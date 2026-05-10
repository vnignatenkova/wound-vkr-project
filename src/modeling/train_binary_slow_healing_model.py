from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_GROUP_CV = True
except ImportError:
    HAS_GROUP_CV = False

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

try:
    from src.config_paths import ensure_project_dirs
except Exception:
    def ensure_project_dirs():
        pass


# ============================================================
# НАСТРОЙКИ
# ============================================================

PROJECT_ROOT = Path.cwd()

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "wound_forecast_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "binary_slow_healing_model"

TARGET_SOURCE_COLUMN = "target_healing_speed_class"
TARGET_BINARY_COLUMN = "target_slow_healing_risk"

GROUP_COLUMN = "task_dir"
SEED = 42
N_SPLITS_DEFAULT = 5

DATASET_OUT = OUTPUT_DIR / "binary_slow_healing_dataset.csv"
METRICS_OUT = OUTPUT_DIR / "binary_slow_healing_model_metrics.json"
OOF_OUT = OUTPUT_DIR / "binary_slow_healing_oof_predictions.csv"
FEATURES_OUT = OUTPUT_DIR / "binary_slow_healing_used_features.json"
IMPORTANCE_OUT = OUTPUT_DIR / "binary_slow_healing_feature_importance.csv"
MODEL_OUT = OUTPUT_DIR / "binary_slow_healing_model.joblib"


# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    if TARGET_SOURCE_COLUMN not in df.columns:
        raise ValueError(f"В датасете нет колонки {TARGET_SOURCE_COLUMN}")

    df[TARGET_SOURCE_COLUMN] = df[TARGET_SOURCE_COLUMN].replace(
        ["nan", "NaN", "None", "none", "", "NULL", "null"],
        np.nan
    )

    df = df[df[TARGET_SOURCE_COLUMN].notna()].copy().reset_index(drop=True)

    # Новый бинарный target:
    # slow = риск медленного заживления
    # not_slow = fast + medium
    df[TARGET_BINARY_COLUMN] = np.where(
        df[TARGET_SOURCE_COLUMN].astype(str) == "slow",
        "slow",
        "not_slow"
    )

    return df


def build_candidate_features(df: pd.DataFrame) -> List[str]:
    """
    Берём признаки начального состояния (*_first) + несколько стартовых служебных признаков.
    Не берём target_future_* и другие признаки будущего, чтобы не было утечки.
    """

    exclude_exact = {
        "task_dir",
        "task_name",
        "phase_label",
        "phase_segment_id",
        TARGET_SOURCE_COLUMN,
        TARGET_BINARY_COLUMN,
        "target_healing_speed_class_id",
    }

    exclude_tokens = [
        "target_",
        "future",
        "final",
        "last_early",
        "early_mean",
        "early_std",
        "delta_early",
        "rel_delta_early",
        "file_name",
        "parsed_date",
        "present_labels_union",
    ]

    allowed_non_first = {
        "parsed_day_start",
        "observed_raw_frame_start",
        "phase_segment_size",
        "n_images_observed",
        "observed_duration_days",
    }

    features = []

    for col in df.columns:
        if col in exclude_exact:
            continue

        if any(tok in col for tok in exclude_tokens):
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if col.endswith("_first") or col in allowed_non_first:
            features.append(col)

    features = list(dict.fromkeys(features))

    if not features:
        raise ValueError("Не найдено ни одного признака для обучения")

    return features


# ============================================================
# МОДЕЛИ
# ============================================================

def build_models() -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                C=0.3,
                max_iter=2000,
                random_state=SEED,
            )),
        ]),

        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),

        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
    }


def build_dummy() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])


def choose_cv(y_enc: np.ndarray, groups: Optional[pd.Series]):
    class_counts = pd.Series(y_enc).value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(N_SPLITS_DEFAULT, min_class_count)

    if n_splits < 2:
        raise ValueError("Слишком мало объектов в одном из классов для CV")

    if groups is not None and HAS_GROUP_CV:
        if groups.nunique() >= n_splits:
            return StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=SEED
            ), n_splits, True

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED
    ), n_splits, False


def compute_metrics(y_true, y_pred, class_names) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def fit_oof(model: Pipeline, X: pd.DataFrame, y_enc: np.ndarray, groups: Optional[pd.Series]):
    cv, n_splits, used_group_cv = choose_cv(y_enc, groups)

    oof_pred = np.empty(len(X), dtype=int)
    fold_metrics = []

    if used_group_cv:
        split_iter = cv.split(X, y_enc, groups)
    else:
        split_iter = cv.split(X, y_enc)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_enc[train_idx]
        y_test = y_enc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        oof_pred[test_idx] = pred

        fold_metrics.append({
            "fold": int(fold_idx),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
        })

    return oof_pred, fold_metrics, n_splits, used_group_cv


def make_feature_importance(model: Pipeline, feature_cols: List[str], model_name: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]

    if model_name == "logistic_regression":
        values = np.abs(estimator.coef_[0])
        out = pd.DataFrame({
            "feature": feature_cols,
            "importance": values,
        })
    elif hasattr(estimator, "feature_importances_"):
        out = pd.DataFrame({
            "feature": feature_cols,
            "importance": estimator.feature_importances_,
        })
    else:
        out = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.nan,
        })

    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_project_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Не найден файл: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = clean_df(df)

    feature_cols = build_candidate_features(df)

    X = df[feature_cols].copy()
    y = df[TARGET_BINARY_COLUMN].astype(str).copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    groups = df[GROUP_COLUMN] if GROUP_COLUMN in df.columns else None

    print("=== Binary slow healing model ===")
    print(f"Датасет: {INPUT_CSV}")
    print(f"Строк после очистки: {len(df)}")
    print(f"Признаков: {len(feature_cols)}")
    print(f"Target: {TARGET_BINARY_COLUMN}")
    print("Распределение классов:")
    print(y.value_counts().to_string())

    models = build_models()

    all_model_results = {}
    best_name = None
    best_score = -1
    best_oof = None
    best_fold_metrics = None
    best_used_group_cv = None
    best_n_splits = None

    for model_name, model in models.items():
        oof_pred, fold_metrics, n_splits, used_group_cv = fit_oof(
            model=model,
            X=X,
            y_enc=y_enc,
            groups=groups,
        )

        metrics = compute_metrics(y_enc, oof_pred, class_names)

        all_model_results[model_name] = {
            "metrics": metrics,
            "fold_metrics": fold_metrics,
            "n_splits": int(n_splits),
            "used_group_cv": bool(used_group_cv),
        }

        print(f"\n--- {model_name} ---")
        print(f"accuracy          = {metrics['accuracy']:.4f}")
        print(f"balanced_accuracy = {metrics['balanced_accuracy']:.4f}")
        print(f"f1_macro          = {metrics['f1_macro']:.4f}")
        print("confusion_matrix:")
        print(np.array(metrics["confusion_matrix"]))

        # Выбираем лучшую модель по balanced_accuracy.
        # Если равенство — по accuracy.
        score = metrics["balanced_accuracy"] + metrics["accuracy"] * 0.0001
        if score > best_score:
            best_score = score
            best_name = model_name
            best_oof = oof_pred
            best_fold_metrics = fold_metrics
            best_used_group_cv = used_group_cv
            best_n_splits = n_splits

    # Dummy baseline
    dummy = build_dummy()
    dummy_oof, dummy_fold_metrics, _, _ = fit_oof(
        model=dummy,
        X=X,
        y_enc=y_enc,
        groups=groups,
    )
    dummy_metrics = compute_metrics(y_enc, dummy_oof, class_names)

    print("\n=== Dummy baseline ===")
    print(f"accuracy          = {dummy_metrics['accuracy']:.4f}")
    print(f"balanced_accuracy = {dummy_metrics['balanced_accuracy']:.4f}")
    print(f"f1_macro          = {dummy_metrics['f1_macro']:.4f}")

    # Финальная модель обучается на всех строках
    final_model = models[best_name]
    final_model.fit(X, y_enc)

    # Сохраняем датасет с новым target
    df.to_csv(DATASET_OUT, index=False, encoding="utf-8-sig")

    # OOF-предсказания
    oof_df = df.copy()
    oof_df["binary_target_class_id"] = y_enc
    oof_df["binary_target_class_name"] = y
    oof_df["binary_pred_class_id"] = best_oof
    oof_df["binary_pred_class_name"] = le.inverse_transform(best_oof)
    oof_df["binary_dummy_pred_class_id"] = dummy_oof
    oof_df["binary_dummy_pred_class_name"] = le.inverse_transform(dummy_oof)
    oof_df.to_csv(OOF_OUT, index=False, encoding="utf-8-sig")

    # Важность признаков
    fi_df = make_feature_importance(final_model, feature_cols, best_name)
    fi_df.to_csv(IMPORTANCE_OUT, index=False, encoding="utf-8-sig")

    # Список признаков
    with open(FEATURES_OUT, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # Модель
    bundle = {
        "model": final_model,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "target_column": TARGET_BINARY_COLUMN,
        "source_target_column": TARGET_SOURCE_COLUMN,
        "group_column": GROUP_COLUMN if GROUP_COLUMN in df.columns else None,
        "class_names": class_names,
        "best_model_name": best_name,
        "seed": SEED,
    }
    joblib.dump(bundle, MODEL_OUT)

    # Метрики
    summary = {
        "config": {
            "input_csv": str(INPUT_CSV),
            "target_source_column": TARGET_SOURCE_COLUMN,
            "target_binary_column": TARGET_BINARY_COLUMN,
            "group_column": GROUP_COLUMN if GROUP_COLUMN in df.columns else None,
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "classes": class_names,
            "class_distribution": y.value_counts().to_dict(),
            "n_splits": int(best_n_splits),
            "used_group_cv": bool(best_used_group_cv),
            "seed": SEED,
        },
        "best_model_name": best_name,
        "dummy_metrics": dummy_metrics,
        "all_model_results": all_model_results,
        "best_model_metrics": all_model_results[best_name]["metrics"],
        "best_fold_metrics": best_fold_metrics,
        "top_feature_importance": fi_df.head(20).to_dict(orient="records"),
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Лучшая модель ===")
    print(best_name)
    print(f"accuracy          = {summary['best_model_metrics']['accuracy']:.4f}")
    print(f"balanced_accuracy = {summary['best_model_metrics']['balanced_accuracy']:.4f}")
    print(f"f1_macro          = {summary['best_model_metrics']['f1_macro']:.4f}")

    print("\nФайлы сохранены:")
    print(f"- {DATASET_OUT}")
    print(f"- {METRICS_OUT}")
    print(f"- {OOF_OUT}")
    print(f"- {IMPORTANCE_OUT}")
    print(f"- {FEATURES_OUT}")
    print(f"- {MODEL_OUT}")


if __name__ == "__main__":
    main()
