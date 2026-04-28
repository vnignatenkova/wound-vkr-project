from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.inspection import permutation_importance


# ============================================================
# НАСТРОЙКИ
# ============================================================

DATASET_PATH = Path("feature_pruning_analysis/top8_dataset.csv")
FEATURES_PATH = Path("feature_pruning_analysis/top8_features.json")

OUTPUT_DIR = Path("compact_healing_model")
MODEL_PATH = OUTPUT_DIR / "compact_healing_model.joblib"
METRICS_PATH = OUTPUT_DIR / "compact_healing_model_metrics.json"
OOF_PATH = OUTPUT_DIR / "compact_healing_model_oof_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "compact_healing_feature_importance.csv"
USED_FEATURES_OUT_PATH = OUTPUT_DIR / "compact_healing_used_features.json"
CONFIG_PATH = OUTPUT_DIR / "compact_healing_config.json"

TARGET_COLUMN = "target_healing_speed_class"
GROUP_COLUMN = "task_dir"   # если есть повторы task, будет групповой CV
N_SPLITS_DEFAULT = 5
SEED = 42


# ============================================================
# ВСПОМОГАТЕЛЬНОЕ
# ============================================================

def clean_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Не найден target-столбец: {target_col}")

    df = df.replace([np.inf, -np.inf], np.nan).copy()
    df[target_col] = df[target_col].replace(
        ["nan", "NaN", "None", "none", "", "NULL", "null"],
        np.nan
    )
    df = df[df[target_col].notna()].copy().reset_index(drop=True)
    return df


def load_inputs() -> tuple[pd.DataFrame, List[str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Не найден dataset: {DATASET_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Не найден список признаков: {FEATURES_PATH}")

    df = pd.read_csv(DATASET_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("Файл features.json пустой или имеет неверный формат")

    df = clean_target_column(df, TARGET_COLUMN)

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"В таблице отсутствуют признаки: {missing_features}")

    return df, feature_cols


def choose_cv(y_enc: np.ndarray, groups: Optional[pd.Series]):
    class_counts = pd.Series(y_enc).value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(N_SPLITS_DEFAULT, min_class_count)

    if n_splits < 2:
        raise ValueError("Слишком мало объектов в одном из классов для кросс-валидации")

    if groups is not None and HAS_GROUP_CV:
        # Проверим, достаточно ли разных групп
        n_groups = groups.nunique()
        if n_groups >= n_splits:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            return cv, n_splits, True

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return cv, n_splits, False


def build_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def build_dummy() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="prior")),
    ])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def fit_oof_predictions(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    groups: Optional[pd.Series],
    model_pipeline: Pipeline,
):
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

        model = build_model() if model_pipeline is None else model_pipeline
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        oof_pred[test_idx] = pred

        fold_metrics.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
        })

    return oof_pred, fold_metrics, n_splits, used_group_cv


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_inputs()

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(str).copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    groups = df[GROUP_COLUMN] if GROUP_COLUMN in df.columns else None

    # --------------------------------------------------------
    # OOF: основная модель
    # --------------------------------------------------------
    model_pipeline = build_model()
    oof_pred, fold_metrics, n_splits, used_group_cv = fit_oof_predictions(
        X=X,
        y_enc=y_enc,
        groups=groups,
        model_pipeline=None,
    )

    model_metrics = compute_metrics(y_enc, oof_pred, class_names)

    # --------------------------------------------------------
    # OOF: dummy baseline
    # --------------------------------------------------------
    cv, _, _ = choose_cv(y_enc, groups)
    dummy_oof = np.empty(len(X), dtype=int)

    if used_group_cv:
        split_iter = cv.split(X, y_enc, groups)
    else:
        split_iter = cv.split(X, y_enc)

    for train_idx, test_idx in split_iter:
        dummy = build_dummy()
        dummy.fit(X.iloc[train_idx], y_enc[train_idx])
        pred = dummy.predict(X.iloc[test_idx])
        dummy_oof[test_idx] = pred

    dummy_metrics = compute_metrics(y_enc, dummy_oof, class_names)

    # --------------------------------------------------------
    # Финальная модель на всех данных
    # --------------------------------------------------------
    final_model = build_model()
    final_model.fit(X, y_enc)

    # permutation importance
    perm = permutation_importance(
        final_model,
        X,
        y_enc,
        n_repeats=30,
        random_state=SEED,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    model_est = final_model.named_steps["model"]
    rf_importance = model_est.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "rf_importance": rf_importance,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values(
        ["perm_importance_mean", "rf_importance"],
        ascending=[False, False]
    ).reset_index(drop=True)

    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Сохранение OOF
    # --------------------------------------------------------
    oof_df = df.copy()
    oof_df["target_class_id"] = y_enc
    oof_df["target_class_name"] = y
    oof_df["pred_class_id"] = oof_pred
    oof_df["pred_class_name"] = le.inverse_transform(oof_pred)
    oof_df["dummy_pred_class_id"] = dummy_oof
    oof_df["dummy_pred_class_name"] = le.inverse_transform(dummy_oof)
    oof_df.to_csv(OOF_PATH, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Сохранение модели
    # --------------------------------------------------------
    bundle = {
        "model": final_model,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN if GROUP_COLUMN in df.columns else None,
        "class_names": class_names,
        "seed": SEED,
    }
    joblib.dump(bundle, MODEL_PATH)

    with open(USED_FEATURES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    config = {
        "dataset_path": str(DATASET_PATH),
        "features_path": str(FEATURES_PATH),
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN if GROUP_COLUMN in df.columns else None,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_splits": int(n_splits),
        "used_group_cv": bool(used_group_cv),
        "classes": class_names,
        "seed": SEED,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    all_metrics = {
        "config": config,
        "fold_metrics": fold_metrics,
        "dummy_metrics": dummy_metrics,
        "model_metrics": model_metrics,
        "top_feature_importance": fi_df.head(20).to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # Вывод
    # --------------------------------------------------------
    print("=== Dummy baseline ===")
    print(f"accuracy            = {dummy_metrics['accuracy']:.4f}")
    print(f"balanced_accuracy   = {dummy_metrics['balanced_accuracy']:.4f}")
    print(f"f1_macro            = {dummy_metrics['f1_macro']:.4f}")

    print("\n=== Compact model ===")
    print(f"accuracy            = {model_metrics['accuracy']:.4f}")
    print(f"balanced_accuracy   = {model_metrics['balanced_accuracy']:.4f}")
    print(f"f1_macro            = {model_metrics['f1_macro']:.4f}")

    print(f"\nOOF предсказания сохранены в: {OOF_PATH.resolve()}")
    print(f"Метрики сохранены в: {METRICS_PATH.resolve()}")
    print(f"Важности признаков сохранены в: {FEATURE_IMPORTANCE_PATH.resolve()}")
    print(f"Финальная модель сохранена в: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()