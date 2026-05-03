from src.config_paths import WOUND_FORECAST_DATASET_CSV, FEATURE_SIGNIFICANCE_DIR, ensure_project_dirs
from pathlib import Path
import json
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
)

from src.config_paths import (
    WOUND_FORECAST_DATASET_CSV,
    FEATURE_SIGNIFICANCE_DIR,
    ensure_project_dirs,
)

INPUT_TABLE = WOUND_FORECAST_DATASET_CSV
TARGET_COLUMN = "target_healing_speed_class"

OUTPUT_DIR = FEATURE_SIGNIFICANCE_DIR
SEED = 42

MIN_NON_NULL_RATIO = 0.60
MIN_UNIQUE_VALUES = 2
TOP_K_FINAL = 30

# Если True — скрипт попытается брать в первую очередь baseline/start-признаки
PREFER_BASELINE_FEATURES = True



def find_input_table() -> Path:
    if INPUT_TABLE is not None:
        if not INPUT_TABLE.exists():
            raise FileNotFoundError(f"Не найден INPUT_TABLE: {INPUT_TABLE}")
        return INPUT_TABLE

    candidates = [
        WOUND_PHASE_DYNAMICS_CSV,
        WOUND_FORECAST_DATASET_CSV,
        WOUND_IMAGE_FEATURES_CSV,
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Не найдена ни одна из таблиц: "
        "wound_phase_dynamics.csv / wound_forecast_dataset.csv / "
        "wound_forecast_inference_dataset.csv / wound_image_features.csv"
    )


def detect_target_column(df: pd.DataFrame) -> str:
    if TARGET_COLUMN is not None:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"TARGET_COLUMN={TARGET_COLUMN} не найден в таблице")
        return TARGET_COLUMN

    candidates = [
        "target_healing_speed_class",
        "target_healing_binary_class",
        "target_binary_progress_class",
        "healing_binary_label",
        "healing_speed_class",
        "healing_class",
        "healing_label",
        "is_healing",
        "progress_label",
        "target",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "Не удалось автоматически найти target-столбец. "
        "Укажи TARGET_COLUMN вручную."
    )


def detect_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique(dropna=True)
        if n_unique <= 10:
            return "classification"
        return "regression"
    return "classification"


def has_any_token(name: str, tokens: List[str]) -> bool:
    low = name.lower()
    return any(tok in low for tok in tokens)


def choose_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    exact_exclude = {
        target_col,
        "task_dir",
        "task_name",
        "file_name",
        "image_path",
        "raw_name",
        "present_labels",
        "predicted_labels_present",
        "parsed_date",
        "phase_label",
        "phase_segment_id",
    }

    # Технические и текстовые поля
    always_exclude_tokens = [
        "path",
        "file",
        "name",
        "date",
        "label",
        "class_id",
        "present_labels",
        "pred_",
        "prediction",
    ]

    # Явная утечка из будущего / из target
    leakage_tokens = [
        "target",
        "future",
        "outcome",
        "progress",
        "healing",
        "improvement",
        "response",
        "delta_",
        "_delta",
        "change",
        "slope",
        "_end",
        "end_",
        "_last",
        "last_",
        "_followup",
        "followup_",
        "_rel_phase_start",
    ]

    # Что считаем baseline-признаками
    baseline_regexes = [
        r"^start_",
        r"_start$",
        r"^baseline_",
        r"_baseline$",
        r"^first_",
        r"_first$",
        r"^initial_",
        r"_initial$",
        r"^t0_",
        r"_t0$",
        r"_rel_task_start$",
    ]

    numeric_cols = []
    for c in df.columns:
        if c in exact_exclude:
            continue
        if has_any_token(c, always_exclude_tokens):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        numeric_cols.append(c)

    baseline_cols = []
    for c in numeric_cols:
        if any(re.search(rx, c.lower()) for rx in baseline_regexes):
            baseline_cols.append(c)

    if PREFER_BASELINE_FEATURES and len(baseline_cols) >= 10:
        candidate_cols = baseline_cols
    else:
        candidate_cols = []
        for c in numeric_cols:
            if has_any_token(c, leakage_tokens):
                continue
            candidate_cols.append(c)

    # фильтр по полноте и вариативности
    final_cols = []
    for c in candidate_cols:
        non_null_ratio = df[c].notna().mean()
        nunique = df[c].nunique(dropna=True)
        if non_null_ratio < MIN_NON_NULL_RATIO:
            continue
        if nunique < MIN_UNIQUE_VALUES:
            continue
        final_cols.append(c)

    return sorted(final_cols)



def run_univariate_classification(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    X_imp = SimpleImputer(strategy="median").fit_transform(X)

    mi = mutual_info_classif(X_imp, y_enc, random_state=SEED)
    f_vals, p_vals = f_classif(X_imp, y_enc)

    out = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi,
        "f_score": f_vals,
        "f_pvalue": p_vals,
        "missing_ratio": X.isna().mean().values,
    })

    out["rank_mi"] = out["mi_score"].rank(ascending=False, method="average")
    out["rank_f"] = out["f_score"].rank(ascending=False, method="average")
    out["rank_uni_avg"] = out[["rank_mi", "rank_f"]].mean(axis=1)
    out = out.sort_values(["rank_uni_avg", "mi_score"], ascending=[True, False]).reset_index(drop=True)
    return out


def run_univariate_regression(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    y_num = pd.to_numeric(y, errors="coerce")
    valid = ~np.isnan(y_num)
    X_imp = X_imp[valid]
    y_num = y_num[valid]

    mi = mutual_info_regression(X_imp, y_num, random_state=SEED)
    f_vals, p_vals = f_regression(X_imp, y_num)

    out = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi,
        "f_score": f_vals,
        "f_pvalue": p_vals,
        "missing_ratio": X.isna().mean().values,
    })

    out["rank_mi"] = out["mi_score"].rank(ascending=False, method="average")
    out["rank_f"] = out["f_score"].rank(ascending=False, method="average")
    out["rank_uni_avg"] = out[["rank_mi", "rank_f"]].mean(axis=1)
    out = out.sort_values(["rank_uni_avg", "mi_score"], ascending=[True, False]).reset_index(drop=True)
    return out



def run_model_classification(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, dict]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    class_counts = pd.Series(y_enc).value_counts().sort_index()
    min_class_count = int(class_counts.min())

    metrics = {
        "problem_type": "classification",
        "classes": list(le.classes_),
        "n_rows": int(len(y_enc)),
        "class_distribution": {str(le.classes_[i]): int(v) for i, v in class_counts.items()},
    }

    if min_class_count >= 2:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        oof_pred = cross_val_predict(model, X, y_enc, cv=cv, method="predict")
        metrics["cv_accuracy"] = float(accuracy_score(y_enc, oof_pred))
        metrics["cv_balanced_accuracy"] = float(balanced_accuracy_score(y_enc, oof_pred))
        metrics["cv_f1_macro"] = float(f1_score(y_enc, oof_pred, average="macro"))
    else:
        metrics["cv_accuracy"] = np.nan
        metrics["cv_balanced_accuracy"] = np.nan
        metrics["cv_f1_macro"] = np.nan

    model.fit(X, y_enc)

    rf = model.named_steps["rf"]
    importances = rf.feature_importances_

    perm = permutation_importance(
        model,
        X,
        y_enc,
        n_repeats=20,
        random_state=SEED,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    out = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": importances,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    })

    out["rank_rf"] = out["rf_importance"].rank(ascending=False, method="average")
    out["rank_perm"] = out["perm_importance_mean"].rank(ascending=False, method="average")
    out["rank_model_avg"] = out[["rank_rf", "rank_perm"]].mean(axis=1)
    out = out.sort_values(["rank_model_avg", "perm_importance_mean"], ascending=[True, False]).reset_index(drop=True)

    return out, metrics


def run_model_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, dict]:
    y_num = pd.to_numeric(y, errors="coerce")
    valid = ~np.isnan(y_num)

    X = X.loc[valid].reset_index(drop=True)
    y_num = y_num[valid]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    if len(X) >= 10:
        n_splits = min(5, len(X))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        oof_pred = cross_val_predict(model, X, y_num, cv=cv, method="predict")
        metrics = {
            "problem_type": "regression",
            "n_rows": int(len(y_num)),
            "cv_r2": float(r2_score(y_num, oof_pred)),
            "cv_mae": float(mean_absolute_error(y_num, oof_pred)),
        }
    else:
        metrics = {
            "problem_type": "regression",
            "n_rows": int(len(y_num)),
            "cv_r2": np.nan,
            "cv_mae": np.nan,
        }

    model.fit(X, y_num)

    rf = model.named_steps["rf"]
    importances = rf.feature_importances_

    perm = permutation_importance(
        model,
        X,
        y_num,
        n_repeats=20,
        random_state=SEED,
        scoring="r2",
        n_jobs=-1,
    )

    out = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": importances,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    })

    out["rank_rf"] = out["rf_importance"].rank(ascending=False, method="average")
    out["rank_perm"] = out["perm_importance_mean"].rank(ascending=False, method="average")
    out["rank_model_avg"] = out[["rank_rf", "rank_perm"]].mean(axis=1)
    out = out.sort_values(["rank_model_avg", "perm_importance_mean"], ascending=[True, False]).reset_index(drop=True)

    return out, metrics



def build_final_ranking(univariate_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    merged = univariate_df.merge(model_df, on="feature", how="outer")

    if "rank_uni_avg" not in merged:
        merged["rank_uni_avg"] = np.nan
    if "rank_model_avg" not in merged:
        merged["rank_model_avg"] = np.nan

    max_uni = np.nanmax(merged["rank_uni_avg"].values) if merged["rank_uni_avg"].notna().any() else 9999
    max_model = np.nanmax(merged["rank_model_avg"].values) if merged["rank_model_avg"].notna().any() else 9999

    merged["rank_uni_filled"] = merged["rank_uni_avg"].fillna(max_uni + 1000)
    merged["rank_model_filled"] = merged["rank_model_avg"].fillna(max_model + 1000)
    merged["final_rank_score"] = 0.5 * merged["rank_uni_filled"] + 0.5 * merged["rank_model_filled"]

    merged = merged.sort_values(
        ["final_rank_score", "perm_importance_mean", "mi_score"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    merged["final_rank"] = np.arange(1, len(merged) + 1)
    return merged



def main():
    ensure_project_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_table = find_input_table()
    df = pd.read_csv(input_table)

    target_col = detect_target_column(df)
    problem_type = detect_problem_type(df[target_col])

    # Приведём inf к NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Удаляем строки без target
    df = df[df[target_col].notna()].copy().reset_index(drop=True)

    feature_cols = choose_feature_columns(df, target_col)
    if not feature_cols:
        raise ValueError("После фильтрации не осталось признаков для анализа.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    if problem_type == "classification":
        univariate_df = run_univariate_classification(X, y)
        model_df, model_metrics = run_model_classification(X, y)
    else:
        univariate_df = run_univariate_regression(X, y)
        model_df, model_metrics = run_model_regression(X, y)

    final_df = build_final_ranking(univariate_df, model_df)

    top_df = final_df.head(TOP_K_FINAL).copy()
    top_features = top_df["feature"].tolist()

    # Сохраняем
    univariate_path = OUTPUT_DIR / "feature_univariate_scores.csv"
    model_path = OUTPUT_DIR / "feature_model_importance.csv"
    final_path = OUTPUT_DIR / "feature_final_ranking.csv"
    top_path = OUTPUT_DIR / "selected_top_features.csv"
    top_txt_path = OUTPUT_DIR / "selected_top_features.txt"
    analysis_dataset_path = OUTPUT_DIR / "analysis_dataset_used.csv"
    summary_path = OUTPUT_DIR / "analysis_summary.json"

    univariate_df.to_csv(univariate_path, index=False, encoding="utf-8-sig")
    model_df.to_csv(model_path, index=False, encoding="utf-8-sig")
    final_df.to_csv(final_path, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    with open(top_txt_path, "w", encoding="utf-8") as f:
        for feat in top_features:
            f.write(feat + "\n")

    df[[target_col] + feature_cols].to_csv(analysis_dataset_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_table": str(input_table),
        "target_column": target_col,
        "problem_type": problem_type,
        "n_rows": int(len(df)),
        "n_candidate_features": int(len(feature_cols)),
        "top_k_saved": int(TOP_K_FINAL),
        "model_metrics": model_metrics,
        "top_features": top_features,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Входная таблица: {input_table}")
    print(f"Target: {target_col}")
    print(f"Тип задачи: {problem_type}")
    print(f"Строк: {len(df)}")
    print(f"Признаков после фильтрации: {len(feature_cols)}")
    print("\nТоп-15 признаков:")
    for i, feat in enumerate(top_features[:15], start=1):
        print(f"{i:02d}. {feat}")

    print(f"\nСохранено в: {OUTPUT_DIR.resolve()}")
    print(f"- {univariate_path.name}")
    print(f"- {model_path.name}")
    print(f"- {final_path.name}")
    print(f"- {top_path.name}")
    print(f"- {summary_path.name}")


if __name__ == "__main__":
    main()
