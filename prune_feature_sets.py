from pathlib import Path
import json
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# ============================================================
# НАСТРОЙКИ
# ============================================================

FEATURE_SET_DIR = Path("feature_set_comparison")
ANALYSIS_DIR = Path("feature_significance_analysis")
OUTPUT_DIR = Path("feature_pruning_analysis")

SOURCE_SET_NAME = "top_visual_only"
CORR_THRESHOLD = 0.85
SEED = 42


# ============================================================
# ЗАГРУЗКА
# ============================================================

def load_inputs():
    summary_path = FEATURE_SET_DIR / "feature_set_comparison_summary.json"
    dataset_path = FEATURE_SET_DIR / f"{SOURCE_SET_NAME}_dataset.csv"
    features_json_path = FEATURE_SET_DIR / f"{SOURCE_SET_NAME}_features.json"
    ranking_path = ANALYSIS_DIR / "feature_final_ranking.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Не найден {summary_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Не найден {dataset_path}")
    if not features_json_path.exists():
        raise FileNotFoundError(f"Не найден {features_json_path}")
    if not ranking_path.exists():
        raise FileNotFoundError(f"Не найден {ranking_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        comparison_summary = json.load(f)

    with open(features_json_path, "r", encoding="utf-8") as f:
        source_features = json.load(f)

    dataset = pd.read_csv(dataset_path)
    ranking_df = pd.read_csv(ranking_path)

    return comparison_summary, dataset, source_features, ranking_df


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


# ============================================================
# МОДЕЛИ
# ============================================================

def build_model(problem_type: str):
    if problem_type == "classification":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            )),
        ])
    else:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            )),
        ])


def evaluate_feature_set_classification(X: pd.DataFrame, y: pd.Series) -> dict:
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    class_counts = pd.Series(y_enc).value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise ValueError("Слишком мало объектов в одном из классов для CV.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    model = build_model("classification")

    oof_pred = cross_val_predict(model, X, y_enc, cv=cv, method="predict")

    return {
        "n_rows": int(len(y_enc)),
        "n_features": int(X.shape[1]),
        "cv_splits": int(n_splits),
        "accuracy": float(accuracy_score(y_enc, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_enc, oof_pred)),
        "f1_macro": float(f1_score(y_enc, oof_pred, average="macro")),
        "classes": list(le.classes_),
    }


def evaluate_feature_set_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    y_num = pd.to_numeric(y, errors="coerce")
    valid = ~np.isnan(y_num)

    X = X.loc[valid].reset_index(drop=True)
    y_num = y_num[valid]

    n_splits = min(5, len(X))
    if n_splits < 2:
        raise ValueError("Слишком мало строк для CV.")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    model = build_model("regression")

    oof_pred = cross_val_predict(model, X, y_num, cv=cv, method="predict")

    return {
        "n_rows": int(len(y_num)),
        "n_features": int(X.shape[1]),
        "cv_splits": int(n_splits),
        "r2": float(r2_score(y_num, oof_pred)),
        "mae": float(mean_absolute_error(y_num, oof_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_num, oof_pred))),
    }


def evaluate_feature_set(X: pd.DataFrame, y: pd.Series, problem_type: str) -> dict:
    if problem_type == "classification":
        return evaluate_feature_set_classification(X, y)
    return evaluate_feature_set_regression(X, y)


# ============================================================
# ОТБОР ПО КОРРЕЛЯЦИИ
# ============================================================

def order_features_by_global_ranking(features: List[str], ranking_df: pd.DataFrame) -> List[str]:
    ranking_map = {
        row["feature"]: row["final_rank"]
        for _, row in ranking_df.iterrows()
        if "final_rank" in ranking_df.columns
    }
    return sorted(features, key=lambda f: ranking_map.get(f, 10**9))


def remove_highly_correlated_features(
    df: pd.DataFrame,
    features: List[str],
    ranking_df: pd.DataFrame,
    threshold: float,
) -> List[str]:
    ordered = order_features_by_global_ranking(features, ranking_df)

    X = df[ordered].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    corr = X.corr(method="spearman").abs()

    keep = []
    removed = set()

    for i, feat in enumerate(ordered):
        if feat in removed:
            continue
        keep.append(feat)
        for j in range(i + 1, len(ordered)):
            other = ordered[j]
            if other in removed:
                continue
            val = corr.loc[feat, other]
            if pd.notna(val) and val >= threshold:
                removed.add(other)

    return keep


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_summary, dataset, source_features, ranking_df = load_inputs()

    target_col = comparison_summary["target_column"]
    problem_type = comparison_summary["problem_type"]

    dataset = clean_target_column(dataset, target_col)

    source_features = [f for f in source_features if f in dataset.columns]
    if not source_features:
        raise ValueError("Не найдено ни одного признака из source set в dataset")

    feature_sets = {
        "top_visual_only_full": source_features,
        "top8": source_features[:8],
        "top6": source_features[:6],
        "top4": source_features[:4],
    }

    corr_reduced = remove_highly_correlated_features(
        df=dataset,
        features=source_features,
        ranking_df=ranking_df,
        threshold=CORR_THRESHOLD,
    )
    feature_sets[f"corr_pruned_{CORR_THRESHOLD}"] = corr_reduced

    results = []
    for set_name, feats in feature_sets.items():
        X = dataset[feats].copy()
        y = dataset[target_col].copy()

        metrics = evaluate_feature_set(X, y, problem_type)
        metrics["feature_set_name"] = set_name
        metrics["features"] = feats
        results.append(metrics)

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "features"}
        for r in results
    ])

    if problem_type == "classification":
        sort_col = "balanced_accuracy"
    else:
        sort_col = "r2"

    results_df = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "pruned_feature_set_metrics.csv", index=False, encoding="utf-8-sig")

    for r in results:
        set_name = r["feature_set_name"]
        feats = r["features"]

        with open(OUTPUT_DIR / f"{set_name}_features.json", "w", encoding="utf-8") as f:
            json.dump(feats, f, ensure_ascii=False, indent=2)

        cols_to_save = [c for c in dataset.columns if c not in source_features]
        reduced_df = dataset[cols_to_save + feats].copy()
        reduced_df.to_csv(OUTPUT_DIR / f"{set_name}_dataset.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_set_name": SOURCE_SET_NAME,
        "target_column": target_col,
        "problem_type": problem_type,
        "corr_threshold": CORR_THRESHOLD,
        "results": results,
    }
    with open(OUTPUT_DIR / "pruned_feature_set_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Источник: {SOURCE_SET_NAME}")
    print(f"Target: {target_col}")
    print(f"Тип задачи: {problem_type}\n")
    print("Сравнение урезанных наборов:")
    print(results_df.to_string(index=False))

    best_row = results_df.iloc[0]
    print("\nЛучший набор:")
    print(best_row["feature_set_name"])

    print(f"\nСохранено в: {OUTPUT_DIR.resolve()}")
    print("- pruned_feature_set_metrics.csv")
    print("- pruned_feature_set_summary.json")


if __name__ == "__main__":
    main()