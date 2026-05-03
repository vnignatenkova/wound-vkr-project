from src.config_paths import FEATURE_SIGNIFICANCE_DIR, FEATURE_SET_COMPARISON_DIR, ensure_project_dirs
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

from src.config_paths import (
    FEATURE_SIGNIFICANCE_DIR,
    FEATURE_SET_COMPARISON_DIR,
    ensure_project_dirs,
)
# Сколько top-признаков брать из рейтинга

ANALYSIS_DIR = FEATURE_SIGNIFICANCE_DIR
OUTPUT_DIR = FEATURE_SET_COMPARISON_DIR

TOP_K_USE = 15

NON_VISUAL_TOKENS = [
    "parsed_day",
    "day_start",
    "day_num",
    "score",
    "proxy",
    "balance",
    "target",
    "healing",
    "progress",
]


SEED = 42


def load_analysis_artifacts():
    summary_path = ANALYSIS_DIR / "analysis_summary.json"
    selected_path = ANALYSIS_DIR / "selected_top_features.csv"
    used_dataset_path = ANALYSIS_DIR / "analysis_dataset_used.csv"
    final_ranking_path = ANALYSIS_DIR / "feature_final_ranking.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Не найден {summary_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Не найден {selected_path}")
    if not used_dataset_path.exists():
        raise FileNotFoundError(f"Не найден {used_dataset_path}")
    if not final_ranking_path.exists():
        raise FileNotFoundError(f"Не найден {final_ranking_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    selected_df = pd.read_csv(selected_path)
    used_df = pd.read_csv(used_dataset_path)
    ranking_df = pd.read_csv(final_ranking_path)

    return summary, selected_df, used_df, ranking_df


# ============================================================
# ВСПОМОГАТЕЛЬНОЕ
# ============================================================

def has_any_token(name: str, tokens: List[str]) -> bool:
    low = name.lower()
    return any(tok in low for tok in tokens)


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


def build_feature_sets(
    target_col: str,
    used_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    all_candidate_features = [c for c in used_df.columns if c != target_col]

    ranked_features = ranking_df["feature"].tolist()
    ranked_features = [f for f in ranked_features if f in all_candidate_features]

    top_features = ranked_features[:TOP_K_USE]

    top_visual_only = [
        f for f in top_features
        if not has_any_token(f, NON_VISUAL_TOKENS)
    ]

    # Если после фильтра осталось слишком мало, добираем дальше по рейтингу
    if len(top_visual_only) < 8:
        for f in ranked_features[TOP_K_USE:]:
            if f in top_visual_only:
                continue
            if has_any_token(f, NON_VISUAL_TOKENS):
                continue
            top_visual_only.append(f)
            if len(top_visual_only) >= 15:
                break

    return {
        "all_candidate_features": all_candidate_features,
        "top_features": top_features,
        "top_visual_only": top_visual_only,
    }


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

    metrics = {
        "n_rows": int(len(y_enc)),
        "n_features": int(X.shape[1]),
        "cv_splits": int(n_splits),
        "accuracy": float(accuracy_score(y_enc, oof_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_enc, oof_pred)),
        "f1_macro": float(f1_score(y_enc, oof_pred, average="macro")),
        "classes": list(le.classes_),
    }
    return metrics


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

    metrics = {
        "n_rows": int(len(y_num)),
        "n_features": int(X.shape[1]),
        "cv_splits": int(n_splits),
        "r2": float(r2_score(y_num, oof_pred)),
        "mae": float(mean_absolute_error(y_num, oof_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_num, oof_pred))),
    }
    return metrics


def evaluate_feature_set(X: pd.DataFrame, y: pd.Series, problem_type: str) -> dict:
    if problem_type == "classification":
        return evaluate_feature_set_classification(X, y)
    return evaluate_feature_set_regression(X, y)


def prepare_reduced_table(
    source_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    meta_cols = [
        c for c in [
            "task_dir",
            "task_name",
            "phase_segment_id",
            "phase_label",
            "target_healing_speed_class",
            "target_healing_binary_class",
        ]
        if c in source_df.columns
    ]

    keep_cols = []
    for c in meta_cols + feature_cols:
        if c not in keep_cols and c in source_df.columns:
            keep_cols.append(c)

    if target_col not in keep_cols and target_col in source_df.columns:
        keep_cols.append(target_col)

    reduced = source_df[keep_cols].copy()
    reduced = clean_target_column(reduced, target_col)
    return reduced



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary, selected_df, used_df, ranking_df = load_analysis_artifacts()

    input_table = Path(summary["input_table"])
    target_col = summary["target_column"]
    problem_type = summary["problem_type"]

    full_source_df = pd.read_csv(input_table)
    full_source_df = clean_target_column(full_source_df, target_col)

    used_df = clean_target_column(used_df, target_col)

    feature_sets = build_feature_sets(
        target_col=target_col,
        used_df=used_df,
        ranking_df=ranking_df,
    )

    results = []
    reduced_tables = {}

    for set_name, feature_cols in feature_sets.items():
        X = used_df[feature_cols].copy()
        y = used_df[target_col].copy()

        metrics = evaluate_feature_set(X, y, problem_type)
        metrics["feature_set_name"] = set_name
        metrics["features"] = feature_cols
        results.append(metrics)

        reduced_tables[set_name] = prepare_reduced_table(
            source_df=full_source_df,
            target_col=target_col,
            feature_cols=feature_cols,
        )

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "features"}
        for r in results
    ])

    if problem_type == "classification":
        sort_col = "balanced_accuracy"
    else:
        sort_col = "r2"

    results_df = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "feature_set_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    for r in results:
        set_name = r["feature_set_name"]
        feature_cols = r["features"]

        with open(OUTPUT_DIR / f"{set_name}_features.json", "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)

        reduced_tables[set_name].to_csv(
            OUTPUT_DIR / f"{set_name}_dataset.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_out = {
        "input_table": str(input_table),
        "target_column": target_col,
        "problem_type": problem_type,
        "top_k_use": TOP_K_USE,
        "non_visual_tokens": NON_VISUAL_TOKENS,
        "results": results,
    }

    with open(OUTPUT_DIR / "feature_set_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    print(f"Входная таблица: {input_table}")
    print(f"Target: {target_col}")
    print(f"Тип задачи: {problem_type}\n")

    print("Сравнение наборов признаков:")
    print(results_df.to_string(index=False))

    print(f"\nСохранено в: {OUTPUT_DIR.resolve()}")
    print("- feature_set_comparison_metrics.csv")
    print("- feature_set_comparison_summary.json")
    print("- all_candidate_features_dataset.csv")
    print("- top_features_dataset.csv")
    print("- top_visual_only_dataset.csv")


if __name__ == "__main__":
    ensure_project_dirs()
    main()
