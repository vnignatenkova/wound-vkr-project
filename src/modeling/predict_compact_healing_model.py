from src.config_paths import COMPACT_MODEL_DIR, FEATURE_PRUNING_DIR, COMPACT_PREDICTIONS_DIR, ensure_project_dirs
from pathlib import Path
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config_paths import COMPACT_MODEL_DIR, FEATURE_PRUNING_DIR, COMPACT_PREDICTIONS_DIR as OUTPUT_DIR, ensure_project_dirs

MODEL_PATH = COMPACT_MODEL_DIR / "compact_healing_model.joblib"

INPUT_DATASET_PATH = FEATURE_PRUNING_DIR / "top8_dataset.csv"

OUTPUT_PREDICTIONS_PATH = OUTPUT_DIR / "compact_healing_predictions.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "compact_healing_prediction_summary.json"

# Если target есть в таблице, скрипт посчитает метрики
DEFAULT_TARGET_COLUMN = "target_healing_speed_class"



def clean_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        return df

    df = df.copy()
    df[target_col] = df[target_col].replace(
        ["nan", "NaN", "None", "none", "", "NULL", "null"],
        np.nan
    )
    return df


def find_existing_target_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    candidates = []
    if preferred is not None:
        candidates.append(preferred)

    candidates.extend([
        "target_healing_speed_class",
        "target_healing_binary_class",
        "target",
        "healing_class",
        "healing_label",
    ])

    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c in df.columns:
            return c
    return None


def compute_metrics(y_true_labels, y_pred_labels, class_names):
    y_true_arr = np.asarray(y_true_labels)
    y_pred_arr = np.asarray(y_pred_labels)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "confusion_matrix": confusion_matrix(
            y_true_arr,
            y_pred_arr,
            labels=class_names,
        ).tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=class_names,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics



def main():
    ensure_project_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена модель: {MODEL_PATH}")

    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Не найден входной CSV: {INPUT_DATASET_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError("Файл модели имеет неожиданный формат")

    model = bundle.get("model")
    feature_columns = bundle.get("feature_columns")
    class_names = bundle.get("class_names")
    trained_target_column = bundle.get("target_column", DEFAULT_TARGET_COLUMN)

    if model is None:
        raise ValueError("В bundle нет ключа 'model'")
    if not feature_columns:
        raise ValueError("В bundle нет feature_columns")
    if not class_names:
        raise ValueError("В bundle нет class_names")

    df = pd.read_csv(INPUT_DATASET_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)

    target_col = find_existing_target_column(df, trained_target_column)
    if target_col is not None:
        df = clean_target_column(df, target_col)

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(
            "Во входной таблице отсутствуют признаки, нужные модели:\n"
            + "\n".join(missing_features)
        )

    X = df[feature_columns].copy()

    pred_class_ids = model.predict(X)
    pred_class_names = [class_names[i] for i in pred_class_ids]

    result_df = df.copy()
    result_df["pred_class_id"] = pred_class_ids
    result_df["pred_class_name"] = pred_class_names

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        for idx, class_name in enumerate(class_names):
            result_df[f"proba_{class_name}"] = proba[:, idx]

        result_df["pred_confidence"] = np.max(proba, axis=1)
    else:
        result_df["pred_confidence"] = np.nan

    metrics = None
    evaluated_rows = 0

    if target_col is not None:
        eval_mask = result_df[target_col].notna()
        evaluated_rows = int(eval_mask.sum())

        if evaluated_rows > 0:
            y_true_labels = result_df.loc[eval_mask, target_col].astype(str).tolist()
            y_pred_labels = result_df.loc[eval_mask, "pred_class_name"].astype(str).tolist()
            metrics = compute_metrics(y_true_labels, y_pred_labels, class_names)

    result_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "model_path": str(MODEL_PATH),
        "input_dataset_path": str(INPUT_DATASET_PATH),
        "output_predictions_path": str(OUTPUT_PREDICTIONS_PATH),
        "n_rows_total": int(len(result_df)),
        "n_features_used": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "class_names": class_names,
        "target_column_found": target_col,
        "n_rows_with_target_for_evaluation": evaluated_rows,
        "metrics_if_target_present": metrics,
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Загружена модель: {MODEL_PATH.resolve()}")
    print(f"Входной CSV: {INPUT_DATASET_PATH.resolve()}")
    print(f"Обработано строк: {len(result_df)}")
    print(f"Использовано признаков: {len(feature_columns)}")

    if metrics is not None:
        print("\n=== Метрики на входной таблице ===")
        print(f"accuracy            = {metrics['accuracy']:.4f}")
        print(f"balanced_accuracy   = {metrics['balanced_accuracy']:.4f}")
        print(f"f1_macro            = {metrics['f1_macro']:.4f}")
    else:
        print("\nTarget во входной таблице не найден или пустой — сохранены только предсказания.")

    print(f"\nПредсказания сохранены в: {OUTPUT_PREDICTIONS_PATH.resolve()}")
    print(f"Сводка сохранена в: {OUTPUT_SUMMARY_PATH.resolve()}")


if __name__ == "__main__":
    main()
