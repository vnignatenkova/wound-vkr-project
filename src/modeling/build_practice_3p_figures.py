from __future__ import annotations

from src.config_paths import PRACTICE_3P_FIGURES_DIR, WOUND_FORECAST_DATASET_CSV, FEATURE_SET_COMPARISON_DIR, FEATURE_PRUNING_DIR, COMPACT_MODEL_DIR, ensure_project_dirs
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.config_paths import (
    PROJECT_ROOT,
    PRACTICE_3P_FIGURES_DIR as FIGURES_DIR,
    WOUND_FORECAST_DATASET_CSV as FORECAST_DATASET,
    FEATURE_SET_COMPARISON_DIR,
    FEATURE_PRUNING_DIR,
    COMPACT_MODEL_DIR,
    ensure_project_dirs,
)

FEATURE_SET_METRICS = FEATURE_SET_COMPARISON_DIR / "feature_set_comparison_metrics.csv"
PRUNED_SET_METRICS = FEATURE_PRUNING_DIR / "pruned_feature_set_metrics.csv"
MODEL_METRICS_JSON = COMPACT_MODEL_DIR / "compact_healing_model_metrics.json"
FEATURE_IMPORTANCE_CSV = COMPACT_MODEL_DIR / "compact_healing_feature_importance.csv"

TARGET_COLUMN = "target_healing_speed_class"

MANDATORY_FIGURES = [
    "fig_01_target_class_distribution.png",
    "fig_02_feature_set_comparison_balanced_accuracy.png",
    "fig_03_pruned_feature_sets_balanced_accuracy.png",
    "fig_04_dummy_vs_compact_metrics.png",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_current_figure(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Не найден обязательный файл: {path}")


def load_json(path: Path) -> dict:
    require_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_numeric(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def find_metric_block(data: dict, candidate_keys: Iterable[str]) -> Optional[dict]:
    for key in candidate_keys:
        if key in data and isinstance(data[key], dict):
            return data[key]
    return None


def parse_model_metrics(path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    data = load_json(path)

    dummy_block = find_metric_block(
        data,
        ["dummy_metrics", "dummy", "baseline", "baseline_metrics"],
    )
    model_block = find_metric_block(
        data,
        ["model_metrics", "compact_model_metrics", "compact_metrics", "metrics"],
    )

    if dummy_block is None and model_block is None:
        dummy_block = {
            "accuracy": data.get("dummy_accuracy"),
            "balanced_accuracy": data.get("dummy_balanced_accuracy"),
            "f1_macro": data.get("dummy_f1_macro"),
        }
        model_block = {
            "accuracy": data.get("accuracy") or data.get("model_accuracy"),
            "balanced_accuracy": data.get("balanced_accuracy") or data.get("model_balanced_accuracy"),
            "f1_macro": data.get("f1_macro") or data.get("model_f1_macro"),
        }

    if dummy_block is None or model_block is None:
        raise ValueError(
            "Не удалось распознать структуру compact_healing_model_metrics.json. "
            "Проверь содержимое файла."
        )

    dummy_metrics = {
        "accuracy": normalize_numeric(dummy_block.get("accuracy")),
        "balanced_accuracy": normalize_numeric(dummy_block.get("balanced_accuracy")),
        "f1_macro": normalize_numeric(dummy_block.get("f1_macro", dummy_block.get("f1_binary"))),
    }
    model_metrics = {
        "accuracy": normalize_numeric(model_block.get("accuracy")),
        "balanced_accuracy": normalize_numeric(model_block.get("balanced_accuracy")),
        "f1_macro": normalize_numeric(model_block.get("f1_macro", model_block.get("f1_binary"))),
    }

    missing_dummy = [k for k, v in dummy_metrics.items() if v is None]
    missing_model = [k for k, v in model_metrics.items() if v is None]
    if missing_dummy or missing_model:
        raise ValueError(
            "В JSON не найдены все необходимые метрики.\n"
            f"Отсутствуют у dummy: {missing_dummy}\n"
            f"Отсутствуют у model: {missing_model}"
        )

    return dummy_metrics, model_metrics


def plot_target_distribution(output_path: Path) -> None:
    require_file(FORECAST_DATASET)
    df = pd.read_csv(FORECAST_DATASET)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"В файле {FORECAST_DATASET.name} нет столбца {TARGET_COLUMN!r}")

    counts = df[TARGET_COLUMN].astype(str).value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.tolist(), counts.values.tolist())
    plt.title("Распределение классов заживления")
    plt.xlabel("Класс заживления")
    plt.ylabel("Число наблюдений")

    for i, value in enumerate(counts.values.tolist()):
        plt.text(i, value, str(value), ha="center", va="bottom")

    save_current_figure(output_path)


def plot_feature_set_comparison(output_path: Path) -> None:
    require_file(FEATURE_SET_METRICS)
    df = pd.read_csv(FEATURE_SET_METRICS)

    required_cols = {"feature_set_name", "balanced_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В файле {FEATURE_SET_METRICS.name} нет колонок: {sorted(missing)}")

    preferred_order = ["all_candidate_features", "top_visual_only", "top_features"]
    order_map = {name: i for i, name in enumerate(preferred_order)}
    df = df.copy()
    df["_sort"] = df["feature_set_name"].map(order_map).fillna(999)
    df = df.sort_values(["_sort", "feature_set_name"])

    labels = df["feature_set_name"].tolist()
    values = df["balanced_accuracy"].astype(float).tolist()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.title("Сравнение наборов признаков")
    plt.xlabel("Набор признаков")
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0, max(values) * 1.2 if values else 1.0)
    plt.xticks(rotation=15)

    for i, value in enumerate(values):
        plt.text(i, value, f"{value:.3f}", ha="center", va="bottom")

    save_current_figure(output_path)


def plot_pruned_feature_sets(output_path: Path) -> None:
    require_file(PRUNED_SET_METRICS)
    df = pd.read_csv(PRUNED_SET_METRICS)

    required_cols = {"feature_set_name", "balanced_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В файле {PRUNED_SET_METRICS.name} нет колонок: {sorted(missing)}")

    preferred_order = ["top_visual_only_full", "corr_pruned_0.85", "top4", "top6", "top8"]
    order_map = {name: i for i, name in enumerate(preferred_order)}
    df = df.copy()
    df["_sort"] = df["feature_set_name"].map(order_map).fillna(999)
    df = df.sort_values(["_sort", "feature_set_name"])

    labels = df["feature_set_name"].tolist()
    values = df["balanced_accuracy"].astype(float).tolist()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title("Сравнение урезанных наборов признаков")
    plt.xlabel("Набор признаков")
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0, max(values) * 1.2 if values else 1.0)
    plt.xticks(rotation=20)

    for i, value in enumerate(values):
        plt.text(i, value, f"{value:.3f}", ha="center", va="bottom")

    save_current_figure(output_path)


def plot_dummy_vs_compact_metrics(output_path: Path) -> None:
    dummy_metrics, model_metrics = parse_model_metrics(MODEL_METRICS_JSON)

    metric_names = ["accuracy", "balanced_accuracy", "f1_macro"]
    dummy_values = [dummy_metrics[name] for name in metric_names]
    model_values = [model_metrics[name] for name in metric_names]

    x = list(range(len(metric_names)))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], dummy_values, width=width, label="Dummy baseline")
    plt.bar([i + width / 2 for i in x], model_values, width=width, label="Compact model")

    plt.title("Сравнение baseline и итоговой модели")
    plt.xlabel("Метрика")
    plt.ylabel("Значение")
    plt.xticks(x, ["Accuracy", "Balanced Accuracy", "F1-macro"])
    plt.ylim(0, max(dummy_values + model_values) * 1.2)

    for i, value in enumerate(dummy_values):
        plt.text(i - width / 2, value, f"{value:.3f}", ha="center", va="bottom")
    for i, value in enumerate(model_values):
        plt.text(i + width / 2, value, f"{value:.3f}", ha="center", va="bottom")

    plt.legend()
    save_current_figure(output_path)


def plot_feature_importance_top8(output_path: Path) -> bool:
    if not FEATURE_IMPORTANCE_CSV.exists():
        print(f"[WARN] Пропущен график важности признаков: не найден {FEATURE_IMPORTANCE_CSV}")
        return False

    df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    if df.empty:
        print(f"[WARN] Пропущен график важности признаков: файл {FEATURE_IMPORTANCE_CSV} пуст")
        return False

    feature_col = None
    importance_col = None

    for col in df.columns:
        lower = col.lower()
        if feature_col is None and ("feature" in lower or "name" in lower or "приз" in lower):
            feature_col = col
        if importance_col is None and ("importance" in lower or "weight" in lower or "score" in lower):
            importance_col = col

    if feature_col is None and len(df.columns) >= 1:
        feature_col = df.columns[0]
    if importance_col is None and len(df.columns) >= 2:
        importance_col = df.columns[1]

    if feature_col is None or importance_col is None:
        print("[WARN] Не удалось определить колонки в файле важности признаков.")
        return False

    tmp = df[[feature_col, importance_col]].copy()
    tmp.columns = ["feature", "importance"]
    tmp["importance"] = pd.to_numeric(tmp["importance"], errors="coerce")
    tmp = tmp.dropna(subset=["importance"]).sort_values("importance", ascending=False).head(8)

    if tmp.empty:
        print("[WARN] После очистки не осталось данных для графика важности признаков.")
        return False

    tmp = tmp.iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(tmp["feature"].astype(str), tmp["importance"].astype(float))
    plt.title("Важности признаков итоговой модели (top-8)")
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")

    save_current_figure(output_path)
    return True


def main() -> None:
    ensure_project_dirs()
    ensure_dir(FIGURES_DIR)

    print(f"Сохраняем графики в: {FIGURES_DIR.resolve()}")

    plot_target_distribution(FIGURES_DIR / "fig_01_target_class_distribution.png")
    print("[OK] Построен fig_01_target_class_distribution.png")

    plot_feature_set_comparison(FIGURES_DIR / "fig_02_feature_set_comparison_balanced_accuracy.png")
    print("[OK] Построен fig_02_feature_set_comparison_balanced_accuracy.png")

    plot_pruned_feature_sets(FIGURES_DIR / "fig_03_pruned_feature_sets_balanced_accuracy.png")
    print("[OK] Построен fig_03_pruned_feature_sets_balanced_accuracy.png")

    plot_dummy_vs_compact_metrics(FIGURES_DIR / "fig_04_dummy_vs_compact_metrics.png")
    print("[OK] Построен fig_04_dummy_vs_compact_metrics.png")

    optional_ok = plot_feature_importance_top8(FIGURES_DIR / "fig_05_top8_feature_importance.png")
    if optional_ok:
        print("[OK] Построен fig_05_top8_feature_importance.png")

    print("\nОбязательные графики готовы:")
    for name in MANDATORY_FIGURES:
        print(f"- {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
