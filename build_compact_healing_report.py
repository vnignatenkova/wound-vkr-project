from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# НАСТРОЙКИ
# ============================================================

PREDICTIONS_PATH = Path("compact_healing_predictions/compact_healing_predictions.csv")
FEATURE_IMPORTANCE_PATH = Path("compact_healing_model/compact_healing_feature_importance.csv")
USED_FEATURES_PATH = Path("compact_healing_model/compact_healing_used_features.json")

OUTPUT_DIR = Path("compact_healing_report")
DETAILED_CSV_PATH = OUTPUT_DIR / "compact_healing_report_detailed.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "compact_healing_report_summary.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "compact_healing_report_summary.json"
MARKDOWN_PATH = OUTPUT_DIR / "compact_healing_report.md"

TOP_FEATURES_FOR_EXPLANATION = 3


# ============================================================
# ЧЕЛОВЕКО-ПОНЯТНЫЕ НАЗВАНИЯ ПРИЗНАКОВ
# ============================================================

FEATURE_RU_MAP = {
    "wound_perimeter_px_first": "периметр раны на первом снимке",
    "wound_L_mean_first": "средняя светлота раны (L*) на первом снимке",
    "suture_zone_L_mean_first": "средняя светлота шовной зоны (L*) на первом снимке",
    "wound_glcm_energy_first": "текстурная энергия раны на первом снимке",
    "wound_glcm_entropy_first": "текстурная энтропия раны на первом снимке",
    "wound_glcm_homogeneity_first": "текстурная однородность раны на первом снимке",
    "suture_zone_glcm_energy_first": "текстурная энергия шовной зоны на первом снимке",
    "suture_zone_glcm_entropy_first": "текстурная энтропия шовной зоны на первом снимке",
    "hyperemia_zone_area_pct_wound_first": "доля гиперемии относительно площади раны",
    "devitalized_area_pct_wound_first": "доля девитализированных тканей относительно площади раны",
    "suture_zone_perimeter_px_first": "периметр шовной зоны на первом снимке",
}


CLASS_RU_MAP = {
    "fast": "быстрое заживление",
    "medium": "умеренное заживление",
    "slow": "замедленное заживление",
}


# ============================================================
# ВСПОМОГАТЕЛЬНОЕ
# ============================================================

def load_inputs():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Не найден файл предсказаний: {PREDICTIONS_PATH}")
    if not FEATURE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(f"Не найден файл важностей: {FEATURE_IMPORTANCE_PATH}")
    if not USED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Не найден файл признаков: {USED_FEATURES_PATH}")

    df = pd.read_csv(PREDICTIONS_PATH)
    fi_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    with open(USED_FEATURES_PATH, "r", encoding="utf-8") as f:
        used_features = json.load(f)

    return df, fi_df, used_features


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def feature_display_name(feature: str) -> str:
    return FEATURE_RU_MAP.get(feature, feature)


def class_display_name(class_name: str) -> str:
    return CLASS_RU_MAP.get(class_name, class_name)


def direction_text(value: float, median: float, tol_ratio: float = 0.05) -> str:
    if pd.isna(value) or pd.isna(median):
        return "неопределён"
    if median == 0:
        if abs(value) < 1e-12:
            return "около медианы"
        return "повышен" if value > 0 else "снижен"

    rel_diff = (value - median) / (abs(median) + 1e-12)
    if abs(rel_diff) <= tol_ratio:
        return "около медианы"
    return "повышен" if rel_diff > 0 else "снижен"


def build_feature_stats(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in feature_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        median = vals.median()
        q25 = vals.quantile(0.25)
        q75 = vals.quantile(0.75)
        iqr = q75 - q25
        std = vals.std()
        stats[c] = {
            "median": safe_float(median),
            "q25": safe_float(q25),
            "q75": safe_float(q75),
            "iqr": safe_float(iqr),
            "std": safe_float(std),
        }
    return stats


def build_importance_map(fi_df: pd.DataFrame) -> Dict[str, float]:
    if "perm_importance_mean" in fi_df.columns:
        scores = fi_df.set_index("feature")["perm_importance_mean"].to_dict()
    elif "rf_importance" in fi_df.columns:
        scores = fi_df.set_index("feature")["rf_importance"].to_dict()
    else:
        scores = {f: 1.0 for f in fi_df["feature"].tolist()}

    cleaned = {}
    for k, v in scores.items():
        try:
            val = float(v)
            if np.isnan(val):
                val = 0.0
            cleaned[k] = max(val, 0.0)
        except Exception:
            cleaned[k] = 0.0

    if sum(cleaned.values()) <= 0:
        for k in cleaned:
            cleaned[k] = 1.0

    return cleaned


def feature_outlier_score(value: float, median: float, iqr: float, std: float) -> float:
    if pd.isna(value) or pd.isna(median):
        return 0.0

    scale = np.nan
    if not pd.isna(iqr) and iqr > 1e-12:
        scale = iqr
    elif not pd.isna(std) and std > 1e-12:
        scale = std
    else:
        scale = max(abs(median), 1.0)

    return abs(value - median) / (scale + 1e-12)


def choose_row_explanation_features(
    row: pd.Series,
    feature_cols: List[str],
    feature_stats: Dict[str, Dict[str, float]],
    importance_map: Dict[str, float],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    candidates = []

    for c in feature_cols:
        value = safe_float(row.get(c, np.nan))
        st = feature_stats.get(c, {})
        median = st.get("median", np.nan)
        iqr = st.get("iqr", np.nan)
        std = st.get("std", np.nan)

        outlier = feature_outlier_score(value, median, iqr, std)
        importance = importance_map.get(c, 0.0)
        score = importance * outlier

        candidates.append((c, score))

    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def build_row_interpretation(
    row: pd.Series,
    feature_cols: List[str],
    feature_stats: Dict[str, Dict[str, float]],
    importance_map: Dict[str, float],
) -> Tuple[str, str]:
    pred_class = str(row.get("pred_class_name", "unknown"))
    pred_ru = class_display_name(pred_class)

    top_feats = choose_row_explanation_features(
        row=row,
        feature_cols=feature_cols,
        feature_stats=feature_stats,
        importance_map=importance_map,
        top_k=TOP_FEATURES_FOR_EXPLANATION,
    )

    expl_parts = []
    expl_short_parts = []

    for feat, _score in top_feats:
        value = safe_float(row.get(feat, np.nan))
        st = feature_stats.get(feat, {})
        median = st.get("median", np.nan)
        dir_text = direction_text(value, median)

        feat_ru = feature_display_name(feat)
        expl_parts.append(f"{feat_ru}: {dir_text}")
        expl_short_parts.append(f"{feat_ru} ({dir_text})")

    short_reason = "; ".join(expl_short_parts) if expl_short_parts else "выделяющиеся признаки не определены"
    full_reason = (
        f"Модель отнесла случай к категории «{pred_ru}». "
        f"Наиболее выделяющиеся значимые признаки: {', '.join(expl_parts)}."
        if expl_parts
        else f"Модель отнесла случай к категории «{pred_ru}»."
    )

    return short_reason, full_reason


def aggregate_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["task_dir", "task_name"] if c in df.columns]
    if not group_cols:
        group_cols = ["task_dir"] if "task_dir" in df.columns else []

    if not group_cols:
        raise ValueError("Не найден ни task_dir, ни task_name для агрегации отчёта")

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {col: val for col, val in zip(group_cols, keys)}

        pred_counts = g["pred_class_name"].value_counts(dropna=False).to_dict()
        main_pred = g["pred_class_name"].mode().iloc[0] if len(g["pred_class_name"].mode()) > 0 else np.nan
        mean_conf = safe_float(g["pred_confidence"].mean()) if "pred_confidence" in g.columns else np.nan

        base["n_records"] = int(len(g))
        base["predicted_main_class"] = main_pred
        base["predicted_main_class_ru"] = class_display_name(str(main_pred)) if pd.notna(main_pred) else np.nan
        base["mean_pred_confidence"] = mean_conf
        base["predicted_class_distribution"] = json.dumps(pred_counts, ensure_ascii=False)

        if "target_class_name" in g.columns:
            valid = g["target_class_name"].notna()
            if valid.any():
                true_mode = g.loc[valid, "target_class_name"].mode()
                true_main = true_mode.iloc[0] if len(true_mode) > 0 else np.nan
                base["true_main_class"] = true_main
                base["true_main_class_ru"] = class_display_name(str(true_main)) if pd.notna(true_main) else np.nan
            else:
                base["true_main_class"] = np.nan
                base["true_main_class_ru"] = np.nan

        if "interpretation_short" in g.columns:
            base["task_short_interpretation"] = " | ".join(g["interpretation_short"].astype(str).tolist())

        rows.append(base)

    return pd.DataFrame(rows)


def build_markdown_report(summary_df: pd.DataFrame, detailed_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Отчёт по прогнозу динамики заживления\n")
    lines.append(f"Всего записей: **{len(detailed_df)}**  ")
    lines.append(f"Всего task: **{summary_df['task_dir'].nunique() if 'task_dir' in summary_df.columns else len(summary_df)}**\n")

    if "predicted_main_class_ru" in summary_df.columns:
        class_dist = summary_df["predicted_main_class_ru"].value_counts().to_dict()
        lines.append("## Распределение итоговых предсказаний по task\n")
        for k, v in class_dist.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Краткие результаты по task\n")
    for _, row in summary_df.iterrows():
        task_name = row.get("task_dir", row.get("task_name", "unknown_task"))
        pred_ru = row.get("predicted_main_class_ru", row.get("predicted_main_class", "неизвестно"))
        conf = row.get("mean_pred_confidence", np.nan)
        interp = row.get("task_short_interpretation", "")

        if pd.notna(conf):
            lines.append(f"### {task_name}")
            lines.append(f"- Итоговый прогноз: **{pred_ru}**")
            lines.append(f"- Средняя уверенность модели: **{conf:.3f}**")
        else:
            lines.append(f"### {task_name}")
            lines.append(f"- Итоговый прогноз: **{pred_ru}**")

        if interp:
            lines.append(f"- Ключевые наблюдения: {interp}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, fi_df, used_features = load_inputs()

    df = df.replace([np.inf, -np.inf], np.nan).copy()

    feature_cols = [c for c in used_features if c in df.columns]
    if not feature_cols:
        raise ValueError("Во входном файле предсказаний не найдены нужные признаки")

    feature_stats = build_feature_stats(df, feature_cols)
    importance_map = build_importance_map(fi_df)

    # Если target_class_name отсутствует, попробуем восстановить из target_healing_speed_class
    if "target_class_name" not in df.columns and "target_healing_speed_class" in df.columns:
        df["target_class_name"] = df["target_healing_speed_class"]

    short_list = []
    full_list = []
    for _, row in df.iterrows():
        short_reason, full_reason = build_row_interpretation(
            row=row,
            feature_cols=feature_cols,
            feature_stats=feature_stats,
            importance_map=importance_map,
        )
        short_list.append(short_reason)
        full_list.append(full_reason)

    df["pred_class_name_ru"] = df["pred_class_name"].astype(str).map(CLASS_RU_MAP).fillna(df["pred_class_name"].astype(str))
    df["interpretation_short"] = short_list
    df["interpretation_full"] = full_list

    if "target_class_name" in df.columns:
        df["target_class_name_ru"] = df["target_class_name"].astype(str).map(CLASS_RU_MAP).fillna(df["target_class_name"].astype(str))

    detailed_df = df.copy()
    detailed_df.to_csv(DETAILED_CSV_PATH, index=False, encoding="utf-8-sig")

    summary_df = aggregate_task_summary(detailed_df)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")

    summary_json = {
        "n_rows_detailed": int(len(detailed_df)),
        "n_tasks": int(summary_df["task_dir"].nunique()) if "task_dir" in summary_df.columns else int(len(summary_df)),
        "predicted_class_distribution": summary_df["predicted_main_class"].value_counts(dropna=False).to_dict()
        if "predicted_main_class" in summary_df.columns else {},
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    md_text = build_markdown_report(summary_df, detailed_df)
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Подробный отчёт сохранён в: {DETAILED_CSV_PATH.resolve()}")
    print(f"Краткий отчёт по task сохранён в: {SUMMARY_CSV_PATH.resolve()}")
    print(f"JSON-сводка сохранена в: {SUMMARY_JSON_PATH.resolve()}")
    print(f"Markdown-отчёт сохранён в: {MARKDOWN_PATH.resolve()}")


if __name__ == "__main__":
    main()