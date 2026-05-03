from pathlib import Path
import numpy as np
import pandas as pd

DATASET_ROOT = Path(".")
INPUT_CSV = DATASET_ROOT / "wound_image_features.csv"
OUTPUT_CSV = DATASET_ROOT / "wound_phase_dynamics.csv"

ALL_LABELS = [
    "wound",
    "fibrin",
    "metal_device",
    "suture_zone",
    "edema_zone",
    "hyperemia_zone",
    "necrosis_zone",
    "granulation_zone",
    "scale_marker",
    "secondary_pigmentation",
    "subcutaneous_fat_no_granulation",
    "fascia_no_granulation",
    "vac_sponge",
    "wound_depths",
    "tendon",
    "purulent_discharge",
]

COLOR_TEXTURE_LABELS = [
    "wound",
    "suture_zone",
    "granulation_zone",
    "necrosis_zone",
    "fibrin",
    "hyperemia_zone",
    "edema_zone",
    "purulent_discharge",
    "secondary_pigmentation",
    "subcutaneous_fat_no_granulation",
    "fascia_no_granulation",
    "wound_depths",
    "tendon",
]

GLOBAL_CONTINUOUS = [
    "suture_to_wound_area_ratio",
    "suture_to_wound_perimeter_ratio",
    "wound_non_suture_area_px",
    "wound_non_suture_area_pct_wound",
    "reparative_area_pct_wound",
    "inflammatory_area_pct_wound",
    "devitalized_area_pct_wound",
    "deep_structure_area_pct_wound",
    "device_related_area_pct_wound",
    "healing_balance_score",
]

GLOBAL_BINARY = [
    "infection_related_proxy_flag",
    "inflammation_related_proxy_flag",
    "reparative_proxy_flag",
    "deep_damage_proxy_flag",
]


def safe_div(a, b):
    if b is None or pd.isna(b):
        return np.nan
    if isinstance(b, (int, float, np.integer, np.floating)) and b == 0:
        return np.nan
    return a / b


def linear_slope(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan
    if np.unique(x).size < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def build_feature_lists(df: pd.DataFrame):
    continuous = []
    binary = []

    # Геометрия по всем зонам
    geom_suffixes = [
        "_area_px",
        "_area_pct_wound",
        "_perimeter_px",
        "_circularity",
    ]

    for label in ALL_LABELS:
        present_col = f"{label}_present"
        if present_col in df.columns:
            binary.append(present_col)

        for suffix in geom_suffixes:
            col = f"{label}{suffix}"
            if col in df.columns:
                continuous.append(col)

    # Цвет + текстура по клинически значимым зонам
    colortex_suffixes = [
        "_L_mean",
        "_a_mean",
        "_b_mean",
        "_glcm_contrast",
        "_glcm_energy",
        "_glcm_homogeneity",
        "_glcm_entropy",
    ]

    for label in COLOR_TEXTURE_LABELS:
        for suffix in colortex_suffixes:
            col = f"{label}{suffix}"
            if col in df.columns:
                continuous.append(col)

    # Глобальные proxy-признаки
    for col in GLOBAL_CONTINUOUS:
        if col in df.columns:
            continuous.append(col)

    for col in GLOBAL_BINARY:
        if col in df.columns:
            binary.append(col)

    # Разницы между швом и всей раной
    for col in [
        "suture_minus_wound_L_mean",
        "suture_minus_wound_a_mean",
        "suture_minus_wound_b_mean",
    ]:
        if col in df.columns:
            continuous.append(col)

    continuous = list(dict.fromkeys(continuous))
    binary = list(dict.fromkeys(binary))
    return continuous, binary


def aggregate_continuous_array(values, x_day, x_step, prefix):
    nonnull = np.isfinite(values).sum()

    out = {
        f"{prefix}_nonnull_count": int(nonnull),
        f"{prefix}_start": np.nan,
        f"{prefix}_end": np.nan,
        f"{prefix}_mean": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
        f"{prefix}_last_minus_first": np.nan,
        f"{prefix}_rel_change": np.nan,
        f"{prefix}_slope_per_day": np.nan,
        f"{prefix}_slope_per_step": np.nan,
    }

    if nonnull == 0:
        return out

    start = values[0]
    end = values[-1]

    out[f"{prefix}_start"] = start
    out[f"{prefix}_end"] = end
    out[f"{prefix}_mean"] = float(np.nanmean(values))
    out[f"{prefix}_std"] = float(np.nanstd(values)) if nonnull >= 2 else 0.0
    out[f"{prefix}_min"] = float(np.nanmin(values))
    out[f"{prefix}_max"] = float(np.nanmax(values))

    if np.isfinite(start) and np.isfinite(end):
        diff = end - start
        out[f"{prefix}_last_minus_first"] = diff
        out[f"{prefix}_rel_change"] = safe_div(diff, start)

    out[f"{prefix}_slope_per_day"] = linear_slope(x_day, values)
    out[f"{prefix}_slope_per_step"] = linear_slope(x_step, values)
    return out


def aggregate_binary_array(values, prefix):
    vals = np.nan_to_num(values, nan=0.0)
    n = len(vals)
    total = int(vals.sum())

    return {
        f"{prefix}_any": int(total > 0),
        f"{prefix}_all": int(total == n and n > 0),
        f"{prefix}_sum": total,
        f"{prefix}_fraction": float(total / n) if n > 0 else np.nan,
        f"{prefix}_start": int(vals[0]) if n > 0 else np.nan,
        f"{prefix}_end": int(vals[-1]) if n > 0 else np.nan,
    }


def build_phase_row(g: pd.DataFrame, continuous_cols, binary_cols):
    g = g.sort_values(["phase_time_index", "raw_frame"]).reset_index(drop=True)

    row = {
        "task_dir": g["task_dir"].iloc[0],
        "task_name": g["task_name"].iloc[0],
        "phase_label": g["phase_label"].iloc[0],
        "phase_segment_id": int(g["phase_segment_id"].iloc[0]),
        "phase_segment_size": int(g["phase_segment_size"].iloc[0]),
        "usable_for_phase_dynamics": int(g["usable_for_phase_dynamics"].iloc[0]),
        "task_n_images": int(g["task_n_images"].iloc[0]),
        "n_images_in_phase": len(g),
        "first_raw_frame": int(g["raw_frame"].iloc[0]),
        "last_raw_frame": int(g["raw_frame"].iloc[-1]),
        "first_file_name": g["file_name"].iloc[0],
        "last_file_name": g["file_name"].iloc[-1],
        "phase_duration_steps": int(g["phase_time_index"].iloc[-1] - g["phase_time_index"].iloc[0]),
        "phase_duration_days": np.nan,
        "parsed_day_start": g["parsed_day_num"].iloc[0],
        "parsed_day_end": g["parsed_day_num"].iloc[-1],
        "parsed_date_start": g["parsed_date"].iloc[0],
        "parsed_date_end": g["parsed_date"].iloc[-1],
    }

    phase_t = pd.to_numeric(g["phase_time_from_start"], errors="coerce").to_numpy(dtype=float)
    start_t = phase_t[0]
    end_t = phase_t[-1]
    if np.isfinite(start_t) and np.isfinite(end_t):
        row["phase_duration_days"] = float(end_t - start_t)

    if "present_labels" in g.columns:
        union = set()
        for txt in g["present_labels"].fillna("").astype(str):
            if txt.strip():
                union.update([x for x in txt.split(";") if x])
        row["present_labels_union"] = ";".join(sorted(union))
        row["n_present_labels_union"] = len(union)

    x_day = phase_t
    x_step = pd.to_numeric(g["phase_time_index"], errors="coerce").to_numpy(dtype=float)

    cont_df = g[continuous_cols].apply(pd.to_numeric, errors="coerce")
    for col in continuous_cols:
        row.update(aggregate_continuous_array(cont_df[col].to_numpy(dtype=float), x_day, x_step, col))

    bin_df = g[binary_cols].apply(pd.to_numeric, errors="coerce")
    for col in binary_cols:
        row.update(aggregate_binary_array(bin_df[col].to_numpy(dtype=float), col))

    # Удобные summary flags
    for src, dst, sign in [
        ("wound_area_px_last_minus_first", "wound_area_decreased_flag", -1),
        ("wound_perimeter_px_last_minus_first", "wound_perimeter_decreased_flag", -1),
        ("wound_a_mean_last_minus_first", "wound_redness_decreased_flag", -1),
        ("wound_glcm_homogeneity_last_minus_first", "wound_homogeneity_increased_flag", +1),
        ("healing_balance_score_last_minus_first", "healing_balance_improved_flag", +1),
    ]:
        val = row.get(src, np.nan)
        if pd.isna(val):
            row[dst] = np.nan
        else:
            row[dst] = int(val < 0) if sign < 0 else int(val > 0)

    return row


def main():
    if not INPUT_CSV.exists():
        print(f"Не найден входной CSV: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    required_cols = {
        "task_dir",
        "phase_segment_id",
        "phase_label",
        "phase_time_index",
        "phase_time_from_start",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Во входном CSV нет обязательных колонок: {sorted(missing)}")

    continuous_cols, binary_cols = build_feature_lists(df)

    print(f"Найдено continuous-признаков: {len(continuous_cols)}")
    print(f"Найдено binary-признаков: {len(binary_cols)}")

    rows = []
    for _, g in df.groupby(["task_dir", "phase_segment_id"], sort=False):
        rows.append(build_phase_row(g, continuous_cols, binary_cols))

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["task_dir", "phase_segment_id"]).reset_index(drop=True)

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Готово. Сохранено: {OUTPUT_CSV.resolve()}")
    print(f"Строк: {len(out_df)}")
    print(f"Столбцов: {len(out_df.columns)}")


if __name__ == "__main__":
    main()