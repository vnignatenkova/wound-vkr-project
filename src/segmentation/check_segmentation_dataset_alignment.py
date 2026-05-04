from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

from src.config_paths import PROJECT_ROOT, ensure_project_dirs


SEG_DATASET_DIR = PROJECT_ROOT / "data" / "segmentation_dataset"
METADATA_CSV = SEG_DATASET_DIR / "metadata.csv"

OUT_DIR = PROJECT_ROOT / "results" / "segmentation" / "dataset_check"
OVERLAY_DIR = OUT_DIR / "gt_overlays"

SEED = 42
SAMPLES_TOTAL = 20
ALPHA = 0.40

# пороги для "подозрительных" масок
MIN_MASK_RATIO_WARN = 0.001     # 0.1%
MAX_MASK_RATIO_WARN = 0.60      # 60%


def make_red_overlay(image_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    out = image_np.astype(np.float32).copy()
    red = np.zeros_like(out, dtype=np.float32)
    red[..., 0] = 255.0
    mask3 = mask_np[..., None].astype(np.float32)
    out = out * (1.0 - alpha * mask3) + red * (alpha * mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def choose_sample_counts(df: pd.DataFrame, total: int = 20):
    counts = {}
    splits = ["train", "val", "test"]
    available = {s: len(df[df["split"] == s]) for s in splits}

    base = total // 3
    rem = total % 3

    for i, s in enumerate(splits):
        counts[s] = min(available[s], base + (1 if i < rem else 0))

    assigned = sum(counts.values())
    if assigned < total:
        for s in splits:
            free = available[s] - counts[s]
            if free > 0:
                add = min(free, total - assigned)
                counts[s] += add
                assigned += add
            if assigned >= total:
                break

    return counts


def main():
    ensure_project_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Не найден файл: {METADATA_CSV}")

    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(METADATA_CSV)
    if len(df) == 0:
        raise ValueError("metadata.csv пустой")

    report_rows = []

    for idx, row in df.iterrows():
        image_path = SEG_DATASET_DIR / "images" / row["image_file"]
        mask_path = SEG_DATASET_DIR / "masks" / row["mask_file"]

        image_exists = image_path.exists()
        mask_exists = mask_path.exists()

        image_w = image_h = None
        mask_w = mask_h = None
        size_match = False
        mask_area_px_real = None
        mask_ratio = None
        suspicious = False
        suspicious_reason = []

        if image_exists and mask_exists:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            image_w, image_h = image.size
            mask_w, mask_h = mask.size
            size_match = (image_w == mask_w) and (image_h == mask_h)

            mask_np = (np.array(mask) > 0).astype(np.uint8)
            mask_area_px_real = int(mask_np.sum())
            mask_ratio = float(mask_area_px_real / (mask_w * mask_h))

            if not size_match:
                suspicious = True
                suspicious_reason.append("size_mismatch")
            if mask_area_px_real == 0:
                suspicious = True
                suspicious_reason.append("empty_mask")
            if mask_ratio is not None and mask_ratio < MIN_MASK_RATIO_WARN:
                suspicious = True
                suspicious_reason.append("very_small_mask")
            if mask_ratio is not None and mask_ratio > MAX_MASK_RATIO_WARN:
                suspicious = True
                suspicious_reason.append("very_large_mask")
        else:
            suspicious = True
            if not image_exists:
                suspicious_reason.append("missing_image")
            if not mask_exists:
                suspicious_reason.append("missing_mask")

        report_rows.append({
            "task_dir": row["task_dir"],
            "split": row["split"],
            "image_file": row["image_file"],
            "mask_file": row["mask_file"],
            "image_exists": image_exists,
            "mask_exists": mask_exists,
            "image_w": image_w,
            "image_h": image_h,
            "mask_w": mask_w,
            "mask_h": mask_h,
            "size_match": size_match,
            "mask_area_px_real": mask_area_px_real,
            "mask_ratio": mask_ratio,
            "suspicious": suspicious,
            "suspicious_reason": ";".join(suspicious_reason) if suspicious_reason else "",
        })

    report_df = pd.DataFrame(report_rows)
    report_csv = OUT_DIR / "dataset_alignment_report.csv"
    report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    suspicious_df = report_df[report_df["suspicious"] == True].copy()
    suspicious_csv = OUT_DIR / "dataset_alignment_suspicious.csv"
    suspicious_df.to_csv(suspicious_csv, index=False, encoding="utf-8-sig")

    # сохраняем случайные GT overlay
    sample_counts = choose_sample_counts(report_df, total=SAMPLES_TOTAL)
    saved_overlays = 0

    for split, n_samples in sample_counts.items():
        split_dir = OVERLAY_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_df = report_df[
            (report_df["split"] == split)
            & (report_df["image_exists"] == True)
            & (report_df["mask_exists"] == True)
            & (report_df["size_match"] == True)
        ].copy()

        if len(split_df) == 0 or n_samples == 0:
            continue

        chosen = split_df.sample(n=min(n_samples, len(split_df)), random_state=SEED)

        for _, row in chosen.iterrows():
            image_path = SEG_DATASET_DIR / "images" / row["image_file"]
            mask_path = SEG_DATASET_DIR / "masks" / row["mask_file"]

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            image_np = np.array(image)
            mask_np = (np.array(mask) > 0).astype(np.uint8)

            overlay_np = make_red_overlay(image_np, mask_np, alpha=ALPHA)

            base = Path(row["image_file"]).stem

            image.save(split_dir / f"{base}_image.jpg")
            Image.fromarray(mask_np * 255).save(split_dir / f"{base}_mask.png")
            Image.fromarray(overlay_np).save(split_dir / f"{base}_gt_overlay.jpg")

            saved_overlays += 1

    summary = {
        "n_rows_total": int(len(report_df)),
        "n_suspicious": int(len(suspicious_df)),
        "n_missing_image": int((report_df["image_exists"] == False).sum()),
        "n_missing_mask": int((report_df["mask_exists"] == False).sum()),
        "n_size_mismatch": int((report_df["size_match"] == False).sum()),
        "n_saved_gt_overlays": int(saved_overlays),
        "split_counts": report_df["split"].value_counts().to_dict(),
        "suspicious_by_split": suspicious_df["split"].value_counts().to_dict() if len(suspicious_df) else {},
        "mask_ratio_mean": float(report_df["mask_ratio"].dropna().mean()),
        "mask_ratio_median": float(report_df["mask_ratio"].dropna().median()),
        "mask_ratio_min": float(report_df["mask_ratio"].dropna().min()),
        "mask_ratio_max": float(report_df["mask_ratio"].dropna().max()),
    }

    summary_json = OUT_DIR / "dataset_alignment_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Проверка segmentation_dataset завершена ===")
    print(f"Всего строк: {len(report_df)}")
    print(f"Подозрительных случаев: {len(suspicious_df)}")
    print(f"GT overlay сохранено: {saved_overlays}")
    print(f"CSV отчёт: {report_csv}")
    print(f"Подозрительные: {suspicious_csv}")
    print(f"JSON summary: {summary_json}")
    print(f"Папка с GT overlay: {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
