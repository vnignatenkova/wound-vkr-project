from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
from PIL import Image

from src.config_paths import PROJECT_ROOT, ensure_project_dirs


SOURCE_DATASET_DIR = Path(os.environ.get(
    "SEG_SOURCE_DATASET_DIR",
    str(PROJECT_ROOT / "data" / "segmentation_dataset")
))
SOURCE_METADATA_CSV = Path(os.environ.get(
    "SEG_METADATA_CSV",
    str(SOURCE_DATASET_DIR / "metadata.csv")
))

CROP_DATASET_DIR = Path(os.environ.get(
    "SEG_CROP_DATASET_DIR",
    str(PROJECT_ROOT / "data" / "segmentation_dataset_crop")
))

IMAGES_DIR = CROP_DATASET_DIR / "images"
MASKS_DIR = CROP_DATASET_DIR / "masks"
METADATA_CSV = CROP_DATASET_DIR / "metadata.csv"
SUMMARY_JSON = CROP_DATASET_DIR / "summary.json"

MARGIN_RATIO = float(os.environ.get("SEG_MARGIN_RATIO", "0.25"))
MIN_MARGIN_PX = int(os.environ.get("SEG_MIN_MARGIN_PX", "48"))
MAKE_SQUARE = os.environ.get("SEG_MAKE_SQUARE", "1") == "1"


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (MASKS_DIR / split).mkdir(parents=True, exist_ok=True)


def get_bbox_from_mask(mask_np: np.ndarray):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def make_crop_box(x1, y1, x2, y2, w, h, margin_ratio=0.25, min_margin_px=48, make_square=True):
    bw = x2 - x1
    bh = y2 - y1

    mx = max(int(round(bw * margin_ratio)), min_margin_px)
    my = max(int(round(bh * margin_ratio)), min_margin_px)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    if not make_square:
        return cx1, cy1, cx2, cy2

    cw = cx2 - cx1
    ch = cy2 - cy1
    side = max(cw, ch)

    cx = (cx1 + cx2) / 2.0
    cy = (cy1 + cy2) / 2.0

    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    nx2 = nx1 + side
    ny2 = ny1 + side

    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > w:
        shift = nx2 - w
        nx1 -= shift
        nx2 = w
    if ny2 > h:
        shift = ny2 - h
        ny1 -= shift
        ny2 = h

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    return nx1, ny1, nx2, ny2


def main():
    ensure_project_dirs()
    ensure_dirs()

    if not SOURCE_METADATA_CSV.exists():
        raise FileNotFoundError(f"Не найден metadata.csv: {SOURCE_METADATA_CSV}")

    df = pd.read_csv(SOURCE_METADATA_CSV)
    if len(df) == 0:
        raise ValueError("Пустой metadata.csv")

    rows = []
    skipped_empty = 0

    for _, row in df.iterrows():
        split = row["split"]

        image_path = SOURCE_DATASET_DIR / "images" / row["image_file"]
        mask_path = SOURCE_DATASET_DIR / "masks" / row["mask_file"]

        if not image_path.exists():
            raise FileNotFoundError(f"Не найдено изображение: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Не найдена маска: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np = np.array(image)
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        h, w = mask_np.shape
        bbox = get_bbox_from_mask(mask_np)
        if bbox is None:
            skipped_empty += 1
            continue

        x1, y1, x2, y2 = bbox
        cx1, cy1, cx2, cy2 = make_crop_box(
            x1, y1, x2, y2, w, h,
            margin_ratio=MARGIN_RATIO,
            min_margin_px=MIN_MARGIN_PX,
            make_square=MAKE_SQUARE,
        )

        crop_img = image.crop((cx1, cy1, cx2, cy2))
        crop_mask = mask.crop((cx1, cy1, cx2, cy2))

        crop_mask_np = (np.array(crop_mask) > 0).astype(np.uint8)
        crop_area = int(crop_mask_np.sum())
        crop_h, crop_w = crop_mask_np.shape

        base = Path(row["image_file"]).stem
        image_out_name = f"{base}_crop.jpg"
        mask_out_name = f"{base}_crop.png"

        image_out_rel = f"{split}/{image_out_name}"
        mask_out_rel = f"{split}/{mask_out_name}"

        image_out_path = IMAGES_DIR / split / image_out_name
        mask_out_path = MASKS_DIR / split / mask_out_name

        crop_img.save(image_out_path, quality=95)
        Image.fromarray((crop_mask_np * 255).astype(np.uint8)).save(mask_out_path)

        rows.append({
            "task_dir": row["task_dir"],
            "task_name": row.get("task_name", row["task_dir"]),
            "split": split,
            "frame_idx": row.get("frame_idx", -1),
            "source_image_file": row["image_file"],
            "source_mask_file": row["mask_file"],
            "image_file": image_out_rel,
            "mask_file": mask_out_rel,
            "orig_width": int(w),
            "orig_height": int(h),
            "crop_x1": int(cx1),
            "crop_y1": int(cy1),
            "crop_x2": int(cx2),
            "crop_y2": int(cy2),
            "crop_width": int(crop_w),
            "crop_height": int(crop_h),
            "mask_area_px": crop_area,
            "mask_ratio": float(crop_area / max(crop_w * crop_h, 1)),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "source_dataset_dir": str(SOURCE_DATASET_DIR),
        "source_metadata_csv": str(SOURCE_METADATA_CSV),
        "crop_dataset_dir": str(CROP_DATASET_DIR),
        "n_rows_total": int(len(out_df)),
        "n_skipped_empty": int(skipped_empty),
        "split_counts": out_df["split"].value_counts().to_dict(),
        "mask_ratio_mean": float(out_df["mask_ratio"].mean()) if len(out_df) else 0.0,
        "mask_ratio_median": float(out_df["mask_ratio"].median()) if len(out_df) else 0.0,
        "mask_ratio_min": float(out_df["mask_ratio"].min()) if len(out_df) else 0.0,
        "mask_ratio_max": float(out_df["mask_ratio"].max()) if len(out_df) else 0.0,
        "margin_ratio": MARGIN_RATIO,
        "min_margin_px": MIN_MARGIN_PX,
        "make_square": MAKE_SQUARE,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Crop dataset готов ===")
    print(f"Строк: {len(out_df)}")
    print(f"Пропущено пустых: {skipped_empty}")
    print(f"Metadata: {METADATA_CSV}")
    print(f"Summary:  {SUMMARY_JSON}")
    if len(out_df):
        print(out_df['split'].value_counts().to_string())
        print(f"mask_ratio mean:   {out_df['mask_ratio'].mean():.4f}")
        print(f"mask_ratio median: {out_df['mask_ratio'].median():.4f}")


if __name__ == "__main__":
    main()
