from pathlib import Path
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config_paths import RAW_DATA_DIR, ensure_project_dirs


SEG_DATASET_DIR = RAW_DATA_DIR.parent / "segmentation_dataset"
IMAGES_DIR = SEG_DATASET_DIR / "images"
MASKS_DIR = SEG_DATASET_DIR / "masks"
METADATA_CSV = SEG_DATASET_DIR / "metadata.csv"
SPLIT_JSON = SEG_DATASET_DIR / "split_tasks.json"
CLASS_MAP_JSON = SEG_DATASET_DIR / "class_map.json"
SUMMARY_JSON = SEG_DATASET_DIR / "dataset_summary.json"

TARGET_LABEL = "ВсяРана"
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
APPLY_EXIF_TRANSPOSE = False


def natural_task_sort_key(path: Path):
    nums = re.findall(r"\d+", path.name)
    if nums:
        return (path.name.rstrip("0123456789_"), int(nums[-1]))
    return (path.name, -1)


def find_task_dirs(root: Path) -> List[Path]:
    task_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("task")]
    return sorted(task_dirs, key=natural_task_sort_key)


def find_data_dir(task_dir: Path) -> Path:
    for candidate in (task_dir / "Data", task_dir / "data"):
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Не найдена папка Data/data в {task_dir}")


def find_manifest_path(task_dir: Path, data_dir: Path) -> Path:
    candidates = [
        data_dir / "manifest.jsonl",
        task_dir / "manifest.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Не найден manifest.jsonl в {task_dir}")


def read_manifest(manifest_path: Path) -> List[dict]:
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "name" in obj:
                records.append(obj)
    return records


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if APPLY_EXIF_TRANSPOSE:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def flatten_shapes(annotations_obj) -> List[dict]:
    all_shapes = []
    if isinstance(annotations_obj, list):
        for item in annotations_obj:
            if isinstance(item, dict):
                all_shapes.extend(item.get("shapes", []))
    elif isinstance(annotations_obj, dict):
        all_shapes.extend(annotations_obj.get("shapes", []))
    return all_shapes


def build_shapes_by_frame(all_shapes: List[dict]) -> Dict[int, List[dict]]:
    by_frame = defaultdict(list)
    for shape in all_shapes:
        frame = shape.get("frame")
        if frame is None:
            continue
        by_frame[frame].append(shape)
    return by_frame


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def build_binary_wound_mask(shapes: List[dict], image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    found = False
    for shape in shapes:
        if shape.get("type") != "polygon":
            continue
        if shape.get("outside", False):
            continue
        if shape.get("label") != TARGET_LABEL:
            continue

        pts = shape.get("points", [])
        if len(pts) < 6:
            continue

        polygon = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
        draw.polygon(polygon, fill=1, outline=1)
        found = True

    if not found:
        return np.zeros((height, width), dtype=np.uint8)

    return np.asarray(mask_img, dtype=np.uint8)


def split_tasks(task_names: List[str]) -> Dict[str, str]:
    rng = np.random.RandomState(SEED)
    task_names = list(sorted(task_names))
    rng.shuffle(task_names)

    n = len(task_names)
    n_train = max(1, int(round(n * TRAIN_RATIO)))
    n_val = max(1, int(round(n * VAL_RATIO)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    train_tasks = set(task_names[:n_train])
    val_tasks = set(task_names[n_train:n_train + n_val])
    test_tasks = set(task_names[n_train + n_val:])

    mapping = {}
    for t in task_names:
        if t in train_tasks:
            mapping[t] = "train"
        elif t in val_tasks:
            mapping[t] = "val"
        else:
            mapping[t] = "test"
    return mapping


def ensure_seg_dirs():
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (MASKS_DIR / split).mkdir(parents=True, exist_ok=True)


def process_task(task_dir: Path, split_map: Dict[str, str]) -> Tuple[List[dict], List[str], int, int]:
    data_dir = find_data_dir(task_dir)
    manifest_path = find_manifest_path(task_dir, data_dir)
    annotations_path = task_dir / "annotations.json"
    task_json_path = task_dir / "task.json"

    if not annotations_path.exists():
        raise FileNotFoundError(f"Не найден {annotations_path}")
    if not task_json_path.exists():
        raise FileNotFoundError(f"Не найден {task_json_path}")

    manifest = read_manifest(manifest_path)
    annotations = read_json(annotations_path)
    task_info = read_json(task_json_path)

    task_name = task_info.get("name", task_dir.name)
    split = split_map[task_dir.name]

    all_shapes = flatten_shapes(annotations)
    shapes_by_frame = build_shapes_by_frame(all_shapes)

    rows: List[dict] = []
    warnings: List[str] = []
    saved_count = 0
    skipped_empty_masks = 0

    for frame_idx, rec in enumerate(manifest):
        filename = rec["name"] + rec["extension"]
        img_path = data_dir / filename

        if not img_path.exists():
            warnings.append(f"[WARN] {task_dir.name}: не найдено изображение {rec['name']}")
            continue

        img = open_image(img_path)
        width, height = img.size
        frame_shapes = shapes_by_frame.get(frame_idx, [])

        mask = build_binary_wound_mask(frame_shapes, (width, height))
        if int(mask.sum()) == 0:
            skipped_empty_masks += 1
            continue

        safe_base = sanitize_filename(rec["name"])
        image_out_name = f"{task_dir.name}__{frame_idx:04d}__{safe_base}{rec['extension']}"
        mask_out_name = f"{task_dir.name}__{frame_idx:04d}__{safe_base}.png"

        image_out_path = IMAGES_DIR / split / image_out_name
        mask_out_path = MASKS_DIR / split / mask_out_name

        img.save(image_out_path)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_out_path)

        rows.append({
            "task_dir": task_dir.name,
            "task_name": task_name,
            "split": split,
            "frame_idx": frame_idx,
            "image_file": f"{split}/{image_out_name}",
            "mask_file": f"{split}/{mask_out_name}",
            "raw_name": rec["name"],
            "width": width,
            "height": height,
            "mask_area_px": int(mask.sum()),
            "n_shapes_total": int(len(frame_shapes)),
            "has_wound_mask": 1,
        })
        saved_count += 1

    return rows, warnings, saved_count, skipped_empty_masks


def main():
    ensure_project_dirs()
    ensure_seg_dirs()

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка с исходными данными: {RAW_DATA_DIR}")

    task_dirs = find_task_dirs(RAW_DATA_DIR)
    if not task_dirs:
        raise ValueError(f"Не найдено ни одной task_* в {RAW_DATA_DIR}")

    print(f"Найдено task-папок: {len(task_dirs)}")

    split_map = split_tasks([t.name for t in task_dirs])

    all_rows: List[dict] = []
    all_warnings: List[str] = []
    total_saved = 0
    total_skipped_empty = 0

    for task_dir in task_dirs:
        try:
            rows, warnings, saved_count, skipped_empty_masks = process_task(task_dir, split_map)
            all_rows.extend(rows)
            all_warnings.extend(warnings)
            total_saved += saved_count
            total_skipped_empty += skipped_empty_masks
            print(f"[OK] {task_dir.name}: сохранено {saved_count} | пропущено пустых масок: {skipped_empty_masks}")
        except Exception as e:
            print(f"[ERROR] {task_dir.name}: {e}")

    if not all_rows:
        raise ValueError("Не удалось сформировать ни одной строки segmentation_dataset")

    df = pd.DataFrame(all_rows).sort_values(["split", "task_dir", "frame_idx"]).reset_index(drop=True)
    df.to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")

    with open(SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(split_map, f, ensure_ascii=False, indent=2)

    with open(CLASS_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({"background": 0, "wound": 1, "target_label": TARGET_LABEL}, f, ensure_ascii=False, indent=2)

    summary = {
        "n_tasks_total": len(task_dirs),
        "n_rows_total": int(len(df)),
        "n_saved_images": int(total_saved),
        "n_skipped_empty_masks": int(total_skipped_empty),
        "split_counts": df["split"].value_counts().to_dict(),
        "task_split_counts": {
            "train": sum(1 for v in split_map.values() if v == "train"),
            "val": sum(1 for v in split_map.values() if v == "val"),
            "test": sum(1 for v in split_map.values() if v == "test"),
        },
        "image_counts_by_split": (
            df.groupby("split")["image_file"].count().to_dict()
            if len(df) > 0 else {}
        ),
        "target_label": TARGET_LABEL,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    warnings_log = SEG_DATASET_DIR / "prepare_dataset_warnings.log"
    if all_warnings:
        with open(warnings_log, "w", encoding="utf-8") as f:
            for line in all_warnings:
                f.write(line + "\n")

    print("\n=== Итог ===")
    print(f"Сохранено строк: {len(df)}")
    print(f"Сохранено изображений: {total_saved}")
    print(f"Пропущено пустых масок: {total_skipped_empty}")
    print(f"Metadata: {METADATA_CSV}")
    print(f"Summary:  {SUMMARY_JSON}")
    if all_warnings:
        print(f"Warnings: {warnings_log}")


if __name__ == "__main__":
    main()
