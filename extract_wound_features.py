from pathlib import Path
import json
import re
import math
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage import color, measure


# ============================================================
# НАСТРОЙКИ
# ============================================================

# Запускать из корня датасета: там, где лежат project.json и task_*
DATASET_ROOT = Path(".")

# Итоговый CSV
OUTPUT_CSV = DATASET_ROOT / "wound_image_features.csv"

# Минимальное число пикселей маски для вычисления цветовых признаков
MIN_PIXELS_FOR_COLOR = 25

# Минимальное число пикселей маски для вычисления текстурных признаков
MIN_PIXELS_FOR_TEXTURE = 50

# Уровни квантования для GLCM
GLCM_LEVELS = 16

# Использовать EXIF-поворот JPEG.
# Обычно для таких аннотаций лучше False, иначе маска может "съехать".
APPLY_EXIF_TRANSPOSE = False


# ============================================================
# КАРТА КЛАССОВ: РУССКОЕ ИМЯ -> БЕЗОПАСНЫЙ АНГЛИЙСКИЙ КЛЮЧ
# ============================================================

LABEL_MAP = {
    "ВсяРана": "wound",
    "Фибрин": "fibrin",
    "Металлоконструкция": "metal_device",
    "Зона шва": "suture_zone",
    "Зона отека вокруг раны": "edema_zone",
    "Зона гиперемии вокруг": "hyperemia_zone",
    "Зона некроза": "necrosis_zone",
    "Зона грануляций": "granulation_zone",
    "Метка для размерности": "scale_marker",
    "Вторичная пигментация": "secondary_pigmentation",
    "Подкожная жир.кл. без грануляций": "subcutaneous_fat_no_granulation",
    "Фасция без грануляций": "fascia_no_granulation",
    "Губка ВАК": "vac_sponge",
    "Глубины раны": "wound_depths",
    "Сухожилие": "tendon",
    "Гнойное отделяемое": "purulent_discharge",
}

# Для каких зон считать цвет и текстуру.
# Технические зоны вроде scale_marker и metal_device обычно не нужны для CIELAB/GLCM.
COLOR_TEXTURE_LABEL_KEYS = [
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

# Для каких зон считать геометрию
GEOMETRY_LABEL_KEYS = list(LABEL_MAP.values())

# Группы зон для агрегированных proxy-признаков
REPARATIVE_LABELS = [
    "granulation_zone",
    "secondary_pigmentation",
]

INFLAMMATORY_LABELS = [
    "hyperemia_zone",
    "edema_zone",
]

DEVITALIZED_LABELS = [
    "necrosis_zone",
    "fibrin",
    "purulent_discharge",
]

DEEP_STRUCTURE_LABELS = [
    "subcutaneous_fat_no_granulation",
    "fascia_no_granulation",
    "tendon",
    "wound_depths",
]

DEVICE_LABELS = [
    "metal_device",
    "vac_sponge",
]


# ============================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================

def natural_task_sort_key(path: Path):
    nums = re.findall(r"\d+", path.name)
    if nums:
        return (path.name.rstrip("0123456789_"), int(nums[-1]))
    return (path.name, -1)


def find_task_dirs(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("task")],
        key=natural_task_sort_key,
    )


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
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Не найден manifest.jsonl в {task_dir}")


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_manifest(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "name" in obj:
                items.append(obj)
    return items


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


def safe_div(a, b):
    if b is None:
        return np.nan
    if isinstance(b, (int, float, np.integer, np.floating)) and b == 0:
        return np.nan
    if pd.isna(b):
        return np.nan
    return a / b


def parse_day_and_date(name: str) -> Tuple[Optional[int], Optional[str], Optional[datetime]]:
    day_num = None
    date_str = None
    date_obj = None

    # Ищем дату YYYY-MM-DD
    m_date = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if m_date:
        date_str = m_date.group(1)
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            date_obj = None

    # Ищем day-12 / day_12 / day 12
    m_day = re.search(r"(?:^|[-_ ])day[-_ ]?(\d+)(?:$|[-_ ])", name, flags=re.IGNORECASE)
    if m_day:
        try:
            day_num = int(m_day.group(1))
        except ValueError:
            day_num = None

    return day_num, date_str, date_obj


def build_shapes_by_frame(all_shapes: List[dict]) -> Dict[int, List[dict]]:
    by_frame = defaultdict(list)
    for shape in all_shapes:
        frame = shape.get("frame")
        if frame is None:
            continue
        by_frame[frame].append(shape)
    return by_frame


def build_label_masks(
    shapes: List[dict],
    image_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    width, height = image_size
    mask_images = {}

    for shape in shapes:
        if shape.get("type") != "polygon":
            continue
        if shape.get("outside", False):
            continue

        raw_label = shape.get("label", "UNKNOWN")
        label_key = LABEL_MAP.get(raw_label)
        if label_key is None:
            continue

        points = shape.get("points", [])
        if len(points) < 6:
            continue

        if label_key not in mask_images:
            mask_images[label_key] = Image.new("L", (width, height), 0)

        polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        draw = ImageDraw.Draw(mask_images[label_key])
        draw.polygon(polygon, outline=1, fill=1)

    masks = {}
    for label_key, mask_img in mask_images.items():
        masks[label_key] = np.array(mask_img, dtype=bool)

    for label_key in GEOMETRY_LABEL_KEYS:
        if label_key not in masks:
            masks[label_key] = np.zeros((height, width), dtype=bool)

    return masks


def bbox_from_mask(mask: np.ndarray):
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return x_min, y_min, x_max, y_max, width, height


def geometry_features_from_mask(mask: np.ndarray, image_area: int, prefix: str) -> dict:
    area_px = int(mask.sum())

    out = {
        f"{prefix}_present": int(area_px > 0),
        f"{prefix}_area_px": area_px,
        f"{prefix}_area_pct_image": safe_div(area_px, image_area),
        f"{prefix}_perimeter_px": np.nan,
        f"{prefix}_perimeter_to_area": np.nan,
        f"{prefix}_circularity": np.nan,
        f"{prefix}_bbox_xmin": np.nan,
        f"{prefix}_bbox_ymin": np.nan,
        f"{prefix}_bbox_xmax": np.nan,
        f"{prefix}_bbox_ymax": np.nan,
        f"{prefix}_bbox_width_px": np.nan,
        f"{prefix}_bbox_height_px": np.nan,
        f"{prefix}_bbox_aspect_ratio": np.nan,
        f"{prefix}_extent": np.nan,
    }

    if area_px == 0:
        return out

    perimeter_px = float(measure.perimeter(mask, neighborhood=8))
    out[f"{prefix}_perimeter_px"] = perimeter_px
    out[f"{prefix}_perimeter_to_area"] = safe_div(perimeter_px, area_px)

    if perimeter_px > 0:
        out[f"{prefix}_circularity"] = 4.0 * math.pi * area_px / (perimeter_px ** 2)

    bbox = bbox_from_mask(mask)
    if bbox is not None:
        x_min, y_min, x_max, y_max, bbox_w, bbox_h = bbox
        out[f"{prefix}_bbox_xmin"] = x_min
        out[f"{prefix}_bbox_ymin"] = y_min
        out[f"{prefix}_bbox_xmax"] = x_max
        out[f"{prefix}_bbox_ymax"] = y_max
        out[f"{prefix}_bbox_width_px"] = bbox_w
        out[f"{prefix}_bbox_height_px"] = bbox_h
        out[f"{prefix}_bbox_aspect_ratio"] = safe_div(bbox_w, bbox_h)
        out[f"{prefix}_extent"] = safe_div(area_px, bbox_w * bbox_h)

    return out


def lab_stats_from_mask(lab_img: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    n = int(mask.sum())

    out = {
        f"{prefix}_lab_n_pixels": n,
        f"{prefix}_L_mean": np.nan,
        f"{prefix}_L_std": np.nan,
        f"{prefix}_L_median": np.nan,
        f"{prefix}_L_q25": np.nan,
        f"{prefix}_L_q75": np.nan,
        f"{prefix}_a_mean": np.nan,
        f"{prefix}_a_std": np.nan,
        f"{prefix}_a_median": np.nan,
        f"{prefix}_a_q25": np.nan,
        f"{prefix}_a_q75": np.nan,
        f"{prefix}_b_mean": np.nan,
        f"{prefix}_b_std": np.nan,
        f"{prefix}_b_median": np.nan,
        f"{prefix}_b_q25": np.nan,
        f"{prefix}_b_q75": np.nan,
    }

    if n < MIN_PIXELS_FOR_COLOR:
        return out

    values = lab_img[mask]
    channel_names = ["L", "a", "b"]

    for idx, ch in enumerate(channel_names):
        arr = values[:, idx]
        out[f"{prefix}_{ch}_mean"] = float(np.mean(arr))
        out[f"{prefix}_{ch}_std"] = float(np.std(arr))
        out[f"{prefix}_{ch}_median"] = float(np.median(arr))
        out[f"{prefix}_{ch}_q25"] = float(np.quantile(arr, 0.25))
        out[f"{prefix}_{ch}_q75"] = float(np.quantile(arr, 0.75))

    return out


def quantize_gray(gray_img: np.ndarray, levels: int) -> np.ndarray:
    q = np.clip(np.round(gray_img * (levels - 1)), 0, levels - 1).astype(np.int32)
    return q


def masked_glcm_features(
    gray_img: np.ndarray,
    mask: np.ndarray,
    prefix: str,
    levels: int = GLCM_LEVELS,
) -> dict:
    out = {
        f"{prefix}_glcm_pairs": 0,
        f"{prefix}_glcm_contrast": np.nan,
        f"{prefix}_glcm_energy": np.nan,
        f"{prefix}_glcm_homogeneity": np.nan,
        f"{prefix}_glcm_correlation": np.nan,
        f"{prefix}_glcm_entropy": np.nan,
        f"{prefix}_glcm_dissimilarity": np.nan,
    }

    n = int(mask.sum())
    if n < MIN_PIXELS_FOR_TEXTURE:
        return out

    q = quantize_gray(gray_img, levels)
    h, w = q.shape

    # offsets: 0°, 45°, 90°, 135°
    offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]

    P_total = np.zeros((levels, levels), dtype=np.float64)

    for dy, dx in offsets:
        if dy >= 0:
            src_y = slice(0, h - dy)
            dst_y = slice(dy, h)
        else:
            src_y = slice(-dy, h)
            dst_y = slice(0, h + dy)

        if dx >= 0:
            src_x = slice(0, w - dx)
            dst_x = slice(dx, w)
        else:
            src_x = slice(-dx, w)
            dst_x = slice(0, w + dx)

        src_mask = mask[src_y, src_x]
        dst_mask = mask[dst_y, dst_x]
        valid = src_mask & dst_mask

        if not np.any(valid):
            continue

        src_vals = q[src_y, src_x][valid].ravel()
        dst_vals = q[dst_y, dst_x][valid].ravel()

        counts = np.bincount(
            src_vals * levels + dst_vals,
            minlength=levels * levels
        ).reshape(levels, levels)

        counts = counts + counts.T
        P_total += counts

    total_pairs = P_total.sum()
    out[f"{prefix}_glcm_pairs"] = int(total_pairs)

    if total_pairs == 0:
        return out

    P = P_total / total_pairs

    I = np.arange(levels).reshape(-1, 1)
    J = np.arange(levels).reshape(1, -1)
    diff = I - J

    contrast = np.sum((diff ** 2) * P)
    dissimilarity = np.sum(np.abs(diff) * P)
    asm = np.sum(P ** 2)
    energy = np.sqrt(asm)
    homogeneity = np.sum(P / (1.0 + diff ** 2))

    mu_i = np.sum(I * P)
    mu_j = np.sum(J * P)
    sigma_i = np.sqrt(np.sum(((I - mu_i) ** 2) * P))
    sigma_j = np.sqrt(np.sum(((J - mu_j) ** 2) * P))

    if sigma_i > 0 and sigma_j > 0:
        correlation = np.sum(((I - mu_i) * (J - mu_j) * P)) / (sigma_i * sigma_j)
    else:
        correlation = np.nan

    P_nonzero = P[P > 0]
    entropy = -np.sum(P_nonzero * np.log2(P_nonzero))

    out[f"{prefix}_glcm_contrast"] = float(contrast)
    out[f"{prefix}_glcm_energy"] = float(energy)
    out[f"{prefix}_glcm_homogeneity"] = float(homogeneity)
    out[f"{prefix}_glcm_correlation"] = float(correlation) if not np.isnan(correlation) else np.nan
    out[f"{prefix}_glcm_entropy"] = float(entropy)
    out[f"{prefix}_glcm_dissimilarity"] = float(dissimilarity)

    return out


def infer_phase(row: dict) -> str:
    has_device = row.get("metal_device_present", 0) == 1
    has_suture = row.get("suture_zone_present", 0) == 1

    if has_device:
        return "device_present"
    if has_suture:
        return "sutured"
    return "open"


def first_non_nan(rows: List[dict], key: str):
    for row in rows:
        val = row.get(key, np.nan)
        if not pd.isna(val):
            return val
    return np.nan


def temporal_sort_key(row: dict):
    if row["_date_obj"] is not None:
        day_num = row["parsed_day_num"] if row["parsed_day_num"] is not None else 10**9
        return (0, row["_date_obj"], day_num, row["raw_frame"])
    if row["parsed_day_num"] is not None:
        return (1, row["parsed_day_num"], row["raw_frame"])
    return (2, row["raw_frame"])


def add_scale_normalized_geometry(row: dict):
    scale_area = row.get("scale_marker_area_px", 0)
    scale_area_sqrt = math.sqrt(scale_area) if scale_area and scale_area > 0 else np.nan

    for label_key in GEOMETRY_LABEL_KEYS:
        row[f"{label_key}_area_norm_scale"] = safe_div(row.get(f"{label_key}_area_px", np.nan), scale_area)
        row[f"{label_key}_perimeter_norm_scale"] = safe_div(row.get(f"{label_key}_perimeter_px", np.nan), scale_area_sqrt)

    return row


def add_wound_relative_geometry(row: dict):
    wound_area = row.get("wound_area_px", np.nan)
    wound_perimeter = row.get("wound_perimeter_px", np.nan)

    for label_key in GEOMETRY_LABEL_KEYS:
        if label_key == "wound":
            continue
        row[f"{label_key}_area_pct_wound"] = safe_div(row.get(f"{label_key}_area_px", np.nan), wound_area)
        row[f"{label_key}_perimeter_pct_wound"] = safe_div(row.get(f"{label_key}_perimeter_px", np.nan), wound_perimeter)

    row["suture_to_wound_area_ratio"] = safe_div(
        row.get("suture_zone_area_px", np.nan),
        wound_area
    )
    row["suture_to_wound_perimeter_ratio"] = safe_div(
        row.get("suture_zone_perimeter_px", np.nan),
        wound_perimeter
    )

    wound_non_suture = np.nan
    if not pd.isna(wound_area) and not pd.isna(row.get("suture_zone_area_px", np.nan)):
        wound_non_suture = max(0.0, row["wound_area_px"] - row["suture_zone_area_px"])
    row["wound_non_suture_area_px"] = wound_non_suture
    row["wound_non_suture_area_pct_wound"] = safe_div(wound_non_suture, wound_area)

    return row


def add_proxy_burden_features(row: dict):
    wound_area = row.get("wound_area_px", np.nan)

    def sum_area(labels: List[str]) -> float:
        vals = [row.get(f"{lbl}_area_px", 0) for lbl in labels]
        vals = [0 if pd.isna(v) else v for v in vals]
        return float(sum(vals))

    reparative_area = sum_area(REPARATIVE_LABELS)
    inflammatory_area = sum_area(INFLAMMATORY_LABELS)
    devitalized_area = sum_area(DEVITALIZED_LABELS)
    deep_structure_area = sum_area(DEEP_STRUCTURE_LABELS)
    device_area = sum_area(DEVICE_LABELS)

    row["reparative_area_px"] = reparative_area
    row["inflammatory_area_px"] = inflammatory_area
    row["devitalized_area_px"] = devitalized_area
    row["deep_structure_area_px"] = deep_structure_area
    row["device_related_area_px"] = device_area

    row["reparative_area_pct_wound"] = safe_div(reparative_area, wound_area)
    row["inflammatory_area_pct_wound"] = safe_div(inflammatory_area, wound_area)
    row["devitalized_area_pct_wound"] = safe_div(devitalized_area, wound_area)
    row["deep_structure_area_pct_wound"] = safe_div(deep_structure_area, wound_area)
    row["device_related_area_pct_wound"] = safe_div(device_area, wound_area)

    row["infection_related_proxy_flag"] = int(
        row.get("purulent_discharge_present", 0) == 1 or row.get("necrosis_zone_present", 0) == 1
    )

    row["inflammation_related_proxy_flag"] = int(
        row.get("hyperemia_zone_present", 0) == 1
        or row.get("edema_zone_present", 0) == 1
        or row.get("purulent_discharge_present", 0) == 1
    )

    row["reparative_proxy_flag"] = int(
        row.get("granulation_zone_present", 0) == 1
        or row.get("secondary_pigmentation_present", 0) == 1
    )

    row["deep_damage_proxy_flag"] = int(
        row.get("subcutaneous_fat_no_granulation_present", 0) == 1
        or row.get("fascia_no_granulation_present", 0) == 1
        or row.get("tendon_present", 0) == 1
        or row.get("wound_depths_present", 0) == 1
    )

    # Это не диагноз, а количественный proxy-баланс тканей в ране
    row["healing_balance_score"] = (
        (0 if pd.isna(row["reparative_area_pct_wound"]) else row["reparative_area_pct_wound"])
        - (0 if pd.isna(row["inflammatory_area_pct_wound"]) else row["inflammatory_area_pct_wound"])
        - (0 if pd.isna(row["devitalized_area_pct_wound"]) else row["devitalized_area_pct_wound"])
        - (0 if pd.isna(row["deep_structure_area_pct_wound"]) else row["deep_structure_area_pct_wound"])
    )

    return row


def add_color_difference_features(row: dict):
    for ch in ["L", "a", "b"]:
        wound_val = row.get(f"wound_{ch}_mean", np.nan)
        suture_val = row.get(f"suture_zone_{ch}_mean", np.nan)

        row[f"suture_minus_wound_{ch}_mean"] = (
            suture_val - wound_val
            if not pd.isna(suture_val) and not pd.isna(wound_val)
            else np.nan
        )

    return row


def add_time_and_phase_features(rows: List[dict]) -> List[dict]:
    if not rows:
        return rows

    rows = sorted(rows, key=temporal_sort_key)

    task_n_images = len(rows)
    is_single = int(task_n_images == 1)

    for idx, row in enumerate(rows):
        row["task_time_index"] = idx
        row["task_n_images"] = task_n_images
        row["is_single_image_task"] = is_single
        row["usable_for_task_dynamics"] = int(task_n_images >= 2)

    base_date = rows[0]["_date_obj"]
    base_day = rows[0]["parsed_day_num"]

    for row in rows:
        if base_date is not None and row["_date_obj"] is not None:
            row["task_time_from_start"] = (row["_date_obj"] - base_date).days
        elif base_day is not None and row["parsed_day_num"] is not None:
            row["task_time_from_start"] = row["parsed_day_num"] - base_day
        else:
            row["task_time_from_start"] = row["task_time_index"]

    for row in rows:
        row["phase_label"] = infer_phase(row)

    segment_id = 0
    prev_phase = None
    for row in rows:
        phase = row["phase_label"]
        if prev_phase is None:
            segment_id = 0
        elif phase != prev_phase:
            segment_id += 1
        row["phase_segment_id"] = segment_id
        prev_phase = phase

    segment_sizes = Counter(row["phase_segment_id"] for row in rows)
    for row in rows:
        seg_id = row["phase_segment_id"]
        row["phase_segment_size"] = segment_sizes[seg_id]
        row["usable_for_phase_dynamics"] = int(segment_sizes[seg_id] >= 2)

    segments = defaultdict(list)
    for row in rows:
        segments[row["phase_segment_id"]].append(row)

    for seg_id, seg_rows in segments.items():
        seg_rows = sorted(seg_rows, key=temporal_sort_key)
        seg_base_date = seg_rows[0]["_date_obj"]
        seg_base_day = seg_rows[0]["parsed_day_num"]

        for idx, row in enumerate(seg_rows):
            row["phase_time_index"] = idx

            if seg_base_date is not None and row["_date_obj"] is not None:
                row["phase_time_from_start"] = (row["_date_obj"] - seg_base_date).days
            elif seg_base_day is not None and row["parsed_day_num"] is not None:
                row["phase_time_from_start"] = row["parsed_day_num"] - seg_base_day
            else:
                row["phase_time_from_start"] = idx

    # Относительные признаки к началу task и фазы:
    # теперь считаем для всех зон по площади и периметру
    rel_keys = []
    for label_key in GEOMETRY_LABEL_KEYS:
        rel_keys.append(f"{label_key}_area_px")
        rel_keys.append(f"{label_key}_perimeter_px")

    task_starts = {k: first_non_nan(rows, k) for k in rel_keys}
    for row in rows:
        for k in rel_keys:
            row[f"{k}_rel_task_start"] = safe_div(row.get(k, np.nan), task_starts[k])

    for seg_id, seg_rows in segments.items():
        seg_starts = {k: first_non_nan(seg_rows, k) for k in rel_keys}
        for row in seg_rows:
            for k in rel_keys:
                row[f"{k}_rel_phase_start"] = safe_div(row.get(k, np.nan), seg_starts[k])

    return rows


# ============================================================
# ОСНОВНАЯ ЛОГИКА ПО ОДНОМУ TASK
# ============================================================

def process_task(task_dir: Path) -> List[dict]:
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
    all_shapes = flatten_shapes(annotations)
    shapes_by_frame = build_shapes_by_frame(all_shapes)

    rows = []

    for frame_idx, rec in enumerate(manifest):
        file_name = rec["name"] + rec["extension"]
        img_path = data_dir / file_name

        if not img_path.exists():
            print(f"[WARNING] Пропускаю: не найдено изображение {img_path}")
            continue

        img = open_image(img_path)
        width, height = img.size
        image_area = width * height

        parsed_day_num, parsed_date_str, parsed_date_obj = parse_day_and_date(rec["name"])

        frame_shapes = shapes_by_frame.get(frame_idx, [])
        masks = build_label_masks(frame_shapes, (width, height))

        rgb = np.asarray(img, dtype=np.float32) / 255.0
        lab = color.rgb2lab(rgb)
        gray = color.rgb2gray(rgb)

        row = {
            "task_dir": task_dir.name,
            "task_name": task_name,
            "raw_frame": frame_idx,
            "file_name": file_name,
            "image_path": str(img_path),
            "image_width": width,
            "image_height": height,
            "image_area_px": image_area,
            "parsed_day_num": parsed_day_num,
            "parsed_date": parsed_date_str,
            "_date_obj": parsed_date_obj,
        }

        # Геометрия по всем классам
        present_labels = []
        for label_key in GEOMETRY_LABEL_KEYS:
            mask = masks[label_key]
            geom = geometry_features_from_mask(mask, image_area, label_key)
            row.update(geom)
            if geom[f"{label_key}_present"] == 1:
                present_labels.append(label_key)

        row["n_present_labels"] = len(present_labels)
        row["present_labels"] = ";".join(sorted(present_labels))

        # CIELAB + GLCM по всем клинически значимым зонам
        for label_key in COLOR_TEXTURE_LABEL_KEYS:
            mask = masks.get(label_key, np.zeros((height, width), dtype=bool))
            row.update(lab_stats_from_mask(lab, mask, label_key))
            row.update(masked_glcm_features(gray, mask, label_key, levels=GLCM_LEVELS))

        # Нормировки и производные frame-level признаки
        row = add_scale_normalized_geometry(row)
        row = add_wound_relative_geometry(row)
        row = add_proxy_burden_features(row)
        row = add_color_difference_features(row)

        rows.append(row)

    rows = add_time_and_phase_features(rows)

    for row in rows:
        row.pop("_date_obj", None)

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    task_dirs = find_task_dirs(DATASET_ROOT)
    if not task_dirs:
        print("Папки task_* не найдены.")
        return

    all_rows = []
    print(f"Найдено task-папок: {len(task_dirs)}")

    for task_dir in task_dirs:
        try:
            rows = process_task(task_dir)
            all_rows.extend(rows)
            print(f"[OK] {task_dir.name}: {len(rows)} изображений")
        except Exception as e:
            print(f"[ERROR] {task_dir.name}: {e}")

    if not all_rows:
        print("Не удалось собрать ни одной строки признаков.")
        return

    df = pd.DataFrame(all_rows)

    sort_cols = []
    for c in ["task_dir", "task_time_index", "raw_frame"]:
        if c in df.columns:
            sort_cols.append(c)
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nГотово. CSV сохранён в: {OUTPUT_CSV.resolve()}")
    print(f"Строк: {len(df)}")
    print(f"Столбцов: {len(df.columns)}")


if __name__ == "__main__":
    main()