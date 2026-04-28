from pathlib import Path
import json
import re
from collections import defaultdict
from PIL import Image, ImageDraw

# Запускать из корневой папки датасета, где лежат task_* и project.json
DATASET_ROOT = Path(".")

# Более подходящее имя для выходной папки
OUTPUT_DIR_NAME = "_segmentation_overlays"

# Если True, Pillow применит EXIF-поворот.
# Для таких аннотаций обычно лучше оставлять False, иначе полигоны могут "съехать".
APPLY_EXIF_TRANSPOSE = False


def natural_task_sort_key(path: Path):
    nums = re.findall(r"\d+", path.name)
    if nums:
        return (path.name.rstrip("0123456789_"), int(nums[-1]))
    return (path.name, -1)


def find_task_dirs(root: Path):
    task_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("task")]
    return sorted(task_dirs, key=natural_task_sort_key)


def find_data_dir(task_dir: Path):
    for candidate in (task_dir / "Data", task_dir / "data"):
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Не найдена папка Data/data в {task_dir}")


def find_manifest_path(task_dir: Path, data_dir: Path):
    candidates = [
        data_dir / "manifest.jsonl",
        task_dir / "manifest.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Не найден manifest.jsonl в {task_dir}")


def read_manifest(manifest_path: Path):
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


def read_label_colors(task_json_path: Path):
    task_info = read_json(task_json_path)
    colors = {}
    for lbl in task_info.get("labels", []):
        colors[lbl["name"]] = lbl["color"]
    return colors


def flatten_shapes(annotations_obj):
    all_shapes = []

    if isinstance(annotations_obj, list):
        for item in annotations_obj:
            if isinstance(item, dict):
                all_shapes.extend(item.get("shapes", []))
    elif isinstance(annotations_obj, dict):
        all_shapes.extend(annotations_obj.get("shapes", []))

    return all_shapes


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def check_points_within_image(points, width, height):
    bad_points = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]
        if x < 0 or x > width or y < 0 or y > height:
            bad_points.append((x, y))
    return bad_points


def draw_polygons(image, shapes, label_colors):
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for shape in shapes:
        if shape.get("type") != "polygon":
            continue
        if shape.get("outside", False):
            continue

        label = shape.get("label", "UNKNOWN")
        points = shape.get("points", [])
        if len(points) < 6:
            continue

        polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]

        color_hex = label_colors.get(label, "#00FF00")
        color_rgb = hex_to_rgb(color_hex)

        fill = (*color_rgb, 60)
        outline = (*color_rgb, 255)

        draw.polygon(polygon, fill=fill, outline=outline)

        x0, y0 = polygon[0]
        draw.text((x0 + 3, y0 + 3), label, fill=(255, 255, 255, 255))

    return img


def open_image(path: Path):
    img = Image.open(path)
    if APPLY_EXIF_TRANSPOSE:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    return img


def build_shapes_by_frame(all_shapes):
    shapes_by_frame = defaultdict(list)
    for shape in all_shapes:
        frame_idx = shape.get("frame")
        if frame_idx is None:
            continue
        shapes_by_frame[frame_idx].append(shape)
    return shapes_by_frame


def sanitize_filename(name: str):
    # Оставляем unicode, убираем только совсем проблемные символы для путей
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def process_task(task_dir: Path):
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
    label_colors = read_label_colors(task_json_path)

    all_shapes = flatten_shapes(annotations)
    shapes_by_frame = build_shapes_by_frame(all_shapes)

    output_dir = task_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(exist_ok=True)

    warnings = []
    processed = 0

    for frame_idx, rec in enumerate(manifest):
        filename = rec["name"] + rec["extension"]
        img_path = data_dir / filename

        if not img_path.exists():
            warnings.append(f"[{task_dir.name}] Нет файла изображения: {img_path}")
            continue

        img = open_image(img_path)
        width, height = img.size

        expected_w = rec.get("width")
        expected_h = rec.get("height")

        if expected_w is not None and expected_h is not None and (width, height) != (expected_w, expected_h):
            warnings.append(
                f"[{task_dir.name}] {filename}: размер файла {width}x{height}, "
                f"в manifest {expected_w}x{expected_h}"
            )

        frame_shapes = shapes_by_frame.get(frame_idx, [])

        for shape in frame_shapes:
            if shape.get("type") != "polygon":
                continue
            bad_points = check_points_within_image(shape.get("points", []), width, height)
            if bad_points:
                warnings.append(
                    f"[{task_dir.name}] frame {frame_idx}, {filename}, label={shape.get('label')} "
                    f"имеет точки вне изображения: {bad_points[:5]}"
                )

        overlay = draw_polygons(img, frame_shapes, label_colors)

        safe_base = sanitize_filename(rec["name"])
        out_path = output_dir / f"{frame_idx:04d}_{safe_base}.png"
        overlay.save(out_path)

        processed += 1

    return processed, len(manifest), warnings


def main():
    task_dirs = find_task_dirs(DATASET_ROOT)

    if not task_dirs:
        print("Папки task_* не найдены.")
        return

    total_tasks = 0
    total_images_processed = 0
    total_images_expected = 0
    all_warnings = []

    print(f"Найдено task-папок: {len(task_dirs)}")

    for task_dir in task_dirs:
        try:
            processed, expected, warnings = process_task(task_dir)
            total_tasks += 1
            total_images_processed += processed
            total_images_expected += expected
            all_warnings.extend(warnings)
            print(f"[OK] {task_dir.name}: сохранено {processed}/{expected}")
        except Exception as e:
            print(f"[ERROR] {task_dir.name}: {e}")

    print("\n=== Итог ===")
    print(f"Обработано task-папок: {total_tasks}")
    print(f"Сохранено изображений: {total_images_processed}/{total_images_expected}")

    if all_warnings:
        log_path = DATASET_ROOT / "segmentation_warnings.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for line in all_warnings:
                f.write(line + "\n")
        print(f"Предупреждения сохранены в: {log_path}")
    else:
        print("Предупреждений нет.")


if __name__ == "__main__":
    main()