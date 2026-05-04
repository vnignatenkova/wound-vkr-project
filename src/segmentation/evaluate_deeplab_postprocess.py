from pathlib import Path
import os
import json
from collections import deque

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.segmentation import deeplabv3_resnet50

from src.config_paths import PROJECT_ROOT, ensure_project_dirs

try:
    from scipy import ndimage as ndi
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


SEG_DATASET_DIR = Path(os.environ.get("SEG_DATASET_DIR", str(PROJECT_ROOT / "data" / "segmentation_dataset")))
METADATA_CSV = Path(os.environ.get("SEG_METADATA_CSV", str(SEG_DATASET_DIR / "metadata.csv")))

RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"
MODEL_PATH = RESULTS_DIR / "deeplab_best_model.pth"
TRAIN_METRICS_JSON = RESULTS_DIR / "deeplab_metrics.json"

SEARCH_CSV = RESULTS_DIR / "deeplab_postprocess_search.csv"
BEST_JSON = RESULTS_DIR / "deeplab_postprocess_best.json"
TEST_PER_IMAGE_CSV = RESULTS_DIR / "deeplab_postprocess_test_per_image.csv"

BATCH_SIZE = 4
NUM_WORKERS = 0
THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.91, 0.02)]
POSTPROCESS_MODES = ["none", "largest_cc", "largest_cc_fill_holes"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image_size(default=384):
    if TRAIN_METRICS_JSON.exists():
        try:
            data = json.loads(TRAIN_METRICS_JSON.read_text(encoding="utf-8"))
            return int(data.get("image_size", default))
        except Exception:
            pass
    return default


class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, image_size: int):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = SEG_DATASET_DIR / "images" / row["image_file"]
        mask_path = SEG_DATASET_DIR / "masks" / row["mask_file"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = (np.array(mask) > 0).astype(np.uint8)

        return {
            "image": image,
            "mask": mask,
            "image_file": row["image_file"],
            "mask_file": row["mask_file"],
            "task_dir": row["task_dir"],
            "split": row["split"],
        }


def build_model():
    return deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=1,
        aux_loss=False,
    )


def get_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["out"]
    return model_output


@torch.no_grad()
def collect_probs_and_targets(model, loader, device):
    model.eval()

    probs_all = []
    targets_all = []
    meta_rows = []

    for batch in loader:
        images = batch["image"].to(device)
        logits = get_logits(model(images))
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

        masks = batch["mask"].numpy()

        probs_all.append(probs)
        targets_all.append(masks)

        for i in range(len(batch["image_file"])):
            meta_rows.append({
                "image_file": batch["image_file"][i],
                "mask_file": batch["mask_file"][i],
                "task_dir": batch["task_dir"][i],
                "split": batch["split"][i],
            })

    probs_all = np.concatenate(probs_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    return probs_all, targets_all, meta_rows


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask.astype(np.uint8)

    if SCIPY_AVAILABLE:
        labeled, num = ndi.label(mask)
        if num == 0:
            return mask.astype(np.uint8)
        sizes = ndi.sum(mask, labeled, index=np.arange(1, num + 1))
        largest_label = int(np.argmax(sizes)) + 1
        return (labeled == largest_label).astype(np.uint8)

    # fallback pure python BFS
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    best_coords = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = 1
            coords = [(y, x)]

            while q:
                cy, cx = q.popleft()
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))
                        coords.append((ny, nx))

            if len(coords) > len(best_coords):
                best_coords = coords

    out = np.zeros((h, w), dtype=np.uint8)
    for y, x in best_coords:
        out[y, x] = 1
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask.astype(np.uint8)

    if SCIPY_AVAILABLE:
        return ndi.binary_fill_holes(mask).astype(np.uint8)

    # fallback flood fill background from borders
    h, w = mask.shape
    inv = ~mask
    visited = np.zeros((h, w), dtype=np.uint8)
    q = deque()

    for x in range(w):
        if inv[0, x]:
            q.append((0, x))
            visited[0, x] = 1
        if inv[h - 1, x] and not visited[h - 1, x]:
            q.append((h - 1, x))
            visited[h - 1, x] = 1

    for y in range(h):
        if inv[y, 0] and not visited[y, 0]:
            q.append((y, 0))
            visited[y, 0] = 1
        if inv[y, w - 1] and not visited[y, w - 1]:
            q.append((y, w - 1))
            visited[y, w - 1] = 1

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        cy, cx = q.popleft()
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and inv[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = 1
                q.append((ny, nx))

    holes = inv & (~visited.astype(bool))
    filled = mask | holes
    return filled.astype(np.uint8)


def apply_postprocess(pred: np.ndarray, mode: str) -> np.ndarray:
    pred = pred.astype(np.uint8)

    if mode == "none":
        return pred
    if mode == "largest_cc":
        return largest_connected_component(pred)
    if mode == "largest_cc_fill_holes":
        pred = largest_connected_component(pred)
        pred = fill_holes(pred)
        return pred

    raise ValueError(f"Unknown mode: {mode}")


def binary_metrics(pred: np.ndarray, target: np.ndarray):
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    tp = int(((pred == 1) & (target == 1)).sum())
    fp = int(((pred == 1) & (target == 0)).sum())
    fn = int(((pred == 0) & (target == 1)).sum())

    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_probs(probs_all: np.ndarray, targets_all: np.ndarray, threshold: float, mode: str):
    rows = []

    for i in range(len(probs_all)):
        pred = (probs_all[i] >= threshold).astype(np.uint8)
        pred = apply_postprocess(pred, mode)
        target = targets_all[i].astype(np.uint8)

        rows.append(binary_metrics(pred, target))

    df = pd.DataFrame(rows)
    return {
        "dice": float(df["dice"].mean()),
        "iou": float(df["iou"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
    }, df


def main():
    ensure_project_dirs()

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Не найден metadata.csv: {METADATA_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена модель: {MODEL_PATH}")

    image_size = load_image_size(default=384)
    device = get_device()

    df = pd.read_csv(METADATA_CSV)
    print(f"metadata: {METADATA_CSV}")
    print(f"rows: {len(df)}")
    print(df["split"].value_counts().to_string())
    print(f"device: {device}")
    print(f"image_size: {image_size}")
    print(f"scipy_available: {SCIPY_AVAILABLE}")

    val_ds = SegmentationDataset(df, "val", image_size=image_size)
    test_ds = SegmentationDataset(df, "test", image_size=image_size)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    val_probs, val_targets, _ = collect_probs_and_targets(model, val_loader, device)
    test_probs, test_targets, test_meta = collect_probs_and_targets(model, test_loader, device)

    search_rows = []
    best_cfg = None
    best_val_dice = -1.0

    for thr in THRESHOLDS:
        for mode in POSTPROCESS_MODES:
            metrics, _ = evaluate_probs(val_probs, val_targets, thr, mode)
            row = {
                "threshold": thr,
                "postprocess_mode": mode,
                "val_dice": metrics["dice"],
                "val_iou": metrics["iou"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
            }
            search_rows.append(row)

            if metrics["dice"] > best_val_dice:
                best_val_dice = metrics["dice"]
                best_cfg = row.copy()

    search_df = pd.DataFrame(search_rows).sort_values(
        ["val_dice", "val_iou", "val_precision"], ascending=False
    ).reset_index(drop=True)
    search_df.to_csv(SEARCH_CSV, index=False, encoding="utf-8-sig")

    best_thr = float(best_cfg["threshold"])
    best_mode = best_cfg["postprocess_mode"]

    test_metrics, test_df = evaluate_probs(test_probs, test_targets, best_thr, best_mode)

    # per-image metrics for test
    per_image_rows = []
    for i in range(len(test_probs)):
        pred = (test_probs[i] >= best_thr).astype(np.uint8)
        pred = apply_postprocess(pred, best_mode)
        target = test_targets[i].astype(np.uint8)
        m = binary_metrics(pred, target)
        meta = test_meta[i]
        per_image_rows.append({
            **meta,
            "threshold": best_thr,
            "postprocess_mode": best_mode,
            "dice": m["dice"],
            "iou": m["iou"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    pd.DataFrame(per_image_rows).to_csv(TEST_PER_IMAGE_CSV, index=False, encoding="utf-8-sig")

    result = {
        "metadata_csv": str(METADATA_CSV),
        "model_path": str(MODEL_PATH),
        "image_size": image_size,
        "best_on_val": best_cfg,
        "test_metrics_with_best_val_config": test_metrics,
        "search_csv": str(SEARCH_CSV),
        "test_per_image_csv": str(TEST_PER_IMAGE_CSV),
    }
    BEST_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BEST ON VAL ===")
    print(f"threshold      = {best_thr:.2f}")
    print(f"postprocess    = {best_mode}")
    print(f"val_dice       = {best_cfg['val_dice']:.4f}")
    print(f"val_iou        = {best_cfg['val_iou']:.4f}")
    print(f"val_precision  = {best_cfg['val_precision']:.4f}")
    print(f"val_recall     = {best_cfg['val_recall']:.4f}")

    print("\n=== TEST WITH BEST VAL CONFIG ===")
    print(f"test_dice      = {test_metrics['dice']:.4f}")
    print(f"test_iou       = {test_metrics['iou']:.4f}")
    print(f"test_precision = {test_metrics['precision']:.4f}")
    print(f"test_recall    = {test_metrics['recall']:.4f}")

    print("\nСохранено:")
    print(f"- {SEARCH_CSV}")
    print(f"- {BEST_JSON}")
    print(f"- {TEST_PER_IMAGE_CSV}")


if __name__ == "__main__":
    main()
