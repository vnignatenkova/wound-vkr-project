from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.segmentation import deeplabv3_resnet50

from src.config_paths import PROJECT_ROOT, ensure_project_dirs


SEG_DATASET_DIR = Path(os.environ.get("SEG_DATASET_DIR", str(PROJECT_ROOT / "data" / "segmentation_dataset")))
METADATA_CSV = Path(os.environ.get("SEG_METADATA_CSV", str(SEG_DATASET_DIR / "metadata.csv")))

RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"
MODEL_PATH = RESULTS_DIR / "deeplab_best_model.pth"
TRAIN_METRICS_JSON = RESULTS_DIR / "deeplab_metrics.json"

SEARCH_CSV = RESULTS_DIR / "deeplab_tta_search.csv"
BEST_JSON = RESULTS_DIR / "deeplab_tta_best.json"
TEST_PER_IMAGE_CSV = RESULTS_DIR / "deeplab_tta_test_per_image.csv"

BATCH_SIZE = 4
NUM_WORKERS = 0
THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.91, 0.02)]


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


def apply_tta(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "hflip":
        return torch.flip(x, dims=[3])
    if mode == "vflip":
        return torch.flip(x, dims=[2])
    if mode == "hvflip":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(f"Unknown TTA mode: {mode}")


def undo_tta(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "hflip":
        return torch.flip(x, dims=[3])
    if mode == "vflip":
        return torch.flip(x, dims=[2])
    if mode == "hvflip":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(f"Unknown TTA mode: {mode}")


@torch.no_grad()
def collect_probs_and_targets_tta(model, loader, device):
    model.eval()

    probs_all = []
    targets_all = []
    meta_rows = []

    tta_modes = ["none", "hflip", "vflip", "hvflip"]

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()

        probs_sum = None
        for mode in tta_modes:
            aug = apply_tta(images, mode)
            logits = get_logits(model(aug))
            probs = torch.sigmoid(logits)
            probs = undo_tta(probs, mode)

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum = probs_sum + probs

        probs_avg = (probs_sum / len(tta_modes)).detach().cpu().numpy()[:, 0]

        probs_all.append(probs_avg)
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


def evaluate_probs(probs_all: np.ndarray, targets_all: np.ndarray, threshold: float):
    rows = []

    for i in range(len(probs_all)):
        pred = (probs_all[i] >= threshold).astype(np.uint8)
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

    val_ds = SegmentationDataset(df, "val", image_size=image_size)
    test_ds = SegmentationDataset(df, "test", image_size=image_size)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    val_probs, val_targets, _ = collect_probs_and_targets_tta(model, val_loader, device)
    test_probs, test_targets, test_meta = collect_probs_and_targets_tta(model, test_loader, device)

    search_rows = []
    best_thr = None
    best_val_dice = -1.0
    best_val_metrics = None

    for thr in THRESHOLDS:
        metrics, _ = evaluate_probs(val_probs, val_targets, thr)
        row = {
            "threshold": thr,
            "val_dice": metrics["dice"],
            "val_iou": metrics["iou"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
        }
        search_rows.append(row)

        if metrics["dice"] > best_val_dice:
            best_val_dice = metrics["dice"]
            best_thr = thr
            best_val_metrics = row.copy()

    search_df = pd.DataFrame(search_rows).sort_values(
        ["val_dice", "val_iou", "val_precision"], ascending=False
    ).reset_index(drop=True)
    search_df.to_csv(SEARCH_CSV, index=False, encoding="utf-8-sig")

    test_metrics, _ = evaluate_probs(test_probs, test_targets, best_thr)

    per_image_rows = []
    for i in range(len(test_probs)):
        pred = (test_probs[i] >= best_thr).astype(np.uint8)
        target = test_targets[i].astype(np.uint8)
        m = binary_metrics(pred, target)
        per_image_rows.append({
            **test_meta[i],
            "threshold": best_thr,
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
        "tta_modes": ["none", "hflip", "vflip", "hvflip"],
        "best_on_val": best_val_metrics,
        "test_metrics_with_best_val_threshold": test_metrics,
        "search_csv": str(SEARCH_CSV),
        "test_per_image_csv": str(TEST_PER_IMAGE_CSV),
    }
    BEST_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BEST ON VAL WITH TTA ===")
    print(f"threshold      = {best_thr:.2f}")
    print(f"val_dice       = {best_val_metrics['val_dice']:.4f}")
    print(f"val_iou        = {best_val_metrics['val_iou']:.4f}")
    print(f"val_precision  = {best_val_metrics['val_precision']:.4f}")
    print(f"val_recall     = {best_val_metrics['val_recall']:.4f}")

    print("\n=== TEST WITH BEST VAL TTA THRESHOLD ===")
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
