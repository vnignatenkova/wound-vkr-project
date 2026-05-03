from pathlib import Path
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from src.config_paths import ensure_project_dirs

SEG_DATASET_DIR = Path("data/segmentation_dataset")
METADATA_CSV = SEG_DATASET_DIR / "metadata.csv"

RESULTS_DIR = Path("results/segmentation")
BEST_MODEL_PATH = RESULTS_DIR / "best_model.pth"
LAST_MODEL_PATH = RESULTS_DIR / "last_model.pth"
HISTORY_CSV = RESULTS_DIR / "train_history.csv"
METRICS_JSON = RESULTS_DIR / "segmentation_metrics.json"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42
THRESHOLD = 0.5


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_tf = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
        ])
        self.mask_resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = SEG_DATASET_DIR / "images" / row["image_file"]
        mask_path = SEG_DATASET_DIR / "masks" / row["mask_file"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_tf(image)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "image_file": row["image_file"],
            "mask_file": row["mask_file"],
            "task_dir": row["task_dir"],
        }


def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def precision_recall_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))

    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    return precision, recall


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score_from_logits(logits, masks, THRESHOLD)
        total_iou += iou_score_from_logits(logits, masks, THRESHOLD)
        p, r = precision_recall_from_logits(logits, masks, THRESHOLD)
        total_precision += p
        total_recall += r
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
        "precision": total_precision / max(n_batches, 1),
        "recall": total_recall / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    return run_epoch(model, loader, optimizer=None, device=device, train=False)


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ensure_project_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Не найден metadata.csv: {METADATA_CSV}")

    set_seed(SEED)
    device = get_device()

    df = pd.read_csv(METADATA_CSV)
    print(f"Всего строк в segmentation_dataset: {len(df)}")
    print(df["split"].value_counts().to_string())
    print(f"device: {device}")

    train_ds = SegmentationDataset(df, "train")
    val_ds = SegmentationDataset(df, "val")
    test_ds = SegmentationDataset(df, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    best_val_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    print(f"\nЛучшая val Dice: {best_val_dice:.4f}")
    print(f"best_model: {BEST_MODEL_PATH}")

    best_model = UNet().to(device)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_metrics = evaluate_test(best_model, test_loader, device)
    metrics = {
        "device": str(device),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "best_val_dice": best_val_dice,
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    save_json(METRICS_JSON, metrics)

    print("\n=== TEST ===")
    print(f"test_loss      = {test_metrics['loss']:.4f}")
    print(f"test_dice      = {test_metrics['dice']:.4f}")
    print(f"test_iou       = {test_metrics['iou']:.4f}")
    print(f"test_precision = {test_metrics['precision']:.4f}")
    print(f"test_recall    = {test_metrics['recall']:.4f}")
    print(f"\nСохранено:")
    print(f"- {BEST_MODEL_PATH}")
    print(f"- {LAST_MODEL_PATH}")
    print(f"- {HISTORY_CSV}")
    print(f"- {METRICS_JSON}")


if __name__ == "__main__":
    main()
