from pathlib import Path
import json
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.config_paths import ensure_project_dirs


SEG_DATASET_DIR = Path("data/segmentation_dataset")
METADATA_CSV = SEG_DATASET_DIR / "metadata.csv"

RESULTS_DIR = Path("results/segmentation")
BEST_MODEL_PATH = RESULTS_DIR / "best_model_v2.pth"
LAST_MODEL_PATH = RESULTS_DIR / "last_model_v2.pth"
HISTORY_CSV = RESULTS_DIR / "train_history_v2.csv"
METRICS_JSON = RESULTS_DIR / "segmentation_metrics_v2.json"

IMAGE_SIZE = 384
BATCH_SIZE = 6
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
SEED = 42

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
EARLY_STOPPING_PATIENCE = 10


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

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

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
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
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
    def __init__(self, df: pd.DataFrame, split: str, image_size: int = 384, augment: bool = False):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _apply_geom_aug(self, image: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = random.uniform(-15.0, 15.0)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        return image, mask

    def _apply_color_aug(self, image: Image.Image):
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = SEG_DATASET_DIR / "images" / row["image_file"]
        mask_path = SEG_DATASET_DIR / "masks" / row["mask_file"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._apply_geom_aug(image, mask)
            image = self._apply_color_aug(image)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "image_file": row["image_file"],
            "mask_file": row["mask_file"],
            "task_dir": row["task_dir"],
        }


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        pos_weight = self.pos_weight_tensor.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


def compute_pos_weight(train_df: pd.DataFrame) -> float:
    pos = float(train_df["mask_area_px"].sum())
    total = float((train_df["width"] * train_df["height"]).sum())
    neg = max(total - pos, 1.0)
    pos = max(pos, 1.0)
    ratio = neg / pos
    return float(min(max(ratio, 1.0), 20.0))


def metrics_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float):
    preds = (probs >= threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + 1e-7) / (union + 1e-7)).mean().item()

    iou_union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = ((intersection + 1e-7) / (iou_union + 1e-7)).mean().item()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))

    precision = ((tp + 1e-7) / (tp + fp + 1e-7)).mean().item()
    recall = ((tp + 1e-7) / (tp + fn + 1e-7)).mean().item()

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def run_train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device, thresholds):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_probs = []
    all_targets = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        probs = torch.sigmoid(logits).detach().cpu()
        targets = masks.detach().cpu()

        all_probs.append(probs)
        all_targets.append(targets)

        total_loss += loss.item()
        total_batches += 1

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    best_threshold = None
    best_metrics = None
    best_dice = -1.0

    for thr in thresholds:
        metrics = metrics_from_probs(probs, targets, thr)
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_threshold = thr
            best_metrics = metrics

    best_metrics["loss"] = total_loss / max(total_batches, 1)
    best_metrics["best_threshold"] = best_threshold
    return best_metrics


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

    train_df = df[df["split"] == "train"].copy()
    pos_weight = compute_pos_weight(train_df)
    print(f"pos_weight: {pos_weight:.4f}")

    train_ds = SegmentationDataset(df, "train", image_size=IMAGE_SIZE, augment=True)
    val_ds = SegmentationDataset(df, "val", image_size=IMAGE_SIZE, augment=False)
    test_ds = SegmentationDataset(df, "test", image_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = UNet().to(device)
    criterion = BCEDiceLoss(pos_weight=pos_weight, bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    history = []
    best_val_dice = -1.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = run_eval_epoch(model, val_loader, criterion, device, THRESHOLDS)
        scheduler.step(val_metrics["dice"])

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_best_threshold": val_metrics["best_threshold"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_thr={val_metrics['best_threshold']:.2f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_threshold = val_metrics["best_threshold"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: нет улучшения {EARLY_STOPPING_PATIENCE} эпох подряд.")
            break

    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    print(f"\nЛучшая val Dice: {best_val_dice:.4f}")
    print(f"Лучший threshold: {best_threshold:.2f}")
    print(f"Лучшая эпоха: {best_epoch}")
    print(f"best_model: {BEST_MODEL_PATH}")

    best_model = UNet().to(device)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    test_metrics = run_eval_epoch(best_model, test_loader, criterion, device, [best_threshold])

    metrics = {
        "device": str(device),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "epochs_trained": len(history),
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "pos_weight": pos_weight,
        "best_val_dice": best_val_dice,
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
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
