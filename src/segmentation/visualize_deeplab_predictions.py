from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.segmentation import deeplabv3_resnet50

from src.config_paths import PROJECT_ROOT, ensure_project_dirs


SEG_DATASET_DIR = PROJECT_ROOT / "data" / "segmentation_dataset"
METADATA_CSV = SEG_DATASET_DIR / "metadata.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "segmentation"
MODEL_PATH = RESULTS_DIR / "deeplab_best_model.pth"
METRICS_JSON = RESULTS_DIR / "deeplab_metrics.json"
VIS_DIR = RESULTS_DIR / "deeplab_visualizations"

IMAGE_SIZE = 384
SAMPLES_PER_SPLIT = 6
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model():
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=1,
        aux_loss=False,
    )
    return model


def get_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["out"]
    return model_output


def load_threshold():
    if METRICS_JSON.exists():
        data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        return float(data.get("best_threshold", 0.5))
    return 0.5


def make_red_overlay(image_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    out = image_np.astype(np.float32).copy()
    red = np.zeros_like(out, dtype=np.float32)
    red[..., 0] = 255.0
    mask3 = mask_np[..., None].astype(np.float32)
    out = out * (1.0 - alpha * mask3) + red * (alpha * mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_green_overlay(image_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    out = image_np.astype(np.float32).copy()
    green = np.zeros_like(out, dtype=np.float32)
    green[..., 1] = 255.0
    mask3 = mask_np[..., None].astype(np.float32)
    out = out * (1.0 - alpha * mask3) + green * (alpha * mask3)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_mask(mask_np: np.ndarray, path: Path):
    Image.fromarray((mask_np.astype(np.uint8) * 255)).save(path)


def main():
    ensure_project_dirs()
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Не найден metadata.csv: {METADATA_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена модель: {MODEL_PATH}")

    random.seed(SEED)
    device = get_device()
    threshold = load_threshold()

    print(f"device: {device}")
    print(f"threshold: {threshold:.2f}")

    df = pd.read_csv(METADATA_CSV)

    model = build_model().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    saved = 0

    for split in ["train", "val", "test"]:
        split_dir = VIS_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_df = df[df["split"] == split].copy()
        if len(split_df) == 0:
            continue

        n = min(SAMPLES_PER_SPLIT, len(split_df))
        split_df = split_df.sample(n=n, random_state=SEED).reset_index(drop=True)

        for _, row in split_df.iterrows():
            image_path = SEG_DATASET_DIR / "images" / row["image_file"]
            mask_path = SEG_DATASET_DIR / "masks" / row["mask_file"]

            image = Image.open(image_path).convert("RGB")
            gt_mask = Image.open(mask_path).convert("L")

            orig_w, orig_h = image.size

            image_resized = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR)
            x = TF.to_tensor(image_resized)
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = get_logits(model(x))
                probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            pred_small = (probs >= threshold).astype(np.uint8)
            pred_mask = Image.fromarray(pred_small * 255).resize((orig_w, orig_h), resample=Image.NEAREST)
            pred_mask_np = (np.array(pred_mask) > 0).astype(np.uint8)

            gt_mask_np = (np.array(gt_mask) > 0).astype(np.uint8)
            image_np = np.array(image)

            gt_overlay = make_red_overlay(image_np, gt_mask_np)
            pred_overlay = make_green_overlay(image_np, pred_mask_np)

            base = Path(row["image_file"]).stem
            Image.fromarray(image_np).save(split_dir / f"{base}_image.jpg")
            save_mask(gt_mask_np, split_dir / f"{base}_gt_mask.png")
            save_mask(pred_mask_np, split_dir / f"{base}_pred_mask.png")
            Image.fromarray(gt_overlay).save(split_dir / f"{base}_gt_overlay.jpg")
            Image.fromarray(pred_overlay).save(split_dir / f"{base}_pred_overlay.jpg")

            saved += 1

    summary = {
        "model_path": str(MODEL_PATH),
        "threshold": threshold,
        "samples_per_split": SAMPLES_PER_SPLIT,
        "saved_items": saved,
        "output_dir": str(VIS_DIR),
    }
    (VIS_DIR / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Сохранено примеров: {saved}")
    print(f"Папка: {VIS_DIR}")


if __name__ == "__main__":
    main()
