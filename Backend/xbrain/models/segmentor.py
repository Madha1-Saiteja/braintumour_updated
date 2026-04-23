"""
segmentor.py
SwinUNETR brain tumor segmentation.
Mirrors exactly the architecture used in segmentation.ipynb.
"""

import os
import numpy as np
import cv2
import torch
from monai.networks.nets import SwinUNETR

IMG_SIZE = 224


def build_segmentor(
    checkpoint_path: str,
    in_channels: int = 3,
    num_classes: int = 1,
    device: torch.device | None = None
) -> tuple[SwinUNETR, torch.device]:
    """
    Build SwinUNETR and load saved checkpoint.
    Architecture matches segmentation.ipynb Section 6 exactly.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=24,
        use_checkpoint=True,
        spatial_dims=2,
    ).to(device)

    # Handle both file and directory checkpoints
    if os.path.isdir(checkpoint_path):
        # Load from directory (PyTorch zip format)
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        # Load from file
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        epoch = checkpoint.get("epoch", "?")
        dice = checkpoint.get("best_dice", "?")
        print(f"[Segmentor] Loaded checkpoint — epoch {epoch}, best Dice {dice}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[Segmentor] Weights loaded from: {checkpoint_path}")

    model.eval()
    return model, device


def segment(
    model: SwinUNETR,
    img_rgb: np.ndarray,
    device: torch.device,
    img_size: int = IMG_SIZE,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Run SwinUNETR on an RGB numpy image.
    Returns binary mask (H, W) as float32 [0.0 or 1.0].
    """
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = (
        torch.from_numpy(img_norm)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        mask = (pred > threshold).float()

    return mask.squeeze().cpu().numpy()


def compute_tumor_stats(mask: np.ndarray) -> dict:
    """
    Compute tumor area statistics from binary mask.
    Returns pixel count, percentage, and bounding box.
    """
    total_pixels = mask.size
    binary_mask = (mask > 0).astype(np.uint8)
    tumor_pixels = int(binary_mask.sum())
    tumor_pct = round(100.0 * tumor_pixels / total_pixels, 2) if total_pixels > 0 else 0.0

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if rows.any() and cols.any():
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        bbox = {
            "rmin": int(row_indices[0]),
            "rmax": int(row_indices[-1]),
            "cmin": int(col_indices[0]),
            "cmax": int(col_indices[-1]),
        }
    else:
        bbox = None

    return {
        "tumor_pixels": tumor_pixels,
        "total_pixels": total_pixels,
        "tumor_area_pct": tumor_pct,
        "has_mask": tumor_pixels > 0,
        "bounding_box": bbox,
    }


def get_segmentation_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 50, 50),
    alpha: float = 0.45
) -> np.ndarray:
    """
    Blend segmentation mask onto original image with contour.
    Returns overlay as uint8 RGB.
    """
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    mask_resized = cv2.resize(
        mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
    )
    mask_resized = (mask_resized > 0.5).astype(np.float32)

    overlay = img_resized.copy()

    if mask_resized.sum() > 0:
        color_layer = np.zeros_like(img_resized)
        color_layer[:, :] = color

        mask_3ch = np.stack([mask_resized] * 3, axis=-1)
        overlay = img_resized * (1 - alpha * mask_3ch) + color_layer * (alpha * mask_3ch)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    else:
        overlay = overlay.astype(np.uint8)

    return overlay