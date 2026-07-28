"""
preprocessing/pipeline.py
--------------------------
Handles all image preprocessing steps:
  1. CLAHE contrast enhancement
  2. Resize to 224x224
  3. Normalize (ImageNet stats)
  4. Hash-based deduplication
  5. Dataset-level stats logging
"""

import os
import cv2
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# CLAHE Enhancement
# ─────────────────────────────────────────────

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast — critical for fundus images where
    lesions can be very subtle against the retinal background.

    Works on LAB color space so hue is preserved.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to apply_clahe")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


# ─────────────────────────────────────────────
# Resize
# ─────────────────────────────────────────────

def resize_image(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resize to square (target_size x target_size).
    Uses INTER_AREA for downscaling (sharpest), INTER_LINEAR for upscaling.
    """
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > target_size or w > target_size) else cv2.INTER_LINEAR
    return cv2.resize(image, (target_size, target_size), interpolation=interp)


# ─────────────────────────────────────────────
# Normalize
# ─────────────────────────────────────────────

def normalize_image(image: np.ndarray,
                    mean: list = [0.485, 0.456, 0.406],
                    std: list  = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Normalize to ImageNet stats (float32, channels last).
    These stats are used because EfficientNet-B0 was pretrained on ImageNet.
    """
    image = image.astype(np.float32) / 255.0
    mean  = np.array(mean, dtype=np.float32)
    std   = np.array(std,  dtype=np.float32)
    image = (image - mean) / std
    return image


# ─────────────────────────────────────────────
# Full preprocessing pipeline for one image
# ─────────────────────────────────────────────

def preprocess_image(image_path: str, config: dict) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for a single image.
    Returns float32 numpy array of shape (224, 224, 3) or None on failure.
    """
    cfg = config["preprocessing"]

    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not read image: {image_path}")
        return None

    # Step 1: CLAHE
    image = apply_clahe(
        image,
        clip_limit=cfg["clahe_clip_limit"],
        grid_size=tuple(cfg["clahe_grid_size"])
    )

    # Step 2: Resize
    image = resize_image(image, target_size=cfg["image_size"])

    # Step 3: Convert BGR → RGB (PyTorch uses RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 4: Normalize
    image = normalize_image(
        image,
        mean=cfg["normalize_mean"],
        std=cfg["normalize_std"]
    )

    return image


# ─────────────────────────────────────────────
# Hash-based Deduplication
# ─────────────────────────────────────────────

def get_image_hash(image_path: str) -> Optional[str]:
    """
    Compute MD5 hash of raw image bytes.
    Used to detect exact duplicate images across datasets.
    """
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Hash failed for {image_path}: {e}")
        return None


def deduplicate_paths(image_paths: list) -> list:
    """
    Given a list of image paths, return only unique images (by MD5 hash).
    Logs how many duplicates were removed.
    """
    seen_hashes = set()
    unique_paths = []
    duplicate_count = 0

    for path in image_paths:
        h = get_image_hash(path)
        if h is None:
            continue
        if h in seen_hashes:
            duplicate_count += 1
        else:
            seen_hashes.add(h)
            unique_paths.append(path)

    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate images. Kept {len(unique_paths)}.")

    return unique_paths


# ─────────────────────────────────────────────
# Batch preprocessing (save to processed folder)
# ─────────────────────────────────────────────

def preprocess_and_save(image_paths: list, output_dir: str, config: dict) -> dict:
    """
    Validates all images are readable and preprocessable.
    Does NOT save .npy files — preprocessing happens on-the-fly
    during training to avoid disk space issues.
    Returns a dict mapping original path → original path (for compatibility).
    """
    valid_paths = {}
    failed = 0

    for i, img_path in enumerate(image_paths):
        # Just validate the image is readable — don't save to disk
        image = cv2.imread(img_path)
        if image is None:
            failed += 1
            continue
        valid_paths[img_path] = img_path   # identity map — used at training time

        if (i + 1) % 500 == 0:
            logger.info(f"Validated {i + 1}/{len(image_paths)} images...")

    logger.info(f"Done. Valid: {len(valid_paths)}, Failed/unreadable: {failed}")
    return valid_paths


# ─────────────────────────────────────────────
# Quick visual sanity check (saves sample grid)
# ─────────────────────────────────────────────

def save_sample_grid(image_paths: list, output_path: str, config: dict,
                     n_samples: int = 9) -> None:
    """
    Save a 3x3 grid of preprocessed images for visual sanity checking.
    Denormalizes back to uint8 for display.
    """
    import math
    samples = image_paths[:n_samples]
    cols = int(math.sqrt(n_samples))
    rows = math.ceil(n_samples / cols)

    mean = np.array(config["preprocessing"]["normalize_mean"])
    std  = np.array(config["preprocessing"]["normalize_std"])
    size = config["preprocessing"]["image_size"]

    grid_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(samples):
                img = preprocess_image(samples[idx], config)
                if img is not None:
                    # Denormalize for display
                    img = (img * std + mean)
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                img = np.zeros((size, size, 3), dtype=np.uint8)
            row_imgs.append(img)
        grid_rows.append(np.hstack(row_imgs))

    grid = np.vstack(grid_rows)
    cv2.imwrite(output_path, grid)
    logger.info(f"Sample grid saved to: {output_path}")


# ─────────────────────────────────────────────
# Entry point — run pipeline on all raw data
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config("configs/config.yaml")

    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])

    # Collect all image paths
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_paths = [
        str(p) for p in raw_dir.rglob("*")
        if p.suffix.lower() in image_extensions
    ]
    logger.info(f"Found {len(all_paths)} images in {raw_dir}")

    # Deduplicate
    all_paths = deduplicate_paths(all_paths)
    logger.info(f"After deduplication: {len(all_paths)} unique images")

    # Preprocess and save
    path_map = preprocess_and_save(all_paths, str(processed_dir), config)

    # Save sample grid for visual check
    sample_grid_path = str(Path(config["paths"]["outputs"]) / "sample_grid.jpg")
    save_sample_grid(all_paths[:9], sample_grid_path, config)