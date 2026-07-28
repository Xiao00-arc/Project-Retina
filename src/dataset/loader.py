"""
dataset/loader.py
------------------
Unified PyTorch Dataset that:
  1. Loads annotations from ODIR-5K, RFMiD, JSIEC
  2. Maps each dataset's labels to our global 46-class schema
  3. Returns multi-hot encoded label vectors
  4. Feeds preprocessed images to the model
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 46


# ─────────────────────────────────────────────
# Load configs
# ─────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_label_map(map_path: str = "configs/label_map.json") -> dict:
    with open(map_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Per-dataset annotation loaders
# Returns: list of (image_path, label_vector) tuples
# ─────────────────────────────────────────────

def load_odir5k(cfg: dict, label_map: dict) -> List[Tuple[str, np.ndarray]]:
    """
    Load ODIR-5K — each row has left + right eye image.
    CSV columns: ID, filename, N, D, G, C, A, H, M, O
    Images live in: data/raw/ODIR-5K/ODIR-5K/Training Images/
    """
    ann_file = cfg["datasets"]["odir5k"]["annotation_file"]
    folder   = cfg["datasets"]["odir5k"]["folder"]
    mapping  = label_map["odir5k_map"]

    if not os.path.exists(ann_file):
        logger.warning(f"ODIR-5K annotation not found: {ann_file}")
        return []

    df = pd.read_csv(ann_file)
    samples = []

    for _, row in df.iterrows():
        # Build label vector from N,D,G,C,A,H,M,O columns
        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        for col, global_id in mapping.items():
            if col in row and row[col] == 1:
                label_vec[global_id] = 1.0

        # ODIR-5K has a 'filename' column with just the image name
        fname = str(row["filename"]).strip()
        img_path = os.path.join(folder, fname)

        # Also try left/right fundus columns if filename not found
        if not os.path.exists(img_path):
            for col in ["Left-Fundus", "Right-Fundus"]:
                if col in row:
                    alt_path = os.path.join(folder, str(row[col]).strip())
                    if os.path.exists(alt_path):
                        samples.append((alt_path, label_vec))
            continue

        if os.path.exists(img_path):
            samples.append((img_path, label_vec))

    logger.info(f"ODIR-5K loaded: {len(samples)} samples")
    return samples


def load_rfmid(cfg: dict, label_map: dict) -> List[Tuple[str, np.ndarray]]:
    """Load RFMiD — multi-label, 46 binary columns per image."""
    ann_file = cfg["datasets"]["rfmid"]["annotation_file"]
    folder   = cfg["datasets"]["rfmid"]["folder"]
    mapping  = label_map["rfmid_map"]

    if not os.path.exists(ann_file):
        logger.warning(f"RFMiD annotation not found: {ann_file}")
        return []

    df = pd.read_csv(ann_file)
    samples = []

    for _, row in df.iterrows():
        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)

        # RFMiD has a 'Disease_Risk' column — if 0, it's normal
        if "Disease_Risk" in row and row["Disease_Risk"] == 0:
            label_vec[0] = 1.0   # Normal
        else:
            for col, global_id in mapping.items():
                if col in row and row[col] == 1:
                    label_vec[global_id] = 1.0

        img_id    = str(row[cfg["datasets"]["rfmid"]["image_col"]])
        img_path  = os.path.join(folder, img_id + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(folder, img_id + ".jpg")

        if os.path.exists(img_path):
            samples.append((img_path, label_vec))

    logger.info(f"RFMiD loaded: {len(samples)} samples")
    return samples


def load_jsiec(cfg: dict, label_map: dict) -> List[Tuple[str, np.ndarray]]:
    """
    Load JSIEC — folder-based labels with numbered prefixes.
    Format: '0.0.Normal', '0.3.DR1', '10.0.Possible glaucoma' etc.
    """
    folder  = Path(cfg["datasets"]["jsiec"]["folder"]) / "1000images"
    mapping = label_map["jsiec_map"]

    if not folder.exists():
        logger.warning(f"JSIEC folder not found: {folder}")
        return []

    # Extended map handling JSIEC's specific folder naming
    jsiec_keyword_map = {
        "normal":           0,
        "dr1":              1, "dr2": 1, "dr3": 1,
        "diabetic":         1,
        "glaucoma":         2, "possible glaucoma": 2,
        "cataract":         3,
        "amd":              4, "macular degeneration": 4,
        "hypertensive":     5, "hypertension": 5,
        "myopia":           6, "pathologic myopia": 6,
        "tessellated":      8,
        "drusen":           9,
        "optic atrophy":    12,
        "optic cup":        12,
        "macular hole":     22,
        "retinitis":        23,
        "branch retinal vein": 44,
        "central retinal vein": 45,
        "branch retinal artery": 38,
        "central retinal artery": 30,
        "epiretinal":       43,
        "laser":            42,
        "vitreous":         35,
        "membrane":         10,
    }

    samples = []
    skipped_folders = []

    for disease_folder in sorted(folder.iterdir()):
        if not disease_folder.is_dir():
            continue

        raw_name = disease_folder.name.strip()

        # Strip numeric prefix: '0.1.Tessellated fundus' → 'tessellated fundus'
        parts = raw_name.split(".", 2)
        clean_name = parts[-1].strip().lower() if len(parts) >= 2 else raw_name.lower()

        # Match against keyword map
        global_id = -1
        for keyword, gid in jsiec_keyword_map.items():
            if keyword in clean_name:
                global_id = gid
                break

        if global_id == -1:
            skipped_folders.append(raw_name)
            continue

        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        label_vec[global_id] = 1.0

        count = 0
        for img_path in disease_folder.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append((str(img_path), label_vec))
                count += 1

    if skipped_folders:
        logger.warning(f"JSIEC: skipped {len(skipped_folders)} unmatched folders: {skipped_folders[:5]}")

    logger.info(f"JSIEC loaded: {len(samples)} samples from {folder}")
    return samples


# ─────────────────────────────────────────────
# Augmentation transforms
# ─────────────────────────────────────────────

def get_transforms(config: dict, mode: str = "train"):
    """
    Returns torchvision transforms for train / val / test.
    Train gets augmentation. Val/test gets only resize + normalize.
    """
    aug_cfg = config["augmentation"]
    pre_cfg = config["preprocessing"]
    size    = pre_cfg["image_size"]
    mean    = pre_cfg["normalize_mean"]
    std     = pre_cfg["normalize_std"]

    if mode == "train":
        return T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.RandomHorizontalFlip(p=0.5 if aug_cfg["horizontal_flip"] else 0.0),
            T.RandomRotation(degrees=aug_cfg["rotation_degrees"]),
            T.ColorJitter(
                brightness=aug_cfg["brightness"],
                contrast=aug_cfg["contrast"],
                saturation=aug_cfg["saturation"]
            ),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)) if aug_cfg["gaussian_blur"] else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])


# ─────────────────────────────────────────────
# Unified PyTorch Dataset
# ─────────────────────────────────────────────

class IrisDiseaseDataset(Dataset):
    """
    Unified dataset combining ODIR-5K + RFMiD + JSIEC.
    Returns (image_tensor, label_vector, image_path).
    """

    def __init__(self, samples: List[Tuple[str, np.ndarray]],
                 transform=None, apply_clahe: bool = True,
                 config: dict = None):
        self.samples     = samples
        self.transform   = transform
        self.apply_clahe = apply_clahe
        self.config      = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config["preprocessing"]["clahe_clip_limit"],
            tileGridSize=tuple(config["preprocessing"]["clahe_grid_size"])
        ) if config else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_vec = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return blank image if unreadable (shouldn't happen after cleaning)
            size = self.config["preprocessing"]["image_size"] if self.config else 224
            image = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply CLAHE
        if self.apply_clahe and self.clahe is not None:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

        # Apply transforms (augmentation / normalize)
        if self.transform:
            image = self.transform(image)

        label_tensor = torch.tensor(label_vec, dtype=torch.float32)
        return image, label_tensor, img_path


# ─────────────────────────────────────────────
# Train / Val / Test split
# ─────────────────────────────────────────────

def split_samples(samples: list, config: dict, seed: int = 42):
    """Stratified-ish split — shuffles and splits by ratio."""
    from sklearn.model_selection import train_test_split

    train_r = config["training"]["train_split"]
    val_r   = config["training"]["val_split"]

    train_data, temp = train_test_split(samples, test_size=(1 - train_r), random_state=seed)
    val_data, test_data = train_test_split(
        temp,
        test_size=config["training"]["test_split"] / (val_r + config["training"]["test_split"]),
        random_state=seed
    )

    logger.info(f"Split — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


# ─────────────────────────────────────────────
# Weighted sampler for class imbalance
# ─────────────────────────────────────────────

def get_weighted_sampler(samples: list) -> WeightedRandomSampler:
    """
    Computes per-sample weights so rare diseases are sampled more often.
    Weight of a sample = inverse frequency of its rarest positive label.
    """
    label_counts = np.zeros(NUM_CLASSES)
    for _, label_vec in samples:
        label_counts += label_vec

    # Avoid div-by-zero for unseen classes
    label_counts = np.where(label_counts == 0, 1, label_counts)
    label_weights = 1.0 / label_counts

    sample_weights = []
    for _, label_vec in samples:
        positive_weights = label_vec * label_weights
        # Weight of sample = weight of its rarest class
        w = positive_weights.max() if positive_weights.max() > 0 else 1.0
        sample_weights.append(w)

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ─────────────────────────────────────────────
# Build all DataLoaders in one call
# ─────────────────────────────────────────────

def build_dataloaders(config: dict, label_map: dict):
    """
    Full pipeline:
    1. Load all datasets
    2. Deduplicate
    3. Split
    4. Build DataLoaders with weighted sampling for train
    """

    # Load all three datasets
    all_samples = []
    all_samples.extend(load_odir5k(config, label_map))
    all_samples.extend(load_rfmid(config, label_map))
    all_samples.extend(load_jsiec(config, label_map))

    logger.info(f"Total samples before dedup: {len(all_samples)}")

    # Deduplicate by image path
    seen = set()
    unique_samples = []
    for path, label in all_samples:
        if path not in seen:
            seen.add(path)
            unique_samples.append((path, label))

    logger.info(f"Total samples after dedup: {len(unique_samples)}")

    # Split
    train_data, val_data, test_data = split_samples(unique_samples, config,
                                                     seed=config["project"]["seed"])

    # Transforms
    train_tf = get_transforms(config, mode="train")
    val_tf   = get_transforms(config, mode="val")

    # Datasets
    train_ds = IrisDiseaseDataset(train_data, transform=train_tf,
                                   apply_clahe=True, config=config)
    val_ds   = IrisDiseaseDataset(val_data,   transform=val_tf,
                                   apply_clahe=True, config=config)
    test_ds  = IrisDiseaseDataset(test_data,  transform=val_tf,
                                   apply_clahe=True, config=config)

    # Weighted sampler for training
    sampler = get_weighted_sampler(train_data)

    bs          = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    pin_memory  = config["hardware"]["pin_memory"]

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                               num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                               num_workers=num_workers, pin_memory=pin_memory)

    logger.info("DataLoaders ready.")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config    = load_config("configs/config.yaml")
    label_map = load_label_map("configs/label_map.json")

    train_loader, val_loader, test_loader = build_dataloaders(config, label_map)

    # Peek at first batch
    for images, labels, paths in train_loader:
        print(f"Image batch shape : {images.shape}")   # (16, 3, 224, 224)
        print(f"Label batch shape : {labels.shape}")   # (16, 46)
        print(f"Active labels in first sample: {labels[0].nonzero().flatten().tolist()}")
        break