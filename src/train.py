"""
src/train.py
-------------
Full training loop for OcuNet.
Features:
  - Mixed precision (AMP) for RTX 3050 VRAM efficiency
  - Early stopping
  - Best model checkpointing
  - Per-epoch metrics (loss, F1, AUC)
  - Progress bars
"""

import os
import sys
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.ocunet import build_model
from src.dataset.loader import build_dataloaders, load_config, load_label_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 46


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(all_labels: np.ndarray, all_probs: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute F1 and AUC for multi-label outputs."""
    preds = (all_probs >= threshold).astype(int)

    # Macro F1
    f1 = f1_score(all_labels, preds, average="macro", zero_division=0)

    # AUC — only for classes that appear in labels
    try:
        active_cols = [i for i in range(NUM_CLASSES) if all_labels[:, i].sum() > 0]
        if active_cols:
            auc = roc_auc_score(
                all_labels[:, active_cols],
                all_probs[:, active_cols],
                average="macro"
            )
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    return {"f1": round(f1, 4), "auc": round(auc, 4)}


# ─────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler,
              device, mode="train") -> dict:
    """Run one full epoch. mode = 'train' or 'val'."""
    is_train = (mode == "train")
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    all_labels  = []
    all_probs   = []

    pbar = tqdm(loader, desc=f"  {mode.upper():5}", leave=False, ncols=90)

    with torch.set_grad_enabled(is_train):
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(scaler is not None)):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_labels = np.vstack(all_labels)
    all_probs  = np.vstack(all_probs)
    avg_loss   = total_loss / len(loader)
    metrics    = compute_metrics(all_labels, all_probs)
    metrics["loss"] = round(avg_loss, 4)

    return metrics


# ─────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter    = 0
        return self.stop


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(config_path: str = "configs/config.yaml"):
    config    = load_config(config_path)
    label_map = load_label_map("configs/label_map.json")

    # Reproducibility
    torch.manual_seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    logger.info("Building dataloaders...")
    train_loader, val_loader, _ = build_dataloaders(config, label_map)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"OcuNet trainable parameters: {total_params:,}")

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # LR scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3,
                                   factor=0.5, verbose=True)

    # Mixed precision
    use_amp = config["hardware"]["mixed_precision"] and device.type == "cuda"
    scaler  = GradScaler() if use_amp else None
    logger.info(f"Mixed precision (AMP): {use_amp}")

    # ── Resume from checkpoint if it exists ──
    output_dir     = Path(config["paths"]["outputs"])
    output_dir.mkdir(exist_ok=True)
    best_ckpt_path = output_dir / "ocunet_best.pth"
    history_path   = output_dir / "training_history.json"

    start_epoch   = 1
    best_val_loss = float("inf")
    history       = {"train": [], "val": []}

    if best_ckpt_path.exists():
        logger.info(f"Resuming from checkpoint: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        logger.info(f"Resumed from epoch {ckpt['epoch']} | "
                    f"Best val loss: {best_val_loss:.4f} | "
                    f"Val F1: {ckpt['val_f1']:.4f} | "
                    f"Val AUC: {ckpt['val_auc']:.4f}")

        # Load history if exists
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
    else:
        logger.info("No checkpoint found — starting fresh from epoch 1")

    # Early stopping
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    num_epochs = config["training"]["num_epochs"]
    logger.info(f"\nTraining from epoch {start_epoch} to {num_epochs}...\n")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch:03d}/{num_epochs}")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                                   scaler, device, mode="train")
        val_metrics   = run_epoch(model, val_loader,   criterion, None,
                                   scaler, device, mode="val")

        scheduler.step(val_metrics["loss"])

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"  Train — Loss: {train_metrics['loss']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    best_val_loss,
                "val_f1":      val_metrics["f1"],
                "val_auc":     val_metrics["auc"],
                "config":      config
            }, best_ckpt_path)
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")

        # Early stopping check
        if early_stopping(val_metrics["loss"]):
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

        print()

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"Best val loss : {best_val_loss:.4f}")
    logger.info(f"Model saved   : {best_ckpt_path}")
    logger.info(f"History saved : {history_path}")

    return model, history


if __name__ == "__main__":
    train()