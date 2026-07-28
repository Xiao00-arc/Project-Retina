"""
src/gradcam.py
---------------
Generates Grad-CAM heatmaps showing WHERE on the retina
OcuNet detected each disease.

Outputs:
  - outputs/gradcam/<disease_name>_<image_id>.png
  - Side-by-side: original | heatmap overlay | prediction labels
"""

import os
import sys
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.ocunet import build_model, OcuNet
from src.dataset.loader import (build_dataloaders, load_config,
                                  load_label_map, get_transforms)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 46


# ─────────────────────────────────────────────
# Grad-CAM implementation
# ─────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.
    Hooks into the last convolutional block (features[-1]).
    """

    def __init__(self, model: OcuNet):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Target layer: last block of EfficientNet features
        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor,
                 class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class.
        Returns heatmap as numpy array (H, W) in [0, 1].
        """
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)
        score  = logits[0, class_idx]

        # Backward pass for target class
        self.model.zero_grad()
        score.backward()

        # Grad-CAM formula: global average pool gradients → weight activations
        weights     = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam         = (weights * self.activations).sum(dim=1).squeeze()  # (H, W)
        cam         = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()


# ─────────────────────────────────────────────
# Overlay heatmap on image
# ─────────────────────────────────────────────

def overlay_heatmap(original_bgr: np.ndarray,
                     heatmap: np.ndarray,
                     alpha: float = 0.45) -> np.ndarray:
    """Resize heatmap to image size and overlay as colormap."""
    h, w   = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay         = cv2.addWeighted(original_bgr, 1 - alpha,
                                       heatmap_color, alpha, 0)
    return overlay


# ─────────────────────────────────────────────
# Denormalize tensor back to uint8 image
# ─────────────────────────────────────────────

def tensor_to_bgr(tensor: torch.Tensor, config: dict) -> np.ndarray:
    """Convert normalized tensor back to BGR uint8 for display."""
    mean = np.array(config["preprocessing"]["normalize_mean"])
    std  = np.array(config["preprocessing"]["normalize_std"])

    img = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ─────────────────────────────────────────────
# Save single Grad-CAM result
# ─────────────────────────────────────────────

def save_gradcam_figure(original_bgr: np.ndarray,
                         overlay: np.ndarray,
                         heatmap: np.ndarray,
                         true_labels: list,
                         pred_labels: list,
                         pred_probs: list,
                         image_path: str,
                         output_path: str,
                         disease_names: dict):
    """Save a 3-panel figure: original | overlay | prediction bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Original
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Fundus Image", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Grad-CAM overlay
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    true_str = ", ".join(true_labels) if true_labels else "Normal"
    axes[1].set_title(f"Grad-CAM Heatmap\nTrue: {true_str}", fontsize=10)
    axes[1].axis("off")

    # Panel 3: Prediction confidence bars
    if pred_labels:
        colors = ["#1D9E75" if l in true_labels else "#D85A30" for l in pred_labels]
        y_pos  = range(len(pred_labels))
        axes[2].barh(y_pos, pred_probs, color=colors, edgecolor="none")
        axes[2].set_yticks(list(y_pos))
        axes[2].set_yticklabels(pred_labels, fontsize=9)
        axes[2].set_xlim(0, 1)
        axes[2].set_xlabel("Confidence")
        axes[2].set_title("Predicted Diseases", fontsize=11)
        axes[2].invert_yaxis()
        axes[2].axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)

        legend = [
            mpatches.Patch(color="#1D9E75", label="Correct"),
            mpatches.Patch(color="#D85A30", label="False positive"),
        ]
        axes[2].legend(handles=legend, fontsize=8, loc="lower right")
    else:
        axes[2].text(0.5, 0.5, "No disease\ndetected",
                     ha="center", va="center", fontsize=12)
        axes[2].axis("off")

    img_name = Path(image_path).name
    plt.suptitle(f"OcuNet — {img_name}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# Main Grad-CAM generation
# ─────────────────────────────────────────────

def generate_gradcam(config_path: str = "configs/config.yaml",
                      n_samples: int = 20,
                      threshold: float = 0.5):
    """
    Generate Grad-CAM heatmaps for n_samples from the test set.
    Saves results to outputs/gradcam/
    """
    config     = load_config(config_path)
    label_map  = load_label_map("configs/label_map.json")
    output_dir = Path(config["paths"]["outputs"]) / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    ckpt_path = Path(config["paths"]["outputs"]) / "ocunet_best.pth"
    if not ckpt_path.exists():
        logger.error("No checkpoint found. Train first!")
        return

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Model loaded from epoch {ckpt['epoch']}")

    # Grad-CAM
    gradcam = GradCAM(model)

    # Disease names
    disease_names = {int(k): v for k, v in label_map["global_labels"].items()}

    # Get test samples
    _, _, test_loader = build_dataloaders(config, label_map)
    transform = get_transforms(config, mode="val")

    saved = 0
    logger.info(f"Generating Grad-CAM for {n_samples} test images...")

    for images, labels, paths in test_loader:
        for i in range(len(images)):
            if saved >= n_samples:
                break

            image_tensor = images[i].to(device)
            label_vec    = labels[i].numpy()
            image_path   = paths[i]

            # Get predictions
            with torch.no_grad():
                logits = model(image_tensor.unsqueeze(0))
                probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

            pred_mask  = probs >= threshold
            true_mask  = label_vec >= 0.5

            true_labels = [disease_names[j] for j in range(NUM_CLASSES)
                           if true_mask[j]]
            pred_labels = [disease_names[j] for j in range(NUM_CLASSES)
                           if pred_mask[j]]
            pred_probs_list = [float(probs[j]) for j in range(NUM_CLASSES)
                               if pred_mask[j]]

            # Generate Grad-CAM for the highest-confidence predicted class
            if pred_mask.any():
                target_class = int(probs.argmax())
            elif true_mask.any():
                target_class = int(label_vec.argmax())
            else:
                target_class = 0

            heatmap      = gradcam.generate(image_tensor, target_class)
            original_bgr = tensor_to_bgr(images[i], config)
            overlay      = overlay_heatmap(original_bgr, heatmap)

            # Save figure
            disease_tag = disease_names.get(target_class, "unknown").replace(" ", "_")
            out_path    = output_dir / f"{saved:03d}_{disease_tag}.png"
            save_gradcam_figure(
                original_bgr, overlay, heatmap,
                true_labels, pred_labels, pred_probs_list,
                image_path, str(out_path), disease_names
            )
            saved += 1

        if saved >= n_samples:
            break

    logger.info(f"Grad-CAM complete! {saved} images saved to: {output_dir}")
    logger.info("Open outputs/gradcam/ to view the heatmaps.")


if __name__ == "__main__":
    generate_gradcam(n_samples=20)