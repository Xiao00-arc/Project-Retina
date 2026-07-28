"""
src/evaluate.py
----------------
Evaluates OcuNet on the test set.
Outputs:
  - Per-disease AUC, F1, Precision, Recall
  - Overall metrics summary
  - Confusion matrix heatmap
  - Full classification report saved to outputs/
"""

import os
import sys
import json
import yaml
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, classification_report
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.ocunet import build_model
from src.dataset.loader import build_dataloaders, load_config, load_label_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 46


# ─────────────────────────────────────────────
# Run inference on test set
# ─────────────────────────────────────────────

def run_inference(model, loader, device) -> tuple:
    """Returns (all_labels, all_probs) numpy arrays."""
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="  Evaluating", ncols=80):
            images = images.to(device)
            logits = model(images)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.vstack(all_labels), np.vstack(all_probs)


# ─────────────────────────────────────────────
# Per-disease metrics
# ─────────────────────────────────────────────

def per_disease_metrics(labels: np.ndarray, probs: np.ndarray,
                         label_map: dict, threshold: float = 0.5) -> list:
    """Compute AUC, F1, Precision, Recall for each disease."""
    preds        = (probs >= threshold).astype(int)
    disease_names = {int(k): v for k, v in label_map["global_labels"].items()}
    results      = []

    for i in range(NUM_CLASSES):
        name       = disease_names.get(i, f"Class {i}")
        true_col   = labels[:, i]
        pred_col   = preds[:, i]
        prob_col   = probs[:, i]
        n_positive = int(true_col.sum())

        if n_positive == 0:
            continue   # Skip classes with no positive samples in test set

        try:
            auc  = roc_auc_score(true_col, prob_col)
        except Exception:
            auc  = 0.0

        f1   = f1_score(true_col, pred_col, zero_division=0)
        prec = precision_score(true_col, pred_col, zero_division=0)
        rec  = recall_score(true_col, pred_col, zero_division=0)

        results.append({
            "disease":    name,
            "n_samples":  n_positive,
            "auc":        round(auc,  4),
            "f1":         round(f1,   4),
            "precision":  round(prec, 4),
            "recall":     round(rec,  4),
        })

    # Sort by AUC descending
    results.sort(key=lambda x: x["auc"], reverse=True)
    return results


# ─────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────

def print_results(results: list, overall: dict):
    print("\n" + "=" * 72)
    print("  OcuNet — PER DISEASE EVALUATION RESULTS")
    print("=" * 72)
    print(f"  {'Disease':<40} {'Samples':>7} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print("  " + "-" * 70)

    for r in results:
        flag = " ⚠" if r["auc"] < 0.7 else ""
        print(f"  {r['disease']:<40} {r['n_samples']:>7} "
              f"{r['auc']:>6.4f} {r['f1']:>6.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f}{flag}")

    print("=" * 72)
    print(f"  OVERALL")
    print(f"  Macro AUC       : {overall['macro_auc']:.4f}")
    print(f"  Macro F1        : {overall['macro_f1']:.4f}")
    print(f"  Macro Precision : {overall['macro_precision']:.4f}")
    print(f"  Macro Recall    : {overall['macro_recall']:.4f}")
    print(f"  Test samples    : {overall['n_samples']}")
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────
# Plot: AUC bar chart per disease
# ─────────────────────────────────────────────

def plot_auc_chart(results: list, output_path: str):
    names  = [r["disease"] for r in results]
    aucs   = [r["auc"]     for r in results]
    colors = ["#1D9E75" if a >= 0.85 else "#BA7517" if a >= 0.70 else "#D85A30"
              for a in aucs]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.38)))
    bars = ax.barh(names, aucs, color=colors, edgecolor="none")

    # Value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=8)

    ax.axvline(x=0.5, color="red",    linestyle="--", linewidth=0.8, alpha=0.5, label="Random (0.5)")
    ax.axvline(x=0.8, color="orange", linestyle="--", linewidth=0.8, alpha=0.5, label="Good (0.8)")
    ax.axvline(x=0.9, color="green",  linestyle="--", linewidth=0.8, alpha=0.5, label="Excellent (0.9)")

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC Score", fontsize=11)
    ax.set_title("OcuNet — Per-Disease AUC on Test Set", fontsize=13, pad=12)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1D9E75", label="AUC ≥ 0.85 (Excellent)"),
        Patch(facecolor="#BA7517", label="AUC 0.70–0.85 (Good)"),
        Patch(facecolor="#D85A30", label="AUC < 0.70 (Needs work)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"AUC chart saved: {output_path}")


# ─────────────────────────────────────────────
# Plot: training history curves
# ─────────────────────────────────────────────

def plot_training_curves(history_path: str, output_path: str):
    if not Path(history_path).exists():
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs      = range(1, len(history["train"]) + 1)
    train_loss  = [h["loss"] for h in history["train"]]
    val_loss    = [h["loss"] for h in history["val"]]
    train_auc   = [h["auc"]  for h in history["train"]]
    val_auc     = [h["auc"]  for h in history["val"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, label="Train loss", color="#1D9E75", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val loss",   color="#D85A30", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    # AUC
    ax2.plot(epochs, train_auc, label="Train AUC", color="#1D9E75", linewidth=2)
    ax2.plot(epochs, val_auc,   label="Val AUC",   color="#D85A30", linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_title("Training vs Validation AUC")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("OcuNet — Training History", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved: {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def evaluate(config_path: str = "configs/config.yaml"):
    config    = load_config(config_path)
    label_map = load_label_map("configs/label_map.json")
    output_dir = Path(config["paths"]["outputs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load best model
    ckpt_path = output_dir / "ocunet_best.pth"
    if not ckpt_path.exists():
        logger.error(f"No checkpoint found at {ckpt_path}. Train first!")
        return

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded model from epoch {ckpt['epoch']} "
                f"(val_loss={ckpt['val_loss']:.4f}, val_auc={ckpt['val_auc']:.4f})")

    # Test dataloader
    _, _, test_loader = build_dataloaders(config, label_map)
    logger.info(f"Test batches: {len(test_loader)}")

    # Run inference
    labels, probs = run_inference(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)

    # Overall metrics
    active_cols = [i for i in range(NUM_CLASSES) if labels[:, i].sum() > 0]
    overall = {
        "n_samples":       len(labels),
        "macro_auc":       round(roc_auc_score(labels[:, active_cols],
                                                probs[:, active_cols],
                                                average="macro"), 4),
        "macro_f1":        round(f1_score(labels, preds,
                                           average="macro", zero_division=0), 4),
        "macro_precision": round(precision_score(labels, preds,
                                                   average="macro", zero_division=0), 4),
        "macro_recall":    round(recall_score(labels, preds,
                                               average="macro", zero_division=0), 4),
    }

    # Per-disease metrics
    results = per_disease_metrics(labels, probs, label_map)
    print_results(results, overall)

    # Save results JSON
    report = {"overall": overall, "per_disease": results}
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved: {report_path}")

    # Plots
    plot_auc_chart(results, str(output_dir / "per_disease_auc.png"))
    plot_training_curves(
        str(output_dir / "training_history.json"),
        str(output_dir / "training_curves.png")
    )

    return results, overall


if __name__ == "__main__":
    evaluate()