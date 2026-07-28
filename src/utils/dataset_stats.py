"""
utils/dataset_stats.py
-----------------------
Run this BEFORE training to:
  - Check class distribution across all datasets
  - Detect imbalance visually
  - Verify label integrity
  - Print a summary report
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(p="configs/config.yaml"):
    with open(p) as f: return yaml.safe_load(f)

def load_label_map(p="configs/label_map.json"):
    with open(p) as f: return json.load(f)


def compute_class_distribution(samples: list, num_classes: int = 46,
                                label_map: dict = None) -> dict:
    """Count how many samples have each disease label active."""
    counts = np.zeros(num_classes, dtype=int)
    for _, label_vec in samples:
        counts += label_vec.astype(int)

    disease_names = {}
    if label_map:
        disease_names = {int(k): v for k, v in label_map["global_labels"].items()}

    distribution = {}
    for i, count in enumerate(counts):
        name = disease_names.get(i, f"Class {i}")
        distribution[name] = int(count)

    return distribution


def plot_class_distribution(distribution: dict, output_path: str = "outputs/class_distribution.png"):
    """Save a horizontal bar chart of class distribution."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort by count descending
    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    names  = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))
    colors = ["#1D9E75" if c > 500 else "#BA7517" if c > 100 else "#D85A30"
              for c in counts]

    bars = ax.barh(names, counts, color=colors, edgecolor="none")

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)

    ax.set_xlabel("Number of samples", fontsize=11)
    ax.set_title("Class distribution across combined datasets", fontsize=13, pad=12)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1D9E75", label=">500 samples (well-represented)"),
        Patch(facecolor="#BA7517", label="100-500 samples (moderate)"),
        Patch(facecolor="#D85A30", label="<100 samples (rare — needs SMOTE)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Class distribution chart saved: {output_path}")


def print_summary_report(samples: list, distribution: dict):
    """Print a clean summary to terminal."""
    total = len(samples)
    normal_count = distribution.get("Normal", 0)
    diseased_count = total - normal_count

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY REPORT")
    print("=" * 60)
    print(f"  Total images        : {total:,}")
    print(f"  Normal images       : {normal_count:,} ({100*normal_count/total:.1f}%)")
    print(f"  Diseased images     : {diseased_count:,} ({100*diseased_count/total:.1f}%)")
    print(f"  Unique diseases     : {sum(1 for v in distribution.values() if v > 0)}")

    # Imbalance warning
    counts = list(distribution.values())
    if counts:
        max_c, min_c = max(counts), min(c for c in counts if c > 0)
        ratio = max_c / max(min_c, 1)
        print(f"\n  Max class count     : {max_c:,}")
        print(f"  Min class count     : {min_c:,}")
        print(f"  Imbalance ratio     : {ratio:.1f}x", end=" ")
        if ratio > 20:
            print("  ⚠ HIGH — weighted sampler essential")
        elif ratio > 5:
            print("  ⚠ MODERATE — weighted sampler recommended")
        else:
            print("  ✓ Acceptable")

    # Rare class warning
    rare = [name for name, count in distribution.items() if 0 < count < 100]
    if rare:
        print(f"\n  Rare classes (<100 samples): {len(rare)}")
        for name in rare[:10]:
            print(f"    - {name}: {distribution[name]}")
        if len(rare) > 10:
            print(f"    ... and {len(rare) - 10} more")

    print("=" * 60 + "\n")


def check_label_integrity(samples: list, num_classes: int = 46) -> bool:
    """
    Verify label vectors are valid:
    - Shape must be (46,)
    - Values must be 0 or 1 only
    - At least one label must be active per sample
    """
    errors = 0
    for i, (path, label_vec) in enumerate(samples):
        if len(label_vec) != num_classes:
            logger.error(f"Sample {i} ({path}): wrong label length {len(label_vec)}")
            errors += 1
        if not np.all((label_vec == 0) | (label_vec == 1)):
            logger.error(f"Sample {i} ({path}): non-binary values in label")
            errors += 1
        if label_vec.sum() == 0:
            logger.warning(f"Sample {i} ({path}): no active labels (all zeros)")

    if errors == 0:
        logger.info(f"Label integrity check PASSED for {len(samples)} samples")
        return True
    else:
        logger.error(f"Label integrity check FAILED: {errors} errors found")
        return False


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.dataset.loader import load_odir5k, load_rfmid, load_jsiec

    config    = load_config("configs/config.yaml")
    label_map = load_label_map("configs/label_map.json")

    all_samples = []
    all_samples.extend(load_odir5k(config, label_map))
    all_samples.extend(load_rfmid(config, label_map))
    all_samples.extend(load_jsiec(config, label_map))

    logger.info(f"Total samples loaded: {len(all_samples)}")

    # Check integrity
    check_label_integrity(all_samples)

    # Compute and plot distribution
    distribution = compute_class_distribution(all_samples, label_map=label_map)
    print_summary_report(all_samples, distribution)
    plot_class_distribution(distribution, output_path="outputs/class_distribution.png")
