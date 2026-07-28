"""
utils/download_datasets.py
---------------------------
Downloads all three datasets from Kaggle automatically.
Run this FIRST before anything else.

Requirements:
  pip install kaggle
  Place kaggle.json at ~/.kaggle/kaggle.json
  (Download from: https://www.kaggle.com/settings → API → Create Token)
"""

import os
import zipfile
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

DATASETS = [
    {
        "name": "ODIR-5K",
        "kaggle_id": "andrewmvd/ocular-disease-recognition-odir5k",
        "output_dir": RAW_DIR / "ODIR-5K",
        "description": "8 eye diseases, 10,000 fundus images"
    },
    {
        "name": "RFMiD",
        "kaggle_id": "andrewmvd/retinal-disease-classification",
        "output_dir": RAW_DIR / "RFMiD",
        "description": "46 conditions, 3,200 images"
    },
    {
        "name": "JSIEC",
        "kaggle_id": "linchundan/fundusimage1000",
        "output_dir": RAW_DIR / "JSIEC",
        "description": "39 conditions, ~10,000 images"
    }
]


def check_kaggle_api():
    """Verify kaggle CLI and credentials are set up."""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Kaggle API ready: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("\n" + "=" * 60)
    print("  KAGGLE API NOT FOUND")
    print("=" * 60)
    print("  1. pip install kaggle")
    print("  2. Go to: https://www.kaggle.com/settings")
    print("  3. API section → 'Create New Token'")
    print("  4. Move kaggle.json to: ~/.kaggle/kaggle.json")
    print("  5. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac)")
    print("=" * 60 + "\n")
    return False


def download_dataset(dataset: dict):
    """Download and extract a single Kaggle dataset."""
    out_dir = dataset["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    existing = list(out_dir.glob("*.csv")) + list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
    if existing:
        logger.info(f"{dataset['name']} already exists — skipping download.")
        return

    logger.info(f"Downloading {dataset['name']} ({dataset['description']})...")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset["kaggle_id"],
        "-p", str(out_dir),
        "--unzip"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info(f"{dataset['name']} downloaded successfully → {out_dir}")
    else:
        logger.error(f"Failed to download {dataset['name']}: {result.stderr}")
        print(f"\n  Manual download URL:")
        print(f"  https://www.kaggle.com/datasets/{dataset['kaggle_id']}\n")


def verify_downloads():
    """Check that each dataset folder has the expected files."""
    print("\n" + "=" * 60)
    print("  DOWNLOAD VERIFICATION")
    print("=" * 60)

    all_ok = True
    for ds in DATASETS:
        out_dir = ds["output_dir"]
        img_files = list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.png"))
        csv_files = list(out_dir.rglob("*.csv"))

        status = "✓" if (img_files and csv_files) else "✗"
        if not (img_files and csv_files):
            all_ok = False

        print(f"  {status} {ds['name']:12} — {len(img_files):,} images, {len(csv_files)} CSV files")

    print("=" * 60)
    if all_ok:
        print("  All datasets ready. Run preprocessing next.\n")
    else:
        print("  Some datasets missing — check above.\n")
    return all_ok


if __name__ == "__main__":
    if not check_kaggle_api():
        exit(1)

    for dataset in DATASETS:
        download_dataset(dataset)

    verify_downloads()

    print("Next step:")
    print("  python src/preprocessing/pipeline.py")
    print("  python src/utils/dataset_stats.py\n")
