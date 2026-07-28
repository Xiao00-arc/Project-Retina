# Iris Disease Detection using CNN
### Multi-label, 46-disease classification from fundus images
**Hardware target: RTX 3050 4GB | Python 3.10 | PyTorch 2.2 + CUDA 12.1**

---

## Project Structure

```
iris_disease_detection/
│
├── configs/
│   ├── config.yaml          ← All hyperparameters, paths, hardware settings
│   └── label_map.json       ← Unified disease ID map across all 3 datasets
│
├── data/
│   ├── raw/                 ← Downloaded datasets go here (gitignored)
│   │   ├── ODIR-5K/
│   │   ├── RFMiD/
│   │   └── JSIEC/
│   ├── processed/           ← CLAHE + resized images saved as .npy
│   └── augmented/           ← (used at runtime, not pre-saved)
│
├── src/
│   ├── dataset/
│   │   └── loader.py        ← Unified Dataset + DataLoader builder
│   ├── preprocessing/
│   │   └── pipeline.py      ← CLAHE, resize, normalize, deduplication
│   └── utils/
│       ├── download_datasets.py  ← Kaggle auto-download
│       └── dataset_stats.py      ← Class distribution, integrity check
│
├── outputs/                 ← Charts, logs, checkpoints
├── requirements.txt
└── README.md
```

---

## Quick Start

### Step 1 — Install dependencies

```bash
# Install PyTorch with CUDA 12.1 (RTX 3050)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install -r requirements.txt
```

### Step 2 — Set up Kaggle API

```bash
pip install kaggle
# Download kaggle.json from https://www.kaggle.com/settings → API
mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3 — Download datasets

```bash
python src/utils/download_datasets.py
```

Downloads ODIR-5K (~10K images), RFMiD (~3.2K images), JSIEC (~10K images) into `data/raw/`.

### Step 4 — Run preprocessing pipeline

```bash
python src/preprocessing/pipeline.py
```

Applies CLAHE → resize → normalize → deduplication. Saves to `data/processed/`.

### Step 5 — Check dataset stats

```bash
python src/utils/dataset_stats.py
```

Prints class distribution report and saves `outputs/class_distribution.png`.

### Step 6 — Verify data loader

```bash
python src/dataset/loader.py
```

Should print:
```
Image batch shape : torch.Size([16, 3, 224, 224])
Label batch shape : torch.Size([16, 46])
Active labels in first sample: [1, 4]
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Backbone | EfficientNet-B0 | Best accuracy/VRAM tradeoff for RTX 3050 |
| Loss | Binary Cross Entropy | Multi-label: each class is independent |
| Sampler | WeightedRandomSampler | Rare classes seen more often per batch |
| Preprocessing | CLAHE in LAB space | Preserves hue, enhances local contrast |
| Deduplication | MD5 hash | Removes exact duplicates across datasets |
| Mixed precision | torch.cuda.amp | Saves ~1.2GB VRAM on RTX 3050 |

---

## Coming Next

- `src/model/efficientnet.py` — EfficientNet-B0 + custom head
- `src/train.py` — Training loop with AMP + early stopping
- `src/evaluate.py` — AUC, F1, accuracy per disease
- `src/gradcam.py` — Visual explainability heatmaps
