# Forensic Bruise Segmentation

Automatic bruise detection and segmentation in forensic medical images using YOLO and MedSAM.

## Repository Structure

```
bruise_segmentation/
├── configs/
│   └── config.py              ← ALL paths, hyperparameters, constants here
├── src/
│   └── utils.py               ← shared functions (pixel_metrics, binarize, load_gt, ITA, augmentation)
├── scripts/
│   ├── 01_train_yolo.py       ← train YOLO on paul/sarah/gbarimah/majority
│   ├── 02_generate_probability_maps.py  ← save .npy maps (run before 03)
│   ├── 03_threshold_analysis.py         ← 9 methods × 8 models, no GPU
│   ├── 04_temperature_scaling.py        ← spread near-binary maps, re-run threshold
│   └── 05_skin_tone_balancing.py        ← ITA classification + augmentation
├── notebooks/                 ← exploratory notebooks
├── results/                   ← CSVs, plots (gitignored except summaries)
├── docs/                      ← project documentation
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone
git clone <repo_url>
cd bruise_segmentation

# Install dependencies
pip install -r requirements.txt

# Configure paths — edit ONE file
nano configs/config.py
# Change USERNAME = 'tbommawa' to your username
```

## Run Order

Run scripts in this exact order. Each script depends on the previous one.

```bash
# Step 1 — Train models (GPU required, ~2-4 hours each)
python scripts/01_train_yolo.py --dataset paul
python scripts/01_train_yolo.py --dataset sarah
python scripts/01_train_yolo.py --dataset gbarimah
python scripts/01_train_yolo.py --dataset majority        # trains n/s/m/l

# Step 2 — Generate probability maps (GPU required, ~1-2 hours)
# MUST run before Step 3 — otherwise threshold analysis falls back to
# re-running inference and takes 19+ hours
python scripts/02_generate_probability_maps.py

# Step 3 — Threshold analysis (CPU only, ~10 minutes)
python scripts/03_threshold_analysis.py

# Step 4 — Temperature scaling (CPU only, ~20-40 minutes)
python scripts/04_temperature_scaling.py

# Step 5 — Skin tone balancing (CPU only, ~2 hours)
python scripts/05_skin_tone_balancing.py
```

Or run Steps 2→3 chained (automatic, safe):
```bash
python scripts/02_generate_probability_maps.py && python scripts/03_threshold_analysis.py
```

## Key Results

| Model | Dice | IoU | Small Dice | Large Dice |
|---|---|---|---|---|
| majority_yolo26m | **0.7974** | **0.6960** | 0.7555 | 0.8179 |
| majority_yolo26l | 0.7963 | 0.6933 | 0.7241 | 0.8318 |
| majority_yolo26n | 0.7928 | 0.6894 | 0.7498 | 0.8140 |
| gbarimah_yolo26l | 0.7653 | 0.6515 | 0.6940 | 0.8002 |
| sarah_yolo26l    | 0.7428 | 0.6268 | 0.7070 | 0.7603 |
| paul_yolo26l     | 0.7313 | 0.6119 | 0.7163 | 0.7386 |

All models evaluated on the same fixed 304-image test set with majority vote ground truth.

## Learning Rate Search (Optuna)

One model (`majority_yolo26l_best_lr`) was trained using **Optuna** Bayesian hyperparameter optimisation to find the optimal learning rate instead of relying on the Ultralytics default.

**How it works:**
1. Optuna defines a search space for the learning rate (e.g. log-uniform between `1e-5` and `1e-2`)
2. It runs multiple training trials, each with a different learning rate sampled from the search space
3. After each trial it evaluates validation Dice and uses that score to guide the next sample — favouring regions of the search space that showed improvement
4. The best learning rate found is used to train the final model

**Result:** `majority_yolo26l_best_lr` achieved Dice **0.7895**, slightly below the standard `majority_yolo26m` (Dice **0.7974**). This shows that for this dataset size and task, **model size and data quality matter more than the exact learning rate**. The medium model (26m) with default learning rate outperformed the large model (26l) with Optuna-tuned learning rate.

Script: `scripts/01_train_yolo.py --dataset majority --size l` with Optuna integration in `12_optuna_lr_search.py`.

## 5-Fold Cross Validation

Evaluation only — no retraining. The full dataset was split into 5 folds (seed=42). Each model was evaluated on each held-out fold to measure stability across different data partitions.

| Model | Type | CV Dice | CV Std | Notes |
|---|---|---|---|---|
| medsam_majority_aug | MedSAM | **0.9399** | **0.0012** | Triangle threshold |
| medsam_gbarimah_aug | MedSAM | 0.9398 | 0.0013 | Triangle threshold |
| majority_yolo26l_best_lr | YOLO | 0.8116 | 0.0089 | Optuna LR |
| majority_yolo26l | YOLO | 0.8112 | 0.0106 | Standard training |
| gbarimah_yolo26l | YOLO | 0.7667 | 0.0056 | Gbarimah masks |

> **MedSAM is 7× more stable than YOLO** — CV std 0.0012 vs 0.0106. The frozen ViT-B encoder (pretrained on 1.5M medical images) prevents overfitting even on 2,127 training images. Only 4.3% of MedSAM parameters are trainable.

## Key Findings

1. **Label quality > quantity** — majority vote on 3,040 images beats individual annotators on 15,000+ images
2. **Confidence threshold matters** — optimal conf=0.10 gives +0.025 Dice over default 0.25, zero retraining
3. **YOLO outputs near-binary maps** — all 9 threshold methods give identical Dice due to sigmoid on large logits
4. **Temperature scaling** — post-hoc fix (T=5) spreads near-binary values without retraining
5. **Skin tone imbalance** — 225× more dark skin images than light skin; fixed with ITA-based geometric augmentation

## Configuration

All parameters are in `configs/config.py`. To reproduce results exactly:

- `SEED = 42`
- `TEST_RATIO = 0.10`, `VAL_RATIO = 0.20`
- `YOLO_CONF = 0.10` for evaluation
- `PROB_MAP_CONF = 0.05` for probability map generation
- `TEMPERATURES = [1, 2, 5, 10, 20]`

Do not change these values if you want to reproduce the reported numbers.

## Dataset

- 3,040 images where all 3 annotators (Paul, Sarah, Gbarimah) marked a bruise
- Majority vote ground truth: pixel = bruise if ≥ 2/3 annotators agree
- Split: 2,127 train / 609 val / 304 test (fixed, seed=42)

## Server

- GPU: NVIDIA A100-PCIE-40GB (2× GPUs)
- YOLO trains on GPU 1, MedSAM on GPU 0
- Conda env: `bruise`

## Adding a New Script

1. Import config: `from configs.config import ...`
2. Import shared functions: `from src.utils import pixel_metrics, load_gt, binarize`
3. Never hardcode paths — use config variables
4. Never copy-paste `pixel_metrics`, `binarize`, or `load_gt` — import from `src/utils.py`
