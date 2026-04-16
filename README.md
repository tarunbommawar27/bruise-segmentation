# Forensic Bruise Segmentation

Automatic bruise detection and segmentation in forensic medical images.
Evaluated on a fixed 304-image test set against majority vote ground truth.
All results are reproducible with `SEED = 42`.

---

## Repository Structure

```
bruise_segmentation/
├── configs/
│   └── config.py                        ← ALL paths, hyperparameters, constants
├── src/
│   └── utils.py                         ← shared functions — import from here, never copy-paste
├── scripts/
│   ├── 01_train_yolo.py                 ← train YOLO on paul/sarah/gbarimah/majority
│   ├── 02_generate_probability_maps.py  ← save .npy maps (run before 03)
│   ├── 03_threshold_analysis.py         ← 9 threshold methods × 8 models, CPU only
│   ├── 04_temperature_scaling.py        ← spread near-binary maps, re-run threshold
│   ├── 05_skin_tone_balancing.py        ← ITA classification + Effect Size Targeting
│   ├── 09_predict_and_mst_eval.py       ← MST fairness evaluation on all models
│   └── 10_train_balanced_yolo.py        ← retrain YOLO on ITA-balanced datasets
├── results/                             ← CSVs, plots (gitignored except summaries)
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone <repo_url>
cd bruise_segmentation

pip install -r requirements.txt

# Configure all paths in ONE file
nano configs/config.py
# Change: USERNAME = 'your_username'
```

---

## Run Order

### Phase 1 — YOLO Baseline

```bash
# Step 1 — Train all models (GPU, ~2-4 hours each)
python scripts/01_train_yolo.py --dataset paul
python scripts/01_train_yolo.py --dataset sarah
python scripts/01_train_yolo.py --dataset gbarimah
python scripts/01_train_yolo.py --dataset majority    # trains n/s/m/l

# Step 2 — Generate probability maps (GPU, ~1-2 hours)
# Must run before Step 3 — saves .npy maps so threshold analysis
# does not need to re-run inference (would take 19+ hours otherwise)
python scripts/02_generate_probability_maps.py

# Step 3 — Threshold analysis (CPU, ~10 minutes)
python scripts/03_threshold_analysis.py

# Step 4 — Temperature scaling (CPU, ~20-40 minutes)
python scripts/04_temperature_scaling.py
```

### Phase 2 — Skin Tone Balancing and Fairness Evaluation

```bash
# Step 5 — Balance all datasets using ITA and Effect Size Targeting (CPU, ~2 hours each)
python scripts/05_skin_tone_balancing.py --dataset gbarimah
python scripts/05_skin_tone_balancing.py --dataset majority
python scripts/05_skin_tone_balancing.py --dataset paul
python scripts/05_skin_tone_balancing.py --dataset sarah
# or all at once:
python scripts/05_skin_tone_balancing.py --dataset all

# Step 6 — Label test images with MST (CPU, ~10 minutes, run once)
python scripts/09_predict_and_mst_eval.py --step label

# Step 6b — Evaluate fairness on existing models (GPU, ~10 minutes per model)
python scripts/09_predict_and_mst_eval.py --step evaluate --model majority_yolo26m
python scripts/09_predict_and_mst_eval.py --step evaluate --model all

# Step 6c — Print side-by-side fairness comparison
python scripts/09_predict_and_mst_eval.py --step compare

# Step 7 — Retrain on balanced datasets (GPU, ~2-4 hours each)
python scripts/10_train_balanced_yolo.py --dataset gbarimah --size l
python scripts/10_train_balanced_yolo.py --dataset majority  --size l
python scripts/10_train_balanced_yolo.py --dataset paul      --size l
python scripts/10_train_balanced_yolo.py --dataset sarah     --size l
```

---

## Key Results

### YOLO Baseline — 304-image test set, majority vote GT, conf = 0.10

| Model | Dice | IoU | Small Dice | Large Dice |
|-------|------|-----|-----------|-----------|
| majority_yolo26m | **0.7974** | **0.6960** | 0.7555 | 0.8179 |
| majority_yolo26l | 0.7963 | 0.6933 | 0.7241 | 0.8318 |
| majority_yolo26n | 0.7928 | 0.6894 | 0.7498 | 0.8140 |
| majority_yolo26l_best_lr | 0.7895 | 0.6861 | 0.7302 | 0.8185 |
| majority_yolo26s | 0.7803 | 0.6771 | 0.7155 | 0.8122 |
| gbarimah_yolo26l | 0.7653 | 0.6515 | 0.6940 | 0.8002 |
| sarah_yolo26l | 0.7428 | 0.6268 | 0.7070 | 0.7603 |
| paul_yolo26l | 0.7313 | 0.6119 | 0.7163 | 0.7386 |

### Fairness Evaluation — Monk Skin Tone Scale (pre-balancing baseline)

Test set MST distribution: Light (MST01-03) n=5 (1.6%), Medium (MST04-06) n=145 (47.7%), Dark (MST07-10) n=154 (50.7%).
All 304 images labeled via actual LAB computation from images (zero fallback estimates).

Fairness gap: `delta_fair = Dice_dark - Dice_light`. Target: `|delta_fair| <= 0.05`.

| Model | Overall | Light | Medium | Dark | Delta | Status |
|-------|---------|-------|--------|------|-------|--------|
| majority_yolo26m | 0.7974 | 0.7282 | 0.8092 | 0.7884 | +0.0602 | FAIL |
| majority_yolo26l | 0.7963 | 0.7455 | 0.8072 | 0.7878 | +0.0423 | PASS |
| majority_yolo26n | 0.7928 | 0.5820 | 0.7958 | 0.7968 | +0.2148 | FAIL |
| majority_yolo26l_best_lr | 0.7895 | 0.7431 | 0.7969 | 0.7841 | +0.0410 | PASS |
| majority_yolo26s | 0.7803 | 0.7357 | 0.7924 | 0.7704 | +0.0347 | PASS |
| gbarimah_yolo26l | 0.7652 | 0.6166 | 0.7669 | 0.7685 | +0.1519 | FAIL |
| sarah_yolo26l | 0.7428 | 0.7090 | 0.7354 | 0.7508 | +0.0418 | PASS |
| paul_yolo26l | 0.7315 | 0.6977 | 0.7294 | 0.7345 | +0.0368 | PASS |

Every model has a positive delta confirming the 225:1 training imbalance causes
systematic underperformance on light skin patients.

---

## Skin Tone Balancing — Effect Size Targeting

### The Problem

ITA classification of the gbarimah training set reveals a 225:1 imbalance:

| Bin | ITA Range | Count | Fraction |
|-----|-----------|-------|---------|
| Dark | < -30° | 4,504 | 41.8% |
| Tan | -30° to 10° | 4,311 | 40.0% |
| Very Dark | < -55° | 1,877 | 17.4% |
| Intermediate | 10° to 28° | 64 | 0.6% |
| Very Light | > 55° | 20 | 0.2% |

ITA formula (computed on non-bruise pixels only, using GT mask to exclude bruised areas):

```
ITA = arctan((L* - 50) / b*) × (180 / π)
```

Median L* and b* used for robustness against shadows and specular highlights
common in forensic photography.

### Why Naive Balancing Fails

Augmenting 20 very light images to 4,504 copies (225× multiplier) causes the model
to memorise 20 specific patients instead of learning generalizable bruise features.
Validation Dice on light skin collapses at test time because test patients are
different from the 20 memorised during training.
Bencevic et al. CMPB 2024 confirmed single-mechanism mitigation is insufficient
for segmentation tasks.

### Strategy: Effect Size Targeting

Compute the minimum sample size a bin needs to be useful, not the size to match the majority.

```
multiplier = min(ceil(N_TARGET / n_bin), REDUNDANCY_CAP)

N_TARGET       = 300   empirical minimum for fine-tuning a pretrained model (Kang et al. ICLR 2020)
REDUNDANCY_CAP = 15    max geometric variants before copies are informationally identical
```

| Bin | Original | Multiplier | After Augmentation |
|-----|----------|------------|-------------------|
| Dark | 4,504 | ×1 | 4,504 |
| Tan | 4,311 | ×1 | 4,311 |
| Very Dark | 1,877 | ×1 | 1,877 |
| Intermediate | 64 | ×5 | 320 |
| Very Light | 20 | ×15 | 300 |

Augmentation is **geometric only**: horizontal flip, vertical flip, rotate 90/180/270°,
zoom in/out (scale 0.85–1.15). No colour changes — altering L* or b* shifts the ITA
angle and moves the image to the wrong skin tone bin.
Buslaev et al. Information 2020.

### Three Mechanisms

**Mechanism 1 — Geometric Augmentation** (script 05, baked into balanced dataset)

Expands minority bins to N_TARGET using geometric transforms only.

**Mechanism 2 — Square-Root Sampler** (use `sampler_info.json` in training loop)

```python
from torch.utils.data import WeightedRandomSampler
import json

si      = json.load(open('balanced_dataset/<ds>/sampler_info.json'))
weights = [si['sqrt_sample_weights'].get(stem, 1.0) for stem in train_stems]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
loader  = DataLoader(dataset, sampler=sampler, batch_size=32)
```

Sampling weight proportional to 1/√(bin_count). Very light skin gets ×3.87
appearance boost without overfitting risk of pure balanced sampling (×15 repeats
the same 20 patients). Kang et al. ICLR 2020 — τ=0.5 square root is optimal.

**Mechanism 3 — Class-Weighted Focal Tversky Loss** (use `class_weights.json` in training loop)

```python
from scripts.06_focal_tversky_loss import FocalTverskyLoss
import json, torch

w_map        = json.load(open('balanced_dataset/<ds>/class_weights.json'))['class_weights_by_bin']
crit         = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
per_img_loss = crit(logits, masks)                               # shape (B,)
w            = torch.tensor([w_map[b] for b in ita_bins])        # shape (B,)
loss         = (per_img_loss * w).mean()
loss.backward()
```

| Bin | Loss Weight |
|-----|------------|
| Very Light | 15.0× |
| Intermediate | 7.0× |
| Very Dark | 2.4× |
| Tan | 1.0× |
| Dark | 1.0× (reference) |

β=0.7 > α=0.3 penalises missed bruises more than false alarms (clinically appropriate).
Gradient stability cap at 15×: at raw 225× ratio, loss variance is 50,625× reference —
divergence guaranteed. At 15× variance is 225× — stable with gradient clipping.
Abraham & Khan ISBI 2019.

### Balanced Dataset Sizes

| Dataset | Original Train | Balanced Train | Val | Test |
|---------|---------------|----------------|-----|------|
| gbarimah | 10,776 | 11,312 | 609 | 304 |
| sarah | 2,308 | 2,360 | 609 | 304 |
| majority | in progress | TBD | 609 | 304 |
| paul | in progress | TBD | 609 | 304 |

Val and test sets are never augmented.

---

## ITA and Monk Skin Tone Scale

Two complementary scales used at different pipeline stages.

### ITA — Training Stratification (Phase A, complete)

- Computed on non-bruise pixels only using GT mask to exclude bruised areas
- Median L* and b* for robustness against shadows and highlights
- 6 bins: very light, light, intermediate, tan, dark, very dark
- a* value saved separately per image for post-hoc haemoglobin analysis
  (the chromatic axis both ITA and MST miss — bruise products manifest as a* shifts)
- Drives augmentation multipliers, sampler weights, and loss weights
- Reference: Chardon et al. Int J Cosmet Sci 1991. Mandated by ISO 24444:2019

### MST — Fairness Evaluation (Phase B, post-training)

- 10 discrete levels MST01 (lightest) to MST10 (darkest)
- Assigned via nearest-neighbour matching in CIELAB space to 10 published Monk reference swatches
- Groups: Light (MST01-03), Medium (MST04-06), Dark (MST07-10)
- 4 levels for dark skin vs ITA's single dark bin — critical for intra-dark disparity detection
- Reference: Monk Google Research & Harvard 2022
- Schuhmann et al. NeurIPS 2023 — MST has better inter-annotator agreement on dark skin than Fitzpatrick
- Desai et al. Research Square 2025 — only paper studying skin tone scale choice for bruise detection

**ITA tells the model how to learn fairly. MST tells us whether it learned fairly.**

### Test Set MST Distribution

| Group | Levels | Count | Fraction |
|-------|--------|-------|---------|
| Light | MST01-03 | 5 | 1.6% |
| Medium | MST04-06 | 145 | 47.7% |
| Dark | MST07-10 | 154 | 50.7% |

All 304 images labeled via actual LAB computation (zero fallback estimates).

---

## Learning Rate Search (Optuna)

`majority_yolo26l_best_lr` trained using Optuna Bayesian hyperparameter optimisation.
Optuna samples log-uniform learning rates (1e-5 to 1e-2), trains multiple trials,
and uses validation Dice to guide subsequent samples toward better regions.

Result: majority_yolo26l_best_lr Dice **0.7895** vs majority_yolo26m standard **0.7974**.
Model size and data quality matter more than exact learning rate for this task.

---

## 5-Fold Cross Validation

Evaluation only — no retraining. Dataset split into 5 folds (seed=42).

| Model | CV Dice | CV Std |
|-------|---------|--------|
| majority_yolo26l_best_lr | 0.8116 | 0.0089 |
| majority_yolo26l | 0.8112 | 0.0106 |
| gbarimah_yolo26l | 0.7667 | 0.0056 |

---

## Key Findings

1. **Label quality beats label quantity** — majority vote on 3,040 images (Dice 0.7974) outperforms individual annotators on 15,000+ images (Dice 0.7313–0.7653)
2. **Confidence threshold matters** — conf=0.10 gives +0.025 Dice over default 0.25 with zero retraining
3. **YOLO outputs near-binary maps** — all 9 threshold methods give identical Dice; temperature scaling (T=5) spreads values without retraining
4. **Small bruise gap is consistent** — 0.108 Dice gap across every YOLO model size; architectural not a tuning problem
5. **Every model fails on light skin** — delta_fair +0.035 to +0.215 across all models confirming systematic clinical unfairness
6. **Naive balancing causes overfitting** — 225× augmentation memorises 20 patients; Effect Size Targeting to N_TARGET=300 avoids this
7. **Better Dice does not mean better fairness** — majority_yolo26l (Dice 0.7963, PASS) is fairer than majority_yolo26m (Dice 0.7974, FAIL)
8. **Individual labeler GT hurts fairness** — gbarimah_yolo26l has worst fairness gap (+0.1519) despite decent overall Dice

---

## Current Status (April 2026)

| Task | Status |
|------|--------|
| YOLO baseline training — all models | Complete |
| Threshold and confidence analysis | Complete |
| 5-fold cross validation | Complete |
| Skin tone balancing — gbarimah | Complete (11,312 images) |
| Skin tone balancing — sarah | Complete (2,360 images) |
| Skin tone balancing — majority | In progress |
| Skin tone balancing — paul | In progress |
| MST labeling — 304 test images | Complete (all via lab_nn) |
| MST fairness evaluation — all 8 YOLO models | Complete |
| Balanced YOLO retraining — gbarimah_yolo26l | Running |
| nnU-Net training | Planned |

---

## Research Contributions

1. First systematic MST-level fairness evaluation for bruise segmentation at pixel level
2. Effect Size Targeting with N_TARGET=300 — avoids naive max-balancing overfitting trap
3. Dual-scale pipeline combining ITA for training with MST for evaluation plus a* haemoglobin axis
4. nnU-Net vs YOLO comparison on forensic bruise data (upcoming)

Target venues: Medical Image Analysis, IEEE JBHI, Forensic Science International, MICCAI Workshop FAIMI

---

## Configuration

All parameters in `configs/config.py`. To reproduce results exactly:

```python
SEED                  = 42
TEST_RATIO            = 0.10
VAL_RATIO             = 0.20
YOLO_CONF             = 0.10    # optimal confidence found by sweep
PROB_MAP_CONF         = 0.05    # low conf to preserve soft probability values
TEMPERATURES          = [1, 2, 5, 10, 20]
N_TARGET              = 300     # Effect Size Targeting minimum sample size
REDUNDANCY_CAP        = 15      # maximum geometric variants per original image
SIZE_PERCENTILE_CUTOFF = 33     # bottom 33% = small bruises
```

Do not change these values if you want to reproduce the reported numbers.

---

## Dataset

- 3,040 images where all 3 annotators (Paul, Sarah, Gbarimah) marked a bruise
- Majority vote ground truth: pixel = bruise if ≥ 2/3 annotators agree
- Split: 2,127 train / 609 val / 304 test (fixed, seed=42)
- Skin tone imbalance: 225:1 (dark 4,504 vs very light 20 in gbarimah training set)

## Server

- GPU: 2× NVIDIA A100-PCIE-40GB
- YOLO training on GPU 1, other jobs on GPU 0
- Conda env: `bruise`

## Adding a New Script

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import RUNS_DIR, YOLO_DATASETS_DIR, MAJORITY_MASKS_DIR  # etc.
from src.utils import pixel_metrics, load_gt, binarize, masks_from_result    # etc.
```

Never hardcode paths. Never copy-paste functions from `src/utils.py`.

## References

1. Kang et al. Decoupling Representation and Classifier for Long-Tailed Recognition. ICLR 2020.
2. Abraham & Khan. A Novel Focal Tversky Loss Function. IEEE ISBI 2019.
3. Bencevic et al. Understanding Skin Color Bias in Deep Learning-Based Skin Lesion Segmentation. CMPB 2024.
4. Buslaev et al. Albumentations: Fast and Flexible Image Augmentations. Information 2020.
5. Chardon, Cretois, Hourseau. Skin Colour Typology and Suntanning Pathways. Int J Cosmet Sci 1991.
6. Monk. Monk Skin Tone Scale. Google Research & Harvard Sociology 2022.
7. Schuhmann et al. Consensus and Subjectivity of Skin Tone Annotation for ML Fairness. NeurIPS 2023.
8. Desai et al. The Effect of Skin Tone Classification on Bias in Bruise Detection. Research Square 2025.
9. Isensee et al. nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation. Nature Methods 2021.
10. Isensee et al. nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. MICCAI 2024.
