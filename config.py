"""
configs/config.py
==================
Single source of truth for ALL paths, hyperparameters, and constants.
Change your username or base path here and every script updates automatically.
Never hardcode paths inside scripts — always import from here.
"""

from pathlib import Path

# ── User / server ──────────────────────────────────────────────────────────────
USERNAME = 'tbommawa'
BASE     = Path(f'/home/{USERNAME}/bruise_detection')

# ── Core data directories ──────────────────────────────────────────────────────
RAW_DATA_DIR       = BASE / 'raw_data'
MASK_DIR           = RAW_DATA_DIR / 'masks'
THREE_WAY_DIR      = BASE / 'three_way_intersection'
IMAGE_DIR          = THREE_WAY_DIR / 'images'
YOLO_DATASETS_DIR  = THREE_WAY_DIR / 'yolo_datasets'
MAJORITY_MASKS_DIR = THREE_WAY_DIR / 'majority_masks'
RUNS_DIR           = THREE_WAY_DIR / 'runs'

# ── Per-labeler mask directories ───────────────────────────────────────────────
PAUL_MASK_DIR     = MASK_DIR / 'paul'     / 'yes'
SARAH_MASK_DIR    = MASK_DIR / 'sarah'    / 'yes'
GBARIMAH_MASK_DIR = MASK_DIR / 'gbarimah' / 'yes'

# ── Output directories ─────────────────────────────────────────────────────────
PROBABILITY_MAPS_DIR          = THREE_WAY_DIR / 'probability_maps'
THRESHOLD_RESULTS_DIR         = THREE_WAY_DIR / 'threshold_analysis_results'
TEMPERATURE_SCALING_DIR       = THREE_WAY_DIR / 'temperature_scaling_results'
BALANCED_DATASET_DIR          = THREE_WAY_DIR / 'balanced_dataset'
LR_PARAMETERS_DIR             = THREE_WAY_DIR / 'lr_parameters'
PIPELINE_VIS_DIR              = THREE_WAY_DIR / 'pipeline_visualizations'

# ── Model weights ──────────────────────────────────────────────────────────────
MEDSAM_PRETRAINED_WEIGHTS = Path(f'/home/{USERNAME}/medsam_vit_b.pth')

# ── Labelers ───────────────────────────────────────────────────────────────────
LABELERS = ['paul', 'sarah', 'gbarimah']

# ── YOLO models to evaluate ────────────────────────────────────────────────────
YOLO_MODELS = [
    'majority_yolo26m',
    'majority_yolo26l',
    'majority_yolo26n',
    'majority_yolo26s',
    'majority_yolo26l_best_lr',
    'gbarimah_yolo26l',
    'sarah_yolo26l',
    'paul_yolo26l',
]

# ── Dataset split ──────────────────────────────────────────────────────────────
SEED       = 42
TEST_RATIO = 0.10   # 10%
VAL_RATIO  = 0.20   # 20%
# Train = remaining 70%

# ── YOLO training hyperparameters ─────────────────────────────────────────────
YOLO_IMGSZ    = 640
YOLO_EPOCHS   = 100
YOLO_PATIENCE = 20
YOLO_BATCH    = 8
YOLO_DEVICE   = 0       # GPU index
YOLO_CONF     = 0.10    # optimal confidence for evaluation (found by sweep)
YOLO_IOU      = 0.45

# ── Probability map generation ────────────────────────────────────────────────
PROB_MAP_CONF = 0.05    # very low — keep all soft values before threshold

# ── Threshold methods ─────────────────────────────────────────────────────────
THRESHOLD_METHODS = [
    'fixed_0.50', 'fixed_0.30', 'fixed_0.10',
    'otsu', 'triangle', 'li', 'kapur', 'mcet', 'balanced',
]

# ── Temperature scaling ───────────────────────────────────────────────────────
TEMPERATURES = [1, 2, 5, 10, 20]

# ── ITA skin tone bins (Chardon et al. 1991) ──────────────────────────────────
ITA_BINS = [
    ('very_light',     55,  180),
    ('light',          28,   55),
    ('intermediate',   10,   28),
    ('tan',           -30,   10),
    ('dark',          -55,  -30),
    ('very_dark',    -180,  -55),
]

# ── Skin tone balancing ───────────────────────────────────────────────────────
MAX_IMAGE_PIXELS_ITA = 2_000_000   # resize images above this before ITA compute
AUGMENTATION_SCALE   = (0.85, 1.15)
N_TARGET             = 300         # Effect Size Targeting: min samples per bin
REDUNDANCY_CAP       = 15          # max geometric variants per original image

# ── MedSAM ────────────────────────────────────────────────────────────────────
MEDSAM_IMGSZ      = 1024
MEDSAM_LR         = 1e-4
MEDSAM_EPOCHS     = 50
MEDSAM_PATIENCE   = 10
MEDSAM_BATCH      = 4
MEDSAM_DEVICE     = 0

# ── Size stratification ───────────────────────────────────────────────────────
SIZE_PERCENTILE_CUTOFF = 33    # bottom 33% = small bruises

# ── K-Fold cross validation ───────────────────────────────────────────────────
N_FOLDS = 5

# ── GPU devices ───────────────────────────────────────────────────────────────
YOLO_GPU_DEVICE   = 1   # GPU for YOLO training
MEDSAM_GPU_DEVICE = 0   # GPU for MedSAM training
