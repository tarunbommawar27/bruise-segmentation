"""
scripts/01_train_yolo.py
=========================
Train YOLO segmentation models for all labelers and majority vote.

Usage:
    # Train all datasets
    python scripts/01_train_yolo.py

    # Train a specific dataset only
    python scripts/01_train_yolo.py --dataset majority
    python scripts/01_train_yolo.py --dataset paul
    python scripts/01_train_yolo.py --dataset gbarimah

    # Train a specific model size
    python scripts/01_train_yolo.py --dataset majority --size n
    # sizes: n (nano), s (small), m (medium), l (large)

All hyperparameters are loaded from configs/config.py.
Results saved to: runs/{dataset}_yolo26{size}/
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    YOLO_DATASETS_DIR, RUNS_DIR,
    YOLO_IMGSZ, YOLO_EPOCHS, YOLO_PATIENCE,
    YOLO_BATCH, YOLO_GPU_DEVICE, SEED,
)

# ── Model weights ──────────────────────────────────────────────────────────────
BASE_WEIGHTS = {
    'n': 'yolo26n-seg.pt',
    's': 'yolo26s-seg.pt',
    'm': 'yolo26m-seg.pt',
    'l': 'yolo26l-seg.pt',
}

# ── Datasets to train ─────────────────────────────────────────────────────────
DATASETS = ['paul', 'sarah', 'gbarimah', 'majority']

# ── Sizes per dataset ─────────────────────────────────────────────────────────
# majority gets all 4 sizes; labelers get large only by default
DATASET_SIZES = {
    'majority':  ['n', 's', 'm', 'l'],
    'paul':      ['l'],
    'sarah':     ['l'],
    'gbarimah':  ['l'],
}


def train(dataset: str, size: str):
    yaml_path  = YOLO_DATASETS_DIR / dataset / 'data.yaml'
    run_name   = f'{dataset}_yolo26{size}'
    weights    = BASE_WEIGHTS[size]

    if not yaml_path.exists():
        print(f'SKIP {run_name} — data.yaml not found at {yaml_path}')
        return

    out_dir = RUNS_DIR / run_name
    if (out_dir / 'weights' / 'best.pt').exists():
        print(f'SKIP {run_name} — already trained')
        return

    print(f'\n{"="*60}')
    print(f'Training: {run_name}')
    print(f'  Weights : {weights}')
    print(f'  Data    : {yaml_path}')
    print(f'  Epochs  : {YOLO_EPOCHS}  Batch: {YOLO_BATCH}  ImgSz: {YOLO_IMGSZ}')
    print(f'{"="*60}\n')

    model = YOLO(weights)
    model.train(
        data      = str(yaml_path),
        epochs    = YOLO_EPOCHS,
        imgsz     = YOLO_IMGSZ,
        batch     = YOLO_BATCH,
        patience  = YOLO_PATIENCE,
        device    = YOLO_GPU_DEVICE,
        seed      = SEED,
        project   = str(RUNS_DIR),
        name      = run_name,
        exist_ok  = True,
        task      = 'segment',
        save      = True,
        plots     = True,
        val       = True,
    )
    print(f'\nDone: {run_name}')


def main():
    parser = argparse.ArgumentParser(description='Train YOLO segmentation models')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=DATASETS + ['all'],
                        help='Dataset to train (default: all)')
    parser.add_argument('--size', type=str, default=None,
                        choices=['n', 's', 'm', 'l'],
                        help='Model size (default: use DATASET_SIZES per dataset)')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        sizes = [args.size] if args.size else DATASET_SIZES.get(ds, ['l'])
        for sz in sizes:
            train(ds, sz)

    print('\nAll training complete.')


if __name__ == '__main__':
    main()
