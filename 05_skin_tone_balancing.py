"""
scripts/05_skin_tone_balancing.py
====================================
Balance training datasets by skin tone using ITA (Individual Typology Angle).

STRATEGY — Effect Size Targeting
---------------------------------
The training set has a 225:1 skin tone imbalance (dark 4,504 vs very light 20).
Naive balancing (augmenting 20 images to 4,504 copies) causes severe overfitting:
the model memorises 20 specific patients instead of learning generalizable features.

Effect Size Targeting computes the minimum sample size a bin needs to contribute
useful gradient signal, not the size to match the majority:

  Formula: multiplier = min(ceil(N_TARGET / n_bin), REDUNDANCY_CAP)

  N_TARGET       = 300  empirical minimum for fine-tuning a pretrained model
                        (Kang et al. ICLR 2020)
  REDUNDANCY_CAP = 15   beyond 15 geometric variants of the same image,
                        new copies carry zero additional information

  Applied to gbarimah train set:
    dark         4,504  already >> N_TARGET  x1   stays 4,504
    tan          4,311  already >> N_TARGET  x1   stays 4,311
    very_dark    1,877  already >> N_TARGET  x1   stays 1,877
    intermediate    64  ceil(300/64)=5       x5   becomes 320
    very_light      20  ceil(300/20)=15      x15  becomes 300

THREE MECHANISMS
----------------
This script handles Mechanism 1 and produces files for Mechanisms 2 and 3.

  Mechanism 1 — Geometric augmentation (this script)
    Expands minority bins to N_TARGET. No colour changes ever — colour
    changes shift the ITA angle and move images to the wrong bin.
    (Buslaev et al. Information 2020)

  Mechanism 2 — Square-root sampler (use sampler_info.json in training)
    p_bin proportional to 1/sqrt(n_bin_after_aug). Very light gets a 3.87x
    boost without overfitting risk of pure balanced sampling.
    (Kang et al. ICLR 2020 — tau=0.5 is optimal)

  Mechanism 3 — Class-weighted Focal Tversky loss (use class_weights.json)
    w_bin = min(n_dark / n_bin_after_aug, cap). Beta=0.7 > alpha=0.3
    penalises missed bruises more than false alarms (clinically appropriate).
    Gradient stability cap at 15x: at 225x raw ratio, loss variance is
    50,625x reference — divergence guaranteed. At 15x variance is 225x, stable.
    (Abraham & Khan ISBI 2019)

USAGE
-----
    python scripts/05_skin_tone_balancing.py --dataset gbarimah
    python scripts/05_skin_tone_balancing.py --dataset majority
    python scripts/05_skin_tone_balancing.py --dataset paul
    python scripts/05_skin_tone_balancing.py --dataset sarah
    python scripts/05_skin_tone_balancing.py --dataset all

OUTPUTS (balanced_dataset/<dataset>/)
--------------------------------------
    images/train/         augmented train images
    images/val/           original val images (no augmentation)
    images/test/          original test images (no augmentation)
    masks/train/          corresponding masks
    masks/val/
    masks/test/
    dataset.yaml          YOLO-format dataset config with correct paths
    skin_tone_report.csv  per-image ITA bin and augmentation info
    class_weights.json    class weights for Focal Tversky loss (Mechanism 3)
    sampler_info.json     sqrt-sampling weights for WeightedRandomSampler (Mechanism 2)
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    YOLO_DATASETS_DIR,
    MAJORITY_MASKS_DIR,
    GBARIMAH_MASK_DIR,
    PAUL_MASK_DIR,
    SARAH_MASK_DIR,
    BALANCED_DATASET_DIR,
    SEED,
    AUGMENTATION_SCALE,
)
from src.utils import compute_ita, ita_to_bin, augment_pair

random.seed(SEED)
np.random.seed(SEED)

# ── Effect Size Targeting constants ───────────────────────────────────────────
N_TARGET       = 300
REDUNDANCY_CAP = 15

# ── Gradient stability caps for class weights ─────────────────────────────────
CLASS_WEIGHT_CAPS = {
    'dark':         1.0,
    'tan':          1.0,
    'very_dark':    2.4,
    'intermediate': 7.0,
    'very_light':   15.0,
    'light':        3.0,
    'unknown':      1.0,
}

BATCH_SIZE    = 32
MIN_PER_BATCH = {'very_light': 1, 'intermediate': 2}

# ── Dataset configurations ────────────────────────────────────────────────────
DATASETS_CFG = {
    'majority': {
        'imgs':  YOLO_DATASETS_DIR / 'majority' / 'images',
        'masks': MAJORITY_MASKS_DIR,
    },
    'gbarimah': {
        'imgs':  YOLO_DATASETS_DIR / 'gbarimah' / 'images',
        'masks': GBARIMAH_MASK_DIR,
    },
    'paul': {
        'imgs':  YOLO_DATASETS_DIR / 'paul' / 'images',
        'masks': PAUL_MASK_DIR,
    },
    'sarah': {
        'imgs':  YOLO_DATASETS_DIR / 'sarah' / 'images',
        'masks': SARAH_MASK_DIR,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_mask(stem: str, masks_dir: Path, h: int, w: int) -> np.ndarray:
    """Load and resize a GT mask. Returns uint8 binary (0 or 1)."""
    for ext in ['.jpg', '.png', '.jpeg']:
        p = masks_dir / (stem + ext)
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return (cv2.resize(m, (w, h),
                                   interpolation=cv2.INTER_NEAREST) > 127
                        ).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def _multiplier(n_bin: int) -> int:
    """Effect Size Targeting: min(ceil(N_TARGET / n_bin), REDUNDANCY_CAP)."""
    if n_bin <= 0 or n_bin >= N_TARGET:
        return 1
    return min(math.ceil(N_TARGET / n_bin), REDUNDANCY_CAP)


def _class_weights(bin_counts_after: dict) -> dict:
    """
    Compute per-bin class weights from post-augmentation counts.
    Reference bin is dark (weight = 1.0).
    All others: min(n_dark / n_bin, CLASS_WEIGHT_CAPS[bin]).
    """
    dark_n = max(bin_counts_after.get('dark', 1), 1)
    return {
        b: round(min(dark_n / max(n, 1), CLASS_WEIGHT_CAPS.get(b, 1.0)), 3)
        for b, n in bin_counts_after.items()
        if n > 0
    }


def _sqrt_weights(stem_to_bin: dict) -> dict:
    """
    Per-image sqrt-sampling weights for WeightedRandomSampler.
    w_stem = 1 / sqrt(bin_count), normalised so dark bin = 1.0.
    """
    bin_counts: dict = {}
    for b in stem_to_bin.values():
        bin_counts[b] = bin_counts.get(b, 0) + 1

    dark_w = 1.0 / math.sqrt(max(bin_counts.get('dark', 1), 1))
    return {
        stem: round((1.0 / math.sqrt(max(bin_counts.get(b, 1), 1))) / dark_w, 6)
        for stem, b in stem_to_bin.items()
    }


# ── Main balancing function ───────────────────────────────────────────────────

def balance(dataset_name: str, cfg: dict):
    """Run Effect Size Targeting for one dataset."""
    print(f'\n{"="*60}')
    print(f'Balancing: {dataset_name}')
    print(f'{"="*60}')

    imgs_dir  = cfg['imgs']
    masks_dir = cfg['masks']
    out_root  = BALANCED_DATASET_DIR / dataset_name

    if out_root.exists():
        print(f'  Removing previous output at {out_root}')
        shutil.rmtree(out_root)

    report_rows:        list = []
    train_bin_counts:   dict = {}
    train_stem_to_bin:  dict = {}

    for split in ['train', 'val', 'test']:
        split_img_dir = imgs_dir / split
        if not split_img_dir.exists():
            print(f'  {split}: not found — skipping')
            continue

        img_paths = sorted(
            list(split_img_dir.glob('*.jpg')) +
            list(split_img_dir.glob('*.png'))
        )
        print(f'\n  {split}: {len(img_paths)} images')

        out_img_dir  = out_root / 'images' / split
        out_mask_dir = out_root / 'masks'  / split
        out_img_dir .mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        # ── Compute ITA for every image ───────────────────────────────────────
        ita_data = []
        for img_path in tqdm(img_paths, desc=f'    ITA ({split})', leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                ita_data.append({
                    'stem': img_path.stem, 'path': str(img_path),
                    'ita': None, 'bin': 'unknown', 'h': 0, 'w': 0,
                })
                continue
            h, w = img.shape[:2]
            gt   = _load_mask(img_path.stem, masks_dir, h, w)
            ita  = compute_ita(img, gt)
            bin_ = ita_to_bin(ita)
            ita_data.append({
                'stem': img_path.stem, 'path': str(img_path),
                'ita':  round(ita, 3) if not math.isnan(ita) else None,
                'bin':  bin_, 'h': h, 'w': w,
            })

        ita_df = pd.DataFrame(ita_data)
        counts = ita_df['bin'].value_counts().to_dict()
        print(f'    ITA distribution: {dict(sorted(counts.items()))}')

        # ── Copy originals ────────────────────────────────────────────────────
        for row in ita_data:
            src = Path(row['path'])
            shutil.copy2(src, out_img_dir / src.name)

            saved = False
            for ext in ['.jpg', '.png', '.jpeg']:
                mp = masks_dir / (row['stem'] + ext)
                if mp.exists():
                    shutil.copy2(mp, out_mask_dir / (row['stem'] + '.jpg'))
                    saved = True
                    break
            if not saved and row['h'] > 0:
                cv2.imwrite(
                    str(out_mask_dir / (row['stem'] + '.jpg')),
                    np.zeros((row['h'], row['w']), dtype=np.uint8),
                )

            report_rows.append({
                'dataset':  dataset_name, 'split': split,
                'stem':     row['stem'],  'bin':   row['bin'],
                'ita':      row['ita'],   'augmented': False,
                'aug_type': 'original',
            })

        # ── Augment train split only ──────────────────────────────────────────
        if split != 'train':
            continue

        known_bins = [b for b in counts if b != 'unknown']

        for bin_name in known_bins:
            n_orig    = counts.get(bin_name, 0)
            mult      = _multiplier(n_orig)
            needed    = n_orig * mult - n_orig
            bin_imgs  = ita_df[ita_df['bin'] == bin_name]

            if needed <= 0:
                print(f'    {bin_name:<16} {n_orig:5d}  x1  (no aug needed)')
                train_bin_counts[bin_name] = n_orig
                for row in bin_imgs.itertuples(index=False):
                    train_stem_to_bin[row.stem] = bin_name
                continue

            total_after = n_orig + needed
            print(f'    {bin_name:<16} {n_orig:5d} → {total_after}  '
                  f'x{mult}  (+{needed} copies)')

            pool      = list(bin_imgs.itertuples(index=False))
            aug_count = 0
            idx       = 0

            while aug_count < needed:
                row = pool[idx % len(pool)]
                idx += 1
                img = cv2.imread(row.path)
                if img is None:
                    continue
                h, w     = img.shape[:2]
                mask_raw = _load_mask(row.stem, masks_dir, h, w)
                mask_255 = (mask_raw * 255).astype(np.uint8)

                aug_img, aug_mask, aug_type = augment_pair(
                    img, mask_255, AUGMENTATION_SCALE
                )
                new_stem = f'{row.stem}_aug{aug_count:04d}_{aug_type}'
                cv2.imwrite(str(out_img_dir  / (new_stem + '.jpg')), aug_img)
                cv2.imwrite(str(out_mask_dir / (new_stem + '.jpg')), aug_mask)

                report_rows.append({
                    'dataset':  dataset_name, 'split':  split,
                    'stem':     new_stem,     'bin':    bin_name,
                    'ita':      row.ita,      'augmented': True,
                    'aug_type': aug_type,
                })
                train_stem_to_bin[new_stem] = bin_name
                aug_count += 1

            train_bin_counts[bin_name] = total_after
            for row in bin_imgs.itertuples(index=False):
                train_stem_to_bin[row.stem] = bin_name

        total_train = len(list(out_img_dir.glob('*.jpg')))
        print(f'    Total train images after balancing: {total_train}')

    # ── Class weights (Mechanism 3) ───────────────────────────────────────────
    cw             = _class_weights(train_bin_counts)
    imbalance_ratio = max(cw.values(), default=1.0)

    if imbalance_ratio > 20:
        reg_note  = 'High (>20x) — moderate regularisation'
        dropout, wd = 0.3, 5e-4
    elif imbalance_ratio > 10:
        reg_note  = 'Moderate (10-20x) — light regularisation'
        dropout, wd = 0.2, 1e-4
    else:
        reg_note  = 'Low (<10x) — minimal regularisation'
        dropout, wd = 0.1, 1e-5

    (out_root / 'class_weights.json').write_text(json.dumps({
        'class_weights_by_bin': cw,
        'regularization': {
            'imbalance_ratio': round(imbalance_ratio, 2),
            'dropout':         dropout,
            'weight_decay':    wd,
            'note':            reg_note,
        },
        'usage': (
            'per_img_loss = FocalTverskyLoss()(logits, masks)  # shape (B,). '
            'w = tensor([class_weights_by_bin[bin] for bin in ita_bins]). '
            'loss = (per_img_loss * w).mean(). '
            'See scripts/06_focal_tversky_loss.py.'
        ),
    }, indent=2))

    print(f'\n  CLASS WEIGHTS (capped inverse frequency):')
    for b, w in sorted(cw.items(), key=lambda x: -x[1]):
        print(f'    {b:<16} {w:.1f}x')
    print(f'  REGULARIZATION: {reg_note}')
    print(f'    dropout={dropout}   weight_decay={wd}')

    # ── Sqrt-sampling weights (Mechanism 2) ───────────────────────────────────
    sw_map       = _sqrt_weights(train_stem_to_bin)
    stems_by_bin: dict = {}
    for stem, b in train_stem_to_bin.items():
        stems_by_bin.setdefault(b, []).append(stem)

    (out_root / 'sampler_info.json').write_text(json.dumps({
        'batch_size':          BATCH_SIZE,
        'min_per_batch':       MIN_PER_BATCH,
        'stems_by_bin':        stems_by_bin,
        'sqrt_sample_weights': sw_map,
        'sampling_note': (
            'Use sqrt_sample_weights with WeightedRandomSampler. '
            'Pure balanced (1/n) causes overfitting on minority images. '
            'Sqrt (1/sqrt(n)) is the principled safe compromise. '
            '(Kang et al. ICLR 2020, tau=0.5 optimal.)'
        ),
    }, indent=2))

    # ── Skin tone report ──────────────────────────────────────────────────────
    pd.DataFrame(report_rows).to_csv(
        out_root / 'skin_tone_report.csv', index=False
    )

    # ── dataset.yaml ─────────────────────────────────────────────────────────
    with open(out_root / 'dataset.yaml', 'w') as f:
        yaml.dump({
            'path':  str(out_root),
            'train': 'images/train',
            'val':   'images/val',
            'test':  'images/test',
            'nc':    1,
            'names': ['bruise'],
        }, f, default_flow_style=False)

    print(f'\n  Files written to: {out_root}')
    for fname in ['skin_tone_report.csv', 'class_weights.json',
                  'sampler_info.json', 'dataset.yaml']:
        print(f'    {fname}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Skin tone balancing using ITA and Effect Size Targeting.'
    )
    parser.add_argument(
        '--dataset', default='all',
        choices=list(DATASETS_CFG.keys()) + ['all'],
        help='Dataset to balance. Default: all.',
    )
    args = parser.parse_args()

    datasets = (list(DATASETS_CFG.keys()) if args.dataset == 'all'
                else [args.dataset])
    for ds in datasets:
        balance(ds, DATASETS_CFG[ds])

    print('\n' + '='*60)
    print('DONE. Next steps in your training script:')
    print()
    print('  1. Load class weights:')
    print('       cw = json.load(open("balanced_dataset/<ds>/class_weights.json"))')
    print('       w_map = cw["class_weights_by_bin"]')
    print()
    print('  2. Build WeightedRandomSampler:')
    print('       si = json.load(open("balanced_dataset/<ds>/sampler_info.json"))')
    print('       weights = [si["sqrt_sample_weights"].get(s,1.0) for s in stems]')
    print('       sampler = WeightedRandomSampler(weights, len(weights))')
    print()
    print('  3. Weighted Focal Tversky loss:')
    print('       from scripts.06_focal_tversky_loss import FocalTverskyLoss')
    print('       crit = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)')
    print('       per_img = crit(logits, masks)          # (B,)')
    print('       w = tensor([w_map[b] for b in ita_bins])')
    print('       loss = (per_img * w).mean()')
    print('='*60)


if __name__ == '__main__':
    main()
