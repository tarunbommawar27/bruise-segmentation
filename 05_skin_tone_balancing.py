"""
scripts/05_skin_tone_balancing.py
===================================
Balance training datasets by skin tone using ITA (Individual Typology Angle).

ITA = arctan((L* - 50) / b*) × 180 / π
Computed on NON-bruise skin pixels only so bruise discolouration
does not affect the skin tone reading.

Augmentation: geometric only (flips, rotations, zoom).
No colour changes ever — skin tone must remain in the same ITA bin.

Val and test sets are NEVER augmented — only original images.

Usage:
    # Balance both datasets
    python scripts/05_skin_tone_balancing.py

    # Balance a specific dataset
    python scripts/05_skin_tone_balancing.py --dataset gbarimah
    python scripts/05_skin_tone_balancing.py --dataset majority

Output:
    balanced_dataset/
        {dataset}/
            images/train/   ← originals + augmented copies
            images/val/     ← originals only
            images/test/    ← originals only
            masks/train/
            masks/val/
            masks/test/
            dataset.yaml
            skin_tone_report.csv
"""

import argparse
import shutil
import random
import math
import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    YOLO_DATASETS_DIR, MAJORITY_MASKS_DIR,
    PAUL_MASK_DIR, SARAH_MASK_DIR, GBARIMAH_MASK_DIR,
    BALANCED_DATASET_DIR, SEED, AUGMENTATION_SCALE,
)
from src.utils import compute_ita, ita_to_bin, augment_pair

random.seed(SEED)
np.random.seed(SEED)

DATASETS_CFG = {
    'majority': {
        'imgs':  YOLO_DATASETS_DIR / 'majority' / 'images',
        'masks': MAJORITY_MASKS_DIR,
    },
    'gbarimah': {
        'imgs':  YOLO_DATASETS_DIR / 'gbarimah' / 'images',
        'masks': GBARIMAH_MASK_DIR,
    },
}


def load_mask(stem: str, masks_dir: Path, h: int, w: int) -> np.ndarray:
    for ext in ['.jpg', '.png', '.jpeg']:
        p = masks_dir / (stem + ext)
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                return (m > 127).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


def balance(dataset_name: str, cfg: dict):
    print(f'\n{"="*60}')
    print(f'Balancing: {dataset_name}')
    print(f'{"="*60}')

    imgs_dir  = cfg['imgs']
    masks_dir = cfg['masks']
    out_root  = BALANCED_DATASET_DIR / dataset_name

    # Clean up any incomplete previous run
    if out_root.exists():
        print(f'Removing previous output: {out_root}')
        shutil.rmtree(out_root)

    report_rows = []

    for split in ['train', 'val', 'test']:
        split_img_dir = imgs_dir / split
        if not split_img_dir.exists():
            print(f'  {split}: not found, skipping')
            continue

        img_paths = sorted(split_img_dir.glob('*.jpg'))
        print(f'\n  {split}: {len(img_paths)} images')

        out_img_dir  = out_root / 'images' / split
        out_mask_dir = out_root / 'masks'  / split
        out_img_dir .mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        # Compute ITA for all images
        ita_data = []
        for img_path in tqdm(img_paths, desc=f'    ITA ({split})', leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                ita_data.append({'stem': img_path.stem, 'path': str(img_path),
                                 'ita': None, 'bin': 'unknown', 'h': 0, 'w': 0})
                continue
            h, w  = img.shape[:2]
            gt    = load_mask(img_path.stem, masks_dir, h, w)
            ita   = compute_ita(img, gt)
            bin_  = ita_to_bin(ita)
            ita_data.append({
                'stem': img_path.stem, 'path': str(img_path),
                'ita':  round(ita, 2) if not math.isnan(ita) else None,
                'bin':  bin_, 'h': h, 'w': w,
            })

        ita_df = pd.DataFrame(ita_data)
        counts = ita_df['bin'].value_counts().to_dict()
        print(f'    ITA before: {dict(sorted(counts.items()))}')

        # Copy originals
        for row in ita_data:
            src = Path(row['path'])
            shutil.copy2(src, out_img_dir / src.name)
            mask_saved = False
            for ext in ['.jpg', '.png', '.jpeg']:
                mp = masks_dir / (row['stem'] + ext)
                if mp.exists():
                    shutil.copy2(mp, out_mask_dir / (row['stem'] + '.jpg'))
                    mask_saved = True
                    break
            if not mask_saved and row['h'] > 0:
                cv2.imwrite(str(out_mask_dir / (row['stem'] + '.jpg')),
                            np.zeros((row['h'], row['w']), dtype=np.uint8))

        for row in ita_data:
            report_rows.append({
                'dataset': dataset_name, 'split': split,
                'stem': row['stem'], 'bin': row['bin'],
                'ita': row['ita'], 'augmented': False, 'aug_type': 'original',
            })

        # Augment training split only
        if split != 'train':
            continue

        known_bins   = [b for b in counts if b != 'unknown']
        if not known_bins:
            continue
        target_count = max(counts.get(b, 0) for b in known_bins)
        print(f'    Target count per bin: {target_count}')

        for bin_name in known_bins:
            bin_imgs  = ita_df[ita_df['bin'] == bin_name]
            current_n = len(bin_imgs)
            needed    = target_count - current_n
            if needed <= 0:
                print(f'    {bin_name:<16} {current_n} — no augmentation needed')
                continue
            print(f'    {bin_name:<16} {current_n} → +{needed} copies')

            pool      = list(bin_imgs.itertuples(index=False))
            aug_count = 0
            idx       = 0
            while aug_count < needed:
                row = pool[idx % len(pool)]; idx += 1
                img = cv2.imread(row.path)
                if img is None: continue
                h, w     = img.shape[:2]
                mask_raw = load_mask(row.stem, masks_dir, h, w)
                mask_255 = (mask_raw * 255).astype(np.uint8)

                aug_img, aug_mask, aug_type = augment_pair(
                    img, mask_255, AUGMENTATION_SCALE
                )
                new_stem = f'{row.stem}_aug{aug_count:04d}_{aug_type}'
                cv2.imwrite(str(out_img_dir  / (new_stem + '.jpg')), aug_img)
                cv2.imwrite(str(out_mask_dir / (new_stem + '.jpg')), aug_mask)
                report_rows.append({
                    'dataset': dataset_name, 'split': split,
                    'stem': new_stem, 'bin': bin_name, 'ita': row.ita,
                    'augmented': True, 'aug_type': aug_type,
                })
                aug_count += 1

        total_train = len(list(out_img_dir.glob('*.jpg')))
        print(f'    Total train images: {total_train}')

    # Save report and yaml
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(out_root / 'skin_tone_report.csv', index=False)

    with open(out_root / 'dataset.yaml', 'w') as f:
        yaml.dump({
            'path':  str(out_root),
            'train': 'images/train',
            'val':   'images/val',
            'test':  'images/test',
            'nc': 1,
            'names': ['bruise'],
        }, f, default_flow_style=False)

    print(f'\n  Saved: {out_root}')


def main():
    parser = argparse.ArgumentParser(description='Skin tone balancing using ITA')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=list(DATASETS_CFG.keys()) + ['all'])
    args = parser.parse_args()

    datasets = list(DATASETS_CFG.keys()) if args.dataset == 'all' \
               else [args.dataset]

    for ds in datasets:
        balance(ds, DATASETS_CFG[ds])

    print('\nAll balancing complete.')


if __name__ == '__main__':
    main()
