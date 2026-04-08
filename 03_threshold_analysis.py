"""
scripts/03_threshold_analysis.py
=================================
Apply all 9 threshold methods to the saved .npy probability maps.
No GPU needed. Reads from probability_maps/, writes CSVs to
threshold_analysis_results/.

WHY WE TEST 9 METHODS:
  The standard approach is to use a fixed threshold (e.g. 0.50). But
  automatic methods like Otsu, Triangle, and Li adapt to each image's
  histogram. We test all to find if any method beats fixed thresholds.

KEY FINDING:
  All 8 non-balanced methods give identical Dice scores for YOLO models.
  Root cause: YOLO's sigmoid activation on large logits (+-5 to +-8)
  produces near-binary probability maps. Any threshold cuts through the
  same empty gap and produces the same binary mask.

TWO GROUND TRUTH OPTIONS:
  Option A — annotator-specific GT
    majority_test -> majority vote masks
    gbarimah_test -> raw Gbarimah annotator masks
  Option B — majority vote GT for both (fair apples-to-apples comparison)
  Running both lets us measure how much GT choice affects Dice scores.

IMPORTANT: Run Script 02 first. If .npy maps are missing this script
  will fail. Do not run threshold analysis before probability maps exist.

Usage:
    python scripts/03_threshold_analysis.py

Output:
    threshold_analysis_results/
        option_a/
            majority_yolo_threshold_methods.csv    <- mean Dice per model+method
            majority_yolo_threshold_per_image.csv  <- Dice for every image
            gbarimah_yolo_threshold_methods.csv
            gbarimah_yolo_threshold_per_image.csv
            summary_best_methods.csv               <- best method per model
        option_b/  (same structure)
        comparison_a_vs_b.csv                      <- option A vs B side by side
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    PROBABILITY_MAPS_DIR, THRESHOLD_RESULTS_DIR,
    YOLO_MODELS, THRESHOLD_METHODS,
)
from src.utils import binarize, load_gt, pixel_metrics

THRESHOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Ground truth options ───────────────────────────────────────────────────────
# Option A: use each annotator's own masks as GT for their test set
# Option B: use majority vote GT for both — this is the fair comparison
GT_OPTIONS = {
    'option_a': {
        'majority_test': 'majority',   # majority test -> majority vote GT
        'gbarimah_test': 'gbarimah',   # gbarimah test -> raw gbarimah GT
    },
    'option_b': {
        'majority_test': 'majority',   # majority test -> majority vote GT
        'gbarimah_test': 'majority',   # gbarimah test -> majority vote GT (fair)
    },
}


def run_option(gt_option: str, gt_cfg: dict):
    """Run threshold analysis for one GT option across all test sets and models."""
    print(f'\n{"="*60}')
    print(f'Ground truth option: {gt_option.upper()}')
    print(f'{"="*60}')

    out_dir = THRESHOLD_RESULTS_DIR / gt_option
    out_dir.mkdir(parents=True, exist_ok=True)

    for test_name, gt_type in gt_cfg.items():
        prob_base = PROBABILITY_MAPS_DIR / test_name / 'yolo'
        if not prob_base.exists():
            print(f'SKIP {test_name} — probability maps not found.')
            print(f'  Run Script 02 first to generate .npy files.')
            continue

        # Read the ordered image list saved by Script 02
        # This ensures we always process images in the same order
        img_list = PROBABILITY_MAPS_DIR / test_name / 'image_list.txt'
        if img_list.exists():
            stems = img_list.read_text().strip().split('\n')
        else:
            # Fallback: collect stems from the first available model directory
            stems = sorted([
                p.stem for p in (prob_base / YOLO_MODELS[0]).glob('*.npy')
            ])

        print(f'\n  {test_name}  ({len(stems)} images)  GT={gt_type}')

        # agg_rows: one row per model+method (mean Dice across all images)
        # per_rows: one row per model+method+image (individual Dice scores)
        agg_rows = []
        per_rows = []

        for model_name in YOLO_MODELS:
            model_dir = prob_base / model_name
            if not model_dir.exists():
                print(f'    SKIP {model_name} — maps not found')
                continue

            # Load all probability maps for this model into memory once.
            # Avoids repeated disk reads when iterating over 9 methods.
            # cache: {stem: (prob_array, height, width)}
            cache = {}
            for stem in stems:
                p = model_dir / (stem + '.npy')
                if p.exists():
                    prob = np.load(str(p))
                    cache[stem] = (prob, prob.shape[0], prob.shape[1])

            # Apply each of the 9 threshold methods to every cached map
            for method in THRESHOLD_METHODS:
                rows = []
                for stem, (prob, h, w) in cache.items():
                    # Load the ground truth mask for this image
                    gt = load_gt(stem, h, w, gt_type)

                    # Apply threshold method -> binary mask + threshold value used
                    pred, tv = binarize(prob, method)

                    # Compute pixel-level metrics against GT mask
                    d, i, p_, r, s = pixel_metrics(pred, gt)

                    rows.append({
                        'model':       model_name,
                        'method':      method,
                        'image':       stem,
                        'gt_option':   gt_option,
                        'dice':        round(d,  4),
                        'iou':         round(i,  4),
                        'precision':   round(p_, 4),
                        'recall':      round(r,  4),
                        'specificity': round(s,  4),
                        'threshold':   round(tv, 4),
                    })
                per_rows.extend(rows)

                # Aggregate: mean metrics across all 304 images for this method
                df = pd.DataFrame(rows)
                agg_rows.append({
                    'model':        model_name,
                    'method':       method,
                    'gt_option':    gt_option,
                    'n':            len(df),
                    'dice_mean':    round(df.dice.mean(),      4),
                    'dice_std':     round(df.dice.std(),       4),
                    'iou_mean':     round(df.iou.mean(),       4),
                    'prec_mean':    round(df.precision.mean(), 4),
                    'recall_mean':  round(df.recall.mean(),    4),
                    'mean_thresh':  round(df.threshold.mean(), 4),
                })

            # Print the best method for this model in this option
            model_agg = [r for r in agg_rows if r['model'] == model_name]
            best = max(model_agg, key=lambda x: x['dice_mean'])
            print(f'    {model_name:<30} best={best["method"]:<14} '
                  f'Dice={best["dice_mean"]:.4f}')

        # Save aggregate and per-image CSVs for this test set and GT option
        prefix = 'majority_yolo' if 'majority' in test_name else 'gbarimah_yolo'
        pd.DataFrame(agg_rows).to_csv(
            out_dir / f'{prefix}_threshold_methods.csv', index=False)
        pd.DataFrame(per_rows).to_csv(
            out_dir / f'{prefix}_threshold_per_image.csv', index=False)

        # Summary: one row per model showing the single best method
        agg_df  = pd.DataFrame(agg_rows)
        best_df = agg_df.loc[agg_df.groupby('model')['dice_mean'].idxmax()]
        best_df.to_csv(out_dir / 'summary_best_methods.csv', index=False)

        print(f'  Saved -> {out_dir}')


# ── Run both GT options ────────────────────────────────────────────────────────
for gt_option, gt_cfg in GT_OPTIONS.items():
    run_option(gt_option, gt_cfg)

# ── Build comparison table: option A Dice vs option B Dice per model ──────────
print('\nBuilding comparison table (option A vs option B)...')
rows = []
for gt_option in GT_OPTIONS:
    for csv_name in ['majority_yolo_threshold_methods.csv',
                     'gbarimah_yolo_threshold_methods.csv']:
        path = THRESHOLD_RESULTS_DIR / gt_option / csv_name
        if not path.exists():
            continue
        df       = pd.read_csv(path)
        # Take only the best method per model for the comparison
        best     = df.loc[df.groupby('model')['dice_mean'].idxmax()]
        test_set = 'majority_test' if 'majority' in csv_name else 'gbarimah_test'
        for _, r in best.iterrows():
            rows.append({
                'model':       r['model'],
                'test_set':    test_set,
                'gt_option':   gt_option,
                'dice_mean':   r['dice_mean'],
                'best_method': r['method'],
            })

# Pivot so option_a and option_b Dice are side by side for easy comparison
comp  = pd.DataFrame(rows)
pivot = comp.pivot_table(
    index=['model', 'test_set'], columns='gt_option', values='dice_mean'
).reset_index()
pivot.columns.name = None

# Compute difference: positive = option_b higher = majority GT is more favourable
if 'option_a' in pivot.columns and 'option_b' in pivot.columns:
    pivot['diff'] = (pivot['option_b'] - pivot['option_a']).round(4)

pivot.to_csv(THRESHOLD_RESULTS_DIR / 'comparison_a_vs_b.csv', index=False)
print(pivot.to_string(index=False))
print(f'\nSaved: {THRESHOLD_RESULTS_DIR}/comparison_a_vs_b.csv')
print('Done.')
