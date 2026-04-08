"""
scripts/04_temperature_scaling.py
===================================
Apply temperature scaling to saved .npy probability maps and re-run
threshold analysis on the scaled maps.

WHY TEMPERATURE SCALING:
  YOLO outputs near-binary probability maps — almost all pixel values are
  either 0.001 (skin) or 0.999 (bruise) with nothing in between. This
  happens because YOLO's training objective pushes logits to large values
  (+-5 to +-8) and sigmoid crushes them to extremes. As a result all 8
  auto-threshold methods give identical Dice (Script 03 finding).

  Temperature scaling fixes this without retraining:
    1. Convert probability p back to logit: logit = log(p / (1-p))
    2. Divide logit by temperature T: scaled_logit = logit / T
    3. Convert back: result = sigmoid(scaled_logit)

  T=1  -> no change (original)
  T=2  -> moderate spread
  T=5  -> strong spread — threshold methods start diverging
  T=10 -> very soft — Dice may start dropping
  T=20 -> too soft

CRITICAL: Original .npy maps are NEVER modified.
  Scaled maps are written to a completely separate directory.
  Re-running this script is always safe.

Usage:
    python scripts/04_temperature_scaling.py

Output:
    temperature_scaling_results/
        scaled_maps/T{n}/{test_set}/yolo/{model}/{stem}.npy  <- scaled copies
        threshold_results/T{n}_{test_set}_threshold_methods.csv
        dice_vs_temperature.csv   <- pivot: model x T, best Dice at each T
        best_per_model.csv        <- best T and method per model
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    PROBABILITY_MAPS_DIR, TEMPERATURE_SCALING_DIR,
    YOLO_MODELS, THRESHOLD_METHODS, TEMPERATURES,
)
from src.utils import apply_temperature, binarize, load_gt, pixel_metrics

TEMPERATURE_SCALING_DIR.mkdir(parents=True, exist_ok=True)
(TEMPERATURE_SCALING_DIR / 'scaled_maps').mkdir(exist_ok=True)
(TEMPERATURE_SCALING_DIR / 'threshold_results').mkdir(exist_ok=True)

# ── Ground truth: use majority vote GT for both test sets (option B) ───────────
# This matches Script 03's option_b — fair apples-to-apples comparison
GT_OPTIONS = {
    'majority_test': 'majority',
    'gbarimah_test': 'majority',
}

print(f'Temperature values to test: {TEMPERATURES}')
print(f'Original maps (never modified): {PROBABILITY_MAPS_DIR}')
print(f'Scaled maps saved to         : {TEMPERATURE_SCALING_DIR}\n')

# Collect all results — one row per (T, test_set, model, method)
all_results = []


def get_stems(test_name: str) -> list:
    """Return the ordered list of image stems for a test set."""
    img_list = PROBABILITY_MAPS_DIR / test_name / 'image_list.txt'
    if img_list.exists():
        return img_list.read_text().strip().split('\n')
    # Fallback if image_list.txt was not saved by Script 02
    for m in YOLO_MODELS:
        d = PROBABILITY_MAPS_DIR / test_name / 'yolo' / m
        if d.exists():
            return sorted(p.stem for p in d.glob('*.npy'))
    return []


def get_map_dir(T: int, test_name: str, model_name: str) -> Path:
    """
    Return the directory containing probability maps at temperature T.
    T=1 means no scaling — use the original maps directly.
    T>1 means use the scaled copies in a separate subdirectory.
    """
    if T == 1:
        # Original maps — read-only, never written to
        return PROBABILITY_MAPS_DIR / test_name / 'yolo' / model_name
    # Scaled copies live in a completely separate directory tree
    return (TEMPERATURE_SCALING_DIR / 'scaled_maps' /
            f'T{T}' / test_name / 'yolo' / model_name)


for T in TEMPERATURES:
    print(f'\n{"="*60}')
    print(f'Temperature T = {T}{"  (original, no scaling)" if T == 1 else ""}')
    print(f'{"="*60}')

    for test_name, gt_type in GT_OPTIONS.items():
        stems = get_stems(test_name)
        if not stems:
            print(f'  SKIP {test_name} — run Script 02 first')
            continue

        print(f'\n  {test_name}  ({len(stems)} images)')

        for model_name in YOLO_MODELS:
            orig_dir   = PROBABILITY_MAPS_DIR / test_name / 'yolo' / model_name
            scaled_dir = get_map_dir(T, test_name, model_name)
            scaled_dir.mkdir(parents=True, exist_ok=True)

            if not orig_dir.exists():
                continue

            # ── Step 1: Apply temperature scaling and cache results ────────────
            # For T=1 we just load originals directly — no scaling needed.
            # For T>1 we load originals, apply scaling, save to separate dir.
            cache = {}
            for stem in stems:
                orig_path   = orig_dir   / (stem + '.npy')
                scaled_path = scaled_dir / (stem + '.npy')

                if not orig_path.exists():
                    continue

                orig = np.load(str(orig_path))

                if T > 1:
                    if not scaled_path.exists():
                        # Apply temperature and save to separate directory
                        # Original file is never touched
                        scaled = apply_temperature(orig, T)
                        np.save(str(scaled_path), scaled)
                    else:
                        # Already scaled from a previous run — just load it
                        scaled = np.load(str(scaled_path))
                else:
                    # T=1: use original as-is
                    scaled = orig

                cache[stem] = (scaled, scaled.shape[0], scaled.shape[1])

            # ── Step 2: Run all 9 threshold methods on the scaled maps ─────────
            for method in THRESHOLD_METHODS:
                rows = []
                for stem, (prob, h, w) in cache.items():
                    gt       = load_gt(stem, h, w, gt_type)
                    pred, tv = binarize(prob, method)
                    d, i, p_, r, s = pixel_metrics(pred, gt)
                    rows.append({
                        'temperature': T,
                        'test_set':    test_name,
                        'model':       model_name,
                        'method':      method,
                        'image':       stem,
                        'gt_type':     gt_type,
                        'dice':        round(d,  4),
                        'iou':         round(i,  4),
                        'precision':   round(p_, 4),
                        'recall':      round(r,  4),
                        'threshold':   round(tv, 4),
                    })

                df = pd.DataFrame(rows)
                all_results.append({
                    'temperature':  T,
                    'test_set':     test_name,
                    'model':        model_name,
                    'method':       method,
                    'gt_type':      gt_type,
                    'n':            len(df),
                    'dice_mean':    round(df.dice.mean(),      4),
                    'dice_std':     round(df.dice.std(),       4),
                    'iou_mean':     round(df.iou.mean(),       4),
                    'prec_mean':    round(df.precision.mean(), 4),
                    'recall_mean':  round(df.recall.mean(),    4),
                    'mean_thresh':  round(df.threshold.mean(), 4),
                })

            # Print best method for this model at this temperature
            model_res = [r for r in all_results
                         if r['temperature'] == T and
                            r['test_set']    == test_name and
                            r['model']       == model_name]
            best = max(model_res, key=lambda x: x['dice_mean'])
            print(f'    {model_name:<30} T={T}  '
                  f'best={best["method"]:<14} Dice={best["dice_mean"]:.4f}')

# ── Save per-temperature CSVs ──────────────────────────────────────────────────
all_df = pd.DataFrame(all_results)
tr_dir = TEMPERATURE_SCALING_DIR / 'threshold_results'

for T in TEMPERATURES:
    for test_name in GT_OPTIONS:
        sub = all_df[(all_df.temperature == T) & (all_df.test_set == test_name)]
        if len(sub):
            sub.to_csv(tr_dir / f'T{T}_{test_name}_threshold_methods.csv',
                       index=False)

# ── Pivot table: model x temperature -> best Dice ─────────────────────────────
# Restrict to majority_test for the summary so results are comparable
maj = all_df[all_df.test_set == 'majority_test']
best_per_T = maj.loc[maj.groupby(['temperature', 'model'])['dice_mean'].idxmax()]
pivot = best_per_T.pivot_table(
    index='model', columns='temperature', values='dice_mean'
)
# Rename columns for clarity: 1 -> T=1, 5 -> T=5, etc.
pivot.columns = [f'T={c}' for c in pivot.columns]
pivot.to_csv(TEMPERATURE_SCALING_DIR / 'dice_vs_temperature.csv')

# Best overall T and method per model
best_overall = maj.loc[maj.groupby('model')['dice_mean'].idxmax()]
best_overall.to_csv(TEMPERATURE_SCALING_DIR / 'best_per_model.csv', index=False)

# ── Print summary tables ───────────────────────────────────────────────────────
print('\n' + '='*60)
print('DICE vs TEMPERATURE (majority_test, best method at each T)')
print('='*60)
print(pivot.round(4).to_string())

# Check at which temperature threshold methods start diverging
# Divergence = max Dice - min Dice across non-balanced methods > 0.001
print('\n' + '='*60)
print('DIVERGENCE — did threshold methods spread at higher T?')
print('  (spread > 0.001 means temperature scaling is working)')
print('='*60)
for T in TEMPERATURES:
    sub = maj[maj.temperature == T]
    diverged = []
    for model in YOLO_MODELS:
        m = sub[(sub.model == model) & (sub.method != 'balanced')]
        if len(m) == 0:
            continue
        spread = m['dice_mean'].max() - m['dice_mean'].min()
        if spread > 0.001:
            diverged.append(f'{model} (spread={spread:.4f})')
    status = f'DIVERGED for {len(diverged)} models' if diverged else 'identical for all models'
    print(f'  T={T}: {status}')
    for d in diverged:
        print(f'    {d}')

print(f'\nAll results saved to: {TEMPERATURE_SCALING_DIR}')
print('Done.')
