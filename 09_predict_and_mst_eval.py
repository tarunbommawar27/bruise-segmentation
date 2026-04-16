"""
scripts/09_predict_and_mst_eval.py
====================================
Monk Skin Tone (MST) fairness evaluation on the fixed 304-image test set.

WHY MST EVALUATION:
  ITA-based training balancing fixes the gradient imbalance but we need to
  verify it actually improved fairness at test time. MST provides 4 levels
  of resolution for dark skin (MST07-10) vs ITA's single dark bin, critical
  for detecting intra-dark performance differences.
  (Desai et al. Research Square 2025 — only paper studying skin tone scale
  choice for bruise detection specifically)

  ITA tells the model HOW to learn fairly.
  MST tells us WHETHER it learned fairly.

FAIRNESS CRITERION:
  delta_fair = Dice_dark - Dice_light
  Target: |delta_fair| <= 0.05
  Positive delta = model performs better on dark skin (majority training class).

MST ASSIGNMENT:
  Nearest-neighbour matching in CIELAB space to 10 published Monk swatch
  reference values. stone library used if available, falls back to manual
  nearest-neighbour matching.
  (Monk, Google Research & Harvard Sociology, 2022)

STEPS
-----
  --step label
    Assigns ITA bin and MST level to every test image.
    Reads ITA from balanced_dataset/<ds>/skin_tone_report.csv.
    Computes full LAB from each image directly (lab_nn method).
    Saves test_labels.csv. CPU only, ~10 minutes. Run once.

  --step evaluate --model <name>
    Runs YOLO inference on 304 test images, saves prediction masks,
    computes Dice per ITA bin and MST group, reports fairness gap.
    GPU required, ~10 minutes per model.

  --step compare
    Prints side-by-side fairness table across all evaluated models.
    Reads saved JSON results. CPU only.

  --step all
    Runs label -> evaluate all models -> compare.

USAGE
-----
    python scripts/09_predict_and_mst_eval.py --step label
    python scripts/09_predict_and_mst_eval.py --step evaluate --model majority_yolo26m
    python scripts/09_predict_and_mst_eval.py --step evaluate --model all
    python scripts/09_predict_and_mst_eval.py --step compare
    python scripts/09_predict_and_mst_eval.py --step all

OUTPUTS
-------
    mst_fairness_results/
        test_labels.csv                     ITA + MST per test image
        <model>_mst_fairness.json           per-model fairness summary
        <model>_per_image.csv               per-image Dice + skin tone
        fairness_comparison_all_models.json all models side by side
    prediction_masks/<model>/
        <stem>.png                          binary prediction masks (0 or 255)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── Repo imports ──────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    YOLO_DATASETS_DIR,
    MAJORITY_MASKS_DIR,
    GBARIMAH_MASK_DIR,
    PAUL_MASK_DIR,
    SARAH_MASK_DIR,
    BALANCED_DATASET_DIR,
    RUNS_DIR,
    THREE_WAY_DIR,
    YOLO_MODELS,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_GPU_DEVICE,
    SIZE_PERCENTILE_CUTOFF,
)
from src.utils import pixel_metrics, load_gt, masks_from_result

# ── Output directories ────────────────────────────────────────────────────────
MST_RESULTS_DIR  = THREE_WAY_DIR / 'mst_fairness_results'
PRED_MASKS_DIR   = THREE_WAY_DIR / 'prediction_masks'
MST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_MASKS_DIR .mkdir(parents=True, exist_ok=True)

# ── Monk Skin Tone reference swatches in CIELAB ───────────────────────────────
# Source: Monk (2022), Google Research & Harvard Sociology.
# 10 reference swatch LAB values from lightest (MST01) to darkest (MST10).
MONK_LAB = {
    'MST01': (87.0,  5.0, 14.0),
    'MST02': (79.0,  7.0, 17.0),
    'MST03': (70.0, 10.0, 20.0),
    'MST04': (61.0, 13.0, 22.0),
    'MST05': (52.0, 16.0, 22.0),
    'MST06': (42.0, 16.0, 19.0),
    'MST07': (33.0, 14.0, 14.0),
    'MST08': (24.0, 10.0,  9.0),
    'MST09': (16.0,  7.0,  4.0),
    'MST10': ( 9.0,  3.0,  2.0),
}

MST_GROUPS = {
    'Light':  ['MST01', 'MST02', 'MST03'],
    'Medium': ['MST04', 'MST05', 'MST06'],
    'Dark':   ['MST07', 'MST08', 'MST09', 'MST10'],
}

# ── Per-model optimal confidence (found by sweep in 01_train_yolo.py) ─────────
# Uses YOLO_CONF from config as default; override here for individual models.
MODEL_CONF = {
    'majority_yolo26m':        0.10,
    'majority_yolo26l':        0.10,
    'majority_yolo26n':        0.10,
    'majority_yolo26s':        0.10,
    'majority_yolo26l_best_lr':0.10,
    'gbarimah_yolo26l':        0.05,
    'sarah_yolo26l':           0.15,
    'paul_yolo26l':            0.10,
}

# Fixed 304-image test set — majority split used for all evaluation
TEST_IMG_DIR = YOLO_DATASETS_DIR / 'majority' / 'images' / 'test'


# ── Skin tone helpers ─────────────────────────────────────────────────────────

def _assign_mst(L: float, a: float, b: float) -> str:
    """Nearest-neighbour MST assignment in CIELAB space."""
    best, best_d = 'MST05', float('inf')
    for level, (Lr, ar, br) in MONK_LAB.items():
        d = (L - Lr)**2 + (a - ar)**2 + (b - br)**2
        if d < best_d:
            best_d, best = d, level
    return best


def _ita_to_mst_fallback(ita: float) -> str:
    """Estimate MST from ITA when image cannot be loaded."""
    if math.isnan(ita): return 'MST05'
    if ita >  55: return 'MST01'
    if ita >  41: return 'MST02'
    if ita >  28: return 'MST03'
    if ita >  15: return 'MST04'
    if ita >   3: return 'MST05'
    if ita > -10: return 'MST06'
    if ita > -30: return 'MST07'
    if ita > -45: return 'MST08'
    if ita > -55: return 'MST09'
    return 'MST10'


def _mst_group(level: str) -> str:
    for g, levels in MST_GROUPS.items():
        if level in levels:
            return g
    return 'Unknown'


def _compute_skin_lab(img_bgr: np.ndarray, gt_bin: np.ndarray):
    """
    Compute median LAB on non-bruise skin pixels.
    Returns (L, a, b) or None if not enough skin pixels.
    """
    try:
        h, w = img_bgr.shape[:2]
        if h * w > 2_000_000:
            s = (2_000_000 / (h * w)) ** 0.5
            img_bgr = cv2.resize(img_bgr, (int(w*s), int(h*s)))
            gt_bin  = cv2.resize(gt_bin,  (int(w*s), int(h*s)),
                                 interpolation=cv2.INTER_NEAREST)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        L   = lab[:,:,0] * (100.0 / 255.0)
        a   = lab[:,:,1] - 128.0
        b   = lab[:,:,2] - 128.0
        skin = (gt_bin == 0)
        if skin.sum() < 500:
            return None
        return float(np.median(L[skin])), float(np.median(a[skin])), \
               float(np.median(b[skin]))
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — LABEL
# ═════════════════════════════════════════════════════════════════════════════

def step_label():
    """
    Assign ITA bin + MST level to all 304 test images.
    ITA is read from skin_tone_report.csv; MST is computed from actual
    LAB values extracted from each image (lab_nn method).
    """
    cache_path = MST_RESULTS_DIR / 'test_labels.csv'
    if cache_path.exists():
        print(f'  Labels already exist at {cache_path}')
        print(f'  Delete that file and rerun to regenerate.')
        return pd.read_csv(cache_path)

    print('\n' + '='*60)
    print('STEP 1 — Label test images with ITA and MST')
    print('='*60)

    # Load ITA values from balancing report (gbarimah or majority, same test set)
    ita_lookup: dict = {}
    for ds in ['gbarimah', 'majority']:
        report = BALANCED_DATASET_DIR / ds / 'skin_tone_report.csv'
        if report.exists():
            df_r = pd.read_csv(report)
            test_rows = df_r[(df_r['split'] == 'test') &
                             (df_r['augmented'] == False)]
            for _, row in test_rows.iterrows():
                ita_lookup[row['stem']] = {
                    'ita': row['ita'], 'ita_bin': row['bin']
                }
            print(f'  Loaded ITA for {len(test_rows)} test images from {ds} report')
            break

    # Collect test stems
    test_stems = sorted([
        p.stem for p in TEST_IMG_DIR.glob('*')
        if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}
    ])
    print(f'  Test images found: {len(test_stems)}')

    records = []
    for stem in test_stems:
        ita_info = ita_lookup.get(stem, {})
        ita      = float(ita_info.get('ita',     float('nan')) or float('nan'))
        ita_bin  = str  (ita_info.get('ita_bin', 'unknown'))

        img_path = TEST_IMG_DIR / (stem + '.jpg')
        if not img_path.exists():
            img_path = TEST_IMG_DIR / (stem + '.png')

        mst_level  = None
        mst_method = 'fallback'
        L_val = a_val = b_val = None

        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w    = img.shape[:2]
                gt_bin  = load_gt(stem, h, w, 'majority')
                lab     = _compute_skin_lab(img, gt_bin)
                if lab is not None:
                    L_val, a_val, b_val = lab
                    mst_level  = _assign_mst(L_val, a_val, b_val)
                    mst_method = 'lab_nn'

        if mst_level is None:
            mst_level  = _ita_to_mst_fallback(ita)
            mst_method = 'ita_estimate'

        records.append({
            'stem':       stem,
            'ita':        round(ita, 3) if not math.isnan(ita) else None,
            'ita_bin':    ita_bin,
            'L':          round(L_val, 3) if L_val is not None else None,
            'a_star':     round(a_val, 3) if a_val is not None else None,
            'b_star':     round(b_val, 3) if b_val is not None else None,
            'mst_level':  mst_level,
            'mst_group':  _mst_group(mst_level),
            'mst_method': mst_method,
        })

    df_labels = pd.DataFrame(records)
    df_labels.to_csv(cache_path, index=False)

    # Print distribution
    print(f'\n  Labeling method:')
    for m, n in df_labels['mst_method'].value_counts().items():
        print(f'    {m:<16} {n}')

    print(f'\n  ITA bin distribution (test set):')
    for b in ['very_light','light','intermediate','tan','dark','very_dark','unknown']:
        n = (df_labels['ita_bin'] == b).sum()
        if n > 0:
            print(f'    {b:<16} {n:4d}  {"█"*n}')

    print(f'\n  MST level distribution:')
    for lvl in [f'MST{i:02d}' for i in range(1, 11)]:
        n = (df_labels['mst_level'] == lvl).sum()
        if n > 0:
            print(f'    {lvl}  {n:4d}  {"█"*n}')

    print(f'\n  MST group distribution:')
    for grp in ['Light', 'Medium', 'Dark']:
        n   = (df_labels['mst_group'] == grp).sum()
        pct = 100.0 * n / max(len(df_labels), 1)
        print(f'    {grp:<10} {n:4d}  ({pct:.1f}%)')

    print(f'\n  Saved: {cache_path}')
    return df_labels


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — EVALUATE
# ═════════════════════════════════════════════════════════════════════════════

def step_evaluate(model_name: str, df_labels: pd.DataFrame):
    """
    Run YOLO inference on 304 test images, save prediction masks,
    compute Dice per ITA bin and per MST group, report fairness gap.
    """
    from ultralytics import YOLO

    weights_path = RUNS_DIR / model_name / 'weights' / 'best.pt'
    if not weights_path.exists():
        print(f'  SKIP {model_name} — weights not found at {weights_path}')
        return None

    conf     = MODEL_CONF.get(model_name, YOLO_CONF)
    pred_dir = PRED_MASKS_DIR / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Evaluating: {model_name}')
    print(f'{"="*60}')
    print(f'  Weights : {weights_path}')
    print(f'  Conf    : {conf}')
    print(f'  GT      : majority vote masks')

    # ── Inference ─────────────────────────────────────────────────────────────
    test_images = sorted(
        list(TEST_IMG_DIR.glob('*.jpg')) +
        list(TEST_IMG_DIR.glob('*.png'))
    )

    existing = set(p.stem for p in pred_dir.glob('*.png'))
    if len(existing) >= len(test_images) * 0.95:
        print(f'  Prediction masks already exist ({len(existing)} files) — skipping inference.')
    else:
        print(f'  Running inference on {len(test_images)} images...')
        model = YOLO(str(weights_path))
        for img_path in test_images:
            if img_path.stem in existing:
                continue
            results  = model.predict(
                source  = str(img_path),
                conf    = conf,
                iou     = YOLO_IOU,
                device  = YOLO_GPU_DEVICE,
                save    = False,
                verbose = False,
            )
            result   = results[0]
            h, w     = result.orig_shape
            mask_out = masks_from_result(result, h, w)
            cv2.imwrite(str(pred_dir / f'{img_path.stem}.png'), mask_out)
        print(f'  Saved {len(list(pred_dir.glob("*.png")))} prediction masks to {pred_dir}')

    # ── Compute metrics ────────────────────────────────────────────────────────
    rows = []
    for _, label in df_labels.iterrows():
        stem      = label['stem']
        pred_path = pred_dir / f'{stem}.png'
        if not pred_path.exists():
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue

        h, w      = pred.shape
        pred_bin  = (pred > 127).astype(np.uint8) * 255
        gt        = load_gt(stem, h, w, 'majority') * 255
        bruise_px = int((gt > 127).sum())

        dice, iou, prec, rec, _ = pixel_metrics(pred_bin, gt)

        rows.append({
            'stem':      stem,
            'ita':       label['ita'],
            'ita_bin':   label['ita_bin'],
            'mst_level': label['mst_level'],
            'mst_group': label['mst_group'],
            'dice':      round(dice, 4),
            'iou':       round(iou,  4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'bruise_px': bruise_px,
            'pct_bruise':round(100.0 * bruise_px / max(h * w, 1), 3),
        })

    if not rows:
        print('  No predictions found. Cannot evaluate.')
        return None

    df_eval  = pd.DataFrame(rows)
    cutoff   = float(np.percentile(df_eval['pct_bruise'], SIZE_PERCENTILE_CUTOFF))
    df_eval['size'] = df_eval['pct_bruise'].apply(
        lambda x: 'small' if x <= cutoff else 'large'
    )

    overall    = round(df_eval['dice'].mean(), 4)
    small_dice = round(df_eval[df_eval['size'] == 'small']['dice'].mean(), 4)
    large_dice = round(df_eval[df_eval['size'] == 'large']['dice'].mean(), 4)

    print(f'\n  Overall Dice : {overall:.4f}')
    print(f'  Small Dice   : {small_dice:.4f}')
    print(f'  Large Dice   : {large_dice:.4f}')

    # ── Per ITA bin ────────────────────────────────────────────────────────────
    print(f'\n  {"─"*50}')
    print(f'  Dice by ITA bin:')
    ita_res = {}
    for b in ['very_light','light','intermediate','tan','dark','very_dark']:
        sub = df_eval[df_eval['ita_bin'] == b]
        if len(sub) == 0:
            continue
        d   = round(sub['dice'].mean(), 4)
        s   = round(sub['dice'].std(),  4) if len(sub) > 1 else 0.0
        bar = '█' * int(d * 25)
        print(f'    {b:<16} {d:.4f} ±{s:.4f}  {bar}  n={len(sub)}')
        ita_res[b] = {'mean': d, 'std': s, 'n': int(len(sub))}

    # ── Per MST level ──────────────────────────────────────────────────────────
    print(f'\n  {"─"*50}')
    print(f'  Dice by MST level:')
    mst_res = {}
    for lvl in [f'MST{i:02d}' for i in range(1, 11)]:
        sub = df_eval[df_eval['mst_level'] == lvl]
        if len(sub) == 0:
            continue
        d   = round(sub['dice'].mean(), 4)
        s   = round(sub['dice'].std(),  4) if len(sub) > 1 else 0.0
        grp = _mst_group(lvl)
        bar = '█' * int(d * 25)
        print(f'    {lvl} ({grp:<7}) {d:.4f} ±{s:.4f}  {bar}  n={len(sub)}')
        mst_res[lvl] = {'mean': d, 'std': s, 'n': int(len(sub)), 'group': grp}

    # ── Per MST group + fairness gap ───────────────────────────────────────────
    print(f'\n  {"─"*50}')
    print(f'  Dice by MST group:')
    grp_res    = {}
    delta_fair = None

    for grp in ['Light', 'Medium', 'Dark']:
        sub = df_eval[df_eval['mst_group'] == grp]
        if len(sub) == 0:
            grp_res[grp] = {'mean': None, 'n': 0}
            print(f'    {grp:<10} No test images')
            continue
        d   = round(sub['dice'].mean(), 4)
        s   = round(sub['dice'].std(),  4) if len(sub) > 1 else 0.0
        bar = '█' * int(d * 25)
        print(f'    {grp:<10} {d:.4f} ±{s:.4f}  {bar}  n={len(sub)}')
        grp_res[grp] = {'mean': d, 'std': s, 'n': int(len(sub))}

    dark_d  = grp_res.get('Dark',  {}).get('mean')
    light_d = grp_res.get('Light', {}).get('mean')

    print(f'\n  {"─"*50}')
    if dark_d is not None and light_d is not None:
        delta_fair = round(dark_d - light_d, 4)
        fair       = abs(delta_fair) <= 0.05
        print(f'  Fairness gap (delta_fair = Dice_dark - Dice_light):')
        print(f'    Dark Dice   : {dark_d:.4f}')
        print(f'    Light Dice  : {light_d:.4f}')
        print(f'    delta_fair  : {delta_fair:+.4f}')
        print(f'    Target      : |delta| <= 0.05')
        print(f'    Status      : {"PASS" if fair else "FAIL"}')
        if delta_fair > 0:
            print(f'    Meaning     : Model is better on dark skin (majority class)')
        else:
            print(f'    Meaning     : Model is better on light skin')
    else:
        print('  Fairness gap: cannot compute (Light or Dark group is empty)')

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        'model':         model_name,
        'gt_type':       'majority',
        'n_evaluated':   len(df_eval),
        'overall_dice':  overall,
        'small_dice':    small_dice,
        'large_dice':    large_dice,
        'dice_by_ita_bin':   ita_res,
        'dice_by_mst_level': mst_res,
        'dice_by_mst_group': grp_res,
        'delta_fair':    delta_fair,
        'fairness_pass': bool(abs(delta_fair) <= 0.05) if delta_fair is not None else None,
    }

    json_out = MST_RESULTS_DIR / f'{model_name}_mst_fairness.json'
    csv_out  = MST_RESULTS_DIR / f'{model_name}_per_image.csv'
    json_out.write_text(json.dumps(summary, indent=2))
    df_eval.to_csv(csv_out, index=False)
    print(f'\n  Saved: {json_out}')
    print(f'  Saved: {csv_out}')
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — COMPARE
# ═════════════════════════════════════════════════════════════════════════════

def step_compare():
    """Print side-by-side fairness comparison for all evaluated models."""
    results = {}
    for p in sorted(MST_RESULTS_DIR.glob('*_mst_fairness.json')):
        name = p.stem.replace('_mst_fairness', '')
        results[name] = json.loads(p.read_text())

    if not results:
        print('No results found. Run --step evaluate first.')
        return

    print('\n' + '='*90)
    print('FAIRNESS COMPARISON — All Models')
    print('='*90)
    print(f'{"Model":<35} {"Overall":>8} {"Light":>8} {"Medium":>8} '
          f'{"Dark":>8} {"Delta":>8} {"Status":>6}')
    print('-'*90)

    def fmt(v):
        if isinstance(v, float):
            return f'{v:+.4f}' if v < 0 else f'{v:.4f}'
        return 'N/A'

    for name, d in sorted(results.items(),
                          key=lambda x: -(x[1].get('overall_dice') or 0)):
        grps = d.get('dice_by_mst_group', {})
        l    = (grps.get('Light',  {}) or {}).get('mean')
        m    = (grps.get('Medium', {}) or {}).get('mean')
        dk   = (grps.get('Dark',   {}) or {}).get('mean')
        delta  = d.get('delta_fair')
        status = ('PASS' if d.get('fairness_pass') else 'FAIL') \
                 if d.get('fairness_pass') is not None else 'N/A'
        print(f'{name:<35} {fmt(d.get("overall_dice")):>8} {fmt(l):>8} '
              f'{fmt(m):>8} {fmt(dk):>8} {fmt(delta):>8} {status:>6}')

    out = MST_RESULTS_DIR / 'fairness_comparison_all_models.json'
    out.write_text(json.dumps(results, indent=2))
    print(f'\nSaved full comparison: {out}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Monk Skin Tone fairness evaluation on 304-image test set.'
    )
    parser.add_argument(
        '--step', required=True,
        choices=['label', 'evaluate', 'compare', 'all'],
    )
    parser.add_argument(
        '--model', default='majority_yolo26m',
        help='Model name or "all". Used with --step evaluate.',
    )
    args = parser.parse_args()

    df_labels = None

    if args.step in ('label', 'all'):
        df_labels = step_label()

    if args.step in ('evaluate', 'all'):
        if df_labels is None:
            df_labels = step_label()
        models = YOLO_MODELS if args.model == 'all' else [args.model]
        for m in models:
            try:
                step_evaluate(m, df_labels)
            except Exception as e:
                print(f'  ERROR evaluating {m}: {e}')

    if args.step in ('compare', 'all'):
        step_compare()

    print('\nDone.')


if __name__ == '__main__':
    main()
