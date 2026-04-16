"""
scripts/10_train_balanced_yolo.py
====================================
Retrain YOLO on ITA-balanced datasets and evaluate fairness improvement.

WHY THIS SCRIPT EXISTS:
  Script 05 balanced the training data using Effect Size Targeting.
  This script trains a fresh YOLO model on that balanced data and
  immediately evaluates it on the fixed 304-image test set, comparing
  against the unbalanced baseline and reporting MST fairness gap.

  The key question: did the three-mechanism balancing strategy
  (geometric aug + sqrt sampler + Focal Tversky loss) actually close
  the fairness gap reported in 09_predict_and_mst_eval.py?

WHAT CHANGES COMPARED TO 01_train_yolo.py:
  - Dataset points to balanced_dataset/<ds>/ not yolo_datasets/<ds>/
  - Run name is <ds>_balanced_<size> to distinguish from unbalanced
  - Evaluation runs immediately after training using the MST labels
    from 09_predict_and_mst_eval.py

WHAT STAYS THE SAME:
  - All hyperparameters (epochs, batch, imgsz, patience, seed)
  - Model weights (yolo26l-seg.pt etc.)
  - Evaluation test set (fixed 304-image majority test set)
  - GT masks (majority vote)
  Note: class-weighted Focal Tversky loss and sqrt sampler are
  PyTorch-level mechanisms applied when training a custom model.
  For YOLO's built-in trainer we rely on the augmentation mechanism
  (Mechanism 1) which is already baked into the balanced dataset.

USAGE
-----
    python scripts/10_train_balanced_yolo.py --dataset gbarimah --size l
    python scripts/10_train_balanced_yolo.py --dataset majority  --size l
    python scripts/10_train_balanced_yolo.py --dataset paul      --size l
    python scripts/10_train_balanced_yolo.py --dataset sarah     --size l
    python scripts/10_train_balanced_yolo.py --dataset all       --size l

    # Evaluate only (skip training, use existing weights)
    python scripts/10_train_balanced_yolo.py --dataset gbarimah --size l --eval_only

OUTPUTS
-------
    runs/<ds>_balanced_yolo26<size>/   YOLO training artifacts + weights
    balanced_eval_results/
        <model>_results.json           overall + MST fairness results
        <model>_per_image.csv          per-image Dice + skin tone
    prediction_masks/<model>/
        <stem>.png                     binary prediction masks (0 or 255)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── Repo imports ──────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    BALANCED_DATASET_DIR,
    RUNS_DIR,
    THREE_WAY_DIR,
    YOLO_DATASETS_DIR,
    MAJORITY_MASKS_DIR,
    YOLO_IMGSZ,
    YOLO_EPOCHS,
    YOLO_PATIENCE,
    YOLO_BATCH,
    YOLO_GPU_DEVICE,
    YOLO_IOU,
    SEED,
    SIZE_PERCENTILE_CUTOFF,
)
from src.utils import pixel_metrics, load_gt, masks_from_result

# ── Output directories ────────────────────────────────────────────────────────
EVAL_RESULTS_DIR = THREE_WAY_DIR / 'balanced_eval_results'
PRED_MASKS_DIR   = THREE_WAY_DIR / 'prediction_masks'
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_MASKS_DIR  .mkdir(parents=True, exist_ok=True)

# ── Model size -> weights file ────────────────────────────────────────────────
# Weights are in the home directory — edit USERNAME in configs/config.py
from configs.config import BASE
_HOME = BASE.parent
BASE_WEIGHTS = {
    'n': str(_HOME / 'yolo26n-seg.pt'),
    's': str(_HOME / 'yolo26s-seg.pt'),
    'm': str(_HOME / 'yolo26m-seg.pt'),
    'l': str(_HOME / 'yolo26l-seg.pt'),
    'x': str(_HOME / 'yolo26x-seg.pt'),
}

# ── Optimal conf per dataset (from sweep in 01_train_yolo.py) ─────────────────
DATASET_CONF = {
    'gbarimah': 0.05,
    'majority':  0.10,
    'paul':      0.10,
    'sarah':     0.15,
}

# ── Datasets to train ─────────────────────────────────────────────────────────
DATASETS = ['majority', 'gbarimah', 'paul', 'sarah']

# ── Unbalanced baseline Dice for comparison ───────────────────────────────────
BASELINES = {
    'gbarimah': {
        'model': 'gbarimah_yolo26l',
        'overall_dice': 0.7653, 'small_dice': 0.6940, 'large_dice': 0.8002,
        'delta_fair': 0.1519,
    },
    'majority': {
        'model': 'majority_yolo26m',
        'overall_dice': 0.7974, 'small_dice': 0.7555, 'large_dice': 0.8179,
        'delta_fair': 0.0602,
    },
    'paul': {
        'model': 'paul_yolo26l',
        'overall_dice': 0.7313, 'small_dice': 0.7163, 'large_dice': 0.7386,
        'delta_fair': 0.0368,
    },
    'sarah': {
        'model': 'sarah_yolo26l',
        'overall_dice': 0.7428, 'small_dice': 0.7070, 'large_dice': 0.7603,
        'delta_fair': 0.0418,
    },
}

# Fixed test set
TEST_IMG_DIR = YOLO_DATASETS_DIR / 'majority' / 'images' / 'test'


# ── MST helpers ───────────────────────────────────────────────────────────────

def _load_mst_labels() -> pd.DataFrame | None:
    """Load MST test labels from 09_predict_and_mst_eval.py output."""
    p = THREE_WAY_DIR / 'mst_fairness_results' / 'test_labels.csv'
    if not p.exists():
        return None
    return pd.read_csv(p)


# ── Training ──────────────────────────────────────────────────────────────────

def train(dataset: str, size: str) -> Path | None:
    """Train YOLO on the ITA-balanced dataset."""
    from ultralytics import YOLO

    yaml_path = BALANCED_DATASET_DIR / dataset / 'dataset.yaml'
    if not yaml_path.exists():
        print(f'  SKIP {dataset} — balanced dataset not found at {yaml_path}')
        print(f'  Run: python scripts/05_skin_tone_balancing.py --dataset {dataset}')
        return None

    run_name = f'{dataset}_balanced_yolo26{size}'
    out_dir  = RUNS_DIR / run_name

    if (out_dir / 'weights' / 'best.pt').exists():
        print(f'  SKIP training {run_name} — already trained')
        return out_dir

    train_n = len(list((BALANCED_DATASET_DIR / dataset / 'images' / 'train').glob('*')))
    val_n   = len(list((BALANCED_DATASET_DIR / dataset / 'images' / 'val').glob('*')))

    print(f'\n{"="*60}')
    print(f'Training: {run_name}')
    print(f'{"="*60}')
    print(f'  Balanced dataset : {yaml_path}')
    print(f'  Train images     : {train_n}')
    print(f'  Val images       : {val_n}')
    print(f'  Model weights    : {BASE_WEIGHTS[size]}')
    print(f'  Epochs           : {YOLO_EPOCHS}  Patience: {YOLO_PATIENCE}')
    print(f'  Batch            : {YOLO_BATCH}  ImgSz: {YOLO_IMGSZ}  GPU: {YOLO_GPU_DEVICE}')

    # Load and print class weights for reference
    cw_path = BALANCED_DATASET_DIR / dataset / 'class_weights.json'
    if cw_path.exists():
        cw = json.loads(cw_path.read_text())
        print(f'\n  Class weights (from balancing):')
        for b, w in sorted(cw.get('class_weights_by_bin', {}).items(),
                           key=lambda x: -x[1]):
            print(f'    {b:<16} {w:.1f}x')

    model = YOLO(BASE_WEIGHTS[size])
    model.train(
        data     = str(yaml_path),
        epochs   = YOLO_EPOCHS,
        patience = YOLO_PATIENCE,
        batch    = YOLO_BATCH,
        imgsz    = YOLO_IMGSZ,
        device   = YOLO_GPU_DEVICE,
        seed     = SEED,
        project  = str(RUNS_DIR),
        name     = run_name,
        exist_ok = True,
        task     = 'segment',
        save     = True,
        plots    = True,
        val      = True,
    )

    print(f'\n  Weights saved to: {out_dir}/weights/best.pt')
    return out_dir


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(dataset: str, size: str, run_out: Path):
    """
    Evaluate the balanced model on the fixed 304-image test set.
    Reports overall Dice, small/large Dice, MST fairness gap,
    and comparison vs unbalanced baseline.
    """
    from ultralytics import YOLO

    run_name     = f'{dataset}_balanced_yolo26{size}'
    weights_path = run_out / 'weights' / 'best.pt'

    if not weights_path.exists():
        print(f'  ERROR: weights not found at {weights_path}')
        return

    conf     = DATASET_CONF.get(dataset, 0.10)
    pred_dir = PRED_MASKS_DIR / run_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Evaluating: {run_name}')
    print(f'{"="*60}')
    print(f'  GT source : majority vote masks')
    print(f'  Conf      : {conf}')

    # ── Inference ─────────────────────────────────────────────────────────────
    test_images = sorted(
        list(TEST_IMG_DIR.glob('*.jpg')) +
        list(TEST_IMG_DIR.glob('*.png'))
    )
    existing = set(p.stem for p in pred_dir.glob('*.png'))
    if len(existing) >= len(test_images) * 0.95:
        print(f'  Prediction masks already exist ({len(existing)} files).')
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
        print(f'  Saved {len(list(pred_dir.glob("*.png")))} prediction masks.')

    # ── Metrics ───────────────────────────────────────────────────────────────
    rows = []
    for img_path in test_images:
        stem      = img_path.stem
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
            'dice':      round(dice, 4),
            'iou':       round(iou,  4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'bruise_px': bruise_px,
            'pct_bruise':round(100.0 * bruise_px / max(h * w, 1), 3),
        })

    df = pd.DataFrame(rows)
    cutoff   = float(np.percentile(df['pct_bruise'], SIZE_PERCENTILE_CUTOFF))
    df['size'] = df['pct_bruise'].apply(
        lambda x: 'small' if x <= cutoff else 'large'
    )

    overall    = round(df['dice'].mean(), 4)
    small_dice = round(df[df['size'] == 'small']['dice'].mean(), 4)
    large_dice = round(df[df['size'] == 'large']['dice'].mean(), 4)

    print(f'\n  Overall Dice : {overall:.4f}')
    print(f'  Small Dice   : {small_dice:.4f}')
    print(f'  Large Dice   : {large_dice:.4f}')

    # ── MST fairness breakdown ─────────────────────────────────────────────────
    df_mst     = _load_mst_labels()
    delta_fair = None
    grp_res    = {}

    if df_mst is not None:
        df = df.merge(
            df_mst[['stem', 'ita_bin', 'mst_level', 'mst_group']],
            on='stem', how='left',
        )
        print(f'\n  Dice by MST group:')
        for grp in ['Light', 'Medium', 'Dark']:
            sub = df[df['mst_group'] == grp]
            if len(sub) == 0:
                grp_res[grp] = {'mean': None, 'n': 0}
                print(f'    {grp:<10} No test images')
                continue
            d   = round(sub['dice'].mean(), 4)
            s   = round(sub['dice'].std(),  4) if len(sub) > 1 else 0.0
            bar = '█' * int(d * 25)
            print(f'    {grp:<10} {d:.4f} ±{s:.4f}  {bar}  n={len(sub)}')
            grp_res[grp] = {'mean': d, 'std': s, 'n': int(len(sub))}

        dark_d  = (grp_res.get('Dark',  {}) or {}).get('mean')
        light_d = (grp_res.get('Light', {}) or {}).get('mean')
        if dark_d is not None and light_d is not None:
            delta_fair = round(dark_d - light_d, 4)
            fair       = abs(delta_fair) <= 0.05
            print(f'\n  Fairness gap : {delta_fair:+.4f}  '
                  f'({"PASS" if fair else "FAIL"} — target |delta| <= 0.05)')
    else:
        print('\n  MST labels not found. Run: '
              'python scripts/09_predict_and_mst_eval.py --step label')

    # ── Comparison vs unbalanced baseline ─────────────────────────────────────
    baseline = BASELINES.get(dataset)
    if baseline:
        print(f'\n  {"─"*50}')
        print(f'  vs UNBALANCED BASELINE ({baseline["model"]}):')
        print(f'  {"─"*50}')
        print(f'  {"Metric":<20} {"Baseline":>10} {"Balanced":>10} {"Delta":>10}')
        metrics = [
            ('Overall Dice',   overall,    baseline['overall_dice']),
            ('Small Dice',     small_dice, baseline['small_dice']),
            ('Large Dice',     large_dice, baseline['large_dice']),
        ]
        for label, bal_val, base_val in metrics:
            delta = bal_val - base_val
            print(f'  {label:<20} {base_val:>10.4f} {bal_val:>10.4f} {delta:>+10.4f}')
        if delta_fair is not None:
            d_fair = delta_fair - baseline['delta_fair']
            print(f'  {"Fairness gap":<20} {baseline["delta_fair"]:>+10.4f} '
                  f'{delta_fair:>+10.4f} {d_fair:>+10.4f}')

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        'model':         run_name,
        'dataset':       dataset,
        'balanced':      True,
        'overall_dice':  overall,
        'small_dice':    small_dice,
        'large_dice':    large_dice,
        'delta_fair':    delta_fair,
        'fairness_pass': bool(abs(delta_fair) <= 0.05) if delta_fair is not None else None,
        'mst_groups':    grp_res,
        'baseline':      baseline,
    }
    json_out = EVAL_RESULTS_DIR / f'{run_name}_results.json'
    csv_out  = EVAL_RESULTS_DIR / f'{run_name}_per_image.csv'
    json_out.write_text(json.dumps(summary, indent=2))
    df.to_csv(csv_out, index=False)
    print(f'\n  Saved: {json_out}')
    print(f'  Saved: {csv_out}')
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO on ITA-balanced dataset and evaluate fairness.'
    )
    parser.add_argument(
        '--dataset', default='gbarimah',
        choices=DATASETS + ['all'],
    )
    parser.add_argument(
        '--size', default='l',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO model size. Default: l (large).',
    )
    parser.add_argument(
        '--eval_only', action='store_true',
        help='Skip training, evaluate existing weights only.',
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        run_name = f'{ds}_balanced_yolo26{args.size}'
        run_out  = RUNS_DIR / run_name

        if args.eval_only:
            print(f'\nEval-only mode for {run_name}')
            evaluate(ds, args.size, run_out)
        else:
            out = train(ds, args.size)
            if out is not None:
                evaluate(ds, args.size, out)

    print('\nAll done.')


if __name__ == '__main__':
    main()
