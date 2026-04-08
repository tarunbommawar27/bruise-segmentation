"""
scripts/02_generate_probability_maps.py
========================================
Run each YOLO model once on both test sets and save raw float32
probability maps as .npy files.

WHY WE DO THIS:
  Without this step, Script 03 (threshold analysis) has to re-run YOLO
  inference separately for each of the 9 threshold methods — that is
  9 methods x 8 models x 304 images = 21,888 GPU inference runs taking
  19+ hours. By saving maps once, Script 03 reads files in under 10 min.

WHY conf=0.05 (PROB_MAP_CONF):
  At the default conf=0.25, YOLO discards detections it is less than 25%
  confident about. Bruise edge pixels often score only 0.08-0.12. We lower
  conf to 0.05 so faint edge detections are kept in the saved map.
  NOTE: this does NOT change pixel values inside detections — those come
  from model weights. It only controls which detection regions are reported.

Usage:
    python scripts/02_generate_probability_maps.py

Output:
    probability_maps/
        majority_test/yolo/{model_name}/{stem}.npy   <- float32 (H,W) 0.0-1.0
        gbarimah_test/yolo/{model_name}/{stem}.npy
        majority_test/image_list.txt                 <- stems in sorted order
        gbarimah_test/image_list.txt
        metadata.json                                <- config used to generate
"""

import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    YOLO_DATASETS_DIR, RUNS_DIR, PROBABILITY_MAPS_DIR,
    YOLO_MODELS, YOLO_IMGSZ, YOLO_GPU_DEVICE, PROB_MAP_CONF,
)
from src.utils import prob_map_from_result

# ── Test sets: name -> directory of test images ────────────────────────────────
# Both test sets contain 304 images each — the shared majority test set
# and the gbarimah-specific test set.
TEST_SETS = {
    'majority_test': YOLO_DATASETS_DIR / 'majority' / 'images' / 'test',
    'gbarimah_test': YOLO_DATASETS_DIR / 'gbarimah' / 'images' / 'test',
}

# Create output root if it does not exist yet
PROBABILITY_MAPS_DIR.mkdir(parents=True, exist_ok=True)

print(f'Saving probability maps to: {PROBABILITY_MAPS_DIR}')
print(f'conf = {PROB_MAP_CONF}  (kept low to preserve soft edge values)\n')

# metadata.json tracks what was generated — useful for debugging
metadata = {'models': YOLO_MODELS, 'conf': PROB_MAP_CONF, 'test_sets': {}}

for test_name, test_dir in TEST_SETS.items():

    # Collect test images in sorted order for reproducibility
    test_paths = sorted(test_dir.glob('*.jpg'))
    if not test_paths:
        print(f'WARNING: no images found in {test_dir}')
        print(f'  Check that Script 01 completed and datasets were built.')
        continue

    # Save the ordered stem list so Scripts 03 and 04 always read images
    # in the same order — critical for matching maps to GT masks
    stems = [p.stem for p in test_paths]
    stems_out = PROBABILITY_MAPS_DIR / test_name / 'image_list.txt'
    stems_out.parent.mkdir(parents=True, exist_ok=True)
    stems_out.write_text('\n'.join(stems))

    metadata['test_sets'][test_name] = {
        'dir': str(test_dir),
        'n_images': len(test_paths),
    }

    print(f'Test set: {test_name}  ({len(test_paths)} images)')

    for model_name in YOLO_MODELS:
        # Each model gets its own subdirectory of .npy files
        weights = RUNS_DIR / model_name / 'weights' / 'best.pt'
        if not weights.exists():
            print(f'  SKIP {model_name} — weights not found at {weights}')
            print(f'  Run Script 01 to train this model first.')
            continue

        out_dir = PROBABILITY_MAPS_DIR / test_name / 'yolo' / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip models already fully processed — safe to re-run this script
        done = set(p.stem for p in out_dir.glob('*.npy'))
        if len(done) == len(test_paths):
            print(f'  SKIP {model_name} — all {len(done)} maps already exist')
            continue

        # Load model once per model — not once per image (expensive)
        model = YOLO(str(weights))

        for img_path in tqdm(test_paths, desc=f'  {model_name}', leave=False):
            out_path = out_dir / (img_path.stem + '.npy')

            # Skip individual images already saved — resume-safe after crash
            if out_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f'  WARNING: could not read {img_path.name}')
                continue
            h, w = img.shape[:2]

            # Run YOLO at very low conf to keep all soft probability values.
            # verbose=False suppresses per-image output noise in the terminal.
            results = model.predict(
                source  = str(img_path),
                imgsz   = YOLO_IMGSZ,
                conf    = PROB_MAP_CONF,  # 0.05 — keep faint edge detections
                device  = YOLO_GPU_DEVICE,
                verbose = False,
            )

            # Extract the raw sigmoid probability map (float32, 0.0 to 1.0).
            # Takes element-wise max across all overlapping detections so
            # multiple detected regions contribute their highest confidence.
            prob = prob_map_from_result(results[0], h, w)

            # Save as numpy binary — much faster to load than images
            np.save(str(out_path), prob)

        print(f'  {model_name} done  '
              f'({len(list(out_dir.glob("*.npy")))} maps saved)')

# Save metadata so we can always reproduce exactly what generated these maps
(PROBABILITY_MAPS_DIR / 'metadata.json').write_text(
    json.dumps(metadata, indent=2)
)
print(f'\nMetadata: {PROBABILITY_MAPS_DIR}/metadata.json')
print('Done. Now run Script 03 (threshold analysis) — no GPU needed.')
