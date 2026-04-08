"""
src/utils.py
=============
Shared utility functions used across all scripts.
Import from here instead of copy-pasting into each script.

Usage:
    from src.utils import pixel_metrics, load_gt, binarize, apply_temperature
"""

import cv2
import math
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import (
    MAJORITY_MASKS_DIR, GBARIMAH_MASK_DIR,
    ITA_BINS, MAX_IMAGE_PIXELS_ITA,
)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def pixel_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-8
) -> Tuple[float, float, float, float, float]:
    """
    Compute pixel-level segmentation metrics from binary masks.

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask (uint8, values 0 or 255, or bool).
    gt : np.ndarray
        Ground truth binary mask (uint8, values 0 or 255, or bool).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    (dice, iou, precision, recall, specificity) — all floats in [0, 1]
    """
    p = (pred > 127).astype(bool) if pred.dtype != bool else pred
    g = (gt   > 127).astype(bool) if gt.dtype   != bool else gt

    TP = np.logical_and( p,  g).sum()
    FP = np.logical_and( p, ~g).sum()
    FN = np.logical_and(~p,  g).sum()
    TN = np.logical_and(~p, ~g).sum()

    # Both empty — perfect score
    if p.sum() == 0 and g.sum() == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0

    dice        = float((2 * TP) / (2 * TP + FP + FN + eps))
    iou         = float(TP       / (TP + FP + FN + eps))
    precision   = float(TP       / (TP + FP + eps))
    recall      = float(TP       / (TP + FN + eps))
    specificity = float(TN       / (TN + FP + eps))

    return dice, iou, precision, recall, specificity


def get_bruise_area(mask: np.ndarray) -> int:
    """Return number of bruise pixels (foreground) in a mask."""
    return int((mask > 127).sum())


# ══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_gt(
    stem: str,
    h: int,
    w: int,
    gt_type: str = 'majority'
) -> np.ndarray:
    """
    Load a ground truth mask for a given image stem.

    Parameters
    ----------
    stem : str
        Image filename without extension.
    h, w : int
        Target height and width to resize mask to.
    gt_type : str
        'majority' → majority vote masks
        'gbarimah' → raw Gbarimah annotator masks

    Returns
    -------
    np.ndarray  uint8 (H, W), values 0 or 1
    """
    masks_dir = MAJORITY_MASKS_DIR if gt_type == 'majority' else GBARIMAH_MASK_DIR
    for ext in ['.jpg', '.png', '.jpeg']:
        path = masks_dir / (stem + ext)
        if path.exists():
            gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
                return (gt > 127).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD METHODS  (all corrected formulas)
# ══════════════════════════════════════════════════════════════════════════════

def binarize(
    prob: np.ndarray,
    method: str
) -> Tuple[np.ndarray, float]:
    """
    Apply a threshold method to a float32 probability map.

    Parameters
    ----------
    prob : np.ndarray
        Float32 probability map, values in [0, 1].
    method : str
        One of: fixed_0.50, fixed_0.30, fixed_0.10, otsu, triangle,
                li, kapur, mcet, balanced

    Returns
    -------
    (binary_mask, threshold_value)
        binary_mask : uint8 (H, W), values 0 or 1
        threshold_value : float in [0, 1]
    """
    img8 = (prob * 255).astype(np.uint8)
    if img8.max() == 0:
        return np.zeros_like(img8), 0.0

    # ── Fixed thresholds ───────────────────────────────────────────────────────
    if method == 'fixed_0.50':
        return (img8 >= 127).astype(np.uint8), 0.50
    if method == 'fixed_0.30':
        return (img8 >= 76).astype(np.uint8),  0.30
    if method == 'fixed_0.10':
        return (img8 >= 25).astype(np.uint8),  0.10

    # ── Otsu ──────────────────────────────────────────────────────────────────
    if method == 'otsu':
        t, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (binary > 0).astype(np.uint8), float(t) / 255.0

    # ── Triangle ──────────────────────────────────────────────────────────────
    if method == 'triangle':
        t, binary = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return (binary > 0).astype(np.uint8), float(t) / 255.0

    # ── Li (cross entropy minimisation, iterative) ────────────────────────────
    if method == 'li':
        try:
            from skimage.filters import threshold_li
            tf = float(np.clip(threshold_li(prob), 0.0, 1.0))
            return (prob >= tf).astype(np.uint8), tf
        except Exception:
            return (img8 >= 127).astype(np.uint8), 0.50

    # ── Balanced histogram ────────────────────────────────────────────────────
    if method == 'balanced':
        hist, _ = np.histogram(img8.ravel(), bins=256, range=(0, 256))
        hist     = hist.astype(np.float64)
        best_t   = 0
        for t in range(1, 255):
            if hist[:t+1].sum() >= hist[t+1:].sum():
                best_t = t
                break
        return (img8 >= best_t).astype(np.uint8), float(best_t) / 255.0

    # ── Kapur (corrected — includes log(P_t) term) ────────────────────────────
    if method == 'kapur':
        hist, _ = np.histogram(img8.ravel(), bins=256, range=(0, 256))
        hist     = hist.astype(np.float64)
        total    = hist.sum()
        if total == 0:
            return np.zeros_like(img8), 0.0
        p       = hist / total
        eps     = 1e-12
        plogp   = np.where(p > eps, p * np.log(p + eps), 0.0)
        cum_p   = np.cumsum(p)
        cum_H   = np.cumsum(plogp)
        total_H = cum_H[-1]
        best_t, best_H = 0, -np.inf
        for t in range(1, 255):
            P_t = cum_p[t];   Q_t = 1.0 - P_t
            if P_t < eps or Q_t < eps:
                continue
            H_t  = cum_H[t]
            H_bg = -(H_t / P_t) + np.log(P_t)          # corrected formula
            H_fg = -((total_H - H_t) / Q_t) + np.log(Q_t)
            if H_bg + H_fg > best_H:
                best_H = H_bg + H_fg
                best_t = t
        return (img8 >= best_t).astype(np.uint8), float(best_t) / 255.0

    # ── MCET (Li & Lee 1993 — histogram-based, not pixel-wise) ───────────────
    if method == 'mcet':
        hist, _     = np.histogram(img8.ravel(), bins=256, range=(0, 256))
        hist         = hist.astype(np.float64)
        intensities  = np.arange(256, dtype=np.float64)
        eps          = 1e-12
        best_t, best_ce = 0, np.inf
        for t in range(1, 255):
            w_bg = hist[:t+1].sum();  w_fg = hist[t+1:].sum()
            if w_bg < eps or w_fg < eps:
                continue
            mu_bg = np.sum(intensities[:t+1] * hist[:t+1]) / w_bg
            mu_fg = np.sum(intensities[t+1:] * hist[t+1:]) / w_fg
            if mu_bg <= 0 or mu_fg <= 0:
                continue
            ce = (
                -np.sum(intensities[:t+1] * hist[:t+1] * np.log(mu_bg + eps))
                -np.sum(intensities[t+1:] * hist[t+1:] * np.log(mu_fg + eps))
            )
            if ce < best_ce:
                best_ce = ce
                best_t  = t
        return (img8 >= best_t).astype(np.uint8), float(best_t) / 255.0

    raise ValueError(f'Unknown threshold method: {method!r}')


# ══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE SCALING
# ══════════════════════════════════════════════════════════════════════════════

def apply_temperature(prob: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling to a float32 probability map.

    Divides the logit (inverse sigmoid) by T before re-applying sigmoid.
    T=1 → no change. T>1 → softer (values spread toward 0.5).

    The original array is never modified.

    Parameters
    ----------
    prob : np.ndarray
        Float32 probability map, values in (0, 1).
    T : float
        Temperature. Must be > 0.

    Returns
    -------
    np.ndarray  float32, same shape as input.
    """
    if T == 1:
        return prob.copy()
    eps    = 1e-6
    p      = np.clip(prob, eps, 1.0 - eps)
    logit  = np.log(p / (1.0 - p))
    scaled = logit / T
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ITA — SKIN TONE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ita(img_bgr: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Individual Typology Angle (ITA) for the non-bruise skin region.

    ITA = arctan((L* - 50) / b*) × (180 / π)

    Computed on skin pixels only (gt_mask == 0). Images larger than
    MAX_IMAGE_PIXELS_ITA are resized before Lab conversion to prevent
    hanging on large files (e.g. TAM079 at 4022×6024).

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR image.
    gt_mask : np.ndarray
        Binary mask (0 = skin, 1 = bruise).

    Returns
    -------
    float  ITA angle in degrees, or nan if not computable.
    """
    try:
        h, w = img_bgr.shape[:2]

        # Size guard — resize images above pixel threshold
        if h * w > MAX_IMAGE_PIXELS_ITA:
            scale   = (MAX_IMAGE_PIXELS_ITA / (h * w)) ** 0.5
            new_h   = int(h * scale)
            new_w   = int(w * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            gt_mask = cv2.resize(gt_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        L   = lab[:, :, 0] * (100.0 / 255.0)   # scale to [0, 100]
        b   = lab[:, :, 2] - 128.0              # centre b* around 0

        skin_mask = (gt_mask == 0)
        if skin_mask.sum() < 500:
            return float('nan')

        L_mean = float(L[skin_mask].mean())
        b_mean = float(b[skin_mask].mean())

        if abs(b_mean) < 1e-6:
            return float('nan')

        return math.atan2(L_mean - 50.0, b_mean) * (180.0 / math.pi)

    except Exception:
        return float('nan')


def ita_to_bin(ita: float) -> str:
    """Map an ITA value to a skin tone bin name."""
    if math.isnan(ita):
        return 'unknown'
    for name, lo, hi in ITA_BINS:
        if lo < ita <= hi:
            return name
    return 'unknown'


# ══════════════════════════════════════════════════════════════════════════════
# YOLO MASK EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def masks_from_result(result, h: int, w: int) -> np.ndarray:
    """
    Extract a binary segmentation mask from a YOLO result object.

    Handles both square and non-square images (uses masks.xy for
    non-square to avoid stretching letterbox padding).

    Parameters
    ----------
    result : ultralytics result object
    h, w : int
        Original image height and width.

    Returns
    -------
    np.ndarray  uint8 (H, W), values 0 or 255.
    """
    pred = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None or len(result.masks) == 0:
        return pred

    if h == w:
        # Square image — masks.data is safe
        for seg in result.masks.data.cpu().numpy():
            resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
            pred[resized > 0.5] = 255
    else:
        # Non-square — use polygon coordinates to avoid letterbox stretch
        for xy in result.masks.xy:
            if len(xy) < 3:
                continue
            pts = xy.astype(np.int32)
            cv2.fillPoly(pred, [pts], 255)

    return pred


def prob_map_from_result(result, h: int, w: int) -> np.ndarray:
    """
    Extract a soft float32 probability map from a YOLO result object.

    Takes element-wise maximum across all detected masks so overlapping
    detections contribute their highest confidence value.

    Parameters
    ----------
    result : ultralytics result object
    h, w : int
        Original image height and width.

    Returns
    -------
    np.ndarray  float32 (H, W), values in [0, 1].
    """
    prob = np.zeros((h, w), dtype=np.float32)
    if result.masks is None or len(result.masks) == 0:
        return prob

    for seg in result.masks.data.cpu().numpy():
        resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
        prob    = np.maximum(prob, resized)

    return prob


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

import random

def augment_pair(
    img: np.ndarray,
    mask: np.ndarray,
    scale_range: tuple = (0.85, 1.15)
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Apply one random geometric augmentation to an image and its mask.
    No colour changes are ever applied — skin tone must remain the same ITA bin.

    Augmentation types:
        hflip, vflip, rot90, rot180, rot270, zoom_in, zoom_out

    Parameters
    ----------
    img : np.ndarray  BGR image.
    mask : np.ndarray  Corresponding mask.
    scale_range : tuple  (min_scale, max_scale) for zoom.

    Returns
    -------
    (augmented_img, augmented_mask, aug_type_string)
    """
    aug_type = random.choice(
        ['hflip', 'vflip', 'rot90', 'rot180', 'rot270', 'zoom_in', 'zoom_out']
    )

    if aug_type == 'hflip':
        return cv2.flip(img, 1), cv2.flip(mask, 1), 'hflip'

    if aug_type == 'vflip':
        return cv2.flip(img, 0), cv2.flip(mask, 0), 'vflip'

    if aug_type == 'rot90':
        return (cv2.rotate(img,  cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE), 'rot90')

    if aug_type == 'rot180':
        return (cv2.rotate(img,  cv2.ROTATE_180),
                cv2.rotate(mask, cv2.ROTATE_180), 'rot180')

    if aug_type == 'rot270':
        return (cv2.rotate(img,  cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE), 'rot270')

    if aug_type in ('zoom_in', 'zoom_out'):
        h, w     = img.shape[:2]
        lo, hi   = scale_range
        scale    = random.uniform(lo, (lo + hi) / 2) if aug_type == 'zoom_out' \
                   else random.uniform((lo + hi) / 2, hi)
        new_h    = int(h * scale)
        new_w    = int(w * scale)
        res_img  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        res_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if scale > 1.0:
            # Centre-crop back to original size
            y0       = (new_h - h) // 2
            x0       = (new_w - w) // 2
            out_img  = res_img [y0:y0+h, x0:x0+w]
            out_mask = res_mask[y0:y0+h, x0:x0+w]
        else:
            # Pad with reflection
            py       = (h - new_h) // 2
            px       = (w - new_w) // 2
            out_img  = cv2.copyMakeBorder(res_img,  py, h-new_h-py,
                                          px, w-new_w-px, cv2.BORDER_REFLECT_101)
            out_mask = cv2.copyMakeBorder(res_mask, py, h-new_h-py,
                                          px, w-new_w-px, cv2.BORDER_REFLECT_101)
        return out_img, out_mask, aug_type

    return img, mask, 'none'
