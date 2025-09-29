# utils.py — mask & depth helpers for blueberry measurement

from __future__ import annotations
import numpy as np
import cv2

# ----------------------- MASK CLEANUP -----------------------

def keep_largest_cc(mask_u8: np.ndarray) -> np.ndarray:
    """
    Keep the largest connected component of a binary mask.
    mask_u8: uint8 {0,255}
    """
    if mask_u8.ndim != 2:
        raise ValueError("mask must be HxW")
    bin_ = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask_u8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))  # label index of largest CC (skip background=0)
    out = (labels == k).astype(np.uint8) * 255
    return out

def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """
    Fill internal holes using flood-fill from the border.
    Returns uint8 {0,255}.
    """
    h, w = mask_u8.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)      # floodFill needs +2 border
    filled = mask_u8.copy()
    cv2.floodFill(filled, ff_mask, (0, 0), 255)       # fill background from a corner
    holes = cv2.bitwise_not(filled)                   # holes become 255
    return cv2.bitwise_or(mask_u8, holes)

def morph_close(mask_u8: np.ndarray, k: int = 3) -> np.ndarray:
    """One pass close to seal pinholes; k==0 disables."""
    if k <= 0: return mask_u8
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

def erode(mask_u8: np.ndarray, px: int = 1) -> np.ndarray:
    """Erode border to avoid mixed-depth edge pixels; px==0 disables."""
    if px <= 0: return mask_u8
    kernel = np.ones((px, px), np.uint8)
    return cv2.erode(mask_u8, kernel, iterations=1)

def clean_mask(mask_u8: np.ndarray, do_holefill: bool = True, close_k: int = 3, erode_px: int = 1) -> np.ndarray:
    """
    Largest CC → (hole-fill + close) → erode.
    Returns uint8 {0,255}.
    """
    m = keep_largest_cc(mask_u8)
    if do_holefill:
        m = fill_holes(m)
    if close_k > 0:
        m = morph_close(m, close_k)
    if erode_px > 0:
        m = erode(m, erode_px)
    return m

# ----------------------- DEPTH CLEANUP -----------------------

def sanitize_depth_mm(depth_mm: np.ndarray, max_mm: float = 2000.0) -> np.ndarray:
    """
    Ensure depth is float32 in [0, max_mm]. NaN/Inf/negatives -> 0.
    """
    d = depth_mm.astype(np.float32, copy=True)
    np.nan_to_num(d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    d[d <= 0.0] = 0.0
    d[d > max_mm] = 0.0
    return d

def mask_mad_inliers(depth_mm: np.ndarray, mask_u8: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Robust Z inlier mask inside 'mask_u8' using median absolute deviation (MAD).
    Keeps pixels with |z - median| <= k * MAD.
    Returns uint8 {0,255}, subset of input mask.
    """
    base = (mask_u8 > 0)
    if not np.any(base):
        return np.zeros_like(mask_u8)

    # use only valid depths within base for robust stats
    z_all = depth_mm[base]
    z_valid = z_all[z_all > 0]
    if z_valid.size == 0:
        return np.zeros_like(mask_u8)

    med = np.median(z_valid)
    mad = np.median(np.abs(z_valid - med)) + 1e-6  # avoid zero
    # apply criterion to all base pixels
    keep = np.zeros_like(base, dtype=bool)
    good = (z_all > 0) & (np.abs(z_all - med) <= k * mad)
    ys, xs = np.where(base)
    keep[ys[good], xs[good]] = True
    return (keep.astype(np.uint8) * 255)

# ----------------------- 2D→3D PROJECTION -----------------------

def depth_mask_to_points_mm(depth_mm: np.ndarray, mask_u8: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Project masked depth to 3D points in millimeters.
    depth_mm: float32 HxW in mm (0 for invalid)
    mask_u8:  uint8 {0,255}
    Returns: Nx3 (x,y,z) in mm.
    """
    m = (mask_u8 > 0) & (depth_mm > 0)
    if not np.any(m):
        return np.empty((0, 3), np.float32)
    ys, xs = np.nonzero(m)
    z = depth_mm[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1)

import numpy as np, cv2

def keep_near_core_depth_mm(pts_mm, depth_mm, mask_u8, erode_px=3, trim=0.1, band_mm=12.0):
    """
    Keep only 3D points whose Z is within ±band_mm of a robust 'core' depth.
    - Core = eroded mask (avoids rim mixed pixels).
    - Core depth = trimmed mean of 3x3 median-filtered depths.
    """
    M = (mask_u8 > 0).astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
        core = cv2.erode(M, k)
    else:
        core = M

    core_idx = (core > 0) & np.isfinite(depth_mm) & (depth_mm > 0)
    if core_idx.sum() < 20:
        return pts_mm  # not enough support; leave unchanged

    d = depth_mm.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    d_med = cv2.medianBlur(d, 3)
    vals = np.sort(d_med[core_idx].astype(np.float64))
    n = vals.size
    lo, hi = int(trim*n), int((1.0-trim)*n)
    z_core = vals[lo:hi].mean() if hi > lo else float(np.median(vals))
    keep = np.abs(pts_mm[:,2] - z_core) <= band_mm
    return pts_mm[keep]