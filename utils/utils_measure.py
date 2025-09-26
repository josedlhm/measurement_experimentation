# utils_measure.py — geometric measurements for blueberry size
from __future__ import annotations
import numpy as np
import cv2

# If you placed the helpers in utils.py:
from utils import clean_mask, sanitize_depth_mm, mask_mad_inliers, depth_mask_to_points_mm

# ---------------- Detectron2 mask picker ----------------
def pick_mask(outputs, class_id_keep=None):
    """Return uint8 {0,255} mask for the target instance (largest or class-filtered)."""
    inst = outputs["instances"].to("cpu")
    if len(inst) == 0:
        return None
    if class_id_keep is not None:
        keep = (inst.pred_classes.numpy() == class_id_keep)
        if not keep.any():
            return None
        masks = inst.pred_masks[keep].numpy()
    else:
        masks = inst.pred_masks.numpy()
    areas = masks.sum(axis=(1, 2))
    idx = int(np.argmax(areas))
    return (masks[idx].astype(np.uint8) * 255)

# ---------------- PCA “ellipsoid” extents ----------------
def pca_extents_mm(points_mm: np.ndarray):
    """
    Returns (a, b, c) diameters (mm) along principal axes, sorted desc.
    points_mm: Nx3 array in millimeters.
    """
    if points_mm.shape[0] < 5:
        return np.nan, np.nan, np.nan
    P = points_mm - points_mm.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    proj = P @ Vt.T  # (N,3)
    ranges = proj.max(axis=0) - proj.min(axis=0)  # diameters along principal axes
    a, b, c = np.sort(ranges)[::-1]
    return float(a), float(b), float(c)

def pca_major_diameter_mm(points_mm: np.ndarray):
    a, b, c = pca_extents_mm(points_mm)
    return a

# ---------------- Sphere RANSAC ----------------
def _sphere_from_4(p1, p2, p3, p4):
    A = np.stack([p2 - p1, p3 - p1, p4 - p1], axis=0)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    B = 0.5 * np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1),
        np.dot(p4, p4) - np.dot(p1, p1)
    ], dtype=np.float64)
    c_rel = np.linalg.solve(A, B)
    c = c_rel + p1
    R = np.linalg.norm(p1 - c)
    return c, R

def ransac_sphere_diameter_mm(points_mm: np.ndarray,
                              iters: int = 600,
                              inlier_thr_mm: float = 1.5,
                              min_inlier_ratio: float = 0.2):
    """
    Fit a sphere with RANSAC; return (diameter_mm, inlier_count).
    points_mm: Nx3 float32/64 in millimeters.
    """
    n = points_mm.shape[0]
    if n < 20:
        return np.nan, 0
    idx = np.arange(n)
    best_inl, best_R = -1, np.nan
    for _ in range(iters):
        ids = np.random.choice(idx, size=4, replace=False)
        res = _sphere_from_4(points_mm[ids[0]], points_mm[ids[1]],
                             points_mm[ids[2]], points_mm[ids[3]])
        if res is None:
            continue
        c, R = res
        d = np.abs(np.linalg.norm(points_mm - c, axis=1) - R)
        inl = int((d <= inlier_thr_mm).sum())
        if inl > best_inl:
            best_inl, best_R = inl, R
    if best_inl < max(int(min_inlier_ratio * n), 20):
        return np.nan, best_inl
    return float(2.0 * best_R), best_inl

# ---------------- Per-frame measurement wrapper ----------------
def measure_frame_mm(img_bgr: np.ndarray,
                     depth_mm: np.ndarray,
                     predictor,              # Detectron2 predictor
                     fx: float, fy: float, cx: float, cy: float,
                     max_depth_mm: float = 2000.0,
                     morph_close_k: int = 3,
                     erode_px: int = 1,
                     mad_k: float = 3.0,
                     class_id_keep=None,
                     min_points: int = 500):
    """
    Returns a dict with:
      - 'sphere_mm': diameter (median frame estimate)
      - 'ellipsoid_major_mm': major-axis diameter
      - 'n_points': points used after filtering
      - 'ok': whether any measurement succeeded
    """
    # 1) instance mask
    outputs = predictor(img_bgr)
    raw_mask = pick_mask(outputs, class_id_keep)
    if raw_mask is None:
        return {"ok": False}

    # 2) mask cleanup
    mask = clean_mask(raw_mask, do_holefill=True, close_k=morph_close_k, erode_px=erode_px)

    # 3) depth sanitize + robust Z inliers
    depth = sanitize_depth_mm(depth_mm, max_mm=max_depth_mm)
    inlier_mask = mask_mad_inliers(depth, mask, k=mad_k)

    # 4) 2D→3D
    pts = depth_mask_to_points_mm(depth, inlier_mask, fx, fy, cx, cy)
    if pts.shape[0] < min_points:
        return {"ok": False, "n_points": int(pts.shape[0])}

    # 5) measurements
    sph_mm, inl = ransac_sphere_diameter_mm(pts)
    maj_mm = pca_major_diameter_mm(pts)

    return {
        "ok": (np.isfinite(sph_mm) or np.isfinite(maj_mm)),
        "n_points": int(pts.shape[0]),
        "sphere_mm": sph_mm,
        "ellipsoid_major_mm": maj_mm
    }
