# utils/measure.py — geometric measurements on 3D points (mm)
from __future__ import annotations
import numpy as np

# optional: wrappers for ellipsoid fits (you already vendored these)
from utils.ellipsoids.outer_ellipsoid import outer_ellipsoid_fit
from utils.ellipsoids.inner_ellipsoid import inner_ellipsoid_fit


# ---------- PCA “ellipsoid-like” extents ----------
def pca_extents_mm(points_mm: np.ndarray):
    """
    Return (a, b, c) diameters (mm) along principal axes, sorted desc.
    points_mm: (N,3) float array in millimeters.
    """
    if not isinstance(points_mm, np.ndarray):
        points_mm = np.asarray(points_mm)
    if points_mm.ndim != 2 or points_mm.shape[1] != 3 or points_mm.shape[0] < 5:
        return np.nan, np.nan, np.nan

    P = points_mm - points_mm.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    proj = P @ Vt.T
    ranges = proj.max(axis=0) - proj.min(axis=0)
    a, b, c = np.sort(ranges)[::-1]
    return float(a), float(b), float(c)


def pca_major_diameter_mm(points_mm: np.ndarray) -> float:
    a, _, _ = pca_extents_mm(points_mm)
    return float(a)


# ---------- Sphere RANSAC ----------
def _sphere_from_4(p1, p2, p3, p4):
    A = np.stack([p2 - p1, p3 - p1, p4 - p1], axis=0)
    if abs(np.linalg.det(A)) < 1e-9:
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
                              min_inlier_ratio: float = 0.2) -> tuple[float, int]:
    """
    Fit a sphere by RANSAC; returns (diameter_mm, inlier_count).
    points_mm: (N,3) in millimeters.
    """
    pts = np.asarray(points_mm)
    n = pts.shape[0]
    if n < 20:
        return np.nan, 0

    idx = np.arange(n)
    best_inl, best_R = -1, np.nan
    for _ in range(iters):
        ids = np.random.choice(idx, size=4, replace=False)
        fit = _sphere_from_4(pts[ids[0]], pts[ids[1]], pts[ids[2]], pts[ids[3]])
        if fit is None:
            continue
        c, R = fit
        d = np.abs(np.linalg.norm(pts - c, axis=1) - R)
        inl = int((d <= inlier_thr_mm).sum())
        if inl > best_inl:
            best_inl, best_R = inl, R

    if best_inl < max(int(min_inlier_ratio * n), 20):
        return np.nan, best_inl
    return float(2.0 * best_R), best_inl


# ---------- Ellipsoid wrappers (operate on points only) ----------
def outer_ellipsoid_major_diameter_mm(points_mm: np.ndarray) -> float:
    """
    Minimum-volume enclosing ellipsoid (MVEE) major diameter.
    (x-c)^T A (x-c) = 1  => radii = 1/sqrt(eigvals(A))
    """
    A, _ = outer_ellipsoid_fit(points_mm)
    evals, _ = np.linalg.eigh(A)
    radii = 1.0 / np.sqrt(np.clip(evals, 1e-12, None))
    return float(2.0 * radii.max())


def inner_ellipsoid_major_diameter_mm(points_mm: np.ndarray) -> float:
    """
    Maximum-volume inscribed ellipsoid (MIE) major diameter.
    ||B(x-d)||_2 <= 1  => radii = 1/singular_values(B)
    """
    B, _ = inner_ellipsoid_fit(points_mm)
    s = np.linalg.svd(B, compute_uv=False)
    radii = np.clip(s, 1e-12, None)   
    return float(2.0 * radii.max())


# ---------- Convenience: run all methods on same points ----------
def measure_all_on_points(points_mm: np.ndarray) -> dict:
    sph, _ = ransac_sphere_diameter_mm(points_mm)
    pca = pca_major_diameter_mm(points_mm)
    try:
        outer = outer_ellipsoid_major_diameter_mm(points_mm)
    except Exception:
        outer = np.nan
    try:
        inner = inner_ellipsoid_major_diameter_mm(points_mm)
    except Exception:
        inner = np.nan
    return {
        "sphere_mm": sph,
        "pca_mm": pca,
        "outer_mm": outer,
        "inner_mm": inner,
    }
