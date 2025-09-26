import numpy as np
from .outer import outer_ellipsoid_fit
from .inner import inner_ellipsoid_fit

def fit_outer(points: np.ndarray):
    """Return (radii, center) for minimum-volume enclosing ellipsoid."""
    A, c = outer_ellipsoid_fit(points)
    evals, evecs = np.linalg.eigh(A)
    radii = 1.0 / np.sqrt(evals.clip(min=1e-12))
    return radii, c

def fit_inner(points: np.ndarray):
    """Return (radii, center) for maximum-volume inscribed ellipsoid."""
    B, d = inner_ellipsoid_fit(points)
    s = np.linalg.svd(B, compute_uv=False)
    radii = 1.0 / s.clip(min=1e-12)
    return radii, d