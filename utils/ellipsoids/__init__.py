import numpy as np
from .outer_ellipsoid import outer_ellipsoid_fit
from .inner_ellipsoid import inner_ellipsoid_fit

def fit_outer(points: np.ndarray):
    """Return (radii, center) for minimum-volume enclosing ellipsoid."""
    A, c = outer_ellipsoid_fit(points)               # (x-c)^T A (x-c) <= 1
    evals, _ = np.linalg.eigh(A)
    radii = 1.0 / np.sqrt(np.clip(evals, 1e-12, None))  # semi-axes
    return radii, c

def fit_inner(points: np.ndarray):
    """Return (radii, center) for maximum-volume inscribed ellipsoid."""
    B, d = inner_ellipsoid_fit(points)               # ||B^{-1}(x-d)|| <= 1
    s = np.linalg.svd(B, compute_uv=False)           # semi-axes = singular values
    radii = np.clip(s, 1e-12, None)                  
    return radii, d
            