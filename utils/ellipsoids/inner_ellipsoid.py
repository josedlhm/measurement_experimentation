# utils/ellipsoid/inner.py
# -*- coding: utf-8 -*-
"""
Inner ellipsoid fitting (maximum-volume inscribed ellipsoid).
Original author: Raluca Sandu
"""

import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull


def inner_ellipsoid_fit(points: np.ndarray):
    """
    Find the maximum-volume ellipsoid inscribed in the convex hull of given points.
    
    Args:
        points (ndarray): Array of shape (N, d) with N points in d dimensions (2D or 3D).
    
    Returns:
        (B, d): 
            B (ndarray): Symmetric positive-definite matrix defining the ellipsoid.
            d (ndarray): Center of the ellipsoid.
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, d)")

    dim = points.shape[1]
    A, b, _ = _get_hull(points)

    # Define optimization variables
    B = cp.Variable((dim, dim), PSD=True)  # Ellipsoid matrix
    d = cp.Variable(dim)                   # Ellipsoid center

    # Constraints: ellipsoid must be inside convex hull
    constraints = [cp.norm(B @ A[i], 2) + A[i] @ d <= b[i] for i in range(len(A))]

    # Maximize volume <=> maximize log_det(B)
    prob = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
    optval = prob.solve()

    if optval in (np.inf, None) or B.value is None or d.value is None:
        raise RuntimeError("Inner ellipsoid fitting failed or infeasible.")

    return B.value, d.value


def _get_hull(points: np.ndarray):
    """Return half-space representation (A, b) of convex hull Ax <= b."""
    dim = points.shape[1]
    hull = ConvexHull(points)
    A = hull.equations[:, 0:dim]
    b = hull.equations[:, dim]
    return A, -b, hull
