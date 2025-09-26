# utils/ellipsoid/outer.py
# -*- coding: utf-8 -*-
"""
Minimum-volume enclosing ellipsoid (outer ellipsoid / MVEE).
Original author: Raluca Sandu
"""

import numpy as np
import numpy.linalg as la


def outer_ellipsoid_fit(points: np.ndarray, tol: float = 1e-3):
    """
    Find the minimum-volume ellipsoid enclosing a set of points.
    Returns A, c such that (x - c)^T A (x - c) = 1.

    Args:
        points (ndarray): (N, d) array of points (d = 2 or 3).
        tol (float): convergence tolerance for Khachiyan's algorithm.

    Returns:
        A (ndarray): (d, d) SPD matrix of the ellipsoid.
        c (ndarray): (d,) center of the ellipsoid.
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (N, d)")

    P = np.asmatrix(points)                 # N x d
    N, d = P.shape
    if N < d + 1:
        raise ValueError("At least d+1 points are required to compute MVEE.")

    # Khachiyan algorithm
    Q = np.column_stack((P, np.ones(N))).T  # (d+1) x N
    u = np.ones(N) / N
    err = 1.0 + tol

    while err > tol:
        X = Q @ np.diag(u) @ Q.T
        M = np.diag(Q.T @ la.inv(X) @ Q)    # N-long vector
        j = int(np.argmax(M))
        step = (M[j] - d - 1.0) / ((d + 1) * (M[j] - 1.0))
        new_u = (1 - step) * u
        new_u[j] += step
        err = la.norm(new_u - u)
        u = new_u

    c = np.squeeze(np.asarray(u @ P))       # center (d,)
    A = la.inv(P.T @ np.diag(u) @ P - np.outer(c, c)) / d  # (d,d)

    return np.asarray(A, dtype=float), c.astype(float)
