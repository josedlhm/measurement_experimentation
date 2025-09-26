#!/usr/bin/env python3
# eval_measurements_one.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# --- your utils ---
from utils import clean_mask, sanitize_depth_mm, mask_mad_inliers, depth_mask_to_points_mm
from utils_measure import pick_mask, ransac_sphere_diameter_mm, pca_major_diameter_mm
from utils_ellipsoid.outer import outer_ellipsoid_fit
from utils_ellipsoid.inner import inner_ellipsoid_fit

# ---- Config ----
ROOT = Path("berry_dataset")
SAMPLES = ROOT / "samples"
META_CSV = ROOT / "metadata.csv"
WEIGHTS = "weights/arandanos_medidas.pth"
CONFIG  = "weights/arandanos_medidas.yaml"
INTRINSICS = [1272.44, 1272.67, 920.062, 618.949]  # fx, fy, cx, cy

# ---- Predictor ----
def load_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cuda"  # or "cpu"
    return DefaultPredictor(cfg)

predictor = load_predictor()
fx, fy, cx, cy = INTRINSICS

# ---- Per-frame measurement ----
def measure_all_methods(img_bgr, depth_mm):
    out = {"ok": False}
    outputs = predictor(img_bgr)
    raw_mask = pick_mask(outputs)
    if raw_mask is None:
        return out

    mask = clean_mask(raw_mask, do_holefill=True, close_k=3, erode_px=1)
    depth = sanitize_depth_mm(depth_mm, max_mm=2000.0)
    inlier_mask = mask_mad_inliers(depth, mask, k=3.0)
    pts = depth_mask_to_points_mm(depth, inlier_mask, fx, fy, cx, cy)
    if pts.shape[0] < 200:
        return out

    # sphere
    sph, _ = ransac_sphere_diameter_mm(pts)
    # pca
    pca = pca_major_diameter_mm(pts)

    # outer ellipsoid
    try:
        A, c = outer_ellipsoid_fit(pts)
        _, D, _ = np.linalg.svd(A)
        axes = 1.0 / np.sqrt(D)
        outer_major = float(np.max(axes) * 2.0)  # diameter
    except Exception:
        outer_major = np.nan

    # inner ellipsoid
    try:
        B, d = inner_ellipsoid_fit(pts)
        U, s, Vt = np.linalg.svd(B)
        axes = 1.0 / np.sqrt(s)
        inner_major = float(np.max(axes) * 2.0)
    except Exception:
        inner_major = np.nan

    return {
        "ok": True,
        "sphere_mm": sph,
        "pca_mm": pca,
        "outer_mm": outer_major,
        "inner_mm": inner_major
    }

# ---- Main loop ----
def main():
    meta = pd.read_csv(META_CSV)

    # 🔑 Pick ONE sample_id to test
    SAMPLE_ID = "0"   # change this manually
    row = meta.loc[meta["sample_id"] == int(SAMPLE_ID)].iloc[0]

    sid = str(row["sample_id"])
    sdir = SAMPLES / sid
    img_dir = sdir / "images"
    depth_dir = sdir / "depth"

    sph, pca, outer, inner = [], [], [], []

    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        dpth_path = depth_dir / f"{name}.npy"
        if not dpth_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        depth_mm = np.load(dpth_path)

        res = measure_all_methods(img_bgr, depth_mm)
        if not res["ok"]:
            continue
        if np.isfinite(res["sphere_mm"]): sph.append(res["sphere_mm"])
        if np.isfinite(res["pca_mm"]):    pca.append(res["pca_mm"])
        if np.isfinite(res["outer_mm"]):  outer.append(res["outer_mm"])
        if np.isfinite(res["inner_mm"]):  inner.append(res["inner_mm"])

    result = {
        "sample_id": sid,
        "gt_weight_g": row["weight_g"],
        "gt_caliber_mm": row["caliber_mm"],
        "sphere_mm": np.median(sph) if sph else np.nan,
        "pca_mm": np.median(pca) if pca else np.nan,
        "outer_mm": np.median(outer) if outer else np.nan,
        "inner_mm": np.median(inner) if inner else np.nan,
        "n_frames": len(sph) + len(pca)
    }
    print(f"✅ Result for sample {sid}: {result}")

if __name__ == "__main__":
    main()
