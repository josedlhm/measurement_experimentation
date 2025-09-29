#!/usr/bin/env python3
# eval_measurements_one.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# --- preprocessing utils (mask/depth/points) ---
from utils.pre_processing import (
    clean_mask,              # kept imported (not used in raw-only run)
    sanitize_depth_mm,       # kept imported (not used in raw-only run)
    mask_mad_inliers,        # kept imported (not used in raw-only run)
    depth_mask_to_points_mm,
    keep_near_core_depth_mm
)

# --- measurement utils (3D point methods) ---
from utils.measure import (
    ransac_sphere_diameter_mm,
    pca_major_diameter_mm,
    outer_ellipsoid_major_diameter_mm,
    inner_ellipsoid_major_diameter_mm,
)

# --- ellipsoid solvers (if you want direct access) ---
from utils.ellipsoids.outer_ellipsoid import outer_ellipsoid_fit
from utils.ellipsoids.inner_ellipsoid import inner_ellipsoid_fit

# ---- Config ----
ROOT = Path("/Volumes/USBDATA/berry_dataset")
SAMPLES = ROOT / "samples"
META_CSV = ROOT / "metadata.csv"
WEIGHTS = "./weights/arandanos_medidas.pth"
CONFIG  = "./weights/arandanos_medidas.yaml"
fx, fy, cx, cy = (1272.44, 1272.67, 920.062, 618.949)  # intrinsics (pixels; depth in mm)

# ---- Detectron2 predictor ----
def load_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = getattr(
            cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", 0.5
        )
    cfg.INPUT.FORMAT = "BGR"
    return DefaultPredictor(cfg)

predictor = load_predictor()

# ---- Detectron2 mask picker (kept local to avoid mixing into utils) ----
def pick_mask(outputs, class_id_keep=None):
    """
    Return uint8 {0,255} mask for the target instance (largest or class-filtered).
    """
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

# ---- Per-frame: preprocess → points → run 4 measurements (RAW-ONLY) ----
def measure_all_methods_on_frame(img_bgr, depth_mm):
    # 1) mask from detectron
    outputs  = predictor(img_bgr)
    raw_mask = pick_mask(outputs)
    if raw_mask is None:
        return None

    # 2) RAW selection only (no clean_mask, no sanitize_depth_mm, no MAD inliers)
    #    Build boolean mask directly from Detectron output.
    inlier = (raw_mask > 0)

    # 3) project to 3D points (mm) with raw depth + raw mask
    pts = depth_mask_to_points_mm(depth_mm, inlier, fx, fy, cx, cy)
    pts = keep_near_core_depth_mm(pts, depth_mm, raw_mask, erode_px=3, trim=0.1, band_mm=12.0)

    if pts.shape[0] < 200:
        return None
    


    # 4) run all 4 methods on the SAME raw points
    sph, _ = ransac_sphere_diameter_mm(pts)
    pca    = pca_major_diameter_mm(pts)
    try:
        outer = outer_ellipsoid_major_diameter_mm(pts)
    except Exception:
        outer = np.nan
    try:
        inner = inner_ellipsoid_major_diameter_mm(pts)
    except Exception:
        inner = np.nan

    return {"sphere_mm": sph, "pca_mm": pca, "outer_mm": outer, "inner_mm": inner}

# ---- Main (ONE SAMPLE) ----
def main():
    meta = pd.read_csv(META_CSV)

    SAMPLE_ID = "2"  # <— change here for quick test
    rows = meta.loc[meta["sample_id"] == int(SAMPLE_ID)]
    if rows.empty:
        raise ValueError(f"sample_id {SAMPLE_ID} not found in {META_CSV}")

    row = rows.iloc[0]
    sid = str(row["sample_id"])
    img_dir = (SAMPLES / sid / "images")
    depth_dir = (SAMPLES / sid / "depth")

    # accumulate per-frame measurements
    frames_used = 0
    sph, pca, outer, inner = [], [], [], []

    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        dpth_path = depth_dir / f"{name}.npy"
        if not dpth_path.exists():
            continue

        img_bgr  = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        depth_mm = np.load(dpth_path)

        res = measure_all_methods_on_frame(img_bgr, depth_mm)
        if res is None:
            continue

        frames_used += 1
        if np.isfinite(res["sphere_mm"]): sph.append(res["sphere_mm"])
        if np.isfinite(res["pca_mm"]):    pca.append(res["pca_mm"])
        if np.isfinite(res["outer_mm"]):  outer.append(res["outer_mm"])
        if np.isfinite(res["inner_mm"]):  inner.append(res["inner_mm"])

    result = {
        "sample_id": sid,
        "gt_weight_g": float(row["weight_g"]),
        "gt_caliber_mm": float(row["caliber_mm"]),
        "sphere_mm": np.median(sph) if sph else np.nan,
        "pca_mm":    np.median(pca) if pca else np.nan,
        "outer_mm":  np.median(outer) if outer else np.nan,
        "inner_mm":  np.median(inner) if inner else np.nan,
        "n_frames":  frames_used,
    }
    print(f"✅ Result for sample {sid}: {result}")

if __name__ == "__main__":
    main()
