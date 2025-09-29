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
    depth_mask_to_points_mm,
    keep_near_core_depth_mm,   # core-depth band-pass
)

# --- measurement utils (ONLY inner-ellipsoid used) ---
from utils.measure import inner_ellipsoid_major_diameter_mm

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

# ---- Detectron2 mask picker ----
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

# ---- Per-sample runner using inner ellipsoid only ----
def measure_inner_ellipsoid_for_sample(sample_row, erode_px, trim, band_mm):
    sid = str(int(sample_row["sample_id"]))
    img_dir  = (SAMPLES / sid / "images")
    depth_dir = (SAMPLES / sid / "depth")

    diams = []

    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        dpth_path = depth_dir / f"{name}.npy"
        if not dpth_path.exists():
            continue

        img_bgr  = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        depth_mm = np.load(dpth_path)

        outputs  = predictor(img_bgr)
        raw_mask = pick_mask(outputs)
        if raw_mask is None:
            continue

        inlier = (raw_mask > 0)
        pts = depth_mask_to_points_mm(depth_mm, inlier, fx, fy, cx, cy)  # (N,3) mm
        if pts.shape[0] < 50:
            continue

        # essential prefilter
        pts = keep_near_core_depth_mm(
            pts, depth_mm, raw_mask,
            erode_px=erode_px, trim=trim, band_mm=band_mm
        )
        if pts.shape[0] < 50:
            continue

        try:
            d_mm = inner_ellipsoid_major_diameter_mm(pts)
        except Exception:
            d_mm = np.nan

        if np.isfinite(d_mm):
            diams.append(d_mm)

    return np.median(diams) if diams else np.nan

# ---- Grid search main ----
def main():
    meta = pd.read_csv(META_CSV)

    # Parameter grid (small but effective). Tweak as needed.
    erode_grid = [2,3 ]
    trim_grid  = [0.05, 0.1, .15]
    band_grid  = [13.0, 14.0]  # mm  (~0.3–0.6 × 20mm)

    # collect samples that exist on disk
    rows = [
        row for _, row in meta.iterrows()
        if (SAMPLES / str(int(row["sample_id"])) / "images").exists()
        and (SAMPLES / str(int(row["sample_id"])) / "depth").exists()
    ]
    if not rows:
        raise RuntimeError("No samples with images/ and depth/ found on disk.")

    # name of GT column (allow either)
    gt_col = "gt_caliber_mm" if "gt_caliber_mm" in meta.columns else "caliber_mm"

    leaderboard = []  # (MAE, (e,t,b), n_valid)

    for e in erode_grid:
        for t in trim_grid:
            for b in band_grid:
                preds, gts = [], []
                for row in rows:
                    pred = measure_inner_ellipsoid_for_sample(row, e, t, b)
                    gt   = float(row[gt_col])
                    if np.isfinite(pred):
                        preds.append(pred)
                        gts.append(gt)

                if not preds:
                    print(f"[grid] erode={e} trim={t:.2f} band={b:.1f} → no valid preds")
                    continue

                errs = np.abs(np.array(preds) - np.array(gts))
                mae  = float(np.median(errs))
                leaderboard.append((mae, (e, t, b), len(preds)))
                print(f"[grid] erode={e} trim={t:.2f} band={b:.1f} → MAE={mae:.2f} mm on {len(preds)} samples")

    if not leaderboard:
        print("No valid results; try loosening band or erode.")
        return

    leaderboard.sort(key=lambda x: x[0])
    mae, (e, t, b), n = leaderboard[0]
    print("\n🏁 Best params:")
    print(f"   erode_px={e}, trim={t}, band_mm={b} → MAE={mae:.2f} mm (n={n})")

    print("\nTop 5:")
    for i, (mae_i, (ee, tt, bb), nn) in enumerate(leaderboard[:5], 1):
        print(f"{i:>2}. erode={ee} trim={tt:.2f} band={bb:.1f} | MAE={mae_i:.2f} mm, n={nn}")

if __name__ == "__main__":
    main()
