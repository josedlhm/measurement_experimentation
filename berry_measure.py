#!/usr/bin/env python3
# eval_berries_rmse_fixed.py
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# --- preprocessing utils ---
from utils.pre_processing import (
    depth_mask_to_points_mm,
    keep_near_core_depth_mm,   # core-depth band-pass
)

# --- measurement utils (uses your inner-ellipsoid major diameter) ---
from utils.measure import inner_ellipsoid_major_diameter_mm

# ============================
# FIXED PARAMS (from your berries tuning)
# ============================
ERODE_PX  = 4
TRIM_FRAC = 0.10
BAND_MM   = 12.0
BORDER_MARGIN_PX = 4     # skip detections within this many px of the image edge
MIN_PTS = 200             # minimum 3D points after filtering to accept a frame
MIN_VALID_FRAMES = 1     # per-sample minimum frames to keep the sample
GT_COL_FALLBACK = "caliber_mm"  # if gt_caliber_mm not present
# ============================

# ---- Dataset / Model Config (berries) ----
ROOT = Path("/Volumes/USBDATA/berry_dataset")
SAMPLES = ROOT / "samples"
META_CSV = ROOT / "metadata.csv"
WEIGHTS = "./weights/arandanos_medidas.pth"
CONFIG  = "./weights/arandanos_medidas.yaml"
fx, fy, cx, cy = (1272.44, 1272.67, 920.062, 618.949)  # intrinsics (px; depth in mm)

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

# ---- Utilities ----
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

def mask_touches_border(mask_u8: np.ndarray, margin_px: int = 1) -> bool:
    """True if any positive pixel lies within 'margin_px' of image border."""
    if mask_u8 is None:
        return True
    h, w = mask_u8.shape[:2]
    m = max(1, margin_px)
    return (mask_u8[:m, :].any() or mask_u8[h-m:, :].any()
            or mask_u8[:, :m].any() or mask_u8[:, w-m:].any())

def per_sample_prediction(sample_row):
    """Return median predicted diameter (mm) for a sample, or np.nan if invalid."""
    sid = str(int(sample_row["sample_id"]))
    img_dir  = (SAMPLES / sid / "images")
    depth_dir = (SAMPLES / sid / "depth")
    if not img_dir.exists() or not depth_dir.exists():
        return np.nan, 0  # no frames

    vals = []
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

        # Guard: skip detections touching the image border
        if mask_touches_border(raw_mask, margin_px=BORDER_MARGIN_PX):
            continue

        inlier = (raw_mask > 0)
        pts = depth_mask_to_points_mm(depth_mm, inlier, fx, fy, cx, cy)  # (N,3) mm
        if pts.shape[0] < MIN_PTS:
            continue

        # fixed core-depth band-pass
        pts = keep_near_core_depth_mm(
            pts, depth_mm, raw_mask,
            erode_px=ERODE_PX, trim=TRIM_FRAC, band_mm=BAND_MM
        )
        if pts.shape[0] < MIN_PTS:
            continue

        # measure (inner-ellipsoid major diameter)
        try:
            d_mm = inner_ellipsoid_major_diameter_mm(pts)
        except Exception:
            d_mm = np.nan

        if np.isfinite(d_mm):
            vals.append(d_mm)

    if len(vals) < MIN_VALID_FRAMES:
        return np.nan, len(vals)

    return float(np.median(vals)), len(vals)

def rmse(y_pred, y_true):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    e = y_pred - y_true
    return float(np.sqrt(np.mean(e**2)))

# ---- Main ----
def main():
    meta = pd.read_csv(META_CSV)
    gt_col = "gt_caliber_mm" if "gt_caliber_mm" in meta.columns else GT_COL_FALLBACK

    # collect samples that exist on disk
    rows = [
        row for _, row in meta.iterrows()
        if (SAMPLES / str(int(row["sample_id"])) / "images").exists()
        and (SAMPLES / str(int(row["sample_id"])) / "depth").exists()
    ]
    if not rows:
        raise RuntimeError("No samples with images/ and depth/ found on disk.")

    records = []
    for row in rows:
        sid = int(row["sample_id"])
        gt  = float(row[gt_col])
        pred, nframes = per_sample_prediction(row)
        status = "ok" if np.isfinite(pred) else "skip"
        print(f"[{status}] sample {sid}: frames_used={nframes}, pred={pred if np.isfinite(pred) else 'NaN'}")
        if np.isfinite(pred):
            records.append({"sample_id": sid, "gt_caliber_mm": gt, "pred_mm": pred})

    if not records:
        print("No valid predictions after guards.")
        return

    df = pd.DataFrame(records)
    df["err_mm"] = df["pred_mm"] - df["gt_caliber_mm"]
    overall_rmse = rmse(df["pred_mm"].values, df["gt_caliber_mm"].values)

    print(f"\nFixed preprocessing:"
          f" erode_px={ERODE_PX}, trim={TRIM_FRAC}, band_mm={BAND_MM}, border_margin_px={BORDER_MARGIN_PX}")
    print(f"Samples evaluated: {len(df)}")
    print(f"RMSE (mm): {overall_rmse:.2f}")
    print(f"Median Abs Error (mm): {np.median(np.abs(df['err_mm'])):.2f}")
    print(f"Mean Abs Error (mm): {np.mean(np.abs(df['err_mm'])):.2f}")

    # Save per-sample table
    out_csv = ROOT / "berries_inner_major_fixed_rmse.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n📄 Saved per-sample results to: {out_csv}")

if __name__ == "__main__":
    main()
