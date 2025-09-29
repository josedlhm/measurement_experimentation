#!/usr/bin/env python3
# eval_inner_ellipsoid_axes_rmse.py
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

# --- inner-ellipsoid solver (returns B, d) ---
from utils.ellipsoids.inner_ellipsoid import inner_ellipsoid_fit

# ============================
# FIXED PARAMS (set these)
# ============================
ERODE_PX  = 2       # from your grid
TRIM_FRAC = 0.10    # from your grid
BAND_MM   = 12.0    # from your grid
BORDER_MARGIN_PX = 2    # exclude masks touching border within this margin
GT_COL_FALLBACK = "caliber_mm"  # if gt_caliber_mm not present
MIN_PTS = 50            # minimum points after filtering
# ============================

# ---- Dataset / Model Config ----
ROOT = Path("/Volumes/USBDATA/grape_dataset")
SAMPLES = ROOT / "samples"
META_CSV = ROOT / "metadata.csv"
WEIGHTS = "./weights/uvas_medidas.pth"
CONFIG  = "./weights/uvas_medidas.yaml"
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

def mask_touches_border(mask_u8: np.ndarray, margin_px: int = 1) -> bool:
    """True if any positive pixel lies within 'margin_px' of image border."""
    if mask_u8 is None:
        return True
    h, w = mask_u8.shape[:2]
    m = max(0, margin_px)
    top    = mask_u8[:m, :]
    bottom = mask_u8[h-m:, :]
    left   = mask_u8[:, :m]
    right  = mask_u8[:, w-m:]
    return (top.any() or bottom.any() or left.any() or right.any())

def inner_ellipsoid_axes_mm(pts_mm: np.ndarray):
    """
    Fit inner ellipsoid: (x - d)^T B (x - d) <= 1  with B SPD.
    Return semi-axes (a >= b >= c) in mm. NaNs if fail.
    """
    if pts_mm is None or pts_mm.shape[0] < 10:
        return np.nan, np.nan, np.nan
    try:
        B, d = inner_ellipsoid_fit(pts_mm)  # expected API from your utils
        # eigen-decomp: B = R diag(1/a^2, 1/b^2, 1/c^2) R^T
        evals, _ = np.linalg.eigh(B)
        # numerical guard
        evals = np.maximum(evals, 1e-12)
        axes = 1.0 / np.sqrt(evals)  # a,b,c (unordered)
        axes = np.sort(axes)[::-1]   # descending: a >= b >= c
        a, b, c = axes.tolist()
        return float(a), float(b), float(c)
    except Exception:
        return np.nan, np.nan, np.nan

# ---- Per-sample: compute all inner-ellipsoid based candidates (median across frames) ----
def compute_sample_candidates(sample_row):
    sid = str(int(sample_row["sample_id"]))
    img_dir   = (SAMPLES / sid / "images")
    depth_dir = (SAMPLES / sid / "depth")

    majors, inters, minors, means, equivs = [], [], [], [], []

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

        # Exclude masks touching image borders (likely incomplete detections)
        if mask_touches_border(raw_mask, margin_px=BORDER_MARGIN_PX):
            continue

        inlier = (raw_mask > 0)
        pts = depth_mask_to_points_mm(depth_mm, inlier, fx, fy, cx, cy)  # (N,3) in mm
        if pts.shape[0] < MIN_PTS:
            continue

        # fixed prefilter
        pts = keep_near_core_depth_mm(
            pts, depth_mm, raw_mask,
            erode_px=ERODE_PX, trim=TRIM_FRAC, band_mm=BAND_MM
        )
        if pts.shape[0] < MIN_PTS:
            continue

        # Inner ellipsoid axes
        a, b, c = inner_ellipsoid_axes_mm(pts)
        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
            continue

        # Candidate diameters (mm)
        D_major = 2.0 * a
        D_inter = 2.0 * b
        D_minor = 2.0 * c
        D_mean  = (2.0 * a + 2.0 * b + 2.0 * c) / 3.0
        D_equiv = 2.0 * (a * b * c) ** (1.0 / 3.0)  # volume-equivalent sphere diameter

        majors.append(D_major)
        inters.append(D_inter)
        minors.append(D_minor)
        means.append(D_mean)
        equivs.append(D_equiv)

    # per-sample medians (NaN if none)
    med_major = np.median(majors) if majors else np.nan
    med_inter = np.median(inters) if inters else np.nan
    med_minor = np.median(minors) if minors else np.nan
    med_mean  = np.median(means)  if means  else np.nan
    med_equiv = np.median(equivs) if equivs else np.nan

    return med_major, med_inter, med_minor, med_mean, med_equiv

def rmse_col(df: pd.DataFrame, col: str):
    x = df[[col, "gt_caliber_mm"]].dropna()
    if x.empty:
        return np.inf, 0
    err = x[col].values - x["gt_caliber_mm"].values
    return float(np.sqrt(np.mean(err**2))), int(len(x))

# ---- Main: evaluate RMSE per candidate, rank, save CSV ----
def main():
    meta = pd.read_csv(META_CSV)
    gt_col = "gt_caliber_mm" if "gt_caliber_mm" in meta.columns else GT_COL_FALLBACK

    # samples present on disk
    rows = [
        row for _, row in meta.iterrows()
        if (SAMPLES / str(int(row["sample_id"])) / "images").exists()
        and (SAMPLES / str(int(row["sample_id"])) / "depth").exists()
    ]
    if not rows:
        raise RuntimeError("No samples with images/ and depth/ found on disk.")

    recs = []
    for row in rows:
        sid = int(row["sample_id"])
        gt  = float(row[gt_col])

        med_major, med_inter, med_minor, med_mean, med_equiv = compute_sample_candidates(row)

        recs.append({
            "sample_id": sid,
            "gt_caliber_mm": gt,
            "inner_major_mm": med_major,
            "inner_intermediate_mm": med_inter,
            "inner_minor_mm": med_minor,
            "inner_mean_mm": med_mean,
            "inner_equiv_sphere_mm": med_equiv,
        })

    df = pd.DataFrame(recs)

    # RMSE per candidate
    scores = []
    for col in [
        "inner_major_mm",
        "inner_intermediate_mm",
        "inner_minor_mm",
        "inner_mean_mm",
        "inner_equiv_sphere_mm",
    ]:
        rm, n = rmse_col(df, col)
        scores.append((col, rm, n))

    scores.sort(key=lambda x: x[1])
    best_col, best_rmse, best_n = scores[0]

    # Print summary
    print(f"\nFixed preprocessing: erode_px={ERODE_PX}, trim={TRIM_FRAC}, band_mm={BAND_MM}, border_margin_px={BORDER_MARGIN_PX}")
    print("RMSE by inner-ellipsoid derived diameter (mm) [lower is better]:")
    for col, rm, n in scores:
        print(f"  {col:26s} → RMSE={rm:.2f} mm on {n} samples")

    print(f"\n🏁 Best proxy for caliper: {best_col} (RMSE={best_rmse:.2f} mm, n={best_n})")

    # Save per-sample table
    out_csv = ROOT / "inner_ellipsoid_axes_comparison.csv"
    # also include errors per candidate for quick review
    for col in ["inner_major_mm", "inner_intermediate_mm", "inner_minor_mm", "inner_mean_mm", "inner_equiv_sphere_mm"]:
        df[f"err_{col}_mm"] = df[col] - df["gt_caliber_mm"]
    df.to_csv(out_csv, index=False)
    print(f"\n📄 Saved per-sample results to: {out_csv}")

if __name__ == "__main__":
    main()
