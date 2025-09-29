#!/usr/bin/env python3
# eval_inner_ellipsoid_all_samples.py
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
    keep_near_core_depth_mm,
)

# --- inner-ellipsoid solver (generator form: ||B(x-d)|| <= 1) ---
from utils.ellipsoids.inner_ellipsoid import inner_ellipsoid_fit

# ============================
# FIXED PARAMS (from your grid)
# ============================
ERODE_PX  = 3
TRIM_FRAC = 0.15
BAND_MM   = 12.0
BORDER_MARGIN_PX = 10
MIN_PTS = 200
GT_COL_FALLBACK = "caliber_mm"  # if gt_caliber_mm not present
# ============================

# ---- Dataset / Model Config (grapes) ----
ROOT = Path("/Volumes/USBDATA/grape_dataset")
SAMPLES = ROOT / "samples"
META_CSV = ROOT / "metadata.csv"
WEIGHTS = "./weights/uvas_medidas.pth"
CONFIG  = "./weights/uvas_medidas.yaml"
fx, fy, cx, cy = (1272.44, 1272.67, 920.062, 618.949)

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
    if mask_u8 is None:
        return True
    h, w = mask_u8.shape[:2]
    m = max(1, margin_px)
    return (mask_u8[:m, :].any() or mask_u8[h-m:, :].any()
            or mask_u8[:, :m].any() or mask_u8[:, w-m:].any())

def ellipsoid_derived_diameters_from_B(B: np.ndarray):
    """
    B from generator form: ||B(x-d)|| <= 1. Semi-axes = singular values of B.
    Return a dict of candidate diameters (mm).
    """
    s = np.linalg.svd(B, compute_uv=False)
    s = np.sort(np.clip(s, 1e-12, None))[::-1]  # a >= b >= c
    a, b, c = s

    # Singles
    D_major = 2.0 * a
    D_inter = 2.0 * b
    D_minor = 2.0 * c

    # Two-axis (grip-like)
    D_avg_two = (D_major + D_inter) / 2.0
    D_gmean_two = 2.0 * np.sqrt(a * b)
    D_hmean_two = 4.0 * a * b / (a + b)

    # Three-axis summaries
    D_mean = 2.0 * (a + b + c) / 3.0
    D_rms  = 2.0 * np.sqrt((a*a + b*b + c*c) / 3.0)
    D_vesd = 2.0 * (a * b * c) ** (1.0 / 3.0)

    # Surface-area equivalent sphere (Thomsen p≈1.6075)
    p = 1.6075
    SA = 4.0 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3.0) ** (1.0 / p)
    D_saesd = np.sqrt(SA / np.pi)

    return {
        "inner_major_mm": D_major,
        "inner_intermediate_mm": D_inter,
        "inner_minor_mm": D_minor,
        "inner_avg_two_largest_mm": D_avg_two,
        "inner_gmean_two_largest_mm": D_gmean_two,
        "inner_hmean_two_largest_mm": D_hmean_two,
        "inner_mean_mm": D_mean,
        "inner_rms_mm": D_rms,
        "inner_equiv_sphere_mm": D_vesd,
        "inner_sa_equiv_sphere_mm": D_saesd,
        "_a_mm": a, "_b_mm": b, "_c_mm": c,
    }

def compute_sample_candidates(sample_id: int, gt_value: float):
    sid = str(sample_id)
    img_dir   = (SAMPLES / sid / "images")
    depth_dir = (SAMPLES / sid / "depth")
    if not img_dir.exists() or not depth_dir.exists():
        return None

    per_frame = []
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

        if mask_touches_border(raw_mask, margin_px=BORDER_MARGIN_PX):
            continue

        inlier = (raw_mask > 0)
        pts = depth_mask_to_points_mm(depth_mm, inlier, fx, fy, cx, cy)
        if pts.shape[0] < MIN_PTS:
            continue

        pts = keep_near_core_depth_mm(pts, depth_mm, raw_mask,
                                      erode_px=ERODE_PX, trim=TRIM_FRAC, band_mm=BAND_MM)
        if pts.shape[0] < MIN_PTS:
            continue

        try:
            B, d = inner_ellipsoid_fit(pts)
        except Exception:
            continue

        meas = ellipsoid_derived_diameters_from_B(B)
        per_frame.append(meas)

    if not per_frame:
        return None

    df = pd.DataFrame(per_frame)
    cols = [c for c in df.columns if c.endswith("_mm") and not c.startswith("_")]
    med = df[cols].median(axis=0)

    record = {"sample_id": int(sid), "gt_caliber_mm": float(gt_value)}
    record.update({c: float(med[c]) for c in cols})
    return record

def rmse_col(df: pd.DataFrame, col: str):
    x = df[[col, "gt_caliber_mm"]].dropna()
    if x.empty:
        return np.inf, 0
    e = x[col].to_numpy() - x["gt_caliber_mm"].to_numpy()
    return float(np.sqrt(np.mean(e**2))), int(len(x))

def main():
    meta = pd.read_csv(META_CSV)
    gt_col = "gt_caliber_mm" if "gt_caliber_mm" in meta.columns else GT_COL_FALLBACK

    # gather samples that exist on disk
    rows = [
        (int(r["sample_id"]), float(r[gt_col]))
        for _, r in meta.iterrows()
        if (SAMPLES / str(int(r["sample_id"])) / "images").exists()
        and (SAMPLES / str(int(r["sample_id"])) / "depth").exists()
    ]
    if not rows:
        raise RuntimeError("No samples with images/ and depth/ found on disk.")

    records = []
    for sid, gt in rows:
        rec = compute_sample_candidates(sid, gt)
        if rec is not None:
            records.append(rec)
            print(f"[ok] sample {sid}: summarized from medians")
        else:
            print(f"[skip] sample {sid}: no valid frames after filtering")

    if not records:
        print("No valid samples.")
        return

    df = pd.DataFrame(records)

    # rank candidates by RMSE
    cols = [
        "inner_major_mm",
        "inner_intermediate_mm",
        "inner_minor_mm",
        "inner_avg_two_largest_mm",
        "inner_gmean_two_largest_mm",
        "inner_hmean_two_largest_mm",
        "inner_mean_mm",
        "inner_rms_mm",
        "inner_equiv_sphere_mm",
        "inner_sa_equiv_sphere_mm",
    ]
    scores = []
    for c in cols:
        rm, n = rmse_col(df, c)
        scores.append((c, rm, n))
    scores.sort(key=lambda x: x[1])

    # print summary
    print(f"\nFixed preprocessing: erode_px={ERODE_PX}, trim={TRIM_FRAC}, band_mm={BAND_MM}, border_margin_px={BORDER_MARGIN_PX}")
    print("RMSE by inner-ellipsoid derived diameter (mm) across samples:")
    for c, rm, n in scores:
        print(f"  {c:28s} → RMSE={rm:.2f} mm on {n} samples")

    best_col, best_rmse, best_n = scores[0]
    print(f"\n🏁 Best proxy for caliper: {best_col} (RMSE={best_rmse:.2f} mm, n={best_n})")

    # save per-sample predictions + errors
    out_csv = ROOT / "inner_ellipsoid_axes_comparison_all_samples.csv"
    for c in cols:
        df[f"err_{c}_mm"] = df[c] - df["gt_caliber_mm"]
    df.to_csv(out_csv, index=False)
    print(f"\n📄 Saved per-sample results to: {out_csv}")

if __name__ == "__main__":
    main()
