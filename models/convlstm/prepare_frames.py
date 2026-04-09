"""
models/convlstm/prepare_frames.py
==================================
STEP 1 — Spatial frame preparation for the ConvLSTM model.

Reads : data/tiffs/ntl_YYYY_MM.tif   (144 monthly GeoTIFFs, 35×45 pixels)
Writes: models/convlstm/frames.npz          (X_train, y_train, X_test, y_test, window_size)
        models/convlstm/frame_scaler.pkl    (MinMaxScaler fit on train pixels only)
        models/convlstm/frame_metadata.json (dates, H, W, transform, CRS, split info)

Train / Test split (identical to SARIMA/LSTM for consistency):
  Train : Jan 2014 – Dec 2023  (120 frames)
  Test  : Jan 2024 – Dec 2025  ( 24 frames)

Sliding window (W=12, stride=1):
  X[i] = frames[i : i+W]          shape (W, H, W_img, 1)  — input sequence
  y[i] = frames[i+W]              shape (H, W_img, 1)      — next frame to predict

  Train sequences : 120 - 12 = 108    shape (108, 12, 35, 45, 1)
  Test  sequences :  24              shape ( 24, 12, 35, 45, 1)
    └─ seeded from last W train frames, rolled forward one step at a time

NoData handling:
  Source TIFFs have nodata=-inf  →  replaced with 0.0 before scaling.

Normalisation:
  MinMaxScaler fitted ONLY on flattened train pixels  →  [0, 1] range.
  Same scaler applied to test frames (no leakage).

Memory: full tensor (132, 12, 35, 45, 1) float32 ≈ 9.52 MB — CPU-trivial.

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/convlstm/prepare_frames.py [--window W]
"""

import os
import sys
import json
import glob
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

import rasterio
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─── Paths / city config ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

_parser = argparse.ArgumentParser(description="Prepare ConvLSTM spatial frame sequences")
_parser.add_argument("--city", default="kharagpur",
                     help="City key from src/cities.py  (default: kharagpur)")
_parser.add_argument("--window", type=int, default=12,
                     help="Sliding window size in months (default: 12)")
ARGS = _parser.parse_args()
CITY = ARGS.city.lower().strip()

TIFF_DIR    = _cities.get_tiff_dir(CITY, ROOT)
MODEL_DIR   = _cities.get_convlstm_model_dir(CITY, ROOT)
OUTPUT_DIR  = _cities.get_convlstm_dir(CITY, ROOT)

FRAMES_NPZ    = os.path.join(MODEL_DIR, "frames.npz")
SCALER_PKL    = os.path.join(MODEL_DIR, "frame_scaler.pkl")
METADATA_JSON = os.path.join(MODEL_DIR, "frame_metadata.json")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Split boundary (same as SARIMA / LSTM) ───────────────────────────────────
TRAIN_END  = pd.Timestamp("2023-12-01")
TEST_START = pd.Timestamp("2024-01-01")

DEFAULT_WINDOW = 12   # one full seasonal cycle


# ─── 1. Discover and sort TIFFs ──────────────────────────────────────────────

def find_tiffs() -> list[tuple[pd.Timestamp, str]]:
    """Return sorted list of (date, filepath) for all TIFFs."""
    paths = sorted(glob.glob(os.path.join(TIFF_DIR, "ntl_*.tif")))
    if not paths:
        sys.exit(f"[ERROR] No TIFFs found in {TIFF_DIR}")

    entries = []
    for p in paths:
        # filename: ntl_YYYY_MM.tif
        base = os.path.splitext(os.path.basename(p))[0]   # ntl_2014_01
        parts = base.split("_")                            # ['ntl', '2014', '01']
        if len(parts) != 3:
            print(f"[WARN] Skipping unexpected filename: {p}")
            continue
        try:
            date = pd.Timestamp(f"{parts[1]}-{parts[2]}-01")
        except ValueError:
            print(f"[WARN] Cannot parse date from: {p}")
            continue
        entries.append((date, p))

    entries.sort(key=lambda x: x[0])
    print(f"[INFO] Found {len(entries)} TIFFs  "
          f"({entries[0][0].strftime('%Y-%m')} – "
          f"{entries[-1][0].strftime('%Y-%m')})")
    return entries


# ─── 2. Load all frames into a (T, H, W, 1) float32 array ────────────────────

def load_frames(entries: list[tuple[pd.Timestamp, str]]) -> tuple[np.ndarray, dict]:
    """
    Load every TIFF into a stacked array.
    NoData (-inf, nan, negative) is replaced with 0.0.
    Returns:
        frames     : (T, H, W, 1) float32
        geo_meta   : dict with rasterio transform, CRS for later GeoTIFF export
    """
    # Read shape and geo info from first TIFF
    with rasterio.open(entries[0][1]) as src:
        H, W   = src.height, src.width
        # Store transform as a flat list (Affine is not JSON-serialisable)
        transform = list(src.transform)
        crs       = src.crs.to_string() if src.crs else "EPSG:4326"

    print(f"[INFO] Grid: {H} × {W} pixels  |  CRS: {crs}")

    T      = len(entries)
    frames = np.zeros((T, H, W, 1), dtype=np.float32)

    for idx, (date, path) in enumerate(entries):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)          # (H, W)

        # Replace nodata (-inf, +inf, NaN, negatives) with 0
        arr = np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)

        frames[idx, :, :, 0] = arr

    vmin = frames.min()
    vmax = frames.max()
    print(f"[INFO] Loaded {T} frames  |  value range: [{vmin:.4f}, {vmax:.4f}]")

    geo_meta = {
        "transform": transform,
        "crs": crs,
        "H": int(H),
        "W": int(W),
    }
    return frames, geo_meta


# ─── 3. Fit scaler on train pixels only (no leakage) ─────────────────────────

def fit_scaler(frames: np.ndarray,
               train_end_idx: int) -> MinMaxScaler:
    """
    Fit MinMaxScaler on all pixels from train frames flattened to 1-D.
    train_end_idx: the last index (exclusive) that belongs to train.
    """
    train_pixels = frames[:train_end_idx].flatten().reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_pixels)
    print(f"[INFO] Scaler fit on train pixels  "
          f"(data_min={scaler.data_min_[0]:.4f}, "
          f"data_max={scaler.data_max_[0]:.4f})")
    return scaler


def apply_scaler(frames: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Scale a (T, H, W, 1) array using a fitted scaler."""
    T, H, W, C = frames.shape
    flat        = frames.reshape(-1, 1)
    scaled_flat = scaler.transform(flat)
    return scaled_flat.reshape(T, H, W, C).astype(np.float32)


# ─── 4. Build sliding-window sequences ───────────────────────────────────────

def make_sequences(frames: np.ndarray,
                   window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of width W over a (T, H, W_img, 1) array.

    X[i] = frames[i : i+window_size]     shape (window_size, H, W_img, 1)
    y[i] = frames[i + window_size]        shape (H, W_img, 1)
    """
    T = frames.shape[0]
    X, y = [], []
    for i in range(T - window_size):
        X.append(frames[i : i + window_size])   # (W, H, W_img, 1)
        y.append(frames[i + window_size])        # (H, W_img, 1)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


def make_test_sequences(train_frames_scaled: np.ndarray,
                        test_frames_scaled: np.ndarray,
                        window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build test sequences using a rolling window.
    Seed: last W frames of train.
    For each test step i (0..23):
        X[i] = seed[-W+i:] + test[:i]    (always W frames)
        y[i] = test[i]
    This ensures X_test.shape == (24, W, H, W_img, 1).
    """
    seed   = train_frames_scaled[-window_size:]   # (W, H, W_img, 1)
    pool   = np.concatenate([seed, test_frames_scaled], axis=0)  # (W+24, ...)
    T_test = test_frames_scaled.shape[0]

    X, y = [], []
    for i in range(T_test):
        X.append(pool[i : i + window_size])
        y.append(test_frames_scaled[i])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


# ─── 5. Save outputs ──────────────────────────────────────────────────────────

def save_npz(X_train, y_train, X_test, y_test, window_size: int) -> None:
    np.savez_compressed(
        FRAMES_NPZ,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        window_size=np.array([window_size]),   # scalar saved as 1-element array
    )
    size_mb = os.path.getsize(FRAMES_NPZ) / 1024 / 1024
    print(f"[INFO] Saved {FRAMES_NPZ}  ({size_mb:.2f} MB)")


def save_metadata(entries: list[tuple[pd.Timestamp, str]],
                  geo_meta: dict,
                  train_end_idx: int,
                  window_size: int) -> None:
    dates = [d.strftime("%Y-%m-%d") for d, _ in entries]
    meta  = {
        "dates":        dates,
        "n_frames":     len(entries),
        "H":            geo_meta["H"],
        "W":            geo_meta["W"],
        "transform":    geo_meta["transform"],
        "crs":          geo_meta["crs"],
        "train_end":    entries[train_end_idx - 1][0].strftime("%Y-%m-%d"),
        "test_start":   entries[train_end_idx][0].strftime("%Y-%m-%d"),
        "window_size":  window_size,
        "train_frames": train_end_idx,
        "test_frames":  len(entries) - train_end_idx,
    }
    with open(METADATA_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved {METADATA_JSON}")


# ─── 6. Summary ──────────────────────────────────────────────────────────────

def print_summary(X_train, y_train, X_test, y_test, scaler) -> None:
    print("\n" + "=" * 60)
    print("  CONVLSTM FRAME PREPARATION — SUMMARY")
    print("=" * 60)
    print(f"  X_train : {X_train.shape}   float32")
    print(f"  y_train : {y_train.shape}   float32")
    print(f"  X_test  : {X_test.shape}    float32")
    print(f"  y_test  : {y_test.shape}    float32")
    print(f"  Scaler  : [{scaler.data_min_[0]:.4f}, {scaler.data_max_[0]:.4f}] → [0, 1]")
    total_mb = (
        X_train.nbytes + y_train.nbytes +
        X_test.nbytes  + y_test.nbytes
    ) / 1024 / 1024
    print(f"  Total array memory : {total_mb:.2f} MB")
    print("=" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = ARGS.window

    print(f"\n[ConvLSTM] Step 1 — prepare_frames.py  (city={CITY}, window={W})\n")

    # 1. Find TIFFs
    entries = find_tiffs()

    # 2. Determine train/test split index
    train_end_idx = sum(1 for d, _ in entries if d <= TRAIN_END)
    test_start_idx = train_end_idx
    n_train = train_end_idx
    n_test  = len(entries) - train_end_idx
    print(f"[INFO] Split: train={n_train} frames  |  test={n_test} frames  "
          f"(boundary: {TRAIN_END.strftime('%Y-%m')} / {TEST_START.strftime('%Y-%m')})")

    if n_train < W + 1:
        sys.exit(f"[ERROR] Not enough train frames ({n_train}) for window size {W}")
    if n_test < 1:
        sys.exit(f"[ERROR] No test frames found after {TRAIN_END.strftime('%Y-%m')}")

    # 3. Load raw frames
    frames, geo_meta = load_frames(entries)

    # 4. Fit scaler on train only
    scaler = fit_scaler(frames, train_end_idx)
    joblib.dump(scaler, SCALER_PKL)
    print(f"[INFO] Saved {SCALER_PKL}")

    # 5. Scale all frames
    frames_scaled = apply_scaler(frames, scaler)

    train_scaled = frames_scaled[:train_end_idx]   # (120, H, W, 1)
    test_scaled  = frames_scaled[train_end_idx:]   # ( 24, H, W, 1)

    # 6. Build sequences
    X_train, y_train = make_sequences(train_scaled, W)
    X_test,  y_test  = make_test_sequences(train_scaled, test_scaled, W)

    print(f"[INFO] Train sequences : {X_train.shape[0]}  "
          f"(X: {X_train.shape}, y: {y_train.shape})")
    print(f"[INFO] Test  sequences : {X_test.shape[0]}   "
          f"(X: {X_test.shape}, y: {y_test.shape})")

    # 7. Save
    save_npz(X_train, y_train, X_test, y_test, W)
    save_metadata(entries, geo_meta, train_end_idx, W)

    print_summary(X_train, y_train, X_test, y_test, scaler)
    print("[ConvLSTM] Step 1 complete. Next: python models/convlstm/build_convlstm.py\n")


if __name__ == "__main__":
    main()
