"""
src/extract_actuals.py
======================
Extract mean brightness and spatial arrays for Jan–Mar 2026 (actual observed
NTL data) for both cities. Outputs match the formats expected by compute_validation.py.

Outputs per city:
  outputs/{city}/sarima/actuals_2026.csv           — date, mean_rad
  outputs/{city}/convlstm/actual_frames_2026.npz   — (3, H, W) raw float32 arrays

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python src/extract_actuals.py
    python src/extract_actuals.py --city kolkata
    python src/extract_actuals.py --all   # both cities
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

NODATA_THRESHOLD = 1000.0

# Months to validate
VALIDATION_MONTHS = [(2026, 1), (2026, 2), (2026, 3)]


def extract_for_city(city: str) -> None:
    cfg        = _cities.get_city(city)
    tiff_dir   = _cities.get_tiff_dir(city, ROOT)
    sarima_dir = _cities.get_sarima_dir(city, ROOT)
    clstm_dir  = _cities.get_convlstm_dir(city, ROOT)

    print(f"\n{'='*60}")
    print(f"City: {cfg['display_name']}")
    print(f"TIFF dir: {tiff_dir}")

    rows        = []
    raw_frames  = []

    for year, month in VALIDATION_MONTHS:
        tiff_path = os.path.join(tiff_dir, f"ntl_{year}_{month:02d}.tif")
        date_str  = f"{year}-{month:02d}-01"

        if not os.path.exists(tiff_path):
            print(f"  [SKIP] {date_str} — file not found: {tiff_path}")
            continue

        with rasterio.open(tiff_path) as src:
            arr    = src.read(1).astype(np.float32)
            nodata = src.nodata

        # ── Mask nodata (same as preprocess.load_raster) ──────────────────
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr > NODATA_THRESHOLD] = np.nan
        arr[arr < 0]               = np.nan

        mean_rad = float(np.nanmean(arr))
        rows.append({"date": date_str, "mean_rad": mean_rad})
        print(f"  {date_str}  mean_rad = {mean_rad:.6f}")

        # ── Raw array for ConvLSTM spatial comparison ──────────────────────
        # Replace NaN with 0 (same as prepare_frames.load_frames)
        raw = np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)
        raw_frames.append(raw)

    if not rows:
        print("  [ERROR] No valid months extracted.")
        return

    # ── Save scalar CSV ────────────────────────────────────────────────────
    os.makedirs(sarima_dir, exist_ok=True)
    csv_path = os.path.join(sarima_dir, "actuals_2026.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # same file also in lstm dir for convenience
    lstm_dir = _cities.get_lstm_dir(city, ROOT)
    os.makedirs(lstm_dir, exist_ok=True)
    lstm_csv = os.path.join(lstm_dir, "actuals_2026.csv")
    pd.DataFrame(rows).to_csv(lstm_csv, index=False)
    print(f"  Saved: {lstm_csv}")

    # ── Save spatial frames NPZ ────────────────────────────────────────────
    if raw_frames:
        os.makedirs(clstm_dir, exist_ok=True)
        npz_path = os.path.join(clstm_dir, "actual_frames_2026.npz")
        stacked  = np.stack(raw_frames, axis=0)          # (N, H, W)
        dates    = [r["date"] for r in rows]
        np.savez_compressed(npz_path, frames=stacked, dates=np.array(dates))
        print(f"  Saved: {npz_path}  shape={stacked.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract Jan–Mar 2026 actuals")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--city", default="kharagpur",
                       help="City key (default: kharagpur)")
    group.add_argument("--all", action="store_true",
                       help="Process all registered cities")
    args = parser.parse_args()

    if args.all:
        import cities as _c
        for city in _c.CITIES:
            extract_for_city(city)
    else:
        extract_for_city(args.city)

    print("\nDone.")


if __name__ == "__main__":
    main()
