"""
models/sarima/extract_brightness.py
====================================
STEP 1 — Extract mean brightness from all monthly VIIRS NTL GeoTIFFs.

For each available ntl_YYYY_MM.tif:
  - Load the raster (nodata already masked to NaN by load_raster)
  - Compute mean_rad  : mean of all valid (non-NaN) pixel values
  - Compute valid_pixel_ratio : valid_pixels / total_pixels  (used later for cleaning)
  - Also store median_rad and std_rad for reference

Output: outputs/sarima/mean_brightness.csv

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/sarima/extract_brightness.py
"""

import os
import sys

import numpy as np
import pandas as pd

# ─── Make src/ importable regardless of cwd ──────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import preprocess as pp

# ─── Output path ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(ROOT, "outputs", "sarima")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "mean_brightness.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Main extraction ─────────────────────────────────────────────────────────

def extract_mean_brightness() -> pd.DataFrame:
    dates = pp.get_available_dates()
    total = len(dates)
    print(f"Found {total} monthly TIFFs. Starting extraction...\n")

    rows = []
    for i, (year, month) in enumerate(dates, start=1):
        tiff_path = pp.get_tiff_path(year, month)

        # ── progress indicator ───────────────────────────────────────────────
        label = f"{year}-{month:02d}"
        print(f"  [{i:>3}/{total}]  {label} ...", end="  ", flush=True)

        # ── skip if file missing ─────────────────────────────────────────────
        if not os.path.exists(tiff_path):
            print(f"MISSING — skipped")
            rows.append(_missing_row(year, month))
            continue

        try:
            raster = pp.load_raster(year, month)
            arr    = raster["array"]          # 2D float32, NaN where invalid
            stats  = pp.get_stats(arr)

            total_pixels = int(arr.size)
            valid_pixels = stats["valid_pixels"]
            valid_ratio  = valid_pixels / total_pixels if total_pixels > 0 else 0.0

            row = {
                "date":              pd.Timestamp(year=year, month=month, day=1),
                "year":              year,
                "month":             month,
                "mean_rad":          round(stats["mean"],   6),
                "median_rad":        round(stats["median"], 6),
                "std_rad":           round(stats["std"],    6),
                "min_rad":           round(stats["min"],    6),
                "max_rad":           round(stats["max"],    6),
                "valid_pixels":      valid_pixels,
                "total_pixels":      total_pixels,
                "valid_pixel_ratio": round(valid_ratio,     4),
            }
            print(f"mean={stats['mean']:.4f}  valid={valid_ratio:.2%}")

        except Exception as exc:
            print(f"ERROR — {exc}")
            row = _missing_row(year, month)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")
    return df


def _missing_row(year: int, month: int) -> dict:
    """Return a row of NaN values for a missing or corrupt TIFF."""
    return {
        "date":              pd.Timestamp(year=year, month=month, day=1),
        "year":              year,
        "month":             month,
        "mean_rad":          float("nan"),
        "median_rad":        float("nan"),
        "std_rad":           float("nan"),
        "min_rad":           float("nan"),
        "max_rad":           float("nan"),
        "valid_pixels":      0,
        "total_pixels":      0,
        "valid_pixel_ratio": float("nan"),
    }


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total rows          : {len(df)}")
    print(f"  Date range          : {df['date'].min().date()} → {df['date'].max().date()}")

    missing = df["mean_rad"].isna().sum()
    print(f"  Missing / NaN rows  : {missing}")

    low_cov = df[df["valid_pixel_ratio"] < 0.5]
    print(f"  Low coverage (<50%) : {len(low_cov)} months")
    if not low_cov.empty:
        for _, r in low_cov.iterrows():
            print(f"      {r['date'].date()}  valid_ratio={r['valid_pixel_ratio']:.2%}")

    valid = df.dropna(subset=["mean_rad"])
    print(f"\n  mean_rad  range     : {valid['mean_rad'].min():.4f} – {valid['mean_rad'].max():.4f}")
    print(f"  mean_rad  overall Ø : {valid['mean_rad'].mean():.4f}")
    print("=" * 60)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = extract_mean_brightness()
    print_summary(df)
