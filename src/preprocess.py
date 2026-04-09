"""
preprocess.py
Handles all raster loading, masking, stats, and time series building.
All heavy functions are designed to be wrapped with @st.cache_data in the app.

City support
------------
All path-dependent functions accept an optional ``city`` keyword (default
``"kharagpur"``).  Existing call-sites that supply no argument continue to
work without modification; the legacy ``data/tiffs/`` directory is used for
Kharagpur.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds

# ─── Paths ───────────────────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC      = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)

# Legacy module-level constant — kept so any direct import of TIFFS_DIR
# still resolves to the Kharagpur directory without change.
TIFFS_DIR = os.path.join(_ROOT, "data", "tiffs")


def get_tiff_dir(city: str = "kharagpur") -> str:
    """
    Return the GeoTIFF directory for *city*.

    Kharagpur uses the legacy ``data/tiffs/`` path so that existing data
    does not need to be moved.  Other cities use ``data/{city}/tiffs/``.
    """
    import cities as _cities
    return _cities.get_tiff_dir(city, _ROOT)

# VIIRS avg_rad fill / nodata threshold — values above this are sensor fill
NODATA_THRESHOLD = 1000.0

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


# ─── Date discovery ──────────────────────────────────────────────────────────

def get_available_dates(city: str = "kharagpur") -> list[tuple[int, int]]:
    """Return sorted list of (year, month) tuples for all downloaded tifs."""
    tiffs_dir = get_tiff_dir(city)
    pattern = re.compile(r"ntl_(\d{4})_(\d{2})\.tif$")
    dates = []
    for fname in os.listdir(tiffs_dir):
        m = pattern.match(fname)
        if m:
            dates.append((int(m.group(1)), int(m.group(2))))
    return sorted(dates)


def date_to_index(dates: list[tuple[int, int]], year: int, month: int) -> int:
    return dates.index((year, month))


def index_to_label(dates: list[tuple[int, int]], idx: int) -> str:
    y, m = dates[idx]
    return f"{MONTH_NAMES[m - 1]} {y}"


# ─── Raster loading ──────────────────────────────────────────────────────────

def get_tiff_path(year: int, month: int, city: str = "kharagpur") -> str:
    return os.path.join(get_tiff_dir(city), f"ntl_{year}_{month:02d}.tif")


def load_raster(year: int, month: int, city: str = "kharagpur") -> dict:
    """
    Load a single monthly NTL GeoTIFF.
    Returns dict with:
        array     : 2D float32 numpy array (nodata → np.nan)
        bounds    : (west, south, east, north)
        transform : rasterio Affine transform
        crs       : CRS string
        shape     : (height, width)
    """
    path = get_tiff_path(year, month, city)
    with rasterio.open(path) as src:
        raw = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = str(src.crs)
        bounds = src.bounds  # BoundingBox(left, bottom, right, top)

    # Mask nodata / fill values
    arr = raw.copy()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr > NODATA_THRESHOLD] = np.nan
    arr[arr < 0] = np.nan

    return {
        "array": arr,
        "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
        "transform": transform,
        "crs": crs,
        "shape": arr.shape,
    }


# ─── Statistics ──────────────────────────────────────────────────────────────

def get_stats(array: np.ndarray) -> dict:
    """Compute basic stats on a masked (nan) array."""
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return {"min": 0, "max": 0, "mean": 0, "std": 0,
                "median": 0, "valid_pixels": 0, "total_pixels": array.size}
    return {
        "min":          float(np.nanmin(valid)),
        "max":          float(np.nanmax(valid)),
        "mean":         float(np.nanmean(valid)),
        "median":       float(np.nanmedian(valid)),
        "std":          float(np.nanstd(valid)),
        "valid_pixels": int(valid.size),
        "total_pixels": int(array.size),
    }


def get_lit_area_km2(array: np.ndarray, threshold: float = 0.5) -> float:
    """
    Estimate lit area in km² — pixels with radiance > threshold.
    VIIRS DNB at 500m → each pixel ≈ 0.25 km².
    """
    lit_pixels = np.sum(array[np.isfinite(array)] > threshold)
    return float(lit_pixels * 0.25)


def pct_change_from_baseline(current_mean: float, baseline_mean: float) -> float:
    if baseline_mean == 0:
        return 0.0
    return ((current_mean - baseline_mean) / baseline_mean) * 100.0


# ─── Difference map ──────────────────────────────────────────────────────────

def compute_difference(arr1: np.ndarray, arr2: np.ndarray) -> dict:
    """
    Compute arr2 - arr1 (T2 minus T1).
    Both arrays must have the same shape.
    Returns diff array and stats.
    """
    diff = arr2 - arr1
    valid = diff[np.isfinite(diff)]
    stats = {
        "min":      float(np.nanmin(valid)) if valid.size else 0,
        "max":      float(np.nanmax(valid)) if valid.size else 0,
        "mean":     float(np.nanmean(valid)) if valid.size else 0,
        "increased": int(np.sum(valid > 0.1)),
        "decreased": int(np.sum(valid < -0.1)),
        "unchanged": int(np.sum(np.abs(valid) <= 0.1)),
    }
    return {"diff": diff, "stats": stats}


# ─── Time series ─────────────────────────────────────────────────────────────

def build_timeseries_df(dates: list[tuple[int, int]] = None,
                        city: str = "kharagpur") -> pd.DataFrame:
    """
    Build a DataFrame with monthly mean/max/min radiance for all available dates.
    This is slow the first time — cache it with @st.cache_data in the app.
    """
    if dates is None:
        dates = get_available_dates(city)

    rows = []
    for year, month in dates:
        try:
            raster = load_raster(year, month, city)
            stats = get_stats(raster["array"])
            rows.append({
                "date":       pd.Timestamp(year=year, month=month, day=1),
                "year":       year,
                "month":      month,
                "month_name": MONTH_NAMES[month - 1],
                "mean_rad":   stats["mean"],
                "max_rad":    stats["max"],
                "min_rad":    stats["min"],
                "median_rad": stats["median"],
                "std_rad":    stats["std"],
                "lit_area_km2": get_lit_area_km2(raster["array"]),
            })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        # Add % change from 2014 baseline (mean of 2014)
        baseline = df[df["year"] == df["year"].min()]["mean_rad"].mean()
        df["pct_change"] = df["mean_rad"].apply(
            lambda x: pct_change_from_baseline(x, baseline)
        )
    return df
