"""
VIIRS NTL Data Downloader
Downloads VNP46A3 (monthly NTL composites) from Google Earth Engine
and saves them as GeoTIFF files locally.

Before running:
  1. Register at https://earthengine.google.com
  2. Create a GCP project and enable Earth Engine API
  3. Run: earthengine authenticate

Usage:
    python src/download_data.py                    # downloads Kharagpur (default)
    python src/download_data.py --city kolkata     # downloads Kolkata
"""

import argparse
import calendar
import os
import sys

import ee
import geemap

# ─── Make src/ importable regardless of cwd ──────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

# ─── CLI ─────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Download VIIRS NTL GeoTIFFs from GEE.")
_parser.add_argument(
    "--city", default="kharagpur",
    help="City key defined in src/cities.py  (default: kharagpur)",
)
ARGS = _parser.parse_args()
CITY = ARGS.city.lower().strip()

# ─── City-resolved CONFIG ────────────────────────────────────────────────────
_city_cfg = _cities.get_city(CITY)

# GEE settings — shared across all cities
GEE_PROJECT = "voltaic-flag-465406-q9"
COLLECTION  = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
BAND        = "avg_rad"   # average DNB radiance (nW/cm²/sr)

# Time range
START_YEAR = 2014
END_YEAR   = 2025

# City-specific values
BBOX       = _city_cfg["bbox"]   # [min_lon, min_lat, max_lon, max_lat]
SCALE      = _city_cfg["scale"]  # GEE export resolution in metres

# Output folder — kharagpur keeps legacy path; other cities use data/{city}/tiffs/
OUTPUT_DIR = _cities.get_tiff_dir(CITY, ROOT)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def authenticate():
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("GEE authenticated and initialized.")


def download_monthly_ntl():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aoi = ee.Geometry.Rectangle(BBOX)

    total = (END_YEAR - START_YEAR + 1) * 12
    count = 0

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            count += 1
            filename = os.path.join(OUTPUT_DIR, f"ntl_{year}_{month:02d}.tif")

            if os.path.exists(filename):
                print(f"[{count}/{total}] Skipping (already exists): {filename}")
                continue

            last_day = calendar.monthrange(year, month)[1]
            start_date = f"{year}-{month:02d}-01"
            end_date   = f"{year}-{month:02d}-{last_day}"

            print(f"[{count}/{total}] Downloading {year}-{month:02d} ...", end=" ", flush=True)

            try:
                image = (
                    ee.ImageCollection(COLLECTION)
                    .filterDate(start_date, end_date)
                    .filterBounds(aoi)
                    .select(BAND)
                    .median()
                    .clip(aoi)
                )

                geemap.download_ee_image(
                    image,
                    filename=filename,
                    region=aoi,
                    scale=SCALE,
                    crs="EPSG:4326",
                )
                print("Done")

            except Exception as e:
                print(f"FAILED — {e}")


if __name__ == "__main__":
    print(f"City      : {_city_cfg['display_name']}")
    print(f"BBox      : {BBOX}")
    print(f"Output dir: {OUTPUT_DIR}\n")
    authenticate()
    download_monthly_ntl()
    print(f"\nAll files saved to: {OUTPUT_DIR}")
