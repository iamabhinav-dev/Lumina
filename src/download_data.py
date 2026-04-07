"""
VIIRS NTL Data Downloader
Downloads VNP46A3 (monthly NTL composites) from Google Earth Engine
and saves them as GeoTIFF files locally.

Before running:
  1. Register at https://earthengine.google.com
  2. Create a GCP project and enable Earth Engine API
  3. Run: earthengine authenticate
"""

import ee
import geemap
import os
import calendar

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Your GEE project ID (from Google Cloud Console)
GEE_PROJECT = "voltaic-flag-465406-q9"

# Kharagpur bounding box [min_lon, min_lat, max_lon, max_lat]
KHARAGPUR_BBOX = [87.25, 22.30, 87.45, 22.45]

# Time range (NOAA VIIRS DNB monthly starts January 2014)
START_YEAR = 2014
END_YEAR   = 2025

# GEE collection and band
COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
BAND       = "avg_rad"  # average DNB radiance (nW/cm²/sr)

# Output folder (project root / data / tiffs)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tiffs")

# Spatial resolution in meters (VIIRS native ~500m)
SCALE = 500

# ─── MAIN ────────────────────────────────────────────────────────────────────

def authenticate():
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
    print("GEE authenticated and initialized.")


def download_monthly_ntl():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aoi = ee.Geometry.Rectangle(KHARAGPUR_BBOX)

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
    authenticate()
    download_monthly_ntl()
    print(f"\nAll files saved to: {OUTPUT_DIR}")
