# 🌃 Lumina — VIIRS Night Time Light Analysis Dashboard

> **Multi-city NTL intelligence platform:** download → preprocess → forecast → validate — powered by SARIMA, LSTM, and ConvLSTM, deployed on Streamlit.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Data: VIIRS](https://img.shields.io/badge/Data-VIIRS%20VCMSLCFG-informational)](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG)

---

## Table of Contents

1. [Overview](#overview)
2. [Live Demo](#live-demo)
3. [Features](#features)
4. [Data Source](#data-source)
5. [Cities Covered](#cities-covered)
6. [Pipeline Architecture](#pipeline-architecture)
7. [Models](#models)
8. [2026 Validation Results](#2026-validation-results)
9. [Dashboard Pages](#dashboard-pages)
10. [Project Structure](#project-structure)
11. [Local Setup](#local-setup)
12. [Running the Pipeline](#running-the-pipeline)
13. [Deployment](#deployment)

---

## Overview

**Lumina** is a full end-to-end platform for analysing [VIIRS Night Time Light (NTL)](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG) data over Indian cities.  
Starting from raw Google Earth Engine exports, it builds a complete pipeline that:

- Downloads monthly NTL composites as GeoTIFFs via GEE
- Preprocesses and caches raster data for fast interactive access
- Runs three independent forecasting models (statistical → deep learning → spatial deep learning)
- Validates forecasts against real 2026 VIIRS data
- Surfaces everything in an 8-page interactive Streamlit dashboard

The project was originally developed as a Bachelor's Thesis Project (BTP) focusing on the IIT Kharagpur region, and later extended to cover the Kolkata Metropolitan Area.

---

## Live Demo

The dashboard is deployed on Streamlit Community Cloud:

> **[https://lumina-ntl.streamlit.app](https://lumina-ntl.streamlit.app)**  
> *(Select a city from the sidebar to begin)*

---

## Features

| Category | Capability |
|---|---|
| **Data** | Automated VIIRS download via Google Earth Engine for any bounding box |
| **Exploration** | Interactive monthly NTL map browser with time slider + animation |
| **Change Detection** | Two-date comparison with signed difference maps and pixel-level stats |
| **Statistical Forecasting** | SARIMA(0,1,1)(0,1,1)[12] with 95% confidence intervals |
| **Deep Learning Forecasting** | Stacked LSTM with MC Dropout uncertainty quantification |
| **Spatial Forecasting** | ConvLSTM encoder–decoder: full pixel-level 12-month spatial maps |
| **Multi-city** | City selector — all pages and all models update instantaneously |
| **Live Validation** | Jan–Mar 2026 real VIIRS data compared against all model forecasts |
| **GeoTIFF Export** | Download any forecast month as a georeferenced GeoTIFF |

---

## Data Source

| Property | Detail |
|---|---|
| **Collection** | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` |
| **Band** | `avg_rad` — average DNB radiance (nW/cm²/sr) |
| **Temporal coverage** | January 2014 → present (monthly composites) |
| **Resolution** | 500 m/pixel (GEE export scale) |
| **Access** | [Google Earth Engine](https://earthengine.google.com) — free registration required |
| **Pipeline coverage** | Jan 2014 – Mar 2026 (144 training months + 3 validation months) |

VIIRS (Visible Infrared Imaging Radiometer Suite) DNB data is cloud-filtered and stray-light corrected, making it the standard source for change-detection studies of artificial night-time illumination.

---

## Cities Covered

### Kharagpur — IIT Kharagpur Region

| Property | Value |
|---|---|
| Bounding box | 87.25°E – 87.45°E, 22.30°N – 22.45°N |
| Grid size | ~35 × 45 pixels |
| Character | Small city / campus — low baseline radiance, clear seasonal signal |

### Kolkata — Metropolitan Area

| Property | Value |
|---|---|
| Bounding box | 88.20°E – 88.55°E, 22.40°N – 22.75°N |
| Grid size | ~70 × 70 pixels |
| Character | Major metro — high radiance, complex spatial patterns |

---

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  Phase 0 — Data Acquisition                                   │
│  src/download_data.py  →  GEE API  →  data/{city}/tiffs/     │
└──────────────────────────────┬────────────────────────────────┘
                               │ ntl_YYYY_MM.tif (GeoTIFF)
┌──────────────────────────────▼────────────────────────────────┐
│  Phase 1 — Preprocessing                                      │
│  src/preprocess.py  →  mean_brightness_clean.csv              │
│  Masks nodata (>1000 nW/cm²/sr), computes nanmean per frame   │
└──────────────────────────────┬────────────────────────────────┘
                               │ scalar time series
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
│  SARIMA     │         │  LSTM       │         │  ConvLSTM   │
│  Phase 2    │         │  Phase 3    │         │  Phase 4    │
│  (statsmod) │         │  (TF/Keras) │         │  (TF/Keras) │
│  forecast.  │         │  forecast.  │         │  forecast_  │
│  csv        │         │  csv        │         │  frames.npz │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       └───────────────────────┼───────────────────────┘
                               │ all forecasts Jan-Dec 2026
┌──────────────────────────────▼────────────────────────────────┐
│  Phase 5 — Live Validation (2026)                             │
│  src/extract_actuals.py  →  actuals_2026.csv                  │
│  src/compute_validation.py  →  validation_2026.json           │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│  Phase 6 — Interactive Dashboard                              │
│  streamlit run dashboard/app.py                               │
│  8 pages covering exploration, forecasting, and validation    │
└───────────────────────────────────────────────────────────────┘
```

---

## Models

### 1. SARIMA — Statistical Baseline

**Model:** `SARIMA(0,1,1)(0,1,1)[12]`

- Seasonal ARIMA with one non-seasonal and one seasonal moving-average term
- Trained on mean monthly radiance from Jan 2014 to Dec 2023 (120 months)
- Test set: Jan–Dec 2024 (12 months); full 2026 forecast uses all 132 months

| Metric | Value |
|---|---|
| MAE | 0.838 nW/cm²/sr |
| RMSE | 1.295 nW/cm²/sr |
| MAPE | 13.75% |
| MASE | 1.260 |

**Strengths:** Interpretable trend + seasonal decomposition, no GPU required, excellent on short monthly series.

---

### 2. LSTM — Deep Learning Scalar Model

**Architecture:** Stacked LSTM (128 → 32 units) with MC Dropout

- Input: sliding window of 12 months → predict next month
- MinMaxScaler normalisation to [0, 1]
- Monte Carlo Dropout (50 forward passes at inference) → uncertainty intervals
- Trained on the same scalar series as SARIMA for a direct apples-to-apples comparison

| Metric | Value |
|---|---|
| MAE | 0.981 nW/cm²/sr |
| RMSE | 1.390 nW/cm²/sr |
| MAPE | 17.50% |
| MASE | 1.440 |

**Strengths:** Non-linear pattern capture, native uncertainty quantification via MC Dropout.

---

### 3. ConvLSTM — Spatial Deep Learning Model

**Architecture:** ConvLSTM2D encoder–decoder (full pixel-level prediction)

- Input: sequence of NTL spatial frames (H × W grids) with convLSTM layers
- Output: predicted full spatial frame for each future month
- Enables pixel-level change detection, hotspot tracking, and urban growth direction analysis
- SARIMA and LSTM only predict a scalar (mean brightness); ConvLSTM predicts the entire map

**Evaluation metrics:** SSIM, pixel-wise RMSE, mean pixel MAE  
**Unique capability:** Spatial forecasts downloadable as georeferenced GeoTIFFs

---

## 2026 Validation Results

Three months of real VIIRS data (Jan, Feb, Mar 2026) were downloaded and compared against all model forecasts. Results were generated on 2026-04-10.

### Kharagpur

| Month | Actual (nW/cm²/sr) | SARIMA Predicted | SARIMA Err% | LSTM Predicted | LSTM Err% |
|---|---|---|---|---|---|
| Jan 2026 | 9.154 | 8.182 | 10.6% | 7.617 | 16.8% |
| Feb 2026 | 9.462 | 8.486 | 10.3% | 7.390 | 21.9% |
| Mar 2026 | 8.869 | 8.058 | 9.1% | 7.648 | 13.8% |
| **Mean** | — | — | **10.0%** | — | **17.5%** |

ConvLSTM mean % error: **15.5%** &nbsp;|&nbsp; 🥇 **Best model: SARIMA**

### Kolkata

| Month | Actual (nW/cm²/sr) | SARIMA Predicted | SARIMA Err% | LSTM Predicted | LSTM Err% |
|---|---|---|---|---|---|
| Jan 2026 | 20.823 | 18.132 | 12.9% | 16.467 | 20.9% |
| Feb 2026 | 21.537 | 18.345 | 14.8% | 16.628 | 22.8% |
| Mar 2026 | 22.328 | 17.654 | 20.9% | 16.840 | 24.6% |
| **Mean** | — | — | **16.2%** | — | **22.8%** |

ConvLSTM mean % error: **29.9%** &nbsp;|&nbsp; 🥇 **Best model: SARIMA**

> SARIMA is the best-performing model on both cities for the Jan–Mar 2026 validation window.

---

## Dashboard Pages

| # | Page | Description |
|---|---|---|
| 1 | 🗺️ **NTL Explorer** | Browse every monthly NTL map with an interactive time slider; animated playback; Folium choropleth overlay |
| 2 | 🔍 **Change Detection** | Select any two months; view signed difference map, histogram of pixel changes, and summary stats |
| 3 | 📊 **Charts & Trends** | Time series, rolling mean, year-over-year overlay, seasonal decomposition, histogram |
| 4 | 🔮 **SARIMA Forecast** | Full-year 2026 forecast with 95% CI; model diagnostics; ACF/PACF plots; 2026 validation expander |
| 5 | 🧠 **LSTM Forecast** | LSTM forecast with MC Dropout CI bands; head-to-head comparison with SARIMA; 2026 validation expander |
| 6 | 🗺️ **ConvLSTM Spatial** | Pixel-level spatial forecast maps for each month; test-set evaluation; GeoTIFF download; 2026 validation expander |
| 7 | ⚖️ **City Comparison** | Side-by-side metrics for Kharagpur vs Kolkata across all three models; 2026 live validation for both cities |
| 8 | ✅ **2026 Validation** | Dedicated validation page — model ranking cards, month-by-month accuracy table, time series with actual points, ConvLSTM spatial pixel-MAE heatmaps |

---

## Project Structure

```
Lumina/
├── README.md
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python 3.12 (Streamlit Cloud pin)
│
├── src/                          # Pipeline scripts
│   ├── cities.py                 # City registry — bboxes, paths, colours
│   ├── download_data.py          # GEE downloader (VNP46A3 → GeoTIFF)
│   ├── preprocess.py             # Raster loading, masking, time-series building
│   ├── utils.py                  # Shared helpers
│   ├── extract_actuals.py        # Extract 2026 actuals from new TIFFs
│   └── compute_validation.py     # Compute per-model error metrics for 2026
│
├── models/                       # Trained model artefacts
│   ├── sarima/                   # joblib-pickled SARIMA models per city
│   ├── lstm/                     # Keras .h5 models + MinMaxScaler
│   └── convlstm/                 # Keras .h5 spatial models + scaler
│
├── outputs/                      # Forecast outputs and validation data
│   ├── sarima/                   # forecast.csv (Jan-Dec 2026)
│   ├── lstm/                     # forecast.csv (Jan-Dec 2026)
│   ├── convlstm/                 # forecast_frames.npz, actual_frames_2026.npz
│   ├── kharagpur/                # validation_2026.json, per-model CSVs
│   └── kolkata/                  # validation_2026.json, per-model CSVs
│
├── data/
│   ├── tiffs/                    # Kharagpur GeoTIFFs (ntl_YYYY_MM.tif)
│   └── kolkata/tiffs/            # Kolkata GeoTIFFs
│
├── dashboard/
│   ├── app.py                    # Streamlit entry point + city selector
│   └── pages/
│       ├── 01_explorer.py
│       ├── 02_change.py
│       ├── 03_charts.py
│       ├── 04_forecast.py
│       ├── 05_lstm.py
│       ├── 06_convlstm.py
│       ├── 07_compare.py
│       └── 08_validation.py
│
└── kaggle_upload/                # Notebook for GPU training on Kaggle
```

---

## Local Setup

### Prerequisites

- Python 3.12
- A Google Earth Engine account (for data download only)
- ~2 GB disk space for GeoTIFFs

### 1. Clone the repository

```bash
git clone https://github.com/iamabhinav-dev/Lumina.git
cd Lumina
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Authenticate Google Earth Engine (download only)

```bash
pip install earthengine-api geemap
earthengine authenticate
```

---

## Running the Pipeline

> **Note:** Pre-downloaded TIFFs and pre-computed outputs are already committed. You only need to re-run these scripts if you want to extend the dataset or re-train the models.

### Download new TIFFs

```bash
python src/download_data.py --city kharagpur
python src/download_data.py --city kolkata
# Skips months that are already downloaded
```

### Extract 2026 actuals from new TIFFs

```bash
python src/extract_actuals.py --all
# Outputs: outputs/{city}/sarima/actuals_2026.csv
#          outputs/{city}/convlstm/actual_frames_2026.npz
```

### Compute validation metrics

```bash
python src/compute_validation.py --all
# Outputs: outputs/{city}/validation_2026.json
#          outputs/{city}/convlstm/val_pixel_mae_2026_0{1,2,3}.npz
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## Deployment

The app is deployed on **Streamlit Community Cloud** directly from this repository.

Key configuration files:
- `runtime.txt` — pins Python 3.12 (required for TensorFlow wheel availability)
- `requirements.txt` — all dependencies including `tensorflow`, `scikit-learn`, `rasterio`

To redeploy after pushing changes: open the Streamlit Cloud dashboard → **Reboot app**.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Dashboard | `streamlit` |
| Maps | `folium` + `streamlit-folium` |
| Charts | `plotly` |
| Raster I/O | `rasterio` |
| Arrays | `numpy` |
| Data wrangling | `pandas` |
| Statistical model | `statsmodels` (SARIMA) |
| Deep learning | `tensorflow` / Keras (LSTM, ConvLSTM) |
| Scalers | `scikit-learn` (MinMaxScaler) |
| Image utils | `Pillow`, `matplotlib` (colormaps) |
| GEE access | `earthengine-api`, `geemap` |
| Model serialisation | `joblib` |

---

## Acknowledgements

- [NOAA VIIRS DNB Monthly V1 (VCMSLCFG)](https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG) — satellite data source
- [Google Earth Engine](https://earthengine.google.com) — geospatial computation platform
- [Streamlit](https://streamlit.io) — dashboard framework and cloud hosting
