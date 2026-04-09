# Kolkata NTL Multi-City Extension — Detailed Plan

**Objective:** Replicate the full VIIRS NTL analysis pipeline (data download → SARIMA → LSTM → ConvLSTM → dashboard) for Kolkata, and integrate a unified multi-city comparison layer into the existing Streamlit dashboard.

**Current date:** 9 April 2026  
**Repo:** `iamabhinav-dev/BTP-2`

---

## 1. Architecture Decision

Rather than duplicating all scripts, every script gains a `--city` flag backed by a central `src/cities.py` config. Path resolution becomes:

```
data/{city}/tiffs/ntl_YYYY_MM.tif
outputs/{city}/sarima/
outputs/{city}/lstm/
outputs/{city}/convlstm/
models/convlstm/{city}_frames.npz
models/convlstm/{city}_frame_scaler.pkl
models/convlstm/{city}_frame_metadata.json
```

`kharagpur` is treated as the default city so all existing callers continue to work unchanged.

---

## 2. City Configuration — `src/cities.py`

Create `src/cities.py` as the **single source of truth** for all city parameters:

```python
CITIES = {
    "kharagpur": {
        "display_name": "Kharagpur",
        "bbox":  [87.25, 22.30, 87.45, 22.45],   # [min_lon, min_lat, max_lon, max_lat]
        "center": [22.375, 87.35],                 # [lat, lon] for map centering
        "state": "West Bengal",
        "description": "IIT Kharagpur region — small city, ~35×45 pixel grid",
        "scale": 500,
        "color": "#4C72B0",                        # chart colour
    },
    "kolkata": {
        "display_name": "Kolkata",
        "bbox":  [88.20, 22.40, 88.55, 22.75],
        "center": [22.575, 88.375],
        "state": "West Bengal",
        "description": "Kolkata Metropolitan Area — major metro, ~70×70 pixel grid",
        "scale": 500,
        "color": "#DD8452",
        "color": "#DD8452",
    },
}

def get_city(city: str) -> dict:
    if city not in CITIES:
        raise ValueError(f"Unknown city '{city}'. Available: {list(CITIES.keys())}")
    return CITIES[city]
```

---

## 3. Phase 0 — Infrastructure Changes

### 3.1 `src/cities.py`
- New file (see §2 above)

### 3.2 `src/download_data.py` — add `--city` flag
**Current:** hardcoded `KHARAGPUR_BBOX`, saves to `data/tiffs/`  
**Change:**
- Import `cities.get_city(city)`
- `--city` CLI argument (default `kharagpur`)
- `OUTPUT_DIR = data/{city}/tiffs/`
- All other logic unchanged

### 3.3 `src/preprocess.py` — city-aware TIFFS_DIR
**Current:** `TIFFS_DIR` hardcoded to `data/tiffs/`  
**Change:**
- `get_tiff_dir(city="kharagpur")` function returns `data/{city}/tiffs/`
- All existing functions gain optional `city="kharagpur"` parameter
- `get_tiff_path(year, month, city="kharagpur")`  
- `get_available_dates(city="kharagpur")`  
- `load_raster(year, month, city="kharagpur")`  
- **Existing callers supply no argument → behaviour unchanged**

### 3.4 All SARIMA / LSTM / ConvLSTM scripts — add `--city` flag
Each script's path block becomes:
```python
parser = argparse.ArgumentParser()
parser.add_argument("--city", default="kharagpur")
ARGS = parser.parse_args()
CITY = ARGS.city

OUTPUT_DIR = os.path.join(ROOT, "outputs", CITY, "sarima")   # or lstm / convlstm
```
Input CSV / NPZ paths similarly parameterised. No algorithmic changes.

---

## 4. Phase 1 — Data Download (Kolkata)

**Script:** `src/download_data.py --city kolkata`  
**GEE collection:** `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`, band `avg_rad`  
**Bounding box:** `[88.20, 22.40, 88.55, 22.75]`  
**Period:** Jan 2014 – Dec 2025 → **144 GeoTIFFs**  
**Output dir:** `data/kolkata/tiffs/`  
**Expected grid size:** ~70 × 70 pixels at 500 m resolution  
**Estimated time:** 30–60 min (GEE export, depends on quota)

### Expected output
```
data/kolkata/tiffs/
  ntl_2014_01.tif … ntl_2025_12.tif   (144 files, ~10–15 KB each)
```

---

## 5. Phase 2 — SARIMA Pipeline (Kolkata)

Run the 6 scripts in sequence, all with `--city kolkata`:

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 2.1 | `models/sarima/extract_brightness.py --city kolkata` | `data/kolkata/tiffs/*.tif` | `outputs/kolkata/sarima/mean_brightness.csv` |
| 2.2 | `models/sarima/clean_brightness.py --city kolkata` | `mean_brightness.csv` | `mean_brightness_clean.csv`, `train_test_split.csv` |
| 2.3 | `models/sarima/eda.py --city kolkata` | `mean_brightness_clean.csv` | `plots/eda_*.png` |
| 2.4 | `models/sarima/find_order.py --city kolkata` | `mean_brightness_clean.csv` | `best_order.json` |
| 2.5 | `models/sarima/train.py --city kolkata` | `mean_brightness_clean.csv`, `best_order.json` | `sarima_model.pkl`, `sarima_model_full.pkl` |
| 2.6 | `models/sarima/evaluate.py --city kolkata` | `sarima_model.pkl`, `train_test_split.csv` | `evaluation_metrics.json`, `plots/eval_*.png` |
| 2.7 | `models/sarima/forecast.py --city kolkata` | `sarima_model_full.pkl` | `forecast.csv`, `plots/forecast.png` |

**Train/Test split:** Jan 2014 – Dec 2023 train / Jan 2024 – Dec 2025 test (identical to Kharagpur).

---

## 6. Phase 3 — LSTM Pipeline (Kolkata)

Run all LSTM scripts with `--city kolkata`:

| Step | Script | Notes |
|------|--------|-------|
| 3.1 | `models/lstm/prepare_sequences.py --city kolkata` | Window=12, reads `outputs/kolkata/sarima/mean_brightness_clean.csv` |
| 3.2 | `models/lstm/tune.py --city kolkata` | Keras Tuner, 30 trials. Output: `outputs/kolkata/lstm/tuner/` |
| 3.3 | `models/lstm/train.py --city kolkata` | Best params from tuner. Output: `lstm_model.keras`, `training_history.json` |
| 3.4 | `models/lstm/evaluate.py --city kolkata` | Output: `evaluation_metrics.json`, `plots/*.png` |
| 3.5 | `models/lstm/forecast.py --city kolkata` | Output: `forecast.csv`, `plots/forecast.png` |

**Architecture** (will be re-tuned; may differ from Kharagpur):
- LSTM(units_1 → units_2) + MC Dropout
- Tuner search space same as Kharagpur

---

## 7. Phase 4 — ConvLSTM Pipeline (Kolkata)

Kolkata grid (~70×70) is 4× larger than Kharagpur (35×45), so training is heavier.

| Step | Script | Notes |
|------|--------|-------|
| 4.1 | `models/convlstm/prepare_frames.py --city kolkata` | Output: `models/convlstm/kolkata_frames.npz`, `kolkata_frame_scaler.pkl`, `kolkata_frame_metadata.json` |
| 4.2 | `models/convlstm/build_convlstm.py` | Auto-adapts to input shape from metadata. No change needed. |
| 4.3 | `models/convlstm/train_convlstm.py --city kolkata` | Output: `outputs/kolkata/convlstm/convlstm_model.keras`. **Allow ~2–3 hours** on CPU at 70×70. |
| 4.4 | `models/convlstm/evaluate_convlstm.py --city kolkata` | Output: `evaluation_metrics.json`, `plots/*.png` |
| 4.5 | `models/convlstm/forecast_convlstm.py --city kolkata` | Output: `forecast_frames.npz`, 36 GeoTIFFs, GIF |

**Grid size note:** `prepare_frames.py` will auto-detect H×W from the actual downloaded TIFFs, so no manual size configuration is needed.

**Filter adjustment:** With 70×70 spatial input, we may keep the same `(16,32,32)` encoder or try `(8,16,16)` if CPU training is too slow (check at Step 4.3).

---

## 8. Phase 5 — Dashboard Integration

This is the most visible change. The dashboard gets a **global city selector** and a new **cross-city comparison** page.

### 8.1 Global city selector in `dashboard/app.py`

Add a `st.selectbox` (or radio) in the sidebar to select the active city. Store selection in `st.session_state["city"]`. All pages read `st.session_state.get("city", "kharagpur")`.

### 8.2 Update existing pages 01–06

Each existing page will:
1. Read the city from `session_state`
2. Point all path variables to `outputs/{city}/...` and `data/{city}/tiffs/`
3. Update title: `"🌃 NTL Explorer — {city_cfg['display_name']}"`
4. Update folium map bounds and center to the selected city

**Pages requiring path changes:**
| Page | Path variables to update |
|------|--------------------------|
| `01_explorer.py` | `TIFFS_DIR` via `preprocess.get_tiff_dir(city)` |
| `02_change.py`   | `TIFFS_DIR` |
| `03_charts.py`   | `SARIMA_DIR` (for mean_brightness_clean.csv) |
| `04_forecast.py` | `SARIMA_DIR` |
| `05_lstm.py`     | `LSTM_DIR`, `SARIMA_DIR` |
| `06_convlstm.py` | `CLSTM_DIR`, `MODELS_DIR` (for frames.npz etc.) |

### 8.3 New page: `07_compare.py` — Cross-City Comparison

The most impactful addition. A single page showing both cities side-by-side.

**Sections:**

#### A. Mean Radiance Time Series (Plotly, dual-trace)
```
x-axis: date (Jan 2014 – Dec 2025)
Kharagpur trace: blue
Kolkata trace:   orange
Overlaid 2026 forecast bands for each
```

#### B. Model Accuracy Comparison Table
```
| Metric | SARIMA KGP | SARIMA KOL | LSTM KGP | LSTM KOL | ConvLSTM KGP | ConvLSTM KOL |
|--------|-----------|-----------|---------|---------|-------------|-------------|
| MAE    | ...       | ...       | ...     | ...     | ...         | ...         |
| RMSE   | ...       | ...       | ...     | ...     | ...         | ...         |
| MAPE   | ...       | ...       | ...     | ...     | ...         | ...         |
| MASE   | ...       | ...       | ...     | ...     | ...         | ...         |
```

#### C. Spatial Forecast Comparison (2026, side-by-side maps)
```
Month slider → shows mean ConvLSTM forecast map for each city
```

#### D. Year-over-Year Growth
```
Bar chart: annual mean radiance for each city, 2014–2025
Trend lines overlaid
```

---

## 9. File / Directory Structure After Completion

```
data/
  tiffs/                          ← Kharagpur TIFFs (existing)
  kolkata/
    tiffs/                        ← NEW: 144 Kolkata TIFFs

src/
  cities.py                       ← NEW
  download_data.py                ← UPDATED (--city flag)
  preprocess.py                   ← UPDATED (city param)
  utils.py                        ← unchanged

models/
  sarima/                         ← UPDATED (--city flag each script)
  lstm/                           ← UPDATED (--city flag each script)
  convlstm/
    kolkata_frames.npz            ← NEW
    kolkata_frame_scaler.pkl      ← NEW
    kolkata_frame_metadata.json   ← NEW
    prepare_frames.py             ← UPDATED (--city flag)
    build_convlstm.py             ← unchanged (adapts to input shape)
    train_convlstm.py             ← UPDATED (--city flag)
    evaluate_convlstm.py          ← UPDATED (--city flag)
    forecast_convlstm.py          ← UPDATED (--city flag)

outputs/
  sarima/                         ← Kharagpur (existing)
  lstm/                           ← Kharagpur (existing)
  convlstm/                       ← Kharagpur (existing)
  kolkata/
    sarima/                       ← NEW
    lstm/                         ← NEW
    convlstm/
      convlstm_model.keras        ← NEW
      evaluation_metrics.json     ← NEW
      forecast_frames.npz         ← NEW
      forecast_metadata.json      ← NEW
      forecast_tiffs/             ← NEW (36 GeoTIFFs)
      plots/                      ← NEW

dashboard/
  app.py                          ← UPDATED (city selector, nav row)
  pages/
    01_explorer.py                ← UPDATED (city-aware paths)
    02_change.py                  ← UPDATED
    03_charts.py                  ← UPDATED
    04_forecast.py                ← UPDATED
    05_lstm.py                    ← UPDATED
    06_convlstm.py                ← UPDATED
    07_compare.py                 ← NEW (cross-city comparison)
```

---

## 10. Execution Order (Step-by-Step)

```
Phase 0 — Infrastructure  (30 min coding)
  [0.1]  Create src/cities.py
  [0.2]  Update src/download_data.py      --city flag
  [0.3]  Update src/preprocess.py         city param
  [0.4]  Update models/sarima/*.py        --city flag (7 scripts)
  [0.5]  Update models/lstm/*.py          --city flag (6 scripts)
  [0.6]  Update models/convlstm/*.py      --city flag (4 scripts)

Phase 1 — Download  (~30–60 min runtime)
  [1.1]  python src/download_data.py --city kolkata

Phase 2 — SARIMA  (~20 min runtime)
  [2.1]  python models/sarima/extract_brightness.py  --city kolkata
  [2.2]  python models/sarima/clean_brightness.py    --city kolkata
  [2.3]  python models/sarima/eda.py                 --city kolkata
  [2.4]  python models/sarima/find_order.py          --city kolkata
  [2.5]  python models/sarima/train.py               --city kolkata
  [2.6]  python models/sarima/evaluate.py            --city kolkata
  [2.7]  python models/sarima/forecast.py            --city kolkata

Phase 3 — LSTM  (~45–60 min runtime, tuning included)
  [3.1]  python models/lstm/prepare_sequences.py  --city kolkata
  [3.2]  python models/lstm/tune.py               --city kolkata
  [3.3]  python models/lstm/train.py              --city kolkata
  [3.4]  python models/lstm/evaluate.py           --city kolkata
  [3.5]  python models/lstm/forecast.py           --city kolkata

Phase 4 — ConvLSTM  (~2–3 hours runtime)
  [4.1]  python models/convlstm/prepare_frames.py    --city kolkata
  [4.2]  python models/convlstm/train_convlstm.py    --city kolkata
  [4.3]  python models/convlstm/evaluate_convlstm.py --city kolkata
  [4.4]  python models/convlstm/forecast_convlstm.py --city kolkata

Phase 5 — Dashboard  (~45 min coding)
  [5.1]  Add global city selector to dashboard/app.py
  [5.2]  Update pages 01–06 (city-aware paths)
  [5.3]  Create dashboard/pages/07_compare.py
```

---

## 11. Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| GEE download quota exceeded (600 tiles/day limit) | Low — 144 tiles well under limit | Download in one session |
| Kolkata bbox includes Howrah/KMDA pixels with high industrial NTL → scaler outliers | Medium | Review histogram after extraction; clip 99th percentile if needed |
| ConvLSTM training too slow on 70×70 grid (CPU) | Medium | Reduce filters to `(8,16,16)` first; monitor epoch time; set patience=15 |
| SARIMA order differs significantly from Kharagpur | Low | `find_order.py` runs auto_arima independently per city — no issue |
| Existing dashboard pages break after preprocess.py edits | Low | Default `city="kharagpur"` preserves all existing call signatures |

---

## 12. Expected Kolkata Results (Hypothesis)

Kolkata is a major metropolitan area. Compared to Kharagpur we expect:

- **Higher absolute radiance** — CBD, port, industrial zones (Howrah) contribute high NTL
- **Stronger trend signal** — urban growth and expansion clearly visible 2014–2025
- **Lower MAPE** for all models — larger signal-to-noise ratio makes prediction easier
- **Higher SSIM** for ConvLSTM — more spatial structure (roads, rivers, clusters) gives the model more to learn
- **More interesting change detection** — Durga Puja lighting spike (October), COVID dip (2020), industrial seasonal cycles

This contrast is the thesis narrative: *"Small-town vs Metro — how NTL predictability scales with city size."*

---

## 13. Dashboard City Selector UX

```
sidebar:
  🌆 Select City
  ○ Kharagpur
  ● Kolkata

  (changing this rebuilds all page data via st.session_state)
```

All existing pages dynamically re-title and re-load data for the selected city. Page 07 always shows both cities simultaneously.

---

*Plan written: 9 April 2026*
