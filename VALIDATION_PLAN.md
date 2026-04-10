# Validation Plan — Real 2026 Data vs Model Forecasts

**Context:** Today is April 10, 2026. VIIRS VCMSLCFG data for Jan, Feb, and Mar 2026 is now
available on Google Earth Engine. All three models (SARIMA, LSTM, ConvLSTM) have already
forecast the full year 2026. This plan covers downloading the new data, computing actuals,
comparing against forecasts, and surfacing results in a new dashboard page.

---

## What We Have

| Item | Kharagpur | Kolkata |
|------|-----------|---------|
| TIFFs | `data/tiffs/` up to `ntl_2025_12.tif` | `data/kolkata/tiffs/` up to `ntl_2025_12.tif` |
| SARIMA forecast | `outputs/sarima/forecast.csv` → Jan–Dec 2026 | `outputs/kolkata/sarima/forecast.csv` → Jan–Dec 2026 |
| LSTM forecast | `outputs/lstm/forecast.csv` → Jan–Dec 2026 | `outputs/kolkata/lstm/forecast.csv` → Jan–Dec 2026 |
| ConvLSTM forecast | `outputs/convlstm/forecast_frames.npz` + `forecast_metadata.json` → 12 spatial frames | `outputs/kolkata/convlstm/forecast_frames.npz` → 12 spatial frames |

## What We Will Add

- 3 new TIFFs per city: `ntl_2026_01.tif`, `ntl_2026_02.tif`, `ntl_2026_03.tif`
- 1 new JSON per city: `outputs/{city}/validation_2026.json` — monthly actuals + per-model errors
- 1 new dashboard page: `dashboard/pages/08_validation.py`

---

## Phase 1 — Download Jan–Mar 2026 TIFFs

**Script:** Extend `src/download_data.py` — change `END_YEAR = 2026` and run for both cities.

> GEE must be authenticated locally (`earthengine authenticate` already done).

```bash
# In the venv
python src/download_data.py --city kharagpur
python src/download_data.py --city kolkata
```

This will skip all existing files and only download the 3 new months for each city.

**Expected output:**
```
data/tiffs/ntl_2026_01.tif
data/tiffs/ntl_2026_02.tif
data/tiffs/ntl_2026_03.tif
data/kolkata/tiffs/ntl_2026_01.tif
data/kolkata/tiffs/ntl_2026_02.tif
data/kolkata/tiffs/ntl_2026_03.tif
```

---

## Phase 2 — Extract Mean Brightness for 2026 Actuals

**Script:** Write `src/extract_actuals.py`

For each city and each of the 3 new TIFFs:
1. Open the TIFF with `rasterio`
2. Mask nodata/negative values
3. Compute `np.nanmean()` of the valid pixels → scalar mean radiance
4. Store as a list of dicts: `[{"date": "2026-01-01", "mean_rad": X}, ...]`

Also for ConvLSTM, read the full pixel array (H×W) to compare against the spatial forecast frame.

**Output per city:** A Python dict structure saved as JSON:
```json
{
  "city": "kharagpur",
  "months": [
    {
      "date": "2026-01-01",
      "mean_rad": 8.34,
      "pixel_grid": null
    }
  ]
}
```

> `pixel_grid` is stored separately as `.npz` files for spatial comparison (ConvLSTM).

**Specific outputs:**
```
outputs/sarima/actuals_2026.csv          (date, mean_rad)
outputs/kolkata/sarima/actuals_2026.csv
outputs/convlstm/actual_frames_2026.npz  (3, H, W) raw arrays
outputs/kolkata/convlstm/actual_frames_2026.npz
```

---

## Phase 3 — Compute Error Metrics

**Script:** `src/compute_validation.py`

For each city, compute for all 3 available months (Jan, Feb, Mar 2026):

### Scalar models (SARIMA, LSTM)

| Metric | Formula |
|--------|---------|
| MAE | $|actual - predicted|$ |
| % Error | $\frac{|actual - predicted|}{actual} \times 100$ |

Read `forecast.csv`, filter to `date in ["2026-01-01", "2026-02-01", "2026-03-01"]`,
compare against actuals.

### Spatial model (ConvLSTM)

For each of the 3 months, load:
- `actual_frames_2026.npz[i]` → actual H×W array (after same preprocessing/scaling as training)
- `forecast_frames.npz[i]` → predicted H×W array (already in original scale via scaler inverse)

Compute per-pixel MAE map and overall mean MAE.

**Output per city:**
```
outputs/{city}/validation_2026.json
```
Structure:
```json
{
  "city": "kharagpur",
  "generated": "2026-04-10",
  "months": ["2026-01", "2026-02", "2026-03"],
  "sarima": [
    {"date": "2026-01-01", "actual": 8.34, "predicted": 8.18, "mae": 0.16, "pct_error": 1.9}
  ],
  "lstm": [
    {"date": "2026-01-01", "actual": 8.34, "predicted": 8.21, "mae": 0.13, "pct_error": 1.6}
  ],
  "convlstm": [
    {"date": "2026-01-01", "actual_mean": 8.34, "predicted_mean": 8.07, "mae": 0.27, "pct_error": 3.2, "pixel_mae_npz": "outputs/convlstm/val_pixel_mae_2026_01.npz"}
  ]
}
```

Also save per-month pixel-MAE grids as `.npz` for the spatial heatmap in the dashboard.

---

## Phase 4 — Dashboard Page `08_validation.py`

**New page:** `dashboard/pages/08_validation.py`

### Layout

#### Header
- Title: "2026 Forecast Validation — Jan to Mar"
- City selector from `st.session_state`
- Short explainer: "Three months of real VIIRS data now available — how well did each model do?"

#### Section A — Scalar Accuracy Table (SARIMA + LSTM)

A side-by-side table for the 3 months:

| Month | Actual | SARIMA Pred | SARIMA Err% | LSTM Pred | LSTM Err% |
|-------|--------|-------------|-------------|-----------|-----------|
| Jan 2026 | 8.34 | 8.18 | 1.9% | 8.21 | 1.6% |
| Feb 2026 | ... | ... | ... | ... | ... |
| Mar 2026 | ... | ... | ... | ... | ... |
| **Mean** | — | — | **X%** | — | **Y%** |

Colour cells: green < 5%, yellow 5–10%, red > 10%.

#### Section B — Time Series Chart

Plotly line chart:
- Solid line: actual historical data (last 12 months context + 3 new months)
- SARIMA forecast line + CI band
- LSTM forecast line + CI band
- Highlight the 3 validation months with vertical shading

#### Section C — ConvLSTM Spatial Comparison

Three column layout, one per month (Jan / Feb / Mar 2026):
- Left: Actual spatial map (from `actual_frames_2026.npz`) — `Greys_r`
- Right: Forecast spatial map (from `forecast_frames.npz`) — `Greys_r`
- Bottom: Pixel-MAE heatmap — `Reds` colormap

Overall mean pixel-MAE shown as metric cards above each column.

#### Section D — Model Ranking Summary

Three metric cards side by side:
```
SARIMA     LSTM       ConvLSTM
X.X%       X.X%       X.X%
mean err   mean err   mean err
```
with a small "best model" badge on the lowest error model.

---

## Phase 5 — Update Existing Pages (Optional Enhancements)

### `04_forecast.py` (SARIMA)
- Add a small "Validation" expander at the bottom showing the Jan–Mar actual vs forecast table.

### `05_lstm.py` (LSTM)
- Same: add "Validation" expander with actuals vs predictions.

### `06_convlstm.py` (ConvLSTM)
- Add a "Jan–Mar 2026 Actual vs Forecast" tab to the existing spatial viewer.

### `07_compare.py` (City Comparison)
- Add a row to the accuracy table: "2026 Validation MAE" column so both cities' live accuracy is visible side by side.

---

## Phase 6 — Commit & Push to GitHub

```bash
git add data/tiffs/ntl_2026_*.tif
git add data/kolkata/tiffs/ntl_2026_*.tif
git add outputs/*/actuals_2026.csv
git add outputs/*/validation_2026.json
git add outputs/*/actual_frames_2026.npz
git add outputs/*/val_pixel_mae_2026_*.npz
git add src/extract_actuals.py
git add src/compute_validation.py
git add dashboard/pages/08_validation.py
git add dashboard/pages/04_forecast.py  # minor update
git add dashboard/pages/05_lstm.py      # minor update
git add dashboard/pages/06_convlstm.py  # minor update
git add dashboard/pages/07_compare.py   # minor update
git commit -m "Add: 2026 real-data validation — Jan-Mar actuals vs SARIMA/LSTM/ConvLSTM forecasts"
git push origin main
```

---

## Execution Order

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6
Download     Extract      Compute      New page     Update        Push
TIFFs        actuals      errors       08_valid     existing
(GEE)        (rasterio)   (Python)     (Streamlit)  pages
```

Phases 1–3 are pure Python scripts, runnable locally with the venv activated.
Phases 4–5 are dashboard code, no GEE needed, deployable to Streamlit Cloud.

---

## File Changes Summary

| File | Action |
|------|--------|
| `src/download_data.py` | Change `END_YEAR = 2025` → `END_YEAR = 2026` |
| `src/extract_actuals.py` | **NEW** — rasterio loop, saves actuals CSV + .npz |
| `src/compute_validation.py` | **NEW** — loads forecast CSVs + actuals, writes validation JSON |
| `dashboard/pages/08_validation.py` | **NEW** — full validation dashboard page |
| `dashboard/pages/04_forecast.py` | Minor update — add validation expander |
| `dashboard/pages/05_lstm.py` | Minor update — add validation expander |
| `dashboard/pages/06_convlstm.py` | Minor update — add actual-vs-forecast spatial tab |
| `dashboard/pages/07_compare.py` | Minor update — add validation row to accuracy table |
| `data/tiffs/ntl_2026_0{1,2,3}.tif` | **NEW** — 3 Kharagpur TIFFs |
| `data/kolkata/tiffs/ntl_2026_0{1,2,3}.tif` | **NEW** — 3 Kolkata TIFFs |
| `outputs/{city}/actuals_2026.csv` | **NEW** — mean radiance for Jan-Mar 2026 |
| `outputs/{city}/validation_2026.json` | **NEW** — full per-model error breakdown |
| `outputs/{city}/convlstm/actual_frames_2026.npz` | **NEW** — spatial actual arrays |
| `outputs/{city}/convlstm/val_pixel_mae_2026_*.npz` | **NEW** — pixel-level MAE grids |
