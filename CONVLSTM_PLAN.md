# ConvLSTM Spatial Forecast Plan
## VIIRS NTL — Kharagpur, West Bengal

**Objective:** Predict full NTL spatial maps (35×45 pixel grids) for future months using a
ConvLSTM encoder-decoder, enabling pixel-level analysis of urban growth direction, hotspot
expansion, and peri-urban electrification — a capability SARIMA and vanilla LSTM cannot provide.

---

## Context & Motivation

| Model | MAE | RMSE | MAPE | MASE | Predicts |
|---|---|---|---|---|---|
| SARIMA(0,1,1)(0,1,1)[12] | **0.838** | **1.295** | **13.75%** | **1.260** | scalar (mean) |
| LSTM(128→32) + MC Dropout | 0.981 | 1.390 | 17.50% | 1.440 | scalar (mean) |
| **ConvLSTM (proposed)** | TBD | TBD | TBD | TBD | **full 35×45 spatial frame** |

ConvLSTM operates in a different problem space — it is not a direct replacement for SARIMA/LSTM
but a **complementary spatial model**. Its value is measured by SSIM, pixel-wise RMSE, and
qualitative map quality, not just mean-radiance metrics.

---

## Data Facts

| Property | Value |
|---|---|
| Input TIFFs | 144 monthly files (`ntl_2014_01.tif` → `ntl_2025_12.tif`) |
| Grid size | **35 × 45 pixels** (all files identical shape, verified) |
| CRS | EPSG:4326 |
| Bounds | 87.249°E–87.451°E, 22.296°N–22.453°N |
| Pixel dtype | float32 |
| Value range | 0.147 → 31.63 nW/cm²/sr |
| Non-fill pixels per frame | 1,575 / 1,575 (full grid, no mask needed) |
| NoData sentinel | `-inf` (clipped to 0 during preprocessing) |

### Sequence construction (window W=12, stride=1)

```
Total frames:           144
Train frames:           120  (Jan 2014 – Dec 2023)
Test frames:             24  (Jan 2024 – Dec 2025)

Train sequences: 120 - 12 = 108   shape: (108, 12, 35, 45, 1)
Test  sequences:  24              shape: ( 24, 12, 35, 45, 1)
  └─ seeded from train[-12:] + test[:N] as rolling window

Full sequences tensor (132, 12, 35, 45, 1) float32 ≈ 9.52 MB  ← CPU-trivial
```

---

## Pipeline Overview

```
data/tiffs/                     ← 144 monthly GeoTIFFs
    │
    ▼ Step 1: prepare_frames.py
models/convlstm/
    ├── frames.npz              ← (144, 35, 45, 1) normalised tensor + scaler
    ├── frame_scaler.pkl        ← MinMaxScaler fitted on train pixels only
    ├── frame_metadata.json     ← dates, H, W, transform, CRS
    │
    ▼ Step 2: build_convlstm.py
    ├── (model factory — no file output)
    │
    ▼ Step 3: train_convlstm.py
outputs/convlstm/
    ├── convlstm_model.keras    ← trained model weights
    ├── training_history.json   ← loss / val_loss per epoch
    ├── plots/
    │   ├── loss_curve.png
    │   └── sample_predictions.png  ← 3 × (input seq | predicted | actual) grids
    │
    ▼ Step 4: evaluate_convlstm.py
    ├── evaluation_metrics.json ← SSIM, PSNR, pixel MAE, pixel RMSE (per-month)
    ├── plots/
    │   ├── error_maps.png       ← per-pixel mean absolute error heatmap
    │   ├── ssim_timeseries.png  ← SSIM score over test months
    │   └── pred_vs_actual_grid.png
    │
    ▼ Step 5: forecast_convlstm.py
    ├── forecast_frames.npz     ← (12, 35, 45, 1) mean+CI arrays
    ├── forecast_metadata.json  ← dates (Jan–Dec 2026), CI info, transform, CRS
    ├── forecast_tiffs/          ← ★ NEW: georeferenced GeoTIFFs
    │   ├── ntl_2026_01_mean.tif     mean forecast frame
    │   ├── ntl_2026_01_lower95.tif  lower CI frame
    │   ├── ntl_2026_01_upper95.tif  upper CI frame
    │   └── ... (×12 months)
    ├── plots/
    │   └── forecast_animation.gif  ← 12-frame animated forecast
    │
    ▼ Step 6: dashboard/pages/06_convlstm.py
    └── Streamlit page — animated maps, pixel inspector, diff maps, SSIM plot
        ★ Two display modes:
          [Test period]    Actual | Predicted | Difference + opacity slider
          [Forecast period] Mean forecast | Lower CI | Upper CI
```

---

## Step-by-Step Implementation

---

### Step 1 — `models/convlstm/prepare_frames.py`

**Goal:** Load all 144 TIFFs → normalise → build sliding-window sequences → save `.npz`.

**Key design decisions:**

1. **NoData handling:** Replace `-inf` with `0.0` before scaling.
2. **Normalisaton:** `MinMaxScaler` fitted **only on train pixels flattened** (no leakage).
   Apply same scaler to test. Pixel values → `[0, 1]`.
3. **Window construction (same logic as LSTM's `prepare_sequences.py`):**
   - Train: `frames[0:108]` → X, `frames[12:120]` → y, reshape to `(108, 12, 35, 45, 1)`
   - Test: seed from `frames[108:120]` (last 12 train), roll forward 24 steps
4. **Metadata:** Save date list, H, W, rasterio transform, CRS to JSON for georeference in dashboard.

**Outputs:**
- `models/convlstm/frames.npz` — keys: `X_train, y_train, X_test, y_test, window_size`
- `models/convlstm/frame_scaler.pkl` — joblib-saved MinMaxScaler
- `models/convlstm/frame_metadata.json` — `{dates, H, W, transform, crs, train_end, test_start}`

---

### Step 2 — `models/convlstm/build_convlstm.py`

**Goal:** Define the model architecture as a reusable factory function.

**Architecture — Encoder-Decoder ConvLSTM:**

```
Input: (batch, T=12, H=35, W=45, 1)
    │
    ├─ ConvLSTM2D(32, kernel=3, padding=same, return_sequences=True)
    ├─ BatchNormalization
    ├─ ConvLSTM2D(64, kernel=3, padding=same, return_sequences=True)
    ├─ BatchNormalization
    ├─ ConvLSTM2D(64, kernel=3, padding=same, return_sequences=False)
    ├─ BatchNormalization
    │
    ├─ Reshape / RepeatVector to (T_out=1, H, W, 64)
    │
    ├─ ConvLSTM2D(64, kernel=3, padding=same, return_sequences=True)
    ├─ BatchNormalization
    ├─ ConvLSTM2D(32, kernel=3, padding=same, return_sequences=True)
    ├─ BatchNormalization
    │
    └─ TimeDistributed(Conv2D(1, kernel=1, activation=sigmoid))

Output: (batch, T_out, H, W, 1)   ← T_out=1 (predict next frame)
```

**Notes:**
- `padding='same'` preserves 35×45 spatial dimensions throughout — no need for padding to power-of-2.
- `sigmoid` output consistent with `[0,1]` normalised targets.
- Loss: **MAE** (robust to outliers from monsoon dip months).
- Optimizer: Adam, lr=1e-3.
- Metric: RMSE (pixel-wise).

**Factory signature:**
```python
def build_convlstm(
    input_shape=(12, 35, 45, 1),
    filters_enc=(32, 64, 64),
    filters_dec=(64, 32),
    kernel_size=3,
    lr=1e-3,
) -> tf.keras.Model
```

**Defaults object** mirrors LSTM's `DEFAULTS` dict for consistency.

---

### Step 3 — `models/convlstm/train_convlstm.py`

**Goal:** Train model, save weights + history.

**Training protocol:**

| Setting | Value | Rationale |
|---|---|---|
| Batch size | 8 | Small dataset (108 sequences); larger batches reduce gradient noise |
| Max epochs | 500 | ConvLSTM converges slowly |
| Val split | Last 12 of 108 train samples | ~11%, matching LSTM approach |
| EarlyStopping | patience=40, restore\_best=True | Generous — ConvLSTM plateaus slowly |
| ReduceLROnPlateau | patience=15, factor=0.5, min\_lr=1e-6 | |
| ModelCheckpoint | save `convlstm_model.keras` at best val\_loss | |

**Prediction strategy (iterative 1-step):**

Rather than training a multi-step decoder (complex), train to predict **1 frame ahead**.
At forecast time, auto-regressively apply the model 12 times.
This keeps training straightforward and the architecture interpretable.

**Output:** For each training sample, target `y` is a single frame `(35, 45, 1)`.

**Sample prediction plot:** At end of training, plot 3 random test-set examples as:
```
[12 input frames (thumbnails)] → [Predicted frame] vs [Actual frame] | [Error map]
```

**Outputs:**
- `outputs/convlstm/convlstm_model.keras`
- `outputs/convlstm/training_history.json`
- `outputs/convlstm/plots/loss_curve.png`
- `outputs/convlstm/plots/sample_predictions.png`

---

### Step 4 — `models/convlstm/evaluate_convlstm.py`

**Goal:** Evaluate on the 24 test months using spatial metrics.

**Metrics (all on inverse-transformed pixel values):**

| Metric | Formula | Interpretation |
|---|---|---|
| Pixel MAE | mean\|actual − pred\| over all pixels+months | Overall spatial accuracy |
| Pixel RMSE | sqrt(mean(actual − pred)²) | Penalises large spatial errors |
| SSIM | Structural Similarity Index (skimage) | Perceptual map quality — 1=perfect |
| PSNR | 10·log₁₀(MAX²/MSE) | Signal-to-noise quality |
| Mean-radiance MAE | MAE on spatial mean per month | Comparable to SARIMA/LSTM scalar metric |

**Evaluation flow:**
1. Auto-regressively predict all 24 test frames (seed from last 12 train frames).
2. Inverse-transform to original radiance units.
3. Compute per-month SSIM, PSNR, pixel MAE.
4. Compute mean-radiance MAE/RMSE for direct SARIMA/LSTM comparison.

**Plots:**
- `error_maps.png` — heatmap of per-pixel mean absolute error over test period
- `ssim_timeseries.png` — SSIM per month Jan 2024–Dec 2025
- `pred_vs_actual_grid.png` — 4×6 grid: predicted / actual / difference for all 24 test months

**Outputs:**
- `outputs/convlstm/evaluation_metrics.json`
  ```json
  {
    "pixel_MAE": ..., "pixel_RMSE": ...,
    "mean_SSIM": ..., "mean_PSNR": ...,
    "mean_rad_MAE": ..., "mean_rad_RMSE": ...,
    "per_month": [{"date": ..., "SSIM": ..., "pixel_MAE": ...}, ...]
  }
  ```
- `outputs/convlstm/plots/` (3 plots above)

---

### Step 5 — `models/convlstm/forecast_convlstm.py`

**Goal:** Produce 12 future spatial frames (Jan–Dec 2026), plus uncertainty via MC Dropout.

**MC Dropout for spatial uncertainty:**
- Re-add `SpatialDropout2D` in the decoder path (or use standard Dropout) with `training=True`.
- Run `N=50` stochastic forward passes per auto-regressive step.
  - Memory: `50 × 35 × 45 × 1 × 4 bytes × 12 steps ≈ 4.7 MB` — trivial.
- Per-pixel CI: 2.5th / 97.5th percentile across 50 samples.

**Output files:**
- `outputs/convlstm/forecast_frames.npz`
  ```
  keys: mean_forecast, lower_95, upper_95
  shapes: each (12, 35, 45)  ← 12 future months, inverse-transformed to nW/cm²/sr
  ```
- `outputs/convlstm/forecast_metadata.json` — `{dates: ["2026-01-01", ...], H, W, transform, crs}`
- `outputs/convlstm/plots/forecast_animation.gif` — 12-frame animated GIF

**★ GeoTIFF export (NEW):**

Using the rasterio `transform` and `CRS` saved in `frame_metadata.json`, write each forecast
frame back to a properly georeferenced GeoTIFF — identical projection and extent to source TIFFs.

```python
# Pseudocode — ~15 lines using rasterio.open(..., 'w')
for i, date in enumerate(forecast_dates):
    for band_name, array in [("mean", mean[i]), ("lower95", lower[i]), ("upper95", upper[i])]:
        fname = f"ntl_{date.strftime('%Y_%m')}_{band_name}.tif"
        with rasterio.open(fname, 'w', driver='GTiff', height=H, width=W,
                           count=1, dtype='float32', crs=crs,
                           transform=transform) as dst:
            dst.write(array.astype('float32'), 1)
```

Output GeoTIFFs open directly in QGIS/ArcGIS alongside the original `data/tiffs/` files —
projection matches exactly, no manual registration needed.

- `outputs/convlstm/forecast_tiffs/ntl_2026_01_mean.tif` … × 36 files (12 months × 3 bands)

**CLI:** `python forecast_convlstm.py --horizon 12 --n_samples 50`

---

### Step 6 — `dashboard/pages/06_convlstm.py`

**Goal:** Interactive spatial dashboard page.

**Layout:**

```
Sidebar:
  - View mode toggle: "📊 Evaluate on Test Set" | "🔮 Forecast Future"
  - Month slider (adapts to selected mode)
  - Show CI uncertainty toggle  (forecast mode only)
  - Opacity slider 0–100%       (test mode diff map only)  ★ NEW
  - ▶ Re-run Forecast button (subprocess call)
  - ⬇ Download Forecast TIFFs  (zips forecast_tiffs/, streams download)  ★ NEW

Page:
  ┌─ Header metrics row ─────────────────────────────────────────────┐
  │  Pixel MAE │ Pixel RMSE │ Mean SSIM │ Mean PSNR │ Mean-rad MAE  │
  └──────────────────────────────────────────────────────────────────┘

  ══ TEST SET MODE (Jan 2024 – Dec 2025) ════════════════════════════

  ┌─ Chart 1: Animated Test Evaluation ──────────────────────────────┐
  │  Plotly Heatmap animation — 24 frames stepping through test months│
  │  Frame slider + play/pause controls                              │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ Chart 2: Month Inspector (Test) ─────────────────────────────────┐
  │  3 columns:                                                       │
  │    [Actual TIFF]  |  [ConvLSTM Predicted]  |  [Difference map]   │
  │                                                                   │
  │  Difference map has opacity slider (0–100%) to blend             │
  │  actual and predicted — shows spatial agreement visually  ★ NEW  │
  │                                                                   │
  │  Shared YlOrRd colorscale + lat/lon tick labels                  │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ Chart 3: Test-set evaluation ───────────────────────────────────┐
  │  Left: SSIM timeseries (Jan 2024–Dec 2025)                       │
  │  Right: error map heatmap (mean pixel error over test period)    │
  └──────────────────────────────────────────────────────────────────┘

  ══ FORECAST MODE (Jan 2026 – Dec 2026) ════════════════════════════

  ┌─ Chart 1: Animated Forecast Map ─────────────────────────────────┐
  │  Plotly Heatmap animation — 12 frames (Jan–Dec 2026)             │
  │  Colorscale: YlOrRd                                              │
  │  Frame slider + play/pause controls                              │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ Chart 2: Month Inspector (Forecast) ─────────────────────────────┐
  │  3 columns:                                                       │
  │    [Mean forecast] | [Lower 95% CI] | [Upper 95% CI]             │
  │  No real image exists — uncertainty maps shown instead           │
  │  Shared colorscale, lat/lon tick labels                          │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ Download ────────────────────────────────────────────────────────┐
  │  ⬇ Download Forecast GeoTIFFs (ZIP)  ★ NEW                       │
  │  Streams outputs/convlstm/forecast_tiffs/*.tif as zip            │
  │  Opens in QGIS/ArcGIS — same CRS+transform as source data       │
  └──────────────────────────────────────────────────────────────────┘

  expander: "ℹ️ Model Information"
    - Architecture table, training details, MC Dropout explanation
    - Comparison to SARIMA/LSTM with note on different problem space
    - Note: test period has ground truth; forecast period does not
```

**Update `dashboard/app.py`** nav table: add `🗺️ ConvLSTM Spatial Forecast` row.

---

### Display Mode Logic Summary

| Period | Ground truth available? | What is shown side by side |
|---|---|---|
| Test (Jan 2024 – Dec 2025) | ✅ Yes — from `data/tiffs/` | Actual \| Predicted \| Difference + opacity slider |
| Forecast (Jan 2026 – Dec 2026) | ❌ No — truly future | Mean forecast \| Lower CI \| Upper CI |

---

## Hyperparameter Tuning (Optional — between Steps 3 & 4)

If Step 3 results are poor, a lightweight manual grid search:

| Parameter | Values to try |
|---|---|
| Encoder filters | `(16,32,32)`, `(32,64,64)`, `(64,128,64)` |
| Kernel size | 3, 5 |
| Learning rate | 1e-3, 5e-4, 1e-4 |
| Batch size | 4, 8, 16 |

Given the small dataset, full Keras Tuner search is overkill — manual search over 6–8 configs is sufficient.

---

## Expected Outcomes & Thesis Narrative

### Best-case scenario
ConvLSTM achieves SSIM > 0.85 on test frames, correctly predicts spatial growth patterns,
and identifies specific zones (IIT campus, railway colony, new residential areas) brightening
year-on-year. Mean-radiance MAE competitive with LSTM (~1.0).

### Realistic scenario
SSIM ~0.70–0.80. Spatial structure is preserved but fine detail is blurry (common for
generative spatial models on small datasets). Mean-radiance MAE ≈ 1.0–1.5.
The model outperforms a naïve persistence baseline (copy last frame) on SSIM.

### Worst-case scenario
Model outputs near-constant blurry maps. This is still a valid finding:
> *"Spatial prediction of NTL at sub-city scale requires more temporal training data
> or auxiliary features (population density, land-use change) to capture fine
> spatial dynamics. Even with 144 months, spatial complexity exceeds the model's
> capacity under current data constraints."*

### Thesis contribution regardless of accuracy
- **First spatial NTL forecast for Kharagpur** using deep learning
- Demonstrates the pipeline from raw GeoTIFF → pixel-level prediction → geospatial dashboard
- Provides per-pixel uncertainty maps — novel compared to SARIMA/LSTM CIs on scalar mean
- Identifies high-uncertainty spatial zones (useful for urban planners)

---

## File Layout After Completion

```
models/
  convlstm/
    __init__.py
    prepare_frames.py       ← Step 1
    build_convlstm.py       ← Step 2
    train_convlstm.py       ← Step 3
    evaluate_convlstm.py    ← Step 4
    forecast_convlstm.py    ← Step 5

outputs/
  convlstm/
    convlstm_model.keras
    training_history.json
    evaluation_metrics.json
    forecast_frames.npz
    forecast_metadata.json
    forecast_tiffs/          ← ★ NEW — downloadable GeoTIFFs
      ntl_2026_01_mean.tif
      ntl_2026_01_lower95.tif
      ntl_2026_01_upper95.tif
      ... (×12 months = 36 files)
    plots/
      loss_curve.png
      sample_predictions.png
      error_maps.png
      ssim_timeseries.png
      pred_vs_actual_grid.png
      forecast_animation.gif

dashboard/pages/
  06_convlstm.py            ← Step 6
```

---

## Dependencies

All already installed in the venv except `scikit-image` (for SSIM/PSNR):

```bash
pip install scikit-image
```

| Package | Use |
|---|---|
| `tensorflow >= 2.21` | ConvLSTM2D, training |
| `rasterio` | GeoTIFF loading, transform metadata |
| `scikit-learn` | MinMaxScaler |
| `scikit-image` | SSIM, PSNR metrics |
| `joblib` | scaler serialisation |
| `numpy` | array ops |
| `matplotlib` | training plots, animation GIF |
| `plotly` | dashboard charts |
| `streamlit` | dashboard |

---

## Progress Tracker

- [ ] Step 1 — `prepare_frames.py` — load TIFFs, normalise, build sequences
- [ ] Step 2 — `build_convlstm.py` — architecture factory
- [ ] Step 3 — `train_convlstm.py` — train, save weights + history
- [ ] Step 4 — `evaluate_convlstm.py` — SSIM, PSNR, pixel metrics, test plots
- [ ] Step 5 — `forecast_convlstm.py` — 12-month spatial MC Dropout forecast
- [ ] Step 6 — `dashboard/pages/06_convlstm.py` — animated spatial dashboard page
