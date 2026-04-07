# SARIMA Prediction Model — Plan
### VIIRS Night-Time Light (NTL) Forecasting · Kharagpur

---

## 0. Overview

**Goal:** Predict future monthly mean NTL radiance (avg_rad) for Kharagpur using a
SARIMA (Seasonal AutoRegressive Integrated Moving Average) model.

**Why SARIMA first?**
- Lightweight: runs without GPU, no heavy ML framework needed.
- Interpretable: explicit trend + seasonal components.
- Fits well to monthly time-series data with yearly seasonality (m = 12).
- Serves as the **baseline model** that any future deep-learning approach must beat.

**Input data already available:** 144 monthly GeoTIFFs (`ntl_YYYY_MM.tif`, Jan 2014 – Dec 2025).

**Output:** A single scalar per month → `mean_brightness` (mean of all valid pixels),
then a SARIMA model trained on that series to forecast N months ahead.

---

## 1. Project Structure After This Plan

```
BTP/
├── data/
│   └── tiffs/                      # 144 GeoTIFFs (already downloaded)
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   └── utils.py
├── models/
│   ├── sarima/
│   │   ├── train.py                # ← NEW: end-to-end training script
│   │   ├── evaluate.py             # ← NEW: evaluate on test split
│   │   └── forecast.py             # ← NEW: load saved model, predict N months
├── notebooks/
│   └── 01_sarima_exploration.ipynb # ← NEW: EDA + ACF/PACF analysis
├── outputs/
│   └── sarima/
│       ├── mean_brightness.csv     # ← NEW: extracted time series
│       ├── sarima_model.pkl        # ← NEW: saved fitted model
│       ├── forecast.csv            # ← NEW: forecast results
│       └── plots/                  # ← NEW: diagnostic plots
├── DASHBOARD_PLAN.md
├── SARIMA_PLAN.md                  # ← this file
└── requirements.txt
```

---

## 2. Dependencies

Only lightweight, CPU-friendly packages are needed.

```
# Add to requirements.txt
statsmodels>=0.14.0      # SARIMA/SARIMAX implementation
pmdarima>=2.0.0          # auto_arima for automatic parameter selection
scikit-learn>=1.3.0      # metrics (MAE, RMSE)
matplotlib>=3.7.0        # diagnostic plots
seaborn>=0.12.0          # (optional) nicer plots
joblib>=1.3.0            # save/load fitted model
```

Install command:

```bash
pip install statsmodels pmdarima scikit-learn matplotlib seaborn joblib
```

> All of the above run on CPU only and are fast on a standard laptop.

---

## 3. Step-by-Step Pipeline

```
GeoTIFFs (144 files)
      │
      ▼
[STEP 1] Extract mean brightness → mean_brightness.csv
      │
      ▼
[STEP 2] Clean & validate the time series
      │
      ▼
[STEP 3] EDA + stationarity checks (notebook)
      │
      ▼
[STEP 4] Find SARIMA (p,d,q)(P,D,Q,12) parameters
      │
      ▼
[STEP 5] Train / test split & fit model
      │
      ▼
[STEP 6] Evaluate (MAE, RMSE, MASE, plot)
      │
      ▼
[STEP 7] Save model, generate forecast CSV
      │
      ▼
[STEP 8] (Optional) Plug forecast into Streamlit dashboard
```

---

## 4. Detailed Steps

---

### STEP 1 — Extract Mean Brightness

**Script:** `models/sarima/train.py` (extraction section) or a standalone helper.

**What to do:**
- Loop over all 144 TIFFs using the existing `preprocess.load_raster()`.
- For each raster get the **mean of all valid (non-NaN) pixels** → `mean_rad`.
- Also record `valid_pixel_ratio` = valid_pixels / total_pixels (needed for cleaning).
- Save to `outputs/sarima/mean_brightness.csv`.

**Output CSV schema:**

| date       | year | month | mean_rad | valid_pixel_ratio |
|------------|------|-------|----------|-------------------|
| 2014-01-01 | 2014 | 1     | 3.47     | 0.92              |
| …          | …    | …     | …        | …                 |

**Why mean and not sum/median?**
- Mean brightness is scale-invariant (insensitive to the exact bounding box size).
- It is the standard metric for NTL urbanisation studies.
- Median could also be computed but mean is used in the SARIMA series.

---

### STEP 2 — Data Cleaning

This is critical because VIIRS composites can have residual issues:

#### 2a. Invalid / low-coverage months

- Flag any month where `valid_pixel_ratio < 0.5` (more than 50 % of pixels are NaN).
  - These likely suffered from extreme cloud cover or sensor dropout.
  - **Action:** Mark as `NaN` in the series, interpolate linearly.

#### 2b. Outlier detection (radiance spikes)

- VIIRS VCMSLCFG is cloud-free but can still contain:
  - **Stray light artefacts** in high-latitude months (not relevant for Kharagpur).
  - **Festival / event lighting** (Diwali, Dec–Jan): these are **real** signals, keep them.
  - **Sensor calibration jumps**: can appear as sudden step-changes.
- Use **IQR method** on the full series:
  - Flag values outside `[Q1 − 3·IQR, Q3 + 3·IQR]` (use 3× not 1.5× to be conservative).
  - Review manually; replace with linear interpolation only if clearly non-physical.

#### 2c. Missing months

- Check for gaps in the date sequence.
- If any TIFF is missing, insert a `NaN` row and interpolate.

#### 2d. Smoothing (optional, keep original too)

- Apply a **3-month centred moving average** as an additional column (`mean_rad_smooth`).
- SARIMA is fitted on the **raw** `mean_rad` (not smoothed), but the smoothed version is
  useful for visualisation.

---

### STEP 3 — EDA & Stationarity (Notebook)

**File:** `notebooks/01_sarima_exploration.ipynb`

#### Plots to generate:

1. **Full time-series line plot** — raw mean_rad over time (2014–2025).
2. **Monthly seasonality box plot** — box per calendar month to visualise seasonal pattern.
3. **Year-over-year overlay** — each year as a separate line.
4. **Rolling mean & std** — 12-month rolling window to visually check trend/stationarity.
5. **ACF (Autocorrelation Function)** — lags 0–48.
6. **PACF (Partial ACF)** — lags 0–48.

#### Stationarity tests:

| Test | Library | What it checks |
|------|---------|----------------|
| **ADF (Augmented Dickey-Fuller)** | `statsmodels.tsa.stattools.adfuller` | Unit root (non-stationarity) |
| **KPSS** | `statsmodels.tsa.stattools.kpss` | Stationarity directly |

- If ADF p-value > 0.05 → series is non-stationary → apply first-order differencing `d=1`.
- If strong seasonality remains after differencing → apply seasonal differencing `D=1`.

---

### STEP 4 — Parameter Selection for SARIMA(p,d,q)(P,D,Q,12)

SARIMA has 7 hyperparameters: `(p, d, q) × (P, D, Q, m)` where `m = 12` for monthly data.

#### Option A — Automatic (recommended for first run)

Use **`pmdarima.auto_arima`**:
```python
import pmdarima as pm

model = pm.auto_arima(
    train_series,
    seasonal=True,
    m=12,
    d=None,           # auto-detect
    D=None,           # auto-detect
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    information_criterion='aic',
    stepwise=True,    # faster than exhaustive search
    trace=True,       # prints progress
    error_action='ignore',
    suppress_warnings=True,
    n_jobs=1          # keep single-core so it runs well on any PC
)
```

`auto_arima` with `stepwise=True` is fast — typically finishes in < 2 minutes even on
a laptop.

#### Option B — Manual (guided by ACF/PACF)

Read from the plots in Step 3:
- **p**: number of significant spikes in PACF before cut-off.
- **q**: number of significant spikes in ACF before cut-off.
- **d**: 1 if series is non-stationary (from ADF).
- **P, Q**: same logic applied to the seasonal lags (multiples of 12).
- **D**: 1 if seasonal pattern persists after seasonal differencing.

Typical starting point for NTL monthly data: **SARIMA(1,1,1)(1,1,1,12)**.

---

### STEP 5 — Train / Test Split & Fitting

#### Split strategy:

| Set | Date range | Size |
|-----|-----------|------|
| **Train** | Jan 2014 – Dec 2023 | 120 months (10 years) |
| **Test** | Jan 2024 – Dec 2025 | 24 months (2 years) |

> Rationale: keep the last 2 full years as out-of-sample evaluation.
> This is a **time-series split** — never shuffle; always keep temporal order.

#### Fitting with `statsmodels`:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# order from auto_arima or manual selection
order         = (p, d, q)
seasonal_order = (P, D, Q, 12)

model = SARIMAX(
    train_series,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
result = model.fit(disp=False)   # disp=False suppresses convergence log noise
print(result.summary())
```

> `SARIMAX` = `SARIMA` + optional exogenous variables (we use none for now).
> Memory footprint is tiny: a 120-point series with these orders fits in < 10 MB.

---

### STEP 6 — Evaluation

#### Metrics:

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| **MAE** | $\frac{1}{n}\sum |y - \hat{y}|$ | Same units as radiance; interpretable |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y-\hat{y})^2}$ | Penalises large errors |
| **MAPE** | $\frac{100}{n}\sum \frac{|y-\hat{y}|}{y}$ | % error, scale-free |
| **MASE** | MAE / MAE(naïve) | Compares against seasonal naïve baseline |

#### Diagnostic plots:

1. **Forecast vs Actual** — predicted test values overlaid on true values.
2. **Residual plot** — check for patterns (should be white noise).
3. **Residual ACF** — spikes should fall within confidence bands.
4. **Q-Q plot** — normality of residuals.

All of these are available directly from `result.plot_diagnostics()`.

---

### STEP 7 — Save Model & Generate Forecast

#### Save fitted model:
```python
import joblib
joblib.dump(result, "outputs/sarima/sarima_model.pkl")
```

#### Load and forecast:
```python
result = joblib.load("outputs/sarima/sarima_model.pkl")

# Forecast 12 months ahead (2026)
forecast = result.get_forecast(steps=12)
forecast_df = forecast.summary_frame(alpha=0.05)  # includes 95% CI
forecast_df.to_csv("outputs/sarima/forecast.csv")
```

**Forecast CSV schema:**

| date | mean_forecast | mean_ci_lower | mean_ci_upper |
|------|--------------|---------------|---------------|
| 2026-01-01 | 4.12 | 3.85 | 4.39 |
| … | … | … | … |

---

### STEP 8 — Dashboard Integration (Optional Next Step)

Once the model and forecast CSV exist, they can be linked into the Streamlit dashboard:

- Add a new page **`dashboard/pages/04_forecast.py`**:
  - Load `mean_brightness.csv` + `forecast.csv`.
  - Show a Plotly line chart: historical mean_rad + forecast + 95% CI shaded band.
  - Allow the user to choose forecast horizon (6 / 12 / 24 months) and re-run on demand.

---

## 5. PC Performance Considerations

| Concern | Solution |
|---------|---------|
| Extracting 144 TIFFs is slow | Cache result to `mean_brightness.csv` — only runs once |
| `auto_arima` can be slow | Use `stepwise=True`, `n_jobs=1`, limit `max_p=3, max_P=2` |
| Fitting SARIMAX | `disp=False`, model is tiny (~120 points) → < 5 seconds |
| Forecast generation | Instantaneous once model is fitted |
| RAM usage | Entire pipeline < 500 MB RAM comfortably |
| Notebook EDA | All plots use `matplotlib` / `seaborn` — no GPU needed |

Total first-run time estimate on a mid-range laptop:
- TIFF extraction: ~30–60 seconds (I/O bound)
- auto_arima: ~1–3 minutes
- Model fit + evaluation: < 10 seconds
- All subsequent runs: near-instant (cached CSV + saved model)

---

## 6. File-by-File Implementation Order

| # | File | What to implement |
|---|------|------------------|
| 1 | `models/sarima/train.py` | Extract mean_rad → clean → fit SARIMA → save model |
| 2 | `notebooks/01_sarima_exploration.ipynb` | EDA plots, ACF/PACF, stationarity tests |
| 3 | `models/sarima/evaluate.py` | Load model → predict test set → compute metrics → plots |
| 4 | `models/sarima/forecast.py` | Load model → forecast N months → save CSV |
| 5 | `dashboard/pages/04_forecast.py` | Streamlit forecast page (after model is trained) |

---

## 7. Known Limitations of SARIMA (to address in future models)

| Limitation | Description |
|-----------|-------------|
| Linearity | SARIMA assumes linear relationships; NTL may have non-linear growth phases |
| Fixed seasonality | Assumes constant 12-month pattern; real seasonality can drift |
| No external regressors | Festivals (Diwali), monsoon, pandemic years not explicitly modelled |
| Univariate | Uses only mean_rad; spatial structure of pixels ignored |
| Short series | 144 points is adequate but limits very high-order seasonal models |

These limitations motivate future models: **SARIMAX** (with exogenous regressors),
**LSTM**, or **Prophet**.

---

## 8. Quick Reference: SARIMA Notation

$$\text{SARIMA}(p, d, q)(P, D, Q)_m$$

| Symbol | Meaning |
|--------|---------|
| $p$ | AR order — uses last $p$ values |
| $d$ | Differencing — removes trend |
| $q$ | MA order — uses last $q$ errors |
| $P$ | Seasonal AR order |
| $D$ | Seasonal differencing |
| $Q$ | Seasonal MA order |
| $m$ | Seasonal period (12 for monthly) |

The model equation in backshift notation:

$$\Phi_P(B^m)\,\phi_p(B)\,(1-B)^d(1-B^m)^D\,y_t = \Theta_Q(B^m)\,\theta_q(B)\,\varepsilon_t$$

where $B$ is the backshift operator and $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$.

---

*End of SARIMA Plan*
