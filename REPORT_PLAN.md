# BTP-2 Report Plan — Lumina
## CS47006 (Project Part-II) | Spring 2025–2026 | IIT Kharagpur

---

## 0. Quick Reference

| Item | Value |
|---|---|
| Course | CS47006 — PROJECT PART 2 |
| Semester | Spring 2025–2026 |
| Student | Abhinav Kumar Singh (22CS30005) |
| Supervisor | Professor Soumya Kanti Ghosh |
| Title | *Lumina: Multi-Model Forecasting of Night-Time Light Data* |
| Format | LaTeX (same structure as BTP-1.tex) |
| Logo file | `reportImages/kgplogo.png` |
| Submission date | April 2026 |

---

## 1. File Structure to Create

```
report/
├── main.tex              ← single master LaTeX file
├── references.bib        ← BibTeX bibliography
└── figures/              ← symlink or copy of plots
    ├── kgplogo.png
    ├── kharagpur/
    │   ├── sarima/
    │   ├── lstm/
    │   └── convlstm/
    └── kolkata/
        ├── sarima/
        ├── lstm/
        └── convlstm/
```

---

## 2. Front Matter Pages

### 2.1 Title Page
- IIT KGP logo (`kgplogo.png`)
- "Project-II (CS47006) Report submitted to..."
- Degree: Bachelor of Technology and Master of Technology in CSE
- Student: Abhinav Kumar Singh (22CS30005)
- Supervisor: Professor Soumya Kanti Ghosh
- Title: **Lumina: Multi-Model Forecasting of Night-Time Light Data**
- Date: April 2026

### 2.2 Declaration
- Same template as BTP-1, updated date to April 2026

### 2.3 Certificate
- Signed by Prof. Soumya Kanti Ghosh
- Spring Semester 2025–26

### 2.4 Abstract (~250 words)
Key points to cover:
- VIIRS NTL data from GEE (Jan 2014 – Mar 2026, 2 cities)
- Three forecasting models: SARIMA, LSTM, ConvLSTM
- Live 2026 validation (Jan–Mar 2026 actual vs predicted)
- Streamlit dashboard (Lumina) with 8 pages
- Best model: SARIMA on both cities
- Future: extend with population/GDP for a research paper

### 2.5 Acknowledgements
- Prof. Soumya Kanti Ghosh (supervisor)
- Family and friends

### 2.6 Table of Contents (manual, like BTP-1)

---

## 3. Chapter Plan

---

### Chapter 1 — Introduction

#### 1.1 Background
- What is NTL data and why it matters
- VIIRS VCMSLCFG monthly composites (avg_rad band)
- NTL as a proxy for urbanization, economic activity, electrification
- Gap: VIIRS gives us 10+ years of calibrated data — but nobody has built a practical
  multi-model forecasting platform for Indian cities

#### 1.2 Motivation
- Existing BTP-1 work gave us quality NTL reconstruction (DMSP → VIIRS)
- BTP-2 extends that: given a long VIIRS time series, **can we forecast future NTL?**
- Why forecast NTL?
  - Urban growth planning
  - Energy demand estimation
  - Disaster recovery monitoring
  - Proxy for GDP / economic activity
  - Input for future multi-modal models (population + GDP + NTL)
- Need for a platform that is multi-city, multi-model, and publicly accessible

#### 1.3 Objectives
- Download and preprocess monthly VIIRS NTL for Kharagpur and Kolkata (2014–2025)
- Implement and evaluate three forecasting models:
  - SARIMA (statistical baseline)
  - LSTM (scalar deep learning)
  - ConvLSTM (spatial deep learning)
- Validate all forecasts against real Jan–Mar 2026 VIIRS data
- Build an interactive dashboard (Lumina) exposing all results
- Propose a future roadmap for multi-modal NTL analytics

#### 1.4 Scope and Contributions
- Two cities: Kharagpur (small, campus-dominated) and Kolkata (metro)
- 144 months of training data (Jan 2014 – Dec 2023) + 12 test months (2024)
- Live 3-month validation against unseen 2026 data
- Open-source codebase + deployed dashboard

---

### Chapter 2 — Literature Review

> **Note:** Only one paper is formally cited in this report — the paper available in `PrevWork/41597_2024_Article_4228.pdf`. All other background context is written descriptively without formal citation.

#### 2.1 The Chen et al. (2024) NTLSRU-Net Paper *(the only cited reference)*

**Paper:** X. Chen, Z. Wang, F. Zhang, G. Shen, and Q. Chen,
*"A global annual simulated VIIRS nighttime light dataset from 1992 to 2023,"*
Scientific Data, vol. 11, no. 596, 2024. (Nature Publishing Group)

**What to cover:**
- The paper proposes NTLSRU-Net: a U-Net architecture to reconstruct VIIRS-like radiance from legacy DMSP-OLS + NDVI inputs
- Closes the historical gap: DMSP records exist since 1992 but are uncalibrated; VIIRS is calibrated but only from 2012
- Two-channel input (DMSP + NDVI) → single-channel VIIRS-like output at 500 m resolution
- Trained on paired DMSP–VIIRS data from 2012–2013; pretrained weights provided
- Validated with R², RMSE — demonstrates high fidelity reconstruction of 30-year NTL record

**Connection to BTP-1 and BTP-2:**
- BTP-1 reproduced and validated this pipeline for an Indian region (Buxar, Bihar, 2012)
- BTP-2 takes the **output** of this pipeline as a starting point — specifically the VIIRS time series it enables — and asks: *given a long calibrated NTL time series, can we forecast future values?*
- The availability of a clean, calibrated VIIRS-like record (enabled by Chen et al.'s work) is the direct motivation for building the forecasting platform in this project

#### 2.2 Background Context *(descriptive, no formal citations)*
Write these sections as general background prose without citing specific papers:

- **NTL as a socioeconomic proxy:** Briefly explain that NTL data has been widely used as a proxy for urbanization, electricity access, and economic activity. No specific citation needed — this is well-established domain knowledge.
- **SARIMA:** Describe as a classical Box-Jenkins seasonal time series model. Mention it is a standard statistical baseline for monthly data with seasonality.
- **LSTM:** Describe as a recurrent neural network variant designed to capture long-range temporal dependencies in sequential data, commonly used for time series prediction.
- **ConvLSTM:** Describe as an extension of LSTM that incorporates convolutional operations, enabling spatial-temporal sequence modelling — used here for pixel-level NTL forecasting.
- **MC Dropout:** Mention briefly as a technique to obtain uncertainty estimates from neural networks by keeping dropout active at inference time.

---

### Chapter 3 — Data Acquisition and Preprocessing

#### 3.1 Data Source
- VIIRS VCMSLCFG (NOAA) via Google Earth Engine
- Band: `avg_rad` (nW/cm²/sr)
- Resolution: 500 m/pixel
- Temporal: Jan 2014 – Mar 2026 (monthly composites)

#### 3.2 Study Areas

| Property | Kharagpur | Kolkata |
|---|---|---|
| Bounding box | 87.25°E–87.45°E, 22.30°N–22.45°N | 88.20°E–88.55°E, 22.40°N–22.75°N |
| Grid size | ~35 × 45 px | ~70 × 70 px |
| Character | Small city / campus | Major metro |

#### 3.3 Preprocessing Pipeline
- GEE export → GeoTIFF per month
- Mask nodata pixels (avg_rad > 1000 nW/cm²/sr → NaN)
- Compute mean brightness per frame (nanmean)
- Output: `mean_brightness_clean.csv` (scalar time series)
- For ConvLSTM: stack frames as spatial arrays → `frames.npz` + `frame_metadata.json`

**Figures to include:**
- `sarima/plots/eda_01_timeseries.png` — raw time series (Kharagpur)
- `kolkata/sarima/plots/eda_01_timeseries.png` — raw time series (Kolkata)
- `sarima/plots/cleaning_report.png` — data cleaning summary

#### 3.4 Train / Test / Validation Split
- Training: Jan 2014 – Dec 2023 (120 months)
- Test: Jan 2024 – Dec 2024 (12 months)
- Live validation: Jan 2026 – Mar 2026 (3 months, downloaded April 2026)

---

### Chapter 4 — Methodology

#### 4.1 SARIMA — Statistical Baseline

**What it is:**
- Seasonal ARIMA with differencing
- Captures trend + seasonality in scalar time series

**Order selection:**
- Grid search over (p,d,q)(P,D,Q)[12]
- Kharagpur best: SARIMA(0,1,1)(0,1,1)[12], AIC = 189.95
- Kolkata best: SARIMA(0,1,2)(0,1,1)[12], AIC = 444.22
- Verified with auto_arima

**Training:**
- Fit on 120-month series using statsmodels

**Figures to include:**
- `sarima/plots/eda_05_acf_pacf_raw.png` — ACF/PACF raw
- `sarima/plots/eda_06_acf_pacf_diff1.png` — after first difference
- `sarima/plots/eda_07_acf_pacf_diff1_s1.png` — after seasonal difference
- `sarima/plots/eval_01_forecast_vs_actual.png` — test forecast
- `sarima/plots/eval_05_diagnostics_panel.png` — residual diagnostics
- `sarima/plots/forecast.png` — 2026 forecast with CI

**Metrics:**
| Metric | Kharagpur | Kolkata |
|---|---|---|
| MAE | 0.838 | 2.545 |
| RMSE | 1.295 | 4.052 |
| MAPE | 13.75% | 24.82% |
| MASE | 1.260 | 1.151 |

---

#### 4.2 LSTM — Deep Learning Scalar Model

**Architecture:**
```
Input (window_size, 1)
  → LSTM(128, return_sequences=True)
  → Dropout(0.3)
  → LSTM(32, return_sequences=False)
  → Dropout(0.3)
  → Dense(16, relu)
  → Dense(1)
Loss: MAE | Optimizer: Adam(lr=0.0005)
```

**Hyperparameter tuning:**
- Keras Tuner Random Search (30 trials)
- Kharagpur: window=12, units=[128,32], dropout=0.3, lr=0.0005
- Kolkata: window=24, units=[128,16], dropout=0.2, lr=0.0005

**Uncertainty:**
- MC Dropout: 50 forward passes at inference → mean ± std

**Figures to include:**
- `lstm/plots/train_test_split.png` — data split
- `lstm/plots/loss_curve.png` — training/validation loss
- `lstm/plots/eval_01_forecast_vs_actual.png` — test set prediction
- `lstm/plots/eval_02_residuals.png` — residual plot
- `lstm/plots/forecast.png` — 2026 forecast with uncertainty bands

**Metrics:**
| Metric | Kharagpur | Kolkata |
|---|---|---|
| MAE | 0.981 | 3.357 |
| RMSE | 1.390 | 5.147 |
| MAPE | 17.50% | 30.50% |
| MASE | 1.440 | 1.512 |

---

#### 4.3 ConvLSTM — Spatial Deep Learning Model

**Architecture (encoder-decoder):**
```
Input (batch, T=12, H, W, 1)
  → ConvLSTM2D(32) → BatchNorm
  → ConvLSTM2D(64) → BatchNorm
  → ConvLSTM2D(64) → BatchNorm
  → Lambda reshape → (batch, 1, H, W, 64)
  → ConvLSTM2D(64) → BatchNorm
  → ConvLSTM2D(32) → BatchNorm
  → TimeDistributed(Conv2D(1, sigmoid))
  → squeeze → (batch, H, W, 1)
Loss: MAE | Optimizer: Adam | padding=same
```

**Why ConvLSTM matters:**
- SARIMA and LSTM only predict a single scalar (mean brightness)
- ConvLSTM predicts the **entire spatial map** — pixel by pixel
- Enables: spatial change detection, hotspot tracking, urban growth direction

**Evaluation metrics (spatial):**
- SSIM, PSNR, pixel-wise MAE, pixel-wise RMSE
- Also: mean radiance MAE (for comparison with scalar models)

**Figures to include:**
- `convlstm/plots/loss_curve.png` — training loss
- `convlstm/plots/pred_vs_actual_grid.png` — predicted vs actual spatial frames
- `convlstm/plots/error_maps.png` — per-pixel error heatmap
- `convlstm/plots/ssim_timeseries.png` — SSIM over test months
- `convlstm/plots/forecast_panel.png` — 2026 spatial forecast grid

**Metrics:**
| Metric | Kharagpur | Kolkata |
|---|---|---|
| pixel_MAE | 5.14 | 18.19 |
| pixel_RMSE | 7.66 | 46.90 |
| mean_SSIM | 0.477 | 0.536 |
| mean_PSNR (dB) | 24.78 | 16.27 |
| mean_rad_MAE | 0.861 | 10.785 |

---

### Chapter 5 — System Architecture and Dashboard

#### 5.1 Overall Pipeline Architecture
- Describe all 6 phases (Data Acquisition → Preprocessing → SARIMA → LSTM → ConvLSTM → Dashboard)
- Include the ASCII pipeline diagram from README as a figure or verbatim block

#### 5.2 Codebase Structure
- `src/` — download, preprocess, utilities
- `models/sarima/`, `models/lstm/`, `models/convlstm/` — model pipelines
- `outputs/` — all results, plots, models
- `dashboard/` — Streamlit app

#### 5.3 Dashboard — Lumina
Describe each of the 8 pages briefly:

| Page | Purpose |
|---|---|
| NTL Explorer | Monthly map browser with time slider + animation |
| Change Detection | Two-date signed difference maps |
| Charts & Trends | Time series, rolling mean, seasonal decomposition |
| SARIMA Forecast | 2026 forecast + CI + diagnostics |
| LSTM Forecast | MC Dropout forecast + SARIMA comparison |
| ConvLSTM Spatial | Pixel-level maps + GeoTIFF download |
| City Comparison | Side-by-side Kharagpur vs Kolkata |
| 2026 Validation | Live validation against Jan–Mar 2026 actual data |

#### 5.4 Deployment
- Streamlit Community Cloud: https://lumina-ntl.streamlit.app
- `requirements.txt`, `runtime.txt` for reproducibility

---

### Chapter 6 — Results and Discussion

#### 6.1 Model Comparison — Kharagpur

**Test set (2024):**

| Model | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|
| SARIMA | 0.838 | 1.295 | 13.75% | 1.260 |
| LSTM | 0.981 | 1.390 | 17.50% | 1.440 |
| ConvLSTM (mean_rad) | 0.861 | 1.382 | — | — |

**Live validation (Jan–Mar 2026):**

| Month | Actual | SARIMA Pred | SARIMA Err% | LSTM Pred | LSTM Err% |
|---|---|---|---|---|---|
| Jan 2026 | 9.154 | 8.182 | 10.6% | 7.617 | 16.8% |
| Feb 2026 | 9.462 | 8.486 | 10.3% | 7.390 | 21.9% |
| Mar 2026 | 8.869 | 8.058 | 9.1% | 7.648 | 13.8% |
| **Mean** | — | — | **10.0%** | — | **17.5%** |

ConvLSTM mean error: **15.5%** | Best: **SARIMA**

#### 6.2 Model Comparison — Kolkata

**Test set (2024):**

| Model | MAE | RMSE | MAPE | MASE |
|---|---|---|---|---|
| SARIMA | 2.545 | 4.052 | 24.82% | 1.151 |
| LSTM | 3.357 | 5.147 | 30.50% | 1.512 |
| ConvLSTM (mean_rad) | 10.785 | 12.546 | — | — |

**Live validation (Jan–Mar 2026):**

| Month | Actual | SARIMA Pred | SARIMA Err% | LSTM Pred | LSTM Err% |
|---|---|---|---|---|---|
| Jan 2026 | 20.823 | 18.132 | 12.9% | 16.467 | 20.9% |
| Feb 2026 | 21.537 | 18.345 | 14.8% | 16.628 | 22.8% |
| Mar 2026 | 22.328 | 17.654 | 20.9% | 16.840 | 24.6% |
| **Mean** | — | — | **16.2%** | — | **22.8%** |

ConvLSTM mean error: **29.9%** | Best: **SARIMA**

#### 6.3 Discussion
- Why SARIMA outperforms deep learning here:
  - Short monthly time series (120 points) — not enough for deep learning advantage
  - Strong seasonal pattern (period=12) that SARIMA captures directly
  - LSTM/ConvLSTM need more data to generalize
- Why Kolkata errors are higher than Kharagpur:
  - More complex spatial patterns, higher absolute radiance
  - More urban noise in the signal
- ConvLSTM's strength is spatial insight, not scalar accuracy
  - The pixel-level maps reveal **which areas** are growing, not just the average
- Limitation: only NTL data used — adding GDP/population could improve accuracy

---

### Chapter 7 — Conclusion and Future Work

#### 7.1 Conclusion
- Successfully built an end-to-end NTL forecasting platform for two Indian cities
- SARIMA is the best scalar forecaster (10.0% error on Kharagpur, 16.2% on Kolkata)
- ConvLSTM adds a unique spatial forecasting dimension
- All results validated against real 2026 VIIRS data — not just test-set performance
- Live dashboard deployed and accessible

#### 7.2 Future Work — Near Term
- Extend to more Indian cities (Mumbai, Delhi, Chennai, Bangalore)
- Retrain ConvLSTM with more data and larger spatial windows
- Add Transformer-based temporal model (Temporal Fusion Transformer)
- Improve ConvLSTM with attention mechanism

#### 7.3 Future Work — Research Paper Proposal *(not yet done)*
This is the key proposed direction that this BTP-2 work enables:

**Proposed Paper Title:** *"Multi-Modal Night-Time Light Analytics: Combining VIIRS, Population, and GDP for Urban Growth Prediction"*

Key additions over current work:
1. **Population trend data** — census projections, WorldPop datasets
2. **GDP proxy** — district-level GDP, economic activity indices
3. **Correlation analysis** — NTL growth vs GDP growth vs population growth
4. **Multi-modal model** — combine NTL time series + population + GDP as input features
5. **Policy applications** — identify under-lit regions, predict electricity demand, detect economic anomalies

Why current BTP-2 is the foundation:
- Clean NTL pipeline already built
- Two cities with validated forecasts
- Dashboard infrastructure ready for extension
- 2026 validation confirms model reliability

---

## 4. Figures Summary Table

| Figure | File Path | Used In |
|---|---|---|
| Kharagpur raw time series | `outputs/sarima/plots/eda_01_timeseries.png` | Ch. 3 |
| Kolkata raw time series | `outputs/kolkata/sarima/plots/eda_01_timeseries.png` | Ch. 3 |
| Cleaning report | `outputs/sarima/plots/cleaning_report.png` | Ch. 3 |
| Train/test split | `outputs/sarima/plots/train_test_split.png` | Ch. 3 |
| Seasonal boxplot | `outputs/sarima/plots/eda_02_seasonal_boxplot.png` | Ch. 4.1 |
| YoY overlay | `outputs/sarima/plots/eda_03_yoy_overlay.png` | Ch. 4.1 |
| ACF/PACF raw | `outputs/sarima/plots/eda_05_acf_pacf_raw.png` | Ch. 4.1 |
| ACF/PACF diff1 | `outputs/sarima/plots/eda_06_acf_pacf_diff1.png` | Ch. 4.1 |
| ACF/PACF diff1+s1 | `outputs/sarima/plots/eda_07_acf_pacf_diff1_s1.png` | Ch. 4.1 |
| SARIMA test forecast | `outputs/sarima/plots/eval_01_forecast_vs_actual.png` | Ch. 4.1 |
| SARIMA residuals | `outputs/sarima/plots/eval_02_residuals.png` | Ch. 4.1 |
| SARIMA diagnostics | `outputs/sarima/plots/eval_05_diagnostics_panel.png` | Ch. 4.1 |
| SARIMA 2026 forecast | `outputs/sarima/plots/forecast.png` | Ch. 4.1 |
| LSTM loss curve | `outputs/lstm/plots/loss_curve.png` | Ch. 4.2 |
| LSTM test forecast | `outputs/lstm/plots/eval_01_forecast_vs_actual.png` | Ch. 4.2 |
| LSTM residuals | `outputs/lstm/plots/eval_02_residuals.png` | Ch. 4.2 |
| LSTM 2026 forecast | `outputs/lstm/plots/forecast.png` | Ch. 4.2 |
| ConvLSTM loss | `outputs/convlstm/plots/loss_curve.png` | Ch. 4.3 |
| ConvLSTM pred grid | `outputs/convlstm/plots/pred_vs_actual_grid.png` | Ch. 4.3 |
| ConvLSTM error maps | `outputs/convlstm/plots/error_maps.png` | Ch. 4.3 |
| ConvLSTM SSIM series | `outputs/convlstm/plots/ssim_timeseries.png` | Ch. 4.3 |
| ConvLSTM forecast panel | `outputs/convlstm/plots/forecast_panel.png` | Ch. 4.3 |

*(Same set repeated for Kolkata using `outputs/kolkata/` paths)*

---

## 5. LaTeX Packages to Use

```latex
\usepackage{geometry}          % margins
\usepackage{graphicx}          % figures
\usepackage{float}             % [H] placement
\usepackage{booktabs}          % professional tables (\toprule etc.)
\usepackage{caption}           % caption formatting
\usepackage{setspace}          % line spacing
\usepackage{amsmath}           % equations
\usepackage{ragged2e}          % \justifying
\usepackage{parskip}           % paragraph spacing
\usepackage{listings}          % code blocks
\usepackage{xcolor}            % syntax highlighting
\usepackage{hyperref}          % clickable links
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{times}
```

---

## 6. Things to Confirm Before Writing

- [ ] Submission date — April 2026 (exact date?)
- [ ] Are we including Kolkata plots in the same detail as Kharagpur, or just summary tables?
- [ ] Dashboard screenshot — do you want to add one as a figure?
- [ ] Bibliography: do you have specific papers to cite, or should I suggest standard ones?
- [ ] Any section you want expanded more vs. kept short?
- [ ] LaTeX file location — `report/main.tex` inside the BTP folder?

---

## 7. Writing Phases (Tracked)

> Use this as the execution checklist. Mark `[x]` as each item is done.
> **Do not skip phases or reorder steps** — each phase builds on the previous one.

---

### Phase 1 — Setup & Skeleton
*Goal: get the LaTeX file compiling with all structure in place before any prose is written.*

- [ ] Create `report/` directory inside `/home/abhinav/Desktop/BTP/`
- [ ] Create `report/figures/` and copy all required plot PNGs from `outputs/` (see Section 4 figures table)
- [ ] Copy `reportImages/kgplogo.png` → `report/figures/kgplogo.png`
- [ ] Create `report/main.tex` — full document skeleton with all chapters as `% TODO` stubs
- [ ] Create `report/references.bib` — add the single Chen et al. (2024) BibTeX entry
- [ ] Verify `main.tex` compiles to PDF without errors (even with empty chapters)

---

### Phase 2 — Front Matter ✅ COMPLETE
*Goal: complete all pages before Chapter 1.*

- [x] Title page (CS47006, Spring 2025–26, logo, supervisor, title, April 2026)
- [x] Declaration (adapt from BTP-1, update date)
- [x] Certificate (adapt from BTP-1, update semester to Spring 2025–26)
- [x] Acknowledgements (Prof. Soumya Kanti Ghosh + family)
- [x] Abstract (~250 words — write LAST, placeholder for now)
- [x] Manual Table of Contents (placeholder page numbers — finalize in Phase 5)

---

### Phase 3 — Core Technical Chapters ✅ COMPLETE
*Goal: write the factual, data-driven content first — no opinion, just what was done.*

- [x] **Chapter 3** — Data Acquisition & Preprocessing
  - [x] 3.1 Data Source (VIIRS, GEE, avg_rad band)
  - [x] 3.2 Study Areas table (Kharagpur + Kolkata bounding boxes)
  - [x] 3.3 Preprocessing Pipeline (GeoTIFF → mask → nanmean → CSV + NPZ)
  - [x] 3.4 Train/Test/Validation Split (120 train / 24 test / 3 live validation)
  - [x] Insert figures: eda_01_timeseries (both cities), cleaning_report, train_test_split

- [x] **Chapter 4** — Methodology
  - [x] 4.1 SARIMA: model description, order selection, ACF/PACF figures, metrics table
  - [x] 4.2 LSTM: architecture block, hyperparameter tuning, MC Dropout, loss curve, metrics table
  - [x] 4.3 ConvLSTM: encoder-decoder architecture, spatial metrics explanation, figures

- [x] **Chapter 5** — System Architecture & Dashboard
  - [x] 5.1 Pipeline diagram (verbatim ASCII 6-stage pipeline)
  - [x] 5.2 Codebase structure (ASCII tree)
  - [x] 5.3 Dashboard pages table (8 pages)
  - [x] 5.4 Deployment (Streamlit Cloud URL)

**Note:** Test split confirmed as 24 months (Jan 2024 – Dec 2025), not 12 as originally planned.

---

### Phase 4 — Results, Framing & Conclusion ✅ COMPLETE
*Goal: write the interpretive and analytical content.*

- [x] **Chapter 6** — Results & Discussion
  - [x] 6.1 Kharagpur model comparison table (test 2024 + live 2026 validation)
  - [x] 6.2 Kolkata model comparison table (test 2024 + live 2026 validation)
  - [x] 6.3 Discussion (why SARIMA wins, why Kolkata harder, ConvLSTM spatial value, limitations)

- [x] **Chapter 1** — Introduction
  - [x] 1.1 Background (NTL data, VIIRS, connection to BTP-1/Chen et al.)
  - [x] 1.2 Motivation (why forecast NTL, future use cases)
  - [x] 1.3 Objectives
  - [x] 1.4 Scope and Contributions

- [x] **Chapter 2** — Literature Review
  - [x] 2.1 Chen et al. (2024) — the only cited paper — summary + connection to BTP-1 and BTP-2
  - [x] 2.2 Background context prose (NTL proxy, SARIMA, LSTM, ConvLSTM, MC Dropout — no citations)

- [x] **Chapter 7** — Conclusion & Future Work
  - [x] 7.1 Conclusion
  - [x] 7.2 Near-term future work
  - [x] 7.3 Research paper proposal (NTL + population + GDP)

---

### Phase 5 — Finishing ✅ COMPLETE
*Goal: finalize everything, fix TOC, compile clean PDF.*

- [x] Write Abstract (250 words — now that all chapters exist)
- [x] Fix Table of Contents page numbers to match actual compiled output
- [x] Verify all figures render correctly (no missing file errors)
- [x] Verify all `\cite{chen2024}` references resolve correctly from `references.bib`
- [x] Final full compile → check PDF looks correct cover to cover
- [x] Save final PDF as `report/BTP-2-Report.pdf`
