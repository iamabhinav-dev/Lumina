# LSTM NTL Forecast — Detailed Plan
### Kharagpur Night Time Light — Deep Learning Model

---

## 0. Goal & Design Rationale

Build a **univariate LSTM** model on the same `mean_brightness_clean.csv`
scalar time series used by SARIMA. This gives a **direct apples-to-apples**
comparison (same input, same train/test split, same evaluation metrics).

### Why LSTM over a simpler DL model?
| Choice | Reason |
|---|---|
| LSTM over GRU | Standard baseline in time-series DL literature; slightly better for longer sequences |
| Univariate first | 144-month series is small; adding exogenous features risks overfitting |
| Sliding window | Converts the 1-D series into supervised (X, y) pairs the network can learn from |
| MinMaxScaler | LSTM is sensitive to input scale; normalize to [0, 1] before training |
| Keras/TensorFlow | Already in wide use for time series; easier to iterate than raw PyTorch |

---

## 1. Project Structure (new files)

```
BTP/
├── models/
│   ├── sarima/           ← existing
│   └── lstm/
│       ├── __init__.py
│       ├── prepare_sequences.py   # Step 1 — windowing + scaling
│       ├── build_model.py         # Step 2 — architecture definition
│       ├── train.py               # Step 3 — fit, early stopping, callbacks
│       ├── tune.py                # Step 4 — hyperparameter search (Keras Tuner)
│       ├── evaluate.py            # Step 5 — MAE/RMSE/MAPE/MASE + plots
│       └── forecast.py            # Step 6 — multi-step forecast + CI via MC Dropout
├── outputs/
│   ├── sarima/           ← existing
│   └── lstm/
│       ├── scaler.pkl             # fitted MinMaxScaler
│       ├── lstm_model.keras       # trained model (SavedModel format)
│       ├── best_params.json       # best window_size + hyperparams
│       ├── train_test_split.csv   # same schema as sarima/train_test_split.csv
│       ├── evaluation_metrics.json
│       ├── forecast.csv           # date, mean_forecast, lower_95, upper_95
│       └── plots/
│           ├── loss_curve.png
│           ├── train_test_split.png
│           ├── eval_01_forecast_vs_actual.png
│           ├── eval_02_residuals.png
│           ├── eval_03_residual_acf.png
│           └── forecast.png
└── dashboard/
    └── pages/
        └── 05_lstm.py             # Step 7 — Streamlit forecast page
```

---

## 2. Dependencies

```bash
pip install tensorflow scikit-learn keras-tuner
```

`tensorflow` brings Keras. `keras-tuner` handles hyperparameter search.
`scikit-learn` provides `MinMaxScaler` and cross-validation utilities.

---

## 3. Step-by-Step Pipeline

---

### Step 1 — Prepare Sequences (`prepare_sequences.py`)

**Concept — Sliding Window (look-back)**

Convert the 1-D series of length T into supervised learning pairs:

```
Series: [x1, x2, x3, ..., xT]

window_size = W:
  X[0] = [x1, x2, ..., xW]    → y[0] = x(W+1)
  X[1] = [x2, x3, ..., x(W+1)] → y[1] = x(W+2)
  ...
```

Shape fed to LSTM: `(samples, W, 1)` — one feature (mean_rad).

**Key decisions:**
- **Train/test split**: identical to SARIMA — train Jan 2014–Dec 2023 (120 months),
  test Jan 2024–Dec 2025 (24 months). Fit scaler **on train only** to avoid leakage.
- **window_size**: treated as a hyperparameter; candidates = [6, 12, 18, 24].
  12 is the natural choice (one full seasonal cycle) and the expected winner.
- **Scaling**: `MinMaxScaler(feature_range=(0, 1))` fit on train `mean_rad`.
  Inverse-transform predictions before computing metrics.
- Persist the scaler as `outputs/lstm/scaler.pkl` (needed at inference time).

**Outputs:**
- `outputs/lstm/train_test_split.csv` — `date, mean_rad, split` (same schema as SARIMA)
- Console report: series length, window size, train/test sample counts

---

### Step 2 — Architecture (`build_model.py`)

**Baseline architecture (Stacked LSTM):**

```
Input  →  (window_size, 1)

LSTM(64, return_sequences=True)    — first layer, passes full sequence
Dropout(0.2)                       — regularisation
LSTM(32, return_sequences=False)   — second layer, outputs last hidden state
Dropout(0.2)
Dense(16, activation='relu')       — bottleneck
Dense(1)                           — single-step output (next month's radiance)
```

**Why this architecture:**
- Two LSTM layers: first learns local patterns, second learns temporal dependencies
- Dropout (0.2): prevents memorisation on a small dataset (< 200 samples)
- Dense bottleneck: non-linear mapping before final regression output
- Single Dense(1) output: one-step-ahead prediction (multi-step uses iterated forecasting)

**Compile settings:**
- Loss: `MAE` (more robust to the occasional spike than MSE)
- Optimizer: `Adam(learning_rate=1e-3)` with ReduceLROnPlateau callback
- Metric: `RMSE` (tracked via `tf.keras.metrics.RootMeanSquaredError`)

**`build_model(window_size, units_1, units_2, dropout, lr)` function**
returns a compiled `tf.keras.Model` — called by both `train.py` and `tune.py`.

---

### Step 3 — Training (`train.py`)

**Training loop with callbacks:**

| Callback | Config | Purpose |
|---|---|---|
| `EarlyStopping` | `patience=20, restore_best_weights=True` | Halt when val loss stops improving |
| `ReduceLROnPlateau` | `patience=10, factor=0.5, min_lr=1e-5` | Halve LR on plateau |
| `ModelCheckpoint` | save `lstm_model.keras` at best val_loss | Persist best weights |

**Split for training:**
- Of the 120 train months → 96 for fitting, 24 for validation (last 2 years of train)
- Test set remains fully held out

**Epochs:** 300 max (early stopping typically triggers at 80–150)
**Batch size:** 16 (small dataset — smaller batch = more gradient noise = better generalisation)

**Outputs:**
- `outputs/lstm/lstm_model.keras`
- `outputs/lstm/plots/loss_curve.png` — train vs val loss per epoch
- `outputs/lstm/plots/train_test_split.png` — actual vs predicted on train+test
- Console: final train/val loss, epoch stopped at

---

### Step 4 — Hyperparameter Search (`tune.py`)

Uses **Keras Tuner (RandomSearch)** to find the best combination of:

| Hyperparameter | Search Space |
|---|---|
| `window_size` | [6, 12, 18, 24] |
| `units_1` (LSTM layer 1) | [32, 64, 128] |
| `units_2` (LSTM layer 2) | [16, 32, 64] |
| `dropout` | [0.1, 0.2, 0.3] |
| `learning_rate` | [1e-3, 5e-4, 1e-4] |

**Strategy:**
- `max_trials=30`, `executions_per_trial=1`
- Objective: `val_loss` (MAE)
- Fast re-build via `build_model()` from `build_model.py`

**Outputs:**
- `outputs/lstm/best_params.json` — winning hyperparameter set
- Console: top-5 trial table

**Note:** `train.py` reads `best_params.json` if it exists; otherwise falls back
to the baseline architecture in Step 2.

---

### Step 5 — Evaluation (`evaluate.py`)

Same metrics as SARIMA for direct comparison:

| Metric | Formula | Note |
|---|---|---|
| MAE | mean(|actual − forecast|) | Primary metric |
| RMSE | sqrt(mean((actual − forecast)²)) | Penalises large errors |
| MAPE | mean(|actual − forecast| / actual) × 100 | % error |
| MASE | MAE / MAE_seasonal_naïve | < 1 = beats baseline |

**Plots saved to `outputs/lstm/plots/`:**

1. `eval_01_forecast_vs_actual.png` — test period: actual (green) vs LSTM (red dashed) + SARIMA (blue dashed) overlay for comparison
2. `eval_02_residuals.png` — residuals over time
3. `eval_03_residual_acf.png` — ACF of residuals (should be white noise)

**Outputs:**
- `outputs/lstm/evaluation_metrics.json` — `{"MAE":…,"RMSE":…,"MAPE":…,"MASE":…}`
- Comparison table printed to console: SARIMA vs LSTM side by side

---

### Step 6 — Forecast (`forecast.py`)

**Multi-step iterated forecasting:**

For horizon H months beyond Dec 2025:
1. Start with the last `window_size` months of the full series as seed
2. Predict month 1 → append prediction → drop oldest → repeat H times
3. Note: errors compound — CI widens faster than SARIMA

**Uncertainty via Monte Carlo Dropout:**

LSTM has Dropout layers. At inference, normally Dropout is OFF. Enabling it
at test time and running N=200 forward passes gives a distribution of predictions:
- Mean → point forecast
- 2.5th / 97.5th percentile → 95% CI

This is called **MC Dropout** and is a standard lightweight Bayesian approximation
for neural networks — no extra parameters needed.

```python
# MC Dropout inference
model.trainable = False          # freeze weights
# but keep dropout ON by passing training=True to predict
preds = np.stack([model(X_seed, training=True) for _ in range(200)])
mean = preds.mean(axis=0)
ci_low  = np.percentile(preds, 2.5, axis=0)
ci_high = np.percentile(preds, 97.5, axis=0)
```

**CLI argument:** `--horizon N` (default 12), same as `sarima/forecast.py`

**Outputs:**
- `outputs/lstm/forecast.csv` — `date, mean_forecast, lower_95, upper_95`
  (identical schema to SARIMA forecast for easy dashboard reuse)
- `outputs/lstm/plots/forecast.png`

---

### Step 7 — Dashboard Page (`dashboard/pages/05_lstm.py`)

Mirror the structure of `04_forecast.py` (SARIMA page) but with LSTM-specific additions:

**Sections:**

1. **Header + 5-metric row** — MAE, RMSE, MAPE, MASE, forecast horizon
   - Each metric shows LSTM value with SARIMA value as delta for direct comparison

2. **Chart 1 — Main forecast chart**
   - Historical series (context window)
   - LSTM forecast line + MC Dropout 95% CI shaded band
   - SARIMA forecast overlaid (dashed, faded) for comparison
   - Sidebar toggle to show/hide SARIMA overlay

3. **Chart 2 — Head-to-head on test set**
   - Actual (green) vs LSTM (red) vs SARIMA (blue) on Jan 2024–Dec 2025
   - Error bar chart below (LSTM error vs SARIMA error per month)
   - Winner annotation per month (which model was closer)

4. **Chart 3 — Forecast table + bar chart** (same as SARIMA page)

5. **Training diagnostics expander**
   - Loss curve (train vs val loss by epoch)
   - Residual ACF plot
   - MC Dropout uncertainty explanation

6. **Model info expander**
   - Architecture table, hyperparams from `best_params.json`
   - Training details (epochs, early stopping, batch size)

7. **Sidebar controls**
   - Forecast horizon selector (6 / 12 / 18 / 24 months)
   - History context slider
   - Toggle: Show SARIMA overlay
   - Toggle: 95% CI band
   - "▶ Forecast N months" button (re-runs `forecast.py` via subprocess)

**Also update `dashboard/app.py`:** add `05_lstm.py` row to the navigation table.

---

## 4. Execution Order

```bash
cd /home/abhinav/Desktop/BTP
source venv/bin/activate
pip install tensorflow keras-tuner    # one-time

python models/lstm/prepare_sequences.py   # Step 1
python models/lstm/tune.py                # Step 4 (optional, ~5–10 min)
python models/lstm/train.py               # Step 3 (uses best_params.json if present)
python models/lstm/evaluate.py            # Step 5
python models/lstm/forecast.py            # Step 6
# Step 7: dashboard page created by agent
```

Run `tune.py` before `train.py` if you want the hyperparameter search;
skip it to use the baseline architecture directly.

---

## 5. Expected Outcomes

| Metric | SARIMA (test) | LSTM (expected) |
|---|---|---|
| MAE | 0.838 | 0.60–0.85 |
| RMSE | 1.295 | 0.90–1.30 |
| MAPE | 13.75% | 8–14% |
| MASE | 1.260 | 0.90–1.20 |

LSTM should match or beat SARIMA on MAPE/MAE because it can model the non-linear
acceleration in 2024–2025 NTL growth without assuming a fixed seasonal structure.
However, with only 144 data points, the margin may be narrow — the comparison
itself is the thesis contribution.

---

## 6. Thesis Narrative (How to present)

1. SARIMA — classical approach, strong interpretability, good seasonal fit
2. LSTM — data-driven, learns non-linear trend, MC Dropout gives uncertainty
3. **Key question to answer:** Does deep learning bring meaningful improvement
   for a short NTL time series, or does the classical model generalise better?
4. Either result is valid and interesting — discuss trade-offs (data hunger,
   interpretability, training time, uncertainty quantification)
