"""
models/lstm/forecast.py
=========================
STEP 6 — Multi-step forecast with MC Dropout uncertainty estimation.

Reads : outputs/lstm/lstm_model.keras
        outputs/lstm/scaler.pkl
        outputs/sarima/mean_brightness_clean.csv   (full history as seed)

Saves : outputs/lstm/forecast.csv                  (date, mean_forecast, lower_95, upper_95)
        outputs/lstm/plots/forecast.png

Forecasting strategy — iterated one-step-ahead:
  1. Seed with the last `window_size` months of the full observed series.
  2. Predict month t+1, append it to the window, drop the oldest, repeat.
  3. Errors compound over the horizon — CI widens naturally via MC Dropout.

Uncertainty — Monte Carlo Dropout (MC Dropout):
  Dropout layers are normally OFF at inference.  Passing training=True keeps
  them ON, so each forward pass samples a different sub-network.
  Running N=200 stochastic passes gives a distribution of predictions:
    mean        → point forecast
    2.5th pct   → lower 95% CI bound
    97.5th pct  → upper 95% CI bound

CLI argument:
  --horizon N   months to forecast beyond Dec 2025 (default 12)

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/lstm/forecast.py
    python models/lstm/forecast.py --horizon 24
"""

import os
import sys
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf

# ─── Paths / city config ─────────────────────────────────────────────────────
import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

_parser = argparse.ArgumentParser(description="LSTM — Step 6: Forecast")
_parser.add_argument("--city", default="kharagpur",
                     help="City key from src/cities.py  (default: kharagpur)")
_parser.add_argument("--horizon", type=int, default=12,
                     help="Months to forecast beyond Dec 2025 (default: 12)")
ARGS = _parser.parse_args()
CITY    = ARGS.city.lower().strip()
HORIZON = ARGS.horizon

LSTM_DIR    = _cities.get_lstm_dir(CITY, ROOT)
SARIMA_DIR  = _cities.get_sarima_dir(CITY, ROOT)
PLOTS_DIR   = os.path.join(LSTM_DIR, "plots")

MODEL_PATH  = os.path.join(LSTM_DIR, "lstm_model.keras")
SCALER_PKL  = os.path.join(LSTM_DIR, "scaler.pkl")
SEQ_NPZ     = os.path.join(LSTM_DIR, "sequences.npz")
INPUT_CSV   = os.path.join(SARIMA_DIR, "mean_brightness_clean.csv")
FC_CSV      = os.path.join(LSTM_DIR, "forecast.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_END  = pd.Timestamp("2023-12-01")
MC_SAMPLES = 200     # number of stochastic forward passes for MC Dropout
DEFAULT_HORIZON = 12


# ─── MC Dropout forward pass (batched) ───────────────────────────────────────

@tf.function
def _predict_batch(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    """Single compiled forward pass with dropout active (training=True)."""
    return model(x, training=True)


def mc_forecast(model: tf.keras.Model,
                seed_window: np.ndarray,
                horizon: int,
                n_samples: int = MC_SAMPLES) -> np.ndarray:
    """
    Iterated multi-step forecast using Monte Carlo Dropout — batched.

    Instead of n_samples × horizon individual calls (slow), we run ONE
    batched forward pass per horizon step:
      - Stack n_samples identical windows → (n_samples, W, 1)
      - model((n_samples, W, 1), training=True) → (n_samples,) in one call
      - Each sample gets a different dropout mask → stochastic diversity
    Total model calls: horizon  (12 by default, instead of n_samples × horizon)

    Parameters
    ----------
    model       : compiled Keras model (with Dropout layers)
    seed_window : (window_size, 1) array of scaled values — the starting context
    horizon     : number of future steps to predict
    n_samples   : number of parallel stochastic samples

    Returns
    -------
    all_preds : (n_samples, horizon) array of raw scaled predictions
    """
    # Tile seed to (n_samples, window_size, 1) — each sample starts identically
    windows = np.tile(seed_window[np.newaxis], (n_samples, 1, 1)).astype(np.float32)
    all_preds = np.zeros((n_samples, horizon), dtype=np.float32)

    for h in range(horizon):
        x_batch = tf.constant(windows)                        # (N, W, 1)
        preds   = _predict_batch(model, x_batch).numpy().flatten()  # (N,)
        all_preds[:, h] = preds
        # Roll every sample's window: drop oldest column, append new prediction
        windows = np.roll(windows, -1, axis=1)
        windows[:, -1, 0] = preds

    return all_preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(horizon: int) -> None:
    print("=" * 60)
    print("LSTM Step 6 — Forecast")
    print(f"  Horizon  : {horizon} months beyond Dec 2025")
    print(f"  Method   : MC Dropout  ({MC_SAMPLES} stochastic passes)")
    print("=" * 60)

    # ── Load model + scaler ───────────────────────────────────────────────────
    for p in [MODEL_PATH, SCALER_PKL, SEQ_NPZ, INPUT_CSV]:
        if not os.path.exists(p):
            sys.exit(f"[ERROR] Missing: {p}\nRun previous steps first.")

    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PKL)
    data   = np.load(SEQ_NPZ)
    window_size = int(data["window_size"].flat[0])

    print(f"[INFO] Model loaded  |  window_size={window_size}")

    # ── Build seed from full observed series ──────────────────────────────────
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    full_vals   = df["mean_rad"].values.astype(np.float32)
    full_scaled = scaler.transform(full_vals.reshape(-1, 1)).flatten()

    # Seed = last window_size months of the complete history (Jan 2014 – Dec 2025)
    seed_window = full_scaled[-window_size:].reshape(-1, 1)  # (W, 1)
    print(f"[INFO] Seed window: "
          f"{df['date'].iloc[-window_size].strftime('%Y-%m')} – "
          f"{df['date'].iloc[-1].strftime('%Y-%m')}")

    # ── MC Dropout forecast ───────────────────────────────────────────────────
    print(f"\n[INFO] Running {MC_SAMPLES} MC Dropout passes × {horizon} steps …")
    all_preds_scaled = mc_forecast(model, seed_window, horizon, MC_SAMPLES)
    # shape: (MC_SAMPLES, horizon)

    # Inverse-transform each sample
    all_preds = np.zeros_like(all_preds_scaled)
    for s in range(MC_SAMPLES):
        all_preds[s] = scaler.inverse_transform(
            all_preds_scaled[s].reshape(-1, 1)
        ).flatten()

    # Aggregate across MC samples
    fc_mean  = all_preds.mean(axis=0)
    fc_lower = np.percentile(all_preds, 2.5,  axis=0)
    fc_upper = np.percentile(all_preds, 97.5, axis=0)

    # ── Build forecast dates ──────────────────────────────────────────────────
    last_date = df["date"].iloc[-1]
    fc_dates  = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )

    fc_df = pd.DataFrame({
        "date"          : fc_dates,
        "mean_forecast" : fc_mean.round(6),
        "lower_95"      : fc_lower.round(6),
        "upper_95"      : fc_upper.round(6),
    })
    fc_df.to_csv(FC_CSV, index=False)
    print(f"[INFO] Saved → {FC_CSV}")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n  {'Month':<10}  {'Forecast':>10}  {'Lower 95%':>10}  {'Upper 95%':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for _, row in fc_df.iterrows():
        m = pd.Timestamp(row["date"]).strftime("%Y-%m")
        print(f"  {m:<10}  {row['mean_forecast']:>10.3f}  "
              f"{row['lower_95']:>10.3f}  {row['upper_95']:>10.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    context_cutoff = df["date"].max() - pd.DateOffset(years=4)
    hist_ctx = df[df["date"] >= context_cutoff]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Faint full history
    ax.plot(df["date"], df["mean_rad"],
            color="#cccccc", linewidth=0.8, zorder=1)

    # Context history
    ax.plot(hist_ctx["date"], hist_ctx["mean_rad"],
            color="#4C72B0", linewidth=1.8, label="Historical (last 4y)", zorder=2)

    # CI band
    ax.fill_between(fc_dates, fc_lower, fc_upper,
                    color="#D62728", alpha=0.15, label="95% CI (MC Dropout)", zorder=3)

    # Forecast line
    ax.plot(fc_dates, fc_mean,
            color="#D62728", linewidth=2.2, linestyle="--",
            marker="o", markersize=5, label=f"LSTM forecast ({horizon}m)", zorder=4)

    # SARIMA forecast overlay if available
    sarima_fc_csv = os.path.join(SARIMA_DIR, "forecast.csv")
    if os.path.exists(sarima_fc_csv):
        sarima_fc = pd.read_csv(sarima_fc_csv, parse_dates=["date"])
        sarima_fc = sarima_fc.head(horizon)
        ax.plot(sarima_fc["date"], sarima_fc["mean_forecast"],
                color="#4C72B0", linewidth=1.8, linestyle=":",
                marker="s", markersize=4,
                label="SARIMA forecast (reference)", zorder=3, alpha=0.75)

    ax.axvline(df["date"].max(), color="gray",
               linestyle=":", linewidth=1.2, alpha=0.7)
    ax.annotate("History | Forecast",
                xy=(df["date"].max(), ax.get_ylim()[1]),
                xytext=(5, -15), textcoords="offset points",
                fontsize=8, color="gray")

    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.set_title(f"LSTM — {horizon}-Month Forecast with MC Dropout 95% CI")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, "forecast.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot → {out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    mc_spread = (fc_upper - fc_lower).mean()
    print(f"\n  MC Dropout spread (mean CI width): {mc_spread:.3f} nW/cm²/sr")
    print(f"  Forecast range : "
          f"{fc_mean.min():.3f} – {fc_mean.max():.3f} nW/cm²/sr")

    print("\n" + "=" * 60)
    print("✓ Step 6 complete. Outputs:")
    print(f"  {FC_CSV}")
    print(f"  {out}")
    print("\nNext: python dashboard/pages/05_lstm.py  (via streamlit run dashboard/app.py)")


if __name__ == "__main__":
    if HORIZON < 1 or HORIZON > 36:
        sys.exit("[ERROR] --horizon must be between 1 and 36.")
    main(HORIZON)
