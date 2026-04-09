"""
models/lstm/evaluate.py
=========================
STEP 5 — Evaluate the trained LSTM on the test set.

Reads : outputs/lstm/lstm_model.keras
        outputs/lstm/sequences.npz
        outputs/lstm/scaler.pkl
        outputs/lstm/train_test_split.csv
        outputs/sarima/evaluation_metrics.json   (for side-by-side comparison)

Saves : outputs/lstm/evaluation_metrics.json
        outputs/lstm/plots/eval_01_forecast_vs_actual.png
        outputs/lstm/plots/eval_02_residuals.png
        outputs/lstm/plots/eval_03_residual_acf.png

Metrics (identical to SARIMA for direct comparison):
  MAE   — Mean Absolute Error
  RMSE  — Root Mean Squared Error
  MAPE  — Mean Absolute Percentage Error (%)
  MASE  — Mean Absolute Scaled Error (vs seasonal naïve, m=12)

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/lstm/evaluate.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LSTM_DIR    = os.path.join(ROOT, "outputs", "lstm")
SARIMA_DIR  = os.path.join(ROOT, "outputs", "sarima")
PLOTS_DIR   = os.path.join(LSTM_DIR, "plots")

MODEL_PATH   = os.path.join(LSTM_DIR, "lstm_model.keras")
SEQ_NPZ      = os.path.join(LSTM_DIR, "sequences.npz")
SCALER_PKL   = os.path.join(LSTM_DIR, "scaler.pkl")
SPLIT_CSV    = os.path.join(LSTM_DIR, "train_test_split.csv")
OUT_METRICS  = os.path.join(LSTM_DIR, "evaluation_metrics.json")
SARIMA_METRICS = os.path.join(SARIMA_DIR, "evaluation_metrics.json")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Seasonal naïve period for MASE denominator
M = 12


# ─── Metric helpers ───────────────────────────────────────────────────────────

def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))

def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))

def mape(actual, pred):
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)

def mase(actual, pred, train_actual):
    """MASE: MAE / MAE of seasonal naïve on training set."""
    naive_errors = np.abs(train_actual[M:] - train_actual[:-M])
    naive_mae    = np.mean(naive_errors)
    return float(mae(actual, pred) / naive_mae)


# ─── Load everything ──────────────────────────────────────────────────────────

def load_all():
    for p in [MODEL_PATH, SEQ_NPZ, SCALER_PKL, SPLIT_CSV]:
        if not os.path.exists(p):
            sys.exit(f"[ERROR] Missing: {p}\nRun the previous steps first.")

    model  = tf.keras.models.load_model(MODEL_PATH)
    data   = np.load(SEQ_NPZ)
    scaler = joblib.load(SCALER_PKL)
    split  = pd.read_csv(SPLIT_CSV, parse_dates=["date"])

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    return model, scaler, X_train, y_train, X_test, y_test, split


# ─── Predict & inverse-transform ──────────────────────────────────────────────

def predict(model, scaler, X_train, y_train, X_test, y_test):
    pred_train_s = model.predict(X_train, verbose=0).flatten()
    pred_test_s  = model.predict(X_test,  verbose=0).flatten()

    actual_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    actual_test  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_train   = scaler.inverse_transform(pred_train_s.reshape(-1, 1)).flatten()
    pred_test    = scaler.inverse_transform(pred_test_s.reshape(-1, 1)).flatten()

    return actual_train, pred_train, actual_test, pred_test


# ─── Plot 1: forecast vs actual ───────────────────────────────────────────────

def plot_forecast_vs_actual(split_df, actual_test, pred_test, window_size):
    train_df = split_df[split_df["split"] == "train"]
    test_df  = split_df[split_df["split"] == "test"]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Full history (faint)
    ax.plot(split_df["date"], split_df["mean_rad"],
            color="#bbbbbb", linewidth=0.9, label="Historical", zorder=1)

    # Test actual
    ax.plot(test_df["date"].values, actual_test,
            color="#2CA02C", linewidth=2.2, label="Actual (test)", zorder=3)

    # LSTM forecast
    ax.plot(test_df["date"].values, pred_test,
            color="#D62728", linewidth=2, linestyle="--",
            label="LSTM forecast (test)", zorder=4)

    # SARIMA overlay if available
    sarima_split = os.path.join(SARIMA_DIR, "train_test_split.csv")
    sarima_model_pkl = os.path.join(SARIMA_DIR, "sarima_model.pkl")
    if os.path.exists(sarima_model_pkl) and os.path.exists(sarima_split):
        try:
            import joblib as jl
            sm = jl.load(sarima_model_pkl)
            fc = sm.get_forecast(steps=len(actual_test))
            sarima_pred = fc.predicted_mean.values
            ax.plot(test_df["date"].values, sarima_pred,
                    color="#4C72B0", linewidth=1.8, linestyle=":",
                    label="SARIMA forecast (test)", zorder=2, alpha=0.8)
        except Exception:
            pass

    ax.axvline(pd.Timestamp("2024-01-01"), color="gray",
               linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.set_title("LSTM — Test Forecast vs Actual  (Jan 2024 – Dec 2025)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "eval_01_forecast_vs_actual.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot → {out}")


# ─── Plot 2: residuals ────────────────────────────────────────────────────────

def plot_residuals(test_dates, actual_test, pred_test):
    residuals = actual_test - pred_test

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Residuals over time
    ax = axes[0]
    ax.bar(test_dates, residuals,
           color=["#2CA02C" if r >= 0 else "#D62728" for r in residuals],
           alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Residuals over Time  (actual − forecast)")
    ax.set_ylabel("Residual (nW/cm²/sr)")
    ax.grid(True, alpha=0.25)

    # Residual histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=10, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.axvline(np.mean(residuals), color="#D62728", linewidth=1.5,
                linestyle="--", label=f"Mean={np.mean(residuals):.3f}")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual (nW/cm²/sr)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "eval_02_residuals.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot → {out}")


# ─── Plot 3: residual ACF ─────────────────────────────────────────────────────

def plot_residual_acf(actual_test, pred_test):
    from statsmodels.graphics.tsaplots import plot_acf
    residuals = actual_test - pred_test

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, lags=min(12, len(residuals) - 2), ax=ax,
             color="#4C72B0", vlines_kwargs={"colors": "#4C72B0"})
    ax.set_title("ACF of LSTM Residuals  (should be near zero for white noise)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "eval_03_residual_acf.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LSTM Step 5 — Evaluation")
    print("=" * 60)

    model, scaler, X_train, y_train, X_test, y_test, split_df = load_all()
    data        = np.load(SEQ_NPZ)
    window_size = int(data["window_size"].flat[0])

    print(f"[INFO] Model loaded: {MODEL_PATH}")
    print(f"[INFO] Window size : {window_size}")

    actual_train, pred_train, actual_test, pred_test = predict(
        model, scaler, X_train, y_train, X_test, y_test
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "MAE"  : round(mae(actual_test, pred_test), 4),
        "RMSE" : round(rmse(actual_test, pred_test), 4),
        "MAPE" : round(mape(actual_test, pred_test), 4),
        "MASE" : round(mase(actual_test, pred_test, actual_train), 4),
    }

    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved → {OUT_METRICS}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    test_df   = split_df[split_df["split"] == "test"]
    test_dates = test_df["date"].values

    plot_forecast_vs_actual(split_df, actual_test, pred_test, window_size)
    plot_residuals(test_dates, actual_test, pred_test)
    plot_residual_acf(actual_test, pred_test)

    # ── Per-month error table ─────────────────────────────────────────────────
    error_df = pd.DataFrame({
        "date"    : test_df["date"].values,
        "actual"  : actual_test.round(4),
        "lstm"    : pred_test.round(4),
        "error"   : (actual_test - pred_test).round(4),
        "abs_err" : np.abs(actual_test - pred_test).round(4),
    })

    # ── Side-by-side with SARIMA ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<8}  {'LSTM':>10}  {'SARIMA':>10}  {'Winner':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

    sarima_metrics = {}
    if os.path.exists(SARIMA_METRICS):
        with open(SARIMA_METRICS) as f:
            sarima_metrics = json.load(f)

    for k in ["MAE", "RMSE", "MAPE", "MASE"]:
        lstm_val   = metrics[k]
        sarima_val = sarima_metrics.get(k, float("nan"))
        if not np.isnan(sarima_val):
            winner = "LSTM ✓" if lstm_val < sarima_val else "SARIMA ✓"
            suffix = "%" if k == "MAPE" else ""
            print(f"  {k:<8}  {lstm_val:>9.4f}{suffix}  "
                  f"{sarima_val:>9.4f}{suffix}  {winner:>8}")
        else:
            print(f"  {k:<8}  {lstm_val:>10.4f}  {'N/A':>10}")

    print(f"\n  MASE note: < 1.0 = beats seasonal naïve baseline")
    print(f"\n  Per-month errors (test set):")
    print(f"  {'Month':<10}  {'Actual':>8}  {'LSTM':>8}  {'|Error|':>8}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for _, row in error_df.iterrows():
        month = pd.Timestamp(row["date"]).strftime("%Y-%m")
        print(f"  {month:<10}  {row['actual']:>8.3f}  "
              f"{row['lstm']:>8.3f}  {row['abs_err']:>8.3f}")

    print("\n" + "=" * 60)
    print("✓ Step 5 complete. Outputs:")
    print(f"  {OUT_METRICS}")
    print(f"  {os.path.join(PLOTS_DIR, 'eval_01_forecast_vs_actual.png')}")
    print(f"  {os.path.join(PLOTS_DIR, 'eval_02_residuals.png')}")
    print(f"  {os.path.join(PLOTS_DIR, 'eval_03_residual_acf.png')}")
    print("\nNext: python models/lstm/forecast.py")


if __name__ == "__main__":
    main()
