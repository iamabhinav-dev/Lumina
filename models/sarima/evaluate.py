"""
models/sarima/evaluate.py
==========================
STEP 6 — Evaluate SARIMA on the test split.

Reads : outputs/sarima/sarima_model.pkl
        outputs/sarima/train_test_split.csv
Saves : outputs/sarima/evaluation_metrics.json
        outputs/sarima/plots/eval_01_forecast_vs_actual.png
        outputs/sarima/plots/eval_02_residuals.png
        outputs/sarima/plots/eval_03_residual_acf.png
        outputs/sarima/plots/eval_04_qq.png
        outputs/sarima/plots/eval_05_diagnostics.png  (statsmodels built-in)

Metrics computed
----------------
  MAE   — Mean Absolute Error (same units as radiance)
  RMSE  — Root Mean Squared Error
  MAPE  — Mean Absolute Percentage Error
  MASE  — Mean Absolute Scaled Error (vs seasonal naïve baseline)

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/sarima/evaluate.py
"""

import argparse
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
import scipy.stats as stats

from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

# ─── Paths / city config ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

_parser = argparse.ArgumentParser()
_parser.add_argument("--city", default="kharagpur",
                     help="City key from src/cities.py  (default: kharagpur)")
ARGS = _parser.parse_args()
CITY = ARGS.city.lower().strip()

OUTPUT_DIR = _cities.get_sarima_dir(CITY, ROOT)
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
MODEL_PKL  = os.path.join(OUTPUT_DIR, "sarima_model.pkl")
SPLIT_CSV  = os.path.join(OUTPUT_DIR, "train_test_split.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─── Load artefacts ───────────────────────────────────────────────────────────

def load_data():
    result = joblib.load(MODEL_PKL)
    print(f"Loaded model : SARIMA{result.model.order}×{result.model.seasonal_order}")

    df = pd.read_csv(SPLIT_CSV, parse_dates=["date"])
    train = df[df["split"] == "train"].set_index("date")["mean_rad"]
    test  = df[df["split"] == "test"].set_index("date")["mean_rad"]
    print(f"Train : {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} months)")
    print(f"Test  : {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} months)")
    return result, train, test


# ─── Generate forecast on test period ────────────────────────────────────────

def make_forecast(result, n_steps: int):
    fc = result.get_forecast(steps=n_steps)
    mean  = fc.predicted_mean
    ci    = fc.conf_int(alpha=0.05)
    return mean, ci


# ─── Metrics ─────────────────────────────────────────────────────────────────

def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))

def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

def mape(actual, predicted):
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def mase(actual, predicted, train, m=12):
    """
    Mean Absolute Scaled Error.
    Scale = MAE of seasonal naïve forecast on the training set.
    Seasonal naïve: y_hat_t = y_{t-m}
    """
    naive_errors = np.abs(train.values[m:] - train.values[:-m])
    scale = float(np.mean(naive_errors))
    if scale == 0:
        return float("nan")
    return mae(actual, predicted) / scale

def compute_all_metrics(actual: pd.Series, predicted: pd.Series, train: pd.Series) -> dict:
    a = actual.values
    p = predicted.values
    return {
        "MAE":  round(mae(a, p), 4),
        "RMSE": round(rmse(a, p), 4),
        "MAPE": round(mape(a, p), 4),
        "MASE": round(mase(a, p, train), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Forecast vs Actual (full series context) ─────────────────────────

def plot_forecast_vs_actual(train, test, fc_mean, fc_ci, metrics) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train.index, train.values,
            color="#4C72B0", linewidth=1.3, label="Train (actual)")
    ax.plot(test.index, test.values,
            color="#2CA02C", linewidth=2.0, label="Test (actual)", zorder=4)
    ax.plot(fc_mean.index, fc_mean.values,
            color="#D62728", linewidth=2.0, linestyle="--",
            label="SARIMA forecast", zorder=5)
    ax.fill_between(fc_ci.index,
                    fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
                    color="#D62728", alpha=0.15, label="95% CI")

    ax.axvline(pd.Timestamp("2024-01-01"), color="gray",
               linestyle=":", linewidth=1.5)

    # Metrics text box
    txt = (f"MAE  = {metrics['MAE']:.3f}\n"
           f"RMSE = {metrics['RMSE']:.3f}\n"
           f"MAPE = {metrics['MAPE']:.1f}%\n"
           f"MASE = {metrics['MASE']:.3f}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_title(
        "SARIMA(0,1,1)(0,1,1)[12] — Forecast vs Actual (Test: 2024-01 – 2025-12)",
        fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "eval_01_forecast_vs_actual.png")


# ── Plot 2: Residuals over time ───────────────────────────────────────────────

def plot_residuals(result, train) -> None:
    resid = result.resid.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    fig.suptitle("Residual Analysis — Training Set", fontsize=12, fontweight="bold")

    # Residuals over time
    axes[0].plot(resid.index, resid.values, color="#4C72B0", linewidth=1.0)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals over Time (should be white noise ~N(0,σ²))")
    axes[0].grid(True, alpha=0.3)

    # Residual histogram
    axes[1].hist(resid.values, bins=20, color="#4C72B0", edgecolor="white",
                 alpha=0.75, density=True)
    mu, sigma = resid.mean(), resid.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma),
                 color="red", linewidth=1.5, label=f"N({mu:.2f}, {sigma:.2f}²)")
    axes[1].set_title("Residual Distribution vs Normal")
    axes[1].set_xlabel("Residual value")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "eval_02_residuals.png")


# ── Plot 3: Residual ACF ──────────────────────────────────────────────────────

def plot_residual_acf(result) -> None:
    resid = result.resid.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    fig.suptitle("Residual ACF & PACF — should have no significant spikes",
                 fontsize=12, fontweight="bold")

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf( resid, lags=36, ax=axes[0], alpha=0.05,
              title="Residual ACF", color="#4C72B0")
    plot_pacf(resid, lags=36, ax=axes[1], alpha=0.05, method="ywm",
              title="Residual PACF", color="#E74C3C")

    for ax in axes:
        ax.axvline(x=12, color="green", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, "eval_03_residual_acf.png")


# ── Plot 4: Q-Q plot ──────────────────────────────────────────────────────────

def plot_qq(result) -> None:
    resid = result.resid.dropna()

    fig, ax = plt.subplots(figsize=(6, 6))
    (osm, osr), (slope, intercept, r) = stats.probplot(resid.values, dist="norm")
    ax.scatter(osm, osr, color="#4C72B0", s=20, alpha=0.7, label="Residuals")
    x_line = np.array([min(osm), max(osm)])
    ax.plot(x_line, slope * x_line + intercept,
            color="red", linewidth=1.5, label=f"Normal line (R²={r**2:.3f})")
    ax.set_title("Q-Q Plot of Residuals", fontsize=12, fontweight="bold")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "eval_04_qq.png")


# ── Plot 5: statsmodels built-in diagnostic panel ────────────────────────────

def plot_diagnostics_panel(result) -> None:
    fig = result.plot_diagnostics(figsize=(12, 8))
    fig.suptitle("SARIMA Model Diagnostics (statsmodels)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "eval_05_diagnostics_panel.png")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("TEST-SET EVALUATION METRICS")
    print("=" * 60)
    print(f"  MAE   = {metrics['MAE']:.4f}  nW/cm²/sr")
    print(f"         (avg absolute error in radiance units)")
    print(f"  RMSE  = {metrics['RMSE']:.4f}  nW/cm²/sr")
    print(f"         (penalises large errors more than MAE)")
    print(f"  MAPE  = {metrics['MAPE']:.2f} %")
    print(f"         (% error relative to actual values)")
    print(f"  MASE  = {metrics['MASE']:.4f}")
    mase_verdict = "better than seasonal naïve" if metrics['MASE'] < 1 else "worse than seasonal naïve"
    print(f"         (< 1 = {mase_verdict})")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("─" * 60)
    print("STEP 6 — SARIMA EVALUATION")
    print("─" * 60 + "\n")

    result, train, test = load_data()

    # Forecast on test period
    fc_mean, fc_ci = make_forecast(result, n_steps=len(test))
    fc_mean.index = test.index   # align dates

    # Metrics
    metrics = compute_all_metrics(test, fc_mean, train)
    print_metrics(metrics)

    # Save metrics JSON
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {METRICS_JSON}")

    # Plots
    print("\nGenerating evaluation plots…")
    plot_forecast_vs_actual(train, test, fc_mean, fc_ci, metrics)
    plot_residuals(result, train)
    plot_residual_acf(result)
    plot_qq(result)
    plot_diagnostics_panel(result)

    print("\n" + "=" * 60)
    print("STEP 6 COMPLETE")
    print("=" * 60)
    print(f"  → Run forecast.py (Step 7) to predict future months.")
    print("=" * 60)


if __name__ == "__main__":
    main()
