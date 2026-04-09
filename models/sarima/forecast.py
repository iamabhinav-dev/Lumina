"""
models/sarima/forecast.py
==========================
STEP 7 — Refit on full series, generate future forecast, save CSV & plot.

Why refit on full data:
  The model in sarima_model.pkl was trained on 120 months (up to Dec 2023).
  For forecasting future months (2026+) we should use ALL 144 months of
  available data to get the best possible starting state.

Reads : outputs/sarima/best_order.json
        outputs/sarima/mean_brightness_clean.csv
Saves : outputs/sarima/sarima_model_full.pkl     (full-data model)
        outputs/sarima/forecast.csv
        outputs/sarima/plots/forecast.png

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/sarima/forecast.py [--horizon 12]

    --horizon N  : months to forecast ahead (default 12)
"""

import os
import sys
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ─── Paths / city config ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

_parser = argparse.ArgumentParser(description="SARIMA future forecast.")
_parser.add_argument("--city", default="kharagpur",
                     help="City key from src/cities.py  (default: kharagpur)")
_parser.add_argument("--horizon", type=int, default=12,
                     help="Months to forecast ahead (default 12)")
ARGS = _parser.parse_args()
CITY    = ARGS.city.lower().strip()
HORIZON = ARGS.horizon

OUTPUT_DIR    = _cities.get_sarima_dir(CITY, ROOT)
PLOTS_DIR     = os.path.join(OUTPUT_DIR, "plots")
INPUT_CSV     = os.path.join(OUTPUT_DIR, "mean_brightness_clean.csv")
ORDER_JSON    = os.path.join(OUTPUT_DIR, "best_order.json")
FULL_MODEL_PKL = os.path.join(OUTPUT_DIR, "sarima_model_full.pkl")
FORECAST_CSV  = os.path.join(OUTPUT_DIR, "forecast.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─── Load full series ─────────────────────────────────────────────────────────

def load_full_series() -> pd.Series:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    series = df["mean_rad"]
    print(f"Full series: {series.index[0].date()} → {series.index[-1].date()}  ({len(series)} months)")
    return series


def load_order() -> tuple[tuple, tuple]:
    with open(ORDER_JSON) as f:
        cfg = json.load(f)
    order          = tuple(cfg["order"])
    seasonal_order = tuple(cfg["seasonal_order"])
    print(f"Order: SARIMA{order} × {seasonal_order}  (from best_order.json)")
    return order, seasonal_order


# ─── Refit on full series ─────────────────────────────────────────────────────

def fit_full_model(series: pd.Series, order: tuple, seasonal_order: tuple):
    print(f"\nRefitting SARIMAX{order}×{seasonal_order} on all {len(series)} observations…",
          flush=True)
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="MS",
    )
    result = model.fit(disp=False)
    print(f"Done.  AIC={result.aic:.4f}  BIC={result.bic:.4f}")
    joblib.dump(result, FULL_MODEL_PKL)
    print(f"Full model saved → {FULL_MODEL_PKL}")
    return result


# ─── Generate forecast ────────────────────────────────────────────────────────

def generate_forecast(result, series: pd.Series, horizon: int) -> pd.DataFrame:
    """
    Forecast `horizon` months beyond the last observation.
    Returns DataFrame with date, mean_forecast, lower_95, upper_95.
    """
    fc = result.get_forecast(steps=horizon)
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int(alpha=0.05)

    # Build future date index (month starts)
    last_date   = series.index[-1]
    future_idx  = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )
    fc_mean.index = future_idx
    fc_ci.index   = future_idx

    df_fc = pd.DataFrame({
        "date":           future_idx,
        "mean_forecast":  fc_mean.values.round(6),
        "lower_95":       fc_ci.iloc[:, 0].values.round(6),
        "upper_95":       fc_ci.iloc[:, 1].values.round(6),
    })

    return df_fc


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_forecast(series: pd.Series, df_fc: pd.DataFrame, horizon: int) -> None:
    # Show last 36 months of history for context
    context = series.iloc[-36:]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical (context window)
    ax.plot(context.index, context.values,
            color="#4C72B0", linewidth=1.6, label="Historical (last 3 years)")

    # Faint full history
    ax.plot(series.index, series.values,
            color="#4C72B0", linewidth=0.6, alpha=0.25)

    # Forecast
    ax.plot(df_fc["date"], df_fc["mean_forecast"],
            color="#D62728", linewidth=2.2, linestyle="--",
            marker="o", markersize=4, label=f"Forecast ({horizon} months)")
    ax.fill_between(df_fc["date"],
                    df_fc["lower_95"], df_fc["upper_95"],
                    color="#D62728", alpha=0.15, label="95% Confidence Interval")

    # Vertical boundary
    ax.axvline(series.index[-1], color="gray", linestyle=":", linewidth=1.5)
    ax.text(series.index[-1], ax.get_ylim()[1] * 0.98,
            "  History | Forecast", color="gray", fontsize=9, va="top")

    # Annotate each forecast point with its value
    for _, row in df_fc.iterrows():
        ax.annotate(
            f"{row['mean_forecast']:.2f}",
            xy=(row["date"], row["mean_forecast"]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=7.5, color="#D62728",
        )

    ax.set_title(
        f"SARIMA(0,1,1)(0,1,1)[12] — {horizon}-Month Forecast\n"
        f"Fitted on full series (Jan 2014 – Dec 2025)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "forecast.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved   → {path}")


# ─── Print forecast table ─────────────────────────────────────────────────────

def print_forecast_table(df_fc: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("FORECAST TABLE")
    print("=" * 62)
    print(f"  {'Date':<14}  {'Forecast':>10}  {'Lower 95%':>10}  {'Upper 95%':>10}")
    print(f"  {'─'*58}")
    for _, row in df_fc.iterrows():
        print(f"  {str(row['date'].date()):<14}  "
              f"{row['mean_forecast']:>10.4f}  "
              f"{row['lower_95']:>10.4f}  "
              f"{row['upper_95']:>10.4f}")
    print("=" * 62)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("─" * 62)
    print(f"STEP 7 — SARIMA FORECAST  (city = {CITY}, horizon = {HORIZON} months)")
    print("─" * 62 + "\n")

    series = load_full_series()
    order, seasonal_order = load_order()

    result = fit_full_model(series, order, seasonal_order)

    df_fc = generate_forecast(result, series, HORIZON)

    # Save CSV
    df_fc.to_csv(FORECAST_CSV, index=False)
    print(f"Forecast CSV → {FORECAST_CSV}")

    print_forecast_table(df_fc)

    plot_forecast(series, df_fc, HORIZON)

    print("\n" + "=" * 62)
    print("STEP 7 COMPLETE")
    print("=" * 62)
    print(f"  Forecast spans: {df_fc['date'].iloc[0].date()} → {df_fc['date'].iloc[-1].date()}")
    print(f"  Mean forecast range: {df_fc['mean_forecast'].min():.4f} – {df_fc['mean_forecast'].max():.4f}")
    print(f"  → Next: integrate into Streamlit dashboard (Step 8).")
    print("=" * 62)


if __name__ == "__main__":
    main()
