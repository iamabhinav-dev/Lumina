"""
models/sarima/train.py
=======================
STEP 5 — Train/test split & fit SARIMA model.

Reads : outputs/sarima/mean_brightness_clean.csv
        outputs/sarima/best_order.json          (from Step 4)
Saves : outputs/sarima/sarima_model.pkl         (fitted model)
        outputs/sarima/train_test_split.csv      (split record)
        outputs/sarima/plots/train_test_split.png

Split
-----
  Train : Jan 2014 – Dec 2023  (120 months)
  Test  : Jan 2024 – Dec 2025  (24 months)

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/sarima/train.py
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

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR  = os.path.join(ROOT, "outputs", "sarima")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
INPUT_CSV   = os.path.join(OUTPUT_DIR, "mean_brightness_clean.csv")
ORDER_JSON  = os.path.join(OUTPUT_DIR, "best_order.json")
MODEL_PKL   = os.path.join(OUTPUT_DIR, "sarima_model.pkl")
SPLIT_CSV   = os.path.join(OUTPUT_DIR, "train_test_split.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Split boundary ──────────────────────────────────────────────────────────
TRAIN_END = pd.Timestamp("2023-12-01")
TEST_START = pd.Timestamp("2024-01-01")


# ─── Load data ────────────────────────────────────────────────────────────────

def load_series() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    train = df.loc[:TRAIN_END, "mean_rad"]
    test  = df.loc[TEST_START:, "mean_rad"]

    print(f"Full series : {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} months)")
    print(f"Train split : {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} months)")
    print(f"Test  split : {test.index[0].date()} → {test.index[-1].date()}  ({len(test)} months)")

    return train, test, df


def load_order() -> tuple[tuple, tuple]:
    with open(ORDER_JSON) as f:
        cfg = json.load(f)
    order          = tuple(cfg["order"])
    seasonal_order = tuple(cfg["seasonal_order"])
    print(f"\nLoaded order from {ORDER_JSON}")
    print(f"  SARIMA{order} × {seasonal_order}")
    print(f"  AIC (train, from Step 4) = {cfg['aic']}")
    return order, seasonal_order


# ─── Fit model ────────────────────────────────────────────────────────────────

def fit_model(train: pd.Series, order: tuple, seasonal_order: tuple):
    print(f"\nFitting SARIMAX{order}×{seasonal_order} on {len(train)} observations…", flush=True)

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="MS",
    )
    result = model.fit(disp=False)
    print("Done.")
    return result


# ─── In-sample diagnostics ───────────────────────────────────────────────────

def print_model_summary(result) -> None:
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    # Print just the key parameter table, not the full statsmodels output
    summary = result.summary()
    # Extract the coefficient table section
    lines = str(summary).split("\n")
    in_coeff = False
    for line in lines:
        if "coef" in line and "std err" in line:
            in_coeff = True
        if in_coeff:
            print(" ", line)
        if in_coeff and line.strip() == "":
            # stop after the first blank line after the coeff block
            consecutive_blank = True
            break
    print(f"\n  Log-Likelihood : {result.llf:.4f}")
    print(f"  AIC            : {result.aic:.4f}")
    print(f"  BIC            : {result.bic:.4f}")
    print(f"  HQIC           : {result.hqic:.4f}")


# ─── Train-set in-sample fitted values plot ──────────────────────────────────

def save_split_plot(train: pd.Series, test: pd.Series, result) -> None:
    fitted = result.fittedvalues

    # In-sample one-step-ahead forecast on the test set
    fc = result.get_forecast(steps=len(test))
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train.index, train.values,
            color="#4C72B0", linewidth=1.4, label="Train (actual)")
    ax.plot(train.index, fitted.values,
            color="#FF7F0E", linewidth=1.1, linestyle="--",
            alpha=0.8, label="In-sample fitted")
    ax.plot(test.index, test.values,
            color="#2CA02C", linewidth=1.8, label="Test (actual)")
    ax.plot(fc_mean.index, fc_mean.values,
            color="#D62728", linewidth=1.8, linestyle="--",
            label="Forecast (test period)")
    ax.fill_between(
        fc_ci.index,
        fc_ci.iloc[:, 0], fc_ci.iloc[:, 1],
        color="#D62728", alpha=0.15, label="95% CI",
    )

    # Vertical line at train/test boundary
    ax.axvline(TEST_START, color="gray", linestyle=":", linewidth=1.5)
    ax.text(TEST_START, ax.get_ylim()[1] * 0.97, "  Train | Test",
            color="gray", fontsize=9, va="top")

    ax.set_title(
        f"SARIMA{result.model.order}×{result.model.seasonal_order} — Train/Test Split\n"
        f"Train: 2014-01 – 2023-12  |  Test: 2024-01 – 2025-12",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.set_xlabel("")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "train_test_split.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {path}")


# ─── Save artefacts ──────────────────────────────────────────────────────────

def save_model(result) -> None:
    joblib.dump(result, MODEL_PKL)
    print(f"Model saved → {MODEL_PKL}")


def save_split_csv(train: pd.Series, test: pd.Series) -> None:
    train_df = pd.DataFrame({"date": train.index, "mean_rad": train.values, "split": "train"})
    test_df  = pd.DataFrame({"date": test.index,  "mean_rad": test.values,  "split": "test"})
    pd.concat([train_df, test_df], ignore_index=True).to_csv(SPLIT_CSV, index=False)
    print(f"Split CSV   → {SPLIT_CSV}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("─" * 60)
    print("STEP 5 — SARIMA TRAINING")
    print("─" * 60)

    train, test, df_full = load_series()
    order, seasonal_order = load_order()

    result = fit_model(train, order, seasonal_order)
    print_model_summary(result)

    save_split_plot(train, test, result)
    save_model(result)
    save_split_csv(train, test)

    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE")
    print("=" * 60)
    print(f"  Model   : SARIMA{result.model.order}×{result.model.seasonal_order}")
    print(f"  AIC     : {result.aic:.4f}")
    print(f"  BIC     : {result.bic:.4f}")
    print(f"  → Run evaluate.py (Step 6) to score on the test split.")
    print("=" * 60)


if __name__ == "__main__":
    main()
