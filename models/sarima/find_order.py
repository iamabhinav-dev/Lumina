"""
models/sarima/find_order.py
============================
STEP 4 — Find SARIMA(p,d,q)(P,D,Q,12) parameters via auto_arima.

Reads : outputs/sarima/mean_brightness_clean.csv
        (uses the TRAIN split only: Jan 2014 – Dec 2023, 120 months)
Saves : outputs/sarima/best_order.json   ← consumed by train.py in Step 5
Prints: candidate model comparison table + chosen order

Strategy
--------
  1. Run auto_arima with stepwise=True (fast) on the training split.
  2. Also run a small manual grid search over the most common NTL orders
     to cross-check (keeps runtime low by limiting max_p/q to 2).
  3. Compare AIC scores of all candidates, print table, save the winner.

Runtime: typically 1–3 minutes on a laptop (stepwise search).

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/sarima/find_order.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pmdarima as pm
from itertools import product

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "sarima")
INPUT_CSV  = os.path.join(OUTPUT_DIR, "mean_brightness_clean.csv")
ORDER_JSON = os.path.join(OUTPUT_DIR, "best_order.json")

# ─── Train/test split boundary (Step 5 uses the same) ────────────────────────
TRAIN_END_YEAR  = 2023
TRAIN_END_MONTH = 12   # Jan 2014 – Dec 2023 → 120 months

# ─── Grid search bounds (kept small for PC performance) ──────────────────────
MAX_P = 2
MAX_Q = 2
MAX_P_SEASONAL = 1
MAX_Q_SEASONAL = 1


# ─── Load & split ─────────────────────────────────────────────────────────────

def load_train_series() -> pd.Series:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    mask = (df["date"].dt.year < TRAIN_END_YEAR) | (
        (df["date"].dt.year == TRAIN_END_YEAR) &
        (df["date"].dt.month <= TRAIN_END_MONTH)
    )
    train = df.loc[mask, ["date", "mean_rad"]].set_index("date")["mean_rad"]
    print(f"Training series: {train.index[0].date()} → {train.index[-1].date()}  ({len(train)} months)")
    return train


# ─── Auto ARIMA (stepwise) ────────────────────────────────────────────────────

def run_auto_arima(train: pd.Series) -> pm.ARIMA:
    print("\n[1/2] Running auto_arima (stepwise=True)…  (may take 1–3 min)")
    model = pm.auto_arima(
        train,
        seasonal=True,
        m=12,
        d=1,            # forced from Step 3 (first diff achieves stationarity)
        D=None,         # let auto_arima decide seasonal differencing
        max_p=MAX_P, max_q=MAX_Q,
        max_P=MAX_P_SEASONAL, max_Q=MAX_Q_SEASONAL,
        max_order=6,    # p+q+P+Q ≤ 6 to stay lightweight
        information_criterion="aic",
        stepwise=True,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        n_jobs=1,
    )
    print(f"\nauto_arima winner : SARIMA{model.order}×{model.seasonal_order}")
    print(f"  AIC = {model.aic():.4f}")
    return model


# ─── Manual grid cross-check ─────────────────────────────────────────────────

def grid_search(train: pd.Series, auto_order, auto_seasonal) -> list[dict]:
    """
    Fit a small set of SARIMA candidates and rank by AIC.
    Includes the auto_arima winner + a set of common NTL orders.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Fixed d=1 from Step 3; try D=0 and D=1
    candidates = set()
    for p, q, P, Q, D in product(
        range(0, MAX_P + 1),
        range(0, MAX_Q + 1),
        range(0, MAX_P_SEASONAL + 1),
        range(0, MAX_Q_SEASONAL + 1),
        [0, 1],
    ):
        if p + q + P + Q == 0:
            continue   # degenerate model
        candidates.add(((p, 1, q), (P, D, Q, 12)))

    # Always include the auto_arima winner
    candidates.add((auto_order, auto_seasonal))

    print(f"\n[2/2] Grid cross-check: {len(candidates)} candidate models…")
    results = []
    for i, (order, seasonal_order) in enumerate(sorted(candidates), 1):
        try:
            fit = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            results.append({
                "order":          str(order),
                "seasonal_order": str(seasonal_order),
                "aic":            round(fit.aic, 4),
                "bic":            round(fit.bic, 4),
                "params":         fit.params.shape[0],
            })
            tag = " ← auto_arima" if (order == auto_order and seasonal_order == auto_seasonal) else ""
            print(f"  [{i:>3}/{len(candidates)}] SARIMA{order}×{seasonal_order}  "
                  f"AIC={fit.aic:.2f}  BIC={fit.bic:.2f}{tag}")
        except Exception as exc:
            print(f"  [{i:>3}/{len(candidates)}] SARIMA{order}×{seasonal_order}  FAILED: {exc}")

    return results


# ─── Print ranking table ─────────────────────────────────────────────────────

def print_ranking(results: list[dict], top_n: int = 10) -> None:
    if not results:
        return
    df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    print(f"\n{'─'*70}")
    print(f"TOP {min(top_n, len(df))} MODELS BY AIC")
    print(f"{'─'*70}")
    print(f"  {'Rank':<5} {'SARIMA order':<20} {'Seasonal order':<22} {'AIC':>9}  {'BIC':>9}  {'#params':>7}")
    print(f"  {'─'*65}")
    for rank, row in df.head(top_n).iterrows():
        marker = " ★" if rank == 0 else ""
        print(f"  {rank+1:<5} {row['order']:<20} {row['seasonal_order']:<22} "
              f"{row['aic']:>9.2f}  {row['bic']:>9.2f}  {row['params']:>7}{marker}")
    print(f"{'─'*70}")


# ─── Save best order ──────────────────────────────────────────────────────────

def save_best_order(auto_model: pm.ARIMA, grid_results: list[dict]) -> dict:
    # Best from grid (lowest AIC)
    grid_best = min(grid_results, key=lambda x: x["aic"]) if grid_results else None

    # Parse auto_arima orders
    auto_order    = list(auto_model.order)
    auto_seasonal = list(auto_model.seasonal_order)

    # If grid finds something strictly better (>2 AIC improvement), use it
    chosen_order    = auto_order
    chosen_seasonal = auto_seasonal
    chosen_source   = "auto_arima"

    if grid_best:
        import ast
        g_ord  = list(ast.literal_eval(grid_best["order"]))
        g_seas = list(ast.literal_eval(grid_best["seasonal_order"]))
        if (grid_best["aic"] < auto_model.aic() - 2.0 and
                g_ord != auto_order or g_seas != auto_seasonal):
            chosen_order    = g_ord
            chosen_seasonal = g_seas
            chosen_source   = "grid_search"

    payload = {
        "order":              chosen_order,
        "seasonal_order":     chosen_seasonal,
        "source":             chosen_source,
        "aic":                round(auto_model.aic() if chosen_source == "auto_arima"
                                    else grid_best["aic"], 4),
        "auto_arima_order":   auto_order,
        "auto_arima_seasonal": auto_seasonal,
        "auto_arima_aic":     round(auto_model.aic(), 4),
        "train_end":          f"{TRAIN_END_YEAR}-{TRAIN_END_MONTH:02d}",
    }

    with open(ORDER_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BEST ORDER SELECTED")
    print(f"{'='*60}")
    print(f"  SARIMA{tuple(chosen_order)} × {tuple(chosen_seasonal)}")
    print(f"  Source : {chosen_source}")
    print(f"  AIC    : {payload['aic']}")
    print(f"\n  Saved → {ORDER_JSON}")
    print(f"{'='*60}")

    return payload


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    train = load_train_series()

    # Step 4a: auto_arima
    auto_model = run_auto_arima(train)

    # Step 4b: grid cross-check
    grid_results = grid_search(train, auto_model.order, auto_model.seasonal_order)

    # Print ranking
    print_ranking(grid_results)

    # Save winner
    save_best_order(auto_model, grid_results)


if __name__ == "__main__":
    main()
