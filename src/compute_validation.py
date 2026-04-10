"""
src/compute_validation.py
=========================
Compare Jan–Mar 2026 actual NTL (from extract_actuals.py) against all three
model forecasts (SARIMA, LSTM, ConvLSTM) and write a validation JSON.

Inputs (must already exist):
  outputs/{city}/sarima/actuals_2026.csv
  outputs/{city}/sarima/forecast.csv
  outputs/{city}/lstm/forecast.csv
  outputs/{city}/convlstm/forecast_frames.npz
  outputs/{city}/convlstm/actual_frames_2026.npz
  outputs/{city}/convlstm/forecast_metadata.json

Outputs per city:
  outputs/{city}/validation_2026.json
  outputs/{city}/convlstm/val_pixel_mae_2026_01.npz  (and _02, _03)

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python src/compute_validation.py
    python src/compute_validation.py --city kolkata
    python src/compute_validation.py --all
"""

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

VALIDATION_DATES = ["2026-01-01", "2026-02-01", "2026-03-01"]


def _mae(actual: float, predicted: float) -> float:
    return abs(actual - predicted)


def _pct_error(actual: float, predicted: float) -> float:
    if actual == 0:
        return 0.0
    return abs(actual - predicted) / actual * 100.0


def compute_for_city(city: str) -> None:
    cfg       = _cities.get_city(city)
    sar_dir   = _cities.get_sarima_dir(city, ROOT)
    lstm_dir  = _cities.get_lstm_dir(city, ROOT)
    clstm_dir = _cities.get_convlstm_dir(city, ROOT)

    print(f"\n{'='*60}")
    print(f"City: {cfg['display_name']}")

    # ── Load actuals ──────────────────────────────────────────────────────
    actuals_csv = os.path.join(sar_dir, "actuals_2026.csv")
    if not os.path.exists(actuals_csv):
        print(f"  [ERROR] Missing actuals CSV: {actuals_csv}")
        print("  Run: python src/extract_actuals.py first.")
        return

    actuals_df = pd.read_csv(actuals_csv, parse_dates=["date"])
    actuals_df["date"] = actuals_df["date"].dt.strftime("%Y-%m-%d")
    actuals_map = dict(zip(actuals_df["date"], actuals_df["mean_rad"]))
    available   = sorted(actuals_map.keys())
    print(f"  Actuals available: {available}")

    # ── SARIMA ────────────────────────────────────────────────────────────
    sar_fc = pd.read_csv(os.path.join(sar_dir, "forecast.csv"), parse_dates=["date"])
    sar_fc["date"] = sar_fc["date"].dt.strftime("%Y-%m-%d")
    sar_map = dict(zip(sar_fc["date"], sar_fc["mean_forecast"]))

    sarima_results = []
    for d in available:
        actual    = actuals_map[d]
        predicted = sar_map.get(d)
        if predicted is None:
            continue
        sarima_results.append({
            "date":      d,
            "actual":    round(actual,    6),
            "predicted": round(predicted, 6),
            "mae":       round(_mae(actual, predicted),       6),
            "pct_error": round(_pct_error(actual, predicted), 4),
        })
    mean_sar_pct = round(float(np.mean([r["pct_error"] for r in sarima_results])), 4) if sarima_results else None
    print(f"  SARIMA mean % error: {mean_sar_pct}")

    # ── LSTM ──────────────────────────────────────────────────────────────
    lstm_fc = pd.read_csv(os.path.join(lstm_dir, "forecast.csv"), parse_dates=["date"])
    lstm_fc["date"] = lstm_fc["date"].dt.strftime("%Y-%m-%d")
    lstm_map = dict(zip(lstm_fc["date"], lstm_fc["mean_forecast"]))

    lstm_results = []
    for d in available:
        actual    = actuals_map[d]
        predicted = lstm_map.get(d)
        if predicted is None:
            continue
        lstm_results.append({
            "date":      d,
            "actual":    round(actual,    6),
            "predicted": round(predicted, 6),
            "mae":       round(_mae(actual, predicted),       6),
            "pct_error": round(_pct_error(actual, predicted), 4),
        })
    mean_lstm_pct = round(float(np.mean([r["pct_error"] for r in lstm_results])), 4) if lstm_results else None
    print(f"  LSTM   mean % error: {mean_lstm_pct}")

    # ── ConvLSTM (spatial) ────────────────────────────────────────────────
    actual_npz   = os.path.join(clstm_dir, "actual_frames_2026.npz")
    forecast_npz = os.path.join(clstm_dir, "forecast_frames.npz")

    convlstm_results = []
    if os.path.exists(actual_npz) and os.path.exists(forecast_npz):
        act_data  = np.load(actual_npz)
        act_frames= act_data["frames"]               # (N, H, W)
        act_dates = list(act_data["dates"])

        fc_data   = np.load(forecast_npz)
        fc_frames = fc_data["mean_forecast"]         # (12, H, W)  — Jan 2026 = index 0

        # Map forecast month index: Jan 2026 = 0, Feb = 1, Mar = 2
        for i, d in enumerate(act_dates):
            if i >= fc_frames.shape[0]:
                break
            act_frame  = act_frames[i]               # (H, W)
            pred_frame = fc_frames[i]                # (H, W)

            pixel_mae  = np.abs(act_frame - pred_frame)          # (H, W)
            mean_mae   = float(np.mean(pixel_mae))
            actual_mean   = float(np.mean(act_frame))
            predicted_mean= float(np.mean(pred_frame))
            pct_err    = _pct_error(actual_mean, predicted_mean)

            # Save pixel-level MAE map for dashboard heatmap
            month_tag = d[:7].replace("-", "_")      # "2026_01"
            mae_npz   = os.path.join(clstm_dir, f"val_pixel_mae_{month_tag}.npz")
            np.savez_compressed(mae_npz, pixel_mae=pixel_mae)

            convlstm_results.append({
                "date":           d,
                "actual_mean":    round(actual_mean,    6),
                "predicted_mean": round(predicted_mean, 6),
                "mae":            round(mean_mae,       6),
                "pct_error":      round(pct_err,        4),
                "pixel_mae_npz":  os.path.relpath(mae_npz, ROOT),
            })
        mean_cl_pct = round(float(np.mean([r["pct_error"] for r in convlstm_results])), 4)
        print(f"  ConvLSTM mean % error: {mean_cl_pct}")
    else:
        print(f"  [WARN] ConvLSTM npz missing — skipping spatial validation")
        mean_cl_pct = None

    # ── Best model ────────────────────────────────────────────────────────
    errors = {
        "sarima":   mean_sar_pct,
        "lstm":     mean_lstm_pct,
        "convlstm": mean_cl_pct if convlstm_results else None,
    }
    valid_errors  = {k: v for k, v in errors.items() if v is not None}
    best_model    = min(valid_errors, key=valid_errors.get) if valid_errors else None

    # ── Write JSON ────────────────────────────────────────────────────────
    out_dir = os.path.join(ROOT, "outputs", city if city != "kharagpur" else "",
                           "" if city == "kharagpur" else "")
    # Use sarima_dir parent for kharagpur consistency
    out_dir = os.path.dirname(sar_dir)
    out_path = os.path.join(out_dir, "validation_2026.json")

    result = {
        "city":       city,
        "display":    cfg["display_name"],
        "generated":  str(date.today()),
        "months":     available,
        "best_model": best_model,
        "mean_pct_errors": errors,
        "sarima":     sarima_results,
        "lstm":       lstm_results,
        "convlstm":   convlstm_results,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Best model: {best_model}")


def main():
    parser = argparse.ArgumentParser(description="Compute 2026 validation metrics")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--city", default="kharagpur",
                       help="City key (default: kharagpur)")
    group.add_argument("--all", action="store_true",
                       help="Process all registered cities")
    args = parser.parse_args()

    if args.all:
        import cities as _c
        for city in _c.CITIES:
            compute_for_city(city)
    else:
        compute_for_city(args.city)

    print("\nDone.")


if __name__ == "__main__":
    main()
