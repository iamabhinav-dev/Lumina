"""
Step 5 – ConvLSTM Spatial Forecast
===================================
Generates 12-month ahead pixel-level NTL forecasts (2026-01 … 2026-12) from
the trained ConvLSTM model.

Because the model was trained without dropout (dropout=0.0), Monte-Carlo
Dropout is not available.  Per-pixel prediction intervals (95 %) are computed
from the test-set pixel RMSE:

    lower_95[t] = mean_forecast[t] − 1.96 × pixel_RMSE_test
    upper_95[t] = mean_forecast[t] + 1.96 × pixel_RMSE_test

Outputs (all under outputs/convlstm/):
  • forecast_frames.npz        – mean_forecast, lower_95, upper_95  (12, H, W)
  • forecast_metadata.json     – dates, shape, transform, CRS, method
  • forecast_tiffs/ntl_YYYY_MM_{mean|lower95|upper95}.tif  ×36 GeoTIFFs
  • plots/forecast_animation.gif

Usage:
    python models/convlstm/forecast_convlstm.py [--horizon 12]
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from PIL import Image
import rasterio
from rasterio.transform import Affine
import joblib

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH  = os.path.join(ROOT, "outputs", "convlstm", "convlstm_model.keras")
SCALER_PATH = os.path.join(ROOT, "models",  "convlstm", "frame_scaler.pkl")
META_PATH   = os.path.join(ROOT, "models",  "convlstm", "frame_metadata.json")
FRAMES_PATH = os.path.join(ROOT, "models",  "convlstm", "frames.npz")
EVAL_PATH   = os.path.join(ROOT, "outputs", "convlstm", "evaluation_metrics.json")
OUT_DIR     = os.path.join(ROOT, "outputs", "convlstm")
TIFF_DIR    = os.path.join(OUT_DIR, "forecast_tiffs")
PLOT_DIR    = os.path.join(OUT_DIR, "plots")

for d in (TIFF_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=12,
                    help="Number of months to forecast (default 12)")
ARGS = parser.parse_args()
HORIZON = ARGS.horizon


def main():
    # ── 1. Load model ──────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.join(ROOT, "models"))
    from convlstm.build_convlstm import build_convlstm

    model = build_convlstm()   # dropout=0.0 → deterministic
    model.load_weights(MODEL_PATH)
    print(f"[INFO] Loaded weights — Trainable params: {model.count_params():,}")

    # ── 2. Load data & metadata ────────────────────────────────────────────────
    with open(META_PATH) as f:
        meta = json.load(f)
    H, W      = meta["H"], meta["W"]
    transform = meta["transform"]
    crs_str   = meta["crs"]
    all_dates = meta["dates"]           # 144 entries  2014-01 … 2025-12
    W_SIZE    = int(meta["window_size"])

    data  = np.load(FRAMES_PATH, allow_pickle=True)
    # X_test shape: (24, 12, H, W, 1) – scaled [0,1]
    X_test = data["X_test"]             # (24, 12, 35, 45, 1)

    scaler = joblib.load(SCALER_PATH)

    with open(EVAL_PATH) as f:
        eval_m = json.load(f)
    pixel_rmse_test = eval_m["pixel_RMSE"]   # nW/cm²/sr on original scale
    CI_HALF = 1.96 * pixel_rmse_test
    print(f"[INFO] 95% CI half-width ±{CI_HALF:.4f} nW/cm²/sr  "
          f"(from test pixel RMSE = {pixel_rmse_test:.4f})")

    # ── 3. Build seed window: last W_SIZE frames of test set (scaled) ──────────
    # X_test[-1] has window months [2024-12, 2025-01, …, 2025-11]
    # We want months [2025-01 … 2025-12] → that is the test target frames
    # y_test[-W_SIZE:] gives the last W_SIZE actual frames (scaled).
    # Simpler: the last row of X_test gives us months ending at 2025-11;
    # we need to slide one more step to include 2025-12 in the window.
    # The *prediction* of X_test[-1] is 2026-01's predecessor (2025-12).
    # Use X_test[:, -1:] column-wise to build the rolling seed.

    # All scaled test predictions to build the final 12-frame seed
    y_test_scaled = data["y_test"]          # (24, H, W, 1) – scaled
    # Last W_SIZE actual test frames: indices 24-12 … 23  →  2025-01 … 2025-12
    seed_frames = y_test_scaled[-W_SIZE:]   # (12, H, W, 1) – scaled
    seed_window = seed_frames[np.newaxis, ...]   # (1, 12, H, W, 1)
    print(f"[INFO] Seed window: last {W_SIZE} test frames  "
          f"({all_dates[-W_SIZE]} … {all_dates[-1]})")

    # ── 4. Auto-regressive forecast ────────────────────────────────────────────
    print(f"[INFO] Forecasting {HORIZON} months ahead …")
    current_window = seed_window.copy()          # (1, 12, H, W, 1)  scaled
    forecast_scaled = []

    for step in range(HORIZON):
        pred = model.predict(current_window, verbose=0)  # (1, H, W, 1)
        pred_clipped = np.clip(pred, 0.0, 1.0)
        forecast_scaled.append(pred_clipped[0])          # (H, W, 1)
        # Slide window: drop oldest frame, append new prediction
        current_window = np.concatenate(
            [current_window[:, 1:, :, :, :], pred_clipped[:, np.newaxis, :, :, :]],
            axis=1
        )

    forecast_scaled = np.array(forecast_scaled)  # (12, H, W, 1)

    # ── 5. Inverse-transform to radiance ──────────────────────────────────────
    # scaler was fitted on flattened pixels → reshape to 2-D for inverse_transform
    fc_2d = forecast_scaled.reshape(HORIZON, -1)   # (12, H*W)
    fc_rad = scaler.inverse_transform(fc_2d).reshape(HORIZON, H, W)  # (12,H,W)
    fc_rad = np.clip(fc_rad, 0.0, None).astype(np.float32)

    mean_forecast = fc_rad
    lower_95      = np.clip(fc_rad - CI_HALF, 0.0, None).astype(np.float32)
    upper_95      = np.clip(fc_rad + CI_HALF, 0.0, None).astype(np.float32)

    print(f"[INFO] Forecast radiance range: "
          f"[{fc_rad.min():.4f}, {fc_rad.max():.4f}] nW/cm²/sr")

    # ── 6. Forecast dates ──────────────────────────────────────────────────────
    from datetime import date
    last_date  = date.fromisoformat(all_dates[-1])
    # Advance month-by-month
    def next_months(start: date, n: int):
        dates = []
        y, m = start.year, start.month
        for _ in range(n):
            m += 1
            if m > 12:
                m = 1; y += 1
            dates.append(date(y, m, 1).isoformat())
        return dates

    forecast_dates = next_months(last_date, HORIZON)
    print(f"[INFO] Forecast period: {forecast_dates[0]} … {forecast_dates[-1]}")

    # ── 7. Save NPZ ────────────────────────────────────────────────────────────
    npz_path = os.path.join(OUT_DIR, "forecast_frames.npz")
    np.savez_compressed(
        npz_path,
        mean_forecast=mean_forecast,
        lower_95=lower_95,
        upper_95=upper_95,
    )
    print(f"[INFO] Saved {npz_path}")

    # ── 8. Save forecast_metadata.json ────────────────────────────────────────
    fmeta = {
        "dates":       forecast_dates,
        "horizon":     HORIZON,
        "H":           H,
        "W":           W,
        "transform":   transform,
        "crs":         crs_str,
        "ci_method":   "test_pixel_rmse",
        "pixel_RMSE":  pixel_rmse_test,
        "ci_half":     float(round(CI_HALF, 6)),
    }
    fmeta_path = os.path.join(OUT_DIR, "forecast_metadata.json")
    with open(fmeta_path, "w") as f:
        json.dump(fmeta, f, indent=2)
    print(f"[INFO] Saved {fmeta_path}")

    # ── 9. Export GeoTIFFs ────────────────────────────────────────────────────
    aff = Affine(
        transform[0], transform[1], transform[2],
        transform[3], transform[4], transform[5],
    )
    bands = {
        "mean":    mean_forecast,
        "lower95": lower_95,
        "upper95": upper_95,
    }
    n_tiffs = 0
    for i, d in enumerate(forecast_dates):
        y_str, m_str, _ = d.split("-")
        for band_name, arr in bands.items():
            fname = f"ntl_{y_str}_{m_str}_{band_name}.tif"
            fpath = os.path.join(TIFF_DIR, fname)
            with rasterio.open(
                fpath, "w",
                driver="GTiff",
                height=H, width=W,
                count=1,
                dtype="float32",
                crs=crs_str,
                transform=aff,
            ) as dst:
                dst.write(arr[i].astype(np.float32), 1)
            n_tiffs += 1
    print(f"[INFO] Saved {n_tiffs} GeoTIFFs → {TIFF_DIR}/")

    # ── 10. Animated GIF of mean forecast ─────────────────────────────────────
    vmax = float(np.percentile(mean_forecast, 99))
    norm = Normalize(vmin=0, vmax=max(vmax, 1.0))
    cmap = plt.cm.hot

    frames_pil = []
    for i, d in enumerate(forecast_dates):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=90)
        im = ax.imshow(mean_forecast[i], cmap=cmap, norm=norm,
                       origin="upper", aspect="auto")
        ax.set_title(d, fontsize=10, fontweight="bold")
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("nW/cm²/sr", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        fig.tight_layout(pad=0.5)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w_px, h_px = fig.canvas.get_width_height()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h_px, w_px, 4)
        frames_pil.append(Image.fromarray(rgba, mode="RGBA").convert("RGB"))
        plt.close(fig)

    gif_path = os.path.join(PLOT_DIR, "forecast_animation.gif")
    frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=600,    # ms per frame
        loop=0,
    )
    print(f"[INFO] Saved animated GIF → {gif_path}")

    # ── 11. Summary panel (12-panel grid) ─────────────────────────────────────
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    axes = axes.flatten()
    for i, d in enumerate(forecast_dates):
        ax = axes[i]
        im = ax.imshow(mean_forecast[i], cmap=cmap, norm=norm,
                       origin="upper", aspect="auto")
        ax.set_title(d, fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02,
                 label="nW/cm²/sr")
    fig.suptitle("ConvLSTM 12-Month Forecast — Mean Radiance  (2026)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    panel_path = os.path.join(PLOT_DIR, "forecast_panel.png")
    fig.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved forecast panel → {panel_path}")

    # ── 12. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ConvLSTM Forecast — Step 5 Complete")
    print("=" * 60)
    print(f"  Horizon       : {HORIZON} months  ({forecast_dates[0]} … {forecast_dates[-1]})")
    print(f"  Mean radiance : [{fc_rad.min():.2f}, {fc_rad.max():.2f}] nW/cm²/sr")
    print(f"  95% CI half   : ±{CI_HALF:.2f} nW/cm²/sr  (test pixel RMSE × 1.96)")
    print(f"  GeoTIFFs      : {n_tiffs} files  ({len(forecast_dates)} months × 3 bands)")
    print(f"  NPZ           : {npz_path}")
    print(f"  Metadata      : {fmeta_path}")
    print(f"  GIF           : {gif_path}")
    print(f"  Panel PNG     : {panel_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
