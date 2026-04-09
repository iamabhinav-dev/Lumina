"""
models/convlstm/evaluate_convlstm.py
=======================================
STEP 4 — Spatial evaluation of the ConvLSTM model on the test set.

Reads : models/convlstm/frames.npz            (sequences from Step 1)
        models/convlstm/frame_scaler.pkl       (scaler from Step 1)
        models/convlstm/frame_metadata.json    (geo metadata from Step 1)
        outputs/convlstm/convlstm_model.keras  (trained model from Step 3)

Writes: outputs/convlstm/evaluation_metrics.json
        outputs/convlstm/plots/error_maps.png
        outputs/convlstm/plots/ssim_timeseries.png
        outputs/convlstm/plots/pred_vs_actual_grid.png

Evaluation covers Jan 2024 – Dec 2025 (24 test frames) via auto-regressive
prediction seeded from the last 12 train frames.

Metrics (all in original radiance units nW/cm²/sr):
  Pixel MAE          mean|actual - pred| over all pixels × 24 months
  Pixel RMSE         sqrt(mean(actual - pred)²)
  SSIM               Structural Similarity Index (skimage), per month → mean
  PSNR               Peak Signal-to-Noise Ratio, per month → mean
  Mean-radiance MAE  MAE on spatial mean per month (comparable to SARIMA/LSTM)
  Mean-radiance RMSE

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/convlstm/evaluate_convlstm.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Paths / city config ─────────────────────────────────────────────────────────────────
import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

_parser = argparse.ArgumentParser()
_parser.add_argument("--city", default="kharagpur",
                     help="City key from src/cities.py  (default: kharagpur)")
ARGS = _parser.parse_args()
CITY = ARGS.city.lower().strip()

MODEL_DIR    = _cities.get_convlstm_model_dir(CITY, ROOT)
OUTPUT_DIR   = _cities.get_convlstm_dir(CITY, ROOT)
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")

FRAMES_NPZ    = os.path.join(MODEL_DIR,  "frames.npz")
SCALER_PKL    = os.path.join(MODEL_DIR,  "frame_scaler.pkl")
METADATA_JSON = os.path.join(MODEL_DIR,  "frame_metadata.json")
MODEL_PATH    = os.path.join(OUTPUT_DIR, "convlstm_model.keras")
METRICS_JSON  = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")

SARIMA_METRICS = _cities.get_sarima_dir(CITY, ROOT) + "/evaluation_metrics.json"
LSTM_METRICS   = _cities.get_lstm_dir(CITY, ROOT)   + "/evaluation_metrics.json"

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─── 1. Load model ────────────────────────────────────────────────────────────

# Register custom metric so .keras files produced by any environment can be loaded
@tf.keras.utils.register_keras_serializable(package="convlstm")
def _pixel_rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def load_model() -> tf.keras.Model:
    """Load full saved model (architecture + weights)."""
    for p in [MODEL_PATH, FRAMES_NPZ, SCALER_PKL]:
        if not os.path.exists(p):
            sys.exit(f"[ERROR] {p} not found. Run prior steps first.")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )
    print(f"[INFO] Loaded model from {MODEL_PATH}")
    print(f"[INFO] Trainable params: {model.count_params():,}")
    return model


# ─── 2. Auto-regressive prediction on test set ────────────────────────────────

def predict_test(model, X_test: np.ndarray) -> np.ndarray:
    """
    Predict all 24 test frames sequentially.
    X_test[i] is already seeded correctly from prepare_frames.py — just predict.
    Shape: predictions → (24, H, W, 1) scaled [0,1]
    """
    print(f"[INFO] Predicting {len(X_test)} test frames ...")
    preds = model.predict(X_test, batch_size=8, verbose=0)  # (24, H, W, 1)
    return preds


# ─── 3. Inverse transform ─────────────────────────────────────────────────────

def inverse(arr: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform a (..., 1) or (N, H, W, 1) array to radiance units."""
    shape   = arr.shape
    flat    = arr.reshape(-1, 1)
    orig    = scaler.inverse_transform(flat)
    return orig.reshape(shape)


# ─── 4. Compute spatial metrics ───────────────────────────────────────────────

def compute_metrics(pred_orig: np.ndarray,
                    true_orig: np.ndarray,
                    dates: list[str]) -> dict:
    """
    pred_orig, true_orig: (24, H, W, 1) in original radiance units.
    dates: list of 24 date strings for the test period.
    """
    H, W = true_orig.shape[1], true_orig.shape[2]
    pred2d = pred_orig[:, :, :, 0]   # (24, H, W)
    true2d = true_orig[:, :, :, 0]

    # ── Global pixel metrics ──────────────────────────────────────────────────
    pixel_mae  = float(np.mean(np.abs(true2d - pred2d)))
    pixel_rmse = float(np.sqrt(np.mean((true2d - pred2d) ** 2)))

    # ── Per-month spatial metrics ─────────────────────────────────────────────
    data_range = float(true2d.max() - true2d.min())
    data_range = max(data_range, 1e-6)

    per_month = []
    ssim_list, psnr_list = [], []
    for i in range(len(dates)):
        t = true2d[i]
        p = pred2d[i]

        ssim_val = structural_similarity(t, p, data_range=data_range)
        psnr_val = peak_signal_noise_ratio(t, p, data_range=data_range)
        pm_mae   = float(np.mean(np.abs(t - p)))

        ssim_list.append(float(ssim_val))
        psnr_list.append(float(psnr_val))
        per_month.append({
            "date":      dates[i],
            "SSIM":      float(ssim_val),
            "PSNR":      float(psnr_val),
            "pixel_MAE": pm_mae,
        })

    mean_ssim = float(np.mean(ssim_list))
    mean_psnr = float(np.mean(psnr_list))

    # ── Mean-radiance MAE/RMSE (comparable to SARIMA/LSTM) ───────────────────
    pred_mean = pred2d.mean(axis=(1, 2))   # (24,)
    true_mean = true2d.mean(axis=(1, 2))
    mean_rad_mae  = float(np.mean(np.abs(true_mean - pred_mean)))
    mean_rad_rmse = float(np.sqrt(np.mean((true_mean - pred_mean) ** 2)))

    metrics = {
        "pixel_MAE":     round(pixel_mae,  4),
        "pixel_RMSE":    round(pixel_rmse, 4),
        "mean_SSIM":     round(mean_ssim,  4),
        "mean_PSNR":     round(mean_psnr,  4),
        "mean_rad_MAE":  round(mean_rad_mae,  4),
        "mean_rad_RMSE": round(mean_rad_rmse, 4),
        "per_month":     per_month,
    }
    return metrics


# ─── 5. Error map plot ────────────────────────────────────────────────────────

def plot_error_map(pred_orig: np.ndarray, true_orig: np.ndarray) -> None:
    """Per-pixel mean absolute error heatmap over all 24 test months."""
    error = np.mean(np.abs(true_orig[:, :, :, 0] - pred_orig[:, :, :, 0]), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(true_orig[:, :, :, 0].mean(axis=0),
                         cmap="YlOrRd", vmin=0)
    axes[0].set_title("Mean Actual (Jan 2024 – Dec 2025)", fontsize=10)
    plt.colorbar(im0, ax=axes[0], label="nW/cm²/sr", fraction=0.046)
    axes[0].axis("off")

    im1 = axes[1].imshow(error, cmap="Reds", vmin=0)
    axes[1].set_title("Mean Absolute Error (per pixel)", fontsize=10)
    plt.colorbar(im1, ax=axes[1], label="MAE  nW/cm²/sr", fraction=0.046)
    axes[1].axis("off")

    plt.suptitle("ConvLSTM — Spatial Error Analysis (Test Set)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "error_maps.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[INFO] Saved {out}")


# ─── 6. SSIM timeseries plot ─────────────────────────────────────────────────

def plot_ssim_timeseries(metrics: dict) -> None:
    pm    = metrics["per_month"]
    dates = pd.to_datetime([m["date"] for m in pm])
    ssims = [m["SSIM"] for m in pm]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, ssims, marker="o", linewidth=2, color="#D62728", markersize=5)
    ax.axhline(metrics["mean_SSIM"], color="gray", linestyle="--",
               label=f"Mean SSIM = {metrics['mean_SSIM']:.4f}")
    ax.axhline(0.80, color="green", linestyle=":", alpha=0.6,
               label="SSIM = 0.80 (good threshold)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_ylabel("SSIM")
    ax.set_title("ConvLSTM — SSIM per Test Month (Jan 2024 – Dec 2025)", fontsize=12)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "ssim_timeseries.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[INFO] Saved {out}")


# ─── 7. Predicted vs Actual grid ─────────────────────────────────────────────

def plot_pred_vs_actual_grid(pred_orig: np.ndarray,
                             true_orig: np.ndarray,
                             dates: list[str]) -> None:
    """
    4×6 grid (4 rows = Actual / Predicted / Difference / SSIM annotation,
    6 cols = 6 sample months spaced across the test period).
    """
    n_total  = len(dates)
    n_cols   = 6
    indices  = np.linspace(0, n_total - 1, n_cols, dtype=int)
    n_rows   = 3   # actual, predicted, difference

    vmax = float(true_orig[:, :, :, 0].max())
    row_labels = ["Actual", "Predicted", "| Error |"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows))

    for ci, idx in enumerate(indices):
        t = true_orig[idx, :, :, 0]
        p = pred_orig[idx, :, :, 0]
        e = np.abs(t - p)
        label = pd.Timestamp(dates[idx]).strftime("%b %Y")

        axes[0, ci].set_title(label, fontsize=9, fontweight="bold")
        axes[0, ci].imshow(t, cmap="YlOrRd", vmin=0, vmax=vmax)
        axes[1, ci].imshow(p, cmap="YlOrRd", vmin=0, vmax=vmax)
        im_e = axes[2, ci].imshow(e, cmap="Reds", vmin=0)

        for ri in range(n_rows):
            axes[ri, ci].axis("off")

    for ri, label in enumerate(row_labels):
        axes[ri, 0].set_ylabel(label, fontsize=10, fontweight="bold",
                               rotation=90, labelpad=4)
        axes[ri, 0].yaxis.set_visible(True)
        axes[ri, 0].tick_params(left=False, labelleft=False)

    plt.colorbar(im_e, ax=axes[2, :], orientation="horizontal",
                 fraction=0.03, label="Absolute error  (nW/cm²/sr)")
    plt.suptitle("ConvLSTM — Predicted vs Actual (6 sample test months)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pred_vs_actual_grid.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {out}")


# ─── 8. Print summary ─────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    sarima_m = lstm_m = None
    if os.path.exists(SARIMA_METRICS):
        with open(SARIMA_METRICS) as f:
            sarima_m = json.load(f)
    if os.path.exists(LSTM_METRICS):
        with open(LSTM_METRICS) as f:
            lstm_m = json.load(f)

    print("\n" + "=" * 62)
    print("  CONVLSTM EVALUATION — SUMMARY")
    print("=" * 62)
    print(f"  Pixel MAE          : {metrics['pixel_MAE']:.4f}  nW/cm²/sr")
    print(f"  Pixel RMSE         : {metrics['pixel_RMSE']:.4f}  nW/cm²/sr")
    print(f"  Mean SSIM          : {metrics['mean_SSIM']:.4f}  (1 = perfect)")
    print(f"  Mean PSNR          : {metrics['mean_PSNR']:.4f}  dB")
    print(f"  Mean-radiance MAE  : {metrics['mean_rad_MAE']:.4f}  nW/cm²/sr")
    print(f"  Mean-radiance RMSE : {metrics['mean_rad_RMSE']:.4f}  nW/cm²/sr")
    print("-" * 62)
    print("  Mean-radiance MAE comparison:")
    print(f"    ConvLSTM : {metrics['mean_rad_MAE']:.4f}")
    if sarima_m:
        print(f"    SARIMA   : {sarima_m['MAE']:.4f}")
    if lstm_m:
        print(f"    LSTM     : {lstm_m['MAE']:.4f}")
    print("=" * 62 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n[ConvLSTM] Step 4 — evaluate_convlstm.py\n")

    # Load
    model  = load_model()
    data   = np.load(FRAMES_NPZ)
    scaler = joblib.load(SCALER_PKL)
    with open(METADATA_JSON) as f:
        meta = json.load(f)

    X_test = data["X_test"]   # (24, 12, 35, 45, 1)
    y_test = data["y_test"]   # (24, 35, 45, 1)

    # Test dates (last 24 of 144 months)
    all_dates  = meta["dates"]
    test_dates = all_dates[-24:]

    # Predict
    pred_scaled = predict_test(model, X_test)

    # Inverse transform
    pred_orig = inverse(pred_scaled, scaler)
    true_orig = inverse(y_test,      scaler)

    print(f"[INFO] Pred range: [{pred_orig.min():.4f}, {pred_orig.max():.4f}] nW/cm²/sr")
    print(f"[INFO] True range: [{true_orig.min():.4f}, {true_orig.max():.4f}] nW/cm²/sr")

    # Metrics
    metrics = compute_metrics(pred_orig, true_orig, test_dates)

    # Save
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved {METRICS_JSON}")

    # Plots
    plot_error_map(pred_orig, true_orig)
    plot_ssim_timeseries(metrics)
    plot_pred_vs_actual_grid(pred_orig, true_orig, test_dates)

    # Summary
    print_summary(metrics)

    print("[ConvLSTM] Step 4 complete.")
    print("[ConvLSTM] Next: python models/convlstm/forecast_convlstm.py\n")


if __name__ == "__main__":
    main()
