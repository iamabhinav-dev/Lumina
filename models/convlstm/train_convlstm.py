"""
models/convlstm/train_convlstm.py
====================================
STEP 3 — Train the ConvLSTM encoder-decoder model.

Reads : models/convlstm/frames.npz           (from Step 1)
        models/convlstm/frame_scaler.pkl      (from Step 1)
        models/convlstm/frame_metadata.json   (from Step 1)

Saves : outputs/convlstm/convlstm_model.keras     (best weights via ModelCheckpoint)
        outputs/convlstm/training_history.json
        outputs/convlstm/plots/loss_curve.png
        outputs/convlstm/plots/sample_predictions.png

Validation split (of the 108 train sequences):
  First 96 → fit      (Jan 2014 – Dec 2021)
  Last  12 → validate (Jan 2022 – Dec 2023)

Callbacks:
  EarlyStopping     patience=40, restore_best_weights=True
  ReduceLROnPlateau patience=15, factor=0.5, min_lr=1e-6
  ModelCheckpoint   saves best val_loss model

Training strategy: 1-step-ahead prediction (predict next frame from W input frames).
At forecast time the model is applied auto-regressively.

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/convlstm/train_convlstm.py
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Limit CPU thread pool — biggest lever for RAM on CPU-only machines.
# 4 threads ≈ 4 GB RAM ceiling vs 16+ threads which can spike to 12+ GB.
os.environ.setdefault("OMP_NUM_THREADS",              "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS",       "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS",       "2")
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)   # threads per op
tf.config.threading.set_inter_op_parallelism_threads(2)   # parallel ops
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convlstm.build_convlstm import build_convlstm, DEFAULTS

# ─── Paths / city config ─────────────────────────────────────────────────────────────────
import argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import cities as _cities

_parser = argparse.ArgumentParser()
_parser.add_argument("--city", default="kharagpur",
                     help="City key from src/cities.py  (default: kharagpur)")
_parser.add_argument("--filters_enc", nargs=3, type=int, default=None,
                     metavar=("F1", "F2", "F3"),
                     help="Encoder ConvLSTM filter counts (default: 16 32 32)")
_parser.add_argument("--filters_dec", nargs=2, type=int, default=None,
                     metavar=("F1", "F2"),
                     help="Decoder ConvLSTM filter counts (default: 32 16)")
ARGS = _parser.parse_args()
CITY = ARGS.city.lower().strip()

FILTERS_ENC = tuple(ARGS.filters_enc) if ARGS.filters_enc else DEFAULTS["filters_enc"]
FILTERS_DEC = tuple(ARGS.filters_dec) if ARGS.filters_dec else DEFAULTS["filters_dec"]

MODEL_DIR   = _cities.get_convlstm_model_dir(CITY, ROOT)
OUTPUT_DIR  = _cities.get_convlstm_dir(CITY, ROOT)
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")

FRAMES_NPZ    = os.path.join(MODEL_DIR,  "frames.npz")
SCALER_PKL    = os.path.join(MODEL_DIR,  "frame_scaler.pkl")
METADATA_JSON = os.path.join(MODEL_DIR,  "frame_metadata.json")
MODEL_PATH    = os.path.join(OUTPUT_DIR, "convlstm_model.keras")
HISTORY_JSON  = os.path.join(OUTPUT_DIR, "training_history.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
# Tuned for CPU: larger batch (fewer steps/epoch) + tighter patience
# Expected: ~2-3s/epoch × ~60 epochs ≈ 3-5 minutes total
EPOCHS      = 500
BATCH_SIZE  = 16
VAL_SPLIT   = 12    # last N train sequences used for validation
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─── 1. Load sequences ────────────────────────────────────────────────────────

def load_sequences():
    for p in [FRAMES_NPZ, SCALER_PKL]:
        if not os.path.exists(p):
            sys.exit(f"[ERROR] {p} not found. Run prepare_frames.py first.")

    data    = np.load(FRAMES_NPZ)
    X_train = data["X_train"]   # (108, 12, 35, 45, 1)
    y_train = data["y_train"]   # (108, 35, 45, 1)
    X_test  = data["X_test"]    # ( 24, 12, 35, 45, 1)
    y_test  = data["y_test"]    # ( 24, 35, 45, 1)
    W       = int(data["window_size"].flat[0])
    scaler  = joblib.load(SCALER_PKL)

    print(f"[INFO] Loaded sequences — X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"[INFO]                    X_test : {X_test.shape}   y_test : {y_test.shape}")
    print(f"[INFO] Window size: {W}")
    return X_train, y_train, X_test, y_test, W, scaler


# ─── 2. Train/val split ───────────────────────────────────────────────────────

def split_train_val(X_train, y_train):
    n_val   = VAL_SPLIT
    n_fit   = len(X_train) - n_val
    X_fit   = X_train[:n_fit]
    y_fit   = y_train[:n_fit]
    X_val   = X_train[n_fit:]
    y_val   = y_train[n_fit:]
    print(f"[INFO] Train/val split: {n_fit} fit  |  {n_val} val")
    return X_fit, y_fit, X_val, y_val


# ─── 3. Build + train ─────────────────────────────────────────────────────────

def train(X_fit, y_fit, X_val, y_val):
    print(f"\n[INFO] Building ConvLSTM model ...")
    input_shape = tuple(X_fit.shape[1:])   # (T, H, W, C) — derived from actual data
    print(f"[INFO] input_shape from data: {input_shape}")
    model = build_convlstm(
        input_shape  = input_shape,
        filters_enc  = FILTERS_ENC,
        filters_dec  = FILTERS_DEC,
        kernel_size  = DEFAULTS["kernel_size"],
        dropout      = DEFAULTS["dropout"],
        lr           = DEFAULTS["lr"],
    )
    print(f"[INFO] Trainable params: {model.count_params():,}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
    ]

    print(f"[INFO] Training — max {EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"val={VAL_SPLIT} sequences\n")

    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ─── 4. Save training history ─────────────────────────────────────────────────

def save_history(history) -> dict:
    hist = {
        "loss":         [float(v) for v in history.history["loss"]],
        "val_loss":     [float(v) for v in history.history["val_loss"]],
        "pixel_rmse":   [float(v) for v in history.history.get("pixel_rmse", [])],
        "val_pixel_rmse": [float(v) for v in history.history.get("val_pixel_rmse", [])],
        "filters_enc":  list(FILTERS_ENC),
        "filters_dec":  list(FILTERS_DEC),
    }
    with open(HISTORY_JSON, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[INFO] Saved {HISTORY_JSON}")
    return hist


# ─── 5. Loss curve plot ───────────────────────────────────────────────────────

def plot_loss_curve(hist: dict) -> None:
    epochs     = range(1, len(hist["loss"]) + 1)
    best_epoch = int(np.argmin(hist["val_loss"])) + 1

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, hist["loss"],     label="Train MAE",  linewidth=1.5)
    ax.plot(epochs, hist["val_loss"], label="Val MAE",    linewidth=1.5)
    ax.axvline(best_epoch, color="gray", linestyle="--",
               label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (scaled [0,1])")
    ax.set_title("ConvLSTM — Training Loss Curve")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "loss_curve.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[INFO] Saved {out}")


# ─── 6. Sample predictions plot ───────────────────────────────────────────────

def plot_sample_predictions(model, X_test, y_test, scaler) -> None:
    """
    For 3 random test samples plot:
      [Mean of 12 input frames] | [Predicted frame] | [Actual frame] | [Error map]
    All in original radiance units.
    """
    n_samples = min(3, len(X_test))
    indices   = np.random.choice(len(X_test), size=n_samples, replace=False)
    indices   = np.sort(indices)

    fig, axes = plt.subplots(n_samples, 4,
                             figsize=(14, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input mean\n(last 12 frames)", "Predicted", "Actual", "Error (|actual−pred|)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    H, W = y_test.shape[1], y_test.shape[2]
    vmax_rad = float(scaler.data_max_[0])

    for row, idx in enumerate(indices):
        x_seq  = X_test[idx]   # (12, H, W, 1)
        y_true = y_test[idx]   # (H, W, 1)

        # Predict
        pred_scaled = model.predict(x_seq[np.newaxis], verbose=0)[0]  # (H, W, 1)

        # Inverse transform
        def inv(arr):
            return scaler.inverse_transform(
                arr.reshape(-1, 1)
            ).reshape(H, W)

        input_mean = inv(x_seq.mean(axis=0))    # mean across 12 input frames
        pred_frame = inv(pred_scaled)
        true_frame = inv(y_true)
        error_map  = np.abs(true_frame - pred_frame)

        im0 = axes[row, 0].imshow(input_mean, cmap="YlOrRd", vmin=0, vmax=vmax_rad)
        im1 = axes[row, 1].imshow(pred_frame, cmap="YlOrRd", vmin=0, vmax=vmax_rad)
        im2 = axes[row, 2].imshow(true_frame, cmap="YlOrRd", vmin=0, vmax=vmax_rad)
        im3 = axes[row, 3].imshow(error_map,  cmap="Reds",   vmin=0)

        for col in range(4):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(f"Test #{idx}", fontsize=9)
        plt.colorbar(im2,  ax=axes[row, 2], fraction=0.04, label="nW/cm²/sr")
        plt.colorbar(im3,  ax=axes[row, 3], fraction=0.04, label="error")

    plt.suptitle("ConvLSTM — Sample Test Predictions", fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "sample_predictions.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved {out}")


# ─── 7. Quick test-set MAE for orientation ────────────────────────────────────

def quick_eval(model, X_test, y_test, scaler) -> None:
    pred_scaled = model.predict(X_test, verbose=0)   # (24, H, W, 1)

    H, W = y_test.shape[1], y_test.shape[2]
    pred_orig = scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)).reshape(-1, H, W)
    true_orig = scaler.inverse_transform(
        y_test.reshape(-1, 1)).reshape(-1, H, W)

    pixel_mae  = float(np.mean(np.abs(true_orig - pred_orig)))
    pixel_rmse = float(np.sqrt(np.mean((true_orig - pred_orig) ** 2)))
    mean_mae   = float(np.mean(np.abs(
        true_orig.mean(axis=(1, 2)) - pred_orig.mean(axis=(1, 2))
    )))

    print("\n" + "=" * 55)
    print("  QUICK TEST-SET EVALUATION (original units)")
    print("=" * 55)
    print(f"  Pixel MAE        : {pixel_mae:.4f}  nW/cm²/sr")
    print(f"  Pixel RMSE       : {pixel_rmse:.4f}  nW/cm²/sr")
    print(f"  Mean-radiance MAE: {mean_mae:.4f}  nW/cm²/sr  "
          f"(cf. SARIMA 0.838, LSTM 0.981)")
    print("=" * 55 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n[ConvLSTM] Step 3 — train_convlstm.py\n")

    # 1. Load
    X_train, y_train, X_test, y_test, W, scaler = load_sequences()

    # 2. Split
    X_fit, y_fit, X_val, y_val = split_train_val(X_train, y_train)

    # 3. Train
    model, history = train(X_fit, y_fit, X_val, y_val)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val   = float(min(history.history["val_loss"]))
    n_epochs   = len(history.history["loss"])
    print(f"\n[INFO] Training finished — {n_epochs} epochs  |  "
          f"Best epoch: {best_epoch}  |  Best val MAE: {best_val:.6f}")

    # 4. Save history
    hist = save_history(history)

    # 5. Loss curve
    plot_loss_curve(hist)

    # 6. Sample predictions plot
    plot_sample_predictions(model, X_test, y_test, scaler)

    # 7. Quick eval
    quick_eval(model, X_test, y_test, scaler)

    print(f"[INFO] Model saved to {MODEL_PATH}")
    print("[ConvLSTM] Step 3 complete.")
    print("[ConvLSTM] Next: python models/convlstm/evaluate_convlstm.py\n")


if __name__ == "__main__":
    main()
