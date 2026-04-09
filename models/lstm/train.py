"""
models/lstm/train.py
======================
STEP 3 — Train the LSTM model.

Reads : outputs/lstm/sequences.npz       (from Step 1)
        outputs/lstm/scaler.pkl           (from Step 1)
        outputs/lstm/best_params.json     (from Step 4, optional)

Saves : outputs/lstm/lstm_model.keras     (best weights via ModelCheckpoint)
        outputs/lstm/training_history.json
        outputs/lstm/plots/loss_curve.png
        outputs/lstm/plots/train_test_split.png

Validation split (of the 108 train sequences):
  First 84  → fit      (Jan 2014 – Dec 2020, ~7 years)
  Last  24  → validate (Jan 2021 – Dec 2023, ~2 years)
  Test  24  → held out (Jan 2024 – Dec 2025)

Callbacks:
  EarlyStopping     patience=25, restore_best_weights=True
  ReduceLROnPlateau patience=10, factor=0.5, min_lr=1e-5
  ModelCheckpoint   saves best val_loss model

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/lstm/train.py
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras

# Import build_model from the same package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm.build_model import build_model, DEFAULTS

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

OUTPUT_DIR = _cities.get_lstm_dir(CITY, ROOT)
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")

SEQ_NPZ      = os.path.join(OUTPUT_DIR, "sequences.npz")
SCALER_PKL   = os.path.join(OUTPUT_DIR, "scaler.pkl")
PARAMS_JSON  = os.path.join(OUTPUT_DIR, "best_params.json")
MODEL_PATH   = os.path.join(OUTPUT_DIR, "lstm_model.keras")
HISTORY_JSON = os.path.join(OUTPUT_DIR, "training_history.json")
SPLIT_CSV    = os.path.join(OUTPUT_DIR, "train_test_split.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Training constants ───────────────────────────────────────────────────────
EPOCHS     = 300
BATCH_SIZE = 16
VAL_SPLIT  = 24    # last N train sequences → validation set
SEED       = 42


# ─── 1. Load sequences ────────────────────────────────────────────────────────

def load_sequences():
    if not os.path.exists(SEQ_NPZ):
        sys.exit(f"[ERROR] {SEQ_NPZ} not found. Run prepare_sequences.py first.")

    data = np.load(SEQ_NPZ)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    window_size = int(data["window_size"].flat[0])

    print(f"[INFO] Sequences loaded:")
    print(f"       X_train {X_train.shape}  y_train {y_train.shape}")
    print(f"       X_test  {X_test.shape}   y_test  {y_test.shape}")
    print(f"       Window size: {window_size} months")
    return X_train, y_train, X_test, y_test, window_size


# ─── 2. Load hyperparameters (best_params.json if exists, else DEFAULTS) ──────

def load_params(window_size: int) -> dict:
    if os.path.exists(PARAMS_JSON):
        with open(PARAMS_JSON) as f:
            params = json.load(f)
        print(f"[INFO] Using hyperparameters from best_params.json: {params}")
    else:
        params = {**DEFAULTS, "window_size": window_size}
        print(f"[INFO] best_params.json not found — using baseline defaults: {params}")
    return params


# ─── 3. Validation split ──────────────────────────────────────────────────────

def split_train_val(X_train, y_train):
    X_tr = X_train[:-VAL_SPLIT]
    y_tr = y_train[:-VAL_SPLIT]
    X_val = X_train[-VAL_SPLIT:]
    y_val = y_train[-VAL_SPLIT:]
    print(f"[INFO] Train sequences: {len(X_tr)}  |  Val sequences: {len(X_val)}")
    return X_tr, y_tr, X_val, y_val


# ─── 4. Build callbacks ───────────────────────────────────────────────────────

def make_callbacks() -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


# ─── 5. Plot loss curve ───────────────────────────────────────────────────────

def plot_loss_curve(history: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"],     label="Train MAE", color="#4C72B0", linewidth=1.8)
    ax.plot(epochs, history["val_loss"], label="Val MAE",   color="#DD8452", linewidth=1.8)

    best_ep = int(np.argmin(history["val_loss"])) + 1
    best_val = min(history["val_loss"])
    ax.axvline(best_ep, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"Best epoch {best_ep}\nval MAE={best_val:.4f}",
        xy=(best_ep, best_val),
        xytext=(best_ep + max(1, len(epochs) * 0.05), best_val * 1.05),
        fontsize=8, color="gray",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (scaled)")
    ax.set_title("LSTM Training — Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "loss_curve.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot → {out}")


# ─── 6. Plot train / test predictions ────────────────────────────────────────

def plot_train_test(model, X_train, y_train, X_test, y_test, scaler) -> None:
    pred_train_s = model.predict(X_train, verbose=0).flatten()
    pred_test_s  = model.predict(X_test,  verbose=0).flatten()

    # Inverse-transform to original radiance units
    actual_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    actual_test  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_train   = scaler.inverse_transform(pred_train_s.reshape(-1, 1)).flatten()
    pred_test    = scaler.inverse_transform(pred_test_s.reshape(-1, 1)).flatten()

    # Rebuild date axis from split CSV
    split_df = pd.read_csv(SPLIT_CSV, parse_dates=["date"])
    train_dates = split_df[split_df["split"] == "train"]["date"].values
    test_dates  = split_df[split_df["split"] == "test"]["date"].values

    # train sequences start at index window_size from the train-set start
    window_size = X_train.shape[1]
    train_seq_dates = train_dates[window_size:]
    test_seq_dates  = test_dates

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(split_df["date"], split_df["mean_rad"],
            color="#aaaaaa", linewidth=1, label="Actual (full)", zorder=1)
    ax.plot(train_seq_dates, pred_train,
            color="#4C72B0", linewidth=1.6, label="LSTM fit (train)", zorder=2)
    ax.plot(test_seq_dates, pred_test,
            color="#D62728", linewidth=2, linestyle="--", label="LSTM forecast (test)", zorder=3)
    ax.plot(test_seq_dates, actual_test,
            color="#2CA02C", linewidth=2, label="Actual (test)", zorder=4)

    ax.axvline(pd.Timestamp("2024-01-01"), color="gray",
               linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.set_title("LSTM — Train Fit & Test Forecast vs Actual")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "train_test_split.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot → {out}")


# ─── 7. Main ──────────────────────────────────────────────────────────────────

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("LSTM Step 3 — Training")
    print("=" * 60)

    # --- Load data ---
    X_train, y_train, X_test, y_test, window_size = load_sequences()
    scaler = joblib.load(SCALER_PKL)

    # --- Hyperparameters ---
    params = load_params(window_size)

    # --- Validation split ---
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)

    # --- Build model ---
    model = build_model(
        window_size = params.get("window_size", window_size),
        units_1     = params.get("units_1",     DEFAULTS["units_1"]),
        units_2     = params.get("units_2",     DEFAULTS["units_2"]),
        dropout     = params.get("dropout",     DEFAULTS["dropout"]),
        lr          = params.get("lr",          DEFAULTS["lr"]),
    )
    model.summary(print_fn=lambda x: None)   # silent summary
    print(f"[INFO] Model params: {model.count_params():,}")

    # --- Train ---
    print(f"\n[INFO] Training  (max {EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"early stop patience=25) …\n")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=make_callbacks(),
        verbose=1,
    )

    # --- Save history ---
    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    with open(HISTORY_JSON, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"\n[INFO] Saved → {HISTORY_JSON}")
    print(f"[INFO] Saved → {MODEL_PATH}")

    # --- Diagnostics ---
    best_epoch = int(np.argmin(hist_dict["val_loss"])) + 1
    best_val   = min(hist_dict["val_loss"])
    epochs_ran = len(hist_dict["loss"])
    print(f"\n  Epochs run    : {epochs_ran}  (stopped at epoch {best_epoch})")
    print(f"  Best val MAE  : {best_val:.5f}  (scaled units)")

    # --- Plots ---
    plot_loss_curve(hist_dict)
    plot_train_test(model, X_train, y_train, X_test, y_test, scaler)

    # --- Quick test evaluation (scaled) ---
    pred_test_s  = model.predict(X_test, verbose=0).flatten()
    actual_test  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_test    = scaler.inverse_transform(pred_test_s.reshape(-1, 1)).flatten()
    mae_test = float(np.mean(np.abs(actual_test - pred_test)))
    print(f"\n  Quick test MAE (original units): {mae_test:.4f} nW/cm²/sr")

    print("\n" + "=" * 60)
    print("✓ Step 3 complete. Outputs:")
    print(f"  {MODEL_PATH}")
    print(f"  {HISTORY_JSON}")
    print(f"  {os.path.join(PLOTS_DIR, 'loss_curve.png')}")
    print(f"  {os.path.join(PLOTS_DIR, 'train_test_split.png')}")
    print("\nNext: python models/lstm/evaluate.py")


if __name__ == "__main__":
    main()
