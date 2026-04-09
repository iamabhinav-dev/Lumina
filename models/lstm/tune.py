"""
models/lstm/tune.py
=====================
STEP 4 — Hyperparameter search with Keras Tuner (RandomSearch).

Reads : outputs/sarima/mean_brightness_clean.csv  (raw series)
        outputs/lstm/scaler.pkl                    (fitted scaler from Step 1)

Saves : outputs/lstm/best_params.json              (winning hyperparameters)
        outputs/lstm/tuner/                        (Keras Tuner trial cache)

Search space:
  window_size : [6, 12, 18, 24]  months
  units_1     : [32, 64, 128]    LSTM layer 1
  units_2     : [16, 32, 64]     LSTM layer 2
  dropout     : [0.1, 0.2, 0.3]
  lr          : [1e-3, 5e-4, 1e-4]

Strategy:
  - Keras Tuner HyperModel (overrides build + fit) so window_size affects
    both the model input shape AND the sequence construction per trial.
  - max_trials=30, objective = val_loss (MAE)
  - After search: saves best_params.json, then re-runs train.py with those params.

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/lstm/tune.py
"""

import os
import sys
import json
import warnings
import joblib
import subprocess
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lstm.build_model import build_model

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SARIMA_DIR = os.path.join(ROOT, "outputs", "sarima")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "lstm")
TUNER_DIR  = os.path.join(OUTPUT_DIR, "tuner")

INPUT_CSV   = os.path.join(SARIMA_DIR, "mean_brightness_clean.csv")
SCALER_PKL  = os.path.join(OUTPUT_DIR, "scaler.pkl")
PARAMS_JSON = os.path.join(OUTPUT_DIR, "best_params.json")

os.makedirs(TUNER_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
TRAIN_END  = pd.Timestamp("2023-12-01")
TEST_START = pd.Timestamp("2024-01-01")
VAL_SPLIT  = 24     # last N train sequences → validation
MAX_TRIALS = 30
EPOCHS     = 100    # per trial (early stopping will cut this down)
BATCH_SIZE = 16
SEED       = 42


# ─── Sequence builder (called per trial for each window_size) ─────────────────

def build_sequences(window_size: int, scaler) -> tuple:
    """
    Re-build train/val sequences for a given window_size.
    Scaler is already fit on train — no leakage.
    """
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train_vals = df[df["date"] <= TRAIN_END]["mean_rad"].values.astype(np.float32)
    test_vals  = df[df["date"] >= TEST_START]["mean_rad"].values.astype(np.float32)

    full_vals   = np.concatenate([train_vals, test_vals])
    full_scaled = scaler.transform(full_vals.reshape(-1, 1)).flatten()

    train_scaled = full_scaled[:len(train_vals)]
    test_seed    = full_scaled[len(train_vals) - window_size:]

    def make_seqs(arr):
        X, y = [], []
        for i in range(len(arr) - window_size):
            X.append(arr[i : i + window_size].reshape(-1, 1))
            y.append(arr[i + window_size])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train, y_train = make_seqs(train_scaled)
    X_tr  = X_train[:-VAL_SPLIT]
    y_tr  = y_train[:-VAL_SPLIT]
    X_val = X_train[-VAL_SPLIT:]
    y_val = y_train[-VAL_SPLIT:]

    return X_tr, y_tr, X_val, y_val


# ─── HyperModel ───────────────────────────────────────────────────────────────

class NTLHyperModel(kt.HyperModel):
    """
    Keras Tuner HyperModel that rebuilds sequences per trial
    so window_size is a true hyperparameter affecting both
    input shape and sequence construction.
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        window_size = hp.Choice("window_size", [6, 12, 18, 24])
        units_1     = hp.Choice("units_1",  [32, 64, 128])
        units_2     = hp.Choice("units_2",  [16, 32, 64])
        dropout     = hp.Choice("dropout",  [0.1, 0.2, 0.3])
        lr          = hp.Choice("lr",       [1e-3, 5e-4, 1e-4])
        return build_model(window_size, units_1, units_2, dropout, lr)

    def fit(self, hp: kt.HyperParameters, model: keras.Model, *args, **kwargs):
        window_size = hp.get("window_size")
        X_tr, y_tr, X_val, y_val = build_sequences(window_size, self.scaler)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=7, min_lr=1e-5,
            ),
        ]

        return model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("LSTM Step 4 — Hyperparameter Search (Keras Tuner)")
    print(f"  max_trials : {MAX_TRIALS}")
    print(f"  epochs/trial: {EPOCHS}  (with early stopping)")
    print(f"  Search space: window∈[6,12,18,24], units1∈[32,64,128],")
    print(f"                units2∈[16,32,64], dropout∈[0.1,0.2,0.3],")
    print(f"                lr∈[1e-3,5e-4,1e-4]")
    print("=" * 60)

    if not os.path.exists(SCALER_PKL):
        sys.exit(f"[ERROR] {SCALER_PKL} not found. Run prepare_sequences.py first.")

    scaler = joblib.load(SCALER_PKL)
    hypermodel = NTLHyperModel(scaler=scaler)

    tuner = kt.RandomSearch(
        hypermodel=hypermodel,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=MAX_TRIALS,
        seed=SEED,
        directory=TUNER_DIR,
        project_name="ntl_lstm",
        overwrite=True,
    )

    print(f"\n[INFO] Starting search — {MAX_TRIALS} trials …\n")
    tuner.search()

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TOP 5 TRIALS")
    print("=" * 60)
    top_trials = tuner.oracle.get_best_trials(num_trials=5)
    for rank, trial in enumerate(top_trials, 1):
        hp_vals = trial.hyperparameters.values
        score   = trial.score
        print(f"  #{rank}  val_loss={score:.5f}  |  {hp_vals}")

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_params = {
        "window_size" : best_hp.get("window_size"),
        "units_1"     : best_hp.get("units_1"),
        "units_2"     : best_hp.get("units_2"),
        "dropout"     : best_hp.get("dropout"),
        "lr"          : best_hp.get("lr"),
        "source"      : "keras_tuner_random_search",
        "max_trials"  : MAX_TRIALS,
    }

    with open(PARAMS_JSON, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n[INFO] Best hyperparameters:")
    for k, v in best_params.items():
        print(f"       {k:15s}: {v}")
    print(f"\n[INFO] Saved → {PARAMS_JSON}")

    # ── If window_size changed, re-run prepare_sequences with new window ──────
    saved_window = best_params["window_size"]
    current_npz  = os.path.join(OUTPUT_DIR, "sequences.npz")
    rebuild_seqs = True
    if os.path.exists(current_npz):
        data = np.load(current_npz)
        current_window = int(data["window_size"].flat[0])
        rebuild_seqs = (current_window != saved_window)

    if rebuild_seqs:
        print(f"\n[INFO] window_size changed to {saved_window} — rebuilding sequences …")
        prep_script = os.path.join(
            ROOT, "models", "lstm", "prepare_sequences.py"
        )
        result = subprocess.run(
            [sys.executable, prep_script, "--window", str(saved_window)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[WARN] prepare_sequences.py failed:\n{result.stderr[-500:]}")
        else:
            print(result.stdout.strip())
    else:
        print(f"\n[INFO] window_size unchanged ({saved_window}) — sequences reuse OK.")

    print("\n" + "=" * 60)
    print("✓ Step 4 complete.")
    print(f"  {PARAMS_JSON}")
    print("\nNext: python models/lstm/train.py   (will use best_params.json)")


if __name__ == "__main__":
    main()
