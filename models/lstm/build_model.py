"""
models/lstm/build_model.py
============================
STEP 2 — LSTM architecture definition.

Defines build_model() which is imported by train.py and tune.py.
Can also be run standalone to print the model summary.

Architecture:
  Input  →  (window_size, 1)
  LSTM(units_1, return_sequences=True)
  Dropout(dropout)
  LSTM(units_2, return_sequences=False)
  Dropout(dropout)
  Dense(16, activation='relu')
  Dense(1)

Compile:
  Loss      : MAE   (robust to spikes in radiance data)
  Optimizer : Adam
  Metric    : RMSE

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/lstm/build_model.py          # prints summary with defaults
    python models/lstm/build_model.py --window 12 --units1 64 --units2 32
"""

import os
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF CPU/GPU info messages

import tensorflow as tf
from tensorflow import keras

# ─── Default / baseline hyperparameters ──────────────────────────────────────
DEFAULTS = dict(
    window_size = 12,
    units_1     = 64,
    units_2     = 32,
    dropout     = 0.2,
    lr          = 1e-3,
)


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(
    window_size : int   = DEFAULTS["window_size"],
    units_1     : int   = DEFAULTS["units_1"],
    units_2     : int   = DEFAULTS["units_2"],
    dropout     : float = DEFAULTS["dropout"],
    lr          : float = DEFAULTS["lr"],
) -> keras.Model:
    """
    Build and compile a stacked LSTM regression model.

    Parameters
    ----------
    window_size : int
        Number of look-back time steps (= second dim of input tensor).
    units_1 : int
        Number of units in the first LSTM layer.
    units_2 : int
        Number of units in the second LSTM layer.
    dropout : float
        Dropout rate applied after each LSTM layer (same value for both).
    lr : float
        Adam learning rate.

    Returns
    -------
    keras.Model
        Compiled model ready for training.
    """
    inputs = keras.Input(shape=(window_size, 1), name="input")

    # First LSTM — passes full sequence to the next layer
    x = keras.layers.LSTM(units_1, return_sequences=True, name="lstm_1")(inputs)
    x = keras.layers.Dropout(dropout, name="dropout_1")(x)

    # Second LSTM — outputs only the last hidden state
    x = keras.layers.LSTM(units_2, return_sequences=False, name="lstm_2")(x)
    x = keras.layers.Dropout(dropout, name="dropout_2")(x)

    # Dense bottleneck + regression head
    x = keras.layers.Dense(16, activation="relu", name="dense_hidden")(x)
    outputs = keras.layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="NTL_LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mae",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    return model


# ─── Standalone summary ───────────────────────────────────────────────────────

def main(window_size, units_1, units_2, dropout, lr):
    print("=" * 60)
    print("LSTM Step 2 — Model Architecture")
    print("=" * 60)

    model = build_model(
        window_size=window_size,
        units_1=units_1,
        units_2=units_2,
        dropout=dropout,
        lr=lr,
    )
    model.summary()

    total_params = model.count_params()
    print(f"\n  Input shape  : (batch, {window_size}, 1)")
    print(f"  Output shape : (batch, 1)")
    print(f"  Total params : {total_params:,}")
    print(f"  Loss         : MAE")
    print(f"  Optimizer    : Adam  lr={lr}")
    print("\n✓ Step 2 complete — build_model() is ready to import.")
    print("  Next: python models/lstm/tune.py   (optional hyperparameter search)")
    print("        python models/lstm/train.py  (fits with current / best params)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM — Step 2: Architecture")
    parser.add_argument("--window",  type=int,   default=DEFAULTS["window_size"])
    parser.add_argument("--units1",  type=int,   default=DEFAULTS["units_1"])
    parser.add_argument("--units2",  type=int,   default=DEFAULTS["units_2"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--lr",      type=float, default=DEFAULTS["lr"])
    args = parser.parse_args()
    main(args.window, args.units1, args.units2, args.dropout, args.lr)
