"""
models/lstm/prepare_sequences.py
==================================
STEP 1 — Sliding-window sequence preparation for the LSTM model.

Reads : outputs/sarima/mean_brightness_clean.csv   (shared with SARIMA)
Writes: outputs/lstm/scaler.pkl                    (MinMaxScaler fit on train)
        outputs/lstm/train_test_split.csv           (date, mean_rad, split)
        outputs/lstm/sequences.npz                  (X_train, y_train, X_test, y_test)

Train / Test split (identical to SARIMA for fair comparison):
  Train : Jan 2014 – Dec 2023  (120 months)
  Test  : Jan 2024 – Dec 2025  (24 months)

Sliding window:
  For each position i, X[i] = series[i : i+W],  y[i] = series[i+W]
  Shape: X → (samples, window_size, 1),  y → (samples,)

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/lstm/prepare_sequences.py [--window W]
"""

import os
import sys
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF CPU warnings

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SARIMA_DIR = os.path.join(ROOT, "outputs", "sarima")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "lstm")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")

INPUT_CSV   = os.path.join(SARIMA_DIR, "mean_brightness_clean.csv")
SCALER_PKL  = os.path.join(OUTPUT_DIR, "scaler.pkl")
SPLIT_CSV   = os.path.join(OUTPUT_DIR, "train_test_split.csv")
SEQ_NPZ     = os.path.join(OUTPUT_DIR, "sequences.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─── Split boundary (same as SARIMA) ─────────────────────────────────────────
TRAIN_END  = pd.Timestamp("2023-12-01")
TEST_START = pd.Timestamp("2024-01-01")

# ─── Default window size ──────────────────────────────────────────────────────
DEFAULT_WINDOW = 12   # one full seasonal cycle


# ─── 1. Load & validate ───────────────────────────────────────────────────────

def load_series() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not os.path.exists(INPUT_CSV):
        sys.exit(f"[ERROR] Input not found: {INPUT_CSV}\n"
                 "Run models/sarima/clean_brightness.py first.")

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df["mean_rad"].isna().any():
        n_nan = df["mean_rad"].isna().sum()
        sys.exit(f"[ERROR] {n_nan} NaN values remain in mean_rad. "
                 "Re-run clean_brightness.py.")

    train = df[df["date"] <= TRAIN_END]["mean_rad"].reset_index(drop=True)
    test  = df[df["date"] >= TEST_START]["mean_rad"].reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} months  "
          f"({df['date'].iloc[0].strftime('%Y-%m')} – "
          f"{df['date'].iloc[-1].strftime('%Y-%m')})")
    print(f"[INFO] Train: {len(train)} months  |  Test: {len(test)} months")

    return df, train, test


# ─── 2. Fit scaler on train only (no leakage) ────────────────────────────────

def fit_scaler(train_vals: np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_vals.reshape(-1, 1))
    print(f"[INFO] Scaler fit on train  "
          f"(data_min={scaler.data_min_[0]:.4f}, "
          f"data_max={scaler.data_max_[0]:.4f})")
    return scaler


# ─── 3. Build sliding-window sequences ───────────────────────────────────────

def make_sequences(scaled_series: np.ndarray,
                   window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D scaled array into (X, y) pairs.

    X[i] shape: (window_size, 1)   — look-back window
    y[i]      : scalar             — next value to predict
    """
    X, y = [], []
    for i in range(len(scaled_series) - window_size):
        X.append(scaled_series[i : i + window_size].reshape(-1, 1))
        y.append(scaled_series[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─── 4. Save train/test split CSV (shared schema with SARIMA) ─────────────────

def save_split_csv(df: pd.DataFrame) -> None:
    split_col = df["date"].apply(
        lambda d: "train" if d <= TRAIN_END else "test"
    )
    out = pd.DataFrame({
        "date":     df["date"],
        "mean_rad": df["mean_rad"],
        "split":    split_col,
    })
    out.to_csv(SPLIT_CSV, index=False)
    print(f"[INFO] Saved → {SPLIT_CSV}")


# ─── 5. Main ──────────────────────────────────────────────────────────────────

def main(window_size: int) -> None:
    print("=" * 60)
    print("LSTM Step 1 — Prepare Sequences")
    print(f"  Window size : {window_size} months")
    print("=" * 60)

    # --- Load ---
    df, train_series, test_series = load_series()

    train_vals = train_series.values.astype(np.float32)
    test_vals  = test_series.values.astype(np.float32)

    # --- Scale (fit on train ONLY) ---
    scaler = fit_scaler(train_vals)
    joblib.dump(scaler, SCALER_PKL)
    print(f"[INFO] Saved → {SCALER_PKL}")

    train_scaled = scaler.transform(train_vals.reshape(-1, 1)).flatten()

    # For test sequences we need the tail of train as context seed.
    # Concatenate last (window_size) train months + full test.
    full_scaled = scaler.transform(
        np.concatenate([train_vals, test_vals]).reshape(-1, 1)
    ).flatten()
    train_scaled_full = full_scaled[:len(train_vals)]
    test_seed         = full_scaled[len(train_vals) - window_size:]  # seed + test

    # --- Build sequences ---
    X_train, y_train = make_sequences(train_scaled_full, window_size)
    X_test,  y_test  = make_sequences(test_seed,         window_size)

    print(f"\n[INFO] Sequence shapes:")
    print(f"       X_train : {X_train.shape}   y_train : {y_train.shape}")
    print(f"       X_test  : {X_test.shape}    y_test  : {y_test.shape}")

    # Sanity check: X_test should have exactly len(test_series) samples
    if X_test.shape[0] != len(test_series):
        sys.exit(f"[ERROR] X_test length mismatch: "
                 f"got {X_test.shape[0]}, expected {len(test_series)}")

    # --- Save sequences ---
    np.savez(SEQ_NPZ,
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test,
             window_size=np.array([window_size]))
    print(f"[INFO] Saved → {SEQ_NPZ}")

    # --- Save split CSV ---
    save_split_csv(df)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  CHECK — inverse-transform sanity:")
    y_train_orig = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig  = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    print(f"  Train target range : [{y_train_orig.min():.3f}, {y_train_orig.max():.3f}]")
    print(f"  Test  target range : [{y_test_orig.min():.3f}, {y_test_orig.max():.3f}]")
    print(f"  Scaled train range : [{y_train.min():.4f}, {y_train.max():.4f}]  (expect 0–1)")
    print("=" * 60)
    print("\n✓ Step 1 complete. Outputs:")
    print(f"  {SCALER_PKL}")
    print(f"  {SPLIT_CSV}")
    print(f"  {SEQ_NPZ}")
    print("\nNext: python models/lstm/build_model.py   (or tune.py for hyperparameter search)")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM — Step 1: Prepare Sequences")
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Look-back window size in months (default: {DEFAULT_WINDOW})"
    )
    args = parser.parse_args()

    if args.window < 1 or args.window > 36:
        sys.exit("[ERROR] --window must be between 1 and 36.")

    main(args.window)
