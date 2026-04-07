"""
models/sarima/clean_brightness.py
===================================
STEP 2 — Clean & validate the mean brightness time series.

Reads : outputs/sarima/mean_brightness.csv
Writes: outputs/sarima/mean_brightness_clean.csv
Plot  : outputs/sarima/plots/cleaning_report.png

Cleaning operations (in order):
  1. Ensure complete monthly date index (fill gaps as NaN)
  2. Flag low-coverage months (valid_pixel_ratio < 0.5) → NaN
  3. Detect extreme outliers via 3×IQR rule → NaN
  4. Linear interpolation of all NaN values
  5. Add 3-month centred rolling average column (for visualisation)
  6. Save cleaned CSV + print summary

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/sarima/clean_brightness.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "sarima")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
INPUT_CSV  = os.path.join(OUTPUT_DIR, "mean_brightness.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "mean_brightness_clean.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Thresholds ───────────────────────────────────────────────────────────────
LOW_COVERAGE_THRESHOLD = 0.50   # flag month if < 50% valid pixels
IQR_MULTIPLIER         = 3.0    # conservative: flag only extreme outliers
# Kharagpur is a city — any monthly mean below this is physically impossible
# (sensor dropout / bad composite), not a real measurement.
CITY_FLOOR             = 0.5    # nW/cm²/sr


# ─── STEP 2a — Ensure complete monthly date index ────────────────────────────

def ensure_complete_index(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Insert NaN rows for any missing months in the date sequence."""
    full_idx = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS",   # Month Start
    )
    df = df.set_index("date").reindex(full_idx)
    df.index.name = "date"
    df = df.reset_index()

    # Re-fill year / month from the date index
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    inserted = df[df["mean_rad"].isna() & df["valid_pixel_ratio"].isna()]["date"].tolist()
    return df, inserted


# ─── STEP 2b — Flag low-coverage months ──────────────────────────────────────

def flag_low_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Set mean_rad to NaN for months with valid_pixel_ratio < threshold."""
    mask = df["valid_pixel_ratio"].notna() & (df["valid_pixel_ratio"] < LOW_COVERAGE_THRESHOLD)
    flagged = df.loc[mask, "date"].tolist()
    df.loc[mask, "mean_rad"] = np.nan
    return df, flagged


# ─── STEP 2c — Outlier detection (global IQR + city floor) ───────────────────

def flag_iqr_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Two-pass outlier detection:

      Pass 1 — Global IQR  (k=3, catches series-wide extremes on the high end)
      Pass 2 — City floor  (any value < CITY_FLOOR is physically impossible for
               an urban area — sensor dropout / bad monthly composite)

    Why NOT seasonal statistics:
      The NTL series has a clear upward trend (2014 mean ~3.5 → 2025 mean ~7.5).
      Applying per-month statistics across all years treats recent higher values as
      outliers, which would remove real urbanisation signal. Global IQR + a hard
      physical floor is the correct approach for a trending series.

    Returns: modified df, DataFrame of flagged rows (with original values),
             global lower fence, global upper fence.
    """
    flagged_rows = []

    # ── Pass 1: global IQR ───────────────────────────────────────────────────
    series = df["mean_rad"].dropna()
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - IQR_MULTIPLIER * iqr
    upper_fence = q3 + IQR_MULTIPLIER * iqr

    global_mask = df["mean_rad"].notna() & (
        (df["mean_rad"] < lower_fence) | (df["mean_rad"] > upper_fence)
    )
    if global_mask.any():
        flagged_rows.append(
            df.loc[global_mask, ["date", "mean_rad"]].copy().assign(reason="global_IQR")
        )
        df.loc[global_mask, "mean_rad"] = np.nan

    # ── Pass 2: city floor ───────────────────────────────────────────────────
    floor_mask = df["mean_rad"].notna() & (df["mean_rad"] < CITY_FLOOR)
    if floor_mask.any():
        flagged_rows.append(
            df.loc[floor_mask, ["date", "mean_rad"]].copy().assign(reason="city_floor")
        )
        df.loc[floor_mask, "mean_rad"] = np.nan

    flagged_vals = (
        pd.concat(flagged_rows, ignore_index=True)
        if flagged_rows
        else pd.DataFrame(columns=["date", "mean_rad", "reason"])
    )
    if not flagged_vals.empty:
        flagged_vals = flagged_vals.drop_duplicates(subset="date")

    return df, flagged_vals, lower_fence, upper_fence


# ─── STEP 2d — Linear interpolation ─────────────────────────────────────────

def interpolate_nans(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Linear interpolation of NaN values in mean_rad (time-ordered)."""
    n_before = df["mean_rad"].isna().sum()
    df = df.sort_values("date").reset_index(drop=True)
    df["mean_rad"] = df["mean_rad"].interpolate(method="linear", limit_direction="both")
    df["median_rad"] = df["median_rad"].interpolate(method="linear", limit_direction="both")
    df["std_rad"]    = df["std_rad"].interpolate(method="linear", limit_direction="both")
    return df, int(n_before)


# ─── STEP 2e — 3-month centred rolling average ───────────────────────────────

def add_rolling_mean(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add a centred rolling mean column. min_periods=2 avoids NaN at edges."""
    df = df.sort_values("date").reset_index(drop=True)
    df["mean_rad_smooth"] = (
        df["mean_rad"]
        .rolling(window=window, center=True, min_periods=2)
        .mean()
        .round(6)
    )
    return df


# ─── Diagnostic plot ─────────────────────────────────────────────────────────

def save_cleaning_plot(
    df_raw:     pd.DataFrame,
    df_clean:   pd.DataFrame,
    flagged_dates: list,
    lower_fence: float,
    upper_fence: float,
) -> None:
    """Side-by-side before/after plot with flagged points highlighted."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("VIIRS NTL Mean Brightness — Cleaning Report\nKharagpur", fontsize=14, fontweight="bold")

    dates_raw   = pd.to_datetime(df_raw["date"])
    dates_clean = pd.to_datetime(df_clean["date"])

    # ── Top: raw series ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dates_raw, df_raw["mean_rad"], color="#4C72B0", linewidth=1.3, label="Raw mean_rad")
    ax.axhline(lower_fence, color="red",    linestyle="--", linewidth=0.9, alpha=0.7, label=f"IQR lower ({lower_fence:.3f})")
    ax.axhline(upper_fence, color="orange", linestyle="--", linewidth=0.9, alpha=0.7, label=f"IQR upper ({upper_fence:.3f})")

    # Highlight flagged points
    flag_dt = [pd.Timestamp(d) for d in flagged_dates]
    for fd in flag_dt:
        raw_row = df_raw[df_raw["date"] == fd]
        if not raw_row.empty:
            ax.scatter(fd, raw_row["mean_rad"].values[0],
                       color="red", zorder=5, s=60)

    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.set_title("Before Cleaning")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Bottom: cleaned series ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(dates_clean, df_clean["mean_rad"],
             color="#2ca02c", linewidth=1.3, label="Cleaned mean_rad")
    ax2.plot(dates_clean, df_clean["mean_rad_smooth"],
             color="#ff7f0e", linewidth=1.5, linestyle="--",
             label="3-month rolling mean", alpha=0.85)

    # Mark interpolated positions
    for fd in flag_dt:
        cl_row = df_clean[df_clean["date"] == fd]
        if not cl_row.empty:
            ax2.scatter(fd, cl_row["mean_rad"].values[0],
                        color="magenta", zorder=5, s=60, marker="^",
                        label="_interpolated")

    interp_patch = mpatches.Patch(color="magenta", label="Interpolated value")
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(interp_patch)
    labels.append("Interpolated value")

    ax2.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax2.set_xlabel("Date")
    ax2.set_title("After Cleaning")
    ax2.legend(handles=handles, labels=labels, fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "cleaning_report.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {out}")


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(
    df_raw:         pd.DataFrame,
    df_clean:       pd.DataFrame,
    inserted_dates: list,
    low_cov_dates:  list,
    flagged_vals:   pd.DataFrame,
    n_interpolated: int,
    lower_fence:    float,
    upper_fence:    float,
) -> None:
    print("\n" + "=" * 62)
    print("CLEANING SUMMARY")
    print("=" * 62)

    # ── 2a
    print(f"\n[2a] Date continuity")
    if inserted_dates:
        print(f"     Inserted {len(inserted_dates)} missing month(s):")
        for d in inserted_dates:
            print(f"       {d.date()}")
    else:
        print(f"     ✓ No gaps — all 144 months present")

    # ── 2b
    print(f"\n[2b] Low coverage (<{LOW_COVERAGE_THRESHOLD:.0%} valid pixels)")
    if low_cov_dates:
        print(f"     Flagged {len(low_cov_dates)} month(s):")
        for d in low_cov_dates:
            print(f"       {d.date()}")
    else:
        print(f"     ✓ No low-coverage months")

    # ── 2c
    print(f"\n[2c] Outlier detection  (global 3×IQR + city floor ≥{CITY_FLOOR})")
    print(f"     Global fence: [{lower_fence:.4f},  {upper_fence:.4f}]  nW/cm²/sr")
    if flagged_vals.empty:
        print(f"     ✓ No outliers detected")
    else:
        print(f"     Flagged {len(flagged_vals)} value(s):")
        for _, row in flagged_vals.iterrows():
            direction = "LOW" if row["mean_rad"] < CITY_FLOOR else "HIGH"
            reason = row.get("reason", "")
            print(f"       {row['date'].date()}  mean_rad = {row['mean_rad']:.4f}  [{direction}]  ({reason})")

    # ── 2d
    print(f"\n[2d] Linear interpolation")
    print(f"     Interpolated {n_interpolated} NaN value(s)")

    # ── 2e
    print(f"\n[2e] 3-month centred rolling mean → column 'mean_rad_smooth'")
    print(f"     Added successfully")

    # ── Final stats
    valid = df_clean["mean_rad"].dropna()
    print(f"\n── Cleaned series statistics ───────────────────────────────")
    print(f"   Rows            : {len(df_clean)}")
    print(f"   Remaining NaNs  : {df_clean['mean_rad'].isna().sum()}")
    print(f"   mean_rad range  : {valid.min():.4f} – {valid.max():.4f}")
    print(f"   mean_rad mean   : {valid.mean():.4f}")
    print(f"   mean_rad std    : {valid.std():.4f}")

    changed = (df_raw["mean_rad"].values != df_clean["mean_rad"].values)
    print(f"   Values changed  : {int(np.sum(changed))}")
    print("=" * 62)


# ─── Main ─────────────────────────────────────────────────────────────────────

def clean_brightness() -> pd.DataFrame:
    df_raw = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df_raw)} rows from {INPUT_CSV}")

    df = df_raw.copy()

    # ── 2a: complete index
    df, inserted = ensure_complete_index(df)

    # ── 2b: low coverage
    df, low_cov_flagged = flag_low_coverage(df)

    # ── 2c: IQR outliers  (capture original values before nullifying)
    df, flagged_vals, lower_fence, upper_fence = flag_iqr_outliers(df)

    # Collect all flagged dates for the plot
    all_flagged_dates = (
        list(inserted) +
        list(low_cov_flagged) +
        (flagged_vals["date"].tolist() if not flagged_vals.empty else [])
    )

    # ── 2d: interpolate
    df, n_interp = interpolate_nans(df)

    # ── 2e: rolling average
    df = add_rolling_mean(df)

    # ── Round floats for clean CSV
    for col in ["mean_rad", "median_rad", "std_rad", "min_rad", "max_rad", "mean_rad_smooth"]:
        if col in df.columns:
            df[col] = df[col].round(6)

    # Add a flag column so cleaning decisions are transparent
    df["cleaned"] = df["date"].isin(all_flagged_dates).astype(int)

    # ── Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned CSV → {OUTPUT_CSV}")

    # ── Summary + plot
    print_summary(df_raw, df, inserted, low_cov_flagged,
                  flagged_vals, n_interp, lower_fence, upper_fence)
    save_cleaning_plot(df_raw, df, all_flagged_dates, lower_fence, upper_fence)

    return df


if __name__ == "__main__":
    clean_brightness()
