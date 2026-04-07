"""
models/sarima/eda.py
=====================
STEP 3 — EDA & Stationarity Analysis

Reads : outputs/sarima/mean_brightness_clean.csv
Saves : outputs/sarima/plots/
          eda_01_timeseries.png
          eda_02_seasonal_boxplot.png
          eda_03_yoy_overlay.png
          eda_04_rolling_stats.png
          eda_05_acf_pacf.png
          eda_06_differenced_acf_pacf.png
Prints: ADF and KPSS stationarity test results + SARIMA order guidance

Usage:
    cd /home/abhinav/Desktop/BTP
    source venv/bin/activate
    python models/sarima/eda.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "sarima")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
INPUT_CSV  = os.path.join(OUTPUT_DIR, "mean_brightness_clean.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = sns.color_palette("tab10")

# ─── Load data ────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Full time series
# ═══════════════════════════════════════════════════════════════════════════════

def plot_timeseries(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df.index, df["mean_rad"], color="#4C72B0", linewidth=1.4,
            label="Mean radiance (cleaned)", zorder=3)
    ax.plot(df.index, df["mean_rad_smooth"], color="#FF7F0E", linewidth=2,
            linestyle="--", label="3-month rolling mean", zorder=4)

    # Annotate interpolated points
    interpolated = df[df["cleaned"] == 1]
    if not interpolated.empty:
        ax.scatter(interpolated.index, interpolated["mean_rad"],
                   color="red", zorder=5, s=55, label="Interpolated (artifact)")

    ax.set_title("VIIRS NTL Mean Brightness — Kharagpur (Jan 2014 – Dec 2025)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.set_xlabel("")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "eda_01_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Monthly seasonality box plot
# ═══════════════════════════════════════════════════════════════════════════════

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

def plot_seasonal_boxplot(df: pd.DataFrame) -> None:
    df2 = df.reset_index().copy()
    df2["month_name"] = df2["date"].dt.month.apply(lambda m: MONTH_ABBR[m - 1])
    df2["month_num"]  = df2["date"].dt.month

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(
        data=df2, x="month_name", y="mean_rad",
        order=MONTH_ABBR, palette="Blues_d",
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
        ax=ax,
    )
    ax.set_title("Monthly Seasonality — Mean Radiance by Calendar Month",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "eda_02_seasonal_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Year-over-year overlay
# ═══════════════════════════════════════════════════════════════════════════════

def plot_yoy(df: pd.DataFrame) -> None:
    df2 = df.reset_index().copy()
    years = sorted(df2["date"].dt.year.unique())
    cmap  = plt.cm.get_cmap("plasma", len(years))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, yr in enumerate(years):
        yr_data = df2[df2["date"].dt.year == yr].sort_values("date")
        months  = yr_data["date"].dt.month.values
        values  = yr_data["mean_rad"].values
        lw = 2.2 if yr in (2014, 2025) else 1.2
        ax.plot(months, values, color=cmap(i), linewidth=lw, label=str(yr))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.set_title("Year-over-Year Overlay — NTL Mean Radiance",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Radiance (nW/cm²/sr)")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "eda_03_yoy_overlay.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Rolling mean & std (12-month window)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_rolling_stats(df: pd.DataFrame) -> None:
    series  = df["mean_rad"]
    roll12  = series.rolling(window=12, center=False)
    r_mean  = roll12.mean()
    r_std   = roll12.std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("Rolling Statistics (12-month window) — Stationarity View",
                 fontsize=13, fontweight="bold")

    axes[0].plot(df.index, series,  color="#4C72B0", linewidth=1.2, alpha=0.5, label="mean_rad")
    axes[0].plot(df.index, r_mean,  color="#E74C3C", linewidth=2,   label="12m rolling mean")
    axes[0].set_ylabel("Mean Radiance")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, r_std, color="#27AE60", linewidth=1.8, label="12m rolling std")
    axes[1].set_ylabel("Rolling Std Dev")
    axes[1].set_xlabel("")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Annotation: if rolling mean drifts → non-stationary
    axes[0].annotate(
        "Rising rolling mean → non-stationary (trend present)",
        xy=(df.index[60], r_mean.iloc[60]),
        xytext=(df.index[24], r_mean.iloc[60] + 0.8),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8, color="gray",
    )

    plt.tight_layout()
    _save(fig, "eda_04_rolling_stats.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — ACF & PACF (raw series)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_acf_pacf(series: pd.Series, title_suffix: str, filename: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle(f"ACF & PACF — {title_suffix}", fontsize=13, fontweight="bold")

    plot_acf( series, lags=48, ax=axes[0], alpha=0.05,
              title="ACF (lags 0–48)", color="#4C72B0")
    plot_pacf(series, lags=48, ax=axes[1], alpha=0.05, method="ywm",
              title="PACF (lags 0–48)", color="#E74C3C")

    for ax in axes:
        ax.axvline(x=12, color="green", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axvline(x=24, color="green", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.axvline(x=36, color="green", linestyle="--", linewidth=0.9, alpha=0.3)
        ax.set_xlabel("Lag (months)")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# STATIONARITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_stationarity_tests(series: pd.Series, label: str) -> dict:
    """Run ADF and KPSS; return dict of results."""
    # ADF — H0: unit root (non-stationary). Reject H0 → stationary.
    adf_res = adfuller(series.dropna(), autolag="AIC")
    adf_stat, adf_p = adf_res[0], adf_res[1]
    adf_crit = adf_res[4]

    # KPSS — H0: stationary. Reject H0 → non-stationary.
    kpss_res = kpss(series.dropna(), regression="c", nlags="auto")
    kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
    kpss_crit = kpss_res[3]

    results = {
        "label":     label,
        "adf_stat":  adf_stat,
        "adf_p":     adf_p,
        "adf_crit":  adf_crit,
        "adf_stationary": adf_p < 0.05,
        "kpss_stat": kpss_stat,
        "kpss_p":    kpss_p,
        "kpss_crit": kpss_crit,
        "kpss_stationary": kpss_p > 0.05,
    }
    return results


def print_stationarity(r: dict) -> None:
    adf_verdict  = "STATIONARY"     if r["adf_stationary"]  else "NON-STATIONARY"
    kpss_verdict = "STATIONARY"     if r["kpss_stationary"] else "NON-STATIONARY"
    adf_colour   = "✓" if r["adf_stationary"]  else "✗"
    kpss_colour  = "✓" if r["kpss_stationary"] else "✗"

    print(f"\n  ── {r['label']} ─────────────────────────────────────────")
    print(f"  ADF test")
    print(f"    Statistic : {r['adf_stat']:.4f}")
    print(f"    p-value   : {r['adf_p']:.4f}")
    print(f"    Critical  : 1%={r['adf_crit']['1%']:.4f}  5%={r['adf_crit']['5%']:.4f}")
    print(f"    Verdict   : {adf_colour} {adf_verdict}  (H0=unit root; reject if p<0.05)")

    print(f"  KPSS test")
    print(f"    Statistic : {r['kpss_stat']:.4f}")
    print(f"    p-value   : {r['kpss_p']:.4f}")
    print(f"    Critical  : 1%={r['kpss_crit']['1%']:.4f}  5%={r['kpss_crit']['5%']:.4f}")
    print(f"    Verdict   : {kpss_colour} {kpss_verdict}  (H0=stationary; reject if p<0.05)")


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENCING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_differenced(df: pd.DataFrame) -> dict[str, pd.Series]:
    s = df["mean_rad"]
    return {
        "raw":              s,
        "first_diff":       s.diff().dropna(),
        "seasonal_diff":    s.diff(12).dropna(),
        "first+seasonal":   s.diff(12).diff().dropna(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SARIMA GUIDANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_sarima_guidance(test_results: dict) -> None:
    raw_stationary      = test_results["raw"]["adf_stationary"]      and test_results["raw"]["kpss_stationary"]
    fd_stationary       = test_results["first_diff"]["adf_stationary"] and test_results["first_diff"]["kpss_stationary"]
    sd_stationary       = test_results["seasonal_diff"]["adf_stationary"] and test_results["seasonal_diff"]["kpss_stationary"]
    fsd_stationary      = test_results["first+seasonal"]["adf_stationary"] and test_results["first+seasonal"]["kpss_stationary"]

    d  = 0 if raw_stationary else 1
    D  = 0
    if not raw_stationary:
        D = 0 if fd_stationary else 1

    print("\n" + "=" * 60)
    print("SARIMA ORDER GUIDANCE")
    print("=" * 60)
    print(f"  Raw series stationary       : {'Yes' if raw_stationary  else 'No'}")
    print(f"  After 1st diff stationary   : {'Yes' if fd_stationary   else 'No'}")
    print(f"  After seasonal diff stat.   : {'Yes' if sd_stationary   else 'No'}")
    print(f"  After 1st+seasonal diff st. : {'Yes' if fsd_stationary  else 'No'}")
    print(f"\n  Recommended  d = {d}  (non-seasonal differencing)")
    print(f"  Recommended  D = {D}  (seasonal differencing, m=12)")
    print()
    print("  Next: read p,q from ACF/PACF of the differenced series.")
    print("  Or run auto_arima in train.py to select all orders automatically.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    df = load_data()
    series = df["mean_rad"]
    print(f"Loaded {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})\n")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("Generating plots...")
    plot_timeseries(df)
    plot_seasonal_boxplot(df)
    plot_yoy(df)
    plot_rolling_stats(df)

    diff_series = prepare_differenced(df)

    plot_acf_pacf(series,
                  "Raw series (mean_rad)",
                  "eda_05_acf_pacf_raw.png")
    plot_acf_pacf(diff_series["first_diff"],
                  "First-differenced (d=1)",
                  "eda_06_acf_pacf_diff1.png")
    plot_acf_pacf(diff_series["first+seasonal"],
                  "First + Seasonal differenced (d=1, D=1, m=12)",
                  "eda_07_acf_pacf_diff1_s1.png")

    # ── Stationarity tests ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STATIONARITY TESTS")
    print("=" * 60)

    test_results = {}
    for key, s in diff_series.items():
        r = run_stationarity_tests(s, key)
        test_results[key] = r
        print_stationarity(r)

    # ── SARIMA guidance ──────────────────────────────────────────────────────
    print_sarima_guidance(test_results)


if __name__ == "__main__":
    main()
