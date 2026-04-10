"""
08_validation.py — 2026 Forecast Validation
Compares Jan–Mar 2026 actual VIIRS NTL against SARIMA / LSTM / ConvLSTM forecasts.
"""

import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.join(PAGE_DIR, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import cities as _cities

st.set_page_config(page_title="2026 Validation", page_icon="✅", layout="wide")

# ─── City ────────────────────────────────────────────────────────────────────
CITY = st.session_state.get("city", "kharagpur")
CFG  = _cities.get_city(CITY)

SARIMA_DIR = _cities.get_sarima_dir(CITY, ROOT)
LSTM_DIR   = _cities.get_lstm_dir(CITY, ROOT)
CLSTM_DIR  = _cities.get_convlstm_dir(CITY, ROOT)

# validation_2026.json lives in the parent of the sarima dir
VAL_JSON_PATH = os.path.join(os.path.dirname(SARIMA_DIR), "validation_2026.json")

MONTH_LABELS = {"2026-01-01": "Jan 2026", "2026-02-01": "Feb 2026", "2026-03-01": "Mar 2026"}

# ─── Load helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading validation results…")
def load_validation(city: str) -> dict | None:
    sarima_dir = _cities.get_sarima_dir(city, ROOT)
    path = os.path.join(os.path.dirname(sarima_dir), "validation_2026.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading historical time series…")
def load_history(city: str) -> pd.DataFrame:
    sarima_dir = _cities.get_sarima_dir(city, ROOT)
    df = pd.read_csv(os.path.join(sarima_dir, "train_test_split.csv"), parse_dates=["date"])
    return df


@st.cache_data(show_spinner="Loading forecasts…")
def load_forecasts(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sar = pd.read_csv(os.path.join(_cities.get_sarima_dir(city, ROOT), "forecast.csv"),
                      parse_dates=["date"])
    lstm = pd.read_csv(os.path.join(_cities.get_lstm_dir(city, ROOT), "forecast.csv"),
                       parse_dates=["date"])
    return sar, lstm


@st.cache_data(show_spinner="Loading spatial frames…")
def load_spatial(city: str) -> tuple[np.ndarray | None, np.ndarray | None, list]:
    clstm_dir  = _cities.get_convlstm_dir(city, ROOT)
    act_path   = os.path.join(clstm_dir, "actual_frames_2026.npz")
    fc_path    = os.path.join(clstm_dir, "forecast_frames.npz")
    if not os.path.exists(act_path) or not os.path.exists(fc_path):
        return None, None, []
    act_data  = np.load(act_path)
    fc_data   = np.load(fc_path)
    return act_data["frames"], fc_data["mean_forecast"], list(act_data["dates"])


@st.cache_data(show_spinner="Loading pixel MAE…")
def load_pixel_mae(city: str, month_tag: str) -> np.ndarray | None:
    clstm_dir = _cities.get_convlstm_dir(city, ROOT)
    path = os.path.join(clstm_dir, f"val_pixel_mae_{month_tag}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["pixel_mae"]


def _arr_to_img(arr: np.ndarray, cmap: str, vmin=None, vmax=None) -> Image.Image:
    """Convert 2D numpy array to PIL Image using matplotlib colormap."""
    norm = plt.Normalize(
        vmin=float(np.nanmin(arr)) if vmin is None else vmin,
        vmax=float(np.nanmax(arr)) if vmax is None else vmax,
    )
    rgba = plt.get_cmap(cmap)(norm(arr))
    return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))


def _color_pct(val: float) -> str:
    if val < 5:
        return "🟢"
    if val < 10:
        return "🟡"
    return "🔴"


# ─── Page ─────────────────────────────────────────────────────────────────────

st.title("✅ 2026 Forecast Validation")
st.caption(
    f"Jan – Mar 2026 real VIIRS NTL data is now available. "
    f"Below we compare actual observed radiance against what each model predicted "
    f"for **{CFG['display_name']}**."
)

val = load_validation(CITY)

if val is None:
    st.error(
        "validation_2026.json not found. "
        "Run `python src/extract_actuals.py --all` "
        "then `python src/compute_validation.py --all` first."
    )
    st.stop()

months    = val["months"]
best_mdl  = val["best_model"]
errors    = val["mean_pct_errors"]
sar_rows  = val["sarima"]
lstm_rows = val["lstm"]
cl_rows   = val["convlstm"]

# ═══════════════════════════════════════════════════════════
# SECTION D — Model ranking (put at top for quick summary)
# ═══════════════════════════════════════════════════════════
st.subheader("🏆 Model Accuracy Summary (3-Month Mean % Error)")

c1, c2, c3 = st.columns(3)
for col, key, label, icon in [
    (c1, "sarima",   "SARIMA",   "📈"),
    (c2, "lstm",     "LSTM",     "🧠"),
    (c3, "convlstm", "ConvLSTM", "🖼️"),
]:
    val_pct = errors.get(key)
    badge   = " 🥇 Best" if key == best_mdl else ""
    with col:
        if val_pct is not None:
            st.metric(
                label=f"{icon} {label}{badge}",
                value=f"{val_pct:.2f}%",
                help="Mean absolute % error across Jan–Mar 2026",
            )
        else:
            st.metric(label=f"{icon} {label}", value="N/A")

st.divider()

# ═══════════════════════════════════════════════════════════
# SECTION A — Scalar accuracy table
# ═══════════════════════════════════════════════════════════
st.subheader("📋 Month-by-Month: SARIMA & LSTM")

table_rows = []
for s, l in zip(sar_rows, lstm_rows):
    label = MONTH_LABELS.get(s["date"], s["date"])
    table_rows.append({
        "Month":         label,
        "Actual":        round(s["actual"], 3),
        "SARIMA Pred":   round(s["predicted"], 3),
        "SARIMA Err%":   f"{_color_pct(s['pct_error'])} {s['pct_error']:.2f}%",
        "LSTM Pred":     round(l["predicted"], 3),
        "LSTM Err%":     f"{_color_pct(l['pct_error'])} {l['pct_error']:.2f}%",
    })

# Mean row
mean_sar = errors.get("sarima", 0)
mean_lstm = errors.get("lstm", 0)
table_rows.append({
    "Month":        "**Mean**",
    "Actual":       "—",
    "SARIMA Pred":  "—",
    "SARIMA Err%":  f"{_color_pct(mean_sar)} **{mean_sar:.2f}%**",
    "LSTM Pred":    "—",
    "LSTM Err%":    f"{_color_pct(mean_lstm)} **{mean_lstm:.2f}%**",
})

st.dataframe(
    pd.DataFrame(table_rows),
    use_container_width=True,
    hide_index=True,
)
st.caption("🟢 < 5% error  |  🟡 5–10%  |  🔴 > 10%")

st.divider()

# ═══════════════════════════════════════════════════════════
# SECTION B — Time Series Chart
# ═══════════════════════════════════════════════════════════
st.subheader("📈 Time Series: Actual vs Forecasted (2025 Context + 2026 Validation)")

hist_df      = load_history(CITY)
sar_fc, lstm_fc = load_forecasts(CITY)

# Context: last 18 months of history
context_cut  = hist_df["date"].max() - pd.DateOffset(months=18)
hist_ctx     = hist_df[hist_df["date"] >= context_cut]

actuals_ext  = pd.DataFrame(sar_rows)[["date", "actual"]].copy()
actuals_ext["date"] = pd.to_datetime(actuals_ext["date"])

# Forecast range: only Jan–Mar 2026
val_dates    = pd.to_datetime(months)
sar_val      = sar_fc[sar_fc["date"].isin(val_dates)]
lstm_val     = lstm_fc[lstm_fc["date"].isin(val_dates)]

fig = go.Figure()

# Historical line
fig.add_trace(go.Scatter(
    x=hist_ctx["date"], y=hist_ctx["mean_rad"],
    name="Historical", line=dict(color="#555", width=2),
    mode="lines",
))

# Validation actuals (big dots + line)
fig.add_trace(go.Scatter(
    x=actuals_ext["date"], y=actuals_ext["actual"],
    name="Actual 2026", line=dict(color="#2ca02c", width=3),
    mode="lines+markers", marker=dict(size=10, symbol="diamond"),
))

# SARIMA forecast + CI band
sar_full = sar_fc[sar_fc["date"] <= pd.Timestamp("2026-03-01")]
fig.add_trace(go.Scatter(
    x=pd.concat([sar_full["date"], sar_full["date"][::-1]]),
    y=pd.concat([sar_full["upper_95"], sar_full["lower_95"][::-1]]),
    fill="toself", fillcolor="rgba(31,119,180,0.12)",
    line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="SARIMA 95% CI",
))
fig.add_trace(go.Scatter(
    x=sar_full["date"], y=sar_full["mean_forecast"],
    name="SARIMA Forecast", line=dict(color="#1f77b4", width=2, dash="dash"),
))

# LSTM forecast + CI band
lstm_full = lstm_fc[lstm_fc["date"] <= pd.Timestamp("2026-03-01")]
fig.add_trace(go.Scatter(
    x=pd.concat([lstm_full["date"], lstm_full["date"][::-1]]),
    y=pd.concat([lstm_full["upper_95"], lstm_full["lower_95"][::-1]]),
    fill="toself", fillcolor="rgba(255,127,14,0.12)",
    line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="LSTM 95% CI",
))
fig.add_trace(go.Scatter(
    x=lstm_full["date"], y=lstm_full["mean_forecast"],
    name="LSTM Forecast", line=dict(color="#ff7f0e", width=2, dash="dot"),
))

# Vertical shading for validation window
for vd in val_dates:
    fig.add_vrect(
        x0=vd - pd.Timedelta(days=15), x1=vd + pd.Timedelta(days=15),
        fillcolor="rgba(44,160,44,0.12)", line_width=0,
        annotation_text=vd.strftime("%b"), annotation_position="top left",
    )

fig.update_layout(
    xaxis_title="Date", yaxis_title="Mean Radiance (nW/cm²/sr)",
    legend=dict(orientation="h", y=-0.2),
    height=420, margin=dict(t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════
# SECTION C — ConvLSTM Spatial Comparison
# ═══════════════════════════════════════════════════════════
st.subheader("🗺️ ConvLSTM Spatial Comparison: Actual vs Forecast")

act_frames, fc_frames, act_dates = load_spatial(CITY)

if act_frames is None:
    st.info("Spatial frames not available.")
else:
    # shared colour scale across all months and both actual/forecast maps
    all_vals = np.concatenate([act_frames.flatten(), fc_frames[:3].flatten()])
    vmin_map = float(np.percentile(all_vals[all_vals > 0], 2))
    vmax_map = float(np.percentile(all_vals, 98))

    for i, d in enumerate(act_dates):
        label    = MONTH_LABELS.get(d, d)
        cl_month = next((r for r in cl_rows if r["date"] == d), None)

        st.markdown(f"#### {label}")
        if cl_month:
            m1, m2, m3 = st.columns(3)
            m1.metric("Actual Mean",    f"{cl_month['actual_mean']:.3f}")
            m2.metric("Forecast Mean",  f"{cl_month['predicted_mean']:.3f}")
            m3.metric("Mean Pixel MAE", f"{cl_month['mae']:.3f}",
                      delta=f"{cl_month['pct_error']:.2f}% error",
                      delta_color="inverse")

        col_act, col_fc, col_mae = st.columns(3)

        act_img = _arr_to_img(act_frames[i], "Greys_r", vmin=vmin_map, vmax=vmax_map)
        fc_img  = _arr_to_img(fc_frames[i],  "Greys_r", vmin=vmin_map, vmax=vmax_map)

        col_act.image(act_img, caption="Actual",   use_container_width=True)
        col_fc.image(fc_img,   caption="Forecast", use_container_width=True)

        # Pixel MAE heatmap
        month_tag = d[:7].replace("-", "_")
        mae_grid  = load_pixel_mae(CITY, month_tag)
        if mae_grid is not None:
            mae_img = _arr_to_img(mae_grid, "Reds")
            col_mae.image(mae_img, caption="Pixel MAE", use_container_width=True)
        else:
            col_mae.info("MAE grid not found")

        st.markdown("---")

    st.caption(
        "**Actual** — real VIIRS observation  |  **Forecast** — ConvLSTM prediction  |  "
        "**Pixel MAE** — per-pixel absolute error (darker red = larger error)"
    )
