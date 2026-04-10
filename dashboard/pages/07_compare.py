"""
07_compare.py — City Comparison Dashboard
Side-by-side comparison: Kharagpur vs Kolkata
  A. Dual time series + 2026 SARIMA forecast
  B. Model accuracy table (SARIMA / LSTM / ConvLSTM × MAE / RMSE / MAPE / MASE)
  C. Spatial forecast maps side by side (ConvLSTM, month slider)
  D. Year-over-year annual mean radiance bar chart
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

st.set_page_config(page_title="City Comparison", page_icon="📊", layout="wide")

PLOTLY_DARK = "plotly_dark"
CITIES      = list(_cities.CITIES.keys())          # ["kharagpur", "kolkata"]
COLORS      = {c: _cities.CITIES[c]["color"]        for c in CITIES}
NAMES       = {c: _cities.CITIES[c]["display_name"] for c in CITIES}


# ─── Helper ────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_ts(city: str) -> pd.DataFrame:
    p = os.path.join(_cities.get_sarima_dir(city, ROOT), "mean_brightness_clean.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_sarima_forecast(city: str) -> pd.DataFrame:
    p = os.path.join(_cities.get_sarima_dir(city, ROOT), "forecast.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_model_metrics(city: str) -> dict:
    """Return {model_name: metrics_dict} for SARIMA / LSTM / ConvLSTM."""
    result = {}
    dirs = {
        "SARIMA":   _cities.get_sarima_dir(city, ROOT),
        "LSTM":     _cities.get_lstm_dir(city, ROOT),
        "ConvLSTM": _cities.get_convlstm_dir(city, ROOT),
    }
    for model, d in dirs.items():
        p = os.path.join(d, "evaluation_metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                result[model] = json.load(f)
    return result


@st.cache_data
def load_convlstm_forecast(city: str) -> dict | None:
    clstm_dir = _cities.get_convlstm_dir(city, ROOT)
    p_npz  = os.path.join(clstm_dir, "forecast_frames.npz")
    p_meta = os.path.join(clstm_dir, "forecast_metadata.json")
    if not os.path.exists(p_npz) or not os.path.exists(p_meta):
        return None
    data = np.load(p_npz)
    with open(p_meta) as f:
        meta = json.load(f)
    return {"mean": data["mean_forecast"], "meta": meta}


# ─── Page ──────────────────────────────────────────────────────────────────────

st.title("📊 City Comparison — Kharagpur vs Kolkata")

st.markdown(
    "Compare daily NTL trends, model accuracy, spatial forecasts, and annual "
    "growth across both cities. Use the **sidebar** to switch the active city "
    "for all other pages."
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — Dual time series + forecast
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("A. NTL Time Series — 2014 to 2025 (+ 2026 SARIMA Forecast)")

fig_ts = go.Figure()

for city in CITIES:
    df = load_ts(city)
    if df.empty:
        continue
    fc = load_sarima_forecast(city)

    fig_ts.add_trace(go.Scatter(
        x=df["date"], y=df["mean_rad"],
        mode="lines",
        name=NAMES[city],
        line=dict(color=COLORS[city], width=1.8),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.3f} nW/cm²/sr<extra></extra>",
    ))

    if not fc.empty:
        if "upper_95" in fc.columns and "lower_95" in fc.columns:
            fig_ts.add_trace(go.Scatter(
                x=pd.concat([fc["date"], fc["date"][::-1]]),
                y=pd.concat([fc["upper_95"], fc["lower_95"][::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(COLORS[city], 0.12),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ))
        fig_ts.add_trace(go.Scatter(
            x=fc["date"], y=fc["mean_forecast"],
            mode="lines",
            name=f"{NAMES[city]} forecast",
            line=dict(color=COLORS[city], width=1.6, dash="dash"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Fc: %{y:.3f}<extra></extra>",
        ))

fig_ts.add_vline(
    x=pd.Timestamp("2024-01-01").timestamp() * 1000,
    line_dash="dot", line_color="gray", line_width=1.2,
    annotation_text="← history | forecast →",
    annotation_position="top right",
    annotation_font_size=9, annotation_font_color="gray",
)
fig_ts.update_layout(
    template=PLOTLY_DARK, height=420,
    xaxis_title="", yaxis_title="Mean Radiance (nW/cm²/sr)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — Model accuracy comparison table
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("B. Model Accuracy Comparison")

# For ConvLSTM the scalar-radiance metrics are stored under "mean_rad_*" keys
CONVLSTM_KEY_MAP = {
    "MAE":  "mean_rad_MAE",
    "RMSE": "mean_rad_RMSE",
    "MAPE": "mean_rad_MAPE",
    "MASE": "mean_rad_MASE",
}

rows = []
for city in CITIES:
    metrics_dict = load_model_metrics(city)
    for model in ["SARIMA", "LSTM", "ConvLSTM"]:
        if model not in metrics_dict:
            continue
        m = metrics_dict[model]
        row = {"City": NAMES[city], "Model": model}
        for mk in ["MAE", "RMSE", "MAPE", "MASE"]:
            actual_key = CONVLSTM_KEY_MAP[mk] if model == "ConvLSTM" else mk
            val = m.get(actual_key, float("nan"))
            row[mk] = round(val, 3) if not np.isnan(val) else float("nan")
        rows.append(row)

if rows:
    acc_df = pd.DataFrame(rows)
    st.dataframe(
        acc_df.style.background_gradient(
            subset=["MAE", "RMSE", "MAPE", "MASE"],
            cmap="RdYlGn_r",
        ).format(
            {"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}", "MASE": "{:.3f}"},
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Quick bar chart comparing MAE across models/cities
    fig_acc = go.Figure()
    for city in CITIES:
        city_rows = [r for r in rows if r["City"] == NAMES[city]]
        fig_acc.add_trace(go.Bar(
            x=[r["Model"] for r in city_rows],
            y=[r["MAE"] for r in city_rows],
            name=NAMES[city],
            marker_color=COLORS[city],
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>MAE: %{y:.3f}<extra></extra>",
        ))
    fig_acc.update_layout(
        template=PLOTLY_DARK, height=320, barmode="group",
        xaxis_title="Model", yaxis_title="MAE (nW/cm²/sr)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_acc, use_container_width=True)
else:
    st.info("Model metrics not yet computed. Run the full pipeline for both cities.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — Spatial forecast maps (ConvLSTM)
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("C. ConvLSTM Spatial Forecast Maps — Side by Side")

fc_data = {city: load_convlstm_forecast(city) for city in CITIES}

if all(v is not None for v in fc_data.values()):
    dates_a = fc_data[CITIES[0]]["meta"]["dates"]
    dates_b = fc_data[CITIES[1]]["meta"]["dates"]
    common_dates = sorted(set(dates_a) & set(dates_b))

    if common_dates:
        sel_date = st.select_slider(
            "Select forecast month",
            options=common_dates,
            value=common_dates[0],
        )

        def _arr_to_image(arr2d: np.ndarray, title: str) -> Image.Image:
            vmin, vmax = 0.0, max(float(np.percentile(arr2d, 99)), 0.01)
            fig, ax = plt.subplots(figsize=(4.5, 4.0), facecolor="#111")
            im = ax.imshow(arr2d, cmap="Greys_r", vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(title, color="white", fontsize=11, pad=6)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.outline.set_edgecolor("white")
            for lbl in cbar.ax.get_yticklabels():
                lbl.set_color("white")
            cbar.set_label("nW/cm²/sr", color="white", fontsize=8)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#111")
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)

        col1, col2 = st.columns(2)
        for col, city in zip([col1, col2], CITIES):
            fcd = fc_data[city]
            dates_list = fcd["meta"]["dates"]
            if sel_date in dates_list:
                idx = dates_list.index(sel_date)
                arr = fcd["mean"][idx]
                img = _arr_to_image(arr, f"{NAMES[city]} — {sel_date[:7]}")
                col.image(img, caption=f"{NAMES[city]} — {sel_date[:7]} ({arr.shape[0]}×{arr.shape[1]} px)", use_container_width=True)
            else:
                col.info(f"No forecast data for {sel_date[:7]} ({NAMES[city]}).")
    else:
        st.info("No common forecast months found between the two cities.")
else:
    missing = [NAMES[c] for c in CITIES if fc_data[c] is None]
    st.info(
        f"ConvLSTM forecast not yet available for: **{', '.join(missing)}**.  \n"
        "Run `python models/convlstm/forecast_convlstm.py --city <city>` to generate."
    )

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — Year-over-year annual mean radiance
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("D. Year-over-Year Annual Mean Radiance (2014–2025)")

fig_yoy = go.Figure()
for city in CITIES:
    df = load_ts(city)
    if df.empty:
        continue
    annual = df.groupby("year")["mean_rad"].mean().reset_index()
    fig_yoy.add_trace(go.Bar(
        x=annual["year"],
        y=annual["mean_rad"],
        name=NAMES[city],
        marker_color=COLORS[city],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Mean: %{y:.3f} nW/cm²/sr<extra></extra>",
    ))

# CAGR annotation
for city in CITIES:
    df = load_ts(city)
    if df.empty or len(df) < 24:
        continue
    annual = df.groupby("year")["mean_rad"].mean()
    if annual.index.min() < annual.index.max():
        n_years = annual.index.max() - annual.index.min()
        cagr = (annual.iloc[-1] / annual.iloc[0]) ** (1 / n_years) - 1
        fig_yoy.add_annotation(
            x=annual.index.max(),
            y=annual.iloc[-1],
            text=f"CAGR {cagr*100:+.1f}%",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            font=dict(size=10, color=COLORS[city]),
            ax=0, ay=-30,
        )

fig_yoy.update_layout(
    template=PLOTLY_DARK, height=380,
    barmode="group",
    xaxis=dict(title="Year", dtick=1),
    yaxis_title="Annual Mean Radiance (nW/cm²/sr)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig_yoy, use_container_width=True)

# ─── Section E: 2026 Live Validation ─────────────────────────────────────────
st.divider()
st.subheader("✅ 2026 Live Validation — Jan to Mar (Both Cities)")

import json as _json_cmp

_val_cols = st.columns(2)
for _ci, _city in enumerate(CITIES):
    _sar_dir = _cities.get_sarima_dir(_city, ROOT)
    _vp = os.path.join(os.path.dirname(_sar_dir), "validation_2026.json")
    with _val_cols[_ci]:
        _cfg2 = _cities.get_city(_city)
        st.markdown(f"**{_cfg2['display_name']}**")
        if not os.path.exists(_vp):
            st.info("validation_2026.json not found")
            continue
        with open(_vp) as _fv:
            _vd = _json_cmp.load(_fv)
        _errs = _vd["mean_pct_errors"]
        _best = _vd["best_model"]
        _ec1, _ec2, _ec3 = st.columns(3)
        for _col, _key, _lbl in [(_ec1, "sarima", "SARIMA"), (_ec2, "lstm", "LSTM"), (_ec3, "convlstm", "ConvLSTM")]:
            _v = _errs.get(_key)
            _badge = " 🥇" if _key == _best else ""
            with _col:
                st.metric(f"{_lbl}{_badge}", f"{_v:.2f}%" if _v is not None else "N/A",
                          help="Mean absolute % error, Jan–Mar 2026")
st.caption("Lower is better. 🥇 = best model for each city.")
