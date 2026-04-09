"""
04_forecast.py — SARIMA Forecast Page
Historical NTL time series + SARIMA forecast with 95% CI.
Allows on-demand re-forecast at different horizons.
"""

import sys
import os
import json
import subprocess
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────
PAGE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(PAGE_DIR, "..", "..")
MODELS_DIR = os.path.join(ROOT, "models", "sarima")

sys.path.insert(0, os.path.join(ROOT, "src"))
import cities as _cities

st.set_page_config(page_title="SARIMA Forecast", page_icon="🔮", layout="wide")

# ─── City selection ─────────────────────────────────────────────────────────────
CITY       = st.session_state.get("city", "kharagpur")
CFG        = _cities.get_city(CITY)
SARIMA_DIR = _cities.get_sarima_dir(CITY, ROOT)

PLOTLY_DARK = "plotly_dark"

MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading historical series…")
def load_history(city: str) -> pd.DataFrame:
    path = os.path.join(_cities.get_sarima_dir(city, ROOT), "mean_brightness_clean.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner="Loading forecast…")
def load_forecast(city: str) -> pd.DataFrame:
    path = os.path.join(_cities.get_sarima_dir(city, ROOT), "forecast.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner="Loading test results…")
def load_test_split(city: str) -> pd.DataFrame:
    path = os.path.join(_cities.get_sarima_dir(city, ROOT), "train_test_split.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def load_metrics(city: str) -> dict:
    path = os.path.join(_cities.get_sarima_dir(city, ROOT), "evaluation_metrics.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_order(city: str) -> dict:
    path = os.path.join(_cities.get_sarima_dir(city, ROOT), "best_order.json")
    with open(path) as f:
        return json.load(f)


def files_ready() -> bool:
    needed = [
        "mean_brightness_clean.csv",
        "forecast.csv",
        "train_test_split.csv",
        "evaluation_metrics.json",
        "best_order.json",
    ]
    return all(os.path.exists(os.path.join(SARIMA_DIR, f)) for f in needed)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🔮 SARIMA Forecast")
st.sidebar.markdown("---")

st.sidebar.markdown("**Forecast Settings**")
horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [6, 12, 18, 24],
    index=1,
    format_func=lambda h: f"{h} months",
)

context_years = st.sidebar.slider(
    "History to display (years)",
    min_value=1, max_value=12, value=4,
)

show_ci = st.sidebar.checkbox("Show 95% confidence interval", value=True)
show_smooth = st.sidebar.checkbox("Show 3-month rolling mean", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Re-run Forecast**")
if st.sidebar.button(f"▶ Forecast {horizon} months", use_container_width=True):
    with st.spinner(f"Running SARIMA forecast ({horizon} months)…"):
        script = os.path.join(MODELS_DIR, "forecast.py")
        venv_python = sys.executable
        result = subprocess.run(
            [venv_python, script, "--horizon", str(horizon), "--city", CITY],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            st.sidebar.success("Forecast updated ✓")
            # Clear cached forecast so it reloads
            load_forecast.clear()
        else:
            st.sidebar.error("Forecast failed — check terminal for details.")
            st.sidebar.code(result.stderr[-800:] if result.stderr else "No stderr")

st.sidebar.markdown("---")
st.sidebar.caption("Model: SARIMA(0,1,1)(0,1,1)[12]  \nFitted on full series (Jan 2014 – Dec 2025)")


# ─── Guard: check files exist ─────────────────────────────────────────────────

st.title(f"🔮 SARIMA Forecast — {CFG['display_name']} NTL")

if not files_ready():
    st.error(
        "Required output files not found. Run the pipeline first:\n\n"
        "```bash\n"
        "python models/sarima/extract_brightness.py\n"
        "python models/sarima/clean_brightness.py\n"
        "python models/sarima/find_order.py\n"
        "python models/sarima/train.py\n"
        "python models/sarima/evaluate.py\n"
        "python models/sarima/forecast.py\n"
        "```"
    )
    st.stop()


# ─── Load data ────────────────────────────────────────────────────────────────

hist_df   = load_history(CITY)
fc_df     = load_forecast(CITY)
split_df  = load_test_split(CITY)
metrics   = load_metrics(CITY)
order_cfg = load_order(CITY)

train_df = split_df[split_df["split"] == "train"]
test_df  = split_df[split_df["split"] == "test"]

order_str = (f"SARIMA{tuple(order_cfg['order'])} × "
             f"{tuple(order_cfg['seasonal_order'])}")

# Clip forecast to requested horizon (in case saved CSV has more)
fc_df = fc_df.head(horizon)

# Context window for history display
context_cutoff = hist_df["date"].max() - pd.DateOffset(years=context_years)
hist_ctx = hist_df[hist_df["date"] >= context_cutoff]


# ─── Metrics row ─────────────────────────────────────────────────────────────

st.markdown(f"**Model:** `{order_str}` &nbsp;|&nbsp; "
            f"Fitted on full series (Jan 2014 – Dec 2025) &nbsp;|&nbsp; "
            f"Test: Jan 2024 – Dec 2025 (24 months)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MAE",  f"{metrics['MAE']:.3f}",  help="Mean Absolute Error (nW/cm²/sr)")
c2.metric("RMSE", f"{metrics['RMSE']:.3f}", help="Root Mean Squared Error (nW/cm²/sr)")
c3.metric("MAPE", f"{metrics['MAPE']:.1f}%", help="Mean Absolute Percentage Error")
c4.metric("MASE", f"{metrics['MASE']:.3f}",
          delta="vs seasonal naïve",
          delta_color="inverse" if metrics["MASE"] < 1 else "normal",
          help="< 1 = beats seasonal naïve baseline")
c5.metric("Forecast horizon", f"{horizon} months",
          help=f"Predicting {fc_df['date'].iloc[0].strftime('%b %Y')} – "
               f"{fc_df['date'].iloc[-1].strftime('%b %Y')}")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Main forecast chart
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("📈 Historical Series & Forecast")

fig = go.Figure()

# Faint full history behind context window
fig.add_trace(go.Scatter(
    x=hist_df["date"], y=hist_df["mean_rad"],
    mode="lines",
    line=dict(color="#4C72B0", width=0.6),
    opacity=0.25,
    showlegend=False,
    hoverinfo="skip",
))

# Context window history
fig.add_trace(go.Scatter(
    x=hist_ctx["date"], y=hist_ctx["mean_rad"],
    mode="lines",
    name=f"Historical (last {context_years}y)",
    line=dict(color="#4C72B0", width=1.8),
))

# Optional 3-month rolling mean
if show_smooth and "mean_rad_smooth" in hist_ctx.columns:
    fig.add_trace(go.Scatter(
        x=hist_ctx["date"], y=hist_ctx["mean_rad_smooth"],
        mode="lines",
        name="3-month rolling mean",
        line=dict(color="#FF7F0E", width=1.8, dash="dot"),
    ))

# CI band
if show_ci:
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
        y=pd.concat([fc_df["upper_95"], fc_df["lower_95"][::-1]]),
        fill="toself",
        fillcolor="rgba(214,39,40,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% CI",
    ))

# Forecast line + markers
fig.add_trace(go.Scatter(
    x=fc_df["date"], y=fc_df["mean_forecast"],
    mode="lines+markers",
    name=f"Forecast ({horizon}m)",
    line=dict(color="#D62728", width=2.2, dash="dash"),
    marker=dict(size=6),
    customdata=np.stack([fc_df["lower_95"], fc_df["upper_95"]], axis=-1),
    hovertemplate=(
        "<b>%{x|%b %Y}</b><br>"
        "Forecast: %{y:.3f}<br>"
        "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
        "<extra></extra>"
    ),
))

# Vertical boundary line
boundary = hist_df["date"].max()
fig.add_vline(
    x=boundary.timestamp() * 1000,
    line_dash="dot", line_color="gray", line_width=1.5,
    annotation_text="History | Forecast",
    annotation_position="top right",
    annotation_font_size=10,
    annotation_font_color="gray",
)

fig.update_layout(
    template=PLOTLY_DARK,
    height=420,
    xaxis_title="",
    yaxis_title="Mean Radiance (nW/cm²/sr)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Test period: actual vs forecast
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🔍 Model Accuracy — Test Period (Jan 2024 – Dec 2025)")

# Re-generate test forecast from the train model for the evaluation view
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=test_df["date"], y=test_df["mean_rad"],
    mode="lines+markers",
    name="Actual (test)",
    line=dict(color="#2CA02C", width=2),
    marker=dict(size=5),
))

# Load the test-period forecast from the pkl (train model)
try:
    import joblib
    train_model = joblib.load(os.path.join(SARIMA_DIR, "sarima_model.pkl"))
    fc_test = train_model.get_forecast(steps=len(test_df))
    fc_test_mean = fc_test.predicted_mean
    fc_test_ci   = fc_test.conf_int(alpha=0.05)
    fc_test_mean.index = pd.to_datetime(test_df["date"].values)
    fc_test_ci.index   = pd.to_datetime(test_df["date"].values)

    if show_ci:
        fig2.add_trace(go.Scatter(
            x=pd.concat([fc_test_ci.index.to_series(), fc_test_ci.index.to_series()[::-1]]),
            y=pd.concat([fc_test_ci.iloc[:, 1], fc_test_ci.iloc[:, 0][::-1]]),
            fill="toself",
            fillcolor="rgba(214,39,40,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% CI",
        ))

    fig2.add_trace(go.Scatter(
        x=fc_test_mean.index, y=fc_test_mean.values,
        mode="lines+markers",
        name="SARIMA forecast",
        line=dict(color="#D62728", width=2, dash="dash"),
        marker=dict(size=5),
    ))

    # Error bars per month
    errors = (test_df["mean_rad"].values - fc_test_mean.values)
    fig2.add_trace(go.Bar(
        x=fc_test_mean.index,
        y=errors,
        name="Error (actual − forecast)",
        marker_color=["#D62728" if e < 0 else "#2CA02C" for e in errors],
        opacity=0.5,
        yaxis="y2",
    ))

    fig2.update_layout(
        yaxis2=dict(
            title="Error (nW/cm²/sr)",
            overlaying="y", side="right",
            showgrid=False, zeroline=True,
            zerolinecolor="gray", zerolinewidth=1,
        ),
    )
except Exception:
    pass  # if model pkl missing or incompatible, just skip the error bars

fig2.update_layout(
    template=PLOTLY_DARK,
    height=380,
    xaxis_title="",
    yaxis_title="Mean Radiance (nW/cm²/sr)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Forecast table + seasonal bar
# ═══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Forecast Table")
    table_df = fc_df.copy()
    table_df["Month"] = table_df["date"].dt.strftime("%b %Y")
    table_df = table_df.rename(columns={
        "mean_forecast": "Forecast",
        "lower_95":      "Lower 95%",
        "upper_95":      "Upper 95%",
    })[["Month", "Forecast", "Lower 95%", "Upper 95%"]]
    table_df["Forecast"]  = table_df["Forecast"].round(3)
    table_df["Lower 95%"] = table_df["Lower 95%"].round(3)
    table_df["Upper 95%"] = table_df["Upper 95%"].round(3)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("📊 Forecast by Month")
    bar_colors = [
        "#D62728" if v == fc_df["mean_forecast"].min()
        else "#2CA02C" if v == fc_df["mean_forecast"].max()
        else "#4C72B0"
        for v in fc_df["mean_forecast"]
    ]
    fig3 = go.Figure(go.Bar(
        x=fc_df["date"].dt.strftime("%b %Y"),
        y=fc_df["mean_forecast"],
        marker_color=bar_colors,
        text=fc_df["mean_forecast"].round(2),
        textposition="outside",
        error_y=dict(
            type="data",
            symmetric=False,
            array=(fc_df["upper_95"] - fc_df["mean_forecast"]).values,
            arrayminus=(fc_df["mean_forecast"] - fc_df["lower_95"]).values,
            visible=show_ci,
        ),
        hovertemplate="<b>%{x}</b><br>Forecast: %{y:.3f}<extra></extra>",
    ))
    fig3.update_layout(
        template=PLOTLY_DARK,
        height=360,
        xaxis_title="",
        yaxis_title="Mean Radiance (nW/cm²/sr)",
        showlegend=False,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ─── Model info expander ─────────────────────────────────────────────────────

with st.expander("ℹ️ Model Information & Limitations"):
    st.markdown(f"""
**Model:** `{order_str}`

| Component | Value | Meaning |
|-----------|-------|---------|
| `d = 1` | First differencing | Removes the upward trend |
| `D = 1` | Seasonal differencing (m=12) | Removes the 12-month seasonality |
| `q = 1` | MA(1) | Smooths short-term noise |
| `Q = 1` | Seasonal MA(1) | Smooths seasonal noise |

**Why this model (Box-Jenkins "airline" model):**
- Parsimonious — only 2 parameters (+ variance)
- Lowest AIC of all 70 candidates tested in Step 4 (AIC = 189.95)
- Both MA coefficients highly significant (p < 0.001)

**Known limitations:**
- Does not model exogenous events (festivals, pandemics, policy changes)
- Assumes the seasonal pattern stays fixed; real NTL seasonality may drift
- Confidence intervals widen quickly beyond 6 months
- MASE = {metrics['MASE']:.2f} — slightly below seasonal naïve on test set due to the
  strong upward trend in 2024–2025 that the model was not trained on

**To improve:** SARIMAX with a linear trend regressor, or retrain after each new year of data.
""")
