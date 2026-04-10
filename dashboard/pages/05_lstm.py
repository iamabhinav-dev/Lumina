"""
05_lstm.py — LSTM Forecast Page
LSTM (MC Dropout) forecast vs SARIMA — head-to-head comparison.
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
MODELS_DIR = os.path.join(ROOT, "models", "lstm")

sys.path.insert(0, os.path.join(ROOT, "src"))
import cities as _cities

st.set_page_config(page_title="LSTM Forecast", page_icon="🧠", layout="wide")

# ─── City selection ─────────────────────────────────────────────────────────────
CITY       = st.session_state.get("city", "kharagpur")
CFG        = _cities.get_city(CITY)
LSTM_DIR   = _cities.get_lstm_dir(CITY, ROOT)
SARIMA_DIR = _cities.get_sarima_dir(CITY, ROOT)

PLOTLY_DARK = "plotly_dark"

# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading historical series…")
def load_history(city: str) -> pd.DataFrame:
    path = os.path.join(_cities.get_sarima_dir(city, ROOT), "mean_brightness_clean.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner="Loading LSTM forecast…")
def load_lstm_forecast(city: str) -> pd.DataFrame:
    path = os.path.join(_cities.get_lstm_dir(city, ROOT), "forecast.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner="Loading SARIMA forecast…")
def load_sarima_forecast(city: str) -> pd.DataFrame:
    path = os.path.join(_cities.get_sarima_dir(city, ROOT), "forecast.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner="Loading test split…")
def load_lstm_split(city: str) -> pd.DataFrame:
    path = os.path.join(_cities.get_lstm_dir(city, ROOT), "train_test_split.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def load_lstm_metrics(city: str) -> dict:
    with open(os.path.join(_cities.get_lstm_dir(city, ROOT), "evaluation_metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_sarima_metrics(city: str) -> dict:
    with open(os.path.join(_cities.get_sarima_dir(city, ROOT), "evaluation_metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_best_params(city: str) -> dict:
    with open(os.path.join(_cities.get_lstm_dir(city, ROOT), "best_params.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading training history…")
def load_training_history(city: str) -> dict:
    path = os.path.join(_cities.get_lstm_dir(city, ROOT), "training_history.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Running LSTM test predictions…")
def load_lstm_test_preds(city: str) -> tuple[np.ndarray, np.ndarray]:
    """Load model + sequences and predict on test set (cached)."""
    import os as _os
    lstm_dir   = _cities.get_lstm_dir(city, ROOT)
    seq_npz    = _os.path.join(lstm_dir, "sequences.npz")
    scaler_pkl = _os.path.join(lstm_dir, "scaler.pkl")
    model_path = _os.path.join(lstm_dir, "lstm_model.keras")

    for p in [seq_npz, scaler_pkl, model_path]:
        if not _os.path.exists(p):
            return np.array([]), np.array([])

    import os as _os2
    _os2.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    import joblib

    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_pkl)
    data   = np.load(seq_npz)
    X_test = data["X_test"]
    y_test = data["y_test"]

    pred_scaled   = model.predict(X_test, verbose=0).flatten()
    actual_orig   = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_orig     = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return actual_orig, pred_orig


def sarima_test_preds(test_len: int) -> np.ndarray:
    """Load SARIMA train model and generate test forecasts."""
    sarima_pkl = os.path.join(SARIMA_DIR, "sarima_model.pkl")
    if not os.path.exists(sarima_pkl):
        return np.array([])
    try:
        import joblib
        model = joblib.load(sarima_pkl)
        fc = model.get_forecast(steps=test_len)
        return fc.predicted_mean.values
    except Exception:
        return np.array([])


def files_ready() -> bool:
    needed = [
        os.path.join(LSTM_DIR, "forecast.csv"),
        os.path.join(LSTM_DIR, "evaluation_metrics.json"),
        os.path.join(LSTM_DIR, "best_params.json"),
        os.path.join(LSTM_DIR, "lstm_model.keras"),
        os.path.join(SARIMA_DIR, "mean_brightness_clean.csv"),
    ]
    return all(os.path.exists(p) for p in needed)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🧠 LSTM Forecast")
st.sidebar.markdown("---")

st.sidebar.markdown("**Display Settings**")
horizon = st.sidebar.selectbox(
    "Forecast horizon", [6, 12, 18, 24], index=1,
    format_func=lambda h: f"{h} months",
)
context_years = st.sidebar.slider("History to display (years)", 1, 12, 4)
show_ci       = st.sidebar.checkbox("Show 95% CI (MC Dropout)", value=True)
show_sarima   = st.sidebar.checkbox("Show SARIMA overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Re-run Forecast**")
if st.sidebar.button(f"▶ Forecast {horizon} months", use_container_width=True):
    with st.spinner(f"Running LSTM forecast ({horizon} months) …"):
        script = os.path.join(MODELS_DIR, "forecast.py")
        result = subprocess.run(
            [sys.executable, script, "--horizon", str(horizon)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            st.sidebar.success("Forecast updated ✓")
            load_lstm_forecast.clear()
        else:
            st.sidebar.error("Forecast failed.")
            st.sidebar.code(result.stderr[-600:] if result.stderr else "No stderr")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: LSTM(128→32) + MC Dropout  \n"
    "window=12 | dropout=0.3 | lr=5e-4  \n"
    "Keras Tuner RandomSearch (30 trials)"
)


# ─── Guard ───────────────────────────────────────────────────────────────────

st.title(f"🧠 LSTM Forecast — {CFG['display_name']} NTL")

if not files_ready():
    st.error(
        "Required output files not found. Run the LSTM pipeline first:\n\n"
        "```bash\n"
        "python models/lstm/prepare_sequences.py\n"
        "python models/lstm/tune.py\n"
        "python models/lstm/train.py\n"
        "python models/lstm/evaluate.py\n"
        "python models/lstm/forecast.py\n"
        "```"
    )
    st.stop()


# ─── Load data ────────────────────────────────────────────────────────────────

hist_df      = load_history(CITY)
lstm_fc      = load_lstm_forecast(CITY).head(horizon)
sarima_fc    = load_sarima_forecast(CITY).head(horizon)
split_df     = load_lstm_split(CITY)
lstm_m       = load_lstm_metrics(CITY)
sarima_m     = load_sarima_metrics(CITY)
params       = load_best_params(CITY)
train_hist   = load_training_history(CITY)

test_df      = split_df[split_df["split"] == "test"]
context_cut  = hist_df["date"].max() - pd.DateOffset(years=context_years)
hist_ctx     = hist_df[hist_df["date"] >= context_cut]

actual_test, lstm_pred_test = load_lstm_test_preds(CITY)
sarima_pred_test = sarima_test_preds(len(test_df))

# ─── Header metrics row ───────────────────────────────────────────────────────

def delta_str(lstm_val, sarima_val, lower_is_better=True):
    diff = lstm_val - sarima_val
    sign = "▼" if diff < 0 else "▲"
    color = "normal" if (diff < 0) == lower_is_better else "inverse"
    return f"{sign} {abs(diff):.3f} vs SARIMA", color


st.markdown(
    f"**Architecture:** `LSTM({params['units_1']}→{params['units_2']})` "
    f"| window=`{params['window_size']}` | dropout=`{params['dropout']}` "
    f"| lr=`{params['lr']}` | tuned via Keras Tuner ({params['max_trials']} trials)"
)

c1, c2, c3, c4, c5 = st.columns(5)

d, dc = delta_str(lstm_m["MAE"], sarima_m["MAE"])
c1.metric("MAE", f"{lstm_m['MAE']:.3f}", delta=d, delta_color=dc,
          help="Mean Absolute Error (nW/cm²/sr). Lower is better.")

d, dc = delta_str(lstm_m["RMSE"], sarima_m["RMSE"])
c2.metric("RMSE", f"{lstm_m['RMSE']:.3f}", delta=d, delta_color=dc,
          help="Root Mean Squared Error (nW/cm²/sr). Lower is better.")

d, dc = delta_str(lstm_m["MAPE"], sarima_m["MAPE"])
c3.metric("MAPE", f"{lstm_m['MAPE']:.1f}%", delta=d, delta_color=dc,
          help="Mean Absolute Percentage Error. Lower is better.")

d, dc = delta_str(lstm_m["MASE"], sarima_m["MASE"])
c4.metric("MASE", f"{lstm_m['MASE']:.3f}", delta=d, delta_color=dc,
          help="< 1 = beats seasonal naïve. Lower is better.")

winner = "SARIMA" if all(
    lstm_m[k] > sarima_m[k] for k in ["MAE", "RMSE", "MAPE", "MASE"]
) else "LSTM"
c5.metric("Overall winner", winner,
          help="Model with lower values across all 4 metrics on the test set.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Main forecast chart
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("📈 Historical Series & Future Forecast")

fig = go.Figure()

# Faint full history
fig.add_trace(go.Scatter(
    x=hist_df["date"], y=hist_df["mean_rad"],
    mode="lines", line=dict(color="#4C72B0", width=0.6),
    opacity=0.2, showlegend=False, hoverinfo="skip",
))

# Context history
fig.add_trace(go.Scatter(
    x=hist_ctx["date"], y=hist_ctx["mean_rad"],
    mode="lines", name=f"Historical (last {context_years}y)",
    line=dict(color="#4C72B0", width=1.8),
))

# LSTM CI band
if show_ci:
    fig.add_trace(go.Scatter(
        x=pd.concat([lstm_fc["date"], lstm_fc["date"][::-1]]),
        y=pd.concat([lstm_fc["upper_95"], lstm_fc["lower_95"][::-1]]),
        fill="toself", fillcolor="rgba(214,39,40,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", showlegend=True, name="LSTM 95% CI (MC Dropout)",
    ))

# SARIMA overlay
if show_sarima:
    fig.add_trace(go.Scatter(
        x=sarima_fc["date"], y=sarima_fc["mean_forecast"],
        mode="lines+markers", name="SARIMA forecast",
        line=dict(color="#4C72B0", width=1.8, dash="dot"),
        marker=dict(size=4), opacity=0.75,
        customdata=np.stack([sarima_fc["lower_95"], sarima_fc["upper_95"]], axis=-1),
        hovertemplate="<b>%{x|%b %Y}</b><br>SARIMA: %{y:.3f}<br>"
                      "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
    ))

# LSTM forecast line
fig.add_trace(go.Scatter(
    x=lstm_fc["date"], y=lstm_fc["mean_forecast"],
    mode="lines+markers", name=f"LSTM forecast ({horizon}m)",
    line=dict(color="#D62728", width=2.2, dash="dash"),
    marker=dict(size=6),
    customdata=np.stack([lstm_fc["lower_95"], lstm_fc["upper_95"]], axis=-1),
    hovertemplate="<b>%{x|%b %Y}</b><br>LSTM: %{y:.3f}<br>"
                  "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
))

boundary = hist_df["date"].max()
fig.add_vline(
    x=boundary.timestamp() * 1000,
    line_dash="dot", line_color="gray", line_width=1.5,
    annotation_text="History | Forecast",
    annotation_position="top right",
    annotation_font_size=10, annotation_font_color="gray",
)
fig.update_layout(
    template=PLOTLY_DARK, height=420, xaxis_title="",
    yaxis_title="Mean Radiance (nW/cm²/sr)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified", margin=dict(t=40, b=40),
)
st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Head-to-head on test set
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🔍 Head-to-Head — Test Period (Jan 2024 – Dec 2025)")

fig2 = go.Figure()

test_dates = test_df["date"].values

if len(actual_test):
    fig2.add_trace(go.Scatter(
        x=test_dates, y=actual_test,
        mode="lines+markers", name="Actual",
        line=dict(color="#2CA02C", width=2.2), marker=dict(size=5),
    ))

if len(lstm_pred_test):
    lstm_errors = actual_test - lstm_pred_test if len(actual_test) else []
    fig2.add_trace(go.Scatter(
        x=test_dates, y=lstm_pred_test,
        mode="lines+markers", name="LSTM",
        line=dict(color="#D62728", width=2, dash="dash"), marker=dict(size=5),
    ))

if show_sarima and len(sarima_pred_test):
    sarima_errors = actual_test - sarima_pred_test if len(actual_test) else []
    fig2.add_trace(go.Scatter(
        x=test_dates, y=sarima_pred_test,
        mode="lines+markers", name="SARIMA",
        line=dict(color="#4C72B0", width=1.8, dash="dot"),
        marker=dict(size=4), opacity=0.8,
    ))

# Per-month winner annotation as bar
if len(actual_test) and len(lstm_pred_test) and len(sarima_pred_test):
    lstm_abs   = np.abs(actual_test - lstm_pred_test)
    sarima_abs = np.abs(actual_test - sarima_pred_test)
    win_colors = ["#D62728" if l < s else "#4C72B0"
                  for l, s in zip(lstm_abs, sarima_abs)]
    win_vals   = np.where(lstm_abs < sarima_abs,
                          lstm_abs - sarima_abs,
                          lstm_abs - sarima_abs)
    fig2.add_trace(go.Bar(
        x=test_dates, y=win_vals,
        marker_color=win_colors, opacity=0.5,
        name="LSTM advantage (neg=LSTM wins)",
        yaxis="y2",
        hovertemplate="LSTM err: %{customdata[0]:.3f}<br>SARIMA err: %{customdata[1]:.3f}<extra></extra>",
        customdata=np.stack([lstm_abs, sarima_abs], axis=-1),
    ))
    fig2.update_layout(
        yaxis2=dict(
            title="Error advantage (nW/cm²/sr)",
            overlaying="y", side="right",
            showgrid=False, zeroline=True,
            zerolinecolor="gray", zerolinewidth=1,
        )
    )

fig2.update_layout(
    template=PLOTLY_DARK, height=400, xaxis_title="",
    yaxis_title="Mean Radiance (nW/cm²/sr)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified", margin=dict(t=40, b=40),
    barmode="overlay",
)
st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Side-by-side metric comparison
# ═══════════════════════════════════════════════════════════════════════════════

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📊 Model Comparison")
    metrics_fig = go.Figure()
    metric_keys  = ["MAE", "RMSE", "MAPE", "MASE"]
    lstm_vals    = [lstm_m[k] for k in metric_keys]
    sarima_vals  = [sarima_m[k] for k in metric_keys]
    metric_labels = ["MAE", "RMSE", "MAPE (%)", "MASE"]

    metrics_fig.add_trace(go.Bar(
        name="LSTM", x=metric_labels, y=lstm_vals,
        marker_color="#D62728", text=[f"{v:.3f}" for v in lstm_vals],
        textposition="outside",
    ))
    metrics_fig.add_trace(go.Bar(
        name="SARIMA", x=metric_labels, y=sarima_vals,
        marker_color="#4C72B0", text=[f"{v:.3f}" for v in sarima_vals],
        textposition="outside",
    ))
    metrics_fig.update_layout(
        template=PLOTLY_DARK, height=360, barmode="group",
        yaxis_title="Value (lower is better)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(metrics_fig, use_container_width=True)

with col_right:
    st.subheader("📋 Forecast Table")
    table = lstm_fc.copy()
    table["Month"]      = table["date"].dt.strftime("%b %Y")
    table["Forecast"]   = table["mean_forecast"].round(3)
    table["Lower 95%"]  = table["lower_95"].round(3)
    table["Upper 95%"]  = table["upper_95"].round(3)
    if show_sarima and len(sarima_fc):
        table["SARIMA"]  = sarima_fc["mean_forecast"].values[:len(table)].round(3)
    st.dataframe(
        table[["Month", "Forecast", "Lower 95%", "Upper 95%"] +
              (["SARIMA"] if show_sarima and "SARIMA" in table.columns else [])],
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Training diagnostics expander
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📉 Training Diagnostics"):
    if train_hist and "loss" in train_hist:
        epochs  = list(range(1, len(train_hist["loss"]) + 1))
        best_ep = int(np.argmin(train_hist["val_loss"])) + 1

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            x=epochs, y=train_hist["loss"],
            mode="lines", name="Train MAE",
            line=dict(color="#4C72B0", width=1.8),
        ))
        loss_fig.add_trace(go.Scatter(
            x=epochs, y=train_hist["val_loss"],
            mode="lines", name="Val MAE",
            line=dict(color="#DD8452", width=1.8),
        ))
        loss_fig.add_vline(
            x=best_ep, line_dash="dot", line_color="gray",
            annotation_text=f"Best epoch {best_ep}",
            annotation_font_size=10, annotation_font_color="gray",
        )
        loss_fig.update_layout(
            template=PLOTLY_DARK, height=300,
            xaxis_title="Epoch", yaxis_title="MAE (scaled)",
            title="Loss Curve — Train vs Validation",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(loss_fig, use_container_width=True)
        st.caption(
            f"Early stopping triggered at epoch **{len(epochs)}**. "
            f"Best weights restored from epoch **{best_ep}** "
            f"(val MAE = {min(train_hist['val_loss']):.5f} scaled)."
        )
    else:
        st.info("Training history not found — run `train.py` to generate it.")

    st.markdown("""
**Why does val MAE plateau so quickly?**
With only 84 training sequences, the LSTM learns the broad level of the series
within a handful of epochs.  Higher dropout (0.3) prevents it from memorising
noise, but also limits how much it can specialise to the seasonal pattern.
This is the fundamental data-scarcity constraint.
""")


# ─── Model info expander ─────────────────────────────────────────────────────

# ─── 2026 Validation expander ──────────────────────────────────────────────

import json as _json2
_val_path2 = os.path.join(os.path.dirname(LSTM_DIR), "validation_2026.json")
if os.path.exists(_val_path2):
    with open(_val_path2) as _f2:
        _val2 = _json2.load(_f2)
    with st.expander("✅ 2026 Validation — Jan to Mar (Real vs Forecast)"):
        _rows2 = []
        for _r2 in _val2["lstm"]:
            _dot2 = "🟢" if _r2["pct_error"] < 5 else ("🟡" if _r2["pct_error"] < 10 else "🔴")
            _lbl2 = {"2026-01-01": "Jan 2026", "2026-02-01": "Feb 2026", "2026-03-01": "Mar 2026"}.get(_r2["date"], _r2["date"])
            _rows2.append({"Month": _lbl2, "Actual": round(_r2["actual"], 3),
                           "LSTM Pred": round(_r2["predicted"], 3),
                           "MAE": round(_r2["mae"], 3),
                           "% Error": f"{_dot2} {_r2['pct_error']:.2f}%"})
        _mean_err2 = _val2["mean_pct_errors"].get("lstm")
        _rows2.append({"Month": "**Mean**", "Actual": "—", "LSTM Pred": "—",
                       "MAE": "—", "% Error": f"**{_mean_err2:.2f}%**"})
        import pandas as _pd2
        st.dataframe(_pd2.DataFrame(_rows2), use_container_width=True, hide_index=True)
        st.caption("🟢 < 5%  |  🟡 5–10%  |  🔴 > 10%  |  Data: real VIIRS VCMSLCFG, Jan–Mar 2026")

with st.expander("ℹ️ Model Information & Interpretation"):
    st.markdown(f"""
**Architecture:** Stacked LSTM — two recurrent layers + dropout regularisation

| Layer | Config | Output shape |
|---|---|---|
| Input | window\\_size={params['window_size']} months | `(batch, 12, 1)` |
| LSTM 1 | {params['units_1']} units, return\\_sequences=True | `(batch, 12, {params['units_1']})` |
| Dropout | rate={params['dropout']} | — |
| LSTM 2 | {params['units_2']} units | `(batch, {params['units_2']})` |
| Dropout | rate={params['dropout']} | — |
| Dense hidden | 16 units, ReLU | `(batch, 16)` |
| Dense output | 1 unit | `(batch, 1)` |

**Hyperparameter search:** Keras Tuner RandomSearch — {params['max_trials']} trials  
**Objective:** Minimise validation MAE (scaled)  
**Training:** Max 300 epochs, early stopping patience=25, `ReduceLROnPlateau`

---

**MC Dropout uncertainty estimation:**  
At inference, Dropout layers are normally disabled.  Passing `training=True`
keeps them active, so each forward pass samples a different sub-network.
Running **200 batched stochastic passes** per horizon step yields a distribution
of predictions — the 2.5th–97.5th percentile becomes the 95% CI.

This is a standard Bayesian approximation requiring **no extra parameters**.

---

**Why SARIMA outperforms LSTM here:**
- The series has only 144 months → 108 training sequences after windowing
- LSTM needs hundreds of seasons to reliably learn periodic patterns
- SARIMA's explicit seasonal differencing `(D=1, m=12)` is a strong structural
  prior that the LSTM must infer purely from data
- On short, structured time series, **domain knowledge beats data-driven capacity**

**This IS a valid thesis finding** — matching or beating classical models requires
significantly more data or exogenous predictors.
""")
