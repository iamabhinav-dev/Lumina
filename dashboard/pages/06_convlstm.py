"""
06_convlstm.py — ConvLSTM Spatial Forecast Page
Pixel-level (35×45) NTL spatial maps for Kharagpur.

Two display modes:
  • Test period  — Actual | Predicted | Difference  (opacity slider)
  • Forecast     — Mean | Lower 95% CI | Upper 95% CI

Download button serves all 36 forecast GeoTIFFs as a ZIP archive.
"""

import io
import json
import os
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import Normalize
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
PAGE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(PAGE_DIR, "..", "..")

sys.path.insert(0, os.path.join(ROOT, "src"))
import cities as _cities

st.set_page_config(
    page_title="ConvLSTM Spatial Forecast",
    page_icon="🗺️",
    layout="wide",
)

# ─── City selection ─────────────────────────────────────────────────────────────
CITY       = st.session_state.get("city", "kharagpur")
CFG        = _cities.get_city(CITY)
CLSTM_DIR  = _cities.get_convlstm_dir(CITY, ROOT)
MODELS_DIR = _cities.get_convlstm_model_dir(CITY, ROOT)
SARIMA_DIR = _cities.get_sarima_dir(CITY, ROOT)
LSTM_DIR   = _cities.get_lstm_dir(CITY, ROOT)
TIFF_DIR   = os.path.join(CLSTM_DIR, "forecast_tiffs")
PLOT_DIR   = os.path.join(CLSTM_DIR, "plots")
CONVLSTM_SCRIPTS_DIR = os.path.join(ROOT, "models", "convlstm")

CMAP = "hot"

# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data
def load_eval_metrics(city: str) -> dict:
    with open(os.path.join(_cities.get_convlstm_dir(city, ROOT), "evaluation_metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_forecast_metadata(city: str) -> dict:
    with open(os.path.join(_cities.get_convlstm_dir(city, ROOT), "forecast_metadata.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading forecast frames…")
def load_forecast_frames(city: str) -> dict:
    data = np.load(os.path.join(_cities.get_convlstm_dir(city, ROOT), "forecast_frames.npz"))
    return {k: data[k] for k in data.files}


@st.cache_data(show_spinner="Loading test frames…")
def load_frames_npz(city: str) -> dict:
    data = np.load(os.path.join(_cities.get_convlstm_model_dir(city, ROOT), "frames.npz"), allow_pickle=True)
    return {k: data[k] for k in data.files}


@st.cache_data(show_spinner="Loading scaler…")
def load_scaler(city: str):
    import joblib
    return joblib.load(os.path.join(_cities.get_convlstm_model_dir(city, ROOT), "frame_scaler.pkl"))


@st.cache_data(show_spinner="Loading frame metadata…")
def load_frame_metadata(city: str) -> dict:
    with open(os.path.join(_cities.get_convlstm_model_dir(city, ROOT), "frame_metadata.json")) as f:
        return json.load(f)


@st.cache_data
def load_sarima_metrics(city: str) -> dict:
    p = os.path.join(_cities.get_sarima_dir(city, ROOT), "evaluation_metrics.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_lstm_metrics(city: str) -> dict:
    p = os.path.join(_cities.get_lstm_dir(city, ROOT), "evaluation_metrics.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner="Running test predictions…")
def compute_test_predictions(city: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict on the test set using keras.models.load_model(compile=False).
    Returns (y_true, y_pred) — both (N, H, W) in original radiance units.
    """
    import os as _os
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    import joblib

    clstm_dir  = _cities.get_convlstm_dir(city, ROOT)
    models_dir = _cities.get_convlstm_model_dir(city, ROOT)

    model_path = _os.path.join(clstm_dir, "convlstm_model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)

    scaler = joblib.load(_os.path.join(models_dir, "frame_scaler.pkl"))
    raw    = np.load(_os.path.join(models_dir, "frames.npz"), allow_pickle=True)
    X_test = raw["X_test"]   # (N, 12, H, W, 1) scaled
    y_test = raw["y_test"]   # (N, H, W, 1)     scaled

    H, W = X_test.shape[2], X_test.shape[3]
    preds_scaled = model.predict(X_test, verbose=0)  # (N, H, W, 1)

    def inv(arr_scaled):
        flat = arr_scaled.reshape(arr_scaled.shape[0], -1)
        orig = scaler.inverse_transform(flat)
        return np.clip(orig.reshape(-1, H, W), 0, None).astype(np.float32)

    y_true = inv(y_test)
    y_pred = inv(preds_scaled)
    return y_true, y_pred


def files_ready() -> bool:
    needed = [
        os.path.join(CLSTM_DIR, "evaluation_metrics.json"),
        os.path.join(CLSTM_DIR, "forecast_frames.npz"),
        os.path.join(CLSTM_DIR, "forecast_metadata.json"),
        os.path.join(MODELS_DIR, "frames.npz"),
        os.path.join(MODELS_DIR, "frame_scaler.pkl"),
        os.path.join(MODELS_DIR, "frame_metadata.json"),
    ]
    return all(os.path.exists(p) for p in needed)


# ─── Helper: render a spatial map as a PIL image ─────────────────────────────

def _frame_to_image(
    arr: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = CMAP,
    figsize: tuple = (3.5, 2.8),
    title: str = "",
    unit: str = "nW/cm²/sr",
) -> Image.Image:
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    im = ax.imshow(arr, cmap=cmap, norm=norm, origin="upper", aspect="auto")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(unit, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w_px, h_px = fig.canvas.get_width_height()
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h_px, w_px, 4)
    plt.close(fig)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _diff_to_image(
    arr: np.ndarray,
    abs_max: float,
    figsize: tuple = (3.5, 2.8),
    title: str = "",
) -> Image.Image:
    """Signed difference map (RdBu_r centred at 0)."""
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    im = ax.imshow(arr, cmap="RdBu_r", norm=norm, origin="upper", aspect="auto")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=3)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("nW/cm²/sr", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w_px, h_px = fig.canvas.get_width_height()
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h_px, w_px, 4)
    plt.close(fig)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _build_zip(tiff_dir: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(tiff_dir)):
            if fname.endswith(".tif"):
                zf.write(os.path.join(tiff_dir, fname), arcname=fname)
    return buf.getvalue()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🗺️ ConvLSTM Spatial")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Display mode",
    ["📅 Forecast (2026)", "🔍 Test Period Evaluation"],
    index=0,
)

if mode == "📅 Forecast (2026)":
    fc_band = st.sidebar.selectbox(
        "Forecast band",
        ["Mean", "Lower 95% CI", "Upper 95% CI"],
        index=0,
    )
    fc_month_idx = st.sidebar.slider("Month (2026)", 1, 12, 1)
    st.sidebar.caption(f"Selected: 2026-{fc_month_idx:02d}")

else:  # Test period
    frame_meta = None   # loaded after guard
    opacity = st.sidebar.slider("Map opacity", 0.3, 1.0, 0.85, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Re-run Forecast**")
if st.sidebar.button("▶ Re-run forecast (12 months)", use_container_width=True):
    with st.spinner("Running ConvLSTM forecast …"):
        import subprocess, sys as _sys
        script = os.path.join(CONVLSTM_SCRIPTS_DIR, "forecast_convlstm.py")
        result = subprocess.run(
            [_sys.executable, script, "--horizon", "12", "--city", CITY],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            st.sidebar.success("Forecast updated ✓")
            load_forecast_frames.clear()
        else:
            st.sidebar.error("Forecast failed.")
            st.sidebar.code(result.stderr[-600:] if result.stderr else "No stderr")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: ConvLSTM(16→32→32) encoder  \n"
    "       + (32→16) decoder  \n"
    "~241K params | CPU-only | TF 2.21  \n"
    "Trained: 73 epochs, best val MAE=0.035"
)


# ─── Guard ────────────────────────────────────────────────────────────────────

st.title(f"🗺️ ConvLSTM Spatial Forecast — {CFG['display_name']} NTL")

if not files_ready():
    st.error(
        "Required output files not found. Run the ConvLSTM pipeline first:\n\n"
        "```bash\n"
        "python models/convlstm/prepare_frames.py\n"
        "python models/convlstm/build_convlstm.py\n"
        "python models/convlstm/train_convlstm.py\n"
        "python models/convlstm/evaluate_convlstm.py\n"
        "python models/convlstm/forecast_convlstm.py\n"
        "```"
    )
    st.stop()


# ─── Load data ────────────────────────────────────────────────────────────────

eval_m       = load_eval_metrics(CITY)
fc_meta      = load_forecast_metadata(CITY)
fc_frames    = load_forecast_frames(CITY)
frame_meta   = load_frame_metadata(CITY)
sarima_m     = load_sarima_metrics(CITY)
lstm_m       = load_lstm_metrics(CITY)

test_dates   = [d for d in frame_meta["dates"] if d >= frame_meta["test_start"]]  # 24
fc_dates     = fc_meta["dates"]   # 12

mean_fc      = fc_frames["mean_forecast"]   # (12, H, W)
lower_fc     = fc_frames["lower_95"]        # (12, H, W)
upper_fc     = fc_frames["upper_95"]        # (12, H, W)

# Colour-scale bounds (shared across forecast bands for consistency)
FC_VMAX = float(np.percentile(upper_fc, 99))
FC_VMIN = 0.0


# ─── Metrics row ─────────────────────────────────────────────────────────────

st.markdown(
    "**Architecture:** `ConvLSTM(16→32→32)` encoder + `(32→16)` decoder  "
    "| ~241K params  "
    "| Test: Jan 2024 – Dec 2025 (24 months)"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

def _delta(new_val, ref_val, label="vs SARIMA"):
    diff = new_val - ref_val
    sign = "▼" if diff < 0 else "▲"
    return f"{sign} {abs(diff):.3f} {label}"

c1.metric("Pixel MAE",  f"{eval_m['pixel_MAE']:.2f}",
          help="Per-pixel MAE on test set (nW/cm²/sr)")
c2.metric("Pixel RMSE", f"{eval_m['pixel_RMSE']:.2f}",
          help="Per-pixel RMSE on test set (nW/cm²/sr)")
c3.metric("Mean SSIM",  f"{eval_m['mean_SSIM']:.4f}",
          help="Structural Similarity Index (0–1). Higher = better.")
c4.metric("Mean PSNR",  f"{eval_m['mean_PSNR']:.2f} dB",
          help="Peak Signal-to-Noise Ratio. Higher = better.")

if sarima_m:
    d = _delta(eval_m["mean_rad_MAE"], sarima_m["MAE"], "vs SARIMA")
    c5.metric("Mean-rad MAE", f"{eval_m['mean_rad_MAE']:.3f}",
              delta=d,
              delta_color="normal" if eval_m["mean_rad_MAE"] < sarima_m["MAE"] else "inverse",
              help="Spatial mean brightness MAE (nW/cm²/sr). Lower is better.")
else:
    c5.metric("Mean-rad MAE", f"{eval_m['mean_rad_MAE']:.3f}")

if sarima_m and lstm_m:
    best = min(
        [("SARIMA", sarima_m["MAE"]), ("LSTM", lstm_m["MAE"]),
         ("ConvLSTM", eval_m["mean_rad_MAE"])],
        key=lambda x: x[1],
    )
    c6.metric("Best model (MAE)", best[0],
              help="Model with lowest mean-radiance MAE on test set.")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE A — FORECAST 2026
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "📅 Forecast (2026)":

    st.subheader("📅 12-Month Spatial Forecast — 2026")

    band_map = {
        "Mean":          mean_fc,
        "Lower 95% CI":  lower_fc,
        "Upper 95% CI":  upper_fc,
    }
    arr      = band_map[fc_band][fc_month_idx - 1]
    date_str = fc_dates[fc_month_idx - 1]

    vmax_local = float(np.percentile(band_map[fc_band], 99))

    col_map, col_info = st.columns([2, 1])

    with col_map:
        img = _frame_to_image(
            arr, vmin=FC_VMIN, vmax=max(vmax_local, 1.0),
            title=f"{fc_band}  —  {date_str}",
            figsize=(5.5, 4.2),
        )
        st.image(img, use_container_width=True)

    with col_info:
        st.markdown(f"#### {date_str}")
        st.markdown(f"**Band:** {fc_band}")
        st.markdown(f"**Spatial mean:** `{arr.mean():.3f}` nW/cm²/sr")
        st.markdown(f"**Min:** `{arr.min():.3f}` &nbsp; **Max:** `{arr.max():.3f}`")
        st.markdown(f"**Grid:** {arr.shape[0]} × {arr.shape[1]} pixels (~500m res)")
        st.markdown("---")
        st.markdown(
            f"**CI half-width:** ±{fc_meta['ci_half']:.2f} nW/cm²/sr  \n"
            f"*(= 1.96 × test pixel RMSE = 1.96 × {fc_meta['pixel_RMSE']:.2f})*"
        )
        st.markdown("---")
        # Download ZIP
        zip_bytes = _build_zip(TIFF_DIR)
        st.download_button(
            label="⬇ Download all GeoTIFFs (ZIP)",
            data=zip_bytes,
            file_name="convlstm_forecast_tiffs.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # --- 12-panel grid ---
    st.markdown("#### All 12 Months — Mean Forecast")
    band_all = mean_fc
    cols = st.columns(6)
    vmax_all = float(np.percentile(band_all, 99))
    for i, d in enumerate(fc_dates):
        with cols[i % 6]:
            img_i = _frame_to_image(
                band_all[i],
                vmin=FC_VMIN, vmax=max(vmax_all, 1.0),
                title=d[5:7],          # "01" … "12"
                figsize=(2.2, 1.9),
            )
            st.image(img_i, caption=d[:7], use_container_width=True)

    # --- Mean radiance time series across forecast ---
    st.markdown("#### Spatial Mean Radiance — Forecast vs Test")
    mean_ts  = mean_fc.mean(axis=(1, 2))
    lower_ts = lower_fc.mean(axis=(1, 2))
    upper_ts = upper_fc.mean(axis=(1, 2))
    fc_dates_pd = pd.to_datetime(fc_dates)

    # Also plot test actuals for context (load from eval metrics per_month)
    test_actual_means = [m["pixel_MAE"] for m in eval_m.get("per_month", [])]
    # Use the saved per-month pixel_MAE for a visual guide; not the actual mean
    # Better: load from frames and compute directly (cached)
    frames_raw = load_frames_npz(CITY)
    scaler     = load_scaler(CITY)
    y_test_sc  = frames_raw["y_test"]  # (24, H, W, 1) scaled
    H_g, W_g   = y_test_sc.shape[1], y_test_sc.shape[2]
    y_test_orig = np.clip(
        scaler.inverse_transform(y_test_sc.reshape(24, -1)).reshape(24, H_g, W_g),
        0, None
    )
    actual_mean_ts  = y_test_orig.mean(axis=(1, 2))
    test_dates_pd   = pd.to_datetime(test_dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates_pd, y=actual_mean_ts,
        mode="lines+markers", name="Actual (test)",
        line=dict(color="#4C72B0", width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_dates_pd.to_series(), fc_dates_pd.to_series()[::-1]]),
        y=np.concatenate([upper_ts, lower_ts[::-1]]),
        fill="toself", fillcolor="rgba(255,140,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=True, name="95% CI",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates_pd, y=mean_ts,
        mode="lines+markers", name="ConvLSTM forecast (mean)",
        line=dict(color="#FF8C00", width=2, dash="dash"),
        marker=dict(size=5),
    ))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Spatial mean radiance (nW/cm²/sr)",
        legend=dict(orientation="h", y=1.08),
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Animated GIF ---
    gif_path = os.path.join(PLOT_DIR, "forecast_animation.gif")
    if os.path.exists(gif_path):
        st.markdown("#### Forecast Animation")
        with open(gif_path, "rb") as gf:
            st.image(gf.read(), caption="12-month ConvLSTM forecast (mean radiance)",
                     use_container_width=False, width=520)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE B — TEST PERIOD EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

else:
    st.subheader("🔍 Test Period Evaluation — Jan 2024 to Dec 2025")

    y_true, y_pred = compute_test_predictions(CITY)

    # Month selector
    test_month_idx = st.slider(
        "Select test month", 1, len(test_dates), 1,
    )
    idx   = test_month_idx - 1
    d_str = test_dates[idx][:7]
    st.caption(f"Selected: {d_str}")

    actual_frame = y_true[idx]
    pred_frame   = y_pred[idx]
    diff_frame   = pred_frame - actual_frame

    # Shared colour scale
    vmax_shared = float(np.percentile(np.concatenate([actual_frame, pred_frame]), 99))
    abs_max_diff = float(np.percentile(np.abs(diff_frame), 99))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Actual — {d_str}**")
        img_a = _frame_to_image(
            actual_frame, vmin=0, vmax=max(vmax_shared, 1.0),
            title="Actual", figsize=(3.5, 2.8),
        )
        st.image(img_a, use_container_width=True, caption="Ground truth")

    with col2:
        st.markdown(f"**Predicted — {d_str}**")
        img_p = _frame_to_image(
            pred_frame * opacity, vmin=0, vmax=max(vmax_shared * opacity, 1.0),
            title=f"Predicted (opacity={opacity:.2f})", figsize=(3.5, 2.8),
        )
        st.image(img_p, use_container_width=True, caption="ConvLSTM prediction")

    with col3:
        st.markdown(f"**Difference (Pred − Actual) — {d_str}**")
        img_d = _diff_to_image(
            diff_frame, abs_max=max(abs_max_diff, 1.0),
            title="Pred − Actual", figsize=(3.5, 2.8),
        )
        st.image(img_d, use_container_width=True, caption="Blue=under, Red=over")

    # Per-frame stats
    frame_mae  = float(np.mean(np.abs(diff_frame)))
    frame_rmse = float(np.sqrt(np.mean(diff_frame ** 2)))
    st.markdown(
        f"**Month stats:** MAE = `{frame_mae:.3f}` nW/cm²/sr  "
        f"| RMSE = `{frame_rmse:.3f}` nW/cm²/sr  "
        f"| Actual mean = `{actual_frame.mean():.3f}`  "
        f"| Pred mean = `{pred_frame.mean():.3f}`"
    )

    st.markdown("---")

    # SSIM time series from eval metrics
    st.subheader("📊 Per-Month Quality Metrics (Test Set)")

    per_month = eval_m.get("per_month", [])
    if per_month:
        pm_df = pd.DataFrame(per_month)
        pm_df["date"] = pd.to_datetime(pm_df["date"])

        tab1, tab2, tab3 = st.tabs(["Pixel MAE", "SSIM", "PSNR"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pm_df["date"], y=pm_df["pixel_MAE"],
                marker_color="#FF8C00", name="Pixel MAE",
            ))
            fig.add_hline(y=eval_m["pixel_MAE"], line_dash="dash",
                          line_color="white", annotation_text=f"Avg {eval_m['pixel_MAE']:.2f}")
            fig.update_layout(
                template="plotly_dark", yaxis_title="nW/cm²/sr",
                height=320, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pm_df["date"], y=pm_df["SSIM"],
                mode="lines+markers",
                line=dict(color="#2ECC71", width=2),
                marker=dict(size=6),
                name="SSIM",
            ))
            fig.add_hline(y=eval_m["mean_SSIM"], line_dash="dash",
                          line_color="white",
                          annotation_text=f"Avg {eval_m['mean_SSIM']:.4f}")
            fig.update_layout(
                template="plotly_dark", yaxis_title="SSIM (0–1)",
                height=320, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pm_df["date"], y=pm_df["PSNR"],
                mode="lines+markers",
                line=dict(color="#3498DB", width=2),
                marker=dict(size=6),
                name="PSNR (dB)",
            ))
            fig.add_hline(y=eval_m["mean_PSNR"], line_dash="dash",
                          line_color="white",
                          annotation_text=f"Avg {eval_m['mean_PSNR']:.2f} dB")
            fig.update_layout(
                template="plotly_dark", yaxis_title="PSNR (dB)",
                height=320, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Saved evaluation plots
    st.subheader("🖼️ Evaluation Plots")
    plot_files = {
        "Prediction vs Actual Grid": os.path.join(PLOT_DIR, "pred_vs_actual_grid.png"),
        "Error Maps":                os.path.join(PLOT_DIR, "error_maps.png"),
        "SSIM Time Series":          os.path.join(PLOT_DIR, "ssim_timeseries.png"),
        "Training Loss Curve":       os.path.join(PLOT_DIR, "loss_curve.png"),
    }
    pcols = st.columns(2)
    for j, (label, path) in enumerate(plot_files.items()):
        if os.path.exists(path):
            with pcols[j % 2]:
                st.image(path, caption=label, use_container_width=True)

    # Download GeoTIFFs
    st.markdown("---")
    st.markdown("#### ⬇ Download Forecast GeoTIFFs")
    zip_bytes = _build_zip(TIFF_DIR)
    st.download_button(
        label="Download all 36 GeoTIFFs (ZIP)",
        data=zip_bytes,
        file_name="convlstm_forecast_tiffs.zip",
        mime="application/zip",
    )
