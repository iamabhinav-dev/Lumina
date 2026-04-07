"""
03_charts.py — Charts & Trends Page
Time series, seasonal patterns, year-over-year, histograms, % change analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import preprocess as pp
import utils as ut

st.set_page_config(page_title="Charts & Trends", page_icon="📊", layout="wide")

PLOTLY_DARK = "plotly_dark"


# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building time series (first run may take a minute)...")
def cached_timeseries():
    return pp.build_timeseries_df()


@st.cache_data(show_spinner="Loading raster...")
def cached_load_raster(year: int, month: int):
    return pp.load_raster(year, month)


@st.cache_data
def cached_dates():
    return pp.get_available_dates()


# ─── Load data ────────────────────────────────────────────────────────────────

dates  = cached_dates()
labels = ut.dates_to_labels(dates)
df     = cached_timeseries()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Charts & Trends")

chart_type = st.sidebar.radio("Chart", [
    "Time Series",
    "Year-over-Year",
    "Seasonal Pattern",
    "% Change from Baseline",
    "Pixel Histogram",
    "Lit Area Over Time",
])

st.sidebar.markdown("---")

selected_idx = st.sidebar.selectbox(
    "Highlighted Month (for histogram / marker)",
    range(len(dates)),
    index=len(dates) - 1,
    format_func=lambda i: labels[i],
)
sel_year, sel_month = dates[selected_idx]

# Year filter for some charts
all_years = sorted(df["year"].unique().tolist()) if not df.empty else []
selected_years = st.sidebar.multiselect(
    "Filter Years (Year-over-Year)",
    all_years,
    default=all_years,
)

# ─── Main panel ───────────────────────────────────────────────────────────────

st.title("📊 Charts & Trends — Kharagpur NTL")

if df.empty:
    st.error("Time series data not available. Check that data/tiffs/ has downloaded files.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
if chart_type == "Time Series":
    st.subheader("📈 Monthly Mean Radiance — 2014 to 2025")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["mean_rad"],
        mode="lines+markers",
        name="Mean Radiance",
        line=dict(color="#f39c12", width=2),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%b %Y}</b><br>Mean: %{y:.3f} nW/cm²/sr<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["max_rad"],
        mode="lines",
        name="Max Radiance",
        line=dict(color="#e74c3c", width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Max: %{y:.3f}<extra></extra>",
    ))

    # Highlight selected month
    sel_row = df[(df["year"] == sel_year) & (df["month"] == sel_month)]
    if not sel_row.empty:
        fig.add_vline(
            x=str(sel_row["date"].iloc[0]),
            line_dash="dash", line_color="#2ecc71", line_width=1.5,
            annotation_text=f"{labels[selected_idx]}",
            annotation_position="top right",
        )

    # Trend line
    x_num = np.arange(len(df))
    z = np.polyfit(x_num, df["mean_rad"].fillna(df["mean_rad"].mean()), 1)
    trend = np.polyval(z, x_num)
    fig.add_trace(go.Scatter(
        x=df["date"], y=trend,
        mode="lines",
        name="Trend",
        line=dict(color="#9b59b6", width=2, dash="longdash"),
    ))

    fig.update_layout(
        template=PLOTLY_DARK,
        xaxis_title="Date",
        yaxis_title="Radiance (nW/cm²/sr)",
        legend=dict(x=0.01, y=0.99),
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown(f"**Trend:** {'⬆️ Increasing' if z[0] > 0 else '⬇️ Decreasing'} — "
                f"slope = {z[0]:.4f} nW/cm²/sr per month")

# ═══════════════════════════════════════════════════════════════════════════════
elif chart_type == "Year-over-Year":
    st.subheader("📆 Year-over-Year Monthly Radiance")

    filtered = df[df["year"].isin(selected_years)] if selected_years else df

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, year in enumerate(sorted(filtered["year"].unique())):
        yr_df = filtered[filtered["year"] == year].sort_values("month")
        fig.add_trace(go.Scatter(
            x=yr_df["month"],
            y=yr_df["mean_rad"],
            mode="lines+markers",
            name=str(year),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{year}</b> — %{{x}}<br>%{{y:.3f}} nW/cm²/sr<extra></extra>",
        ))

    fig.update_layout(
        template=PLOTLY_DARK,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=ut.MONTH_NAMES,
            title="Month",
        ),
        yaxis_title="Mean Radiance (nW/cm²/sr)",
        height=450,
        hovermode="x unified",
        legend=dict(title="Year"),
    )
    st.plotly_chart(fig, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
elif chart_type == "Seasonal Pattern":
    st.subheader("🌦️ Monthly Seasonal Pattern (All Years Averaged)")

    seasonal = df.groupby("month")["mean_rad"].agg(["mean", "std"]).reset_index()
    seasonal.columns = ["month", "mean", "std"]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(seasonal["month"]) + list(seasonal["month"])[::-1],
        y=list(seasonal["mean"] + seasonal["std"]) + list(seasonal["mean"] - seasonal["std"])[::-1],
        fill="toself",
        fillcolor="rgba(243, 156, 18, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±1 Std Dev",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Bar(
        x=seasonal["month"],
        y=seasonal["mean"],
        name="Avg Radiance",
        marker_color="#f39c12",
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Avg: %{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=seasonal["month"],
        y=seasonal["mean"],
        mode="lines+markers",
        name="Trend line",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        template=PLOTLY_DARK,
        xaxis=dict(
            tickmode="array", tickvals=list(range(1, 13)),
            ticktext=ut.MONTH_NAMES, title="Month",
        ),
        yaxis_title="Mean Radiance (nW/cm²/sr)",
        height=420,
        barmode="overlay",
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("**Note:** Dips in certain months may reflect monsoon cloud cover reducing NTL signal.")

# ═══════════════════════════════════════════════════════════════════════════════
elif chart_type == "% Change from Baseline":
    st.subheader("📊 % Change in Mean Radiance vs 2014 Baseline")

    yearly = df.groupby("year")["mean_rad"].mean().reset_index()
    baseline = yearly[yearly["year"] == yearly["year"].min()]["mean_rad"].values[0]
    yearly["pct_change"] = ((yearly["mean_rad"] - baseline) / baseline) * 100

    colors_bar = ["#e74c3c" if v >= 0 else "#3498db" for v in yearly["pct_change"]]

    fig = go.Figure(go.Bar(
        x=yearly["year"],
        y=yearly["pct_change"],
        marker_color=colors_bar,
        text=[f"{v:+.1f}%" for v in yearly["pct_change"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Change: %{y:+.1f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

    fig.update_layout(
        template=PLOTLY_DARK,
        xaxis_title="Year",
        yaxis_title="% Change from 2014 Baseline",
        height=420,
        xaxis=dict(tickmode="linear", dtick=1),
    )
    st.plotly_chart(fig, width='stretch')

    peak_year = int(yearly.loc[yearly["pct_change"].idxmax(), "year"])
    st.markdown(f"**Peak year:** {peak_year} ({yearly.loc[yearly['pct_change'].idxmax(), 'pct_change']:+.1f}% vs 2014)")

# ═══════════════════════════════════════════════════════════════════════════════
elif chart_type == "Pixel Histogram":
    st.subheader(f"📉 Pixel Radiance Distribution — {labels[selected_idx]}")

    raster = cached_load_raster(sel_year, sel_month)
    arr = raster["array"]
    valid = arr[np.isfinite(arr)].flatten()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid,
        nbinsx=80,
        marker_color="#f39c12",
        opacity=0.8,
        name="Pixel count",
        hovertemplate="Radiance: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_DARK,
        xaxis_title="Radiance (nW/cm²/sr)",
        yaxis_title="Pixel Count (log scale)",
        yaxis_type="log",
        height=400,
    )
    st.plotly_chart(fig, width='stretch')

    s = pp.get_stats(arr)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min", f"{s['min']:.3f}")
    c2.metric("Max", f"{s['max']:.3f}")
    c3.metric("Mean", f"{s['mean']:.3f}")
    c4.metric("Std", f"{s['std']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
elif chart_type == "Lit Area Over Time":
    st.subheader("🏙️ Lit Area (km²) Over Time")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["lit_area_km2"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(46, 204, 113, 0.15)",
        line=dict(color="#2ecc71", width=2),
        marker=dict(size=4),
        name="Lit Area",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f} km²<extra></extra>",
    ))

    # Highlight selected
    sel_row = df[(df["year"] == sel_year) & (df["month"] == sel_month)]
    if not sel_row.empty:
        fig.add_vline(
            x=str(sel_row["date"].iloc[0]),
            line_dash="dash", line_color="#f39c12", line_width=1.5,
            annotation_text=labels[selected_idx],
        )

    fig.update_layout(
        template=PLOTLY_DARK,
        xaxis_title="Date",
        yaxis_title="Lit Area (km²)",
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')

    yearly_lit = df.groupby("year")["lit_area_km2"].mean().reset_index()
    st.markdown("**Average Lit Area by Year:**")
    st.dataframe(
        yearly_lit.rename(columns={"year": "Year", "lit_area_km2": "Avg Lit Area (km²)"}),
        hide_index=True,
        width='stretch',
    )
