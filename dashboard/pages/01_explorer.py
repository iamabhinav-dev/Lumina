"""
01_explorer.py — NTL Explorer Page
Browse monthly NTL maps with time slider, play/pause animation, colormap selector.
Fast Preview mode: Plotly client-side animation (smooth, no rerun lag).
"""

import sys
import os
import time
import numpy as np
import plotly.graph_objects as go
import folium
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import preprocess as pp
import cities as _cities
import utils as ut

st.set_page_config(page_title="NTL Explorer", page_icon="🗺️", layout="wide")

# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading raster...")
def cached_load_raster(year: int, month: int, city: str):
    return pp.load_raster(year, month, city)


@st.cache_data(show_spinner="Building time series...")
def cached_timeseries(city: str):
    return pp.build_timeseries_df(city=city)


@st.cache_data
def cached_dates(city: str):
    return pp.get_available_dates(city)


# ─── Session state init ───────────────────────────────────────────────────────

if "date_idx" not in st.session_state:
    st.session_state.date_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_speed" not in st.session_state:
    st.session_state.play_speed = 0.1
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Fast Preview"

# ─── Load data ────────────────────────────────────────────────────────────────

CITY  = st.session_state.get("city", "kharagpur")
CFG   = _cities.get_city(CITY)
dates = cached_dates(CITY)
labels = ut.dates_to_labels(dates)
ts_df = cached_timeseries(CITY)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🗺️ NTL Explorer")

view_mode = st.sidebar.radio(
    "View Mode",
    ["Fast Preview (Plotly)", "Map Explorer (Folium)"],
    index=0,
    key="view_mode_radio",
)

st.sidebar.markdown("---")

if view_mode == "Map Explorer (Folium)":
    # Play controls only relevant for Folium mode
    col_play, col_stop = st.sidebar.columns(2)
    if col_play.button("▶ Play" if not st.session_state.playing else "⏸ Pause"):
        st.session_state.playing = not st.session_state.playing
    if col_stop.button("⏮ Reset"):
        st.session_state.playing = False
        st.session_state.date_idx = 0

    st.session_state.play_speed = st.sidebar.slider(
        "Animation speed (sec/frame)", 0.0, 2.0, st.session_state.play_speed, 0.05
    )

st.sidebar.markdown("---")

# Date slider — shown for Folium mode
if view_mode == "Map Explorer (Folium)":
    st.session_state.date_idx = st.sidebar.slider(
        "Select Month",
        min_value=0,
        max_value=len(dates) - 1,
        value=st.session_state.date_idx,
        format="",
        key="date_slider",
    )
    st.sidebar.markdown(f"**Selected: {labels[st.session_state.date_idx]}**")
    st.sidebar.markdown("---")

    # Colormap + opacity + basemap only needed for Folium mode
    cmap_label = st.sidebar.selectbox("Colormap", list(ut.COLORMAPS.keys()), index=0, key="me_cmap")
    cmap_name = ut.COLORMAPS[cmap_label]
    opacity = st.sidebar.slider("Overlay opacity", 0.3, 1.0, 0.75, 0.05, key="me_opacity")
    st.sidebar.markdown("**Base Map**")
    basemap = st.sidebar.radio(
        "Tile layer",
        ["CartoDB dark_matter", "CartoDB positron", "OpenStreetMap"],
        index=0,
        key="basemap_radio",
    )
else:
    # Fast Preview sidebar settings
    st.sidebar.markdown("**Fast Preview Settings**")
    frame_duration = st.sidebar.slider("Frame duration (ms)", 100, 1000, 300, 50, key="fp_frame_dur")
    transition_dur = st.sidebar.slider("Transition (ms)", 0, 300, 50, 10, key="fp_trans_dur")
    cmap_label = st.sidebar.selectbox("Colormap", list(ut.COLORMAPS.keys()), index=0, key="fp_cmap")
    cmap_name = ut.COLORMAPS[cmap_label]
    year_filter = st.sidebar.multiselect(
        "Filter years (leave empty = all)",
        list(range(2014, 2026)),
        default=[],
        key="fp_year_filter",
    )
    # Defaults for Folium-only variables
    opacity = 0.75
    basemap = "CartoDB dark_matter"

# ─── Auto-play logic (Folium mode only) ──────────────────────────────────────

if view_mode == "Map Explorer (Folium)" and st.session_state.playing:
    if st.session_state.date_idx < len(dates) - 1:
        st.session_state.date_idx += 1
    else:
        st.session_state.playing = False
    time.sleep(st.session_state.play_speed)
    st.rerun()

# ─── Main panel ───────────────────────────────────────────────────────────────

st.title(f"🌃 NTL Explorer — {CFG['display_name']}")

# ══════════════════════════════════════════════════════════════════════════════
# FAST PREVIEW MODE — Plotly client-side animation
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "Fast Preview (Plotly)":
    with st.spinner("Loading all rasters for animation (first time only)..."):
        # Determine which dates to animate
        anim_dates = [
            (y, m) for y, m in dates
            if (not year_filter or y in year_filter)
        ]

        # Load all rasters
        all_arrays = []
        for y, m in anim_dates:
            r = cached_load_raster(y, m, CITY)
            all_arrays.append(r["array"])
            bounds = r["bounds"]

        # Global vmin/vmax across all frames for consistent color scale
        global_min = float(np.nanmin([np.nanmin(a[np.isfinite(a)]) for a in all_arrays if np.any(np.isfinite(a))]))
        global_max = float(np.nanmax([np.nanmax(a[np.isfinite(a)]) for a in all_arrays if np.any(np.isfinite(a))]))

        anim_labels = ut.dates_to_labels(anim_dates)

        # Build Plotly figure with frames
        frames = []
        for i, (arr, lbl) in enumerate(zip(all_arrays, anim_labels)):
            frames.append(go.Frame(
                data=[go.Heatmap(
                    z=arr,
                    zmin=global_min,
                    zmax=global_max,
                    colorscale=cmap_name,
                    showscale=True,
                    colorbar=dict(title="nW/cm²/sr"),
                )],
                name=lbl,
                layout=go.Layout(title_text=f"NTL Radiance — {lbl}"),
            ))

        fig = go.Figure(
            data=[go.Heatmap(
                z=all_arrays[0],
                zmin=global_min,
                zmax=global_max,
                colorscale=cmap_name,
                showscale=True,
                colorbar=dict(title="nW/cm²/sr", tickfont=dict(color="white")),
                hoverongaps=False,
                hovertemplate="Row: %{y}<br>Col: %{x}<br>Radiance: %{z:.3f}<extra></extra>",
            )],
            frames=frames,
            layout=go.Layout(
                title=dict(text=f"NTL Radiance — {anim_labels[0]}", x=0.5, xanchor="center"),
                template="plotly_dark",
                autosize=False,
                width=700,
                height=640,
                margin=dict(l=60, r=80, t=70, b=190),
                xaxis=dict(
                    title=f"← {bounds[0]:.3f}°E  —  {bounds[2]:.3f}°E →",
                    showticklabels=False,
                    showgrid=False,
                ),
                yaxis=dict(
                    title=f"↑ {bounds[3]:.3f}°N  —  {bounds[1]:.3f}°N ↓",
                    showticklabels=False,
                    showgrid=False,
                    autorange="reversed",
                    scaleanchor="x",
                    scaleratio=1,
                ),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    x=0.5,
                    y=-0.02,
                    xanchor="center",
                    yanchor="top",
                    pad=dict(t=5, b=5),
                    bgcolor="#333",
                    bordercolor="#666",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                transition=dict(duration=transition_dur),
                                fromcurrent=True,
                                mode="immediate",
                            )],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            )],
                        ),
                    ],
                )],
                sliders=[dict(
                    active=0,
                    currentvalue=dict(
                        prefix="📅  ",
                        visible=True,
                        xanchor="center",
                        font=dict(size=13),
                    ),
                    len=0.9,
                    x=0.05,
                    pad=dict(t=55, b=10),
                    steps=[
                        dict(
                            method="animate",
                            label=lbl,
                            args=[[lbl], dict(
                                mode="immediate",
                                frame=dict(duration=frame_duration, redraw=True),
                                transition=dict(duration=transition_dur),
                            )],
                        )
                        for lbl in anim_labels
                    ],
                )],
            ),
        )

    _c1, _c2, _c3 = st.columns([1, 8, 1])
    with _c2:
        st.plotly_chart(fig, use_container_width=False)
    st.caption(
        f"📊 {len(anim_dates)} months  |  "
        f"Radiance range: **{global_min:.2f} – {global_max:.2f}** nW/cm²/sr  |  "
        f"Resolution: {all_arrays[0].shape[0]} × {all_arrays[0].shape[1]} pixels (~500 m/px)  |  Use ▶ Play inside the chart to animate."
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAP EXPLORER MODE — Folium interactive map
# ══════════════════════════════════════════════════════════════════════════════
else:
    year, month = dates[st.session_state.date_idx]
    raster = cached_load_raster(year, month, CITY)
    arr = raster["array"]
    bounds = raster["bounds"]
    stats = pp.get_stats(arr)
    lit_area = pp.get_lit_area_km2(arr)

    st.subheader(f"{labels[st.session_state.date_idx]}")

    if not ts_df.empty:
        baseline_mean = ts_df[ts_df["year"] == ts_df["year"].min()]["mean_rad"].mean()
        pct = pp.pct_change_from_baseline(stats["mean"], baseline_mean)
    else:
        pct = 0.0

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Min Radiance", f"{stats['min']:.3f}", help="nW/cm²/sr")
    s2.metric("Max Radiance", f"{stats['max']:.3f}", help="nW/cm²/sr")
    s3.metric("Mean Radiance", f"{stats['mean']:.3f}", help="nW/cm²/sr")
    s4.metric("Lit Area", f"{lit_area:.1f} km²", help="Pixels > 0.5 nW/cm²/sr")
    s5.metric("vs 2014 Baseline", f"{pct:+.1f}%", delta=f"{pct:.1f}%")

    st.markdown("---")

    map_col, info_col = st.columns([3, 1])

    with map_col:
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=basemap)

        folium.Rectangle(
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            color="#ffffff", weight=1.5, fill=False, tooltip=f"{CFG['display_name']} AOI",
        ).add_to(m)

        vmin = float(stats["min"])
        vmax = max(float(stats["max"]), 0.01)
        png_b64 = ut.array_to_png_base64(arr, cmap_name=cmap_name, vmin=vmin, vmax=vmax, opacity=opacity)
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{png_b64}",
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=1.0,
            name="NTL Radiance",
        ).add_to(m)

        legend_html = ut.get_colorbar_html(vmin, vmax, cmap_name)
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)

        st_folium(m, width=700, height=500, returned_objects=[])

    with info_col:
        st.markdown("### 📋 Raster Info")
        st.markdown(f"- **Date:** {labels[st.session_state.date_idx]}")
        st.markdown(f"- **Shape:** {arr.shape[0]} × {arr.shape[1]} px")
        st.markdown(f"- **Valid pixels:** {stats['valid_pixels']:,} / {stats['total_pixels']:,}")
        st.markdown(f"- **Std deviation:** {stats['std']:.3f}")
        st.markdown(f"- **Median:** {stats['median']:.3f}")

        st.markdown("---")
        st.markdown("### 🎮 Controls")
        st.markdown("""
- Use the **slider** in the sidebar to pick a month
- Press **▶ Play** to animate through all months
- Adjust **speed** to control animation pace
- Change **colormap** to alter visualization style
- Toggle **base map** for different backgrounds
        """)

