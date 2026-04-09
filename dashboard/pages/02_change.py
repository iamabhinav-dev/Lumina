"""
02_change.py — Change Detection Page
Side-by-side map comparison + difference map (T2 − T1).
"""

import sys
import os
import folium
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import preprocess as pp
import cities as _cities
import utils as ut

st.set_page_config(page_title="Change Detection", page_icon="🔍", layout="wide")


# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading raster...")
def cached_load_raster(year: int, month: int, city: str):
    return pp.load_raster(year, month, city)


@st.cache_data
def cached_dates(city: str):
    return pp.get_available_dates(city)


# ─── Load data ────────────────────────────────────────────────────────────────

CITY   = st.session_state.get("city", "kharagpur")
CFG    = _cities.get_city(CITY)
dates  = cached_dates(CITY)
labels = ut.dates_to_labels(dates)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🔍 Change Detection")

mode = st.sidebar.radio(
    "View Mode",
    ["Side-by-Side Comparison", "Difference Map (T2 − T1)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Select Dates")

t1_idx = st.sidebar.selectbox("T1 (Before)", range(len(dates)), index=0,
                               format_func=lambda i: labels[i])
t2_idx = st.sidebar.selectbox("T2 (After)", range(len(dates)), index=len(dates) - 1,
                               format_func=lambda i: labels[i])

if t1_idx == t2_idx:
    st.sidebar.warning("T1 and T2 are the same date.")

st.sidebar.markdown("---")
cmap_label = st.sidebar.selectbox("Colormap", list(ut.COLORMAPS.keys()), index=0)
cmap_name = ut.COLORMAPS[cmap_label]
opacity = st.sidebar.slider("Overlay opacity", 0.3, 1.0, 0.75, 0.05)

basemap = st.sidebar.radio(
    "Base Map", ["CartoDB dark_matter", "CartoDB positron", "OpenStreetMap"], index=0
)

# ─── Load rasters ─────────────────────────────────────────────────────────────

y1, m1 = dates[t1_idx]
y2, m2 = dates[t2_idx]

raster1 = cached_load_raster(y1, m1, CITY)
raster2 = cached_load_raster(y2, m2, CITY)

arr1, arr2 = raster1["array"], raster2["array"]
bounds = raster1["bounds"]  # same AOI for both

stats1 = pp.get_stats(arr1)
stats2 = pp.get_stats(arr2)

# ─── Helper: build folium map ─────────────────────────────────────────────────

def make_map(arr, stats, title, cmap_name, opacity, basemap, bounds):
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=basemap)

    folium.Rectangle(
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        color="#ffffff", weight=1.5, fill=False,
    ).add_to(m)

    vmin = float(stats["min"])
    vmax = max(float(stats["max"]), 0.01)
    png_b64 = ut.array_to_png_base64(arr, cmap_name=cmap_name, vmin=vmin, vmax=vmax, opacity=opacity)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{png_b64}",
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=1.0,
        name=title,
    ).add_to(m)

    legend_html = ut.get_colorbar_html(vmin, vmax, cmap_name)
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ─── Main panel ───────────────────────────────────────────────────────────────

st.title("🔍 Change Detection")

if mode == "Side-by-Side Comparison":
    st.subheader(f"Comparing {labels[t1_idx]} vs {labels[t2_idx]}")

    # Stats comparison
    c1, c2, c3, c4 = st.columns(4)
    delta_mean = stats2["mean"] - stats1["mean"]
    delta_lit  = pp.get_lit_area_km2(arr2) - pp.get_lit_area_km2(arr1)

    c1.metric("T1 Mean Radiance", f"{stats1['mean']:.3f}")
    c2.metric("T2 Mean Radiance", f"{stats2['mean']:.3f}",
              delta=f"{delta_mean:+.3f}", delta_color="normal")
    c3.metric("T1 Lit Area", f"{pp.get_lit_area_km2(arr1):.1f} km²")
    c4.metric("T2 Lit Area", f"{pp.get_lit_area_km2(arr2):.1f} km²",
              delta=f"{delta_lit:+.1f} km²", delta_color="normal")

    st.markdown("---")

    map_col1, map_col2 = st.columns(2)

    with map_col1:
        st.markdown(f"#### T1: {labels[t1_idx]}")
        m1_map = make_map(arr1, stats1, labels[t1_idx], cmap_name, opacity, basemap, bounds)
        st_folium(m1_map, width=480, height=420, key="map_t1", returned_objects=[])

    with map_col2:
        st.markdown(f"#### T2: {labels[t2_idx]}")
        m2_map = make_map(arr2, stats2, labels[t2_idx], cmap_name, opacity, basemap, bounds)
        st_folium(m2_map, width=480, height=420, key="map_t2", returned_objects=[])

else:  # Difference Map
    st.subheader(f"Difference Map: {labels[t2_idx]} − {labels[t1_idx]}")

    diff_result = pp.compute_difference(arr1, arr2)
    diff_arr = diff_result["diff"]
    diff_stats = diff_result["stats"]

    import numpy as np
    valid_diff = diff_arr[np.isfinite(diff_arr)]
    vabs = max(abs(float(np.nanmin(valid_diff))), abs(float(np.nanmax(valid_diff)))) if valid_diff.size > 0 else 1.0

    # Stats row
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Mean Change", f"{diff_stats['mean']:+.3f} nW/cm²/sr")
    d2.metric("Increased Pixels", f"{diff_stats['increased']:,}", delta="brighter")
    d3.metric("Decreased Pixels", f"{diff_stats['decreased']:,}", delta="dimmer", delta_color="inverse")
    d4.metric("Unchanged Pixels", f"{diff_stats['unchanged']:,}")

    st.markdown("---")

    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m_diff = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=basemap)

    folium.Rectangle(
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        color="#ffffff", weight=1.5, fill=False,
    ).add_to(m_diff)

    diff_png = ut.diff_array_to_png_base64(diff_arr, opacity=opacity)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{diff_png}",
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        opacity=1.0,
        name="Difference",
    ).add_to(m_diff)

    diff_legend = ut.get_diff_colorbar_html(vabs)
    m_diff.get_root().html.add_child(folium.Element(diff_legend))

    map_col, info_col = st.columns([2, 1])
    with map_col:
        st_folium(m_diff, width=700, height=500, key="map_diff", returned_objects=[])

    with info_col:
        st.markdown("### 📊 Change Summary")
        import plotly.graph_objects as go

        labels_pie = ["Increased", "Decreased", "Unchanged"]
        values_pie = [diff_stats["increased"], diff_stats["decreased"], diff_stats["unchanged"]]
        colors_pie = ["#e74c3c", "#3498db", "#95a5a6"]

        fig = go.Figure(go.Pie(
            labels=labels_pie,
            values=values_pie,
            marker_colors=colors_pie,
            hole=0.4,
            textinfo="label+percent",
        ))
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown(f"""
- **T1:** {labels[t1_idx]}
- **T2:** {labels[t2_idx]}
- **Max increase:** +{diff_stats['max']:.3f}
- **Max decrease:** {diff_stats['min']:.3f}
- 🔴 Red = more light (urbanization/growth)
- 🔵 Blue = less light (power outage/depopulation)
        """)
