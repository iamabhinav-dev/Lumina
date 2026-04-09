"""
app.py — Main Streamlit entry point
VIIRS Night Time Light Dashboard — multi-city
"""

import sys
import os
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cities as _cities

st.set_page_config(
    page_title="VIIRS NTL Dashboard",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── City selector ────────────────────────────────────────────────────────────

CITY_OPTIONS = {cfg["display_name"]: key for key, cfg in _cities.CITIES.items()}

if "city" not in st.session_state:
    st.session_state["city"] = "kharagpur"

selected_display = st.sidebar.radio(
    "🌆 Select City",
    list(CITY_OPTIONS.keys()),
    index=list(CITY_OPTIONS.keys()).index(
        _cities.get_city(st.session_state["city"])["display_name"]
    ),
)
st.session_state["city"] = CITY_OPTIONS[selected_display]
city = st.session_state["city"]
cfg  = _cities.get_city(city)

# ─── Main page ────────────────────────────────────────────────────────────────

bbox     = cfg["bbox"]
area_str = f"{bbox[0]}°E–{bbox[2]}°E, {bbox[1]}°N–{bbox[3]}°N"

st.title("🌃 VIIRS Night Time Light Dashboard")
st.subheader(f"{cfg['display_name']}, {cfg['state']}, India — 2014 to 2025")

st.markdown(f"""
Welcome to the VIIRS NTL (Night Time Light) analysis platform for **{cfg['display_name']}**.

---

### Pages

| Page | Description |
|---|---|
| **🗺️ NTL Explorer** | Browse monthly NTL maps with time slider and animation |
| **🔍 Change Detection** | Compare two dates, visualize difference maps |
| **📊 Charts & Trends** | Time series, seasonal patterns, year-over-year, histograms |
| **🔮 SARIMA Forecast** | SARIMA(0,1,1)(0,1,1)[12] forecast with 95% CI and model diagnostics |
| **🧠 LSTM Forecast** | Stacked LSTM with MC Dropout uncertainty, head-to-head vs SARIMA |
| **🗺️ ConvLSTM Spatial Forecast** | Pixel-level spatial NTL maps — test evaluation + 12-month forecast with CI, GeoTIFF download |
| **📊 City Comparison** | Side-by-side Kharagpur vs Kolkata — time series, model accuracy, spatial maps, growth |

Use the **sidebar** to navigate between pages and select a city.

---

### Data Source
- **Dataset:** NOAA VIIRS DNB Monthly V1 (`VCMSLCFG`)
- **Band:** `avg_rad` — average radiance (nW/cm²/sr)
- **Resolution:** ~500m
- **Coverage:** January 2014 – December 2025
- **Area:** {cfg['display_name']}, {cfg['state']} ({area_str})
""")

col1, col2, col3 = st.columns(3)
col1.metric("Total Months", "144")
col2.metric("Time Range", "2014–2025")
col3.metric("Spatial Resolution", "~500 m")
