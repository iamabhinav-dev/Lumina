"""
app.py — Main Streamlit entry point
VIIRS Night Time Light Dashboard — Kharagpur, West Bengal
"""

import streamlit as st

st.set_page_config(
    page_title="VIIRS NTL — Kharagpur",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌃 VIIRS Night Time Light Dashboard")
st.subheader("Kharagpur, West Bengal, India — 2014 to 2025")

st.markdown("""
Welcome to the VIIRS NTL (Night Time Light) analysis platform for **Kharagpur**.

---

### Pages

| Page | Description |
|---|---|
| **🗺️ NTL Explorer** | Browse monthly NTL maps with time slider and animation |
| **🔍 Change Detection** | Compare two dates, visualize difference maps |
| **📊 Charts & Trends** | Time series, seasonal patterns, year-over-year, histograms |
| **🔮 SARIMA Forecast** | SARIMA(0,1,1)(0,1,1)[12] forecast with 95% CI and model diagnostics |

Use the **sidebar** to navigate between pages.

---

### Data Source
- **Dataset:** NOAA VIIRS DNB Monthly V1 (`VCMSLCFG`)
- **Band:** `avg_rad` — average radiance (nW/cm²/sr)
- **Resolution:** ~500m
- **Coverage:** January 2014 – December 2025
- **Area:** Kharagpur, West Bengal (87.25°E–87.45°E, 22.30°N–22.45°N)
""")

col1, col2, col3 = st.columns(3)
col1.metric("Total Months", "144")
col2.metric("Time Range", "2014–2025")
col3.metric("Spatial Resolution", "~500 m")
