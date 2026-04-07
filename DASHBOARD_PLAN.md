# VIIRS NTL Dashboard — Detailed Plan
### Kharagpur Night Time Light Change Detection

---

## 1. Project Structure

```
BTP/
├── data/
│   └── tiffs/                  # downloaded GeoTIFFs (ntl_YYYY_MM.tif)
├── src/
│   ├── download_data.py        # GEE downloader (done)
│   ├── preprocess.py           # raster reading + caching
│   └── utils.py                # helper functions
├── dashboard/
│   └── app.py                  # main Streamlit app
├── requirements.txt
└── venv/
```

---

## 2. Tech Stack

| Purpose | Library |
|---|---|
| Dashboard framework | `streamlit` |
| Map rendering | `folium` + `streamlit-folium` |
| Charts & plots | `plotly` |
| Raster reading | `rasterio` |
| Array operations | `numpy` |
| Data wrangling | `pandas` |
| Colormap | `matplotlib` (cmaps only) |

Install:
```bash
pip install streamlit folium streamlit-folium plotly rasterio numpy pandas matplotlib
```

---

## 3. Dashboard Layout

```
┌──────────────────────────────────────────────────────────────┐
│  VIIRS Night Time Light — Kharagpur                [sidebar] │
├──────────────────────────────────────────────────────────────┤
│  SIDEBAR:                                                     │
│  • Date Slider (T1 → T2)                                     │
│  • ▶ Play / ⏸ Pause button                                   │
│  • Colormap selector (viridis / hot / magma)                 │
│  • View mode: Single / Compare / Difference                  │
├──────────────────┬───────────────────────────────────────────┤
│                  │                                            │
│   MAP PANEL      │   CHART PANEL                             │
│                  │                                            │
│  Folium map with │  • Time series line chart                 │
│  NTL overlay     │    (avg radiance over time)               │
│  Kharagpur bbox  │  • Monthly bar chart                      │
│  Color legend    │  • Year-over-year comparison              │
│                  │  • Histogram of pixel values              │
├──────────────────┴───────────────────────────────────────────┤
│  BOTTOM: Stats row                                            │
│  Min | Max | Mean Radiance | % Change from baseline          │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. View Modes

### Mode 1: Single View
- One time slider → shows map for selected month
- Map panel shows NTL raster overlay for that month
- Charts update to highlight selected month

### Mode 2: Compare View (Side by Side)
- Two sliders: T1 and T2
- Side-by-side maps rendered as two Folium overlays
- Difference stats shown between T1 and T2

### Mode 3: Difference Map (T2 − T1)
- Single map showing `raster_T2 - raster_T1`
- Red = increased light, Blue = decreased light
- Shows urbanization / light loss clearly

---

## 5. Components Breakdown

### 5.1 `src/preprocess.py`
Handles all data loading, caching, and processing:

```
Functions:
- get_available_dates()         → list of (year, month) tuples from tiffs/
- load_raster(year, month)      → returns (numpy_array, transform, crs, bounds)
- get_stats(array)              → {min, max, mean, std, nodata_count}
- raster_to_png_overlay(array)  → base64 PNG for Folium overlay
- compute_difference(arr1,arr2) → difference array + stats
- build_timeseries_df()         → pandas DataFrame: date | mean_rad | max_rad
```

**Caching strategy:** Use `@st.cache_data` decorator so rasters are only read once per session, not on every slider move.

### 5.2 `src/utils.py`
Small helpers:

```
Functions:
- normalize_array(arr, vmin, vmax) → scales to 0–255 for PNG
- array_to_colormap_image(arr, cmap) → PIL image with colormap applied
- get_colorbar_html(vmin, vmax, cmap) → HTML string for legend
- date_label(year, month)         → "Jan 2014" string
```

### 5.3 `dashboard/app.py`
Main Streamlit application — see Section 6 for full logic.

---

## 6. App Logic Flow

```
App startup
    │
    ├── Load all available dates from data/tiffs/
    ├── Build time series DataFrame (cached)
    │
Sidebar renders
    ├── Date slider (index into dates list)
    ├── Play button (auto-increments slider via st.session_state)
    ├── Colormap dropdown
    └── View mode radio
    │
Main panel renders
    ├── [Single mode]
    │     ├── Load raster for selected date (cached)
    │     ├── Convert to colorized PNG overlay
    │     ├── Render Folium map with overlay + bbox
    │     └── Show charts + stats
    │
    ├── [Compare mode]
    │     ├── Load raster T1 and T2
    │     ├── Render two maps side by side (st.columns)
    │     └── Show difference stats
    │
    └── [Difference mode]
          ├── Compute T2 - T1 array
          ├── Colorize with diverging colormap (RdBu)
          ├── Render single map
          └── Show change detection charts
```

---

## 7. Charts & Visualizations

### Chart 1: Time Series Line Chart
- X-axis: Date (monthly)
- Y-axis: Mean radiance (nW/cm²/sr)
- Vertical line marker at selected date
- Built with `plotly.graph_objects`

### Chart 2: Monthly Seasonal Bar Chart
- Groups by month (Jan–Dec) across all years
- Shows seasonal NTL patterns
- Useful for detecting festivals, winter fog effects

### Chart 3: Year-over-Year Comparison
- Line per year (2014, 2015, ..., 2025)
- X-axis: Month (1–12)
- Shows growth trend clearly

### Chart 4: Pixel Radiance Histogram
- Distribution of all pixel values for selected date
- Log scale on Y-axis
- Shows urban vs rural pixel spread

### Chart 5: Change Detection Summary
- Bar chart showing % change per year vs 2014 baseline
- Red bars = increased light, Blue = decreased

---

## 8. Map Implementation

### Raster Overlay Approach
GeoTIFFs are converted to PNG images and added as `folium.raster_layers.ImageOverlay`:

```python
# Pseudocode
image_png = array_to_colormap_image(raster_array, colormap)
folium.raster_layers.ImageOverlay(
    image=image_png,
    bounds=[[22.30, 87.25], [22.45, 87.45]],
    opacity=0.7,
    name="NTL"
).add_to(m)
```

### Base Map
- `OpenStreetMap` or `CartoDB dark_matter` (dark base suits NTL data perfectly)
- Kharagpur bounding box rectangle overlay
- Layer control for toggling

### Color Legend
- Custom HTML injected into Folium map
- Shows colormap gradient with min/max radiance values

---

## 9. Animation / Play Feature

Using `st.session_state` to implement auto-play:

```python
# Pseudocode
if st.sidebar.button("▶ Play"):
    st.session_state.playing = True

if st.session_state.playing:
    st.session_state.date_index += 1
    time.sleep(0.5)
    st.rerun()
```

Speed control slider: 0.2s – 2s per frame.

---

## 10. Stats Row (Bottom Panel)

| Metric | Description |
|---|---|
| Min Radiance | Darkest pixel value |
| Max Radiance | Brightest pixel value |
| Mean Radiance | Average across all pixels |
| Lit Area (km²) | Pixels above threshold × pixel area |
| % Change from 2014 | (current_mean − baseline_mean) / baseline_mean × 100 |

---

## 11. Build Order (Step by Step)

| Step | Task | File |
|---|---|---|
| 1 | Install dashboard dependencies | `requirements.txt` |
| 2 | Write `preprocess.py` — raster loading + stats | `src/preprocess.py` |
| 3 | Write `utils.py` — colormap + PNG conversion | `src/utils.py` |
| 4 | Build basic app skeleton — layout + sidebar | `dashboard/app.py` |
| 5 | Add single-view map with slider | `dashboard/app.py` |
| 6 | Add time series chart | `dashboard/app.py` |
| 7 | Add play/animation feature | `dashboard/app.py` |
| 8 | Add compare mode (side-by-side) | `dashboard/app.py` |
| 9 | Add difference map mode | `dashboard/app.py` |
| 10 | Add remaining charts (histogram, YoY, seasonal) | `dashboard/app.py` |
| 11 | Add stats row | `dashboard/app.py` |
| 12 | Polish: colors, layout, legend, dark theme | `dashboard/app.py` |

---

## 12. Running the Dashboard

```bash
cd /home/abhinav/Desktop/BTP
source venv/bin/activate
pip install streamlit folium streamlit-folium plotly matplotlib

streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`

---

## 13. NoData Handling

VIIRS data has pixels with fill values (typically `65535` or `−999`). These must be masked before rendering:

```python
# Mask nodata
array = np.where(array > 1000, np.nan, array)  # values > 1000 are fill
array = np.where(array < 0, np.nan, array)
```

---

## 14. Potential Future Additions

- **NDVI overlay** — correlate NTL with vegetation loss
- **Admin boundary** — Kharagpur municipal boundary shapefile overlay
- **Anomaly detection** — flag months with unusual spikes/drops  
- **Export panel** — download chart as PNG or data as CSV
- **Pixel time series** — click on map to see that pixel's NTL over time (needs `streamlit-plotly-events`)
