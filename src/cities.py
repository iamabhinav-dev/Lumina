"""
src/cities.py
==============
Central city configuration for the VIIRS NTL pipeline.

Every pipeline script imports this to resolve city-specific paths and
parameters instead of hardcoding them. Add new cities here only.

Usage
-----
    from cities import get_city, get_tiff_dir, get_sarima_dir, \
                       get_lstm_dir, get_convlstm_dir, get_convlstm_model_dir

    cfg = get_city("kolkata")
    bbox = cfg["bbox"]          # [min_lon, min_lat, max_lon, max_lat]
    color = cfg["color"]        # hex string for charts
"""

import os

# ─── City registry ────────────────────────────────────────────────────────────

CITIES: dict[str, dict] = {
    "kharagpur": {
        "display_name": "Kharagpur",
        "bbox":         [87.25, 22.30, 87.45, 22.45],  # [min_lon, min_lat, max_lon, max_lat]
        "center":       [22.375, 87.35],                 # [lat, lon] for map centering
        "state":        "West Bengal",
        "description":  "IIT Kharagpur region — small city, ~35×45 pixel grid",
        "scale":        500,                             # GEE export resolution (m)
        "color":        "#4C72B0",                       # chart / map colour
    },
    "kolkata": {
        "display_name": "Kolkata",
        "bbox":         [88.20, 22.40, 88.55, 22.75],
        "center":       [22.575, 88.375],
        "state":        "West Bengal",
        "description":  "Kolkata Metropolitan Area — major metro, ~70×70 pixel grid",
        "scale":        500,
        "color":        "#DD8452",
    },
}


# ─── Lookup helper ────────────────────────────────────────────────────────────

def get_city(city: str) -> dict:
    """Return the config dict for *city*; raises ValueError for unknown names."""
    key = city.lower().strip()
    if key not in CITIES:
        raise ValueError(
            f"Unknown city {city!r}. Available: {sorted(CITIES.keys())}"
        )
    return CITIES[key]


# ─── Path helpers ─────────────────────────────────────────────────────────────

def get_tiff_dir(city: str, root: str) -> str:
    """
    Return the GeoTIFF directory for the given city.

    Kharagpur uses the legacy path ``data/tiffs/`` so that existing data
    does not need to be moved.  All other cities use ``data/{city}/tiffs/``.
    """
    key = city.lower().strip()
    if key == "kharagpur":
        return os.path.join(root, "data", "tiffs")
    return os.path.join(root, "data", key, "tiffs")


def get_sarima_dir(city: str, root: str) -> str:
    """
    Return the SARIMA output directory for the given city.

    Kharagpur uses the legacy path ``outputs/sarima/``.
    All other cities use ``outputs/{city}/sarima/``.
    """
    key = city.lower().strip()
    if key == "kharagpur":
        return os.path.join(root, "outputs", "sarima")
    return os.path.join(root, "outputs", key, "sarima")


def get_lstm_dir(city: str, root: str) -> str:
    """
    Return the LSTM output directory for the given city.

    Kharagpur uses the legacy path ``outputs/lstm/``.
    All other cities use ``outputs/{city}/lstm/``.
    """
    key = city.lower().strip()
    if key == "kharagpur":
        return os.path.join(root, "outputs", "lstm")
    return os.path.join(root, "outputs", key, "lstm")


def get_convlstm_dir(city: str, root: str) -> str:
    """
    Return the ConvLSTM output directory for the given city.

    Kharagpur uses the legacy path ``outputs/convlstm/``.
    All other cities use ``outputs/{city}/convlstm/``.
    """
    key = city.lower().strip()
    if key == "kharagpur":
        return os.path.join(root, "outputs", "convlstm")
    return os.path.join(root, "outputs", key, "convlstm")


def get_convlstm_model_dir(city: str, root: str) -> str:
    """
    Return the directory that holds ConvLSTM model artefacts.

    Kharagpur uses the legacy path ``models/convlstm/`` (frames.npz etc.
    sit directly in that folder).  All other cities use
    ``models/convlstm/{city}/``.
    """
    key = city.lower().strip()
    if key == "kharagpur":
        return os.path.join(root, "models", "convlstm")
    return os.path.join(root, "models", "convlstm", key)
