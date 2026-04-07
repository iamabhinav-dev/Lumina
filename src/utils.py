"""
utils.py
Helper functions: colormap conversion, PNG overlay generation, HTML legend.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from PIL import Image

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

COLORMAPS = {
    "Magma (recommended)": "magma",
    "Viridis": "viridis",
    "Hot": "hot",
    "Inferno": "inferno",
    "Yellow-Orange-Red": "YlOrRd",
}

DIFF_COLORMAP = "RdBu_r"  # red=increase, blue=decrease


# ─── Normalization ───────────────────────────────────────────────────────────

def normalize_array(arr: np.ndarray, vmin: float = None, vmax: float = None) -> np.ndarray:
    """Scale array to 0–1 range, ignoring NaN."""
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.float32)
    normed = (arr - vmin) / (vmax - vmin)
    return np.clip(normed, 0, 1)


# ─── Array → PNG overlay ─────────────────────────────────────────────────────

def array_to_png_base64(
    arr: np.ndarray,
    cmap_name: str = "magma",
    vmin: float = None,
    vmax: float = None,
    opacity: float = 0.75,
) -> str:
    """
    Convert a 2D float array to a base64-encoded PNG string
    suitable for folium ImageOverlay.
    NaN values are rendered as fully transparent.
    """
    normed = normalize_array(arr, vmin, vmax)

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(normed)  # shape (H, W, 4)

    # Mask NaN pixels as transparent
    nan_mask = ~np.isfinite(arr)
    rgba[nan_mask, 3] = 0.0
    # Apply opacity to non-NaN pixels
    rgba[~nan_mask, 3] *= opacity

    # Convert to uint8 PIL image
    img_array = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def diff_array_to_png_base64(
    diff_arr: np.ndarray,
    opacity: float = 0.8,
) -> str:
    """
    Convert a difference array (T2−T1) to PNG using RdBu_r diverging colormap.
    Symmetric around 0.
    """
    valid = diff_arr[np.isfinite(diff_arr)]
    if valid.size == 0:
        vabs = 1.0
    else:
        vabs = max(abs(float(np.nanmin(valid))), abs(float(np.nanmax(valid))))
        vabs = vabs if vabs > 0 else 1.0

    return array_to_png_base64(
        diff_arr,
        cmap_name=DIFF_COLORMAP,
        vmin=-vabs,
        vmax=vabs,
        opacity=opacity,
    )


# ─── Colorbar HTML legend ─────────────────────────────────────────────────────

def get_colorbar_html(vmin: float, vmax: float, cmap_name: str = "magma", title: str = "Radiance (nW/cm²/sr)") -> str:
    """Generate an HTML string for a vertical colorbar legend to inject into Folium."""
    cmap = cm.get_cmap(cmap_name)
    gradient_stops = []
    for i in range(10):
        frac = i / 9.0
        r, g, b, _ = cmap(frac)
        gradient_stops.append(
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) {int(frac*100)}%"
        )
    gradient_css = ", ".join(gradient_stops)

    html = f"""
    <div style="
        position: fixed;
        bottom: 30px; right: 15px; z-index: 9999;
        background: rgba(20,20,20,0.85);
        padding: 10px 12px;
        border-radius: 8px;
        color: white;
        font-family: Arial, sans-serif;
        font-size: 12px;
        min-width: 130px;
    ">
        <div style="font-weight:bold; margin-bottom:6px; font-size:11px;">{title}</div>
        <div style="display:flex; align-items:stretch; gap:6px;">
            <div style="
                width: 18px;
                background: linear-gradient(to top, {gradient_css});
                border-radius: 3px;
                height: 120px;
            "></div>
            <div style="display:flex; flex-direction:column; justify-content:space-between; font-size:11px;">
                <span>{vmax:.2f}</span>
                <span>{((vmax+vmin)/2):.2f}</span>
                <span>{vmin:.2f}</span>
            </div>
        </div>
    </div>
    """
    return html


def get_diff_colorbar_html(vabs: float) -> str:
    """Colorbar for difference maps (symmetric RdBu_r)."""
    cmap = cm.get_cmap(DIFF_COLORMAP)
    gradient_stops = []
    for i in range(10):
        frac = i / 9.0
        r, g, b, _ = cmap(frac)
        gradient_stops.append(
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) {int(frac*100)}%"
        )
    gradient_css = ", ".join(gradient_stops)

    html = f"""
    <div style="
        position: fixed;
        bottom: 30px; right: 15px; z-index: 9999;
        background: rgba(20,20,20,0.85);
        padding: 10px 12px;
        border-radius: 8px;
        color: white;
        font-family: Arial, sans-serif;
        font-size: 12px;
        min-width: 130px;
    ">
        <div style="font-weight:bold; margin-bottom:6px; font-size:11px;">Radiance Change</div>
        <div style="display:flex; align-items:stretch; gap:6px;">
            <div style="
                width: 18px;
                background: linear-gradient(to top, {gradient_css});
                border-radius: 3px;
                height: 120px;
            "></div>
            <div style="display:flex; flex-direction:column; justify-content:space-between; font-size:11px;">
                <span style="color:#ef8a62;">+{vabs:.2f}</span>
                <span>0</span>
                <span style="color:#67a9cf;">-{vabs:.2f}</span>
            </div>
        </div>
        <div style="font-size:10px; margin-top:6px; color:#aaa;">Red=Increase<br>Blue=Decrease</div>
    </div>
    """
    return html


# ─── Label helpers ────────────────────────────────────────────────────────────

def date_label(year: int, month: int) -> str:
    return f"{MONTH_NAMES[month - 1]} {year}"


def dates_to_labels(dates: list[tuple[int, int]]) -> list[str]:
    return [date_label(y, m) for y, m in dates]
