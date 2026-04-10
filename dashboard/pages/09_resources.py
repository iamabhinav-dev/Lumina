"""
09_resources.py — Project Resources
BTP-2 Report (PDF) and Presentation (PPT) for easy access.
"""

import os
import base64
import streamlit as st

st.set_page_config(page_title="Resources", page_icon="📁", layout="wide")

st.title("📁 Project Resources")
st.markdown("Download or preview the BTP-2 thesis report and presentation.")

st.divider()

# ─── Helper ──────────────────────────────────────────────────────────────────

def pdf_download_button(pdf_path: str, label: str, filename: str):
    """Render a styled download button for a PDF file."""
    with open(pdf_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = (
        f'<a href="data:application/pdf;base64,{b64}" '
        f'download="{filename}" '
        f'style="display:inline-block;padding:0.5em 1.2em;background:#4CAF50;'
        f'color:white;border-radius:6px;text-decoration:none;font-weight:bold;">'
        f'⬇️ {label}</a>'
    )
    st.markdown(href, unsafe_allow_html=True)


# ─── Report ──────────────────────────────────────────────────────────────────

st.subheader("📄 BTP-2 Thesis Report")
st.markdown(
    "**Lumina: Multi-Model Forecasting of Night-Time Light Data**  \n"
    "Abhinav Kumar Singh · IIT Kharagpur · April 2026"
)

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "BTP-2-Report.pdf")

if os.path.exists(REPORT_PATH):
    pdf_download_button(REPORT_PATH, "Download Report (PDF)", "BTP-2-Report.pdf")

    st.markdown("#### Preview")
    with open(REPORT_PATH, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        f'width="100%" height="800px" type="application/pdf"></iframe>'
    )
    st.markdown(pdf_display, unsafe_allow_html=True)
else:
    st.warning("Report PDF not found. Place `BTP-2-Report.pdf` in `dashboard/assets/`.")

st.divider()

# ─── Presentation ────────────────────────────────────────────────────────────

st.subheader("📊 BTP-2 Presentation Slides")
st.markdown(
    "**Lumina: Multi-Model Forecasting of Night-Time Light Data**  \n"
    "Abhinav Kumar Singh · IIT Kharagpur · April 2026"
)

PPT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "BTP-PPT.pdf")

if os.path.exists(PPT_PATH):
    pdf_download_button(PPT_PATH, "Download Presentation (PDF)", "BTP-2-Presentation.pdf")

    st.markdown("#### Preview")
    with open(PPT_PATH, "rb") as f:
        base64_ppt = base64.b64encode(f.read()).decode("utf-8")
    ppt_display = (
        f'<iframe src="data:application/pdf;base64,{base64_ppt}" '
        f'width="100%" height="800px" type="application/pdf"></iframe>'
    )
    st.markdown(ppt_display, unsafe_allow_html=True)
else:
    st.info(
        "Presentation not uploaded yet.  \n"
        "Place `BTP-PPT.pdf` in `dashboard/assets/` to enable the preview."
    )
