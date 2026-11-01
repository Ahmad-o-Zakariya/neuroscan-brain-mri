"""Single entry point for the NeuroScan research dashboard."""

import streamlit as st

st.set_page_config(
    page_title="NeuroScan | Brain MRI research",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# An explicit navigation registry disables Streamlit's implicit `pages/`
# discovery, keeping the current multi-task workspace in one dashboard.
from src.ui.dashboard import render_dashboard

st.navigation(
    [st.Page(render_dashboard, title="NeuroScan", icon=":material/neurology:", default=True)]
).run()
