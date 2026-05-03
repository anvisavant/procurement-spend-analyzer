"""
app.py — Procurement Spend Analyzer
Entry point for the Streamlit dashboard.

Run:
    streamlit run app.py

Expects a CSV with columns:
    transaction_id, date, vendor_name, description, amount, department
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Spend Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Spend Analyzer")
    st.caption("Powered by McKinsey Spendscape methodology")
    st.divider()
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data (mock_spend.csv)", value=True)
    st.divider()
    st.caption("v0.1.0 — scaffold")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Procurement Spend Analyzer")
st.caption("Auto-categorize supplier spend · surface savings · flag risk")

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df):,} transactions from uploaded file.")
elif use_demo:
    demo_path = Path("data/mock_spend.csv")
    if demo_path.exists():
        df = pd.read_csv(demo_path)
        st.info(f"Demo mode — loaded {len(df):,} transactions from `data/mock_spend.csv`")
    else:
        st.warning("Demo file not found. Upload a CSV or add `data/mock_spend.csv`.")
        st.stop()
else:
    st.info("Upload a CSV file or enable demo mode in the sidebar to get started.")
    st.stop()

# ── Preview ───────────────────────────────────────────────────────────────────
with st.expander("Raw data preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.info("🚧 Dashboard coming in next commits — categorization, charts, risk, savings, anomalies.")
