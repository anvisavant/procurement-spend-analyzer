"""
app.py — Procurement Spend Analyzer
Full Streamlit dashboard with Plotly charts.

Sections:
    1. KPI header row
    2. Spend by category (bar) + Spend over time (line)
    3. Top 10 vendors by spend (table)
    4. Supplier concentration risk heatmap
    5. Savings opportunity sizer

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.categorizer import enrich_dataframe
from src.risk_scorer import ConcentrationRiskScorer
from src.savings_engine import SavingsEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Spend Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme tokens (inline CSS) ─────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: #f9f8f5;
        border: 1px solid #dcd9d5;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
    }
    .kpi-label { font-size: 0.78rem; color: #7a7974; text-transform: uppercase;
                 letter-spacing: 0.06em; margin-bottom: 4px; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #28251d; }
    .kpi-delta { font-size: 0.8rem; color: #437a22; margin-top: 2px; }
    .kpi-delta.neg { color: #a13544; }
    .section-header { font-size: 1.05rem; font-weight: 600;
                      color: #28251d; margin-bottom: 0.2rem; }
    .risk-high   { color: #a12c7b; font-weight: 600; }
    .risk-medium { color: #da7101; font-weight: 600; }
    .risk-low    { color: #437a22; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Color palette for charts ──────────────────────────────────────────────────
CAT_COLORS = [
    "#01696f", "#da7101", "#006494", "#7a39bb", "#437a22",
    "#a12c7b", "#d19900", "#a13544", "#4f98a3", "#bb653b",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Spend Analyzer")
    st.caption("Spendscape-inspired procurement intelligence")
    st.divider()

    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=True)
    st.divider()

    st.markdown("**Filters**")
    # Populated after data loads
    dept_filter = st.multiselect("Department", [], placeholder="All departments")
    cat_filter  = st.multiselect("Category",   [], placeholder="All categories")
    st.divider()
    st.caption("v1.0.0")


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and categorizing transactions…")
def load_data(source) -> pd.DataFrame:
    if source == "demo":
        df = pd.read_csv(Path("data/mock_spend.csv"))
    else:
        df = pd.read_csv(source)

    df["date"]   = pd.to_datetime(df["date"])
    df["month"]  = df["date"].dt.to_period("M").astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Categorize if column missing or has blanks
    if "category" not in df.columns:
        df = enrich_dataframe(df, use_nlp=False)
    else:
        df = enrich_dataframe(df, use_nlp=False)

    return df


if uploaded:
    df = load_data(uploaded)
elif use_demo:
    df = load_data("demo")
else:
    st.info("Upload a CSV or enable demo mode to get started.")
    st.stop()


# ── Apply sidebar filters ─────────────────────────────────────────────────────
all_depts = sorted(df["department"].dropna().unique())
all_cats  = sorted(df["category"].dropna().unique())

# Re-render multiselects with real options (Streamlit limitation workaround)
with st.sidebar:
    dept_filter = st.multiselect("Department", all_depts,
                                 default=[], key="dept_f",
                                 placeholder="All departments")
    cat_filter  = st.multiselect("Category", all_cats,
                                 default=[], key="cat_f",
                                 placeholder="All categories")

dff = df.copy()
if dept_filter:
    dff = dff[dff["department"].isin(dept_filter)]
if cat_filter:
    dff = dff[dff["category"].isin(cat_filter)]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Procurement Spend Analyzer")
st.caption("Auto-categorized spend intelligence · Spendscape methodology")
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI Cards
# ═══════════════════════════════════════════════════════════════════════════════
total_spend   = dff["amount"].sum()
total_vendors = dff["vendor_name"].nunique()
total_txns    = len(dff)
avg_txn       = dff["amount"].mean()
num_cats      = dff["category"].nunique()

# Month-over-month delta for total spend
monthly = dff.groupby("month")["amount"].sum().sort_index()
if len(monthly) >= 2:
    last, prev = monthly.iloc[-1], monthly.iloc[-2]
    mom_delta  = (last - prev) / prev * 100 if prev else 0
    mom_label  = f"{'▲' if mom_delta >= 0 else '▼'} {abs(mom_delta):.1f}% vs prior month"
    mom_class  = "" if mom_delta >= 0 else "neg"
else:
    mom_label, mom_class = "—", ""

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, delta="", delta_class=""):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {"<div class='kpi-delta " + delta_class + "'>" + delta + "</div>" if delta else ""}
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Spend",      f"${total_spend:,.0f}",   mom_label, mom_class)
kpi(k2, "Transactions",     f"{total_txns:,}")
kpi(k3, "Unique Vendors",   f"{total_vendors:,}")
kpi(k4, "Avg Transaction",  f"${avg_txn:,.0f}")
kpi(k5, "Spend Categories", f"{num_cats}")

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Spend by Category + Spend Over Time
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Spend Breakdown</p>', unsafe_allow_html=True)
col_bar, col_line = st.columns([1, 1.6])

with col_bar:
    cat_totals = (
        dff.groupby("category")["amount"].sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig_bar = px.bar(
        cat_totals, x="amount", y="category", orientation="h",
        color="category", color_discrete_sequence=CAT_COLORS,
        labels={"amount": "Total Spend ($)", "category": ""},
        title="Total Spend by Category",
    )
    fig_bar.update_layout(
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        font_color="#28251d", title_font_size=13,
        margin=dict(l=0, r=10, t=40, b=10),
        xaxis=dict(tickformat="$,.0f", gridcolor="#f0ede8"),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_line:
    monthly_cat = (
        dff.groupby(["month", "category"])["amount"].sum()
        .reset_index()
    )
    fig_line = px.line(
        monthly_cat, x="month", y="amount", color="category",
        color_discrete_sequence=CAT_COLORS,
        labels={"amount": "Spend ($)", "month": "", "category": "Category"},
        title="Monthly Spend by Category",
        markers=True,
    )
    fig_line.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_color="#28251d", title_font_size=13,
        margin=dict(l=0, r=10, t=40, b=10),
        yaxis=dict(tickformat="$,.0f", gridcolor="#f0ede8"),
        xaxis=dict(gridcolor="#f0ede8"),
        legend=dict(font=dict(size=10), orientation="v"),
    )
    st.plotly_chart(fig_line, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Top 10 Vendors + MoM Change Table
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Vendor Intelligence</p>', unsafe_allow_html=True)
col_v, col_mom = st.columns(2)

with col_v:
    top_vendors = (
        dff.groupby("vendor_name")["amount"].sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"vendor_name": "Vendor", "amount": "Total Spend"})
    )
    top_vendors["% of Portfolio"] = (
        top_vendors["Total Spend"] / total_spend * 100
    ).map("{:.1f}%".format)
    top_vendors["Total Spend"] = top_vendors["Total Spend"].map("${:,.0f}".format)
    st.markdown("**Top 10 Vendors by Spend**")
    st.dataframe(top_vendors, use_container_width=True, hide_index=True)

with col_mom:
    mom_table = (
        dff.groupby(["category", "month"])["amount"].sum()
        .reset_index()
        .sort_values(["category", "month"])
    )
    mom_table["prev"] = mom_table.groupby("category")["amount"].shift(1)
    mom_table["MoM %"] = ((mom_table["amount"] - mom_table["prev"])
                          / mom_table["prev"] * 100).round(1)
    latest_month = mom_table["month"].max()
    mom_latest = (
        mom_table[mom_table["month"] == latest_month][["category", "amount", "MoM %"]]
        .rename(columns={"category": "Category", "amount": "Spend (Latest Mo.)"})
    )
    mom_latest["Spend (Latest Mo.)"] = mom_latest["Spend (Latest Mo.)"].map("${:,.0f}".format)
    mom_latest["MoM %"] = mom_latest["MoM %"].map(
        lambda x: f"▲ {x:.1f}%" if x > 0 else (f"▼ {abs(x):.1f}%" if x < 0 else "—")
        if pd.notna(x) else "—"
    )
    st.markdown(f"**Month-over-Month Change ({latest_month})**")
    st.dataframe(mom_latest, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Supplier Concentration Risk
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Supplier Concentration Risk</p>',
            unsafe_allow_html=True)

scorer  = ConcentrationRiskScorer(dff)
summary = scorer.category_risk_summary()
heatmap = scorer.risk_heatmap_matrix()

col_risk, col_hmap = st.columns([1.2, 1])

with col_risk:
    display = summary[["category", "top_vendor", "top_vendor_pct",
                        "risk_level", "vendor_count", "total_spend"]].copy()
    display.columns = ["Category", "Top Vendor", "Conc. %", "Risk", "# Vendors", "Spend"]
    display["Spend"] = display["Spend"].map("${:,.0f}".format)
    display["Conc. %"] = display["Conc. %"].map("{:.1f}%".format)

    def style_risk(val):
        colors = {"High": "#fce8f3", "Medium": "#fef3e2", "Low": "#eaf3e8"}
        return f"background-color: {colors.get(val, 'white')}"

    st.dataframe(
        display.style.applymap(style_risk, subset=["Risk"]),
        use_container_width=True, hide_index=True
    )

with col_hmap:
    hm_df = heatmap.reset_index()
    risk_order = ["High", "Medium", "Low"]
    z_vals = hm_df[risk_order].values.tolist()
    fig_hm = go.Figure(go.Heatmap(
        z=z_vals,
        x=risk_order,
        y=hm_df["category"].tolist(),
        colorscale=[[0, "#e8f5e9"], [0.5, "#fff3e0"], [1, "#fce4ec"]],
        showscale=False,
        text=z_vals,
        texttemplate="%{text}",
        hovertemplate="Category: %{y}<br>Risk: %{x}<br>Vendors: %{z}<extra></extra>",
    ))
    fig_hm.update_layout(
        title="Vendor Count by Risk Level",
        title_font_size=13,
        plot_bgcolor="white", paper_bgcolor="white",
        font_color="#28251d",
        margin=dict(l=0, r=0, t=40, b=10),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Savings Opportunity Sizer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">Savings Opportunity Sizer</p>',
            unsafe_allow_html=True)

engine  = SavingsEngine(dff)
opps    = engine.opportunity_table()
lo_tot, hi_tot = engine.total_savings_range()

st.markdown(
    f"**Estimated portfolio savings potential: "
    f"${lo_tot:,.0f} – ${hi_tot:,.0f}** "
    f"*(McKinsey 3–15% category benchmarks)*"
)

col_sav, col_sbar = st.columns([1, 1.4])

with col_sav:
    disp_opps = opps[["rank", "category", "total_spend",
                       "savings_low", "savings_high", "low_pct", "high_pct"]].copy()
    disp_opps.columns = ["#", "Category", "Spend", "Savings Low",
                          "Savings High", "Low %", "High %"]
    for col in ["Spend", "Savings Low", "Savings High"]:
        disp_opps[col] = disp_opps[col].map("${:,.0f}".format)
    for col in ["Low %", "High %"]:
        disp_opps[col] = disp_opps[col].map("{:.0f}%".format)
    st.dataframe(disp_opps, use_container_width=True, hide_index=True)

with col_sbar:
    fig_sav = go.Figure()
    fig_sav.add_trace(go.Bar(
        name="Conservative (low)",
        x=opps["category"],
        y=opps["savings_low"],
        marker_color="#cedcd8",
    ))
    fig_sav.add_trace(go.Bar(
        name="Upside (high)",
        x=opps["category"],
        y=opps["savings_high"] - opps["savings_low"],
        marker_color="#01696f",
        base=opps["savings_low"],
    ))
    fig_sav.update_layout(
        barmode="stack",
        title="Savings Range by Category",
        title_font_size=13,
        plot_bgcolor="white", paper_bgcolor="white",
        font_color="#28251d",
        margin=dict(l=0, r=10, t=40, b=80),
        yaxis=dict(tickformat="$,.0f", gridcolor="#f0ede8"),
        xaxis=dict(tickangle=-35),
        legend=dict(font=dict(size=10)),
        showlegend=True,
    )
    st.plotly_chart(fig_sav, use_container_width=True)

st.divider()
st.caption("Anomaly detection → commit 5 · Procurement Spend Analyzer v1.0")
