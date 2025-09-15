"""
Marketing Intelligence Report — Streamlit App

Usage:
    streamlit run streamlit_bi_dashboard.py

Place CSVs in the `datasets/` folder:
    - Facebook.csv
    - Google.csv
    - TikTok.csv
    - business.csv  (case-insensitive)

Features:
    - Loads and cleans datasets from datasets/
    - Derived KPIs: CTR, CPC, CPM, ROAS, CAC, AOV, Profit Margin
    - Interactive filters: date range, channel, state, campaign search
    - Visuals: KPI cards, time-series, channel comparison, funnel, state heatmap, campaign table
    - Anomaly alerts and PDF export (fpdf2)
"""
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="Marketing Intelligence Report")

# ---------------------- Header & Assessment Info ----------------------
st.title("Marketing Intelligence Report")
with st.expander("Context"):
    st.markdown(
        """
        Datasets (120 days):
        - Facebook.csv, Google.csv, TikTok.csv: campaign-level marketing (date, tactic, state, campaign, impression, clicks, spend, attributed revenue)
        - business.csv: daily business metrics (orders, new_orders, new_customers, total_revenue, gross_profit, cogs)
        """
    )
with st.expander("Task & Evaluation"):
    st.markdown(
        """
        Build an interactive BI dashboard linking marketing activity to business outcomes.
        Evaluation areas: data preparation, visualization & storytelling, product thinking, delivery.
        """
    )

# ---------------------- Dataset discovery & checks ----------------------
DATASET_DIR = "datasets"
expected_files = {
    "Facebook.csv": os.path.join(DATASET_DIR, "Facebook.csv"),
    "Google.csv": os.path.join(DATASET_DIR, "Google.csv"),
    "TikTok.csv": os.path.join(DATASET_DIR, "TikTok.csv"),
}
# Accept business.csv or Business.csv
business_lower = os.path.join(DATASET_DIR, "business.csv")
business_upper = os.path.join(DATASET_DIR, "Business.csv")
expected_files["business.csv"] = business_lower if os.path.exists(business_lower) else business_upper

missing = [name for name, path in expected_files.items() if not os.path.exists(path)]
if missing:
    st.error("Missing CSV files in `datasets/`. Add these files and reload:")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

FB_PATH = expected_files["Facebook.csv"]
G_PATH = expected_files["Google.csv"]
T_PATH = expected_files["TikTok.csv"]
B_PATH = expected_files["business.csv"]

# ---------------------- Data load & prepare ----------------------
@st.cache_data
def load_and_prepare(fb_path, g_path, t_path, b_path):
    def norm_cols(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    fb = norm_cols(pd.read_csv(fb_path)); fb["channel"] = "Facebook"
    g = norm_cols(pd.read_csv(g_path)); g["channel"] = "Google"
    t = norm_cols(pd.read_csv(t_path)); t["channel"] = "TikTok"
    b = norm_cols(pd.read_csv(b_path))

    marketing = pd.concat([fb, g, t], ignore_index=True, sort=False)

    # normalize impressions
    if "impression" in marketing.columns and "impressions" not in marketing.columns:
        marketing["impressions"] = marketing["impression"]

    # find attributed revenue
    ar_cols = [c for c in marketing.columns if "attribut" in c]
    if ar_cols:
        marketing["attributed_revenue"] = pd.to_numeric(marketing[ar_cols[0]], errors="coerce").fillna(0)
    else:
        marketing["attributed_revenue"] = 0.0

    # ensure numeric
    for c in ["impressions", "clicks", "spend", "attributed_revenue"]:
        marketing[c] = pd.to_numeric(marketing.get(c, 0), errors="coerce").fillna(0)

    # business cleanup and numeric casting
    b = b.rename(columns={c: c.strip().lower().replace(" ", "_") for c in b.columns})
    for col in ["orders", "new_orders", "new_customers", "total_revenue", "gross_profit", "cogs"]:
        if col in b.columns:
            b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0)
        else:
            b[col] = 0

    # parse dates
    marketing["date"] = pd.to_datetime(marketing["date"]).dt.date
    b["date"] = pd.to_datetime(b["date"]).dt.date

    # daily aggregates per channel and totals
    m_daily = marketing.groupby(["date", "channel"], as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })
    m_total = m_daily.groupby("date", as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    merged = pd.merge(b, m_total, on="date", how="left").fillna(0)

    # derived metrics (safe divides)
    merged["ctr"] = merged["clicks"] / merged["impressions"].replace(0, np.nan)
    merged["cpc"] = merged["spend"] / merged["clicks"].replace(0, np.nan)
    merged["cpm"] = (merged["spend"] / merged["impressions"].replace(0, np.nan)) * 1000
    merged["roas"] = merged["attributed_revenue"] / merged["spend"].replace(0, np.nan)
    merged["cac"] = merged["spend"] / merged["new_customers"].replace(0, np.nan)
    merged["aov"] = merged["total_revenue"] / merged["orders"].replace(0, np.nan)
    merged["profit_margin"] = merged["gross_profit"] / merged["total_revenue"].replace(0, np.nan)

    return marketing, m_daily, merged

try:
    marketing, marketing_daily, merged = load_and_prepare(FB_PATH, G_PATH, T_PATH, B_PATH)
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()

# ---------------------- Controls / Interactivity ----------------------
st.sidebar.header("Filters and Controls")
min_date, max_date = merged["date"].min(), merged["date"].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

all_channels = sorted(marketing["channel"].dropna().unique().tolist())
channels = st.sidebar.multiselect("Channels", options=all_channels, default=all_channels)

state_available = "state" in marketing.columns
states = sorted(marketing["state"].dropna().unique().tolist()) if state_available else []
selected_states = st.sidebar.multiselect("States", options=states, default=states if states else None)

campaign_available = "campaign" in marketing.columns
campaign_search = st.sidebar.text_input("Campaign search (substring)") if campaign_available else None

smoothing = st.sidebar.checkbox("Apply 7-day rolling smoothing to time series", value=False)
chart_type = st.sidebar.selectbox("Time series chart type", options=["line", "area", "stacked_area"])
roas_threshold = st.sidebar.slider("ROAS lower bound (flag if below)", 0.0, 5.0, 1.0, 0.1)
spend_sigma = st.sidebar.slider("Spend spike threshold (std multiples)", 0.0, 5.0, 2.0, 0.5)

# Apply filters
mask_m = (marketing["date"] >= date_range[0]) & (marketing["date"] <= date_range[1]) & (marketing["channel"].isin(channels))
if state_available and selected_states:
    mask_m &= marketing["state"].isin(selected_states)
if campaign_available and campaign_search:
    mask_m &= marketing["campaign"].str.contains(campaign_search, case=False, na=False)
marketing_filtered = marketing.loc[mask_m].copy()

mask_md = (marketing_daily["date"] >= date_range[0]) & (marketing_daily["date"] <= date_range[1]) & (marketing_daily["channel"].isin(channels))
marketing_daily_filtered = marketing_daily.loc[mask_md].copy()

mask_merged = (merged["date"] >= date_range[0]) & (merged["date"] <= date_range[1])
merged_filtered = merged.loc[mask_merged].copy()

# ---------------------- KPI row ----------------------
st.header("Key Performance Indicators")
st.write("High-level KPIs linking marketing activity to business performance for selected slice.")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Revenue", f"${merged_filtered['total_revenue'].sum():,.0f}")
col2.metric("Gross Profit", f"${merged_filtered['gross_profit'].sum():,.0f}")
col3.metric("Marketing Spend", f"${merged_filtered['spend'].sum():,.0f}")
roas_val = merged_filtered['attributed_revenue'].sum() / (merged_filtered['spend'].sum() or np.nan)
col4.metric("ROAS", f"{roas_val:.2f}" if not np.isnan(roas_val) else "N/A")
cac_val = merged_filtered['spend'].sum() / (merged_filtered['new_customers'].sum() or np.nan)
col5.metric("CAC", f"${cac_val:.2f}" if not np.isnan(cac_val) else "N/A")

# ---------------------- Time series ----------------------
st.header("Time Series: Spend vs Revenue vs Profit")
st.write("Use smoothing and chart type controls in the sidebar to adjust the view.")
ts = merged_filtered[["date", "spend", "total_revenue", "gross_profit"]].sort_values("date")
if smoothing:
    ts = ts.set_index("date").rolling(7, min_periods=1).mean().reset_index()

if chart_type == "line":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["spend"], name="Spend"))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["total_revenue"], name="Revenue"))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["gross_profit"], name="Gross Profit"))
elif chart_type == "area":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["spend"], name="Spend", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["total_revenue"], name="Revenue", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["gross_profit"], name="Gross Profit", fill="tozeroy"))
else:
    fig = px.area(ts, x="date", y=["spend", "total_revenue", "gross_profit"], labels={"value": "USD", "date": "Date"})

fig.update_layout(height=420, xaxis_title="Date", yaxis_title="USD")
st.plotly_chart(fig, use_container_width=True)

# ---------------------- Channel comparison ----------------------
st.header("Channel Comparison")
st.write("Compare channels on spend, attributed revenue, CTR and ROAS.")
channel_agg = marketing_daily_filtered.groupby("channel", as_index=False).agg({
    "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
})
channel_agg["roas"] = channel_agg["attributed_revenue"] / channel_agg["spend"].replace(0, np.nan)
channel_agg["ctr"] = channel_agg["clicks"] / channel_agg["impressions"].replace(0, np.nan)

c_left, c_right = st.columns([2, 1])
with c_left:
    st.plotly_chart(px.bar(channel_agg, x="channel", y=["spend", "attributed_revenue"], barmode="group", title="Spend vs Attributed Revenue"), use_container_width=True)
with c_right:
    st.plotly_chart(px.scatter(channel_agg, x="spend", y="roas", size="impressions", hover_name="channel", title="Spend vs ROAS"), use_container_width=True)

# ---------------------- Conversion funnel ----------------------
st.header("Conversion Funnel")
st.write("Impressions → Clicks → Orders for the selected slice.")
funnel_vals = [marketing_filtered["impressions"].sum(), marketing_filtered["clicks"].sum(), merged_filtered["orders"].sum()]
funnel_fig = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Orders"], x=funnel_vals))
st.plotly_chart(funnel_fig, use_container_width=True)

# ---------------------- ROAS by state heatmap (unique) ----------------------
st.header("ROAS by State (if available)")
if "state" in marketing.columns:
    st.write("Heatmap of ROAS by state and channel.")
    st_data = marketing_filtered.groupby(["state", "channel"], as_index=False).agg({"attributed_revenue": "sum", "spend": "sum"})
    st_data["roas"] = st_data["attributed_revenue"] / st_data["spend"].replace(0, np.nan)
    if not st_data.empty:
        pivot = st_data.pivot(index="channel", columns="state", values="roas").fillna(0)
        heat = px.imshow(pivot, labels={"x": "State", "y": "Channel", "color": "ROAS"}, aspect="auto")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("No data for selected filters.")
else:
    st.info("State column not present.")

# ---------------------- Anomaly alerts ----------------------
st.header("Anomaly Alerts")
st.write("Flags: spend spikes and days where ROAS < threshold.")
spend_mean = marketing_daily_filtered["spend"].mean()
spend_std = marketing_daily_filtered["spend"].std(ddof=0)
spike_threshold = spend_mean + spend_sigma * (spend_std or 0)
spend_spikes = marketing_daily_filtered[marketing_daily_filtered["spend"] > spike_threshold].sort_values(["date", "spend"], ascending=[True, False])
low_roas_days = merged_filtered[merged_filtered["roas"] < roas_threshold].sort_values("date")

if not spend_spikes.empty:
    st.subheader("Spend spikes (channel-level)")
    st.dataframe(spend_spikes[["date", "channel", "spend"]].reset_index(drop=True))
else:
    st.info("No spend spikes detected with current threshold.")

if not low_roas_days.empty:
    st.subheader("Low ROAS days")
    st.dataframe(low_roas_days[["date", "spend", "attributed_revenue", "roas"]].reset_index(drop=True))
else:
    st.info("No low ROAS days detected with current threshold.")

# ---------------------- Campaign table ----------------------
st.header("Campaign Leaderboard")
st.write("Campaign-level aggregation for the selected slice.")
if "campaign" in marketing.columns:
    camp = marketing_filtered.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
    })
    camp["roas"] = camp["attributed_revenue"] / camp["spend"].replace(0, np.nan)
    st.dataframe(camp.sort_values("spend", ascending=False).reset_index(drop=True))
else:
    st.info("Campaign column not found.")

# ---------------------- Export options ----------------------
st.sidebar.header("Export")
st.sidebar.markdown("Download processed merged dataset or a short PDF summary.")

st.sidebar.download_button("Download merged CSV", data=merged_filtered.to_csv(index=False), file_name="merged_data.csv", mime="text/csv")

def build_pdf_bytes(merged_df, camp_df, spend_spikes_df, low_roas_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=14, style="B")
    pdf.cell(0, 8, "Marketing Intelligence — Summary", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}", ln=True)
    pdf.cell(0, 6, f"Total Revenue: ${merged_df['total_revenue'].sum():,.0f}", ln=True)
    pdf.cell(0, 6, f"Marketing Spend: ${merged_df['spend'].sum():,.0f}", ln=True)
    roas_val = merged_df['attributed_revenue'].sum() / (merged_df['spend'].sum() or np.nan)
    pdf.cell(0, 6, f"ROAS: {roas_val:.2f}" if not np.isnan(roas_val) else "ROAS: N/A", ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", size=11, style="B")
    pdf.cell(0, 6, "Top campaigns (by spend)", ln=True)
    pdf.set_font("Helvetica", size=9)
    if camp_df is not None and not camp_df.empty:
        top = camp_df.sort_values("spend", ascending=False).head(8)
        for _, row in top.iterrows():
            name = str(row.get("campaign", ""))[:50]
            pdf.multi_cell(0, 5, f"- {name} | channel: {row.get('channel','')} | spend: ${row.get('spend',0):,.0f} | ROAS: {row.get('roas',np.nan):.2f}")
    else:
        pdf.cell(0, 5, "No campaign data available.", ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", size=11, style="B")
    pdf.cell(0, 6, "Anomalies Detected", ln=True)
    pdf.set_font("Helvetica", size=9)
    if not spend_spikes_df.empty:
        pdf.cell(0, 5, "Spend spikes:", ln=True)
        for _, r in spend_spikes_df.head(6).iterrows():
            pdf.multi_cell(0, 5, f"- {r['date']} | {r['channel']} | spend: ${r['spend']:,.0f}")
    else:
        pdf.cell(0, 5, "No spend spikes detected.", ln=True)
    if not low_roas_df.empty:
        pdf.ln(2)
        pdf.cell(0, 5, "Low ROAS days:", ln=True)
        for _, r in low_roas_df.head(6).iterrows():
            pdf.multi_cell(0, 5, f"- {r['date']} | spend: ${r['spend']:,.0f} | roas: {r['roas']:.2f}")
    else:
        pdf.cell(0, 5, "No low-ROAS days detected.", ln=True)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_bytes).getvalue()

camp_for_pdf = None
if "campaign" in marketing.columns:
    camp_for_pdf = marketing_filtered.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum"
    })
    camp_for_pdf["roas"] = camp_for_pdf["attributed_revenue"] / camp_for_pdf["spend"].replace(0, np.nan)

if st.sidebar.button("Generate PDF Summary"):
    pdf_bytes = build_pdf_bytes(merged_filtered, camp_for_pdf, spend_spikes, low_roas_days)
    st.sidebar.download_button("Download PDF", data=pdf_bytes, file_name="marketing_summary.pdf", mime="application/pdf")
    st.sidebar.success("PDF generated. Click Download PDF to save.")
# ---------------------- End of app ----------------------