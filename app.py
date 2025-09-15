"""
Streamlit BI Dashboard for Marketing Intelligence Assessment
- Default loads CSVs from datasets/
- Adds descriptions for each component
- Unique features: ROAS-by-state heatmap, Anomaly Alerts
- Export summary PDF (text + top campaigns + anomalies)
Usage:
    streamlit run streamlit_bi_dashboard.py
"""
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard")

# ------------------- Header -------------------
st.title("📊 Marketing Intelligence Dashboard")
st.markdown(
    "Interactive dashboard linking campaign-level marketing (Facebook, Google, TikTok) "
    "to daily business outcomes. Use filters at left. Download CSV or a short PDF summary."
)

# ------------------- Dataset paths & checks -------------------
DATASET_DIR = "datasets"
paths = {
    "Facebook.csv": os.path.join(DATASET_DIR, "Facebook.csv"),
    "Google.csv": os.path.join(DATASET_DIR, "Google.csv"),
    "TikTok.csv": os.path.join(DATASET_DIR, "TikTok.csv"),
}
business_lower = os.path.join(DATASET_DIR, "business.csv")
business_upper = os.path.join(DATASET_DIR, "Business.csv")
paths["business.csv"] = business_lower if os.path.exists(business_lower) else business_upper

missing = [name for name, p in paths.items() if not os.path.exists(p)]
if missing:
    st.error("❌ Missing CSV files. Place these in the `datasets/` folder:")
    for m in missing:
        st.write(f"- `{m}`")
    st.stop()

FB_PATH = paths["Facebook.csv"]
G_PATH = paths["Google.csv"]
T_PATH = paths["TikTok.csv"]
B_PATH = paths["business.csv"]

# ------------------- Data load & prep -------------------
@st.cache_data
def load_and_prepare(fb_path, g_path, t_path, b_path):
    def norm(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    fb = norm(pd.read_csv(fb_path)); fb["channel"] = "Facebook"
    g = norm(pd.read_csv(g_path)); g["channel"] = "Google"
    t = norm(pd.read_csv(t_path)); t["channel"] = "TikTok"
    b = norm(pd.read_csv(b_path))

    marketing = pd.concat([fb, g, t], ignore_index=True, sort=False)

    # normalize impressions column name
    if "impression" in marketing.columns and "impressions" not in marketing.columns:
        marketing["impressions"] = marketing["impression"]

    # find attributed revenue column
    ar_cols = [c for c in marketing.columns if "attribut" in c]
    if ar_cols:
        marketing["attributed_revenue"] = pd.to_numeric(marketing[ar_cols[0]], errors="coerce").fillna(0)
    else:
        marketing["attributed_revenue"] = 0.0

    # ensure numeric fields exist
    for c in ["impressions", "clicks", "spend", "attributed_revenue"]:
        marketing[c] = pd.to_numeric(marketing.get(c, 0), errors="coerce").fillna(0)

    # business cleanup: normalize column names and ensure columns exist as Series
    b = b.rename(columns={c: c.strip().lower().replace(" ", "_") for c in b.columns})
    for col in ["orders", "new_orders", "new_customers", "total_revenue", "gross_profit", "cogs"]:
        if col in b.columns:
            b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0)
        else:
            b[col] = 0

    # dates
    marketing["date"] = pd.to_datetime(marketing["date"]).dt.date
    b["date"] = pd.to_datetime(b["date"]).dt.date

    # aggregate marketing daily by channel
    m_daily = marketing.groupby(["date", "channel"], as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # totals per date across channels
    m_total = m_daily.groupby("date", as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # merge with business on date
    merged = pd.merge(b, m_total, on="date", how="left").fillna(0)

    # derived metrics (guard divisions)
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
    st.success("✅ Datasets loaded.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ------------------- Filters -------------------
st.sidebar.header("Filters")
st.sidebar.markdown("Slice the data to focus the analysis.")
min_d, max_d = merged["date"].min(), merged["date"].max()
date_range = st.sidebar.date_input("Date range", [min_d, max_d], min_value=min_d, max_value=max_d)
channels = st.sidebar.multiselect("Channels", options=marketing["channel"].unique(), default=list(marketing["channel"].unique()))

merged_f = merged[(merged["date"] >= date_range[0]) & (merged["date"] <= date_range[1])]
marketing_daily_f = marketing_daily[
    (marketing_daily["date"] >= date_range[0]) & (marketing_daily["date"] <= date_range[1]) &
    (marketing_daily["channel"].isin(channels))
]

# ------------------- KPI Cards -------------------
st.subheader("📊 Business Pulse — Key KPIs")
st.caption("High level metrics to assess revenue, profitability, and marketing efficiency.")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue", f"${merged_f['total_revenue'].sum():,.0f}")
c2.metric("Gross Profit", f"${merged_f['gross_profit'].sum():,.0f}")
c3.metric("Marketing Spend", f"${merged_f['spend'].sum():,.0f}")
roas_val = merged_f['attributed_revenue'].sum() / (merged_f['spend'].sum() or np.nan)
c4.metric("ROAS", f"{roas_val:.2f}" if not np.isnan(roas_val) else "N/A")
cac_val = merged_f['spend'].sum() / (merged_f['new_customers'].sum() or np.nan)
c5.metric("CAC", f"${cac_val:.2f}" if not np.isnan(cac_val) else "N/A")

# ------------------- Trends Over Time -------------------
st.subheader("⏱ Revenue & Spend Trajectory")
st.caption("Daily trend lines showing how spend, revenue, and gross profit move together.")
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["spend"], name="Spend"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["total_revenue"], name="Revenue"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["gross_profit"], name="Gross Profit"))
fig.update_layout(height=420, xaxis_title="Date", yaxis_title="USD")
st.plotly_chart(fig, use_container_width=True)

# ------------------- Channel Comparison -------------------
st.subheader("📈 Channel Efficiency & Scale")
st.caption("Compare spend, attributed revenue, CTR and ROAS across channels.")
channel_agg = marketing_daily_f.groupby("channel", as_index=False).agg({
    "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
})
channel_agg["roas"] = channel_agg["attributed_revenue"] / channel_agg["spend"].replace(0, np.nan)
channel_agg["ctr"] = channel_agg["clicks"] / channel_agg["impressions"].replace(0, np.nan)
col_left, col_right = st.columns([2,1])
with col_left:
    st.plotly_chart(px.bar(channel_agg, x="channel", y=["spend", "attributed_revenue"], barmode="group", title="Spend vs Attributed Revenue (by channel)"), use_container_width=True)
with col_right:
    st.plotly_chart(px.scatter(channel_agg, x="spend", y="roas", size="impressions", hover_name="channel", title="Spend vs ROAS"), use_container_width=True)

# ------------------- Unique: ROAS by State Heatmap -------------------
st.subheader("🗺 ROAS by State (unique)")
if "state" in marketing.columns:
    st.caption("Heatmap showing ROAS by state across all channels. Useful to prioritize geographies.")
    state_agg = marketing.groupby(["state", "channel"], as_index=False).agg({"attributed_revenue": "sum", "spend": "sum"})
    state_agg["roas"] = state_agg["attributed_revenue"] / state_agg["spend"].replace(0, np.nan)
    # pivot for heatmap (channel x state)
    pivot = state_agg.pivot(index="channel", columns="state", values="roas").fillna(0)
    fig_heat = px.imshow(pivot, labels=dict(x="State", y="Channel", color="ROAS"), aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No `state` column found in marketing data. Add 'state' to marketing CSVs to enable heatmap.")

# ------------------- Unique: Anomaly Alerts -------------------
st.subheader("🚨 Anomaly Alerts (simple)")
st.caption("Flags days where spend is a large spike or ROAS is below threshold.")
# compute anomalies
spend_mean = marketing_daily_f["spend"].mean()
spend_std = marketing_daily_f["spend"].std(ddof=0)
spike_threshold = spend_mean + 2 * (spend_std or 0)
spend_spikes = marketing_daily_f[marketing_daily_f["spend"] > spike_threshold].sort_values("date")
low_roas_days = merged_f[merged_f["roas"] < 1.0].sort_values("date")  # ROAS <1 means negative ROI

st.write("**Detected Anomalies:**")
if not spend_spikes.empty or not low_roas_days.empty:
    if not spend_spikes.empty:
        st.markdown("**Spend spikes (per channel):**")
        st.dataframe(spend_spikes[["date", "channel", "spend"]].reset_index(drop=True))
    if not low_roas_days.empty:
        st.markdown("**Days with ROAS < 1.0:**")
        st.dataframe(low_roas_days[["date", "spend", "attributed_revenue", "roas"]].reset_index(drop=True))
else:
    st.success("No anomalies detected by current simple rules.")

# ------------------- Conversion Funnel -------------------
st.subheader("🛒 Conversion Journey")
st.caption("Funnel conversion volumes: impressions → clicks → orders.")
funnel_vals = [marketing_daily_f["impressions"].sum(), marketing_daily_f["clicks"].sum(), merged_f["orders"].sum()]
fig2 = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Orders"], x=funnel_vals))
st.plotly_chart(fig2, use_container_width=True)

# ------------------- Campaign Leaderboard -------------------
st.subheader("🏆 Campaign Performance Table")
st.caption("Sort and inspect campaigns by spend and ROAS to find winners and losers.")
if "campaign" in marketing.columns:
    camp_agg = marketing.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
    })
    camp_agg["roas"] = camp_agg["attributed_revenue"] / camp_agg["spend"].replace(0, np.nan)
    st.dataframe(camp_agg.sort_values("spend", ascending=False).reset_index(drop=True))
else:
    st.info("No `campaign` column found in marketing data.")

# ------------------- Export CSV -------------------
st.sidebar.header("Export")
st.sidebar.markdown("Download the processed merged dataset as CSV.")
st.sidebar.download_button("Download merged CSV", data=merged.to_csv(index=False), file_name="merged_data.csv", mime="text/csv")

# ------------------- Export PDF Summary -------------------
st.sidebar.markdown("Download a short PDF summary (KPIs, top campaigns, anomalies).")

def make_pdf_bytes(merged_df, camp_df, spend_spikes_df, low_roas_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=14, style="B")
    pdf.cell(0, 8, "Marketing Intelligence — Summary", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    # KPIs
    pdf.cell(0, 6, f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}", ln=True)
    pdf.cell(0, 6, f"Total Revenue: ${merged_df['total_revenue'].sum():,.0f}", ln=True)
    pdf.cell(0, 6, f"Gross Profit: ${merged_df['gross_profit'].sum():,.0f}", ln=True)
    pdf.cell(0, 6, f"Marketing Spend: ${merged_df['spend'].sum():,.0f}", ln=True)
    roas_val = merged_df['attributed_revenue'].sum() / (merged_df['spend'].sum() or np.nan)
    pdf.cell(0, 6, f"ROAS: {roas_val:.2f}" if not np.isnan(roas_val) else "ROAS: N/A", ln=True)
    cac_val = merged_df['spend'].sum() / (merged_df['new_customers'].sum() or np.nan)
    pdf.cell(0, 6, f"CAC: ${cac_val:.2f}" if not np.isnan(cac_val) else "CAC: N/A", ln=True)
    pdf.ln(6)
    # Top campaigns
    pdf.set_font("Helvetica", size=11, style="B")
    pdf.cell(0, 6, "Top Campaigns (by spend)", ln=True)
    pdf.set_font("Helvetica", size=9)
    if camp_df is not None and not camp_df.empty:
        top = camp_df.sort_values("spend", ascending=False).head(8)
        for _, row in top.iterrows():
            name = str(row.get("campaign", ""))[:50]
            pdf.cell(0, 5, f"- {name} | channel: {row.get('channel','')} | spend: ${row.get('spend',0):,.0f} | ROAS: {row.get('roas',np.nan):.2f}", ln=True)
    else:
        pdf.cell(0, 5, "No campaign data available.", ln=True)
    pdf.ln(6)
    # Anomalies
    pdf.set_font("Helvetica", size=11, style="B")
    pdf.cell(0, 6, "Anomalies Detected", ln=True)
    pdf.set_font("Helvetica", size=9)
    if not spend_spikes_df.empty:
        pdf.cell(0, 5, "Spend spikes:", ln=True)
        for _, r in spend_spikes_df.head(6).iterrows():
            pdf.cell(0, 5, f"- {r['date']} | {r['channel']} | spend: ${r['spend']:,.0f}", ln=True)
    else:
        pdf.cell(0, 5, "No spend spikes detected.", ln=True)
    if not low_roas_df.empty:
        pdf.ln(2)
        pdf.cell(0, 5, "Low ROAS days (ROAS < 1):", ln=True)
        for _, r in low_roas_df.head(6).iterrows():
            pdf.cell(0, 5, f"- {r['date']} | spend: ${r['spend']:,.0f} | attributed_revenue: ${r['attributed_revenue']:,.0f} | roas: {r['roas']:.2f}", ln=True)
    else:
        pdf.cell(0, 5, "No low-ROAS days detected.", ln=True)
    # finalize
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()

# prepare data for PDF
camp_for_pdf = None
if "campaign" in marketing.columns:
    camp_for_pdf = marketing.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum"
    })
    camp_for_pdf["roas"] = camp_for_pdf["attributed_revenue"] / camp_for_pdf["spend"].replace(0, np.nan)

# create bytes on click
if st.sidebar.button("Generate PDF Summary"):
    pdf_bytes = make_pdf_bytes(merged_f, camp_for_pdf, spend_spikes, low_roas_days)
    st.sidebar.download_button("Download PDF", data=pdf_bytes, file_name="marketing_summary.pdf", mime="application/pdf")
    st.sidebar.success("PDF generated. Click the button to download.")

# End of app
