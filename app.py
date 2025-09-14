"""
Streamlit BI Dashboard for Marketing Intelligence Assessment

Usage:
    streamlit run streamlit_bi_dashboard.py

Features:
    - Cleans and merges all four datasets
    - Derives KPIs: CTR, CPC, CPM, ROAS, CAC, AOV, Profit Margin
    - Provides interactive filters (date, channel)
    - Layout: KPI cards, time-series trends, channel comparisons, funnel, campaign table
    - Export merged dataset as CSV and PDF

Dependencies:
    pip install streamlit pandas numpy plotly fpdf
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard")

# ===================== Data Load & Prep =====================
@st.cache_data
def load_and_prepare():
    def norm(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    base_path = os.path.join("datasets")  # use relative datasets/ folder
    fb = norm(pd.read_csv(os.path.join(base_path, "Facebook.csv"))); fb["channel"] = "Facebook"
    g = norm(pd.read_csv(os.path.join(base_path, "Google.csv"))); g["channel"] = "Google"
    t = norm(pd.read_csv(os.path.join(base_path, "TikTok.csv"))); t["channel"] = "TikTok"
    b = norm(pd.read_csv(os.path.join(base_path, "business.csv")))

    # unify marketing
    marketing = pd.concat([fb,g,t], ignore_index=True)
    if "impression" in marketing.columns and "impressions" not in marketing.columns:
        marketing["impressions"] = marketing["impression"]
    if "attributed_revenue" not in marketing.columns:
        alt = [c for c in marketing.columns if "attributed" in c]
        marketing["attributed_revenue"] = marketing[alt[0]] if alt else 0.0

    for c in ["impressions","clicks","spend","attributed_revenue"]:
        marketing[c] = pd.to_numeric(marketing[c], errors="coerce").fillna(0)

    # business cleanup
    b = b.rename(columns={"#_of_orders":"orders","#_of_new_orders":"new_orders"})
    for c in ["orders","new_orders","new_customers","total_revenue","gross_profit","cogs"]:
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0) if c in b.columns else 0

    for df in [marketing,b]:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # aggregate
    m_daily = marketing.groupby(["date","channel"]).agg({"impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"}).reset_index()
    m_total = m_daily.groupby("date").agg({"impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"}).reset_index()
    merged = pd.merge(b,m_total,on="date",how="left").fillna(0)

    # derived metrics
    merged["ctr"] = merged["clicks"] / merged["impressions"].replace(0,np.nan)
    merged["cpc"] = merged["spend"] / merged["clicks"].replace(0,np.nan)
    merged["cpm"] = (merged["spend"] / merged["impressions"].replace(0,np.nan))*1000
    merged["roas"] = merged["attributed_revenue"] / merged["spend"].replace(0,np.nan)
    merged["cac"] = merged["spend"] / merged["new_customers"].replace(0,np.nan)
    merged["aov"] = merged["total_revenue"] / merged["orders"].replace(0,np.nan)
    merged["profit_margin"] = merged["gross_profit"] / merged["total_revenue"].replace(0,np.nan)

    return marketing, m_daily, merged

marketing, marketing_daily, merged = load_and_prepare()

# ===================== Heading =====================
st.title("📊 Marketing Intelligence Dashboard")

# ===================== Filters =====================
st.sidebar.header("Interactive Filters")
st.sidebar.markdown("Use these to slice data by period or channel.")
min_d, max_d = merged["date"].min(), merged["date"].max()
date_range = st.sidebar.date_input("Date range", [min_d,max_d], min_value=min_d, max_value=max_d)
channels = st.sidebar.multiselect("Select Channels", marketing_daily["channel"].unique(), default=list(marketing_daily["channel"].unique()))

merged_f = merged[(merged["date"]>=date_range[0])&(merged["date"]<=date_range[1])]
marketing_daily_f = marketing_daily[(marketing_daily["date"]>=date_range[0])&(marketing_daily["date"]<=date_range[1])&(marketing_daily["channel"].isin(channels))]

# ===================== KPI Cards =====================
st.subheader("Business Pulse: Key KPIs")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Revenue", f"${merged_f['total_revenue'].sum():,.0f}")
k2.metric("Gross Profit", f"${merged_f['gross_profit'].sum():,.0f}")
k3.metric("Marketing Spend", f"${merged_f['spend'].sum():,.0f}")
k4.metric("ROAS", f"{(merged_f['attributed_revenue'].sum()/merged_f['spend'].sum()):.2f}")
k5.metric("CAC", f"${(merged_f['spend'].sum()/merged_f['new_customers'].sum()):.2f}")

# ===================== Trends Over Time =====================
st.subheader("Revenue & Spend Trends")

fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["spend"], name="Spend"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["total_revenue"], name="Revenue"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["gross_profit"], name="Gross Profit"))
st.plotly_chart(fig,use_container_width=True)

# ===================== Channel Deep Dive =====================
st.subheader("Channel Efficiency & Scale")

channel_agg = marketing_daily_f.groupby("channel").agg({"spend":"sum","attributed_revenue":"sum","impressions":"sum","clicks":"sum"}).reset_index()
channel_agg["roas"] = channel_agg["attributed_revenue"]/channel_agg["spend"].replace(0,np.nan)
channel_agg["ctr"] = channel_agg["clicks"]/channel_agg["impressions"].replace(0,np.nan)

c1,c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(channel_agg,x="channel",y=["spend","attributed_revenue"],barmode="group",title="Spend vs Revenue by Channel"),use_container_width=True)
with c2:
    st.plotly_chart(px.scatter(channel_agg,x="spend",y="roas",size="impressions",hover_name="channel",title="Spend vs ROAS"),use_container_width=True)

# ===================== Conversion Funnel =====================
st.subheader("Conversion Journey")

funnel_vals = [marketing_daily_f["impressions"].sum(), marketing_daily_f["clicks"].sum(), merged_f["orders"].sum()]
fig2 = go.Figure(go.Funnel(y=["Impressions","Clicks","Orders"], x=funnel_vals))
st.plotly_chart(fig2,use_container_width=True)

# ===================== Campaign Leaderboard =====================
st.subheader("Campaign Performance Table")

if "campaign" in marketing.columns:
    camp_agg = marketing.groupby(["campaign","channel"]).agg({"spend":"sum","attributed_revenue":"sum","impressions":"sum","clicks":"sum"}).reset_index()
    camp_agg["roas"] = camp_agg["attributed_revenue"]/camp_agg["spend"].replace(0,np.nan)
    st.dataframe(camp_agg.sort_values("spend",ascending=False))

# ===================== Export =====================
st.sidebar.header("💾 Export Data")

# Download CSV
st.sidebar.download_button("Download merged CSV", data=merged.to_csv(index=False), file_name="merged_data.csv")

# Generate simple PDF report
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Marketing Intelligence Dashboard Report", ln=True, align="C")
    pdf.ln(10)
    # Basic KPIs
    pdf.cell(200, 10, txt=f"Total Revenue: ${df['total_revenue'].sum():,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Gross Profit: ${df['gross_profit'].sum():,.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Marketing Spend: ${df['spend'].sum():,.0f}", ln=True)
    roas = df['attributed_revenue'].sum()/(df['spend'].sum() or 1)
    pdf.cell(200, 10, txt=f"ROAS: {roas:.2f}", ln=True)
    cac = df['spend'].sum()/(df['new_customers'].sum() or 1)
    pdf.cell(200, 10, txt=f"CAC: ${cac:,.2f}", ln=True)
    return pdf.output(dest="S").encode("latin-1")

pdf_bytes = create_pdf(merged_f)
st.sidebar.download_button("Download PDF Report", data=pdf_bytes, file_name="dashboard_report.pdf", mime="application/pdf")
