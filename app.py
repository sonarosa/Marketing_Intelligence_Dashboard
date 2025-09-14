"""
Streamlit BI Dashboard for Marketing Intelligence Assessment

Usage:
    streamlit run app.py

Files required (place in same folder or upload via UI):
    - Facebook.csv
    - Google.csv
    - TikTok.csv
    - Business.csv

Features:
    - Cleans and merges all four datasets
    - Derives KPIs: CTR, CPC, CPM, ROAS, CAC, AOV, Profit Margin
    - Provides interactive filters (date, channel, state, campaign)
    - Layout: KPI cards, time-series trends, channel comparisons, funnel, campaign table
    - Export merged dataset as CSV

Dependencies:
    pip install streamlit pandas numpy plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard")

# ---------------------- File Upload ----------------------
st.title("Marketing Intelligence Dashboard")

col1, col2 = st.columns(2)
with col1:
    fb_file = st.file_uploader("Upload Facebook.csv", type="csv")
    g_file = st.file_uploader("Upload Google.csv", type="csv")
    t_file = st.file_uploader("Upload TikTok.csv", type="csv")
with col2:
    b_file = st.file_uploader("Upload Business.csv", type="csv")

# ---------------------- Data Load ----------------------
@st.cache_data

def load_and_prepare(fb_file, g_file, t_file, b_file):
    def norm(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    # Load
    fb = norm(pd.read_csv(fb_file)); fb["channel"] = "Facebook"
    g = norm(pd.read_csv(g_file)); g["channel"] = "Google"
    t = norm(pd.read_csv(t_file)); t["channel"] = "TikTok"
    b = norm(pd.read_csv(b_file))

    # Marketing unify
    marketing = pd.concat([fb,g,t], ignore_index=True)
    if "impression" in marketing.columns and "impressions" not in marketing.columns:
        marketing["impressions"] = marketing["impression"]
    if "attributed_revenue" not in marketing.columns:
        alt = [c for c in marketing.columns if "attributed" in c]
        if alt:
            marketing["attributed_revenue"] = marketing[alt[0]]
        else:
            marketing["attributed_revenue"] = 0.0

    # Convert to numeric
    for c in ["impressions","clicks","spend","attributed_revenue"]:
        marketing[c] = pd.to_numeric(marketing[c], errors="coerce").fillna(0)

    # Business rename
    b = b.rename(columns={"#_of_orders":"orders","#_of_new_orders":"new_orders"})
    for c in ["orders","new_orders","new_customers","total_revenue","gross_profit","cogs"]:
        if c in b.columns:
            b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0)
        else:
            b[c] = 0

    # Dates
    for df in [marketing,b]:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # Aggregations
    m_daily = marketing.groupby(["date","channel"]).agg({
        "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
    }).reset_index()
    m_total = m_daily.groupby("date").agg({
        "impressions":"sum","clicks":"sum","spend":"sum","attributed_revenue":"sum"
    }).reset_index()

    merged = pd.merge(b,m_total,on="date",how="left").fillna(0)

    # Derived metrics
    merged["ctr"] = merged["clicks"] / merged["impressions"].replace(0,np.nan)
    merged["cpc"] = merged["spend"] / merged["clicks"].replace(0,np.nan)
    merged["cpm"] = (merged["spend"] / merged["impressions"].replace(0,np.nan))*1000
    merged["roas"] = merged["attributed_revenue"] / merged["spend"].replace(0,np.nan)
    merged["cac"] = merged["spend"] / merged["new_customers"].replace(0,np.nan)
    merged["aov"] = merged["total_revenue"] / merged["orders"].replace(0,np.nan)
    merged["profit_margin"] = merged["gross_profit"] / merged["total_revenue"].replace(0,np.nan)

    return marketing, m_daily, merged

if all([fb_file,g_file,t_file,b_file]):
    marketing, marketing_daily, merged = load_and_prepare(fb_file,g_file,t_file,b_file)
else:
    st.stop()

# ---------------------- Filters ----------------------
st.sidebar.header("Filters")
min_d, max_d = merged["date"].min(), merged["date"].max()
date_range = st.sidebar.date_input("Date range", [min_d,max_d], min_value=min_d, max_value=max_d)
channels = st.sidebar.multiselect("Channels", marketing_daily["channel"].unique(), default=list(marketing_daily["channel"].unique()))

merged_f = merged[(merged["date"]>=date_range[0])&(merged["date"]<=date_range[1])]
marketing_daily_f = marketing_daily[(marketing_daily["date"]>=date_range[0])&(marketing_daily["date"]<=date_range[1])&(marketing_daily["channel"].isin(channels))]

# ---------------------- KPIs ----------------------
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Revenue", f"${merged_f['total_revenue'].sum():,.0f}")
k2.metric("Gross Profit", f"${merged_f['gross_profit'].sum():,.0f}")
k3.metric("Marketing Spend", f"${merged_f['spend'].sum():,.0f}")
k4.metric("ROAS", f"{(merged_f['attributed_revenue'].sum()/merged_f['spend'].sum()):.2f}")
k5.metric("CAC", f"${(merged_f['spend'].sum()/merged_f['new_customers'].sum()):.2f}")

# ---------------------- Time Series ----------------------
st.subheader("Spend vs Revenue vs Profit over Time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["spend"], name="Spend"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["total_revenue"], name="Revenue"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["gross_profit"], name="Gross Profit"))
st.plotly_chart(fig,use_container_width=True)

# ---------------------- Channel Comparison ----------------------
st.subheader("Channel Comparison")
channel_agg = marketing_daily_f.groupby("channel").agg({"spend":"sum","attributed_revenue":"sum","impressions":"sum","clicks":"sum"}).reset_index()
channel_agg["roas"] = channel_agg["attributed_revenue"]/channel_agg["spend"].replace(0,np.nan)
channel_agg["ctr"] = channel_agg["clicks"]/channel_agg["impressions"].replace(0,np.nan)

c1,c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(channel_agg,x="channel",y=["spend","attributed_revenue"],barmode="group",title="Spend vs Revenue by Channel"),use_container_width=True)
with c2:
    st.plotly_chart(px.scatter(channel_agg,x="spend",y="roas",size="impressions",hover_name="channel",title="Spend vs ROAS"),use_container_width=True)

# ---------------------- Funnel ----------------------
st.subheader("Funnel: Impressions → Clicks → Orders")
funnel_vals = [marketing_daily_f["impressions"].sum(), marketing_daily_f["clicks"].sum(), merged_f["orders"].sum()]
fig2 = go.Figure(go.Funnel(y=["Impressions","Clicks","Orders"], x=funnel_vals))
st.plotly_chart(fig2,use_container_width=True)

# ---------------------- Campaign Table ----------------------
st.subheader("Campaign Performance Table")
if "campaign" in marketing.columns:
    camp_agg = marketing.groupby(["campaign","channel"]).agg({"spend":"sum","attributed_revenue":"sum","impressions":"sum","clicks":"sum"}).reset_index()
    camp_agg["roas"] = camp_agg["attributed_revenue"]/camp_agg["spend"].replace(0,np.nan)
    st.dataframe(camp_agg.sort_values("spend",ascending=False))

# ---------------------- Export ----------------------
st.sidebar.download_button("Download merged CSV", data=merged.to_csv(index=False), file_name="merged_data.csv")
