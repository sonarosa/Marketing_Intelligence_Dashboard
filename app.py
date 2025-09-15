# streamlit_bi_dashboard.py
"""
Marketing Intelligence Report — Streamlit App

Place CSVs in `datasets/`:
  - datasets/Facebook.csv
  - datasets/Google.csv
  - datasets/TikTok.csv
  - datasets/business.csv

Run:
  pip install -r requirements.txt
  streamlit run streamlit_bi_dashboard.py
"""
import os
import io
import textwrap
import re
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------- Config -----------------------
st.set_page_config(layout="wide", page_title="Marketing Intelligence Report")
DATA_DIR = "datasets"

# ----------------------- Header -----------------------
st.title("Marketing Intelligence Report")
st.markdown(
    "This dashboard links campaign-level marketing (Facebook, Google, TikTok) with daily business outcomes."
)

# ----------------------- File checks -----------------------
expected = {
    "Facebook.csv": os.path.join(DATA_DIR, "Facebook.csv"),
    "Google.csv": os.path.join(DATA_DIR, "Google.csv"),
    "TikTok.csv": os.path.join(DATA_DIR, "TikTok.csv"),
}
business_lower = os.path.join(DATA_DIR, "business.csv")
business_upper = os.path.join(DATA_DIR, "Business.csv")
expected["business.csv"] = business_lower if os.path.exists(business_lower) else business_upper

missing = [n for n, p in expected.items() if not os.path.exists(p)]
if missing:
    st.error("Missing CSV files in `datasets/`. Add these files and reload:")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

fb_path = expected["Facebook.csv"]
g_path = expected["Google.csv"]
t_path = expected["TikTok.csv"]
biz_path = expected["business.csv"]

# ----------------------- Data load & prepare -----------------------
@st.cache_data
def load_and_prepare(fb_p, g_p, t_p, biz_p):
    def norm_cols(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    fb = norm_cols(pd.read_csv(fb_p)); fb["channel"] = "Facebook"
    g = norm_cols(pd.read_csv(g_p)); g["channel"] = "Google"
    t = norm_cols(pd.read_csv(t_p)); t["channel"] = "TikTok"
    b = norm_cols(pd.read_csv(biz_p))

    marketing = pd.concat([fb, g, t], ignore_index=True, sort=False)

    # unify impressions column
    if "impression" in marketing.columns and "impressions" not in marketing.columns:
        marketing["impressions"] = marketing["impression"]

    # attributed revenue detection
    ar_cols = [c for c in marketing.columns if "attribut" in c]
    if ar_cols:
        marketing["attributed_revenue"] = pd.to_numeric(marketing[ar_cols[0]], errors="coerce").fillna(0)
    else:
        marketing["attributed_revenue"] = 0.0

    for c in ["impressions", "clicks", "spend", "attributed_revenue"]:
        marketing[c] = pd.to_numeric(marketing.get(c, 0), errors="coerce").fillna(0)

    # normalize business columns and types
    b = b.rename(columns={c: c.strip().lower().replace(" ", "_") for c in b.columns})
    for col in ["orders", "new_orders", "new_customers", "total_revenue", "gross_profit", "cogs"]:
        if col in b.columns:
            b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0)
        else:
            b[col] = 0

    # parse dates
    marketing["date"] = pd.to_datetime(marketing["date"]).dt.date
    b["date"] = pd.to_datetime(b["date"]).dt.date

    # daily per-channel aggregates
    marketing_daily = marketing.groupby(["date", "channel"], as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # daily totals across channels
    marketing_totals = marketing_daily.groupby("date", as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    merged = pd.merge(b, marketing_totals, on="date", how="left").fillna(0)

    # derived metrics
    merged["ctr"] = merged["clicks"] / merged["impressions"].replace(0, np.nan)
    merged["cpc"] = merged["spend"] / merged["clicks"].replace(0, np.nan)
    merged["cpm"] = (merged["spend"] / merged["impressions"].replace(0, np.nan)) * 1000
    merged["roas"] = merged["attributed_revenue"] / merged["spend"].replace(0, np.nan)
    merged["cac"] = merged["spend"] / merged["new_customers"].replace(0, np.nan)
    merged["aov"] = merged["total_revenue"] / merged["orders"].replace(0, np.nan)
    merged["profit_margin"] = merged["gross_profit"] / merged["total_revenue"].replace(0, np.nan)

    return marketing, marketing_daily, merged

marketing_df, marketing_daily_df, merged_df = load_and_prepare(fb_path, g_path, t_path, biz_path)

# ----------------------- Sidebar controls -----------------------
st.sidebar.header("Filters and options")
min_date, max_date = merged_df["date"].min(), merged_df["date"].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

channels_all = sorted(marketing_df["channel"].dropna().unique().tolist())
channels_sel = st.sidebar.multiselect("Channels", options=channels_all, default=channels_all)

states_exist = "state" in marketing_df.columns
states_all = sorted(marketing_df["state"].dropna().unique().tolist()) if states_exist else []
states_sel = st.sidebar.multiselect("States", options=states_all, default=states_all if states_all else None)

campaign_exist = "campaign" in marketing_df.columns
campaign_query = st.sidebar.text_input("Campaign substring (optional)") if campaign_exist else None

smooth_7 = st.sidebar.checkbox("7-day smoothing", value=False)
ts_style = st.sidebar.selectbox("Time series type", options=["line", "area", "stacked"])
roas_flag = st.sidebar.slider("Flag ROAS <=", 0.0, 5.0, 1.0, 0.1)
spend_sigma = st.sidebar.slider("Spend spike sigma multiplier", 0.0, 5.0, 2.0, 0.5)

# ----------------------- Apply filters -----------------------
mask_mk = (
    (marketing_df["date"] >= date_range[0]) &
    (marketing_df["date"] <= date_range[1]) &
    (marketing_df["channel"].isin(channels_sel))
)
if states_exist and states_sel:
    mask_mk &= marketing_df["state"].isin(states_sel)
if campaign_exist and campaign_query:
    mask_mk &= marketing_df["campaign"].str.contains(campaign_query, case=False, na=False)
marketing_filtered = marketing_df.loc[mask_mk].copy()

mask_mk_daily = (
    (marketing_daily_df["date"] >= date_range[0]) &
    (marketing_daily_df["date"] <= date_range[1]) &
    (marketing_daily_df["channel"].isin(channels_sel))
)
marketing_daily_filtered = marketing_daily_df.loc[mask_mk_daily].copy()

mask_merged = (merged_df["date"] >= date_range[0]) & (merged_df["date"] <= date_range[1])
merged_filtered = merged_df.loc[mask_merged].copy()

# ----------------------- KPIs -----------------------
st.header("Key Performance Indicators")
k_a, k_b, k_c, k_d, k_e = st.columns(5)
k_a.metric("Total Revenue", f"${merged_filtered['total_revenue'].sum():,.0f}")
k_b.metric("Gross Profit", f"${merged_filtered['gross_profit'].sum():,.0f}")
k_c.metric("Marketing Spend", f"${merged_filtered['spend'].sum():,.0f}")
roas_val = merged_filtered['attributed_revenue'].sum() / (merged_filtered['spend'].sum() or np.nan)
k_d.metric("ROAS", f"{roas_val:.2f}" if not np.isnan(roas_val) else "N/A")
cac_val = merged_filtered['spend'].sum() / (merged_filtered['new_customers'].sum() or np.nan)
k_e.metric("CAC", f"${cac_val:.2f}" if not np.isnan(cac_val) else "N/A")

# ----------------------- Time series -----------------------
st.header("Time Series: Spend vs Revenue vs Profit")
ts_df = merged_filtered[["date", "spend", "total_revenue", "gross_profit"]].sort_values("date")
if smooth_7:
    ts_df = ts_df.set_index("date").rolling(7, min_periods=1).mean().reset_index()

if ts_style == "line":
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["spend"], name="Spend"))
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["total_revenue"], name="Revenue"))
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["gross_profit"], name="Gross Profit"))
elif ts_style == "area":
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["spend"], name="Spend", fill="tozeroy"))
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["total_revenue"], name="Revenue", fill="tozeroy"))
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["gross_profit"], name="Gross Profit", fill="tozeroy"))
else:
    fig_ts = px.area(ts_df, x="date", y=["spend", "total_revenue", "gross_profit"])

fig_ts.update_layout(height=420, xaxis_title="Date", yaxis_title="USD")
st.plotly_chart(fig_ts, use_container_width=True)

# ----------------------- Channel comparison -----------------------
st.header("Channel Comparison")
channel_agg = marketing_daily_filtered.groupby("channel", as_index=False).agg({
    "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
})
channel_agg["roas"] = channel_agg["attributed_revenue"] / channel_agg["spend"].replace(0, np.nan)
channel_agg["ctr"] = channel_agg["clicks"] / channel_agg["impressions"].replace(0, np.nan)

left_c, right_c = st.columns([2, 1])
with left_c:
    fig_ch_bar = px.bar(channel_agg, x="channel", y=["spend", "attributed_revenue"], barmode="group", title="Spend vs Attributed Revenue")
    st.plotly_chart(fig_ch_bar, use_container_width=True)
with right_c:
    fig_ch_scatter = px.scatter(channel_agg, x="spend", y="roas", size="impressions", hover_name="channel", title="Spend vs ROAS")
    st.plotly_chart(fig_ch_scatter, use_container_width=True)

# ----------------------- Funnel -----------------------
st.header("Conversion Funnel")
funnel_vals = [marketing_filtered["impressions"].sum(), marketing_filtered["clicks"].sum(), merged_filtered["orders"].sum()]
fig_funnel = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Orders"], x=funnel_vals))
st.plotly_chart(fig_funnel, use_container_width=True)

# ----------------------- ROAS by state (optional) -----------------------
st.header("ROAS by State")
if "state" in marketing_df.columns:
    state_df = marketing_filtered.groupby(["state", "channel"], as_index=False).agg({"attributed_revenue": "sum", "spend": "sum"})
    state_df["roas"] = state_df["attributed_revenue"] / state_df["spend"].replace(0, np.nan)
    if not state_df.empty:
        pivot_roas = state_df.pivot(index="channel", columns="state", values="roas").fillna(0)
        fig_state_heat = px.imshow(pivot_roas, labels={"x": "State", "y": "Channel", "color": "ROAS"})
        st.plotly_chart(fig_state_heat, use_container_width=True)
    else:
        st.info("No state-level data for selected filters.")
else:
    st.info("State column not present in marketing datasets.")

# ----------------------- Anomaly alerts -----------------------
st.header("Anomaly Alerts")
mean_spend = marketing_daily_filtered["spend"].mean()
std_spend = marketing_daily_filtered["spend"].std(ddof=0)
spike_threshold = mean_spend + spend_sigma * (std_spend or 0)
spend_spikes = marketing_daily_filtered[marketing_daily_filtered["spend"] > spike_threshold].sort_values(["date", "spend"], ascending=[True, False])
low_roas_days = merged_filtered[merged_filtered["roas"] < roas_flag].sort_values("date")

if not spend_spikes.empty:
    st.subheader("Spend spikes")
    st.dataframe(spend_spikes[["date", "channel", "spend"]].reset_index(drop=True))
else:
    st.info("No spend spikes detected with current threshold.")

if not low_roas_days.empty:
    st.subheader("Low ROAS days")
    st.dataframe(low_roas_days[["date", "spend", "attributed_revenue", "roas"]].reset_index(drop=True))
else:
    st.info("No low ROAS days detected with current threshold.")

# ----------------------- Campaign leaderboard -----------------------
st.header("Campaign Leaderboard")
if "campaign" in marketing_df.columns:
    camp_summary = marketing_filtered.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
    })
    camp_summary["roas"] = camp_summary["attributed_revenue"] / camp_summary["spend"].replace(0, np.nan)
    st.dataframe(camp_summary.sort_values("spend", ascending=False).reset_index(drop=True))
else:
    st.info("Campaign column not found.")

# ----------------------- Export merged CSV -----------------------
st.sidebar.header("Export")
st.sidebar.download_button("Download merged CSV", data=merged_filtered.to_csv(index=False), file_name="merged_data.csv", mime="text/csv")
# ----------------------- Footer -----------------------
st.markdown("---")
