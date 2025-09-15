# streamlit_bi_dashboard.py
"""
Marketing Intelligence Report — Streamlit App
Place CSVs in `datasets/`:
  - datasets/Facebook.csv
  - datasets/Google.csv
  - datasets/TikTok.csv
  - datasets/business.csv

Run:
  streamlit run streamlit_bi_dashboard.py
"""
import os
import io
import textwrap
import re
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
expected_files = {
    "Facebook.csv": os.path.join(DATA_DIR, "Facebook.csv"),
    "Google.csv": os.path.join(DATA_DIR, "Google.csv"),
    "TikTok.csv": os.path.join(DATA_DIR, "TikTok.csv"),
}
# business may be business.csv or Business.csv
business_lower = os.path.join(DATA_DIR, "business.csv")
business_upper = os.path.join(DATA_DIR, "Business.csv")
expected_files["business.csv"] = business_lower if os.path.exists(business_lower) else business_upper

missing_files = [name for name, path in expected_files.items() if not os.path.exists(path)]
if missing_files:
    st.error("Missing CSV files in `datasets/`. Add these files and reload:")
    for m in missing_files:
        st.write(f"- {m}")
    st.stop()

facebook_path = expected_files["Facebook.csv"]
google_path = expected_files["Google.csv"]
tiktok_path = expected_files["TikTok.csv"]
business_path = expected_files["business.csv"]

# ----------------------- Data load & prepare -----------------------
@st.cache_data
def load_and_prepare(fb_path, g_path, t_path, biz_path):
    def normalize_columns(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    fb_df = normalize_columns(pd.read_csv(fb_path)); fb_df["channel"] = "Facebook"
    g_df = normalize_columns(pd.read_csv(g_path)); g_df["channel"] = "Google"
    t_df = normalize_columns(pd.read_csv(t_path)); t_df["channel"] = "TikTok"
    biz_df = normalize_columns(pd.read_csv(biz_path))

    # combine marketing
    marketing_df = pd.concat([fb_df, g_df, t_df], ignore_index=True, sort=False)

    # normalize impression(s)
    if "impression" in marketing_df.columns and "impressions" not in marketing_df.columns:
        marketing_df["impressions"] = marketing_df["impression"]

    # attributed revenue detection
    ar_candidates = [c for c in marketing_df.columns if "attribut" in c]
    if ar_candidates:
        marketing_df["attributed_revenue"] = pd.to_numeric(marketing_df[ar_candidates[0]], errors="coerce").fillna(0)
    else:
        marketing_df["attributed_revenue"] = 0.0

    # numeric cast for marketing columns
    for col in ["impressions", "clicks", "spend", "attributed_revenue"]:
        marketing_df[col] = pd.to_numeric(marketing_df.get(col, 0), errors="coerce").fillna(0)

    # business numeric fields
    biz_df = biz_df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in biz_df.columns})
    for col in ["orders", "new_orders", "new_customers", "total_revenue", "gross_profit", "cogs"]:
        if col in biz_df.columns:
            biz_df[col] = pd.to_numeric(biz_df[col], errors="coerce").fillna(0)
        else:
            biz_df[col] = 0

    # date columns
    marketing_df["date"] = pd.to_datetime(marketing_df["date"]).dt.date
    biz_df["date"] = pd.to_datetime(biz_df["date"]).dt.date

    # daily channel aggregates
    marketing_daily = marketing_df.groupby(["date", "channel"], as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # daily totals
    marketing_totals = marketing_daily.groupby("date", as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # merge business with marketing totals
    merged = pd.merge(biz_df, marketing_totals, on="date", how="left").fillna(0)

    # derived metrics
    merged["ctr"] = merged["clicks"] / merged["impressions"].replace(0, np.nan)
    merged["cpc"] = merged["spend"] / merged["clicks"].replace(0, np.nan)
    merged["cpm"] = (merged["spend"] / merged["impressions"].replace(0, np.nan)) * 1000
    merged["roas"] = merged["attributed_revenue"] / merged["spend"].replace(0, np.nan)
    merged["cac"] = merged["spend"] / merged["new_customers"].replace(0, np.nan)
    merged["aov"] = merged["total_revenue"] / merged["orders"].replace(0, np.nan)
    merged["profit_margin"] = merged["gross_profit"] / merged["total_revenue"].replace(0, np.nan)

    return marketing_df, marketing_daily, merged

marketing_df, marketing_daily_df, merged_df = load_and_prepare(
    facebook_path, google_path, tiktok_path, business_path
)

# ----------------------- Sidebar controls -----------------------
st.sidebar.header("Filters and options")
min_date, max_date = merged_df["date"].min(), merged_df["date"].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

available_channels = sorted(marketing_df["channel"].dropna().unique().tolist())
selected_channels = st.sidebar.multiselect("Channels", options=available_channels, default=available_channels)

state_column_exists = "state" in marketing_df.columns
available_states = sorted(marketing_df["state"].dropna().unique().tolist()) if state_column_exists else []
selected_states = st.sidebar.multiselect("States", options=available_states, default=available_states if available_states else None)

campaign_column_exists = "campaign" in marketing_df.columns
campaign_search_text = st.sidebar.text_input("Campaign substring (optional)") if campaign_column_exists else None

apply_smoothing = st.sidebar.checkbox("7-day smoothing", value=False)
time_series_style = st.sidebar.selectbox("Time series type", options=["line", "area", "stacked"])
roas_threshold_flag = st.sidebar.slider("Flag ROAS <=", 0.0, 5.0, 1.0, 0.1)
spend_spike_sigma = st.sidebar.slider("Spend spike sigma multiplier", 0.0, 5.0, 2.0, 0.5)

# ----------------------- Apply filters -----------------------
mask_marketing = (
    (marketing_df["date"] >= date_range[0]) &
    (marketing_df["date"] <= date_range[1]) &
    (marketing_df["channel"].isin(selected_channels))
)
if state_column_exists and selected_states:
    mask_marketing &= marketing_df["state"].isin(selected_states)
if campaign_column_exists and campaign_search_text:
    mask_marketing &= marketing_df["campaign"].str.contains(campaign_search_text, case=False, na=False)
marketing_filtered = marketing_df.loc[mask_marketing].copy()

mask_marketing_daily = (
    (marketing_daily_df["date"] >= date_range[0]) &
    (marketing_daily_df["date"] <= date_range[1]) &
    (marketing_daily_df["channel"].isin(selected_channels))
)
marketing_daily_filtered = marketing_daily_df.loc[mask_marketing_daily].copy()

mask_merged = (merged_df["date"] >= date_range[0]) & (merged_df["date"] <= date_range[1])
merged_filtered = merged_df.loc[mask_merged].copy()

# ----------------------- KPIs -----------------------
st.header("Key Performance Indicators")
col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
col_k1.metric("Total Revenue", f"${merged_filtered['total_revenue'].sum():,.0f}")
col_k2.metric("Gross Profit", f"${merged_filtered['gross_profit'].sum():,.0f}")
col_k3.metric("Marketing Spend", f"${merged_filtered['spend'].sum():,.0f}")
roas_value = merged_filtered['attributed_revenue'].sum() / (merged_filtered['spend'].sum() or np.nan)
col_k4.metric("ROAS", f"{roas_value:.2f}" if not np.isnan(roas_value) else "N/A")
cac_value = merged_filtered['spend'].sum() / (merged_filtered['new_customers'].sum() or np.nan)
col_k5.metric("CAC", f"${cac_value:.2f}" if not np.isnan(cac_value) else "N/A")

# ----------------------- Time series -----------------------
st.header("Time Series: Spend vs Revenue vs Profit")
time_series_df = merged_filtered[["date", "spend", "total_revenue", "gross_profit"]].sort_values("date")
if apply_smoothing:
    time_series_df = time_series_df.set_index("date").rolling(7, min_periods=1).mean().reset_index()

if time_series_style == "line":
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=time_series_df["date"], y=time_series_df["spend"], name="Spend"))
    fig_ts.add_trace(go.Scatter(x=time_series_df["date"], y=time_series_df["total_revenue"], name="Revenue"))
    fig_ts.add_trace(go.Scatter(x=time_series_df["date"], y=time_series_df["gross_profit"], name="Gross Profit"))
elif time_series_style == "area":
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=time_series_df["date"], y=time_series_df["spend"], name="Spend", fill="tozeroy"))
    fig_ts.add_trace(go.Scatter(x=time_series_df["date"], y=time_series_df["total_revenue"], name="Revenue", fill="tozeroy"))
    fig_ts.add_trace(go.Scatter(x=time_series_df["date"], y=time_series_df["gross_profit"], name="Gross Profit", fill="tozeroy"))
else:
    fig_ts = px.area(time_series_df, x="date", y=["spend", "total_revenue", "gross_profit"])

fig_ts.update_layout(height=420, xaxis_title="Date", yaxis_title="USD")
st.plotly_chart(fig_ts, use_container_width=True)

# ----------------------- Channel comparison -----------------------
st.header("Channel Comparison")
channel_summary = marketing_daily_filtered.groupby("channel", as_index=False).agg({
    "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
})
channel_summary["roas"] = channel_summary["attributed_revenue"] / channel_summary["spend"].replace(0, np.nan)
channel_summary["ctr"] = channel_summary["clicks"] / channel_summary["impressions"].replace(0, np.nan)

left_col, right_col = st.columns([2, 1])
with left_col:
    st.plotly_chart(px.bar(channel_summary, x="channel", y=["spend", "attributed_revenue"], barmode="group"), use_container_width=True)
with right_col:
    st.plotly_chart(px.scatter(channel_summary, x="spend", y="roas", size="impressions", hover_name="channel"), use_container_width=True)

# ----------------------- Funnel -----------------------
st.header("Conversion Funnel")
funnel_steps = [marketing_filtered["impressions"].sum(), marketing_filtered["clicks"].sum(), merged_filtered["orders"].sum()]
st.plotly_chart(go.Figure(go.Funnel(y=["Impressions", "Clicks", "Orders"], x=funnel_steps)), use_container_width=True)

# ----------------------- ROAS by state (optional) -----------------------
st.header("ROAS by State (if available)")
if "state" in marketing_df.columns:
    state_summary = marketing_filtered.groupby(["state", "channel"], as_index=False).agg({"attributed_revenue": "sum", "spend": "sum"})
    state_summary["roas"] = state_summary["attributed_revenue"] / state_summary["spend"].replace(0, np.nan)
    if not state_summary.empty:
        pivot_roas = state_summary.pivot(index="channel", columns="state", values="roas").fillna(0)
        st.plotly_chart(px.imshow(pivot_roas, labels={"x": "State", "y": "Channel", "color": "ROAS"}), use_container_width=True)
    else:
        st.info("No state-level data for selected filters.")
else:
    st.info("State column not present in marketing datasets.")

# ----------------------- Anomaly alerts -----------------------
st.header("Anomaly Alerts")
avg_spend = marketing_daily_filtered["spend"].mean()
std_spend = marketing_daily_filtered["spend"].std(ddof=0)
spike_threshold = avg_spend + spend_spike_sigma * (std_spend or 0)
spend_spikes_df = marketing_daily_filtered[marketing_daily_filtered["spend"] > spike_threshold].sort_values(["date", "spend"], ascending=[True, False])
low_roas_days_df = merged_filtered[merged_filtered["roas"] < roas_threshold_flag].sort_values("date")

if not spend_spikes_df.empty:
    st.subheader("Spend spikes")
    st.dataframe(spend_spikes_df[["date", "channel", "spend"]].reset_index(drop=True))
else:
    st.info("No spend spikes detected with current threshold.")

if not low_roas_days_df.empty:
    st.subheader("Low ROAS days")
    st.dataframe(low_roas_days_df[["date", "spend", "attributed_revenue", "roas"]].reset_index(drop=True))
else:
    st.info("No low ROAS days detected with current threshold.")

# ----------------------- Campaign leaderboard -----------------------
st.header("Campaign Leaderboard")
if "campaign" in marketing_df.columns:
    campaign_summary_df = marketing_filtered.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
    })
    campaign_summary_df["roas"] = campaign_summary_df["attributed_revenue"] / campaign_summary_df["spend"].replace(0, np.nan)
    st.dataframe(campaign_summary_df.sort_values("spend", ascending=False).reset_index(drop=True))
else:
    st.info("Campaign column not found.")

# ----------------------- Export CSV -----------------------
st.sidebar.header("Export")
st.sidebar.download_button("Download merged CSV", data=merged_filtered.to_csv(index=False), file_name="merged_data.csv", mime="text/csv")
