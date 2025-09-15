"""
Streamlit BI Dashboard for Marketing Intelligence Assessment

Usage:
    streamlit run streamlit_bi_dashboard.py

Place CSVs in the `datasets/` folder:
    - Facebook.csv
    - Google.csv
    - TikTok.csv
    - business.csv  (case-insensitive)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard")

# ------------------- Header -------------------
st.title("Marketing Intelligence Dashboard")
st.markdown("Automatically loading data from the `datasets/` folder. Place your CSVs there and reload the app.")

# ------------------- Dataset paths & checks -------------------
DATASET_DIR = "datasets"
paths = {
    "Facebook.csv": os.path.join(DATASET_DIR, "Facebook.csv"),
    "Google.csv": os.path.join(DATASET_DIR, "Google.csv"),
    "TikTok.csv": os.path.join(DATASET_DIR, "TikTok.csv"),
}
# accept either business.csv or Business.csv, prefer lowercase
business_lower = os.path.join(DATASET_DIR, "business.csv")
business_upper = os.path.join(DATASET_DIR, "Business.csv")
paths["business.csv"] = business_lower if os.path.exists(business_lower) else business_upper

missing = [name for name, p in paths.items() if not os.path.exists(p)]
if missing:
    st.error("❌ Missing CSV files. Please ensure these files are in the `datasets/` folder:")
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

    # ensure numeric fields
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
    st.success("✅ All datasets loaded successfully.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ------------------- Filters -------------------
st.sidebar.header("Interactive Filters")
min_d, max_d = merged["date"].min(), merged["date"].max()
date_range = st.sidebar.date_input("Date range", [min_d, max_d], min_value=min_d, max_value=max_d)
channels = st.sidebar.multiselect("Select Channels", options=marketing_daily["channel"].unique(), default=list(marketing_daily["channel"].unique()))

merged_f = merged[(merged["date"] >= date_range[0]) & (merged["date"] <= date_range[1])]
marketing_daily_f = marketing_daily[(marketing_daily["date"] >= date_range[0]) & (marketing_daily["date"] <= date_range[1]) & (marketing_daily["channel"].isin(channels))]

# ------------------- KPI Cards -------------------
st.subheader("📊 Business Pulse: Key KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"${merged_f['total_revenue'].sum():,.0f}")
k2.metric("Gross Profit", f"${merged_f['gross_profit'].sum():,.0f}")
k3.metric("Marketing Spend", f"${merged_f['spend'].sum():,.0f}")
roas_val = merged_f['attributed_revenue'].sum() / (merged_f['spend'].sum() or np.nan)
k4.metric("ROAS", f"{roas_val:.2f}" if not np.isnan(roas_val) else "N/A")
cac_val = merged_f['spend'].sum() / (merged_f['new_customers'].sum() or np.nan)
k5.metric("CAC", f"${cac_val:.2f}" if not np.isnan(cac_val) else "N/A")

# ------------------- Time Series -------------------
st.subheader("Revenue & Spend trends")
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["spend"], name="Spend"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["total_revenue"], name="Revenue"))
fig.add_trace(go.Scatter(x=merged_f["date"], y=merged_f["gross_profit"], name="Gross Profit"))
st.plotly_chart(fig, use_container_width=True)

# ------------------- Channel Comparison -------------------
st.subheader("Channel Efficiency & Scale")
channel_agg = marketing_daily_f.groupby("channel", as_index=False).agg({
    "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
})
channel_agg["roas"] = channel_agg["attributed_revenue"] / channel_agg["spend"].replace(0, np.nan)
channel_agg["ctr"] = channel_agg["clicks"] / channel_agg["impressions"].replace(0, np.nan)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(channel_agg, x="channel", y=["spend", "attributed_revenue"], barmode="group", title="Spend vs Revenue by Channel"), use_container_width=True)
with c2:
    st.plotly_chart(px.scatter(channel_agg, x="spend", y="roas", size="impressions", hover_name="channel", title="Spend vs ROAS"), use_container_width=True)

# ------------------- Conversion Funnel -------------------
st.subheader("Conversion Journey")
funnel_vals = [marketing_daily_f["impressions"].sum(), marketing_daily_f["clicks"].sum(), merged_f["orders"].sum()]
fig2 = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Orders"], x=funnel_vals))
st.plotly_chart(fig2, use_container_width=True)

# ------------------- Campaign Leaderboard -------------------
st.subheader("Campaign Performance Table")
if "campaign" in marketing.columns:
    camp_agg = marketing.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
    })
    camp_agg["roas"] = camp_agg["attributed_revenue"] / camp_agg["spend"].replace(0, np.nan)
    st.dataframe(camp_agg.sort_values("spend", ascending=False))
else:
    st.info("No 'campaign' column found in marketing datasets.")

# ------------------- Export -------------------
st.sidebar.header("Export Data")
st.sidebar.markdown("Download the processed dataset as CSV.")
st.sidebar.download_button("Download merged CSV", data=merged.to_csv(index=False), file_name="merged_data.csv")
