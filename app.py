"""
Marketing Intelligence Report — Streamlit App

Usage:
    streamlit run streamlit_bi_dashboard.py

Description:
    This app loads the four provided datasets by default from the local `datasets/` folder,
    prepares and merges campaign-level marketing data (Facebook, Google, TikTok) with
    daily business performance data. It computes common marketing and business KPIs
    and presents an interactive dashboard for analysis and export.

Datasets (place in `datasets/`):
    - datasets/Facebook.csv
    - datasets/Google.csv
    - datasets/TikTok.csv
    - datasets/business.csv

Features:
    - Cleans and merges all four datasets
    - Derives KPIs: CTR, CPC, CPM, ROAS, CAC, AOV, Profit Margin
    - Interactive filters (date, channel)
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
import os

st.set_page_config(layout="wide", page_title="Marketing Intelligence Report")

# ------------------- Header & Description -------------------
st.title("Marketing Intelligence Report")
st.markdown(
    "Load default datasets from `datasets/` and explore marketing impact on business outcomes."
)

# ------------------- Load data from datasets/ -------------------
@st.cache_data
def load_and_prepare():
    def norm(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        return df

    base = 'datasets'
    fb = norm(pd.read_csv(os.path.join(base, 'Facebook.csv'))); fb['channel'] = 'Facebook'
    g = norm(pd.read_csv(os.path.join(base, 'Google.csv'))); g['channel'] = 'Google'
    t = norm(pd.read_csv(os.path.join(base, 'TikTok.csv'))); t['channel'] = 'TikTok'
    b = norm(pd.read_csv(os.path.join(base, 'business.csv')))

    marketing = pd.concat([fb, g, t], ignore_index=True, sort=False)

    # normalize column variants
    if 'impression' in marketing.columns and 'impressions' not in marketing.columns:
        marketing['impressions'] = marketing['impression']
    # attributed revenue variants
    ar_cols = [c for c in marketing.columns if 'attributed' in c]
    marketing['attributed_revenue'] = pd.to_numeric(marketing[ar_cols[0]], errors='coerce') if ar_cols else 0.0

    for c in ['impressions', 'clicks', 'spend', 'attributed_revenue']:
        if c in marketing.columns:
            marketing[c] = pd.to_numeric(marketing[c], errors='coerce').fillna(0)
        else:
            marketing[c] = 0

    # business cleanup
    b = b.rename(columns={c: c.strip().lower().replace(' ', '_') for c in b.columns})
    for col in ['orders', 'new_customers', 'total_revenue', 'gross_profit']:
        b[col] = pd.to_numeric(b.get(col, 0), errors='coerce').fillna(0)

    # dates
    marketing['date'] = pd.to_datetime(marketing['date']).dt.date
    b['date'] = pd.to_datetime(b['date']).dt.date

    # aggregate
    m_daily = marketing.groupby(['date', 'channel']).agg({
        'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'
    }).reset_index()

    m_total = m_daily.groupby('date').agg({
        'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'
    }).reset_index()

    merged = pd.merge(b, m_total, on='date', how='left').fillna(0)

    # derived metrics
    merged['ctr'] = merged['clicks'] / merged['impressions'].replace(0, np.nan)
    merged['cpc'] = merged['spend'] / merged['clicks'].replace(0, np.nan)
    merged['cpm'] = (merged['spend'] / merged['impressions'].replace(0, np.nan)) * 1000
    merged['roas'] = merged['attributed_revenue'] / merged['spend'].replace(0, np.nan)
    merged['cac'] = merged['spend'] / merged['new_customers'].replace(0, np.nan)
    merged['aov'] = merged['total_revenue'] / merged['orders'].replace(0, np.nan)
    merged['profit_margin'] = merged['gross_profit'] / merged['total_revenue'].replace(0, np.nan)

    return marketing, m_daily, merged

marketing, marketing_daily, merged = load_and_prepare()

# ------------------- Filters -------------------
st.sidebar.header('Interactive Filters')
min_date, max_date = merged['date'].min(), merged['date'].max()
date_range = st.sidebar.date_input('Date range', value=[min_date, max_date], min_value=min_date, max_value=max_date)
channels = st.sidebar.multiselect('Channels', options=marketing['channel'].unique(), default=list(marketing['channel'].unique()))

mask_dates = (merged['date'] >= date_range[0]) & (merged['date'] <= date_range[1])
merged_f = merged.loc[mask_dates].copy()

mask_md = (marketing_daily['date'] >= date_range[0]) & (marketing_daily['date'] <= date_range[1]) & (marketing_daily['channel'].isin(channels))
marketing_daily_f = marketing_daily.loc[mask_md].copy()

# ------------------- KPI Cards -------------------
st.subheader('Business Pulse — Key KPIs')
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Total Revenue', f"${merged_f['total_revenue'].sum():,.0f}")
col2.metric('Gross Profit', f"${merged_f['gross_profit'].sum():,.0f}")
col3.metric('Marketing Spend', f"${merged_f['spend'].sum():,.0f}")
col4.metric('ROAS', f"{(merged_f['attributed_revenue'].sum() / (merged_f['spend'].sum() or np.nan)):.2f}")
col5.metric('CAC', f"${(merged_f['spend'].sum() / (merged_f['new_customers'].sum() or np.nan)):.2f}")

# ------------------- Time Series -------------------
st.subheader('Revenue & Spend Trajectory')
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged_f['date'], y=merged_f['spend'], name='Spend'))
fig.add_trace(go.Scatter(x=merged_f['date'], y=merged_f['total_revenue'], name='Revenue'))
fig.add_trace(go.Scatter(x=merged_f['date'], y=merged_f['gross_profit'], name='Gross Profit'))
fig.update_layout(height=420, xaxis_title='Date', yaxis_title='USD')
st.plotly_chart(fig, use_container_width=True)

# ------------------- Channel Comparison -------------------
st.subheader('Channel Efficiency & Scale')
channel_agg = marketing_daily_f.groupby('channel').agg({'spend':'sum','attributed_revenue':'sum','impressions':'sum','clicks':'sum'}).reset_index()
channel_agg['roas'] = channel_agg['attributed_revenue'] / channel_agg['spend'].replace(0, np.nan)
channel_agg['ctr'] = channel_agg['clicks'] / channel_agg['impressions'].replace(0, np.nan)

c_left, c_right = st.columns([2,1])
with c_left:
    st.plotly_chart(px.bar(channel_agg, x='channel', y=['spend','attributed_revenue'], barmode='group'), use_container_width=True)
with c_right:
    st.plotly_chart(px.scatter(channel_agg, x='spend', y='roas', size='impressions', hover_name='channel'), use_container_width=True)

# ------------------- Funnel -------------------
st.subheader('Conversion Journey')
funnel_steps = [merged_f['impressions'].sum(), merged_f['clicks'].sum(), merged_f['orders'].sum()]
fig_f = go.Figure(go.Funnel(y=['Impressions','Clicks','Orders'], x=funnel_steps))
st.plotly_chart(fig_f, use_container_width=True)

# ------------------- Campaign Table -------------------
st.subheader('Campaign Leaderboard')
if 'campaign' in marketing.columns:
    camp = marketing.groupby(['campaign','channel']).agg({'spend':'sum','attributed_revenue':'sum','impressions':'sum','clicks':'sum'}).reset_index()
    camp['roas'] = camp['attributed_revenue'] / camp['spend'].replace(0, np.nan)
    st.dataframe(camp.sort_values('spend', ascending=False))
else:
    st.info('No campaign column found in marketing datasets.')

# ------------------- Export -------------------
st.sidebar.header('Export')
st.sidebar.download_button('Download merged CSV', data=merged.to_csv(index=False), file_name='merged_data.csv')
st.sidebar.markdown('Data merged from all four datasets with derived KPIs.')