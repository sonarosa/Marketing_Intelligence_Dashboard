"""
Marketing Intelligence Report — Streamlit App (fpdf2 safe)

Usage:
    streamlit run streamlit_bi_dashboard.py

Place CSVs in the `datasets/` folder:
    - Facebook.csv
    - Google.csv
    - TikTok.csv
    - business.csv  (case-insensitive)

Dependencies (requirements.txt):
    streamlit
    pandas
    numpy
    plotly
    fpdf2
"""
import os
import io
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="Marketing Intelligence Report")

# Header / context
st.title("Marketing Intelligence Report")
with st.expander("Context"):
    st.markdown(
        "Campaign-level marketing (Facebook, Google, TikTok) joined with daily business "
        "(orders, new_customers, revenue, profit). Use this dashboard to link marketing spend "
        "to business outcomes."
    )

# ----------------------- Dataset paths & checks -----------------------
DATA_DIR = "datasets"
expected_paths = {
    "Facebook.csv": os.path.join(DATA_DIR, "Facebook.csv"),
    "Google.csv": os.path.join(DATA_DIR, "Google.csv"),
    "TikTok.csv": os.path.join(DATA_DIR, "TikTok.csv"),
}
business_lower = os.path.join(DATA_DIR, "business.csv")
business_upper = os.path.join(DATA_DIR, "Business.csv")
expected_paths["business.csv"] = business_lower if os.path.exists(business_lower) else business_upper

missing_files = [name for name, path in expected_paths.items() if not os.path.exists(path)]
if missing_files:
    st.error("Missing CSV files in `datasets/`. Add these files and reload:")
    for mf in missing_files:
        st.write(f"- {mf}")
    st.stop()

facebook_path = expected_paths["Facebook.csv"]
google_path = expected_paths["Google.csv"]
tiktok_path = expected_paths["TikTok.csv"]
business_path = expected_paths["business.csv"]

# ----------------------- Data loading & preparation -----------------------
@st.cache_data
def load_and_prepare_data(fb_path, g_path, t_path, b_path):
    def normalize_columns(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    fb_df = normalize_columns(pd.read_csv(fb_path)); fb_df["channel"] = "Facebook"
    g_df = normalize_columns(pd.read_csv(g_path)); g_df["channel"] = "Google"
    t_df = normalize_columns(pd.read_csv(t_path)); t_df["channel"] = "TikTok"
    biz_df = normalize_columns(pd.read_csv(b_path))

    # combined campaign-level marketing dataframe
    marketing_df = pd.concat([fb_df, g_df, t_df], ignore_index=True, sort=False)

    # normalize column name variants
    if "impression" in marketing_df.columns and "impressions" not in marketing_df.columns:
        marketing_df["impressions"] = marketing_df["impression"]

    # attributed revenue detection
    ar_candidates = [c for c in marketing_df.columns if "attribut" in c]
    if ar_candidates:
        marketing_df["attributed_revenue"] = pd.to_numeric(marketing_df[ar_candidates[0]], errors="coerce").fillna(0)
    else:
        marketing_df["attributed_revenue"] = 0.0

    # ensure numeric columns
    for col in ["impressions", "clicks", "spend", "attributed_revenue"]:
        marketing_df[col] = pd.to_numeric(marketing_df.get(col, 0), errors="coerce").fillna(0)

    # business numeric casting and rename
    biz_df = biz_df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in biz_df.columns})
    for col in ["orders", "new_orders", "new_customers", "total_revenue", "gross_profit", "cogs"]:
        if col in biz_df.columns:
            biz_df[col] = pd.to_numeric(biz_df[col], errors="coerce").fillna(0)
        else:
            biz_df[col] = 0

    # parse dates to date objects
    marketing_df["date"] = pd.to_datetime(marketing_df["date"]).dt.date
    biz_df["date"] = pd.to_datetime(biz_df["date"]).dt.date

    # daily aggregates by channel
    marketing_daily_df = marketing_df.groupby(["date", "channel"], as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # daily totals across channels
    marketing_daily_totals = marketing_daily_df.groupby("date", as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # merge business metrics with marketing daily totals
    business_merged_df = pd.merge(biz_df, marketing_daily_totals, on="date", how="left").fillna(0)

    # derived KPIs
    business_merged_df["ctr"] = business_merged_df["clicks"] / business_merged_df["impressions"].replace(0, np.nan)
    business_merged_df["cpc"] = business_merged_df["spend"] / business_merged_df["clicks"].replace(0, np.nan)
    business_merged_df["cpm"] = (business_merged_df["spend"] / business_merged_df["impressions"].replace(0, np.nan)) * 1000
    business_merged_df["roas"] = business_merged_df["attributed_revenue"] / business_merged_df["spend"].replace(0, np.nan)
    business_merged_df["cac"] = business_merged_df["spend"] / business_merged_df["new_customers"].replace(0, np.nan)
    business_merged_df["aov"] = business_merged_df["total_revenue"] / business_merged_df["orders"].replace(0, np.nan)
    business_merged_df["profit_margin"] = business_merged_df["gross_profit"] / business_merged_df["total_revenue"].replace(0, np.nan)

    return marketing_df, marketing_daily_df, business_merged_df

try:
    marketing_df, marketing_daily_df, business_merged_df = load_and_prepare_data(
        facebook_path, google_path, tiktok_path, business_path
    )
except Exception as err:
    st.error(f"Error loading datasets: {err}")
    st.stop()

# ----------------------- Sidebar controls -----------------------
st.sidebar.header("Filters and Controls")
min_date, max_date = business_merged_df["date"].min(), business_merged_df["date"].max()
selected_date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

available_channels = sorted(marketing_df["channel"].dropna().unique().tolist())
selected_channels = st.sidebar.multiselect("Channels", options=available_channels, default=available_channels)

has_state_col = "state" in marketing_df.columns
available_states = sorted(marketing_df["state"].dropna().unique().tolist()) if has_state_col else []
selected_states = st.sidebar.multiselect("States", options=available_states, default=available_states if available_states else None)

has_campaign_col = "campaign" in marketing_df.columns
campaign_query = st.sidebar.text_input("Campaign search (substring)") if has_campaign_col else None

apply_smoothing = st.sidebar.checkbox("Apply 7-day rolling smoothing", value=False)
time_series_type = st.sidebar.selectbox("Time series chart type", options=["line", "area", "stacked_area"])
roas_flag_threshold = st.sidebar.slider("ROAS lower bound (flag if below)", 0.0, 5.0, 1.0, 0.1)
spend_spike_multiplier = st.sidebar.slider("Spend spike threshold (std multiples)", 0.0, 5.0, 2.0, 0.5)

# ----------------------- Apply filters -----------------------
mask_marketing = (
    (marketing_df["date"] >= selected_date_range[0])
    & (marketing_df["date"] <= selected_date_range[1])
    & (marketing_df["channel"].isin(selected_channels))
)
if has_state_col and selected_states:
    mask_marketing &= marketing_df["state"].isin(selected_states)
if has_campaign_col and campaign_query:
    mask_marketing &= marketing_df["campaign"].str.contains(campaign_query, case=False, na=False)
filtered_marketing_df = marketing_df.loc[mask_marketing].copy()

mask_marketing_daily = (
    (marketing_daily_df["date"] >= selected_date_range[0])
    & (marketing_daily_df["date"] <= selected_date_range[1])
    & (marketing_daily_df["channel"].isin(selected_channels))
)
filtered_marketing_daily_df = marketing_daily_df.loc[mask_marketing_daily].copy()

mask_business = (business_merged_df["date"] >= selected_date_range[0]) & (business_merged_df["date"] <= selected_date_range[1])
filtered_business_df = business_merged_df.loc[mask_business].copy()

# ----------------------- KPI cards -----------------------
st.header("Key Performance Indicators")
st.write("High-level KPIs linking marketing activity to business performance for selected filters.")
kpi_col_1, kpi_col_2, kpi_col_3, kpi_col_4, kpi_col_5 = st.columns(5)
kpi_col_1.metric("Total Revenue", f"${filtered_business_df['total_revenue'].sum():,.0f}")
kpi_col_2.metric("Gross Profit", f"${filtered_business_df['gross_profit'].sum():,.0f}")
kpi_col_3.metric("Marketing Spend", f"${filtered_business_df['spend'].sum():,.0f}")
roas_value = filtered_business_df['attributed_revenue'].sum() / (filtered_business_df['spend'].sum() or np.nan)
kpi_col_4.metric("ROAS", f"{roas_value:.2f}" if not np.isnan(roas_value) else "N/A")
cac_value = filtered_business_df['spend'].sum() / (filtered_business_df['new_customers'].sum() or np.nan)
kpi_col_5.metric("CAC", f"${cac_value:.2f}" if not np.isnan(cac_value) else "N/A")

# ----------------------- Time series -----------------------
st.header("Time Series: Spend vs Revenue vs Profit")
st.write("Adjust smoothing and chart type in the sidebar.")
ts_df = filtered_business_df[["date", "spend", "total_revenue", "gross_profit"]].sort_values("date")
if apply_smoothing:
    ts_df = ts_df.set_index("date").rolling(7, min_periods=1).mean().reset_index()

if time_series_type == "line":
    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["spend"], name="Spend"))
    ts_fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["total_revenue"], name="Revenue"))
    ts_fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["gross_profit"], name="Gross Profit"))
elif time_series_type == "area":
    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["spend"], name="Spend", fill="tozeroy"))
    ts_fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["total_revenue"], name="Revenue", fill="tozeroy"))
    ts_fig.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["gross_profit"], name="Gross Profit", fill="tozeroy"))
else:
    ts_fig = px.area(ts_df, x="date", y=["spend", "total_revenue", "gross_profit"])

ts_fig.update_layout(height=420, xaxis_title="Date", yaxis_title="USD")
st.plotly_chart(ts_fig, use_container_width=True)

# ----------------------- Channel comparison -----------------------
st.header("Channel Comparison")
st.write("Compare channels on spend, attributed revenue, CTR and ROAS.")
channel_summary_df = filtered_marketing_daily_df.groupby("channel", as_index=False).agg({
    "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
})
channel_summary_df["roas"] = channel_summary_df["attributed_revenue"] / channel_summary_df["spend"].replace(0, np.nan)
channel_summary_df["ctr"] = channel_summary_df["clicks"] / channel_summary_df["impressions"].replace(0, np.nan)

left_col, right_col = st.columns([2, 1])
with left_col:
    st.plotly_chart(px.bar(channel_summary_df, x="channel", y=["spend", "attributed_revenue"], barmode="group", title="Spend vs Attributed Revenue"), use_container_width=True)
with right_col:
    st.plotly_chart(px.scatter(channel_summary_df, x="spend", y="roas", size="impressions", hover_name="channel", title="Spend vs ROAS"), use_container_width=True)

# ----------------------- Conversion funnel -----------------------
st.header("Conversion Funnel")
st.write("Impressions → Clicks → Orders for the selected slice.")
funnel_values = [filtered_marketing_df["impressions"].sum(), filtered_marketing_df["clicks"].sum(), filtered_business_df["orders"].sum()]
funnel_fig = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Orders"], x=funnel_values))
st.plotly_chart(funnel_fig, use_container_width=True)

# ----------------------- ROAS by state (unique) -----------------------
st.header("ROAS by State (if available)")
if has_state_col:
    state_channel_df = filtered_marketing_df.groupby(["state", "channel"], as_index=False).agg({"attributed_revenue": "sum", "spend": "sum"})
    state_channel_df["roas"] = state_channel_df["attributed_revenue"] / state_channel_df["spend"].replace(0, np.nan)
    if not state_channel_df.empty:
        pivot_roas = state_channel_df.pivot(index="channel", columns="state", values="roas").fillna(0)
        heat_fig = px.imshow(pivot_roas, labels={"x": "State", "y": "Channel", "color": "ROAS"}, aspect="auto")
        st.plotly_chart(heat_fig, use_container_width=True)
    else:
        st.info("No state-level data for selected filters.")
else:
    st.info("State column not present in marketing datasets.")

# ----------------------- Anomaly alerts -----------------------
st.header("Anomaly Alerts")
spend_mean_val = filtered_marketing_daily_df["spend"].mean()
spend_std_val = filtered_marketing_daily_df["spend"].std(ddof=0)
spike_threshold_val = spend_mean_val + spend_spike_multiplier * (spend_std_val or 0)
spend_spikes_df = filtered_marketing_daily_df[filtered_marketing_daily_df["spend"] > spike_threshold_val].sort_values(["date", "spend"], ascending=[True, False])
low_roas_df = filtered_business_df[filtered_business_df["roas"] < roas_flag_threshold].sort_values("date")

if not spend_spikes_df.empty:
    st.subheader("Spend spikes (channel-level)")
    st.dataframe(spend_spikes_df[["date", "channel", "spend"]].reset_index(drop=True))
else:
    st.info("No spend spikes detected with current threshold.")

if not low_roas_df.empty:
    st.subheader("Low ROAS days")
    st.dataframe(low_roas_df[["date", "spend", "attributed_revenue", "roas"]].reset_index(drop=True))
else:
    st.info("No low ROAS days detected with current threshold.")

# ----------------------- Campaign leaderboard -----------------------
st.header("Campaign Leaderboard")
if has_campaign_col:
    campaign_summary_df = filtered_marketing_df.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
    })
    campaign_summary_df["roas"] = campaign_summary_df["attributed_revenue"] / campaign_summary_df["spend"].replace(0, np.nan)
    st.dataframe(campaign_summary_df.sort_values("spend", ascending=False).reset_index(drop=True))
else:
    st.info("Campaign column not found in marketing datasets.")

# ----------------------- Export options (CSV + PDF) -----------------------
st.sidebar.header("Export")
st.sidebar.markdown("Download processed merged dataset or a short PDF summary.")
st.sidebar.download_button("Download merged CSV", data=filtered_business_df.to_csv(index=False), file_name="merged_data.csv", mime="text/csv")

# helper to make PDF-safe text (break long tokens and force latin-1)
def _make_pdf_safe(text_obj: object, wrap_width: int = 80) -> str:
    if text_obj is None:
        return ""
    s = str(text_obj)
    s = s.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    # break very long words to avoid no-space errors inside fpdf2
    tokens = []
    for token in s.split(" "):
        if len(token) > 200:
            chunks = [token[i:i+100] for i in range(0, len(token), 100)]
            tokens.append(" ".join(chunks))
        else:
            tokens.append(token)
    safe_text = " ".join(tokens)
    wrapped = "\n".join(textwrap.fill(line, width=wrap_width) for line in safe_text.splitlines())
    return wrapped.encode("latin-1", "replace").decode("latin-1")

def build_pdf_bytes_safe(biz_df, camp_df, spikes_df, low_roas_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 8, _make_pdf_safe("Marketing Intelligence - Summary", wrap_width=100), ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, _make_pdf_safe(f"Date range: {biz_df['date'].min()} to {biz_df['date'].max()}"), ln=True)
    pdf.cell(0, 6, _make_pdf_safe(f"Total Revenue: ${biz_df['total_revenue'].sum():,.0f}"), ln=True)
    pdf.cell(0, 6, _make_pdf_safe(f"Marketing Spend: ${biz_df['spend'].sum():,.0f}"), ln=True)
    roas_val_pdf = biz_df['attributed_revenue'].sum() / (biz_df['spend'].sum() or np.nan)
    pdf.cell(0, 6, _make_pdf_safe(f"ROAS: {roas_val_pdf:.2f}" if not np.isnan(roas_val_pdf) else "ROAS: N/A"), ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 6, _make_pdf_safe("Top campaigns (by spend)"), ln=True)
    pdf.set_font("Helvetica", size=9)
    if camp_df is not None and not camp_df.empty:
        top_camps = camp_df.sort_values("spend", ascending=False).head(8)
        for _, row in top_camps.iterrows():
            line = f"- {row.get('campaign','')[:120]} | {row.get('channel','')} | spend: ${row.get('spend',0):,.0f} | ROAS: {row.get('roas',np.nan):.2f}"
            pdf.multi_cell(0, 5, _make_pdf_safe(line))
    else:
        pdf.cell(0, 5, _make_pdf_safe("No campaign data available."), ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 6, _make_pdf_safe("Anomalies Detected"), ln=True)
    pdf.set_font("Helvetica", size=9)
    if not spikes_df.empty:
        pdf.cell(0, 5, _make_pdf_safe("Spend spikes:"), ln=True)
        for _, r in spikes_df.head(8).iterrows():
            pdf.multi_cell(0, 5, _make_pdf_safe(f"- {r['date']} | {r['channel']} | spend: ${r['spend']:,.0f}"))
    else:
        pdf.cell(0, 5, _make_pdf_safe("No spend spikes detected."), ln=True)

    if not low_roas_df.empty:
        pdf.ln(2)
        pdf.cell(0, 5, _make_pdf_safe("Low ROAS days:"), ln=True)
        for _, r in low_roas_df.head(8).iterrows():
            pdf.multi_cell(0, 5, _make_pdf_safe(f"- {r['date']} | spend: ${r['spend']:,.0f} | roas: {r['roas']:.2f}"))
    else:
        pdf.cell(0, 5, _make_pdf_safe("No low-ROAS days detected."), ln=True)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return io.BytesIO(pdf_bytes).getvalue()

# prepare campaign dataframe for PDF
campaign_pdf_df = None
if has_campaign_col:
    campaign_pdf_df = filtered_marketing_df.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum"
    })
    campaign_pdf_df["roas"] = campaign_pdf_df["attributed_revenue"] / campaign_pdf_df["spend"].replace(0, np.nan)

if st.sidebar.button("Generate PDF Summary"):
    pdf_bytes = build_pdf_bytes_safe(filtered_business_df, campaign_pdf_df, spend_spikes_df, low_roas_df)
    st.sidebar.download_button("Download PDF", data=pdf_bytes, file_name="marketing_summary.pdf", mime="application/pdf")
    st.sidebar.success("PDF generated. Click Download PDF to save.")
