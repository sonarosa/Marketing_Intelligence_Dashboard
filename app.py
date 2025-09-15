# streamlit_bi_dashboard.py
"""
Marketing Intelligence Report — Streamlit App

Place CSVs in `datasets/`:
  - datasets/Facebook.csv
  - datasets/Google.csv
  - datasets/TikTok.csv
  - datasets/business.csv

Optional TTF for full unicode PDF:
  - fonts/DejaVuSans.ttf

Run:
  streamlit run streamlit_bi_dashboard.py
"""
import os
import io
import re
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

# ----------------------- Config -----------------------
st.set_page_config(layout="wide", page_title="Marketing Intelligence Report")
DATA_DIR = "datasets"
FONT_PATH = os.path.join("fonts", "DejaVuSans.ttf")  # optional

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

missing = [name for name, path in expected_files.items() if not os.path.exists(path)]
if missing:
    st.error("Missing CSV files in `datasets/`. Add these files and reload:")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

facebook_path = expected_files["Facebook.csv"]
google_path = expected_files["Google.csv"]
tiktok_path = expected_files["TikTok.csv"]
business_path = expected_files["business.csv"]

# ----------------------- Data load & prepare -----------------------
@st.cache_data
def load_prepare(fb_path, g_path, t_path, biz_path):
    def normalize(df):
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    fb = normalize(pd.read_csv(fb_path)); fb["channel"] = "Facebook"
    g = normalize(pd.read_csv(g_path)); g["channel"] = "Google"
    t = normalize(pd.read_csv(t_path)); t["channel"] = "TikTok"
    biz = normalize(pd.read_csv(biz_path))

    # combine marketing
    marketing = pd.concat([fb, g, t], ignore_index=True, sort=False)

    # normalize impression(s)
    if "impression" in marketing.columns and "impressions" not in marketing.columns:
        marketing["impressions"] = marketing["impression"]

    # attributed revenue detection
    ar_cols = [c for c in marketing.columns if "attribut" in c]
    if ar_cols:
        marketing["attributed_revenue"] = pd.to_numeric(marketing[ar_cols[0]], errors="coerce").fillna(0)
    else:
        marketing["attributed_revenue"] = 0.0

    # numeric cast
    for col in ["impressions", "clicks", "spend", "attributed_revenue"]:
        marketing[col] = pd.to_numeric(marketing.get(col, 0), errors="coerce").fillna(0)

    # business numeric fields
    biz = biz.rename(columns={c: c.strip().lower().replace(" ", "_") for c in biz.columns})
    for col in ["orders", "new_orders", "new_customers", "total_revenue", "gross_profit", "cogs"]:
        if col in biz.columns:
            biz[col] = pd.to_numeric(biz[col], errors="coerce").fillna(0)
        else:
            biz[col] = 0

    # dates
    marketing["date"] = pd.to_datetime(marketing["date"]).dt.date
    biz["date"] = pd.to_datetime(biz["date"]).dt.date

    # daily channel aggregates
    marketing_daily = marketing.groupby(["date", "channel"], as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # daily totals
    marketing_totals = marketing_daily.groupby("date", as_index=False).agg({
        "impressions": "sum", "clicks": "sum", "spend": "sum", "attributed_revenue": "sum"
    })

    # merge business with marketing totals
    merged = pd.merge(biz, marketing_totals, on="date", how="left").fillna(0)

    # derived metrics
    merged["ctr"] = merged["clicks"] / merged["impressions"].replace(0, np.nan)
    merged["cpc"] = merged["spend"] / merged["clicks"].replace(0, np.nan)
    merged["cpm"] = (merged["spend"] / merged["impressions"].replace(0, np.nan)) * 1000
    merged["roas"] = merged["attributed_revenue"] / merged["spend"].replace(0, np.nan)
    merged["cac"] = merged["spend"] / merged["new_customers"].replace(0, np.nan)
    merged["aov"] = merged["total_revenue"] / merged["orders"].replace(0, np.nan)
    merged["profit_margin"] = merged["gross_profit"] / merged["total_revenue"].replace(0, np.nan)

    return marketing, marketing_daily, merged

marketing_df, marketing_daily_df, merged_df = load_prepare(
    facebook_path, google_path, tiktok_path, business_path
)

# ----------------------- Sidebar controls -----------------------
st.sidebar.header("Filters and options")
min_date, max_date = merged_df["date"].min(), merged_df["date"].max()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

channels_all = sorted(marketing_df["channel"].dropna().unique().tolist())
channels_sel = st.sidebar.multiselect("Channels", options=channels_all, default=channels_all)

state_present = "state" in marketing_df.columns
states_all = sorted(marketing_df["state"].dropna().unique().tolist()) if state_present else []
states_sel = st.sidebar.multiselect("States", options=states_all, default=states_all if states_all else None)

campaign_present = "campaign" in marketing_df.columns
campaign_search = st.sidebar.text_input("Campaign substring (optional)") if campaign_present else None

apply_smooth = st.sidebar.checkbox("7-day smoothing", value=False)
chart_mode = st.sidebar.selectbox("Time series type", options=["line", "area", "stacked"])
roas_flag = st.sidebar.slider("Flag ROAS <=", 0.0, 5.0, 1.0, 0.1)
spend_sigma = st.sidebar.slider("Spend spike sigma multiplier", 0.0, 5.0, 2.0, 0.5)

# ----------------------- Apply filters -----------------------
mask_marketing = (
    (marketing_df["date"] >= date_range[0]) &
    (marketing_df["date"] <= date_range[1]) &
    (marketing_df["channel"].isin(channels_sel))
)
if state_present and states_sel:
    mask_marketing &= marketing_df["state"].isin(states_sel)
if campaign_present and campaign_search:
    mask_marketing &= marketing_df["campaign"].str.contains(campaign_search, case=False, na=False)
marketing_filtered = marketing_df.loc[mask_marketing].copy()

mask_marketing_daily = (
    (marketing_daily_df["date"] >= date_range[0]) &
    (marketing_daily_df["date"] <= date_range[1]) &
    (marketing_daily_df["channel"].isin(channels_sel))
)
marketing_daily_filtered = marketing_daily_df.loc[mask_marketing_daily].copy()

mask_merged = (merged_df["date"] >= date_range[0]) & (merged_df["date"] <= date_range[1])
merged_filtered = merged_df.loc[mask_merged].copy()

# ----------------------- KPIs -----------------------
st.header("Key Performance Indicators")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"${merged_filtered['total_revenue'].sum():,.0f}")
k2.metric("Gross Profit", f"${merged_filtered['gross_profit'].sum():,.0f}")
k3.metric("Marketing Spend", f"${merged_filtered['spend'].sum():,.0f}")
roas_val = merged_filtered['attributed_revenue'].sum() / (merged_filtered['spend'].sum() or np.nan)
k4.metric("ROAS", f"{roas_val:.2f}" if not np.isnan(roas_val) else "N/A")
cac_val = merged_filtered['spend'].sum() / (merged_filtered['new_customers'].sum() or np.nan)
k5.metric("CAC", f"${cac_val:.2f}" if not np.isnan(cac_val) else "N/A")

# ----------------------- Time series -----------------------
st.header("Time Series: Spend vs Revenue vs Profit")
ts_df = merged_filtered[["date", "spend", "total_revenue", "gross_profit"]].sort_values("date")
if apply_smooth:
    ts_df = ts_df.set_index("date").rolling(7, min_periods=1).mean().reset_index()

if chart_mode == "line":
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["spend"], name="Spend"))
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["total_revenue"], name="Revenue"))
    fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["gross_profit"], name="Gross Profit"))
elif chart_mode == "area":
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
channel_summary = marketing_daily_filtered.groupby("channel", as_index=False).agg({
    "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
})
channel_summary["roas"] = channel_summary["attributed_revenue"] / channel_summary["spend"].replace(0, np.nan)
channel_summary["ctr"] = channel_summary["clicks"] / channel_summary["impressions"].replace(0, np.nan)

left_c, right_c = st.columns([2, 1])
with left_c:
    st.plotly_chart(px.bar(channel_summary, x="channel", y=["spend", "attributed_revenue"], barmode="group"), use_container_width=True)
with right_c:
    st.plotly_chart(px.scatter(channel_summary, x="spend", y="roas", size="impressions", hover_name="channel"), use_container_width=True)

# ----------------------- Funnel -----------------------
st.header("Conversion Funnel")
funnel_vals = [marketing_filtered["impressions"].sum(), marketing_filtered["clicks"].sum(), merged_filtered["orders"].sum()]
st.plotly_chart(go.Figure(go.Funnel(y=["Impressions", "Clicks", "Orders"], x=funnel_vals)), use_container_width=True)

# ----------------------- ROAS by state (optional) -----------------------
st.header("ROAS by State (if available)")
if "state" in marketing_df.columns:
    state_df = marketing_filtered.groupby(["state", "channel"], as_index=False).agg({"attributed_revenue": "sum", "spend": "sum"})
    state_df["roas"] = state_df["attributed_revenue"] / state_df["spend"].replace(0, np.nan)
    if not state_df.empty:
        pivot_roas = state_df.pivot(index="channel", columns="state", values="roas").fillna(0)
        st.plotly_chart(px.imshow(pivot_roas, labels={"x": "State", "y": "Channel", "color": "ROAS"}), use_container_width=True)
    else:
        st.info("No state-level data for selected filters.")
else:
    st.info("State column not present in marketing datasets.")

# ----------------------- Anomaly alerts -----------------------
st.header("Anomaly Alerts")
spend_mean = marketing_daily_filtered["spend"].mean()
spend_std = marketing_daily_filtered["spend"].std(ddof=0)
spike_thresh = spend_mean + spend_sigma * (spend_std or 0)
spend_spikes = marketing_daily_filtered[marketing_daily_filtered["spend"] > spike_thresh].sort_values(["date", "spend"], ascending=[True, False])
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
    campaign_summary = marketing_filtered.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum", "impressions": "sum", "clicks": "sum"
    })
    campaign_summary["roas"] = campaign_summary["attributed_revenue"] / campaign_summary["spend"].replace(0, np.nan)
    st.dataframe(campaign_summary.sort_values("spend", ascending=False).reset_index(drop=True))
else:
    st.info("Campaign column not found.")

# ----------------------- Export CSV -----------------------
st.sidebar.header("Export")
st.sidebar.download_button("Download merged CSV", data=merged_filtered.to_csv(index=False), file_name="merged_data.csv", mime="text/csv")

# ----------------------- PDF helpers -----------------------
def _insert_breaks_long_tokens(text: str, chunk: int = 100):
    """Insert spaces into very long tokens so PDF wrapper can break them."""
    return re.sub(r"(\S{" + str(chunk) + r",})", lambda m: " ".join([m.group(0)[i:i+chunk] for i in range(0, len(m.group(0)), chunk)]), text)

def _wrap_text_for_pdf(obj, wrap_width=90):
    s = "" if obj is None else str(obj)
    s = s.replace("—", "-").replace("–", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    s = _insert_breaks_long_tokens(s, chunk=100)
    wrapped = "\n".join(textwrap.fill(line, width=wrap_width) for line in s.splitlines())
    return wrapped

def build_pdf_latin(merged_df_in, campaign_df_in, spikes_df_in, low_roas_df_in):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.multi_cell(usable_w, 8, _wrap_text_for_pdf("Marketing Intelligence - Summary", 100))
    pdf.ln(2)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"Date range: {merged_df_in['date'].min()} to {merged_df_in['date'].max()}"))
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"Total Revenue: ${merged_df_in['total_revenue'].sum():,.0f}"))
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"Marketing Spend: ${merged_df_in['spend'].sum():,.0f}"))
    roas_val = merged_df_in['attributed_revenue'].sum() / (merged_df_in['spend'].sum() or np.nan)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"ROAS: {roas_val:.2f}" if not np.isnan(roas_val) else "ROAS: N/A"))

    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf("Top campaigns (by spend)"))
    pdf.set_font("Helvetica", size=9)
    if campaign_df_in is not None and not campaign_df_in.empty:
        top_camps = campaign_df_in.sort_values("spend", ascending=False).head(8)
        for _, r in top_camps.iterrows():
            line = f"- {str(r.get('campaign',''))[:200]} | {r.get('channel','')} | spend: ${r.get('spend',0):,.0f} | ROAS: {r.get('roas',np.nan):.2f}"
            pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf(line, wrap_width=100))
    else:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("No campaign data available."))

    pdf.ln(6)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf("Anomalies Detected"))
    pdf.set_font("Helvetica", size=9)
    if not spikes_df_in.empty:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("Spend spikes:"))
        for _, r in spikes_df_in.head(8).iterrows():
            pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf(f"- {r['date']} | {r['channel']} | spend: ${r['spend']:,.0f}"))
    else:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("No spend spikes detected."))

    if not low_roas_df_in.empty:
        pdf.ln(2)
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("Low ROAS days:"))
        for _, r in low_roas_df_in.head(8).iterrows():
            pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf(f"- {r['date']} | spend: ${r['spend']:,.0f} | roas: {r['roas']:.2f}"))
    else:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("No low-ROAS days detected."))

    return io.BytesIO(pdf.output(dest="S").encode("latin-1")).getvalue()

def build_pdf_ttf(merged_df_in, campaign_df_in, spikes_df_in, low_roas_df_in, ttf_path):
    if not os.path.exists(ttf_path):
        raise FileNotFoundError("TTF font not found")
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font(family="DejaVu", style="", fname=ttf_path, uni=True)
    pdf.set_auto_page_break(auto=True, margin=12)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("DejaVu", size=14)
    pdf.multi_cell(usable_w, 8, _wrap_text_for_pdf("Marketing Intelligence - Summary", 100))
    pdf.ln(2)
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"Date range: {merged_df_in['date'].min()} to {merged_df_in['date'].max()}"))
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"Total Revenue: ${merged_df_in['total_revenue'].sum():,.0f}"))
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"Marketing Spend: ${merged_df_in['spend'].sum():,.0f}"))
    roas_val = merged_df_in['attributed_revenue'].sum() / (merged_df_in['spend'].sum() or np.nan)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf(f"ROAS: {roas_val:.2f}" if not np.isnan(roas_val) else "ROAS: N/A"))

    pdf.ln(4)
    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf("Top campaigns (by spend)"))
    pdf.set_font("DejaVu", size=9)
    if campaign_df_in is not None and not campaign_df_in.empty:
        top_camps = campaign_df_in.sort_values("spend", ascending=False).head(8)
        for _, r in top_camps.iterrows():
            line = f"- {str(r.get('campaign',''))[:200]} | {r.get('channel','')} | spend: ${r.get('spend',0):,.0f} | ROAS: {r.get('roas',np.nan):.2f}"
            pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf(line, wrap_width=100))
    else:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("No campaign data available."))

    pdf.ln(6)
    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(usable_w, 6, _wrap_text_for_pdf("Anomalies Detected"))
    pdf.set_font("DejaVu", size=9)
    if not spikes_df_in.empty:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("Spend spikes:"))
        for _, r in spikes_df_in.head(8).iterrows():
            pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf(f"- {r['date']} | {r['channel']} | spend: ${r['spend']:,.0f}"))
    else:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("No spend spikes detected."))

    if not low_roas_df_in.empty:
        pdf.ln(2)
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("Low ROAS days:"))
        for _, r in low_roas_df_in.head(8).iterrows():
            pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf(f"- {r['date']} | spend: ${r['spend']:,.0f} | roas: {r['roas']:.2f}"))
    else:
        pdf.multi_cell(usable_w, 5, _wrap_text_for_pdf("No low-ROAS days detected."))

    return io.BytesIO(pdf.output(dest="S").encode("latin-1")).getvalue()

# ----------------------- Prepare data for PDF -----------------------
campaign_pdf_df = None
if "campaign" in marketing_df.columns:
    campaign_pdf_df = marketing_filtered.groupby(["campaign", "channel"], as_index=False).agg({
        "spend": "sum", "attributed_revenue": "sum"
    })
    campaign_pdf_df["roas"] = campaign_pdf_df["attributed_revenue"] / campaign_pdf_df["spend"].replace(0, np.nan)

# ----------------------- PDF generation button -----------------------
if st.sidebar.button("Generate PDF Summary"):
    try:
        if os.path.exists(FONT_PATH):
            pdf_bytes = build_pdf_ttf(merged_filtered, campaign_pdf_df, spend_spikes, low_roas_days, FONT_PATH)
        else:
            pdf_bytes = build_pdf_latin(merged_filtered, campaign_pdf_df, spend_spikes, low_roas_days)
        st.sidebar.download_button("Download PDF", data=pdf_bytes, file_name="marketing_summary.pdf", mime="application/pdf")
        st.sidebar.success("PDF prepared. Click Download PDF to save.")
    except Exception as e:
        st.sidebar.error(f"PDF generation error: {e}")
# ----------------------- End -----------------------