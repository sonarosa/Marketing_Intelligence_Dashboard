# Marketing Intelligence Dashboard  

**Hosted demo:**  
👉 [Streamlit App](https://marketingintelligencedashboard-amkxye6f6mqkaum3mkd3ap.streamlit.app/)  

---

## What We Built  

We designed and developed an **interactive BI dashboard** in **Streamlit** that connects **campaign-level marketing data** (Facebook, Google, TikTok) with **daily business performance data**.  

This dashboard helps stakeholders **see where marketing spend goes, which channels are efficient, and whether spend drives profitable growth**.  

---

## Problem Statement  

Given **4 CSV datasets** capturing **120 days of daily activity**:  

- `Facebook.csv`, `Google.csv`, `TikTok.csv` → campaign-level marketing data:  
  *(date, tactic, state, campaign, impressions, clicks, spend, attributed revenue)*  
- `business.csv` → business performance data:  
  *(date, orders, new_orders, new_customers, total_revenue, gross_profit, COGS)*  

**Objective:**  
- Merge & clean datasets  
- Aggregate campaign → daily totals by channel  
- Derive efficiency metrics (ROAS, CAC, AOV, CTR, CPC, CPM, Profit Margin)  
- Build interactive dashboard with **filters, KPIs, charts, anomaly alerts, exports**  
- Deploy as a **hosted Streamlit app**  

---

## Features  

- **Automatic data load** from `datasets/` (no uploads needed)  
- **Derived KPIs**: CTR, CPC, CPM, ROAS, CAC, AOV, Profit Margin  
- **Interactive filters**:  
  - Date range  
  - Channel  
  - State (if available)  
  - Campaign substring search  
- **Visualizations**:  
  - KPI cards  
  - Time-series (Spend vs Revenue vs Profit)  
  - Channel comparison (bar + scatter)  
  - Conversion funnel (Impressions → Clicks → Orders)  
  - ROAS by state heatmap *(if data has `state` column)*  
- **Anomaly alerts**:  
  - Spend spikes (std. deviation based threshold)  
  - Low ROAS days  
- **Campaign leaderboard** ranked by spend and ROAS  
- **Export merged dataset** as CSV  

---

## Repository Layout  
Marketing_Intelligence_Dashboard/
├── app.py                # main app
├── requirements.txt      # dependencies
├── environment.yml       # optional conda env
├── README.md
└── datasets/
    ├── Facebook.csv
    ├── Google.csv
    ├── TikTok.csv
    └── business.csv

##  Setup Instructions

### 2. Create Python Environment

**Option A — venv**
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate
```
### 3. Create Conda Environment
**Option B — conda**
```bash
conda env create -f environment.yml
conda activate marketing_dashboard
```
### 4. Create Conda Environment
**Create requirements.txt**
```bash
pip install -r requirements.txt
```
```bash
streamlit>=1.36.0
pandas>=2.0.0
numpy>=1.25.0
plotly>=5.20.0
```

### 5. Place CSVs

Ensure files exist in `datasets/`:
datasets/Facebook.csv
datasets/Google.csv
datasets/TikTok.csv
datasets/business.csv

---

### 6. Run Locally

```bash
streamlit run app.py
```

### 7. Deployment (Streamlit Cloud)

1. Push this repo to GitHub:
```bash
1. git add .
2. git commit -m "initial commit"
3. git push origin main
```

2. Go to Streamlit Cloud

3. Click New app → select:

Repository: <your-username>/Marketing_Intelligence_Dashboard

Branch: main

File path: streamlit_bi_dashboard.py

4. Click Deploy
5. Your app will get deployed successfully in Streamlit
