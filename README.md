# Marketing_Intelligence_Dashboard
# 📊 Marketing Intelligence Report — Streamlit Dashboard

**Purpose**  
Interactive BI dashboard that links campaign-level marketing activity (Facebook, Google, TikTok) to daily business outcomes. Designed to answer where marketing spend goes, which channels are efficient, and whether marketing drives profitable growth.

---

## Features

- Loads and cleans 4 CSVs from `datasets/`: `Facebook.csv`, `Google.csv`, `TikTok.csv`, `business.csv`.
- Aggregates campaign-level marketing data to daily totals and joins with business daily metrics.
- Derived KPIs: CTR, CPC, CPM, ROAS, CAC, AOV, Profit Margin.
- Interactive filters: date range and channel.
- Visuals: KPI cards, time-series (Spend vs Revenue vs Profit), channel comparison, conversion funnel, campaign leaderboard.
- Export merged dataset as CSV.

---

## Repo layout

project/
├── streamlit_bi_dashboard.py
├── requirements.txt
├── environment.yml # optional conda env
└── datasets/
├── Facebook.csv
├── Google.csv
├── TikTok.csv
└── business.csv


---

## ⚙️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/marketing_intelligence_dashboard.git
cd marketing_intelligence_dashboard

. Create Python environment
🔹 Option A: Using venv
bash
# create virtual environment
python -m venv .venv

# activate environment
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate

3. Install dependencies
bash
pip install -r requirements.txt
▶️ Run Locally
Make sure your CSVs are in the datasets/ folder.

bash
streamlit run streamlit_bi_dashboard.py
Open in your browser at:
👉 http://localhost:8501

🌍 Deployment on Streamlit Cloud
Push this repo to GitHub.

Go to Streamlit Cloud.

Click "New app" → connect your GitHub repo.

Select:

Repository: your-username/marketing_intelligence_dashboard

Branch: main

File path: streamlit_bi_dashboard.py

Click Deploy 🎉

Your dashboard will be live and shareable!

📖 Business Context
This dashboard helps business stakeholders understand how marketing activity connects with business outcomes:

Business KPIs: Revenue, Profit, Orders, Customers.

Marketing efficiency: Spend, ROAS, CAC, CTR, CPC.

Channel comparisons: Which platform drives efficient growth?

Funnel leakage: From impressions → clicks → orders.

Campaign drilldown: Identify underperforming vs high-ROI campaigns.

It applies product thinking by surfacing actionable insights, not just raw numbers.

📊 Example Insights
Facebook drives highest reach (impressions) but lowest ROAS.

TikTok shows strong CTR but high CAC.

Google campaigns deliver the best balance of scale and efficiency.

Profit margin dips when spend spikes without proportional attributed revenue.

🛠️ Tech Stack
Streamlit — interactive dashboard framework

Pandas — data processing

NumPy — numeric operations

Plotly — charts and visualizations

📌 Requirements
See requirements.txt:

text
streamlit>=1.36.0
pandas>=2.0.0
numpy>=1.25.0
plotly>=5.20.0
📜 License
This project is for educational & assessment purposes.
