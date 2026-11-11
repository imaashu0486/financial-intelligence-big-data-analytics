# Financial Terminal & Intelligence Platform — User Guide

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  

---

## 1. Terminal Navigation Architecture

The web interface is organized into **11 specialized interactive modules** accessible through the left-hand sidebar navigation:

```text
├── Executive Terminal (Home.py)
│
├── 📊 SYNTHETIC BASELINE MODULES (Steps 1–4)
│   ├── 1_📊_Market_Overview.py
│   ├── 2_⚡_Big_Data_Analytics.py
│   ├── 3_🤖_Machine_Learning.py
│   ├── 4_📰_Sentiment_Intelligence.py
│   ├── 5_🔗_Association_Rules.py
│   └── 6_📑_Financial_RAG.py
│
└── 📈 REAL MARKET INTELLIGENCE MODULES (Steps 5–9)
    ├── 7_📈_Real_Financial_Analytics.py
    ├── 8_🌐_Market_Regime_&_Breadth.py
    ├── 9_🎯_Stock_Screening_&_Rankings.py
    ├── 10_🔬_Real_ML_Models.py
    └── 11_🗞️_Real_News_&_Filings.py
```

### Dataset Scope Switcher
In the sidebar, users can toggle between:
- **`Real Market History (NIFTY 50)`:** Activates the 10-year historical dataset across all 50 NSE constituent blue chips.
- **`Synthetic Baseline (10 Equities)`:** Activates the original 10-stock simulated benchmark.

---

## 2. Walkthrough of Real Market Modules

### Module 7: Real Financial Analytics (`7_📈_Real_Financial_Analytics.py`)
- **Key Capabilities:**
  - View arithmetic annualized return ($\mu_{\text{daily}} \times 252$), geometric CAGR, annualized volatility, Sharpe ratio, CAPM Beta, and Jensen's Alpha for all 50 NIFTY equities.
  - Interactive Scatter Plot: Risk vs Return with Sharpe ratio color gradients.
  - Interactive 50×50 Correlation Heatmap: Inspect pairwise co-movements across Indian banking, IT, energy, and automotive sectors.
  - Single Stock Deep-Dive: Candlestick price charts with 20/50/200 SMA overlays and historical drawdown charts.

### Module 8: Market Regime & Breadth (`8_🌐_Market_Regime_&_Breadth.py`)
- **Key Capabilities:**
  - Daily Advance-Decline Ratio timeline with 20-day moving average smoothing.
  - Market Participation Waves: Tracks percentage of constituents trading above their 50-day and 200-day moving averages.
  - Regime Classification: Visualizes market periods categorized into **Expansion** ($>80\%$ above MA200), **Stress** ($<20\%$ above MA200), and **Neutral**.
  - COVID-19 March 2020 Deep Dive: Inspects the severe market crash where participation dropped to 2% (only 1 stock above MA200).

### Module 9: Stock Screening & Multi-Factor Rankings (`9_🎯_Stock_Screening_&_Rankings.py`)
- **Key Capabilities:**
  - Interactive Multi-Factor Composite Scoring: Users adjust weights for Sharpe Ratio, Low Volatility, Drawdown Defense, Jensen's Alpha, and Consistency using sliders.
  - Historical Min-Max Normalization: Immediately recalculates and re-ranks all 50 stocks based on customized historical factor weights (academic screening tool, not automated trading advice).
  - Filtering by Beta range, Sharpe threshold, and minimum return.
  - One-click CSV Export of customized ranking tables.

### Module 10: Real Machine Learning Models (`10_🔬_Real_ML_Models.py`)
- **Key Capabilities:**
  - Performance Benchmarks: Displays chronological test evaluation (unseen 2024 data, 12,250 samples) for Logistic Regression, Random Forest, and Gradient Boosting against the majority-class baseline.
  - Return Regression Metrics: Evaluates Ridge, Random Forest, and HistGradientBoosting models with MAE, RMSE, conventional $R^2$, and relative improvement over the zero-return baseline.
  - Feature Importance Tree: Highlights top explanatory drivers (`benchmark_daily_return`, `relative_return_vs_nifty`, `daily_adjusted_return`).
  - Interactive Feature Prediction Simulator: Users input technical features (return, volatility, moving average ratio) to evaluate single-sample model directional probabilities (`UP` vs `DOWN/FLAT`).

### Module 11: Real News Intelligence & Filings (`11_🗞️_Real_News_&_Filings.py`)
- **Key Capabilities:**
  - 2024 Corporate News Disclosures: Browse categorized headlines (Earnings, Acquisitions, Governance, Macro) with lexicon sentiment scores.
  - Return Association Analysis: Displays empirical correlation between sentiment and stock returns, including formal non-causality disclosure ($r_{\text{same}}=0.0888, p=0.4751; r_{\text{next}}=-0.1471, p=0.2348$).
  - Verified Annual Report MD&A Extracts: Read executive disclosures for Reliance Industries, TCS, HDFC Bank, and Infosys from FY24 annual reports.

---

## 3. Walkthrough of Synthetic Baseline Modules

- **Page 1 (Market Overview):** OHLCV candlestick charts, volume spikes, and returns for 10 simulated equities.
- **Page 2 (Big Data Analytics):** Hadoop MapReduce volume totals and Spark Parquet aggregations.
- **Page 3 (Machine Learning):** Synthetic classification, regression, and K-Means risk clustering ($K=3$).
- **Page 4 (Sentiment Intelligence):** Financial NLP lexicon analysis over 500 news items.
- **Page 5 (Association Rule Mining):** Apriori market-basket rules for equity co-movements.
- **Page 6 (Financial Document RAG):** Semantic search and extractive question answering with citations over 3 synthetic corporate disclosures.

