# Financial Intelligence & Big Data Analytics Platform

A distributed big data processing, financial analytics, and machine learning platform built for the National Stock Exchange of India (NSE) NIFTY 50 equity universe across a 10-year historical period (2015–2024).

The platform integrates multi-node distributed batch compute (Hadoop HDFS, MapReduce, YARN, and Apache Spark) with financial econometrics, leak-free machine learning, natural language processing, market basket mining, and an offline document retrieval pipeline, presented through an interactive Streamlit terminal.

---

## 1. Academic Context

- **Course:** CSBB 422 — Big Data Analytics Lab
- **Institution:** National Institute of Technology Delhi
- **Department:** Computer Science and Engineering
- **Academic Session:** 2026–2027
- **Target Market:** National Stock Exchange of India (NSE) — NIFTY 50 Universe
- **Project Type:** Laboratory Capstone Implementation

---

## 2. Overview

Financial market data is characterized by high volume, low signal-to-noise ratios, non-stationarity, and cross-asset correlations. Standard single-machine analytical setups often struggle when ingesting multi-year historical panels, enforcing strict temporal boundaries, or scaling computationally intensive econometric workloads like multi-factor screening and rolling drawdown analysis across large portfolios.

This project was built to address these challenges by implementing an end-to-end big data analytics pipeline:

1. Ingesting, cleaning, and validating 10 calendar years (2015–2024) of daily market history for all 50 constituent equities of the NIFTY 50 index and the official `^NSEI` benchmark.
2. Ingesting processed market features into a containerized distributed Hadoop/Spark cluster running on Hadoop Distributed File System (HDFS) with replication factor 2.
3. Executing distributed batch aggregations using Hadoop Streaming MapReduce and Apache Spark DataFrames managed by YARN and Spark Standalone.
4. Computing financial risk and return metrics, tail-risk indicators (VaR, CVaR), CAPM beta, Jensen's alpha, multi-factor percentile rankings, and market breadth indicators.
5. Evaluating machine learning models (classification and regression) using strictly chronological train/validation/test partitions with boundary purging to guarantee zero look-ahead leakage.
6. Processing 2024 corporate disclosures and annual report MD&A extracts using domain-adapted financial sentiment lexicons and an offline extractive retrieval-augmented generation (RAG) system.
7. Serving all analytical data products through an interactive, multipage Streamlit financial terminal supporting dual-scope inspection: Real Market History (50 stocks) and Synthetic Baseline (10 stocks).

---

## 3. Key Features

- **10-Year Real Market History (2015–2024):** 121,962 processed daily records covering all 50 constituent equities of the NIFTY 50 index, plus 2,459 official benchmark sessions (`^NSEI`).
- **Distributed Big Data Compute:** Containerized multi-node Hadoop 3.3.6 and Apache Spark 3.5.1 cluster with HDFS block storage ($R=2$), YARN resource management, and Spark DataFrame transformations.
- **Financial Econometric Engine:** Arithmetic and geometric returns, annualized volatility, Sharpe ratios ($R_f=6.0\%$ Indian G-Sec baseline), Sortino ratios, Calmar ratios, historical maximum drawdowns with peak/trough dates, Value at Risk (VaR 95%, 99%), and Expected Shortfall (CVaR 95%).
- **Benchmark-Relative Alpha & Beta:** CAPM Beta, dual-beta (up-market and down-market beta), Jensen's Alpha, Tracking Error, Information Ratio, and full $50 \times 50$ Pearson correlation matrices.
- **Market Breadth & Regime Classification:** Daily advance/decline ratios, constituent participation waves (% of stocks above 20-day, 50-day, and 200-day moving averages), and benchmark trend/volatility regime tagging across 2,467 observation dates.
- **Transparent Multi-Factor Screener:** Configurable composite factor ranking based on Sharpe ratio, volatility mitigation, drawdown defense, Jensen's alpha, and consistency of positive trading days.
- **Leakage-Free Machine Learning:** Chronological train/validation/test partitions with boundary purging, feature scaler isolation on training partitions, and EMH baseline comparisons.
- **Corporate Disclosure Sentiment Analysis:** Domain-adapted lexicon scoring of 2024 corporate news announcements with linear return correlation analysis against same-day ($r = +0.0888, p = 0.4751$) and next-day returns ($r = -0.1471, p = 0.2348$). Neither establishes nor rules out causality.
- **Market Basket Co-Movement Mining:** Apriori association rule mining discovering directional stock co-movements with support, confidence, and lift metrics.
- **Offline Extractive Document RAG:** Semantic vector retrieval using `sentence-transformers/all-MiniLM-L6-v2` and offline extractive answer synthesis over corporate annual report MD&A extracts with full citation attribution.
- **Dual-Scope Financial Terminal:** Multipage Streamlit application featuring 11 specialized dashboard pages with dynamic scope switching between Real Market History and Synthetic Baseline.

---

## 4. System Architecture

The platform architecture separates data ingestion, distributed processing, analytics, and presentation layers on a single physical host:

```text
+-----------------------------------------------------------------------------------+
|                        PHYSICAL HOST (Windows 11 + WSL2)                          |
+-----------------------------------------------------------------------------------+
                                          |
   +--------------------------------------+------------------------------------+
   |                                                                           |
   v                                                                           v
[1. Local Ingestion & Storage]                              [2. Docker Cluster (Bridge Net)]
- Raw Market Data (50 Tickers, 10 Years)                    - hadoop-master (NameNode, RM,
- Benchmark (^NSEI, 2,459 Sessions)                           Spark Master)
- 2024 Corporate News (79 Records)                          - hadoop-worker1 (DataNode, NM,
- FY24 MD&A Extracts (4 Companies)                            Spark Worker)
- Feature Engineering (14 Technical Factors)                - hadoop-worker2 (DataNode, NM,
- Parquet & CSV Feature Storage                               Spark Worker)
                                                            - financial-mongodb (MongoDB 7.0)
                                                                       |
   +-------------------------------------------------------------------+
   |
   v
[3. Distributed Batch Jobs]
- HDFS Ingestion (/data/real/market/): Replication Factor R=2
- MapReduce Job 1: Ticker Volume & Session Counter (Streaming Python)
- MapReduce Job 2: Cross-Ticker Return Distribution
- Spark Job: Stock Statistics, Daily Volume, Risk/Return Rankings (Parquet)
   |
   +-------------------------------------------------------------------+
   |                                                                   |
   v                                                                   v
[4. Analytical & ML Engines]                                [5. Streamlit Dark Terminal]
- Real Econometrics (VaR, Sharpe, Beta, Drawdowns)          - Home.py (Cluster Health & Scope)
- Real ML Pipeline (Ridge, RF, LogReg, GBDT)                - 11 Specialized Analytical Pages
- NLP Sentiment Engine (Lexicon Scoring)                    - Dual-Scope State Routing
- Offline Extractive RAG (Dense Vector Retr.)               - Interactive Plotly Visualizations
- Apriori Association Rules (Market Baskets)                - Filterable Data Tables & CSV Export
```

### Architectural Transparency Note
The distributed cluster consists of 4 Docker containers running on a single physical Windows computer via WSL2. The containers communicate over an internal Docker bridge network (`financial_hadoop_network`). This deployment models distributed service interaction, RPC protocols, multi-node YARN scheduling, and HDFS replication, but shares underlying physical hardware resources (CPU, RAM, and NVMe disk I/O).

---

## 5. Technology Stack

| Component | Technology | Version | Actual Project Role |
| :--- | :--- | :--- | :--- |
| **Runtime Language** | Python | `3.12.6` (64-bit) | Application core, analytics algorithms, data processing, and test suite |
| **Distributed Storage** | Apache Hadoop HDFS | `3.3.6` | Distributed block storage with replication factor $R=2$ across two DataNodes |
| **Cluster Scheduler** | Apache Hadoop YARN | `3.3.6` | Distributed container resource allocation and MapReduce execution |
| **Distributed Compute** | Apache Spark | `3.5.1` (bin-hadoop3) | In-memory distributed DataFrame processing, aggregations, and Parquet generation |
| **Python Spark API** | PySpark | `3.5.1` | Local and cluster Spark execution driver |
| **JVM Runtime** | OpenJDK | `11.0.22` | Java Virtual Machine environment inside Hadoop/Spark Docker containers |
| **NoSQL Database** | MongoDB Community | `7.0` | Containerized document store for corporate news, disclosures, and RAG cache |
| **User Interface** | Streamlit | `1.30+` | Multipage interactive financial terminal with dark-theme styling |
| **Charting Engine** | Plotly | `5.18+` | Interactive candlestick charts, correlation heatmaps, and distribution plots |
| **Machine Learning** | scikit-learn | `1.4+` | Chronological regressors, classifiers, K-Means clustering, and metrics |
| **Vector Embeddings** | sentence-transformers | `2.3+` | Sentence transformer (`all-MiniLM-L6-v2`) for dense document retrieval |
| **Data Processing** | pandas / numpy / scipy | `2.0+` / `1.24+` / `1.11+` | Local feature engineering, time-series operations, and statistical tests |
| **Test Framework** | pytest | `7.0+` | Automated regression, unit, and integration testing (290 tests) |

---

## 6. Data Sources & Dataset Details

### Real Market History (NIFTY 50 Universe)
- **Asset Universe:** All 50 constituent equities of the National Stock Exchange of India (NSE) NIFTY 50 Index as of the 2024 roster.
- **Historical Horizon:** 10 full calendar years, covering **2015-01-01 to 2024-12-31**.
- **Equities Record Count:** Exactly **121,962 daily records** post-cleaning and validation.
- **Benchmark Index:** NIFTY 50 Index (`^NSEI`), containing **2,459 official NSE trading sessions**.
- **Observation Reconciliation:** Individual equities contain up to 2,467 distinct observation dates. The 8 dates not present in the Yahoo Finance `^NSEI` index feed were reconciled against official NSE trading circulars (e.g., special Muhurat trading sessions and exchange-cleared calendar dates). All benchmark-relative calculations use an inner join of 2,459 aligned sessions.

### Real News & Corporate Filings
- **News Corpus:** 79 verified corporate announcements from CY 2024 across constituent equities; 67 announcements align with active trading sessions for price impact and correlation testing (12 occurred on weekends or non-trading exchange holidays).
- **Annual Report Extracts:** 4 verified Management Discussion & Analysis (MD&A) and governance text extracts for FY24 (Reliance Industries, Tata Motors, HDFC Bank, Infosys) totaling ~10,000 words.

### Synthetic Baseline Dataset
- **Asset Universe:** 10 simulated tickers (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `TSLA`, `NVDA`, `JPM`, `V`, `JNJ`).
- **Record Count:** 5,200 daily records across 519 trading days (Phase 1–4 foundational baseline).

---

## 7. Data Processing Workflow

```text
[Raw Data Acquisition]
  - 50 NIFTY 50 equity CSVs + ^NSEI benchmark CSV
  - Automated identity audit reconciling ticker symbols against NSE master list
       │
       ▼
[Data Cleaning & Verification]
  - Null handling, duplicate removal, chronological sorting
  - Adjustment for stock splits and bonus issues using adj_close
  - Data integrity checks (high >= low, volume >= 0)
       │
       ▼
[Feature Engineering (14 Technical Factors)]
  - Daily adjusted returns, 5d/20d/60d momentum returns
  - 20d/60d annualized volatility
  - Price ratios to 20d/50d/200d Simple Moving Averages (SMA)
  - Normalized volume ratio (Volume / 20d SMA Volume)
  - Intraday high-low percentage range (hl_pct_range)
  - Rolling 60-day maximum drawdown
  - Systematic benchmark daily return and relative idiosyncratic return
       │
       ▼
[Storage & Cluster Ingestion]
  - Local storage: consolidated_market_features.parquet (25 MB) and CSV (53 MB)
  - HDFS upload: /data/real/market/ via WebHDFS / docker exec (Replication = 2)
       │
       ▼
[Distributed Compute & Analytical Export]
  - Hadoop MapReduce: volume aggregates and session distribution
  - Apache Spark: stock statistics, daily market totals, rankings
  - Analytical Parquet/CSV exports consumed by Streamlit dashboard
```

---

## 8. Hadoop and Spark Implementation

### Containerized Topology
The cluster runs via Docker Compose with dedicated service definitions:
- **`hadoop-master`:** NameNode (port 9870 Web UI, 9000 RPC), YARN ResourceManager (port 8088 Web UI), JobHistory Server (port 19888), Spark Master (port 8080 Web UI, 7077 RPC).
- **`hadoop-worker1`:** DataNode 1 (port 9864), NodeManager 1 (port 8042), Spark Worker 1 (port 8081).
- **`hadoop-worker2`:** DataNode 2 (host port 9865), NodeManager 2 (host port 8043), Spark Worker 2 (host port 8082).
- **`financial-mongodb`:** MongoDB 7.0 daemon (port 27017).

### HDFS Storage Configuration
HDFS is configured with block replication factor $R=2$ (`HDFS_REPLICATION=2`), matching the two worker DataNodes. Consolidated market features are ingested to `/data/real/market/consolidated_market_features.parquet`.

### MapReduce Batch Jobs
- **Job 1 (Volume & Session Aggregation):** Streaming MapReduce job with mapper parsing `(symbol, volume)` and reducer aggregating total 10-year traded volume and session counts per ticker across 121,962 records. Output: `job1_volume_by_ticker.tsv` (50 stocks).
- **Job 2 (Return Distribution Aggregator):** Aggregates daily return bins across equities to evaluate overall market distribution.

### Apache Spark DataFrame Jobs
PySpark jobs running on Spark Standalone / YARN execute distributed transformations:
- Ingestion of Parquet datasets with schema enforcement.
- Per-ticker aggregation: daily mean return, standard deviation, annualized return, annualized volatility, and session counts.
- Daily market aggregation: total daily market volume across 2,467 observation dates.
- Stock ranking tables written as optimized Parquet partitions to `data/real/distributed/spark/`.

---

## 9. Financial Analytics

All financial econometrics use the Indian 10-Year Government Securities benchmark yield ($R_f = 6.0\%$ annualized) as the risk-free rate:

### Implemented Risk & Return Formulas
- **Arithmetic Annualized Return:** $\bar{R}_{\text{ann}} = \mu_{\text{daily}} \times 252$
- **Compound Annual Growth Rate (CAGR):** $\text{CAGR} = \left(\frac{P_{\text{end}}}{P_{\text{start}}}\right)^{\frac{252}{N}} - 1$
- **Annualized Volatility:** $\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}$
- **Sharpe Ratio:** $\text{Sharpe} = \frac{\bar{R}_{\text{ann}} - 0.060}{\sigma_{\text{ann}}}$ (computed for both $R_f = 6\%$ and $R_f = 0\%$)
- **Sortino Ratio:** $\text{Sortino} = \frac{\bar{R}_{\text{ann}} - 0.060}{\sigma_{\text{downside}}}$, where downside deviation is computed relative to the daily risk-free target ($0.060 / 252$).
- **Maximum Drawdown (MDD):** $\text{MDD} = \min_{t} \left(\frac{P_t - \max_{s \le t} P_s}{\max_{s \le t} P_s}\right)$, evaluated on split/bonus-adjusted prices (`adj_close`) with peak and trough dates.
- **Calmar Ratio:** $\text{Calmar} = \frac{\bar{R}_{\text{ann}}}{|\text{MDD}|}$
- **Value at Risk (Historical VaR):** Empirical 5th percentile ($\text{VaR}_{95\%}$) and 1st percentile ($\text{VaR}_{99\%}$) of daily returns.
- **Conditional VaR (CVaR / Expected Shortfall):** Mean loss conditional on return falling below $-\text{VaR}_{95\%}$.

### Benchmark-Relative Metrics (vs NIFTY 50 Index)
- **CAPM Beta ($\beta$):** $\beta = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$ across 2,459 aligned sessions.
- **Dual Beta:** Up-market beta ($\beta_{\text{up}}$, computed when $R_m > 0$) and down-market beta ($\beta_{\text{down}}$, computed when $R_m < 0$).
- **Jensen's Alpha ($\alpha$):** $\alpha = (\bar{R}_i - R_f) - \beta (\bar{R}_m - R_f)$
- **Tracking Error:** Annualized standard deviation of $(R_i - R_m)$.
- **Information Ratio:** Annualized excess return divided by Tracking Error.
- **Correlation Matrix:** Complete $50 \times 50$ Pearson correlation matrix (2,500 pairs) evaluated across common dates.

### Market Breadth & Regime Classification
- **Daily Advance/Decline Ratio:** Proportion of stocks with positive vs negative daily returns.
- **Moving Average Participation:** Percentage of constituents trading above their respective 20-day, 50-day, and 200-day SMAs.
- **Trend Regimes:** Bull ($\text{Close} > \text{SMA}_{200}$ and $\text{SMA}_{50} > \text{SMA}_{200}$), Bear ($\text{Close} < \text{SMA}_{200}$ and $\text{SMA}_{50} < \text{SMA}_{200}$), or Transition.
- **Volatility Regimes:** Low ($\sigma_{\text{ann}} < 12\%$), Normal ($12\% \le \sigma_{\text{ann}} \le 20\%$), or High ($\sigma_{\text{ann}} > 20\%$).

### Representative Constituent Metrics (2015–2024 Verified Results)

| Ticker | Annualized Return ($\mu_{\text{ann}}$) | Annualized Volatility ($\sigma_{\text{ann}}$) | Sharpe Ratio ($R_f=6\%$) | CAPM Beta ($\beta$) | Jensen's Alpha ($R_f=6\%$) | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADANIENT** | 51.56% | 54.11% | 0.84 | 1.53 | +36.09% | -82.68% |
| **BAJFINANCE** | 37.28% | 36.23% | 0.86 | 1.33 | +23.43% | -60.40% |
| **INFY** | 19.91% | 26.51% | 0.52 | 0.80 | +8.94% | -38.25% |
| **HDFCBANK** | 16.82% | 22.70% | 0.48 | 1.00 | +4.53% | -38.71% |
| **ITC** | 13.46% | 25.15% | 0.30 | 0.74 | +2.65% | -53.47% |

---

## 10. Machine Learning

### Problem Formulations
The evaluated machine learning pipeline addresses two distinct predictive tasks with separate metrics and objectives:
1. **Binary Direction Classification (Categorical):** Predict whether the next-day adjusted close will rise ($r_{t+1} > 0$, class 1) or fall/remain flat ($r_{t+1} \le 0$, class 0). Evaluated using Directional Accuracy, ROC-AUC, Balanced Accuracy, Precision, Recall, and F1 Score.
2. **Continuous Return Magnitude Regression (Continuous):** Predict the exact continuous percentage return $r_{t+1}$. Evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Conventional $R^2$ (variance explained relative to test sample mean), and Relative MSE Reduction over the Zero-Return Baseline ($\Delta\text{MSE}_{\text{zero}}$).

*Note on Task Separation:* Classification accuracy/ROC-AUC and regression $R^2$ measure fundamentally different targets and mathematical losses; they evaluate complementary aspects of predictability and must not be conflated.

### Strict Anti-Leakage Protocol
- **Chronological Splits:**
  - Training: 2015-03-31 to 2021-12-30 (81,619 samples across 50 equities).
  - Validation: 2022-01-03 to 2023-12-28 (24,600 samples across 50 equities).
  - Holdout Test: 2024-01-01 to 2024-12-30 (12,250 unseen test samples across 50 equities; 245 trading days $\times$ 50 stocks).
- **Boundary Purging:** The final observation row of each partition is purged because its next-day target extends into the subsequent partition.
- **Scaler Isolation:** Feature scalers (`StandardScaler`) are fitted strictly on the Training partition only.
- **Dimensionless Feature Space:** Nominal price levels (`close`, `open`) and ticker symbols are excluded from model inputs. Features consist exclusively of stationary ratios and momentum factors.

### Holdout Test Set Results (CY 2024, 12,250 Samples)

#### Classification Models

| Model | Test Accuracy | Balanced Acc | Precision | Recall | F1 Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 50.64% | 50.00% | 50.64% | 100.00% | 0.6723 | 0.5000 |
| **Logistic Regression (L2)** | **50.74%** | 50.53% | 51.04% | 66.90% | 0.5790 | **0.5167** |
| **Random Forest Classifier** | 50.61% | 50.50% | 50.61% | 100.00% | 0.6721 | **0.5134** |
| **HistGradientBoosting Classifier** | 50.70% | **50.72%** | **51.38%** | 46.85% | 0.4899 | 0.5049 |

#### Regression Models

| Model | Test MAE | Test RMSE | Conventional $R^2$ | Relative vs Zero ($\Delta\text{MSE}_{\text{zero}}$) | Directional Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Return Baseline** | 0.011838 | 0.017141 | -0.001334 | 0.000000 | 50.00% |
| **Ridge Regression** | 0.011834 | 0.017110 | +0.002290 | +0.003618 | **51.10%** |
| **Random Forest Regressor** | **0.011823** | **0.016877** | **+0.029265** | **+0.030558** | 50.58% |
| **HistGradientBoosting Regressor** | 0.012029 | 0.017328 | -0.023394 | -0.022031 | 49.74% |

### Academic Assessment of Predictive Bounds
The empirical models show limited predictive capability on the unseen 2024 holdout dataset (classification accuracy ~50.7%, ROC-AUC ~0.517; regression conventional $R^2 \approx 0.029$). These findings align with the Weak-Form Efficient Market Hypothesis (EMH). Crucially, no reliable trading edge has been demonstrated by these models. Execution frictions such as Securities Transaction Tax (STT), brokerage commissions, exchange turnover charges, and market impact slippage were not modeled in this feature pipeline; rigorous transaction-cost-aware backtesting would be required to evaluate whether any theoretical statistical signal possesses practical economic profitability.

---

## 11. News and Sentiment Analysis

- **Coverage:** 79 verified corporate announcements published during calendar year 2024 across NIFTY 50 constituents.
- **Trading Alignment:** 67 announcements align directly with active exchange trading sessions; 12 occurred on weekends or market holidays.
- **Sentiment Scoring:** Domain-adapted financial lexicon analyzer based on Loughran-McDonald terminology, classifying disclosures into Positive, Neutral, or Negative polarity with confidence scores.
- **Event Categorization:** Rule-based classification into Earnings/Results, Capital Raising/Capex, Contracts/Expansion, Corporate Action/Regulatory, or General Announcements.
- **Linear Correlation Analysis:** Pearson correlation between headline sentiment polarity scores and observed equity returns yielded:
  - Same-day return correlation: $r_{\text{same}} = +0.0888$ ($p = 0.4751, N = 67$).
  - Next-day return correlation: $r_{\text{next}} = -0.1471$ ($p = 0.2348, N = 67$).
- **Academic Findings & Non-Causality Clarification:** No statistically significant linear correlation was detected in the observed sample between headline news polarity and daily equity returns. This statistical result demonstrates an absence of linear association at the daily frequency in this specific dataset; however, it does not establish or rule out causality, nor does it preclude non-linear relationships or intraday price reactions that require tick-level order book data to examine.

---

## 12. Association Rule Mining

- **Transaction Construction:** Daily return matrices are converted into transaction baskets where each date forms a transaction. An equity symbol is tagged with `_UP` if its daily return exceeds a threshold (e.g., $+1.0\%$), and `_DOWN` if it falls below $-1.0\%$.
- **Apriori Algorithm:** Mines frequent itemsets satisfying a minimum support threshold ($s \ge s_{\min}$) and generates directional association rules ($A \implies B$).
- **Evaluated Metrics:**
  - **Support:** $P(A \cap B)$, frequency of co-movement across all trading sessions.
  - **Confidence:** $P(B \mid A)$, conditional probability that stock $B$ rises given stock $A$ rose.
  - **Lift:** $\frac{P(B \mid A)}{P(B)}$, ratio of observed co-occurrence to expected co-occurrence under independence.
  - **Conviction:** $\frac{1 - P(B)}{1 - P(B \mid A)}$, measure of rule implication strength.
- **Application:** Identifies historical sector co-movements and systemic trading patterns across market sessions.

---

## 13. Retrieval-Augmented Generation (RAG)

- **Architecture:** An offline, extractive document retrieval system. It does **not** rely on external commercial generative APIs or third-party chatbots.
- **Document Ingestion:** Reads corporate annual reports, MD&A disclosures, and financial statements from `data/documents/` (synthetic baseline) and `data/real/raw/documents/` (real FY24 extracts).
- **Chunking:** Sliding-window text chunking with token overlap (chunk size: 250–500 characters, overlap: 50 characters).
- **Embeddings:** Dense semantic embeddings generated locally via `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Vector Retrieval:** Exact cosine-similarity vector store with top-$K$ nearest-neighbor retrieval.
- **Extractive Answer Generation:** Synthesizes answers by extracting the highest-scoring supported sentences from retrieved passages, providing document and page-level source citations and confidence scores.
- **Evaluation Framework:** Retrieval performance was evaluated against annotated benchmark questions using Hit Rate@K, Precision@K, Recall@K, and Mean Reciprocal Rank (MRR). These metrics assess dense vector passage retrieval effectiveness (whether relevant context passages were correctly prioritized), rather than proving generative text correctness or model factuality.

---

## 14. Dashboard Guide

The Streamlit dashboard (`app/Home.py`) provides 11 analytical pages organized as follows:

| Page Name | Scope Support | Purpose and Capabilities |
| :--- | :---: | :--- |
| **Home.py** | Unified | Executive landing terminal, cluster status overview, global dataset scope selector (Real vs Synthetic), and navigation routing. |
| **1. 📊 Market Overview** | Real & Synthetic | Interactive candlestick charts, SMA overlays (20, 50, 200), volume indicators, Bollinger Bands, and downloadable summary data. Defaults to Real NIFTY 50 (50 stocks, 121,962 rows). |
| **2. ⚡ Big Data Analytics** | Real & Synthetic | MapReduce job telemetry, Spark distributed compute results, stock statistics, daily market volume, and risk-return scatter plots. Displays 50 stocks in Real mode and 10 in Synthetic mode. |
| **3. 🤖 Machine Learning** | Synthetic | Baseline ML model evaluation (Linear Regression, Random Forest, GBDT, K-Means clustering) on the 10-stock synthetic dataset with interactive inference. |
| **4. 📰 Sentiment Intelligence** | Synthetic | Financial news sentiment scoring, Loughran-McDonald lexicon distribution, daily sentiment regimes, and return correlations on synthetic news. |
| **5. 🔗 Association Rules** | Synthetic | Apriori market-basket mining for stock co-movement rules, interactive support/confidence/lift filters, and scatter visualizers. |
| **6. 📑 Financial RAG** | Synthetic | Grounded extractive QA over financial documents, source citations, vector retrieval inspection, and retrieval benchmark metrics. |
| **7. 📈 Real Financial Analytics** | Real | 10-year risk/return metrics, VaR/CVaR tail risk, CAPM beta, Jensen's alpha, and full $50 \times 50$ correlation heatmap across 50 NIFTY equities. |
| **8. 🌐 Market Regime & Breadth** | Real | 10-year market breadth, advance/decline ratios, constituent moving average participation waves (% above 20/50/200 DMA), and historical stress regimes. |
| **9. 🎯 Stock Screening & Rankings** | Real | Configurable multi-factor percentile ranking (Sharpe, volatility mitigation, drawdown defense, alpha, consistency) across all 50 NIFTY constituents. |
| **10. 🔬 Real ML Models** | Real | Chronological out-of-sample ML evaluation (CY 2024, 12,250 samples), ROC-AUC curves, Random Forest feature importances, and interactive feature predictor. |
| **11. 🗞️ Real News & Filings** | Real | 2024 news sentiment NLP, event category distribution, return correlation analysis ($r = 0.0888, p = 0.4751$), and FY24 corporate filing MD&A extract summaries. |

---

## 15. Project Structure

```text
C:\Projects\BigDataProject\
├── app/                                # Streamlit Financial Terminal
│   ├── Home.py                         # Executive landing page & scope manager
│   ├── components/                     # Reusable UI widgets (charts, metrics, layout)
│   ├── pages/                          # 11 analytical dashboard pages
│   └── services/                       # Data access and analytics service layers
├── config/                             # Database and cluster environment configurations
├── data/                               # Dataset repositories
│   ├── real/                           # Real NIFTY 50 dataset assets (2015–2024)
│   │   ├── raw/                        # 50 constituent CSVs, benchmark, news, documents
│   │   ├── processed/                  # consolidated_market_features (Parquet & CSV)
│   │   └── distributed/                # MapReduce and Spark cluster outputs
│   ├── sample/                         # Synthetic baseline data (10 stocks, 5,200 rows)
│   └── processed/                      # Synthetic pipeline analytical outputs
├── docker/                             # Docker cluster container definitions
│   └── cluster/                        # Dockerfile, entrypoint.sh, and Hadoop/Spark configs
├── docker-compose.yml                  # 4-container logical cluster specification
├── docs/                               # Core Academic Documentation Suite (10 files)
│   ├── CLUSTER_DEPLOYMENT_GUIDE.md     # Docker, HDFS, YARN & Spark operations
│   ├── DATA_DICTIONARY.md              # Column definitions, schemas & mathematical formulas
│   ├── FINAL_DEMONSTRATION_GUIDE.md    # 13-step lab viva script and defense Q&A
│   ├── FINAL_PROJECT_REPORT.md         # Comprehensive capstone project report
│   ├── FINANCIAL_ANALYTICS_METHODOLOGY.md # Mathematical definitions and econometric proofs
│   ├── INSTALLATION_AND_SETUP.md       # Workstation prerequisites & setup guide
│   ├── KNOWN_LIMITATIONS.md            # System, cluster, and empirical boundaries
│   ├── ML_METHODOLOGY.md               # Anti-leakage ML design & EMH analysis
│   ├── PROJECT_ARCHITECTURE.md         # Complete system and network topology
│   └── USER_GUIDE.md                   # 11-page terminal navigation manual
├── models/                             # Serialized scikit-learn model registries (.joblib)
├── notebooks/                          # Exploratory data analysis notebooks
├── scripts/                            # Orchestration and batch processing scripts
│   ├── cluster/                        # PowerShell cluster management scripts
│   ├── audit_real_data.py              # Data verification and audit script
│   ├── run_real_data_acquisition.py    # NIFTY 50 data fetcher
│   └── upload_to_hdfs.py               # HDFS dataset ingestion script
├── src/                                # Core Engine Source Code
│   ├── analytics/                      # Econometric risk/return, breadth, and screening
│   ├── bigdata/                        # MapReduce runners and PySpark cluster jobs
│   ├── database/                       # MongoDB client integration
│   ├── ingestion/                      # Market data fetchers and generators
│   ├── mining/                         # Apriori association rules implementation
│   ├── ml/                             # Real and synthetic ML pipelines
│   ├── nlp/                            # Financial sentiment lexicon analyzer
│   ├── preprocessing/                  # Data cleaners and feature engineers
│   ├── rag/                            # Offline document RAG pipeline and evaluation
│   └── utils/                          # Logging and shared helpers
├── tests/                              # Automated test suite (21 test modules)
├── .env.example                        # Template for environment configuration
├── .gitignore                          # Git exclusions for virtualenvs, dist, and caches
├── requirements.txt                    # Pinned Python package dependencies
└── README.md                           # Project documentation entry point
```

---

## 16. Installation and Setup

### Prerequisites
- **Operating System:** Windows 10/11 with WSL2 enabled.
- **Docker Desktop:** Configured with WSL2 backend and at least 4 GB RAM allocated.
- **Python:** Python 3.12 (verified on `3.12.6` 64-bit).
- **PowerShell:** Version 5.1 or PowerShell 7+.

### Setup Instructions (PowerShell)

```powershell
# 1. Clone the repository and navigate to the project root
git clone https://github.com/imaashu0486/financial-intelligence-big-data-analytics.git C:\Projects\BigDataProject
cd C:\Projects\BigDataProject

# 2. Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install package dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 17. Running the Application

To launch the interactive Streamlit terminal:

```powershell
# Activate virtual environment (if not already active)
.\.venv\Scripts\Activate.ps1

# Launch the Streamlit application
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
```

Once running, access the dashboard in your web browser at:
`http://localhost:8501`

Use the sidebar toggle to switch between **Real Market History (NIFTY 50)** (default, 50 stocks) and **Synthetic Baseline (10 Stocks)**.

---

## 18. Running Distributed Services and Jobs

### 1. Build and Start the Docker Cluster
To run the containerized Hadoop and Spark cluster, first verify that distribution archives (`hadoop-3.3.6.tar.gz` and `spark-3.5.1-bin-hadoop3.tgz`) are downloaded in `docker/cluster/dist/`:

```powershell
# Download Hadoop and Spark archives (if not already present)
powershell -ExecutionPolicy Bypass -File scripts/download_cluster_archives.ps1

# Start the cluster containers in the background
docker compose up -d

# Verify container status
docker compose ps
```

Expected running containers:
- `hadoop-master` (NameNode, ResourceManager, Spark Master)
- `hadoop-worker1` (DataNode 1, NodeManager 1, Spark Worker 1)
- `hadoop-worker2` (DataNode 2, NodeManager 2, Spark Worker 2)
- `financial-mongodb` (MongoDB 7.0)

### 2. Cluster Web Interfaces
- HDFS NameNode: [http://localhost:9870](http://localhost:9870)
- YARN ResourceManager: [http://localhost:8088](http://localhost:8088)
- Spark Master: [http://localhost:8080](http://localhost:8080)
- MapReduce JobHistory: [http://localhost:19888](http://localhost:19888)

### 3. Ingest Data to HDFS
```powershell
.\.venv\Scripts\python.exe scripts/upload_to_hdfs.py
```

### 4. Execute Cluster Jobs
```powershell
# Execute MapReduce jobs inside the cluster
powershell -ExecutionPolicy Bypass -File scripts/cluster/run_mapreduce_test.ps1

# Execute distributed PySpark analytics job
powershell -ExecutionPolicy Bypass -File scripts/cluster/run_financial_analytics.ps1
```

### 5. Stop the Cluster
```powershell
docker compose down
```

---

## 19. Testing

The repository contains an automated test suite covering unit tests, data cleaning, feature engineering, distributed pipelines, machine learning anti-leakage, NLP sentiment, RAG retrieval, and frontend service isolation.

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run the standard non-Spark test suite (282 tests)
.\.venv\Scripts\python.exe -m pytest tests --ignore=tests/test_spark_jobs.py -v

# Run the real ML anti-leakage tests (6 tests, included in the 282)
.\.venv\Scripts\python.exe -m pytest tests/test_real_ml_pipeline.py -v

# Run frontend services and regression tests (30 tests, included in the 282)
.\.venv\Scripts\python.exe -m pytest tests/test_frontend_services.py -v

# Run PySpark distributed DataFrame tests (8 tests)
# Note: Requires local socket permissions or execution within cluster container
.\.venv\Scripts\python.exe -m pytest tests/test_spark_jobs.py -v
```

### Verified Test Counts
- **Non-Spark Test Suite:** 282 tests passed in ~23 seconds.
- **PySpark Test Suite:** 8 tests passed in ~119 seconds.
- **Total Unique Tests:** **290 tests collected, 290 passed, 0 failed, 0 skipped (100% pass rate)**.

---

## 20. Results and Implementation Details

- **Dataset Verification:** Reconciled 50 NSE constituent ticker symbols, 121,962 records, 2,467 distinct observation dates, and 2,459 benchmark trading sessions without interpolation or forward-filling of non-trading dates.
- **Cluster Distributed Execution:** Executed MapReduce volume aggregation across 121,962 records and distributed Spark DataFrame jobs generating 50-stock ranking partitions stored in HDFS and Parquet tables.
- **Risk and Return Profile:** Identified Adani Enterprises (`ADANIENT`) as top annualized arithmetic return (+51.56%, CAGR +44.1%) with high volatility (54.11%, Sharpe 0.84), and Bajaj Finance (`BAJFINANCE`) as top Sharpe ratio (0.86, annualized return 37.28%, volatility 36.23%).
- **Market Stress Quantification:** Documented market-wide breadth collapse during March 2020 COVID-19 crash (2% of constituents above 200 SMA) and quantified multi-year drawdown cycles (Tata Motors 3,082-day peak-to-recovery duration).
- **Empirical Machine Learning:** Tested 4 classification models (direction prediction) and 4 regression models (return magnitude prediction) on 12,250 unseen 2024 holdout rows with strict anti-leakage isolation. In classification, Logistic Regression achieved 50.74% accuracy and 0.5167 ROC-AUC against a 50.64% majority baseline. In regression, Random Forest achieved test MAE 0.011823, RMSE 0.016877, and conventional $R^2 = +0.029265$. These two tasks evaluate separate objectives and cannot be conflated.
- **Sentiment & Linear Association:** Disclosed that news headline polarity across 67 market-aligned corporate disclosures exhibits no statistically significant linear correlation with daily equity returns (same-day: $r_{\text{same}} = +0.0888, p = 0.4751$; next-day: $r_{\text{next}} = -0.1471, p = 0.2348$). This finding indicates an absence of linear predictive signal at the daily resolution in this sample, though it does not establish or rule out causality.
- **Offline Extractive RAG Retrieval Evaluation:** Achieved 95.0% Hit Rate@5 and 0.8583 Mean Reciprocal Rank (MRR) on financial document retrieval benchmarks using sentence embeddings. These figures evaluate dense context retrieval ranking rather than generative text accuracy.

---

## 21. Limitations

1. **Survivorship Bias:** The 50 constituent equities reflect the NIFTY 50 index roster as of 2024. Companies delisted, demoted, or restructured between 2015 and 2023 (e.g., Yes Bank, DHFL) are not in the sample. As a result, 10-year aggregate returns reflect a slight positive survivorship bias.
2. **Historical vs Live Data:** The platform operates exclusively on historical end-of-day (EOD) data (2015–2024). It does not connect to live streaming exchange feeds and does not model intraday order books, bid-ask spreads, or market microstructure.
3. **Weak ML Predictive Performance:** Machine learning models demonstrate performance very close to naive baselines (directional accuracy ~50.7%, ROC-AUC ~0.517, regression conventional $R^2 \approx 0.029$). Crucially, no reliable trading edge has been demonstrated by these models.
4. **Unmodeled Execution Frictions:** Models and backtest metrics do not simulate transaction costs, Securities Transaction Tax (STT), exchange turnover fees, GST, brokerage commissions, or execution slippage. Transaction-cost-aware backtesting would be needed to evaluate whether any theoretical statistical signal possesses practical economic profitability.
5. **News Corpus Size:** The real news dataset contains 79 corporate announcements (67 market-aligned sessions) from 2024. While suitable for demonstrating NLP pipelines, it does not constitute a high-frequency real-time financial wire.
6. **Single-Physical-Machine Virtualization:** The Hadoop and Spark cluster runs across Docker containers on a single physical workstation. While network protocols and service daemons are authentic, physical network latency, rack-awareness, and hardware failover are not modeled.

---

## 22. Documentation References

The `docs/` directory contains comprehensive academic documentation:

- [Project Architecture](docs/PROJECT_ARCHITECTURE.md) — System components, cluster topology, container networking, and data flow.
- [Financial Data Dictionary](docs/DATA_DICTIONARY.md) — Full schema definitions, column descriptions, and mathematical formulas.
- [Installation and Setup](docs/INSTALLATION_AND_SETUP.md) — Workstation prerequisites, environment configuration, and verification steps.
- [Cluster Deployment Guide](docs/CLUSTER_DEPLOYMENT_GUIDE.md) — Docker Compose operations, HDFS management, YARN commands, and Spark job submission.
- [User Guide](docs/USER_GUIDE.md) — Step-by-step navigation manual for all 11 Streamlit dashboard pages.
- [Machine Learning Methodology](docs/ML_METHODOLOGY.md) — Anti-leakage architecture, chronological partitions, EMH bounds, and detailed results.
- [Financial Analytics Methodology](docs/FINANCIAL_ANALYTICS_METHODOLOGY.md) — Econometric proofs, risk-free rate calibration, and formula specifications.
- [Known Limitations](docs/KNOWN_LIMITATIONS.md) — Detailed analysis of survivorship bias, data boundaries, and operational constraints.
- [Final Project Report](docs/FINAL_PROJECT_REPORT.md) — Complete capstone project report and academic findings.
- [Final Demonstration Guide](docs/FINAL_DEMONSTRATION_GUIDE.md) — 13-step lab viva demonstration script and technical defense Q&A.

---

## 23. Future Improvements

The following items are realistic extensions for future research and are distinct from implemented features:

- **Dynamic Historical Universe (Survivorship-Bias Correction):** Reconstruct point-in-time constituent rosters for each historical year, incorporating delisted and demoted stocks.
- **Intraday Tick & Order-Book Processing:** Ingest tick-level market data via Apache Kafka streaming to analyze intraday bid-ask spreads and order-flow toxicity (e.g., VPIN).
- **Physical Multi-Node Cluster Deployment:** Deploy the Hadoop/Spark cluster across physically distinct server nodes to evaluate genuine physical network partition tolerances and rack-aware HDFS placement.
- **Execution Cost Simulation:** Incorporate realistic transaction cost models (STT, brokerage, exchange charges, and slippage curves) into portfolio backtesting.
- **High-Throughput News Feeds:** Expand news coverage with automated continuous scrapers for regulatory exchange filings (NSE corporate announcements) and earnings conference call transcripts.
- **Hybrid Neural Retrieval:** Combine dense semantic embeddings with sparse BM25 keyword search (hybrid search) for financial statement table extraction.

---

## 24. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Developed for academic evaluation in **CSBB 422: Big Data Analytics Lab** at the **National Institute of Technology Delhi**. All financial analyses, econometric models, and sentiment scores are produced solely for educational research and do not constitute financial advice.
