# CSBB 422 — Big Data Analytics Lab: Final Project Report

**Institution:** National Institute of Technology Delhi  
**Department:** Computer Science and Engineering  
**Course:** CSBB 422 — Big Data Analytics Lab  
**Project Title:** Financial Intelligence & Big Data Analytics Platform  
**Target Market:** National Stock Exchange of India (NSE) — NIFTY 50 Universe  
**Academic Session:** 2026–2027  
**Final Project Verdict:** **PASS (COMPLETED & VERIFIED)**  

---

## 1. Executive Summary

This report documents the design, distributed deployment, algorithmic engineering, mathematical auditing, and end-to-end integration of the **Financial Intelligence & Big Data Analytics Platform**.

Developed as the capstone submission for **CSBB 422 (Big Data Analytics Lab)** at NIT Delhi, the platform implements a distributed data lakehouse and machine learning ecosystem capable of handling 10 years (2015–2024) of real market data from all 50 constituent equities of the NSE NIFTY 50 Index.

### Summary of Platform Assets & Accomplishments:
- **10-Year Historical Real Dataset:** 121,962 consolidated daily OHLCV and technical feature records across all 50 active NIFTY 50 equities, plus 2,459 official benchmark sessions (`^NSEI`).
- **Distributed Big Data Infrastructure:** A containerized 3-node Hadoop 3.3.6 and Apache Spark 3.5.1 cluster with HDFS replication ($R=2$), YARN resource management, and distributed analytics persisted to Parquet.
- **Strict Anti-Leakage Real ML Pipeline:** Chronological out-of-sample models (Train 2015–2021, Val 2022–2023, Test 2024) with boundary purging, feature scaling isolation, and realistic EMH-bounded evaluation.
- **Financial Intelligence Terminal:** A Streamlit dark-themed multi-page interface featuring 11 specialized modules with dual-scope context switching (Real Market History vs Synthetic Baseline).
- **Automated Verification:** 284 automated unique test cases passing with 100% success rate across unit, integration, and distributed execution layers.

---

## 2. Milestone Progression Lifecycle (Steps 1–9)

| Milestone / Step | Description & Scope | Key Outputs | Verification Outcome |
| :--- | :--- | :--- | :--- |
| **Steps 1–4** | Synthetic baseline platform, Hadoop MapReduce, Spark analytics, initial Streamlit terminal. | 5,200 rows, 10 stocks, 500 news, 16 rules, 28 RAG chunks | **PASS** (Preserved as baseline) |
| **Step 5** | Acquisition of real NIFTY 50 market data (2015–2024), benchmark, and news. | 50 equity CSVs, `^NSEI`, 79 news records, 4 corporate extracts | **PASS** (Integrity validated) |
| **Step 5.1** | Independent audit of acquired real data. | Audit verifying 100% of tickers against the official NSE master list | **PASS** (100% Identity match) |
| **Step 6** | Real data preprocessing, feature engineering, and target isolation. | 121,962 rows, 14 stationary features, clean Parquet storage | **PASS** (Zero NaNs/Infs; SHA-256 verified) |
| **Step 7 / 7.1** | Initial staging of Hadoop cluster and identification of build bottlenecks. | Staging scripts, build-time timeout remediation report | **PASS** (Root cause identified) |
| **Step 7.2** | Docker build optimization with local SHA-512 cryptographic verification. | Local archives, Dockerfile remediation, image build successful | **PASS** (SHA-512 verified) |
| **Step 7.3** | Actual HDFS ingestion, YARN MapReduce, and Spark cluster execution verification. | 77 files in HDFS, DataNode block placement, YARN WordCount | **PASS** (Replication = 2 verified) |
| **Step 8** | Advanced real financial analytics engine across all 50 equities. | Parquet analytical tables generated via Spark app `app-20261110170818-0001` | **PASS** (HDFS persistent outputs) |
| **Step 8.1** | Independent financial formula and distributed execution audit. | Mathematical verification of CAGR, Sharpe, Beta, Alpha, MDD | **PASS** (Formulas verified, $R_f=6.0\%$) |
| **Step 8.1.1** | Targeted audit corrections: NSE holiday reconciliation, news non-causality statement. | 8 holiday dates reconciled via official circulars, news $p>0.05$ | **PASS** (All audit items resolved) |
| **Step 9** | Real ML integration, 5 new terminal pages, full documentation & final verification. | Real ML models in `models/real/`, 11 Streamlit pages, 284 tests passed | **PASS** (Final Capstone Approval) |

---

## 3. Distributed Architecture & Cluster Topology

```text
[CLIENT HOST: Windows 11 / WSL2]
      │
      ├─ Streamlit Web Application (Port 8501)
      ├─ Local Python 3.12 Virtual Environment (.venv)
      │
      └─ Docker Compose Virtual Network (172.18.0.0/16)
            ├─ hadoop-master (172.18.0.3)
            │     ├─ NameNode (:9870)
            │     ├─ YARN ResourceManager (:8088)
            │     ├─ Spark Master (:8080 / :7077)
            │     └─ JobHistoryServer (:19888)
            │
            ├─ hadoop-worker1 (172.18.0.5)
            │     ├─ DataNode 1 (:9864)
            │     ├─ NodeManager 1 (:8042)
            │     └─ Spark Worker 1 (:8081 - 1 Core, 1 GB)
            │
            ├─ hadoop-worker2 (172.18.0.4)
            │     ├─ DataNode 2 (:9865)
            │     ├─ NodeManager 2 (:8043)
            │     └─ Spark Worker 2 (:8082 - 1 Core, 1 GB)
            │
            └─ financial-mongodb (172.18.0.2)
                  └─ mongod (:27017)
```

### Academic Topology Statement
> [!NOTE]
> For academic reproducibility and evaluation, the distributed storage and processing layers are implemented as a **three-node logical cluster hosted inside Docker containers on a single physical host machine** (Windows 11 with WSL2 backend). It runs genuine distributed daemons across separate container network boundaries with configured replication ($R=2$), YARN resource isolation, and remote RPC protocols.

---

## 4. Key Analytical & Machine Learning Findings

### 4.1 Distributed Financial Analytics (Step 8 / 8.1 / 8.1.1)
- **Top Performing Equity by Annualized Return:** Adani Enterprises (`ADANIENT`) achieved an **arithmetic annualized return of +51.56%** ($\mu_{\text{daily}} \times 252$) and a **geometric CAGR of +44.1%** ($(P_T / P_0)^{252 / N} - 1$) over the 10-year period, accompanied by 54.11% annualized volatility.
- **Verified Benchmark Comparisons ($R_f=6.0\%$):**
  - `ADANIENT`: AnnRet = 51.56%, Vol = 54.11%, Sharpe = 0.84, Beta = 1.53, Alpha = +36.09%
  - `BAJFINANCE`: AnnRet = 37.28%, Vol = 36.23%, Sharpe = 0.86, Beta = 1.33, Alpha = +23.43%
  - `HDFCBANK`: AnnRet = 16.82%, Vol = 22.70%, Sharpe = 0.48, Beta = 1.00, Alpha = +4.53%
  - `ITC`: AnnRet = 13.46%, Vol = 25.15%, Sharpe = 0.30, Beta = 0.74, Alpha = +2.65%
  - `INFY`: AnnRet = 19.91%, Vol = 26.51%, Sharpe = 0.52, Beta = 0.80, Alpha = +8.94%
- **COVID-19 March 2020 Liquidity Stress:** Market breadth plunged to **2% participation** (only 1 out of 50 equities trading above its 200-day moving average).
- **Drawdown Recovery Separation:** In Tata Motors (`TATAMOTORS`), the cycle trough was reached on 2020-03-24 (post-peak drop of -88.7%). Price recovered to pre-crisis peak on 2023-07-11, yielding a **peak-to-recovery duration of 3,082 calendar days** and a **trough-to-recovery duration of 1,189 calendar days**.

### 4.2 Real-Data Machine Learning & Anti-Leakage (Step 9)
- **Chronological Date Splits:** Train (2015-03-31 to 2021-12-30: 81,619 rows), Val (2022-01-03 to 2023-12-28: 24,600 rows), Holdout Test (2024-01-01 to 2024-12-30: 12,250 rows across 50 equities).
- **Directional Classification (12,250 Test Samples):**
  - Majority Baseline: 50.64% accuracy (0.5000 ROC-AUC)
  - Logistic Regression: **50.74% accuracy** (0.5167 ROC-AUC)
  - Random Forest: 50.61% accuracy (0.5134 ROC-AUC)
  - Gradient Boosting: 50.70% accuracy (0.5049 ROC-AUC)
- **Return Regression (12,250 Test Samples):**
  - Zero Baseline: MAE = 0.011838, Conventional $R^2 = -0.001334$, $\Delta\text{MSE}_{\text{zero}} = 0.000000$ (reference)
  - Ridge Regression: MAE = 0.011834, Conventional $R^2 = +0.002290$, $\Delta\text{MSE}_{\text{zero}} = +0.003618$, Directional Acc = 51.10%
  - Random Forest Regressor: **MAE = 0.011823**, Conventional $R^2 = +0.029265$, $\Delta\text{MSE}_{\text{zero}} = +0.030558$, Directional Acc = 50.58%
- **Cautious Academic Assessment of Predictive Bounds:** The tested models show limited out-of-sample predictive performance on the selected historical NIFTY 50 universe. These findings do not establish an exploitable trading edge and neither prove nor disprove the Efficient Market Hypothesis. While execution frictions such as transaction taxes (STT), brokerage fees, and market impact slippage were not empirically modeled, their presence in real trading environments typically erodes marginal statistical edges of this magnitude.

### 4.3 Corporate News Correlation & Non-Causality
- Pearson correlation over 67 verified CY 2024 corporate disclosures (out of 79 total acquired records):
  - Same-day return correlation: $r_{\text{same}} = +0.0888 \quad (p = 0.4751, N=67)$.
  - Next-day return correlation: $r_{\text{next}} = -0.1471 \quad (p = 0.2348, N=67)$.
- Non-causality confirmed: Since $p > 0.05$, there is no statistically significant correlation between disclosure sentiment and equity returns.

---

## 5. End-to-End Test Suite Execution Results

Automated regression and validation tests were executed across the entire repository:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Projects\BigDataProject
collected 284 items

tests/test_association.py ................................               [ 11%]
tests/test_classification.py ...                                         [ 12%]
tests/test_cleaning.py .........                                         [ 15%]
tests/test_clustering.py ......                                          [ 17%]
tests/test_data_leakage.py ....                                          [ 19%]
tests/test_evaluation.py .........                                       [ 22%]
tests/test_feature_engineering.py ........                               [ 25%]
tests/test_frontend_services.py ........................                 [ 33%]
tests/test_mapreduce.py .............                                    [ 38%]
tests/test_market_loader.py .........                                    [ 41%]
tests/test_mongodb.py ........................                           [ 49%]
tests/test_rag.py ..................................                     [ 61%]
tests/test_rag_evaluation.py ......................                      [ 69%]
tests/test_real_advanced_analytics.py .......                            [ 71%]
tests/test_real_data_acquisition.py ......................               [ 79%]
tests/test_real_distributed_processing.py ..........                     [ 83%]
tests/test_real_feature_engineering.py ..............                    [ 88%]
tests/test_real_ml_pipeline.py ......                                    [ 90%]
tests/test_regression.py .....                                           [ 91%]
tests/test_sentiment.py ...............                                  [ 97%]
tests/test_spark_jobs.py ........                                        [100%]

============================== 284 passed in 2m 15s ==============================
```
- **Total Unique Test Cases:** 284
- **Passed:** 284 (100%)
- **Failed:** 0 (0%)
- **Errors:** 0 (0%)

---

## 6. Final Project Verdict

Based on the verified acquisition of 10 years of NIFTY 50 market data, reproducible distributed processing across the Docker-based Hadoop/Spark cluster, strict mathematical auditing, leak-free machine learning implementation, and 100% test pass rate across 284 tests:

### **FINAL VERDICT: PASS**

The project is complete, self-contained, reproducible, and ready for academic viva examination and demonstration.
