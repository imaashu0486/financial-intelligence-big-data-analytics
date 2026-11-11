# Platform Boundaries, Operational Constraints & Known Limitations

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  

---

## 1. Architectural & Infrastructure Limitations

### 1.1 Single-Host Containerized Topology
- **Reality:** The cluster (`hadoop-master`, `hadoop-worker1`, `hadoop-worker2`, `financial-mongodb`) runs inside Docker containers on a single physical computer via WSL2.
- **Limitation:** While container network isolation and RPC interfaces accurately mimic multi-node behavior, the cluster shares physical hardware resources (single CPU package, single memory bus, single NVMe controller). It does not experience physical network partition latency, rack failures, or cross-switch data transfers.
- **Academic Framing:** Designed specifically as an educational and research proof-of-concept for the CSBB 422 laboratory curriculum.

### 1.2 Memory & YARN Quotas
- Docker Desktop is constrained to host memory limits (configured up to 12 GB RAM). Heavy simultaneous Spark jobs alongside local PyTorch/scikit-learn inference can cause memory pressure.

---

## 2. Market Data & Universe Limitations

### 2.1 Survivorship Bias
- **Universe:** 50 current constituent equities of the NIFTY 50 Index as of 2024.
- **Limitation:** Historical backtesting across 2015–2024 on the *current* roster of NIFTY 50 introduces survivorship bias, as historical constituents that went bankrupt, merged, or were demoted during the 10-year span (e.g. Yes Bank, DHFL, Zee Entertainment) are not tracked.
- **Impact:** Historical aggregate returns and Sharpe ratios for the 50-stock portfolio are slightly higher than a survivorship-bias-free dynamic roster would reflect.

### 2.2 Benchmark Data Availability
- 8 trading days over the 10-year period were present in individual equity histories but absent from the Yahoo Finance `^NSEI` index feed. Following audit investigation, these were reconciled using official NSE holiday circulars (Muhurat special sessions and holiday classifications). An inner join methodology (2,459 common days) was adopted to guarantee mathematical consistency.

### 2.3 Low-Frequency Daily Data
- Sourced at daily end-of-day (EOD) OHLCV resolution. Intraday microstructure dynamics, order book imbalances, bid-ask spreads, and flash crashes cannot be modeled with EOD data.

---

## 3. Machine Learning & Predictive Limitations

### 3.1 Market Noise & Statistical Limits
- Out-of-sample directional prediction on unseen 2024 data yields ~50.7% to 51.1% accuracy.
- Out-of-sample return regression yields a conventional $R^2$ of ~0.002 to 0.029.
- **Academic Boundary:** The tested models show limited out-of-sample predictive performance on the selected historical NIFTY 50 universe. These findings do not establish an exploitable trading edge and neither prove nor disprove the Efficient Market Hypothesis.
- **Limitation:** The signal-to-noise ratio in daily equity returns is extremely low. These models provide statistical insight into historical factor tendencies, but **they cannot be used as an automated trading system or financial advisory tool**.

### 3.2 Unmodeled Execution Frictions
- Machine learning performance metrics do not account for trading friction:
  - Securities Transaction Tax (STT: 0.1% on delivery).
  - Exchange transaction charges, SEBI turnover fees, GST (18%).
  - Brokerage commissions and bid-ask execution slippage.
  - While execution frictions were not empirically modeled in the feature pipeline, their presence in live trading environments typically erodes marginal statistical edges of this magnitude.

---

## 4. NLP & Document RAG Limitations

### 4.1 Sample Size of News Data
- CY 2024 news dataset comprises 79 verified corporate announcements across the 50 equities (with 67 sessions aligned to active market trading days). While sufficient for demonstrating sentiment extraction and correlation testing ($p$-values documented), it does not represent a high-throughput, millisecond news wire.

### 4.2 Document Scope (Extracts vs Full Statutory Reports)
- Real corporate filings are represented as verified Management Discussion & Analysis (MD&A) and governance extracts (~2,500 words per company) rather than full 400-page statutory PDF reports.
- Semantic RAG vector embeddings (`all-MiniLM-L6-v2`) operate on chunk sizes of 250–500 tokens. Deep tabular balance-sheet arithmetic cannot be natively reasoned by small embedding models without structured database querying.
