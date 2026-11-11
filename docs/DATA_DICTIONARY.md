# Comprehensive Financial Data Dictionary

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  

---

## 1. Overview of Datasets

The platform manages two distinct data domains to maintain absolute analytical integrity:
1. **Real Market Dataset (Steps 5–9):** 10-year historical data (2015–2024) covering all 50 constituent equities of the National Stock Exchange (NSE) NIFTY 50 Index, official benchmark quotes, corporate news disclosures, and verified FY24 annual report MD&A extracts.
2. **Synthetic Baseline Dataset (Steps 1–4):** A controlled benchmark of 10 equities across 520 trading days, 500 financial news items, and 3 simulated corporate reports, preserved for algorithmic baseline comparison.

---

## 2. Real Market Consolidated Features (`consolidated_market_features.parquet`)

**Location:** `data/real/processed/market/consolidated_market_features.parquet`  
**HDFS Location:** `/financial-intelligence/real/market/consolidated_market_features.parquet`  
**Record Count:** 121,962 rows across 50 equities  
**Date Range:** 2015-01-01 to 2024-12-31  

| Column Name | Data Type | Units / Range | Description | Formula / Derivation |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `string` / `date` | `YYYY-MM-DD` | Official NSE trading session date | Exchange trading calendar |
| `symbol` | `string` | e.g. `RELIANCE` | Verified NSE equity ticker symbol | NIFTY 50 constituent roster |
| `open` | `float64` | INR (₹) | Split/dividend adjusted opening price | Sourced via Yahoo Finance API |
| `high` | `float64` | INR (₹) | Split/dividend adjusted session high | Sourced via Yahoo Finance API |
| `low` | `float64` | INR (₹) | Split/dividend adjusted session low | Sourced via Yahoo Finance API |
| `close` | `float64` | INR (₹) | Split/dividend adjusted closing price | Primary valuation series ($P_t$) |
| `volume` | `int64` | Shares traded | Total session equity volume | Exchange reported share volume |
| `daily_adjusted_return` | `float64` | $\pm 0.15$ | Session simple adjusted return | $\frac{P_t - P_{t-1}}{P_{t-1}}$ |
| `log_return` | `float64` | $\pm 0.15$ | Continuous logarithmic return | $\ln(P_t / P_{t-1})$ |
| `return_5d` | `float64` | $\pm 0.30$ | 5-day cumulative rolling return | $\frac{P_t - P_{t-5}}{P_{t-5}}$ |
| `return_20d` | `float64` | $\pm 0.50$ | 20-day cumulative rolling return | $\frac{P_t - P_{t-20}}{P_{t-20}}$ |
| `return_60d` | `float64` | $\pm 1.00$ | 60-day cumulative rolling return | $\frac{P_t - P_{t-60}}{P_{t-60}}$ |
| `volatility_20d` | `float64` | $[0.05, 1.50]$ | 20-day annualized rolling volatility | $\text{std}(r_{t-19 \dots t}) \times \sqrt{252}$ |
| `volatility_60d` | `float64` | $[0.05, 1.20]$ | 60-day annualized rolling volatility | $\text{std}(r_{t-59 \dots t}) \times \sqrt{252}$ |
| `close_to_ma20` | `float64` | $[0.70, 1.40]$ | Ratio of close to 20-day SMA | $\frac{P_t}{\text{SMA}_{20}(P_t)}$ |
| `close_to_ma50` | `float64` | $[0.60, 1.60]$ | Ratio of close to 50-day SMA | $\frac{P_t}{\text{SMA}_{50}(P_t)}$ |
| `close_to_ma200` | `float64` | $[0.40, 2.50]$ | Ratio of close to 200-day SMA | $\frac{P_t}{\text{SMA}_{200}(P_t)}$ |
| `volume_ratio` | `float64` | $[0.10, 25.0]$ | Ratio of volume to 20-day volume SMA | $\frac{V_t}{\text{SMA}_{20}(V_t)}$ |
| `hl_pct_range` | `float64` | $[0.001, 0.25]$ | Normalized intraday price range | $\frac{\text{High}_t - \text{Low}_t}{P_t}$ |
| `rolling_max_drawdown_60d`| `float64` | $[-0.80, 0.00]$ | 60-day rolling maximum drawdown | $\min_{\tau \le t} \left(\frac{P_\tau - \max_{s \le \tau} P_s}{\max_{s \le \tau} P_s}\right)$ |
| `benchmark_daily_return` | `float64` | $\pm 0.10$ | NIFTY 50 index daily return on $t$ | Aligned from `^NSEI` index session |
| `relative_return_vs_nifty`| `float64` | $\pm 0.15$ | Excess return over benchmark | $r_{\text{stock}, t} - r_{\text{benchmark}, t}$ |
| `target_next_day_return` | `float64` | $\pm 0.20$ | Supervised regression target ($r_{t+1}$) | Strictly shifted: $\frac{P_{t+1} - P_t}{P_t}$ |
| `target_direction` | `int64` | $\{0, 1\}$ | Supervised classification target | $1$ if $r_{t+1} > 0$ else $0$ |

---

## 3. Real Benchmark Dataset (`benchmark_features.parquet`)

**Location:** `data/real/processed/market/benchmark_features.parquet`  
**Record Count:** 2,459 trading sessions  
**Index Symbol:** `^NSEI` (NIFTY 50 Index)  

| Column Name | Data Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `date` | `string` | `YYYY-MM-DD` | Trading session date |
| `close` | `float64` | Index Points | NIFTY 50 closing level (ranging ~7,800 to ~26,200) |
| `daily_adjusted_return` | `float64` | Decimal | Daily benchmark percentage return |
| `volatility_20d` | `float64` | Annualized | 20-day annualized rolling index volatility |
| `close_to_ma50` | `float64` | Ratio | Index level relative to 50-day moving average |
| `close_to_ma200` | `float64` | Ratio | Index level relative to 200-day moving average |

---

## 4. Real Analytical Tables (Distributed Spark Outputs)

Stored in HDFS under `/financial-intelligence/real/analytics/`:

### A. Risk & Return Metrics (`risk_return_metrics.parquet`)
- `symbol` (`string`): Equity ticker.
- `cagr` (`float64`): Compound annual growth rate over 10-year period.
- `annualized_return` (`float64`): Mean daily return multiplied by 252.
- `annualized_volatility` (`float64`): Sample standard deviation of daily return multiplied by $\sqrt{252}$.
- `sharpe_ratio_rf6` (`float64`): Excess return over baseline risk-free rate ($R_f=0.060$, Indian 10-Year G-Sec) divided by annualized volatility: $\frac{\text{AnnRet} - 0.060}{\text{AnnVol}}$.
- `sharpe_ratio_rf0` (`float64`): Zero-risk-free Sharpe ratio: $\frac{\text{AnnRet}}{\text{AnnVol}}$.
- `capm_beta` (`float64`): Systematic market risk coefficient relative to NIFTY 50.
- `jensen_alpha_rf6` (`float64`): Risk-adjusted abnormal return vs CAPM expectation ($R_f=0.060$).
- `max_drawdown` (`float64`): Peak-to-trough maximum historical loss.
- `peak_date` / `trough_date` / `recovery_date`: Key historical drawdown regime timestamps.
- `drawdown_duration_days`: Full peak-to-recovery duration in calendar days.
- `trough_to_recovery_days`: Duration from the cycle trough to complete price recovery.

### B. Market Breadth (`market_breadth_daily.parquet`)
- `date` (`string`): Trading session.
- `advances` (`int`): Count of NIFTY 50 equities closing positive.
- `declines` (`int`): Count of NIFTY 50 equities closing negative.
- `advance_ratio` (`float64`): $\frac{\text{advances}}{\text{advances} + \text{declines}}$.
- `pct_above_ma50` (`float64`): Percentage of constituents trading above their 50-day SMA.
- `pct_above_ma200` (`float64`): Percentage of constituents trading above their 200-day SMA.
- `regime` (`string`): Market participation regime (`EXPANSION`, `STRESS`, `NEUTRAL`).

### C. Pairwise Correlation Matrix (`correlation_matrix_50x50.parquet`)
- `symbol_a` (`string`): Base equity.
- `symbol_b` (`string`): Target equity.
- `correlation` (`float64`): Pearson product-moment correlation coefficient over 2,459 aligned sessions.

### D. Composite Stock Rankings (`stock_rankings.parquet`)
- `symbol` (`string`): Equity ticker.
- `composite_score` (`float64`): Weighted multi-factor score ($\in [0.0, 1.0]$).
- `rank` (`int`): Rank ordinal from 1 to 50 based on Sharpe ratio, Momentum, Beta stability, and Low Volatility.

---

## 5. News Intelligence & NLP Dataset (`cleaned_financial_news.csv`)

**Location:** `data/real/processed/news/cleaned_financial_news.csv`  
**Total Records:** 79 corporate disclosure articles (CY 2024 across 50 equities)  
**Trading-Aligned Sessions:** 67 records matched directly with active NSE trading dates for same-day and next-day market return correlation analysis (12 occurred on weekends or non-trading holidays).  

| Column Name | Data Type | Values | Description |
| :--- | :--- | :--- | :--- |
| `article_id` | `string` | e.g. `NEWS_001` | Unique article identifier |
| `date` | `string` | `YYYY-MM-DD` | Publication timestamp |
| `symbol` | `string` | e.g. `TCS` | Associated NIFTY 50 constituent |
| `category` | `string` | `EARNINGS`, `ACQUISITION`, `GOVERNANCE`, `MACRO` | Disclosure category |
| `headline` | `string` | Text | Full disclosure headline |
| `sentiment_score` | `float64` | $[-1.0, +1.0]$ | Financial lexicon polarity score |
| `sentiment_label` | `string` | `POSITIVE`, `NEUTRAL`, `NEGATIVE` | Classified sentiment polarity |

---

## 6. Corporate Filings Dataset (FY24 MD&A Extracts)

**Location:** `data/real/raw/documents/`  
**Files:**
1. `RELIANCE_Annual_Report_FY24_Strategic_Overview.txt` (2,669 bytes): Segment overview covering Jio Platforms, Reliance Retail, Oil-to-Chemicals (O2C), and green energy transitions.
2. `TCS_Annual_Report_FY24_Business_Review.txt` (2,365 bytes): AI cloud transformation, digital services revenue, and BFSI industry commentary.
3. `HDFCBANK_Annual_Report_FY24_Financial_Disclosures.txt` (2,141 bytes): Post-merger balance sheet integration, Net Interest Margins (NIM), and NPA provisions.
4. `INFY_Annual_Report_FY24_Governance_and_Strategy.txt` (2,157 bytes): Generative AI Topaz platform, large-deal Total Contract Value (TCV), and operational margin targets.

---

## 7. Data Lineage & Cryptographic Integrity

- **Instrument Identity Audit:** 100% of ticker symbols were reconciled against the National Stock Exchange of India master equity list.
- **Missing Date Reconciliation:** Eight historical dates with non-standard session structures (e.g. Diwali Muhurat trading sessions) were audited against official NSE circulars.
- **SHA-256 Checksums:** All processed and analytical assets maintain verified cryptographic checksums in data manifests (`data/real/metadata/`), guaranteeing reproducibility.
