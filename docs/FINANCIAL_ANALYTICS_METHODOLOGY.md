# Financial Analytics Mathematical Methodology & Spark Implementation

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  

---

## 1. Overview

This document provides formal mathematical definitions, risk-return metrics, market breadth algorithms, and the distributed PySpark implementations utilized across the NIFTY 50 10-year historical dataset (2015–2024, 121,962 rows).

---

## 2. Core Return & Volatility Formulations

### A. Daily Return Formulations
- **Simple Adjusted Return ($r_t$):**
  $$r_{i,t} = \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}}$$
- **Continuous Compound Logarithmic Return ($R_t$):**
  $$R_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

### B. Compound Annual Growth Rate (CAGR)
CAGR measures the mean geometric annual growth rate of an equity over the multi-year investment horizon:
$$\text{CAGR}_i = \left(\frac{P_{i,T}}{P_{i,0}}\right)^{\frac{252}{N}} - 1$$
Where $N$ is the total number of historical trading sessions ($N = 2,467$).

### C. Annualized Return & Annualized Volatility
Under standard quantitative finance conventions (252 trading days per year):
$$\bar{R}_{\text{ann}, i} = \bar{r}_i \times 252$$
$$\sigma_{\text{ann}, i} = \sqrt{\frac{1}{N-1} \sum_{t=1}^N (r_{i,t} - \bar{r}_i)^2} \times \sqrt{252}$$
*Note: Arithmetic annualized return ($\bar{r} \times 252$) and geometric CAGR are reported as distinct metrics in the platform.*

---

## 3. Risk-Adjusted Performance & Benchmark Analytics

All benchmark calculations are strictly aligned with the NIFTY 50 Index (`^NSEI`, 2,459 common dates) via inner joins.

### A. Sharpe Ratio
The Sharpe ratio evaluates excess portfolio return per unit of total risk against the baseline risk-free rate ($R_f = 6.0\%$ annual yield, representing the Indian 10-Year Government Securities benchmark rate):
$$\text{Sharpe}_i = \frac{\bar{R}_{\text{ann}, i} - R_f}{\sigma_{\text{ann}, i}}$$

### B. Capital Asset Pricing Model (CAPM) Beta
Beta ($\beta$) quantifies the systematic covariance risk of equity $i$ relative to the market benchmark ($m$):
$$\beta_i = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)} = \frac{\sum_{t=1}^T (r_{i,t} - \bar{r}_i)(r_{m,t} - \bar{r}_m)}{\sum_{t=1}^T (r_{m,t} - \bar{r}_m)^2}$$

### C. Jensen's Alpha
Jensen's Alpha ($\alpha$) captures the abnormal risk-adjusted return generated above CAPM expectations:
$$\alpha_i = \bar{R}_{\text{ann}, i} - \left(R_f + \beta_i (\bar{R}_{\text{ann}, m} - R_f)\right)$$

### D. Representative Benchmark Equities (2015–2024 Verified Results)

The following metrics are verified directly from the distributed analytical Parquet tables (`risk_return_metrics.csv` and `benchmark_relative_metrics.csv`):

| Ticker | Arithmetic Return ($\mu_{\text{ann}}$) | Annualized Volatility ($\sigma_{\text{ann}}$) | Sharpe Ratio ($R_f=6.0\%$) | CAPM Beta ($\beta$) | Jensen's Alpha ($R_f=6.0\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ADANIENT** | 51.56% | 54.11% | 0.84 | 1.53 | +36.09% |
| **BAJFINANCE** | 37.28% | 36.23% | 0.86 | 1.33 | +23.43% |
| **HDFCBANK** | 16.82% | 22.70% | 0.48 | 1.00 | +4.53% |
| **ITC** | 13.46% | 25.15% | 0.30 | 0.74 | +2.65% |
| **INFY** | 19.91% | 26.51% | 0.52 | 0.80 | +8.94% |

### E. Maximum Drawdown & Recovery Analytics
Maximum Drawdown (MDD) captures the largest historical percentage drop from peak to trough:
$$\text{MDD}_i = \min_{t \in [0, T]} \left( \frac{P_{i,t} - \max_{s \le t} P_{i,s}}{\max_{s \le t} P_{i,s}} \right)$$

Two distinct drawdown recovery metrics are maintained:
1. **Peak-to-Recovery Duration:** Calendar days from the pre-crisis peak date to the date price fully reattains that peak. (e.g. TATAMOTORS peak 2015-02-03 to recovery 2023-07-11 = 3,082 calendar days).
2. **Trough-to-Recovery Duration:** Calendar days from the cycle trough date to the recovery date. (e.g. TATAMOTORS trough 2020-03-24 to recovery 2023-07-11 = 1,189 calendar days).

---

## 4. Market Breadth & Regime Classification

Market breadth measures the degree of participation across all 50 constituent equities.

### A. Advance-Decline Ratio
$$AD_t = \frac{\sum_{i=1}^{50} \mathbb{I}(r_{i,t} > 0)}{\sum_{i=1}^{50} \mathbb{I}(r_{i,t} > 0) + \sum_{i=1}^{50} \mathbb{I}(r_{i,t} < 0)}$$

### B. Moving Average Participation Rates
$$P_{\text{MA50}, t} = \frac{1}{50} \sum_{i=1}^{50} \mathbb{I}(P_{i,t} > \text{SMA}_{50}(P_{i,t}))$$
$$P_{\text{MA200}, t} = \frac{1}{50} \sum_{i=1}^{50} \mathbb{I}(P_{i,t} > \text{SMA}_{200}(P_{i,t}))$$

### C. Regime Classification Thresholds
- **`EXPANSION`:** $P_{\text{MA200}, t} \ge 0.80$ (80% or more of stocks trading above 200 SMA; occurred in 721 sessions).
- **`STRESS`:** $P_{\text{MA200}, t} \le 0.20$ (20% or fewer of stocks trading above 200 SMA; occurred in 49 sessions, including COVID March 2020).
- **`NEUTRAL`:** $0.20 < P_{\text{MA200}, t} < 0.80$.

---

## 5. Pairwise Correlation Matrix Formulation

For any pair of equities $(i, j)$ across the 50-stock universe:
$$\rho_{ij} = \frac{\sum_{t=1}^T (r_{i,t} - \bar{r}_i)(r_{j,t} - \bar{r}_j)}{\sqrt{\sum_{t=1}^T (r_{i,t} - \bar{r}_i)^2 \sum_{t=1}^T (r_{j,t} - \bar{r}_j)^2}}$$
Computed across $50 \times 50 = 2,500$ equity pairs in parallel using Apache Spark distributed transformations.

---

## 6. News Sentiment Return Correlation & Non-Causality

To evaluate whether corporate news disclosures correlate with stock returns, Pearson correlations were computed over 67 verified corporate announcements matched with active trading days in CY 2024 (out of 79 total acquired records):
- **Same-Day Return Correlation ($r_{\text{same}}$):**
  $$r_{\text{same}} = +0.0888 \quad (p = 0.4751, N = 67)$$
- **Next-Day Return Correlation ($r_{\text{next}}$):**
  $$r_{\text{next}} = -0.1471 \quad (p = 0.2348, N = 67)$$

### Academic Non-Causality Statement
> [!IMPORTANT]
> Both $p$-values significantly exceed the $\alpha = 0.05$ threshold ($p = 0.4751$ and $p = 0.2348$). Therefore, the null hypothesis cannot be rejected. **There is no statistically significant linear association between disclosure sentiment and equity returns in this sample.** This confirms that corporate disclosures in this dataset cannot be interpreted as causal market drivers.
