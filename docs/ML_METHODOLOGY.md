# Machine Learning Methodology & Anti-Leakage Architecture

**Course:** CSBB 422 — Big Data Analytics Lab  
**Institution:** National Institute of Technology Delhi  
**Project:** Financial Intelligence & Big Data Analytics Platform  

---

## 1. Overview & Dual-Dataset ML Framework

This document details the machine learning methodology employed across both:
1. **The Step 9 Real Market ML Pipeline (NIFTY 50):** 121,962 daily feature records covering 50 NSE blue chips over 10 years (2015–2024), evaluated under strict out-of-sample chronological partitioning.
2. **The Phase 4 Synthetic Baseline Pipeline:** 5,200 records across 10 equities, evaluating regression, 3-class direction classification, and K-Means clustering.

---

## 2. Real Market ML Problem Formulations (Step 9)

### Task 1: Next-Day Direction Classification (Binary)
Given an explanatory technical feature vector $\mathbf{x}_{i,t} \in \mathbb{R}^{14}$ observed at the close of trading session $t$ for equity $i$, predict whether the adjusted closing price on day $t+1$ will be higher than day $t$:

$$y_{i,t+1}^{\text{dir}} = \begin{cases} 1, & \text{if } r_{i,t+1} > 0 \quad (\text{UP}) \\ 0, & \text{if } r_{i,t+1} \le 0 \quad (\text{DOWN/FLAT}) \end{cases}$$

Where the objective is to evaluate out-of-sample directional accuracy and Area Under the ROC Curve (ROC-AUC). Zero returns are classified as class 0.

### Task 2: Next-Day Return Magnitude Regression (Continuous)
Predict the continuous percentage return on the next trading day:

$$\hat{r}_{i,t+1} = f(\mathbf{x}_{i,t}; \mathbf{\theta})$$

Where ground truth is:

$$r_{i,t+1} = \frac{P_{i,t+1} - P_{i,t}}{P_{i,t}}$$

Evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Conventional $R^2$, and Relative Improvement over the Zero-Return Baseline.

---

## 3. Strict Anti-Leakage Protocol & Temporal Splitting

Financial time series violate the Independent and Identically Distributed (i.i.d.) assumption. Standard random k-fold cross-validation results in severe look-ahead bias and spurious predictive accuracy. To eliminate data leakage, the real ML pipeline enforces five strict isolation rules:

### A. Chronological Partitioning (No Shuffle)
- **Training Set:** 2015-03-31 to 2021-12-30 (81,619 samples across 50 equities).
- **Validation Set:** 2022-01-03 to 2023-12-28 (24,600 samples across 50 equities).
- **Holdout Test Set:** 2024-01-01 to 2024-12-30 (12,250 samples across 50 equities; 245 trading days $\times$ 50 stocks).

### B. Feature Scaler Isolation
The feature normalization scaler (`StandardScaler`) is fitted **strictly on the Training partition only**. Validation and Test sets are transformed using the fixed training mean ($\mu_{\text{train}}$) and variance ($\sigma_{\text{train}}$).

### C. Panel Boundary Purging
On the final trading day of the Training partition (2021-12-31) and Validation partition (2023-12-29), the target variable $r_{t+1}$ extends into the next partition. To avoid cross-partition transition leakage, boundary observation rows are explicitly purged.

### D. Target Variable Decoupling
The target column `target_next_day_return` and directional label `target_direction` are generated via strict lead shifts (`shift(-1)`) grouped by symbol and immediately decoupled from the explanatory feature matrix.

### E. Scale Invariance
Raw nominal price levels (`close`, `high`, `low`, `open`) and raw identifiers (`symbol`, `date`) are excluded from model inputs. All features are dimensionless ratios, percentages, or normalized rolling statistics.

---

## 4. Explanatory Feature Space (14 Stationary Features)

| Feature Name | Description | Formula | Economic Rationale |
| :--- | :--- | :--- | :--- |
| `daily_adjusted_return` | 1-day percentage return | $\frac{P_t - P_{t-1}}{P_{t-1}}$ | Immediate price momentum |
| `return_5d` | 5-day rolling return | $\frac{P_t - P_{t-5}}{P_{t-5}}$ | Weekly momentum factor |
| `return_20d` | 20-day rolling return | $\frac{P_t - P_{t-20}}{P_{t-20}}$ | Monthly trend factor |
| `return_60d` | 60-day rolling return | $\frac{P_t - P_{t-60}}{P_{t-60}}$ | Quarterly medium-term trend |
| `volatility_20d` | 20-day annualized volatility | $\text{std}(r_{t-19 \dots t}) \times \sqrt{252}$ | Short-term asset risk |
| `volatility_60d` | 60-day annualized volatility | $\text{std}(r_{t-59 \dots t}) \times \sqrt{252}$ | Medium-term volatility regime |
| `close_to_ma20` | Price / 20-day SMA | $\frac{P_t}{\text{SMA}_{20}(P_t)} - 1.0$ | Short-term mean reversion |
| `close_to_ma50` | Price / 50-day SMA | $\frac{P_t}{\text{SMA}_{50}(P_t)} - 1.0$ | Intermediate trend support |
| `close_to_ma200` | Price / 200-day SMA | $\frac{P_t}{\text{SMA}_{200}(P_t)} - 1.0$ | Long-term macro trend |
| `volume_ratio` | Volume / 20-day Volume SMA | $\frac{V_t}{\text{SMA}_{20}(V_t)}$ | Liquidity / institutional accumulation |
| `hl_pct_range` | Normalized intraday range | $\frac{\text{High}_t - \text{Low}_t}{P_t}$ | Intraday price dispersion |
| `rolling_max_drawdown_60d`| 60-day maximum drawdown | Peak-to-trough drop over 60 days | Historical downside stress |
| `benchmark_daily_return` | NIFTY 50 daily return | $\frac{I_t - I_{t-1}}{I_{t-1}}$ | Systematic market co-movement |
| `relative_return_vs_nifty`| Asset return minus benchmark | $r_{\text{stock}, t} - r_{\text{nifty}, t}$ | Cross-sectional idiosyncratic alpha |

---

## 5. Empirical Results on Unseen Holdout Data (CY 2024, $N=12,250$)

### A. Binary Classification Results (12,250 Test Samples)

| Model Name | Test Accuracy | Balanced Acc | Precision | Recall | F1 Score | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | 50.64% | 50.00% | 50.64% | 100.00% | 0.6723 | 0.5000 |
| **Logistic Regression** | **50.74%** | 50.53% | 51.04% | 66.90% | 0.5790 | **0.5167** |
| **Random Forest** | 50.61% | 50.50% | 50.61% | 100.00% | 0.6721 | **0.5134** |
| **Gradient Boosting** | 50.70% | **50.72%** | **51.38%** | 46.85% | 0.4899 | 0.5049 |

### B. Continuous Return Regression Results (12,250 Test Samples)

Two distinct $R^2$ metrics are reported:
1. **Conventional Test-Set $R^2$** (`r2_score`): Evaluates variance explained relative to the sample mean return $\bar{y} = +0.000626$. Because $\bar{y} \ne 0$, the zero-return baseline achieves $R^2 = -0.001334$.
2. **Relative MSE Reduction vs Zero Baseline** ($\Delta\text{MSE}_{\text{zero}} = 1 - \text{MSE}/\text{MSE}_{\text{zero}}$): Measures improvement over the naive 0.0000 reference point.

| Model Name | Test MAE | Test RMSE | Conventional $R^2$ | Relative vs Zero ($\Delta\text{MSE}$) | Directional Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero Baseline** | 0.011838 | 0.017141 | **-0.001334** | **0.000000** | 50.00% |
| **Ridge Regression** | 0.011834 | 0.017110 | **+0.002290** | **+0.003618** | **51.10%** |
| **Random Forest** | **0.011823** | **0.016877** | **+0.029265** | **+0.030558** | 50.58% |
| **Gradient Boosting** | 0.012029 | 0.017328 | -0.023394 | -0.022031 | 49.74% |

### C. Feature Importance Distribution (Random Forest)
1. `benchmark_daily_return`: **18.52%**
2. `relative_return_vs_nifty`: **7.87%**
3. `daily_adjusted_return`: **7.39%**
4. `close_to_ma20`: **7.20%**
5. `return_5d`: **6.55%**
6. `volume_ratio`: **6.45%**
7. `hl_pct_range`: **6.35%**

---

## 6. Efficient Market Hypothesis & Academic Boundaries

> [!IMPORTANT]
> **Cautious Academic Assessment of Predictive Bounds**:  
> The tested models show limited out-of-sample predictive performance on the selected historical NIFTY 50 universe. These findings do not establish an exploitable trading edge and neither prove nor disprove the Efficient Market Hypothesis. While execution frictions such as transaction taxes (STT), brokerage fees, exchange turnover charges, and market impact slippage were not empirically modeled in this feature pipeline, their presence in live trading environments typically erodes marginal statistical edges of this magnitude.  
> 
> Reported test accuracies exceeding 65% on daily liquid large-cap equities (e.g. NIFTY 50) almost invariably indicate subtle data leakage (e.g. look-ahead bias, random shuffle, unpurged overlap). Our evaluated results reflect genuine, leak-free statistical tendencies for educational research.
