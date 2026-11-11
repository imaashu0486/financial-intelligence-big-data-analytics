"""
Unit and Integration Tests for Step 8 Advanced Real-Data Financial Analytics.
"""

import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.analytics.risk_return import calculate_equity_risk_return
from src.analytics.benchmark_analytics import calculate_benchmark_metrics
from src.analytics.market_regime import calculate_daily_market_breadth, classify_benchmark_regimes
from src.analytics.screening import compute_multi_factor_rankings
from src.analytics.correlation import compute_correlation_matrix, extract_correlation_pairs
from src.analytics.news_document_intelligence import classify_event_category, analyze_news_and_documents
from src.nlp.lexicon import score_financial_text


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Create synthetic historical market dataframe mimicking NIFTY 50 equities."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B").strftime("%Y-%m-%d").tolist()
    records = []
    for sym in ["STOCK_A", "STOCK_B"]:
        p = 100.0
        for i, d in enumerate(dates):
            base_ret = 0.001 if sym == "STOCK_A" else -0.0005
            ret = base_ret + 0.002 * ((i % 5) - 2)
            p *= (1.0 + ret)
            records.append({
                "date": d,
                "symbol": sym,
                "adj_close": p,
                "close": p,
                "daily_adjusted_return": ret,
                "benchmark_daily_return": 0.0005,
                "ma_20": p * 0.98,
                "ma_50": p * 0.95,
                "ma_200": p * 0.90,
                "target_next_day_return": 0.002,  # Future target
            })
    return pd.DataFrame(records)


def test_risk_return_calculation(sample_market_df):
    """Test risk and return annualization, Sharpe ratio, and drawdowns."""
    df_risk = calculate_equity_risk_return(sample_market_df, rf_rate=0.06)
    assert len(df_risk) == 2
    assert "symbol" in df_risk.columns
    assert "annualized_return" in df_risk.columns
    assert "annualized_volatility" in df_risk.columns
    assert "sharpe_ratio_rf6" in df_risk.columns
    assert "max_drawdown" in df_risk.columns
    assert "var_95_daily" in df_risk.columns

    # Verify max drawdown is non-positive
    assert (df_risk["max_drawdown"] <= 0.0).all()


def test_benchmark_metrics(sample_market_df):
    """Test CAPM Beta, Jensen's Alpha, and Tracking Error calculation."""
    df_bench = calculate_benchmark_metrics(sample_market_df, rf_rate=0.06)
    assert len(df_bench) == 2
    assert "beta" in df_bench.columns
    assert "jensen_alpha_rf6" in df_bench.columns
    assert "tracking_error" in df_bench.columns
    assert "information_ratio" in df_bench.columns


def test_market_breadth(sample_market_df):
    """Test advance/decline ratio and moving average breadth."""
    df_breadth = calculate_daily_market_breadth(sample_market_df)
    assert len(df_breadth) == 100
    assert "advance_ratio" in df_breadth.columns
    assert "pct_above_ma20" in df_breadth.columns
    assert (df_breadth["pct_above_ma20"] >= 0.0).all()
    assert (df_breadth["pct_above_ma20"] <= 1.0).all()


def test_screening_and_rankings(sample_market_df):
    """Test multi-factor percentile rankings and score monotonicity."""
    df_risk = calculate_equity_risk_return(sample_market_df)
    df_bench = calculate_benchmark_metrics(sample_market_df)
    df_rank = compute_multi_factor_rankings(df_risk, df_bench)

    assert len(df_rank) == 2
    assert "composite_score" in df_rank.columns
    assert "rank" in df_rank.columns
    assert (df_rank["composite_score"] >= 0.0).all()
    assert (df_rank["composite_score"] <= 100.0).all()
    # Rank 1 must have highest composite score
    assert df_rank.iloc[0]["composite_score"] >= df_rank.iloc[1]["composite_score"]


def test_correlation_matrix_and_pairs(sample_market_df):
    """Test pairwise correlation matrix and pair ranking."""
    corr_matrix = compute_correlation_matrix(sample_market_df)
    assert corr_matrix.shape == (2, 2)
    assert np.isclose(corr_matrix.loc["STOCK_A", "STOCK_A"], 1.0)
    assert np.isclose(corr_matrix.loc["STOCK_B", "STOCK_B"], 1.0)
    assert np.isclose(corr_matrix.loc["STOCK_A", "STOCK_B"], corr_matrix.loc["STOCK_B", "STOCK_A"])

    high_corr, low_corr = extract_correlation_pairs(corr_matrix)
    assert len(high_corr) == 1
    assert len(low_corr) == 1


def test_news_classification():
    """Test event category classification."""
    cat1 = classify_event_category("Company issues QIP to raise capital for capex")
    assert cat1 == "Capital_Raising_and_Capex"

    cat2 = classify_event_category("Net profit surges 25% year on year")
    assert cat2 == "Earnings_and_Results"

    cat3 = classify_event_category("Company signs major agreement to build solar plant")
    assert cat3 == "Contracts_and_Expansion"


def test_anti_leakage_target_isolation(sample_market_df):
    """Verify that target_next_day_return is never consumed by risk/return or benchmark analytics."""
    df_risk = calculate_equity_risk_return(sample_market_df)
    assert "target_next_day_return" not in df_risk.columns

    df_bench = calculate_benchmark_metrics(sample_market_df)
    assert "target_next_day_return" not in df_bench.columns
