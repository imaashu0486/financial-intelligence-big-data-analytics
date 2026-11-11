"""
Multi-Factor Stock Screening and Ranking Page (Step 9 Phase C).
Provides transparent, configurable multi-factor stock screening across all 50 NIFTY equities.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.services.analytics_service import AnalyticsService
from src.analytics.screening import compute_multi_factor_rankings


def main() -> None:
    setup_page("Stock Screening", icon="🎯")
    render_sidebar(active_page="Stock Screening")

    render_terminal_header(
        title="MULTI-FACTOR EQUITY SCREENING & RANKINGS",
        subtitle="CONFIGURABLE QUANTITATIVE FACTOR SCORING ACROSS 50 NIFTY EQUITIES",
        badge="NIFTY 50 SCREENER // TRANSPARENT PERCENTILE RANKS",
    )

    st.info(
        "📌 **Academic Screening Disclaimer**: This module provides configurable quantitative multi-factor ranking "
        "computed on 10-year historical metrics (2015–2024). It is a historical screening and educational research tool, "
        "not a live market feed or an automated buy/sell recommendation."
    )

    analytics_svc = AnalyticsService()
    df_risk = analytics_svc.get_real_risk_return_metrics()
    df_bench = analytics_svc.get_real_benchmark_metrics()

    if df_risk.empty or df_bench.empty:
        st.warning("Underlying metrics not found. Please verify Step 8 analytics exports.")
        return

    # Sidebar / Expander Factor Weight Configuration
    st.markdown("### ⚙️ Factor Weight Configuration")
    st.markdown(
        "*Adjust the relative factor importance. Weights are automatically normalized to sum to 100%.*"
    )

    col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
    with col_w1:
        w_sharpe = st.slider("Sharpe Weight", 0.0, 1.0, 0.25, 0.05, help="Risk-adjusted return.")
    with col_w2:
        w_vol = st.slider("Low Volatility Weight", 0.0, 1.0, 0.20, 0.05, help="Reward lower annualized volatility.")
    with col_w3:
        w_dd = st.slider("Drawdown Defense", 0.0, 1.0, 0.20, 0.05, help="Reward lower max peak-to-trough drop.")
    with col_w4:
        w_alpha = st.slider("Jensen's Alpha", 0.0, 1.0, 0.20, 0.05, help="Excess return over CAPM.")
    with col_w5:
        w_cons = st.slider("Return Consistency", 0.0, 1.0, 0.15, 0.05, help="Percentage of positive trading days.")

    weights = {
        "sharpe": w_sharpe,
        "volatility": w_vol,
        "drawdown": w_dd,
        "alpha": w_alpha,
        "consistency": w_cons,
    }

    # Recompute rankings live
    df_ranked = compute_multi_factor_rankings(df_risk, df_bench, weights=weights)

    # Top 3 Ranked Equities Highlights
    top_1 = df_ranked.iloc[0]
    top_2 = df_ranked.iloc[1]
    top_3 = df_ranked.iloc[2]

    metric_cards = [
        {
            "label": "Rank #1 Leader",
            "value": f"{top_1['symbol']} (Score: {top_1['composite_score']:.1f})",
            "delta": f"Sharpe: {top_1['sharpe_ratio_rf6']:.2f} // Beta: {top_1['beta']:.2f}",
            "help": "Top-ranked equity according to current factor weight combination.",
        },
        {
            "label": "Rank #2 Contender",
            "value": f"{top_2['symbol']} (Score: {top_2['composite_score']:.1f})",
            "delta": f"Alpha: {top_2['jensen_alpha_rf6']*100:+.1f}%",
            "help": "Second ranked equity in the universe.",
        },
        {
            "label": "Rank #3 Contender",
            "value": f"{top_3['symbol']} (Score: {top_3['composite_score']:.1f})",
            "delta": f"Return: {top_3['annualized_return']*100:.1f}%",
            "help": "Third ranked equity in the universe.",
        },
    ]
    render_kpi_grid(metric_cards, columns=3)

    st.markdown("<br>", unsafe_allow_html=True)

    # Complete Ranking Table
    st.subheader("Interactive Stock Screening Results (50 Equities)")

    display_cols = [
        "rank", "symbol", "composite_score", "score_sharpe", "score_volatility",
        "score_drawdown", "score_alpha", "score_consistency",
        "annualized_return", "annualized_volatility", "sharpe_ratio_rf6", "beta", "jensen_alpha_rf6"
    ]
    df_disp = df_ranked[display_cols].copy()
    df_disp["annualized_return"] = (df_disp["annualized_return"] * 100).round(2).astype(str) + "%"
    df_disp["annualized_volatility"] = (df_disp["annualized_volatility"] * 100).round(2).astype(str) + "%"
    df_disp["sharpe_ratio_rf6"] = df_disp["sharpe_ratio_rf6"].round(3)
    df_disp["beta"] = df_disp["beta"].round(3)
    df_disp["jensen_alpha_rf6"] = (df_disp["jensen_alpha_rf6"] * 100).round(2).astype(str) + "%"

    st.dataframe(df_disp, use_container_width=True, height=500)

    csv = df_ranked.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Screened Rankings CSV", csv, "nifty50_screened_rankings.csv", "text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning(
        "⚠️ **Disclaimer**: Multi-factor scoring is a quantitative research tool that ranks historical performance factors. "
        "It does NOT constitute investment advice, financial planning, or a buy/sell endorsement for any security."
    )

    render_disclaimer()


if __name__ == "__main__":
    main()

