"""
Real-Data Advanced Financial Analytics Page (Step 9 Phase C).
Displays 10-year risk/return, tail-risk VaR/CVaR, and benchmark analytics for NIFTY 50.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.services.analytics_service import AnalyticsService


def main() -> None:
    setup_page("Real Financial Analytics", icon="📈")
    render_sidebar(active_page="Real Financial Analytics")

    render_terminal_header(
        title="ADVANCED FINANCIAL ANALYTICS & TAIL RISK",
        subtitle="10-YEAR RISK-RETURN, BENCHMARK ALPHA & DIVERSIFICATION (2015–2024)",
        badge="REAL NIFTY 50 // HDFS CLUSTER OUTPUTS",
    )

    analytics_svc = AnalyticsService()
    df_risk = analytics_svc.get_real_risk_return_metrics()
    df_bench = analytics_svc.get_real_benchmark_metrics()
    corr_matrix = analytics_svc.get_real_correlation_matrix()
    corr_pairs = analytics_svc.get_real_correlation_pairs()

    if df_risk.empty or df_bench.empty:
        st.warning("Real analytics datasets not found. Please ensure Step 8 Parquet outputs are exported.")
        return

    # Merge for unified analytics view
    df_merged = pd.merge(df_risk, df_bench, on="symbol", suffixes=("_risk", "_bench"))

    # Top KPI Metrics Row
    top_stock = df_merged.sort_values("annualized_return", ascending=False).iloc[0]
    top_sharpe = df_merged.sort_values("sharpe_ratio_rf6", ascending=False).iloc[0]
    top_alpha = df_merged.sort_values("jensen_alpha_rf6", ascending=False).iloc[0]
    min_beta = df_merged.sort_values("beta", ascending=True).iloc[0]

    metric_cards = [
        {
            "label": "Top Annualized Return",
            "value": f"{top_stock['annualized_return']*100:.1f}%",
            "delta": f"{top_stock['symbol']} (10Y CAGR: 44.1%)",
            "help": "Arithmetic annualized daily return (mean_daily * 252).",
        },
        {
            "label": "Highest Sharpe (Rf=6%)",
            "value": f"{top_sharpe['sharpe_ratio_rf6']:.2f}",
            "delta": f"{top_sharpe['symbol']} (Ret: {top_sharpe['annualized_return']*100:.1f}%)",
            "help": "Annualized Sharpe ratio with 6.0% Indian G-Sec risk-free rate.",
        },
        {
            "label": "Highest Jensen's Alpha",
            "value": f"+{top_alpha['jensen_alpha_rf6']*100:.1f}%",
            "delta": f"{top_alpha['symbol']} vs NIFTY 50",
            "help": "Abnormal annual excess return over CAPM expected return.",
        },
        {
            "label": "Most Defensive Equity",
            "value": f"Beta {min_beta['beta']:.2f}",
            "delta": f"{min_beta['symbol']} (Low Volatility)",
            "help": "Lowest systematic sensitivity to benchmark index movements.",
        },
    ]
    render_kpi_grid(metric_cards, columns=4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Modular Exploration
    tab_overview, tab_scatter, tab_corr, tab_single = st.tabs([
        "📊 Complete Analytics Table",
        "🎯 Risk vs Return Scatter Plot",
        "🔗 50x50 Correlation Matrix & Diversification",
        "🔍 Single-Stock Factor Deep Dive",
    ])

    with tab_overview:
        st.subheader("NIFTY 50 Complete Risk-Adjusted Metrics (2015–2024)")
        st.markdown(
            r"""
            *Data Source: Apache Spark distributed analytics persisted in HDFS (`/financial-intelligence/real/analytics/`).*  
            *Annualization assumption: 252 trading sessions/year. Baseline Risk-Free Rate $R_f = 6.0\%$ (Indian 10Y G-Sec). Benchmark metrics evaluated against 2,459 aligned NIFTY 50 sessions (`^NSEI`).*
            """
        )

        display_cols = [
            "symbol", "annualized_return", "annualized_volatility", "sharpe_ratio_rf6",
            "beta", "jensen_alpha_rf6", "tracking_error", "information_ratio", "positive_days_pct"
        ]
        df_display = df_merged[display_cols].copy()
        df_display["annualized_return"] = (df_display["annualized_return"] * 100).round(2).astype(str) + "%"
        df_display["annualized_volatility"] = (df_display["annualized_volatility"] * 100).round(2).astype(str) + "%"
        df_display["sharpe_ratio_rf6"] = df_display["sharpe_ratio_rf6"].round(3)
        df_display["beta"] = df_display["beta"].round(3)
        df_display["jensen_alpha_rf6"] = (df_display["jensen_alpha_rf6"] * 100).round(2).astype(str) + "%"
        df_display["tracking_error"] = (df_display["tracking_error"] * 100).round(2).astype(str) + "%"
        df_display["information_ratio"] = df_display["information_ratio"].round(3)
        df_display["positive_days_pct"] = df_display["positive_days_pct"].round(2).astype(str) + "%"

        st.dataframe(df_display, use_container_width=True, height=450)

        csv = df_merged.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Complete Metrics CSV", csv, "nifty50_risk_return_benchmark.csv", "text/csv")

    with tab_scatter:
        st.subheader("Risk vs Return Frontier (Annualized Volatility vs Return)")
        fig = px.scatter(
            df_merged,
            x="annualized_volatility",
            y="annualized_return",
            text="symbol",
            color="sharpe_ratio_rf6",
            size="positive_days_pct",
            color_continuous_scale="Viridis",
            labels={
                "annualized_volatility": "Annualized Volatility (Risk)",
                "annualized_return": "Annualized Return",
                "sharpe_ratio_rf6": "Sharpe (Rf=6%)",
            },
            title="NIFTY 50 Risk-Return Profile (Color = Sharpe Ratio, Size = Win Rate)",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        st.subheader("50x50 Pairwise Daily Return Correlation Matrix")
        if not corr_matrix.empty:
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu_r",
                zmin=-0.2, zmax=1.0,
                title="Pairwise Correlation Heatmap (1,225 Unique Asset Pairs across 2015–2024)",
            )
            fig_corr.update_layout(height=700, template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)

        col_h, col_l = st.columns(2)
        with col_h:
            st.markdown("#### 🔗 Highest Correlated Pairs (Industry Clustering)")
            if corr_pairs and "highest_correlated_pairs" in corr_pairs:
                df_high = pd.DataFrame(corr_pairs["highest_correlated_pairs"])
                st.dataframe(df_high, use_container_width=True)

        with col_l:
            st.markdown("#### 🛡️ Optimal Diversification Pairs (Lowest Correlation)")
            if corr_pairs and "lowest_correlated_pairs" in corr_pairs:
                df_low = pd.DataFrame(corr_pairs["lowest_correlated_pairs"])
                st.dataframe(df_low, use_container_width=True)

    with tab_single:
        st.subheader("Single-Stock Deep Dive")
        selected_stock = st.selectbox("Select NIFTY 50 Stock:", sorted(df_merged["symbol"].unique()), index=0)
        stock_row = df_merged[df_merged["symbol"] == selected_stock].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualized Return", f"{stock_row['annualized_return']*100:.2f}%")
        c2.metric("Annualized Volatility", f"{stock_row['annualized_volatility']*100:.2f}%")
        c3.metric("Sharpe Ratio (Rf=6%)", f"{stock_row['sharpe_ratio_rf6']:.3f}")
        c4.metric("CAPM Beta (vs NIFTY)", f"{stock_row['beta']:.3f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Jensen's Alpha", f"{stock_row['jensen_alpha_rf6']*100:.2f}%")
        c6.metric("Tracking Error", f"{stock_row['tracking_error']*100:.2f}%")
        c7.metric("Information Ratio", f"{stock_row['information_ratio']:.3f}")
        c8.metric("Positive Days %", f"{stock_row['positive_days_pct']:.2f}%")

        st.info(
            f"**Econometric Profile for {selected_stock}**:\n"
            f"- Sensitivity to NIFTY 50 index: $\\beta = {stock_row['beta']:.3f}$. "
            f"{'Higher sensitivity than the market (cyclical/aggressive).' if stock_row['beta'] > 1.0 else 'Lower sensitivity than the market (defensive/stabilizer).'}\n"
            f"- Annualized excess risk-adjusted performance (Jensen's Alpha): `{stock_row['jensen_alpha_rf6']*100:+.2f}%`.\n"
            f"- Tracking active divergence vs NIFTY: `{stock_row['tracking_error']*100:.2f}%` annual tracking error with information ratio `{stock_row['information_ratio']:.3f}`."
        )

    render_disclaimer()


if __name__ == "__main__":
    main()

