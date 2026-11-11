"""
Market Regime and Breadth Analytics Page (Step 9 Phase C).
Visualizes 10-year market-wide breadth dynamics, constituent participation, and crisis regimes.
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
    setup_page("Market Regimes", icon="🌐")
    render_sidebar(active_page="Market Regimes")

    render_terminal_header(
        title="MARKET REGIME & BREADTH DYNAMICS",
        subtitle="10-YEAR PARTICIPATION, MOVING AVERAGE BREADTH & STRESS REGIMES (2015–2024)",
        badge="REAL NIFTY 50 // 2,467 SESSIONS",
    )

    analytics_svc = AnalyticsService()
    df_breadth = analytics_svc.get_real_market_breadth()

    if df_breadth.empty:
        st.warning("Market breadth dataset not found. Please verify Step 8 Parquet outputs.")
        return

    df_breadth["date"] = pd.to_datetime(df_breadth["date"])
    df_breadth = df_breadth.sort_values("date").reset_index(drop=True)

    # Top KPI Metrics
    mean_adv = df_breadth["advance_ratio"].mean() * 100
    mean_ma200 = df_breadth["pct_above_ma200"].mean() * 100
    expansion_days = int((df_breadth["pct_above_ma200"] > 0.80).sum())
    stress_days = int((df_breadth["pct_above_ma200"] < 0.20).sum())

    metric_cards = [
        {
            "label": "Mean Advance Ratio",
            "value": f"{mean_adv:.1f}%",
            "delta": "10-Year Baseline",
            "help": "Average daily percentage of advancing stocks across 2,467 sessions.",
        },
        {
            "label": "Mean Above 200d MA",
            "value": f"{mean_ma200:.1f}%",
            "delta": "Long-Term Trend",
            "help": "Average percentage of stocks trading above their 200-day moving average.",
        },
        {
            "label": "Expansion Regime Days",
            "value": f"{expansion_days} Days",
            "delta": f"{expansion_days / len(df_breadth) * 100:.1f}% of History",
            "help": "Sessions with >80% of constituent equities above their 200-day MA.",
        },
        {
            "label": "Acute Distress Days",
            "value": f"{stress_days} Days",
            "delta": f"{stress_days / len(df_breadth) * 100:.1f}% of History",
            "help": "Sessions with <20% of constituent equities above their 200-day MA (e.g. COVID crash).",
        },
    ]
    render_kpi_grid(metric_cards, columns=4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Breadth Chart
    st.subheader("Constituent Moving Average Participation (Breadth Wave)")
    st.markdown(
        """
        *Monitors the percentage of NIFTY 50 equities trading strictly above their 20-day, 50-day, and 200-day simple moving averages.*
        """
    )

    fig_breadth = go.Figure()
    fig_breadth.add_trace(go.Scatter(
        x=df_breadth["date"], y=df_breadth["pct_above_ma200"] * 100,
        mode="lines", name="% Above 200-Day MA (Macro Trend)", line=dict(color="#2962FF", width=2)
    ))
    fig_breadth.add_trace(go.Scatter(
        x=df_breadth["date"], y=df_breadth["pct_above_ma50"] * 100,
        mode="lines", name="% Above 50-Day MA (Intermediate)", line=dict(color="#00C805", width=1.5, dash="dash")
    ))
    fig_breadth.add_trace(go.Scatter(
        x=df_breadth["date"], y=df_breadth["pct_above_ma20"] * 100,
        mode="lines", name="% Above 20-Day MA (Short-Term)", line=dict(color="#FF9800", width=1, dash="dot")
    ))

    # Add Regime Threshold Shading
    fig_breadth.add_hline(y=80, line_dash="dot", line_color="#00C805", annotation_text="Euphoric Expansion (>80%)")
    fig_breadth.add_hline(y=20, line_dash="dot", line_color="#FF3B30", annotation_text="Acute Distress (<20%)")

    fig_breadth.update_layout(
        height=500,
        yaxis_title="% of NIFTY 50 Stocks Above MA",
        xaxis_title="Date",
        template="plotly_dark",
        hovermode="x unified",
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_breadth, use_container_width=True)

    # Historical Drawdown & Crisis Focus
    st.subheader("Historical Market Crisis Highlights")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div style="background-color: #1c1517; border-left: 4px solid #FF3B30; padding: 12px; border-radius: 4px; font-size: 0.85rem;">
                <h5 style="color: #FF5252; margin: 0 0 6px 0;">🔴 March 2020: The COVID-19 Liquidity Shock</h5>
                <p style="margin: 0; color: #E0E3EB;">
                    On <strong>March 23, 2020</strong>, NIFTY 50 200-day MA breadth collapsed to a record all-time low of <strong>4.0%</strong> (only 2 out of 50 equities remained above their 200d MA). Daily advance ratio plummeted to 2.0%, and rolling 21-day annualized volatility spiked to <strong>84.6%</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div style="background-color: #122116; border-left: 4px solid #00C805; padding: 12px; border-radius: 4px; font-size: 0.85rem;">
                <h5 style="color: #69F0AE; margin: 0 0 6px 0;">🟢 2020–2021 & 2023–2024: Broad Liquidity Expansions</h5>
                <p style="margin: 0; color: #E0E3EB;">
                    Following monetary intervention, 200-day MA breadth staged a violent V-shaped recovery to <strong>98.0%</strong> by Dec 2020. During calendar years 2023–2024, constituent participation remained consistently in the 75%–90% band as benchmark indices logged repeated new highs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**Methodological & Descriptive Notice**: Market breadth metrics describe the historical degree of constituent participation. "
        "They are descriptive diagnostic indicators and do not constitute automated predictive timing rules or investment recommendations."
    )

    render_disclaimer()


if __name__ == "__main__":
    main()

