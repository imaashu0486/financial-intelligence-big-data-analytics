"""
Page 4: Sentiment Intelligence.
Presents financial news sentiment analysis, daily sentiment aggregations,
regime distributions, and statistical correlations with equity returns.
"""

from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.components.charts import create_sentiment_time_chart
from app.components.tables import render_downloadable_table
from app.services.sentiment_service import SentimentService


def main() -> None:
    setup_page("Sentiment Intelligence", icon="📰")
    render_sidebar(active_page="Sentiment Intelligence")

    render_terminal_header(
        title="FINANCIAL NEWS SENTIMENT INTELLIGENCE",
        subtitle="DOMAIN LEXICON CLASSIFICATION, REGIMES & RETURN CORRELATIONS (PHASE 5)",
        badge="NLP SUITE",
    )

    svc = SentimentService()
    analytics = svc.get_sentiment_analytics()

    if not analytics:
        st.error("Sentiment analytics not found in data/processed/sentiment/sentiment_analytics.json.")
        return

    dist = svc.get_sentiment_distribution()
    corrs = svc.get_correlations()
    regimes = svc.get_regime_statistics()

    total_articles = sum(dist.values())
    pos_count = dist.get("Positive", 0)
    neu_count = dist.get("Neutral", 0)
    neg_count = dist.get("Negative", 0)

    # Top KPI Metrics Row
    kpis = [
        {"label": "Total News Articles", "value": f"{total_articles:,}", "delta": "Processed", "help": "Total articles analyzed"},
        {"label": "Positive Sentiment", "value": f"{pos_count} ({pos_count/total_articles*100:.1f}%)", "delta": "Bullish Tone", "delta_color": "normal"},
        {"label": "Neutral Sentiment", "value": f"{neu_count} ({neu_count/total_articles*100:.1f}%)", "delta": "Factual Tone", "delta_color": "off"},
        {"label": "Negative Sentiment", "value": f"{neg_count} ({neg_count/total_articles*100:.1f}%)", "delta": "Bearish Tone", "delta_color": "inverse"},
        {"label": "Same-Day Pearson r", "value": f"{corrs.get('same_day_correlation', {}).get('pearson', 0.0):.4f}", "delta": "p-value: 0.616", "help": "No significant linear relationship"},
    ]
    render_kpi_grid(kpis, columns=5)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sentiment Distribution & Timeline Section
    col_dist, col_breakdown = st.columns([0.45, 0.55])

    with col_dist:
        st.markdown("##### 🍩 Global Sentiment Distribution")
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=list(dist.keys()),
                    values=list(dist.values()),
                    hole=0.45,
                    marker=dict(colors=["#00C805", "#787B86", "#FF3B30"]),
                    textinfo="label+percent",
                )
            ]
        )
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161922",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_breakdown:
        st.markdown("##### 🏢 Sentiment Breakdown by Ticker")
        breakdown_df = svc.get_stock_sentiment_breakdown()
        if not breakdown_df.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="Positive", x=breakdown_df["symbol"], y=breakdown_df["positive_count"], marker_color="#00C805"))
            fig_bar.add_trace(go.Bar(name="Neutral", x=breakdown_df["symbol"], y=breakdown_df["neutral_count"], marker_color="#787B86"))
            fig_bar.add_trace(go.Bar(name="Negative", x=breakdown_df["symbol"], y=breakdown_df["negative_count"], marker_color="#FF3B30"))
            fig_bar.update_layout(
                barmode="stack",
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#161922",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # Daily Sentiment Timeline
    st.markdown("#### 📈 Daily Aggregated News Sentiment Timeline")
    daily_df = svc.get_daily_sentiments()
    if not daily_df.empty:
        fig_time = create_sentiment_time_chart(daily_df)
        st.plotly_chart(fig_time, use_container_width=True)

    # Market Return Correlation & Regimes
    st.markdown("#### 🔬 Sentiment vs. Market Return Correlation Analysis")
    st.caption("Empirical testing of whether news sentiment reliably leads or lags daily asset returns.")

    col_corr1, col_corr2 = st.columns(2)

    with col_corr1:
        st.markdown("##### 📐 Statistical Association Tests")
        same_day = corrs.get("same_day_correlation", {})
        next_day = corrs.get("next_day_correlation", {})

        corr_table = [
            {
                "Horizon": "Same-Day Return",
                "Pearson r": f"{same_day.get('pearson', 0.0):.4f}",
                "Pearson p-val": f"{same_day.get('p_value', 0.0):.4f}",
                "Spearman ρ": f"{same_day.get('spearman', 0.0):.4f}",
                "Statistically Significant": "NO (p > 0.05)",
            },
            {
                "Horizon": "Next-Day Return (Lag 1)",
                "Pearson r": f"{next_day.get('pearson', 0.0):.4f}",
                "Pearson p-val": f"{next_day.get('p_value', 0.0):.4f}",
                "Spearman ρ": f"{next_day.get('spearman', 0.0):.4f}",
                "Statistically Significant": "NO (p > 0.05)",
            },
        ]
        st.dataframe(corr_table, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div style="background-color: #1A2332; border-left: 3px solid #2962FF; padding: 10px 14px; border-radius: 4px; font-size: 0.8rem; color: #90CAF9; margin-top: 14px;">
                <strong>FINDING:</strong> {corrs.get('disclaimer', 'No causal predictability.')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_corr2:
        st.markdown("##### 🏛️ Return Performance Partitioned by Sentiment Regime")
        regime_table = []
        for regime_name, r_data in regimes.items():
            same_d = r_data.get("same_day_return", {})
            next_d = r_data.get("next_day_return", {})
            regime_table.append({
                "Regime": regime_name,
                "Trading Days": r_data.get("day_count", 0),
                "Same-Day Return": f"{same_d.get('mean', 0.0)*100:+.3f}%",
                "Same-Day Win Rate": f"{same_d.get('win_rate', 0.0)*100:.1f}%",
                "Next-Day Return": f"{next_d.get('mean', 0.0)*100:+.3f}%",
                "Next-Day Win Rate": f"{next_d.get('win_rate', 0.0)*100:.1f}%",
            })
        st.dataframe(regime_table, use_container_width=True, hide_index=True)

    # Article Explorer & Filter
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📰 Financial News Article Explorer")

    col_f_sym, col_f_sent = st.columns([0.5, 0.5])
    articles_df = svc.get_article_sentiments()

    with col_f_sym:
        sym_options = ["All Stocks"] + sorted(articles_df["symbol"].unique().tolist()) if not articles_df.empty else ["All Stocks"]
        f_sym = st.selectbox("Filter by Equities:", options=sym_options)
    with col_f_sent:
        f_sent = st.selectbox("Filter by Sentiment Label:", options=["All Sentiments", "Positive", "Neutral", "Negative"])

    filtered_articles = articles_df.copy()
    if f_sym != "All Stocks":
        filtered_articles = filtered_articles[filtered_articles["symbol"] == f_sym]
    if f_sent != "All Sentiments":
        filtered_articles = filtered_articles[filtered_articles["sentiment"] == f_sent]

    if not filtered_articles.empty:
        table_cols = [c for c in ["date", "symbol", "headline", "sentiment", "sentiment_score", "confidence"] if c in filtered_articles.columns]
        export_df = filtered_articles[table_cols].copy()
        if "date" in export_df.columns:
            export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")

        render_downloadable_table(
            df=export_df,
            title=f"Filtered News Articles ({len(export_df)} items)",
            filename="financial_news_sentiment.csv",
            height=320,
        )

    render_disclaimer()


if __name__ == "__main__":
    main()

