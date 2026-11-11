"""
Home landing page for the Financial Intelligence and Big Data Analytics Platform.
National Institute of Technology Delhi — CSBB 422 Big Data Analytics Lab.
"""

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.services.data_service import DataService
from app.services.analytics_service import AnalyticsService
from app.services.sentiment_service import SentimentService
from app.services.mining_service import MiningService
from app.services.rag_service import RAGService
from app.services.ml_service import MLService


def main() -> None:
    setup_page("Executive Terminal", icon="🏢")
    render_sidebar(active_page="Home")

    # Scope detection
    scope = st.session_state.get("dataset_scope", "Real Market History (NIFTY 50)")
    is_real = "Real Market" in scope

    badge_text = "NIT DELHI // CSBB 422 // REAL NIFTY 50 PIPELINE" if is_real else "NIT DELHI // CSBB 422 // SYNTHETIC BASELINE"
    subtitle_text = (
        "10-YEAR HISTORICAL REAL BIG DATA ANALYTICS & MACHINE LEARNING PLATFORM"
        if is_real
        else "END-TO-END BIG DATA ANALYTICS & MARKET INTELLIGENCE SUITE"
    )

    render_terminal_header(
        title="EXECUTIVE FINANCIAL INTELLIGENCE DASHBOARD",
        subtitle=subtitle_text,
        badge=badge_text,
    )

    # Initialize services
    data_svc = DataService()
    analytics_svc = AnalyticsService()
    sentiment_svc = SentimentService()
    mining_svc = MiningService()
    rag_svc = RAGService()
    ml_svc = MLService()

    if is_real:
        # Load Real Market stats
        real_kpis = analytics_svc.get_real_market_kpis()
        real_ml_metrics = ml_svc.get_real_metrics()
        real_news = analytics_svc.get_real_news_intelligence()
        cluster_info = analytics_svc.get_cluster_architecture()

        clf_acc = real_ml_metrics.get("classification", {}).get("logistic_regression", {}).get("test_accuracy", 0.5074)
        reg_r2 = real_ml_metrics.get("regression", {}).get("ridge", {}).get("test_r2", 0.0023)

        metric_cards = [
            {
                "label": "Total Market Records",
                "value": f"{real_kpis['total_records']:,}",
                "delta": "2015–2024 (10 Years)",
                "help": "Consolidated daily OHLCV & technical feature records across 50 NIFTY equities.",
            },
            {
                "label": "Tracked Equities",
                "value": f"{real_kpis['symbols_count']} Stocks",
                "delta": "NIFTY 50 Blue-Chips",
                "help": "All 50 active National Stock Exchange (NSE) constituent equities.",
            },
            {
                "label": "Benchmark Sessions",
                "value": f"{real_kpis['benchmark_sessions']:,}",
                "delta": "^NSEI 10-Yr History",
                "help": "Official NIFTY 50 benchmark trading days aligned chronologically.",
            },
            {
                "label": "HDFS Cluster Storage",
                "value": "77 Files",
                "delta": "Replication: 2 (2 Worker Nodes)",
                "help": "Raw and processed datasets stored in HDFS across 2 DataNodes.",
            },
            {
                "label": "Real ML Directional Acc",
                "value": f"{clf_acc*100:.2f}%",
                "delta": f"Test R²: {reg_r2:+.4f}",
                "help": "Unseen 2024 chronological test set evaluation (EMH bounded signal).",
            },
        ]
        render_kpi_grid(metric_cards, columns=5)

        st.markdown("<br>", unsafe_allow_html=True)

        col_arch, col_phases = st.columns([0.55, 0.45])

        with col_arch:
            st.markdown(
                f"""
                <div class="terminal-card">
                    <h4 style="color: #00C805; margin-top: 0;">🏛️ Distributed Cluster Topology (Real Data)</h4>
                    <p style="font-size: 0.85rem; color: #A0A5B5;">
                        <strong>Academic Verification Statement:</strong> {cluster_info['academic_statement']}
                    </p>
                    <div style="font-family: monospace; font-size: 0.78rem; background-color: #0E1117; padding: 12px; border-radius: 4px; border: 1px solid #2A2E39; line-height: 1.5; color: #E0E3EB;">
[DATA INGESTION]  ➔ 50 NIFTY Equities (121,962 rows) + Benchmark (2,459 rows)<br>
      │<br>
[BIG DATA ENGINE] ➔ Hadoop 3.3.6 (HDFS Rep=2) + Apache Spark 3.5.1<br>
      │            ├─ hadoop-master: NameNode (9870), ResourceManager (8088), Spark Master (8080)<br>
      │            ├─ hadoop-worker1: DataNode (9864), NodeManager, Spark Worker (8081)<br>
      │            └─ hadoop-worker2: DataNode (9865), NodeManager, Spark Worker (8082)<br>
      │<br>
[SPARK ANALYTICS] ➔ Distributed Execution (app-20261110170818-0001) ➔ 4 Parquet Tables in HDFS<br>
      │            ├─ Risk & Return Profile (50 stocks)<br>
      │            ├─ Market Breadth & Advance/Decline (2,459 days)<br>
      │            ├─ Pairwise Return Correlation Matrix (50x50)<br>
      │            └─ Composite Multi-Factor Stock Rankings<br>
      │<br>
[REAL MACHINE LEARNING] ➔ Date Split (Train 2015-21, Val 2022-23, Test 2024)<br>
      │                  ├─ Binary Direction: LogReg (50.74%), RF (50.61%), GBDT (50.70%)<br>
      │                  └─ Return Regression: Ridge (R²=0.0023), RF (R²=0.0293)<br>
      │<br>
[ENTERPRISE PERSISTENCE] ➔ financial-mongodb (27017) + Streamlit Terminal UI
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_phases:
            st.markdown(
                """
                <div class="terminal-card">
                    <h4 style="color: #00C805; margin-top: 0;">✅ Lab Project Implementation Lifecycle</h4>
                    <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #2A2E39; color: #787B86;">
                            <th style="padding: 6px 0; text-align: left;">Step</th>
                            <th style="padding: 6px 0; text-align: left;">Scope & Milestone</th>
                            <th style="padding: 6px 0; text-align: right;">Verification</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Steps 1–4</td>
                            <td>Synthetic Baseline & Platform</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (Preserved)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Step 5 / 5.1</td>
                            <td>Real NIFTY 50 Data & Quality Audit</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (100% Identity)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Step 6</td>
                            <td>Real Preprocessing & Features</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (121,962 rows)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Step 7.3</td>
                            <td>HDFS Ingestion & Cluster Verif.</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (Rep=2, YARN)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Step 8 / 8.1</td>
                            <td>Advanced Distributed Analytics</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (4 HDFS Tables)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Step 8.1.1</td>
                            <td>Targeted Audit & Holiday Reconciliation</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (8 Dates Reconciled)</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0;">Step 9</td>
                            <td>Real ML Pipeline & Full Integration</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (End-to-End)</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🧭 Real Market Historical Intelligence Modules")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #00C805; margin-top: 0;">📈 7. Real Financial Analytics</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Distributed risk/return metrics, benchmark alpha, beta, Sharpe ratios, and full 50×50 pairwise correlation matrix.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>7_📈_Real_Financial_Analytics</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #00C805; margin-top: 0;">🌐 8. Market Regime & Breadth</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Advance-decline ratios, moving average participation waves (% > MA50/200), and COVID March 2020 crash deep-dive.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>8_🌐_Market_Regime_&_Breadth</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #00C805; margin-top: 0;">🎯 9. Stock Screening & Rankings</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Interactive historical multi-factor screening (Sharpe, Volatility, Drawdown, Alpha), user weight configuration, and CSV export.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>9_🎯_Stock_Screening_&_Rankings</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c4, c5 = st.columns(2)
        with c4:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #00C805; margin-top: 0;">🔬 10. Real ML Models</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Chronological out-of-sample models (LogReg, RF, GBDT, Ridge), feature importance trees, and interactive single-sample predictor.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>10_🔬_Real_ML_Models</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #00C805; margin-top: 0;">🗞️ 11. Real News & Filings</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        2024 corporate news analysis with strict non-causality correlation tests, plus verified FY24 MD&A annual filing extracts.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>11_🗞️_Real_News_&_Filings</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # Load Synthetic Baseline stats
        kpis = data_svc.get_market_kpis()
        sentiment_stats = sentiment_svc.get_sentiment_analytics()
        mining_stats = mining_svc.get_mining_statistics()
        rag_status = rag_svc.get_index_status()

        metric_cards = [
            {
                "label": "Total Market Records",
                "value": f"{kpis['total_records']:,}",
                "delta": f"{kpis['trading_days']} Trading Days",
                "help": "Total chronological daily OHLCV records across 10 equities.",
            },
            {
                "label": "Tracked Equities",
                "value": f"{kpis['symbols_count']} Stocks",
                "delta": "NSE/BSE Large-Cap",
                "help": "10 major Indian blue-chip stocks.",
            },
            {
                "label": "News Articles Analyzed",
                "value": f"{sentiment_stats.get('pipeline_config', {}).get('total_articles', 500):,}",
                "delta": "Financial Lexicon NLP",
                "help": "Articles processed through financial domain lexicon.",
            },
            {
                "label": "Mined Association Rules",
                "value": f"{mining_stats.get('num_rules', mining_stats.get('total_rules_generated', 0))} Rules",
                "delta": "Lift > 1.10",
                "help": "Frequent market co-movement rules discovered by Apriori.",
            },
            {
                "label": "Indexed RAG Chunks",
                "value": f"{rag_status.get('unique_chunk_count', rag_status.get('chunk_count', 28))} Chunks",
                "delta": f"{len(rag_status['indexed_documents'])} Documents",
                "help": "Vector embeddings for financial reports and risk disclosures.",
            },
        ]
        render_kpi_grid(metric_cards, columns=5)

        st.markdown("<br>", unsafe_allow_html=True)

        col_arch, col_phases = st.columns([0.55, 0.45])

        with col_arch:
            st.markdown(
                """
                <div class="terminal-card">
                    <h4 style="color: #2962FF; margin-top: 0;">🏛️ Synthetic Baseline Topology</h4>
                    <p style="font-size: 0.85rem; color: #A0A5B5;">
                        Original benchmark implementation preserved for reproducibility and comparative baseline studies.
                    </p>
                    <div style="font-family: monospace; font-size: 0.78rem; background-color: #0E1117; padding: 12px; border-radius: 4px; border: 1px solid #2A2E39; line-height: 1.5; color: #E0E3EB;">
[DATA INGESTION]  ➔ Market Data (5,200 rows) + News (500 articles)<br>
      │<br>
[BIG DATA ENGINE] ➔ Hadoop MapReduce (Volume/Words) + Apache Spark 3.5.1<br>
      │            ├─ Data Lake: HDFS Block Storage (Replication Factor: 2)<br>
      │            └─ Cluster: 1 Master Node + 2 Worker Nodes (YARN / Standalone)<br>
      │<br>
[ANALYTICS & ML]  ➔ ML Models (Linear/RF Regression, Direction Classifier, K-Means)<br>
      │            ├─ Sentiment NLP: Fin-Lexicon + Regime Correlation<br>
      │            └─ Association Rules: Apriori Market Co-Movement<br>
      │<br>
[FINANCIAL RAG]   ➔ Vector Store (all-MiniLM-L6-v2) + Semantic Search + Extractive QA<br>
      │<br>
[PERSISTENCE]     ➔ MongoDB NoSQL Database (Port 27017)<br>
[FRONTEND UI]     ➔ Streamlit Dark Terminal + Plotly Interactive Suite
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_phases:
            st.markdown(
                """
                <div class="terminal-card">
                    <h4 style="color: #00C805; margin-top: 0;">✅ Synthetic Baseline Modules</h4>
                    <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #2A2E39; color: #787B86;">
                            <th style="padding: 6px 0; text-align: left;">Phase</th>
                            <th style="padding: 6px 0; text-align: left;">Capability</th>
                            <th style="padding: 6px 0; text-align: right;">Status</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Phase 1</td>
                            <td>Data Ingestion & Feature Eng.</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (5,200 rows)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Phase 2</td>
                            <td>Hadoop MapReduce & PySpark</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (4 Parquet DS)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Phase 3</td>
                            <td>MongoDB NoSQL Integration</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (Collections)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Phase 4</td>
                            <td>Machine Learning Analytics</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (Reg/Clf/K-Means)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Phase 5</td>
                            <td>News Sentiment & Regimes</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (500 Arts / FinNLP)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Phase 6</td>
                            <td>Apriori Association Mining</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (16 Rules)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #2A2E39;">
                            <td style="padding: 6px 0;">Phase 7</td>
                            <td>Financial Document RAG</td>
                            <td style="text-align: right; color: #00C805; font-weight: 600;">PASS (28 Chunks)</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🧭 Synthetic Baseline Navigation")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #2962FF; margin-top: 0;">📊 1. Market Overview</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Interactive candlestick charts, moving average overlays, daily returns, and volume analysis for 10 Indian equities.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>1_📊_Market_Overview</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #2962FF; margin-top: 0;">⚡ 2. Big Data Analytics</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Hadoop MapReduce volume totals & news terms, PySpark Parquet aggregations, rankings, and cluster node specifications.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>2_⚡_Big_Data_Analytics</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #2962FF; margin-top: 0;">🤖 3. Machine Learning</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Chronological test evaluation for Linear/RF Regression, Random Forest Directional Classifier, and K-Means risk clustering.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>3_🤖_Machine_Learning</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #2962FF; margin-top: 0;">📰 4. Sentiment Intelligence</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Financial NLP lexicon analysis, article sentiment distribution, daily sentiment timelines, and correlation with asset returns.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>4_📰_Sentiment_Intelligence</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #2962FF; margin-top: 0;">🔗 5. Association Rule Mining</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Apriori market basket co-movement discovery: support, confidence, and lift metrics for joint price-movement rules.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>5_🔗_Association_Rules</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown(
                """
                <div class="terminal-card" style="height: 160px;">
                    <h5 style="color: #2962FF; margin-top: 0;">📑 6. Financial Document RAG</h5>
                    <p style="font-size: 0.8rem; color: #A0A5B5;">
                        Vector search (SentenceTransformer) + grounded QA over annual reports and risk disclosures with citations.
                    </p>
                    <div style="font-size: 0.75rem; color: #787B86;">Sidebar: <strong>6_📑_Financial_RAG</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_disclaimer()


if __name__ == "__main__":
    main()
