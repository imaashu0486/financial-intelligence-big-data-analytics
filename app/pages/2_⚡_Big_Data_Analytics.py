"""
Page 2: Big Data Analytics.
Presents distributed Hadoop MapReduce and PySpark batch processing results,
execution modes, Parquet data lake analytics, and cluster architecture.
"""

import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.components.tables import render_downloadable_table
from app.services.analytics_service import AnalyticsService


def main() -> None:
    setup_page("Big Data Analytics", icon="⚡")
    render_sidebar(active_page="Big Data Analytics")

    scope = st.session_state.get("dataset_scope", "Real Market History (NIFTY 50)")
    is_real = "real" in scope.lower()

    render_terminal_header(
        title="BIG DATA ANALYTICS & DISTRIBUTED PROCESSING",
        subtitle="10-YEAR DISTRIBUTED MAPREDUCE & SPARK CLUSTER ANALYTICS (2015–2024)" if is_real else "HADOOP MAPREDUCE, APACHE SPARK & PARQUET DATA LAKE (PHASE 2)",
        badge="REAL NIFTY 50" if is_real else "SYNTHETIC BASELINE",
    )

    svc = AnalyticsService()
    arch_info = svc.get_cluster_architecture()

    # Academic & Execution Transparency Alert
    st.markdown(
        f"""
        <div style="background-color: #1A2332; border-left: 4px solid #2962FF; padding: 12px 18px; border-radius: 4px; font-size: 0.82rem; color: #90CAF9; margin-bottom: 20px;">
            <strong>ℹ️ EXECUTION & CLUSTER TRANSPARENCY:</strong> {arch_info['academic_statement']}<br>
            <em>All analytical datasets presented below are generated from validated Hadoop Streaming and PySpark jobs.
            The dashboard displays cached Parquet and text artifacts locally without requiring Docker containers to be running.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Row
    try:
        mr_volume = svc.get_mapreduce_volume(scope=scope)
        spark_stats = svc.get_spark_stock_statistics(scope=scope)
        spark_rankings = svc.get_spark_stock_rankings(scope=scope)
    except Exception as e:
        st.error(f"Error loading distributed analytics data: {e}")
        return

    mr_words = svc.get_mapreduce_word_counts(top_n=100, scope=scope)

    total_shares = mr_volume["total_volume"].sum() if not mr_volume.empty else 0
    kpis = [
        {"label": "MapReduce Volume Aggregated", "value": f"{total_shares:,.0f} Shares", "delta": f"{len(mr_volume)} Tickers" if not mr_volume.empty else "No Data"},
        {"label": "News Terms Extracted", "value": f"{len(mr_words):,} Terms", "delta": "MapReduce Streaming"},
        {"label": "Spark Processed Equities", "value": f"{len(spark_stats)} Equities" if not spark_stats.empty else "N/A", "delta": "Apache Spark 3.5.1"},
        {"label": "HDFS Replication Target", "value": f"Factor {arch_info['hdfs_replication']}", "delta": "2 DataNodes"},
    ]
    render_kpi_grid(kpis, columns=4)

    st.markdown("<br>", unsafe_allow_html=True)

    # TABBED INTERFACE FOR BIG DATA ENGINES
    tab_spark, tab_mr, tab_arch = st.tabs([
        "🔥 Apache Spark Analytics (Parquet)",
        "🐘 Hadoop MapReduce Results",
        "🏗️ Distributed Cluster Architecture",
    ])

    # --- TAB 1: SPARK ANALYTICS ---
    with tab_spark:
        st.markdown("#### 📊 PySpark Stock Statistics & Window Rankings")
        st.caption(f"Computed via PySpark DataFrame operations and window ranking functions over {len(spark_stats)} equities ({'121,962 rows' if is_real else '5,200 rows'}).")

        if not spark_stats.empty:
            col_rank, col_stat = st.columns([0.45, 0.55])

            with col_rank:
                st.markdown("##### 🏆 Spark Stock Volatility & Return Rankings")
                if not spark_rankings.empty:
                    st.dataframe(
                        spark_rankings,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Spark stock rankings Parquet partition not found.")

            with col_stat:
                st.markdown("##### 📈 Return vs. Volatility Comparison")
                vol_col = "avg_volatility" if "avg_volatility" in spark_stats.columns else ("volatility" if "volatility" in spark_stats.columns else None)
                if vol_col:
                    fig_scatter = px.scatter(
                        spark_stats,
                        x=vol_col,
                        y="avg_daily_return",
                        text="symbol",
                        size="avg_volume",
                        color="symbol",
                        title="Spark Aggregated Return vs. Volatility (Bubble Size = Avg Volume)",
                        template="plotly_dark",
                        labels={vol_col: "Average Volatility", "avg_daily_return": "Average Daily Return", "avg_volume": "Average Volume"},
                    )
                    fig_scatter.update_layout(
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#161922",
                        showlegend=False,
                        height=360,
                    )
                    fig_scatter.update_traces(textposition="top center")
                    st.plotly_chart(fig_scatter, use_container_width=True)

            # Full Spark Stats Table
            st.markdown("<br>", unsafe_allow_html=True)
            render_downloadable_table(
                df=spark_stats,
                title="PySpark Comprehensive Stock Statistics Table",
                filename="spark_stock_statistics.csv",
                height=280,
            )

        # Spark Daily Statistics Timeline
        spark_daily = svc.get_spark_daily_statistics(scope=scope)
        if not spark_daily.empty:
            st.markdown("##### 📅 Market-Wide Daily Statistics (PySpark Aggregates)")
            fig_daily = go.Figure()
            if "total_volume" in spark_daily.columns:
                fig_daily.add_trace(
                    go.Scatter(
                        x=spark_daily["date"],
                        y=spark_daily["total_volume"],
                        name="Market Total Volume",
                        line=dict(color="#2962FF", width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(41, 98, 255, 0.1)",
                    )
                )
            fig_daily.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#161922",
                title="<b>Daily Market-Wide Aggregate Volume</b>",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_daily, use_container_width=True)

    # --- TAB 2: HADOOP MAPREDUCE ---
    with tab_mr:
        st.markdown("#### 🐘 Hadoop Streaming MapReduce Aggregations")
        st.caption("Executed via Hadoop Streaming with standard POSIX streaming I/O (`mapper.py` / `reducer.py`).")

        col_mr1, col_mr2 = st.columns(2)

        with col_mr1:
            st.markdown("##### 📦 MapReduce Job 1: Ticker Volume Aggregation")
            if not mr_volume.empty:
                fig_vol = px.bar(
                    mr_volume,
                    x="symbol",
                    y="total_volume",
                    color="total_volume",
                    color_continuous_scale="Viridis",
                    title="Total Cumulative Volume by Equities",
                    template="plotly_dark",
                )
                fig_vol.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#161922", height=380)
                st.plotly_chart(fig_vol, use_container_width=True)
                st.dataframe(mr_volume, use_container_width=True, hide_index=True)
            else:
                st.info("MapReduce volume totals not found in data/processed/mapreduce/volume_totals.txt")

        with col_mr2:
            st.markdown("##### 📰 MapReduce Job 2: Financial News Term Frequencies")
            if not mr_words.empty:
                top_20_words = mr_words.head(20)
                fig_words = px.bar(
                    top_20_words,
                    x="count",
                    y="word",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Tealgrn",
                    title="Top 20 Financial News Terms",
                    template="plotly_dark",
                )
                fig_words.update_layout(
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161922",
                    yaxis=dict(autorange="reversed"),
                    height=380,
                )
                st.plotly_chart(fig_words, use_container_width=True)
                st.dataframe(mr_words.head(30), use_container_width=True, hide_index=True)
            else:
                st.info("MapReduce word counts not found in data/processed/mapreduce/word_counts.txt")

    # --- TAB 3: CLUSTER ARCHITECTURE ---
    with tab_arch:
        st.markdown("#### 🏗️ Docker Compose Multi-Node Cluster Configuration")
        st.markdown(
            """
            <div style="background-color: #1E222D; border: 1px solid #2A2E39; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                <h5 style="color: #2962FF; margin-top: 0;">Multi-Container Orchestration Model</h5>
                <p style="font-size: 0.85rem; color: #A0A5B5;">
                    The distributed deployment isolates master and compute worker daemons into 3 separate Linux containers
                    bridged by an internal virtual network on a single Windows host with Docker Desktop WSL2 backend.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nodes_data = arch_info["cluster_nodes"]
        for node in nodes_data:
            daemons_str = ", ".join(node["daemons"])
            st.markdown(
                f"""
                <div style="background-color: #161922; border-left: 3px solid #2962FF; padding: 12px 18px; margin-bottom: 12px; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">{node['node']}</span>
                        <span style="font-family: monospace; color: #2962FF; font-weight: 600;">{node['ip']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #00C805; font-weight: 600; margin-top: 2px;">{node['role']}</div>
                    <div style="font-size: 0.78rem; color: #A0A5B5; margin-top: 4px;">
                        <strong>Daemons:</strong> {daemons_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("##### 🛠️ Windows Cluster Management Commands")
        st.code(
            """# Start cluster (1 Master + 2 Workers + MongoDB):
.\\scripts\\cluster\\start_cluster.ps1 -Build

# Verify HDFS Live DataNodes & Replication:
.\\scripts\\cluster\\verify_hdfs.ps1

# Run distributed Hadoop Streaming MapReduce on YARN:
.\\scripts\\cluster\\run_mapreduce_test.ps1

# Run Spark standalone job reading and writing to HDFS:
.\\scripts\\cluster\\run_spark_test.ps1

# Stop cluster:
.\\scripts\\cluster\\stop_cluster.ps1""",
            language="powershell",
        )

    render_disclaimer()


if __name__ == "__main__":
    main()

