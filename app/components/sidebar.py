"""
Standardized sidebar navigation, system health status, and metadata monitor.
"""

from pathlib import Path
from typing import Optional
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent

from src.database.mongodb import check_connection


def render_sidebar(active_page: str = "Home") -> None:
    """
    Render global financial terminal sidebar with system health monitors.
    """
    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #2A2E39; margin-bottom: 20px;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.5px;">
                    FINANCIAL INTELLIGENCE
                </div>
                <div style="font-size: 0.75rem; color: #2962FF; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;">
                    Big Data Analytics Platform
                </div>
                <div style="font-size: 0.7rem; color: #787B86; margin-top: 4px;">
                    NIT Delhi // CSBB 422 Lab Project
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dataset Scope Selector
        st.markdown(
            '<div style="font-size: 0.75rem; font-weight: 700; color: #787B86; letter-spacing: 1px; margin-bottom: 6px; text-transform: uppercase;">'
            "DATASET CONTEXT SCOPE"
            "</div>",
            unsafe_allow_html=True,
        )
        if "dataset_scope" not in st.session_state:
            st.session_state["dataset_scope"] = "Real Market History (NIFTY 50)"

        dataset_options = [
            "Real Market History (NIFTY 50)",
            "Synthetic Baseline (10 Equities)",
        ]
        selected_scope = st.radio(
            "Select Active Context:",
            dataset_options,
            index=dataset_options.index(st.session_state["dataset_scope"]),
            key="dataset_scope_radio",
            label_visibility="collapsed",
        )
        st.session_state["dataset_scope"] = selected_scope

        scope_desc = (
            '<div style="background-color: #0d1e3a; border-left: 3px solid #2962FF; padding: 6px 10px; border-radius: 3px; font-size: 0.72rem; color: #82B1FF; margin-bottom: 16px;">'
            '<strong>ACTIVE: REAL NIFTY 50</strong><br>50 Stocks // 10 Years (2015–2024)<br>121,962 records // Cluster Parquet'
            '</div>'
            if "Real" in selected_scope
            else
            '<div style="background-color: #2b1f0d; border-left: 3px solid #FF9800; padding: 6px 10px; border-radius: 3px; font-size: 0.72rem; color: #FFE082; margin-bottom: 16px;">'
            '<strong>ACTIVE: SYNTHETIC BASELINE</strong><br>10 Stocks // 2 Years (2022–2023)<br>5,200 records // Controlled Testing'
            '</div>'
        )
        st.markdown(scope_desc, unsafe_allow_html=True)

        # System Health Indicators
        st.markdown(
            '<div style="font-size: 0.75rem; font-weight: 700; color: #787B86; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase;">'
            "SYSTEM HEALTH MONITOR"
            "</div>",
            unsafe_allow_html=True,
        )

        # MongoDB Ping (fast check with timeout)
        mongo_online = False
        try:
            health = check_connection(timeout_ms=400)
            mongo_online = bool(isinstance(health, dict) and health.get("connected", False))
        except Exception:
            mongo_online = False

        mongo_badge = (
            '<span style="color: #00C805;">● ONLINE</span>'
            if mongo_online
            else '<span style="color: #FFB300;">○ OFFLINE (FALLBACK)</span>'
        )

        # Artifact Checks
        parquet_exists = (ROOT / "data" / "real" / "analytics" / "risk_return_metrics.csv").exists()
        parquet_badge = '<span style="color: #00C805;">● REAL HDFS (4 TBLS)</span>' if parquet_exists else '<span style="color: #FF3B30;">○ MISSING</span>'

        vectorstore_exists = (ROOT / "data" / "vectorstore" / "embeddings.npy").exists()
        vector_badge = '<span style="color: #00C805;">● 28 CHUNKS</span>' if vectorstore_exists else '<span style="color: #FFB300;">○ UNINDEXED</span>'

        real_ml_exist = (ROOT / "models" / "real" / "real_classifier_random_forest.joblib").exists()
        ml_badge = '<span style="color: #00C805;">● REAL ML (6 MDLS)</span>' if real_ml_exist else '<span style="color: #FF3B30;">○ NOT FOUND</span>'

        st.markdown(
            f"""
            <div style="background-color: #161922; border: 1px solid #2A2E39; border-radius: 6px; padding: 12px; font-size: 0.8rem; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #A0A5B5;">MongoDB NoSQL:</span>
                    <span style="font-family: monospace; font-weight: 600;">{mongo_badge}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #A0A5B5;">Analytics Store:</span>
                    <span style="font-family: monospace; font-weight: 600;">{parquet_badge}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #A0A5B5;">Vector Store (RAG):</span>
                    <span style="font-family: monospace; font-weight: 600;">{vector_badge}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #A0A5B5;">Real ML Engine:</span>
                    <span style="font-family: monospace; font-weight: 600;">{ml_badge}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Course Information
        st.markdown(
            """
            <div style="border-top: 1px solid #2A2E39; padding-top: 16px; font-size: 0.75rem; color: #787B86;">
                <div><strong>Course:</strong> CSBB 422 Big Data Analytics</div>
                <div><strong>Institution:</strong> NIT Delhi</div>
                <div><strong>Architecture:</strong> Master + 2 Workers</div>
                <div><strong>Engine:</strong> Hadoop 3.3.6 / Spark 3.5.1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clear App Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

