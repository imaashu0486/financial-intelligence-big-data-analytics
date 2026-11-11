"""
Layout, custom CSS styling, and global headers for the Financial Terminal.
"""

from datetime import datetime
from typing import Optional
import streamlit as st


TERMINAL_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0E1117;
        color: #E0E3EB;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        padding: 16px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #787B86 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #F0F3FA !important;
        font-family: "Roboto Mono", monospace;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Header Container */
    .terminal-header {
        background: linear-gradient(90deg, #161922 0%, #1E222D 100%);
        border-left: 4px solid #2962FF;
        border-radius: 4px;
        padding: 14px 20px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .terminal-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.5px;
    }
    .terminal-subtitle {
        font-size: 0.8rem;
        color: #787B86;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Card Panels */
    .terminal-card {
        background-color: #1E222D;
        border: 1px solid #2A2E39;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Disclaimer Banner */
    .disclaimer-banner {
        background-color: #211915;
        border-left: 3px solid #FF9800;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.78rem;
        color: #FFA726;
        margin-top: 24px;
        margin-bottom: 12px;
    }

    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #2A2E39;
        border-radius: 6px;
    }
</style>
"""


def setup_page(title: str, icon: str = "📈", layout: str = "wide") -> None:
    """
    Configure Streamlit page layout and inject terminal CSS styling.
    """
    st.set_page_config(
        page_title=f"{title} | Financial Intelligence",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


def render_terminal_header(title: str, subtitle: str, badge: Optional[str] = None) -> None:
    """
    Render a branded financial terminal header bar.
    """
    badge_html = f'<span style="background-color: #2962FF; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{badge}</span>' if badge else ""
    header_html = f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">{title} {badge_html}</div>
            <div class="terminal-subtitle">{subtitle}</div>
        </div>
        <div style="text-align: right; font-family: monospace; font-size: 0.8rem; color: #787B86;">
            <div>SYS // NITD-CSBB422</div>
            <div>STATUS // OPERATIONAL</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_disclaimer(custom_text: Optional[str] = None) -> None:
    """
    Render standardized academic & financial compliance disclaimer banner.
    """
    text = custom_text or (
        "ACADEMIC & RESEARCH DISCLAIMER: This platform is developed for CSBB 422 Big Data Analytics Lab at NIT Delhi. "
        "All analytics, ML forecasts, sentiment regimes, and RAG answers are generated for educational demonstration and "
        "do not constitute financial, investment, or legal advice."
    )
    disclaimer_html = f"""
    <div class="disclaimer-banner">
        <strong>⚠️ COMPLIANCE NOTICE:</strong> {text}
    </div>
    """
    st.markdown(disclaimer_html, unsafe_allow_html=True)

