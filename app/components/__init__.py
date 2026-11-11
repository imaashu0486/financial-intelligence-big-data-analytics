"""
Reusable UI components for the Streamlit dashboard.
"""

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_card, render_kpi_grid
from app.components.charts import (
    create_candlestick_chart,
    create_indicators_chart,
    create_confusion_matrix_chart,
    create_cluster_scatter,
    create_sentiment_time_chart,
    create_apriori_scatter,
)

__all__ = [
    "setup_page",
    "render_terminal_header",
    "render_disclaimer",
    "render_sidebar",
    "render_kpi_card",
    "render_kpi_grid",
    "create_candlestick_chart",
    "create_indicators_chart",
    "create_confusion_matrix_chart",
    "create_cluster_scatter",
    "create_sentiment_time_chart",
    "create_apriori_scatter",
]

