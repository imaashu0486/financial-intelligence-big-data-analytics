"""
Metric cards and KPI grid layout components for executive financial dashboards.
"""

from typing import Any, Dict, List, Optional
import streamlit as st


def render_kpi_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
) -> None:
    """
    Render a single standardized KPI metric card.
    """
    st.metric(
        label=label,
        value=str(value),
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def render_kpi_grid(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    """
    Render a responsive grid of metric cards.

    Parameters
    ----------
    metrics : list of dicts with keys: 'label', 'value', optional 'delta', 'help'
    columns : int
    """
    cols = st.columns(columns)
    for i, m in enumerate(metrics):
        with cols[i % columns]:
            render_kpi_card(
                label=m.get("label", ""),
                value=m.get("value", "-"),
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
                help_text=m.get("help"),
            )

