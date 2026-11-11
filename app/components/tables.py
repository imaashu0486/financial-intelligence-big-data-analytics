"""
Standardized table display and export utilities.
"""

from typing import Optional
import pandas as pd
import streamlit as st


def render_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    height: int = 350,
) -> None:
    """
    Render a styled DataFrame with row count and optional CSV download.
    """
    if title:
        st.markdown(f"##### {title}")

    if df.empty:
        st.info("No records found.")
        return

    st.dataframe(
        df,
        height=height,
        use_container_width=True,
    )


def render_downloadable_table(
    df: pd.DataFrame,
    title: str,
    filename: str = "export.csv",
    height: int = 350,
) -> None:
    """
    Render a table alongside a direct CSV export button.
    """
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(f"##### {title} ({len(df):,} records)")
    with col2:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    render_table(df, height=height)

