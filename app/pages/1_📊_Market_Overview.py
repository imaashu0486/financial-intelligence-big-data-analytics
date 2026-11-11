"""
Page 1: Market Overview.
Visualizes financial market data, price trends, volume analysis, technical indicators, and historical records.
"""

import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.layout import setup_page, render_terminal_header, render_disclaimer
from app.components.sidebar import render_sidebar
from app.components.metrics import render_kpi_grid
from app.components.charts import create_candlestick_chart, create_indicators_chart
from app.components.tables import render_downloadable_table
from app.services.data_service import DataService


def main() -> None:
    setup_page("Market Overview", icon="📊")
    render_sidebar(active_page="Market Overview")

    scope = st.session_state.get("dataset_scope", "Real Market History (NIFTY 50)")
    is_real = "real" in scope.lower()

    render_terminal_header(
        title="FINANCIAL MARKET OVERVIEW",
        subtitle="PRICE TRENDS, VOLUME ANALYSIS & TECHNICAL INDICATORS (PHASE 1)",
        badge="REAL NIFTY 50" if is_real else "SYNTHETIC BASELINE",
    )

    data_svc = DataService()
    try:
        df_all = data_svc.load_market_features()
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return

    if df_all.empty:
        st.error("Market data file not found for the selected dataset context.")
        return

    # Top KPI Bar
    kpis = data_svc.get_market_kpis(df_all)
    metric_cards = [
        {"label": "Total Market Records", "value": f"{kpis['total_records']:,}", "delta": "Verified Data"},
        {"label": "Tracked Equities", "value": f"{kpis['symbols_count']} Companies", "delta": "NIFTY 50" if is_real else "Synthetic Universe"},
        {"label": "Date Range", "value": f"{kpis['min_date']} → {kpis['max_date']}", "help": "Chronological bounds"},
        {"label": "Equity Trading Dates", "value": f"{kpis['trading_days']:,} Sessions", "delta": "Distinct Equity Dates" if is_real else "Complete Sample", "help": "Total distinct calendar dates with active equity trading (2,467 distinct equity dates across the 50 stocks; 2,459 aligned benchmark sessions)."},
        {"label": "Avg Daily Volume", "value": f"{kpis['avg_daily_volume']:,.0f}", "help": "Shares per ticker/day"},
    ]
    render_kpi_grid(metric_cards, columns=5)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filtering Controls
    col_sym, col_dates = st.columns([0.4, 0.6])
    symbols = kpis["symbols"]

    with col_sym:
        selected_symbol = st.selectbox(
            "Select Stock Ticker:",
            options=symbols,
            index=0 if "RELIANCE" not in symbols else symbols.index("RELIANCE"),
        )

    with col_dates:
        min_d = df_all["date"].min().date()
        max_d = df_all["date"].max().date()
        date_range = st.date_input(
            "Filter Date Window:",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )

    # Filter data for selected ticker and date bounds
    start_d = str(date_range[0]) if isinstance(date_range, (tuple, list)) and len(date_range) > 0 else str(min_d)
    end_d = str(date_range[1]) if isinstance(date_range, (tuple, list)) and len(date_range) > 1 else str(max_d)

    df_ticker = data_svc.get_ticker_data(
        symbol=selected_symbol,
        df=df_all,
        start_date=start_d,
        end_date=end_d,
    )

    if df_ticker.empty:
        st.warning(f"No records found for {selected_symbol} in the selected date range.")
        return

    # Ticker specific summary cards
    latest_row = df_ticker.iloc[-1]
    company_name = latest_row.get("company_name", selected_symbol)
    sector = latest_row.get("sector", "Equities")
    latest_close = latest_row.get("close", 0.0)
    latest_return = latest_row.get("daily_return", 0.0) * 100
    avg_vol = df_ticker["volume"].mean() if "volume" in df_ticker.columns else 0.0
    volatility_20 = latest_row.get("volatility", 0.0) * 100

    st.markdown(
        f"""
        <div class="terminal-card" style="padding: 14px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF;">{company_name} ({selected_symbol})</span>
                <span style="background-color: #2A2E39; color: #A0A5B5; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 10px;">{sector}</span>
            </div>
            <div style="font-family: monospace; font-size: 1.1rem; font-weight: 700;">
                Latest Close: ₹{latest_close:,.2f}
                <span style="color: {'#00C805' if latest_return >= 0 else '#FF3B30'}; margin-left: 10px;">
                    {'+' if latest_return >= 0 else ''}{latest_return:.2f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Main Candlestick & Volume Chart
    fig_candle = create_candlestick_chart(df_ticker, selected_symbol)
    st.plotly_chart(fig_candle, use_container_width=True)

    # 2. Technical Indicators (RSI & MACD)
    st.markdown("#### 📐 Technical Indicators & Oscillators")
    fig_indicators = create_indicators_chart(df_ticker)
    st.plotly_chart(fig_indicators, use_container_width=True)

    # 3. Latest Market Snapshot
    st.markdown(f"#### 📋 Latest Cross-Market Snapshot ({kpis['symbols_count']} Equities)")
    snapshot = data_svc.get_latest_snapshot(df_all)
    if not snapshot.empty:
        display_cols = [c for c in ["symbol", "company_name", "sector", "date", "open", "high", "low", "close", "volume", "daily_return", "volatility", "rsi_14"] if c in snapshot.columns]
        formatted_snapshot = snapshot[display_cols].copy()
        if "date" in formatted_snapshot.columns:
            formatted_snapshot["date"] = formatted_snapshot["date"].dt.strftime("%Y-%m-%d")
        if "daily_return" in formatted_snapshot.columns:
            formatted_snapshot["daily_return"] = formatted_snapshot["daily_return"].apply(lambda x: f"{x*100:+.2f}%")
        if "volatility" in formatted_snapshot.columns:
            formatted_snapshot["volatility"] = formatted_snapshot["volatility"].apply(lambda x: f"{x*100:.2f}%")
        if "close" in formatted_snapshot.columns:
            formatted_snapshot["close"] = formatted_snapshot["close"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(formatted_snapshot, use_container_width=True, hide_index=True)

    # 4. Historical Data Table for Selected Ticker
    st.markdown("<br>", unsafe_allow_html=True)
    table_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "daily_return", "ma_20", "ma_50", "rsi_14", "macd", "volatility"] if c in df_ticker.columns]
    df_export = df_ticker[table_cols].copy()
    if "date" in df_export.columns:
        df_export["date"] = df_export["date"].dt.strftime("%Y-%m-%d")

    render_downloadable_table(
        df=df_export,
        title=f"Historical Price & Indicator Records: {selected_symbol}",
        filename=f"{selected_symbol}_historical_data.csv",
        height=320,
    )

    render_disclaimer()


if __name__ == "__main__":
    main()

