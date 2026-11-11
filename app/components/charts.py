"""
Interactive Plotly charting components with dark financial terminal styling.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Terminal Color Palette
DARK_BG = "#0E1117"
CARD_BG = "#161922"
GRID_COLOR = "#2A2E39"
TEXT_COLOR = "#E0E3EB"
MUTED_TEXT = "#787B86"
ACCENT_BLUE = "#2962FF"
BULL_GREEN = "#00C805"
BEAR_RED = "#FF3B30"
AMBER_WARN = "#FFB300"


def _apply_dark_theme(fig: go.Figure, height: int = 500) -> go.Figure:
    """Apply consistent terminal dark styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, family="Segoe UI, sans-serif"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(
            bgcolor="rgba(22, 25, 34, 0.8)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
        ),
    )
    return fig


def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Generate institutional OHLC candlestick chart with MA overlays and volume subplot.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC Price",
            increasing_line_color=BULL_GREEN,
            decreasing_line_color=BEAR_RED,
        ),
        row=1,
        col=1,
    )

    # Moving Average Overlays
    if "ma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["ma_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="#FFD600", width=1.5),
            ),
            row=1,
            col=1,
        )
    if "ma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["ma_50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="#29B6F6", width=1.5),
            ),
            row=1,
            col=1,
        )

    # 2. Volume Subplot
    colors = [
        BULL_GREEN if c >= o else BEAR_RED
        for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.8,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"<b>{symbol}</b> Price Action & Trading Volume",
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return _apply_dark_theme(fig, height=540)


def create_indicators_chart(df: pd.DataFrame) -> go.Figure:
    """
    Generate technical oscillator charts: RSI(14) and MACD with histogram.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Relative Strength Index (RSI 14)", "MACD (12, 26, 9)"),
    )

    # RSI Trace
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rsi_14"],
                mode="lines",
                name="RSI 14",
                line=dict(color="#AB47BC", width=1.5),
            ),
            row=1,
            col=1,
        )
        # Threshold lines (70 overbought, 30 oversold)
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color=BEAR_RED,
            annotation_text="Overbought (70)",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color=BULL_GREEN,
            annotation_text="Oversold (30)",
            row=1,
            col=1,
        )

    # MACD Trace
    if "macd" in df.columns and "macd_signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["macd"],
                mode="lines",
                name="MACD Line",
                line=dict(color="#2962FF", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["macd_signal"],
                mode="lines",
                name="Signal Line",
                line=dict(color="#FF9100", width=1.5),
            ),
            row=2,
            col=1,
        )
        # Histogram
        hist = df["macd"] - df["macd_signal"]
        hist_colors = [BULL_GREEN if val >= 0 else BEAR_RED for val in hist]
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=hist,
                name="MACD Histogram",
                marker_color=hist_colors,
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_dark_theme(fig, height=450)


def create_confusion_matrix_chart(cm: List[List[int]], labels: List[str]) -> go.Figure:
    """
    Generate an annotated confusion matrix heatmap.
    """
    z = np.array(cm)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=[[0, "#161922"], [0.5, "#2962FF"], [1.0, "#00C805"]],
            showscale=True,
            text=z,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
        )
    )
    fig.update_layout(
        title="<b>Classification Confusion Matrix</b>",
        xaxis_title="Predicted Direction",
        yaxis_title="True Direction",
    )
    return _apply_dark_theme(fig, height=420)


def create_cluster_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Generate K-Means stock risk-return profile scatter plot.
    """
    fig = go.Figure()

    cluster_colors = {0: "#2962FF", 1: "#FF3B30", 2: "#00C805", 3: "#FFB300"}

    for cluster_id, group in df.groupby("cluster"):
        color = cluster_colors.get(int(cluster_id), "#AB47BC")
        fig.add_trace(
            go.Scatter(
                x=group["avg_volatility"],
                y=group["avg_daily_return"],
                mode="markers+text",
                name=f"Cluster {cluster_id}",
                text=group["symbol"],
                textposition="top center",
                marker=dict(size=14, color=color, line=dict(width=1, color="#FFFFFF")),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Avg Volatility: %{x:.4f}<br>"
                    "Avg Return: %{y:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="<b>K-Means Stock Risk vs. Return Clustering</b>",
        xaxis_title="Average Volatility (Risk)",
        yaxis_title="Average Daily Return",
    )
    return _apply_dark_theme(fig, height=450)


def create_sentiment_time_chart(df: pd.DataFrame) -> go.Figure:
    """
    Generate daily financial news sentiment timeline.
    Robustly handles 'mean_sentiment', 'average_sentiment', or 'sentiment_score',
    aggregating by date when multiple ticker observations exist per day.
    """
    fig = go.Figure()

    if df is None or df.empty:
        fig.update_layout(
            title="<b>Daily Aggregated News Sentiment Score (No Data)</b>",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
        )
        return _apply_dark_theme(fig, height=400)

    work_df = df.copy()
    if "date" in work_df.columns:
        work_df["date"] = pd.to_datetime(work_df["date"])

    # Resolve sentiment column name
    sent_col = None
    for candidate in ["mean_sentiment", "average_sentiment", "sentiment_score"]:
        if candidate in work_df.columns:
            sent_col = candidate
            break

    if sent_col is None:
        # Fallback to any numeric column with sentiment in name
        num_cols = [c for c in work_df.columns if "sentiment" in c.lower() and pd.api.types.is_numeric_dtype(work_df[c])]
        if num_cols:
            sent_col = num_cols[0]

    if sent_col is not None and "date" in work_df.columns:
        # Aggregate to one average sentiment score per date if multiple symbols exist on same day
        daily = (
            work_df.dropna(subset=["date", sent_col])
            .groupby("date", as_index=False)[sent_col]
            .mean()
            .sort_values(by="date")
        )
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily[sent_col],
                mode="lines+markers",
                name="Daily Mean Sentiment",
                line=dict(color=ACCENT_BLUE, width=2),
                marker=dict(size=4),
            )
        )

    fig.add_hline(y=0.15, line_dash="dot", line_color=BULL_GREEN, annotation_text="Positive Regime (+0.15)")
    fig.add_hline(y=-0.15, line_dash="dot", line_color=BEAR_RED, annotation_text="Negative Regime (-0.15)")

    fig.update_layout(
        title="<b>Daily Aggregated News Sentiment Score</b>",
        xaxis_title="Date",
        yaxis_title="Mean Sentiment Score",
    )
    return _apply_dark_theme(fig, height=400)


def create_apriori_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Generate Support vs. Confidence scatter plot with Lift-scaled markers for association rules.
    """
    fig = go.Figure(
        data=go.Scatter(
            x=df["support"],
            y=df["confidence"],
            mode="markers",
            marker=dict(
                size=df["lift"] * 10,
                color=df["lift"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Lift"),
                line=dict(color="#FFFFFF", width=0.5),
            ),
            text=[f"{ant} ➔ {con}" for ant, con in zip(df["antecedent"], df["consequent"])],
            hovertemplate=(
                "<b>Rule:</b> %{text}<br>"
                "Support: %{x:.4f}<br>"
                "Confidence: %{y:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="<b>Mined Market Co-Movement Rules (Support vs. Confidence)</b>",
        xaxis_title="Rule Support",
        yaxis_title="Rule Confidence",
    )
    return _apply_dark_theme(fig, height=450)

