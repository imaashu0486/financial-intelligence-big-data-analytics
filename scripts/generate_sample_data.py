#!/usr/bin/env python3
"""
Generate realistic synthetic sample data for the Financial Intelligence Platform.

Creates:
  - data/sample/market_data.csv   (~5,000 rows — 10 stocks × ~2 years)
  - data/sample/financial_news.csv (~500 rows)

Market prices are generated using Geometric Brownian Motion so they look
plausible. News headlines use curated financial phrase templates.

⚠ All data is SYNTHETIC — for demonstration only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Resolve project root and load config ─────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

from src.utils.logger import logger  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Market data generation (Geometric Brownian Motion)
# ─────────────────────────────────────────────────────────────

# Realistic starting prices for Indian stocks (₹)
_BASE_PRICES = {
    "RELIANCE": 2400.0,
    "TCS": 3500.0,
    "INFY": 1500.0,
    "HDFCBANK": 1600.0,
    "ICICIBANK": 900.0,
    "WIPRO": 450.0,
    "SBIN": 550.0,
    "BAJFINANCE": 7000.0,
    "BHARTIARTL": 800.0,
    "LT": 2100.0,
}

# Per-stock annualised drift (μ) and volatility (σ)
_STOCK_PARAMS = {
    "RELIANCE":   {"mu": 0.08, "sigma": 0.25},
    "TCS":        {"mu": 0.12, "sigma": 0.20},
    "INFY":       {"mu": 0.10, "sigma": 0.22},
    "HDFCBANK":   {"mu": 0.06, "sigma": 0.18},
    "ICICIBANK":  {"mu": 0.09, "sigma": 0.22},
    "WIPRO":      {"mu": 0.05, "sigma": 0.28},
    "SBIN":       {"mu": 0.07, "sigma": 0.24},
    "BAJFINANCE": {"mu": 0.14, "sigma": 0.32},
    "BHARTIARTL": {"mu": 0.11, "sigma": 0.20},
    "LT":         {"mu": 0.09, "sigma": 0.21},
}


def _generate_ohlcv(
    symbol: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate OHLCV data for a single stock using GBM."""
    n = len(dates)
    s0 = _BASE_PRICES.get(symbol, 1000.0)
    params = _STOCK_PARAMS.get(symbol, {"mu": 0.08, "sigma": 0.25})
    mu, sigma = params["mu"], params["sigma"]

    dt = 1 / 252  # trading-day fraction of a year
    # GBM: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    log_returns[0] = 0.0  # first day starts at s0
    close = s0 * np.exp(np.cumsum(log_returns))

    # Derive OHLV from close
    intraday_noise = rng.uniform(0.005, 0.025, size=n)
    high = close * (1 + intraday_noise)
    low = close * (1 - intraday_noise)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=n)

    # Volume: base volume + noise, with occasional spikes
    base_volume = rng.integers(500_000, 5_000_000, size=n)
    spike_mask = rng.random(n) < 0.05  # 5 % chance of volume spike
    base_volume[spike_mask] = base_volume[spike_mask] * rng.integers(3, 8, size=spike_mask.sum())

    return pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": base_volume,
    })


def generate_market_data(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate OHLCV for all configured symbols."""
    sd = cfg["sample_data"]
    symbols = cfg["symbols"]

    # Build trading-day calendar (Mon-Fri, exclude weekends)
    all_dates = pd.bdate_range(start=sd["start_date"], end=sd["end_date"])
    logger.info(f"Generating market data: {len(symbols)} symbols × {len(all_dates)} trading days")

    frames = [_generate_ohlcv(sym, all_dates, rng) for sym in symbols]
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["date", "symbol"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# News data generation
# ─────────────────────────────────────────────────────────────

_BULLISH_TEMPLATES = [
    "{symbol} shares surge {pct}% on strong quarterly results",
    "{symbol} hits 52-week high amid sector rally",
    "{symbol} announces ₹{amount} crore buyback plan",
    "Analysts upgrade {symbol} target price to ₹{price}",
    "{symbol} wins major contract worth ₹{amount} crore",
    "Foreign investors increase stake in {symbol}",
    "{symbol} reports record revenue growth of {pct}%",
    "Bullish outlook for {symbol} as demand rebounds",
    "{symbol} board approves dividend of ₹{div} per share",
    "{symbol} expands operations into new markets",
]

_BEARISH_TEMPLATES = [
    "{symbol} drops {pct}% after disappointing earnings",
    "{symbol} faces regulatory scrutiny over compliance issues",
    "Analysts downgrade {symbol} citing margin pressure",
    "{symbol} reports {pct}% decline in quarterly profit",
    "Global headwinds weigh on {symbol} share price",
    "{symbol} cuts capex guidance amid uncertain outlook",
    "Promoter sells {pct}% stake in {symbol}",
    "{symbol} warns of supply chain disruptions",
    "{symbol} under pressure as sector sentiment weakens",
    "Rating agency downgrades {symbol} outlook to negative",
]

_NEUTRAL_TEMPLATES = [
    "{symbol} to announce Q{q} results on {month} {day}",
    "Market watch: {symbol} trades flat in volatile session",
    "{symbol} appoints new CFO from {source}",
    "Sector rotation: what {symbol}'s movement signals",
    "{symbol} explores strategic partnerships in fintech",
    "Institutional holding in {symbol} remains steady at {pct}%",
    "{symbol} AGM scheduled for next month",
    "Technical analysis: {symbol} approaching key support level",
]

_SOURCES = [
    "Economic Times", "Moneycontrol", "LiveMint",
    "Business Standard", "CNBC-TV18", "BloombergQuint",
    "Financial Express", "Reuters India", "NDTV Profit",
]

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _fill_template(template: str, symbol: str, rng: np.random.Generator) -> str:
    """Replace placeholders with realistic random values."""
    return template.format(
        symbol=symbol,
        pct=rng.integers(2, 18),
        amount=rng.integers(500, 10000),
        price=rng.integers(500, 10000),
        div=rng.integers(5, 50),
        q=rng.integers(1, 5),
        month=rng.choice(_MONTHS),
        day=rng.integers(1, 29),
        source=rng.choice(["internal promotion", "rival firm", "global operations"]),
    )


def generate_news_data(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic financial news headlines."""
    sd = cfg["sample_data"]
    symbols = cfg["symbols"]
    n_news = sd["news_count"]

    dates = pd.bdate_range(start=sd["start_date"], end=sd["end_date"])
    all_templates = _BULLISH_TEMPLATES + _BEARISH_TEMPLATES + _NEUTRAL_TEMPLATES

    rows = []
    for _ in range(n_news):
        sym = rng.choice(symbols)
        date = rng.choice(dates)
        template = rng.choice(all_templates)
        title = _fill_template(template, sym, rng)

        # Description is an expanded version of the title
        description = (
            f"{title}. Market participants are closely watching developments "
            f"around {sym} as trading volumes {'remain elevated' if rng.random() > 0.5 else 'show mixed signals'}. "
            f"Sector peers are {'also impacted' if rng.random() > 0.5 else 'relatively unaffected'} by the news."
        )

        rows.append({
            "date": date,
            "title": title,
            "description": description,
            "source": rng.choice(_SOURCES),
            "symbol": sym,
        })

    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sample data")
    parser.add_argument(
        "--seed", type=int, default=CFG["sample_data"]["seed"],
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Market data
    market_df = generate_market_data(CFG, rng)
    market_path = out_dir / "market_data.csv"
    market_df.to_csv(market_path, index=False)
    logger.info(f"✅ Market data saved: {market_path}  ({len(market_df):,} rows)")

    # News data
    news_df = generate_news_data(CFG, rng)
    news_path = out_dir / "financial_news.csv"
    news_df.to_csv(news_path, index=False)
    logger.info(f"✅ News data saved:   {news_path}  ({len(news_df):,} rows)")

    logger.info("Sample data generation complete ✨")


if __name__ == "__main__":
    main()
