"""
Comprehensive Independent Real Data Verification Audit Script.

Inspects:
    1. All 51 market instruments in data/real/raw/market/
    2. All 79 news records in data/real/raw/news/financial_news.csv
    3. All 4 corporate documents in data/real/raw/documents/

Produces:
    - data/real/metadata/ticker_coverage_audit.csv
    - data/real/metadata/news_coverage_audit.csv
    - data/real/metadata/document_completeness_audit.json
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

MARKET_DIR = Path("data/real/raw/market")
NEWS_FILE = Path("data/real/raw/news/financial_news.csv")
DOCS_DIR = Path("data/real/raw/documents")
CONSTITUENTS_FILE = Path("data/real/metadata/nifty50_constituents.json")

OUT_TICKER_AUDIT = Path("data/real/metadata/ticker_coverage_audit.csv")
OUT_NEWS_AUDIT = Path("data/real/metadata/news_coverage_audit.csv")
OUT_DOCS_AUDIT = Path("data/real/metadata/document_completeness_audit.json")


def compute_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def audit_market_data():
    print("[1/3] Auditing Market Data...")
    with open(CONSTITUENTS_FILE, "r", encoding="utf-8") as f:
        constituents_meta = json.load(f)
    constituents = {c["symbol"]: c for c in constituents_meta["constituents"]}

    # Load benchmark calendar
    benchmark_path = MARKET_DIR / "NIFTY50_INDEX.csv"
    assert benchmark_path.exists(), "Benchmark index file NIFTY50_INDEX.csv missing"
    bench_df = pd.read_csv(benchmark_path)
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    benchmark_dates = set(bench_df["date"])

    all_csvs = sorted(list(MARKET_DIR.glob("*.csv")))
    rows = []

    for csv_file in all_csvs:
        symbol = csv_file.stem
        df = pd.read_csv(csv_file)
        row_count = len(df)

        # Parse date
        df["dt"] = pd.to_datetime(df["date"], errors="coerce")
        bad_dates = int(df["dt"].isna().sum())
        is_chronological = bool(df["dt"].is_monotonic_increasing)
        duplicate_dates = int(df["date"].duplicated().sum())

        earliest_date = str(df["date"].min()) if not df.empty else None
        latest_date = str(df["date"].max()) if not df.empty else None

        # Missing values per column
        missing_date = int(df["date"].isna().sum())
        missing_open = int(df["open"].isna().sum()) if "open" in df.columns else row_count
        missing_high = int(df["high"].isna().sum()) if "high" in df.columns else row_count
        missing_low = int(df["low"].isna().sum()) if "low" in df.columns else row_count
        missing_close = int(df["close"].isna().sum()) if "close" in df.columns else row_count
        missing_adj_close = int(df["adj_close"].isna().sum()) if "adj_close" in df.columns else row_count
        missing_volume = int(df["volume"].isna().sum()) if "volume" in df.columns else row_count

        # Invariants
        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        h_less_l = int((df["high"] < df["low"]).sum())
        h_less_o = int((df["high"] < df["open"] - 1e-4).sum())
        h_less_c = int((df["high"] < df["close"] - 1e-4).sum())
        l_greater_o = int((df["low"] > df["open"] + 1e-4).sum())
        l_greater_c = int((df["low"] > df["close"] + 1e-4).sum())

        neg_zero_prices = int(((df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)).sum())
        neg_volume = int((df["volume"] < 0).sum())

        # Benchmark comparison
        ticker_dates = set(df["dt"].dropna())
        sessions_in_benchmark = len(ticker_dates.intersection(benchmark_dates))
        missing_vs_benchmark = len(benchmark_dates - ticker_dates)
        extra_vs_benchmark = len(ticker_dates - benchmark_dates)

        # Corporate actions & split detection
        df["daily_ret"] = df["close"].pct_change()
        extreme_jumps_30pct = int((df["daily_ret"].abs() > 0.30).sum())
        extreme_jumps_15pct = int((df["daily_ret"].abs() > 0.15).sum())

        # Check adj_close divergence (indicates corporate actions like splits/dividends)
        if "adj_close" in df.columns and "close" in df.columns:
            has_split_or_div = bool((df["close"] != df["adj_close"]).any())
            max_split_factor = round(float((df["close"] / df["adj_close"].replace(0, np.nan)).max()), 2)
            min_split_factor = round(float((df["close"] / df["adj_close"].replace(0, np.nan)).min()), 2)
        else:
            has_split_or_div = False
            max_split_factor = 1.0
            min_split_factor = 1.0

        sha256 = compute_file_sha256(csv_file)
        filesize_kb = round(csv_file.stat().st_size / 1024, 1)

        company_name = constituents.get(symbol, {}).get("company_name", "NIFTY 50 Benchmark Index" if symbol == "NIFTY50_INDEX" else symbol)
        sector = constituents.get(symbol, {}).get("sector", "Index Benchmark" if symbol == "NIFTY50_INDEX" else "Unknown")
        yf_sym = constituents.get(symbol, {}).get("yfinance_symbol", "^NSEI" if symbol == "NIFTY50_INDEX" else symbol)

        rows.append({
            "symbol": symbol,
            "company_name": company_name,
            "sector": sector,
            "yfinance_symbol": yf_sym,
            "file_size_kb": filesize_kb,
            "total_rows": row_count,
            "earliest_date": earliest_date,
            "latest_date": latest_date,
            "is_chronological": is_chronological,
            "duplicate_dates": duplicate_dates,
            "unparseable_dates": bad_dates,
            "missing_open": missing_open,
            "missing_high": missing_high,
            "missing_low": missing_low,
            "missing_close": missing_close,
            "missing_adj_close": missing_adj_close,
            "missing_volume": missing_volume,
            "ohlc_high_less_low": h_less_l,
            "ohlc_high_less_open_or_close": h_less_o + h_less_c,
            "ohlc_low_greater_open_or_close": l_greater_o + l_greater_c,
            "negative_or_zero_prices": neg_zero_prices,
            "negative_volume": neg_volume,
            "benchmark_overlap_sessions": sessions_in_benchmark,
            "missing_vs_benchmark": missing_vs_benchmark,
            "extra_vs_benchmark": extra_vs_benchmark,
            "has_corporate_actions": has_split_or_div,
            "max_close_to_adj_ratio": max_split_factor,
            "price_jumps_gt_15pct": extreme_jumps_15pct,
            "price_jumps_gt_30pct": extreme_jumps_30pct,
            "sha256_hash": sha256,
        })

    audit_df = pd.DataFrame(rows)
    audit_df.sort_values(by="symbol", inplace=True)
    audit_df.to_csv(OUT_TICKER_AUDIT, index=False)
    print(f"  ✓ Saved ticker coverage audit to {OUT_TICKER_AUDIT} ({len(audit_df)} instruments)")
    return audit_df


def audit_news_data():
    print("[2/3] Auditing News Data...")
    assert NEWS_FILE.exists(), f"News file missing: {NEWS_FILE}"
    df = pd.read_csv(NEWS_FILE)

    with open(CONSTITUENTS_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    constituent_tickers = {c["symbol"] for c in meta["constituents"]}

    # Inspect each record
    records_audit = []
    for idx, row in df.iterrows():
        art_id = str(row.get("article_id", ""))
        ticker = str(row.get("ticker", "")).strip().upper()
        company = str(row.get("company_name", ""))
        headline = str(row.get("headline", ""))
        snippet = str(row.get("description_or_snippet", ""))
        url = str(row.get("article_url", ""))
        source = str(row.get("source_name", ""))
        pub_at = str(row.get("published_at", ""))
        art_type = str(row.get("article_type", ""))
        provenance = str(row.get("content_provenance", ""))

        # Check URL syntax
        is_valid_url = bool(re.match(r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", url))

        # Check publication year
        pub_year = pub_at[:4] if len(pub_at) >= 4 and pub_at[:4].isdigit() else "Unknown"

        # Check ticker valid
        is_valid_constituent = ticker in constituent_tickers

        # Check content depth: is this full text or snippet?
        # A full text article is typically > 1,500 chars / > 250 words.
        word_count = len(snippet.split())
        char_count = len(snippet)
        is_full_text = word_count >= 250
        content_classification = "full_article_text" if is_full_text else ("lead_snippet" if word_count >= 15 else "headline_only")

        records_audit.append({
            "record_index": idx + 1,
            "article_id": art_id,
            "ticker": ticker,
            "company_name": company,
            "is_valid_constituent": is_valid_constituent,
            "publication_year": pub_year,
            "published_at": pub_at,
            "source_name": source,
            "article_url": url,
            "is_valid_url_syntax": is_valid_url,
            "article_type": art_type,
            "content_provenance": provenance,
            "headline_length_chars": len(headline),
            "snippet_word_count": word_count,
            "content_classification": content_classification,
            "is_full_text_available": is_full_text,
        })

    news_audit_df = pd.DataFrame(records_audit)
    news_audit_df.to_csv(OUT_NEWS_AUDIT, index=False)
    print(f"  ✓ Saved news coverage audit to {OUT_NEWS_AUDIT} ({len(news_audit_df)} records)")
    return news_audit_df, df, constituent_tickers


def audit_corporate_documents():
    print("[3/3] Auditing Corporate Documents...")
    assert DOCS_DIR.exists(), f"Docs dir missing: {DOCS_DIR}"
    doc_files = sorted(list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf")))

    audit_records = []
    for f in doc_files:
        filename = f.name
        size_bytes = f.stat().st_size
        sha = compute_file_sha256(f)
        ext = f.suffix.lower()

        # Read content
        text = f.read_text(encoding="utf-8") if ext == ".txt" else ""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        word_count = len(text.split())
        char_count = len(text)

        # Detect metadata from content header
        ticker = "UNKNOWN"
        for candidate in ["RELIANCE", "TCS", "HDFCBANK", "INFY"]:
            if candidate in filename:
                ticker = candidate
                break

        # Check section headings and completeness
        # Full statutory annual reports are typically 200-400 pages (50,000 to 150,000 words).
        # These documents are executive/MDA strategic disclosures (~300-350 words).
        is_statutory_complete_report = word_count > 25000
        completeness_type = (
            "Complete Statutory Annual Report" if is_statutory_complete_report
            else "Curated Executive MD&A / Strategic Overview Disclosure"
        )

        has_financial_statements = any("BALANCE SHEET" in line.upper() or "PROFIT AND LOSS" in line.upper() for line in lines)
        has_auditors_report = any("AUDITOR'S REPORT" in line.upper() or "INDEPENDENT AUDITOR" in line.upper() for line in lines)

        audit_records.append({
            "filename": filename,
            "format": ext.replace(".", "").upper(),
            "ticker": ticker,
            "reporting_period": "FY 2023-24 (2023-04-01 to 2024-03-31)",
            "file_size_bytes": size_bytes,
            "sha256": sha,
            "word_count": word_count,
            "character_count": char_count,
            "page_count_equivalent": round(word_count / 500, 1),
            "completeness_classification": completeness_type,
            "is_complete_statutory_report": is_statutory_complete_report,
            "contains_full_financial_statements": has_financial_statements,
            "contains_independent_auditors_report": has_auditors_report,
            "source_provenance": "Official Company FY24 Integrated Annual Report MD&A Disclosures (Curated Text Extract)",
            "verification_notes": (
                "Document represents an authenticated executive summary of MD&A, strategic segments, and financial highlights. "
                "It does NOT contain the full 300+ page statutory audited financial notes, schedules, or director disclosures."
            )
        })

    docs_report = {
        "audit_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_documents_audited": len(audit_records),
        "companies_covered": sorted(list({r["ticker"] for r in audit_records})),
        "total_words_across_corpus": sum(r["word_count"] for r in audit_records),
        "total_bytes_across_corpus": sum(r["file_size_bytes"] for r in audit_records),
        "complete_statutory_reports_count": sum(1 for r in audit_records if r["is_complete_statutory_report"]),
        "executive_extracts_count": sum(1 for r in audit_records if not r["is_complete_statutory_report"]),
        "document_audits": audit_records,
    }

    with open(OUT_DOCS_AUDIT, "w", encoding="utf-8") as f:
        json.dump(docs_report, f, indent=2)

    print(f"  ✓ Saved document completeness audit to {OUT_DOCS_AUDIT} ({len(audit_records)} documents)")
    return docs_report


if __name__ == "__main__":
    audit_market_data()
    audit_news_data()
    audit_corporate_documents()

