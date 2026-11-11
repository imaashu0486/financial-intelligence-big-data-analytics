#!/usr/bin/env python3
"""
MapReduce Job 4: Yearly Trading Volume per Ticker — Mapper.
Input: CSV line from consolidated market features.
Output: symbol:year \t volume
"""

import csv
import sys


def mapper():
    reader = csv.reader(sys.stdin)
    header = None
    symbol_idx = None
    date_idx = None
    volume_idx = None

    for row in reader:
        if not row:
            continue
        if header is None:
            header = [col.strip().lower() for col in row]
            if "symbol" in header:
                symbol_idx = header.index("symbol")
            elif "ticker" in header:
                symbol_idx = header.index("ticker")
            if "date" in header:
                date_idx = header.index("date")
            if "volume" in header:
                volume_idx = header.index("volume")
            continue

        if symbol_idx is not None and date_idx is not None and volume_idx is not None:
            if symbol_idx < len(row) and date_idx < len(row) and volume_idx < len(row):
                symbol = row[symbol_idx].strip().upper()
                dt_str = row[date_idx].strip()
                vol_str = row[volume_idx].strip()
                if not symbol or not dt_str or not vol_str:
                    continue
                year = dt_str[:4]
                try:
                    vol = int(float(vol_str))
                    if vol >= 0 and len(year) == 4 and year.isdigit():
                        sys.stdout.write(f"{symbol}:{year}\t{vol}\n")
                except ValueError:
                    continue


if __name__ == "__main__":
    mapper()

