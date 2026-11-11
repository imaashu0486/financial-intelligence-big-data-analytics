#!/usr/bin/env python3
"""
MapReduce Job 1: Daily Trading-Volume Aggregation by Ticker — Mapper.
Input: CSV line (header or data row).
Output: ticker \t volume
"""

import sys

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("date,") or line.startswith("Date,"):
        continue
    parts = line.split(",")
    if len(parts) >= 8:
        # Expected consolidated schema: date,symbol,open,high,low,close,adj_close,volume,...
        symbol = parts[1].strip().upper()
        vol_str = parts[7].strip()
        try:
            vol = int(float(vol_str))
            if symbol and vol >= 0:
                sys.stdout.write(f"{symbol}\t{vol}\n")
        except (ValueError, IndexError):
            continue

