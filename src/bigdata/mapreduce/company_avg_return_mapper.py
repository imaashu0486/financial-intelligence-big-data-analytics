#!/usr/bin/env python3
"""
MapReduce Job 3: Company-Level 10-Year Average Daily Return — Mapper.
Input: CSV line from consolidated market features.
Output: symbol \t daily_adjusted_return
"""

import csv
import sys


def mapper():
    reader = csv.reader(sys.stdin)
    header = None
    symbol_idx = None
    return_idx = None

    for row in reader:
        if not row:
            continue
        if header is None:
            header = [col.strip().lower() for col in row]
            if "symbol" in header:
                symbol_idx = header.index("symbol")
            elif "ticker" in header:
                symbol_idx = header.index("ticker")
            if "daily_adjusted_return" in header:
                return_idx = header.index("daily_adjusted_return")
            continue

        if symbol_idx is not None and return_idx is not None:
            if symbol_idx < len(row) and return_idx < len(row):
                symbol = row[symbol_idx].strip().upper()
                ret_str = row[return_idx].strip()
                if not symbol or not ret_str:
                    continue
                try:
                    ret_val = float(ret_str)
                    sys.stdout.write(f"{symbol}\t{ret_val:.8f}\n")
                except ValueError:
                    continue


if __name__ == "__main__":
    mapper()

