#!/usr/bin/env python3
"""
MapReduce Job 1: Daily Trading-Volume Aggregation by Ticker — Reducer.
Input: sorted ticker \t volume
Output: ticker \t total_volume \t record_count
"""

import sys


def reducer():
    current_symbol = None
    current_volume = 0
    current_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        symbol, vol_str = parts
        try:
            vol = int(vol_str)
        except ValueError:
            continue

        if symbol == current_symbol:
            current_volume += vol
            current_count += 1
        else:
            if current_symbol is not None:
                sys.stdout.write(f"{current_symbol}\t{current_volume}\t{current_count}\n")
            current_symbol = symbol
            current_volume = vol
            current_count = 1

    if current_symbol is not None:
        sys.stdout.write(f"{current_symbol}\t{current_volume}\t{current_count}\n")


if __name__ == "__main__":
    reducer()

