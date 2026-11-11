#!/usr/bin/env python3
"""
MapReduce Job 4: Yearly Trading Volume per Ticker — Reducer.
Input: sorted symbol:year \t volume
Output: symbol \t year \t total_volume \t trading_days
"""

import sys


def reducer():
    current_key = None
    total_volume = 0
    days_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        key, vol_str = parts
        try:
            vol = int(vol_str)
        except ValueError:
            continue

        if key == current_key:
            total_volume += vol
            days_count += 1
        else:
            if current_key is not None and ":" in current_key:
                sym, yr = current_key.split(":", 1)
                sys.stdout.write(f"{sym}\t{yr}\t{total_volume}\t{days_count}\n")
            current_key = key
            total_volume = vol
            days_count = 1

    if current_key is not None and ":" in current_key:
        sym, yr = current_key.split(":", 1)
        sys.stdout.write(f"{sym}\t{yr}\t{total_volume}\t{days_count}\n")


if __name__ == "__main__":
    reducer()

