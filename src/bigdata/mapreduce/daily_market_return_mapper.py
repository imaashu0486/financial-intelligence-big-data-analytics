#!/usr/bin/env python3
"""
MapReduce Job 2: Cross-Sectional Daily Market Return — Mapper.
Input: CSV line from consolidated market features.
Output: date \t daily_adjusted_return
"""

import csv
import sys


def mapper():
    reader = csv.reader(sys.stdin)
    header = None
    date_idx = None
    return_idx = None

    for row in reader:
        if not row:
            continue
        if header is None:
            header = [col.strip().lower() for col in row]
            if "date" in header:
                date_idx = header.index("date")
            if "daily_adjusted_return" in header:
                return_idx = header.index("daily_adjusted_return")
            continue

        if date_idx is not None and return_idx is not None:
            if date_idx < len(row) and return_idx < len(row):
                date_str = row[date_idx].strip()
                ret_str = row[return_idx].strip()
                if not date_str or not ret_str:
                    continue
                try:
                    ret_val = float(ret_str)
                    sys.stdout.write(f"{date_str}\t{ret_val:.8f}\n")
                except ValueError:
                    continue


if __name__ == "__main__":
    mapper()

