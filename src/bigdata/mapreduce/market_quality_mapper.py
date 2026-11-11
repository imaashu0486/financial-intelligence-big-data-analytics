#!/usr/bin/env python3
"""
MapReduce Job 5: Market Data Quality & Integrity Summary — Mapper.
Input: CSV line from market data.
Output: metric_name \t 1
"""

import csv
import sys


def mapper():
    reader = csv.reader(sys.stdin)
    header = None
    open_idx = None
    high_idx = None
    low_idx = None
    close_idx = None
    vol_idx = None
    ret_idx = None

    for row in reader:
        if not row:
            continue
        if header is None:
            header = [col.strip().lower() for col in row]
            open_idx = header.index("open") if "open" in header else None
            high_idx = header.index("high") if "high" in header else None
            low_idx = header.index("low") if "low" in header else None
            close_idx = header.index("close") if "close" in header else None
            vol_idx = header.index("volume") if "volume" in header else None
            ret_idx = header.index("daily_adjusted_return") if "daily_adjusted_return" in header else None
            continue

        sys.stdout.write("TOTAL_ROWS\t1\n")

        # Parse prices
        try:
            o = float(row[open_idx]) if open_idx is not None and open_idx < len(row) and row[open_idx] else None
            h = float(row[high_idx]) if high_idx is not None and high_idx < len(row) and row[high_idx] else None
            l = float(row[low_idx]) if low_idx is not None and low_idx < len(row) and row[low_idx] else None
            c = float(row[close_idx]) if close_idx is not None and close_idx < len(row) and row[close_idx] else None
            v = float(row[vol_idx]) if vol_idx is not None and vol_idx < len(row) and row[vol_idx] else None
        except ValueError:
            sys.stdout.write("PARSE_ERROR\t1\n")
            continue

        if o is None or h is None or l is None or c is None:
            sys.stdout.write("MISSING_PRICE_FIELDS\t1\n")
            continue

        # Check anomalies
        if h < l:
            sys.stdout.write("ANOMALY_HIGH_LESS_THAN_LOW\t1\n")
        elif o <= 0 or h <= 0 or l <= 0 or c <= 0:
            sys.stdout.write("ANOMALY_ZERO_OR_NEGATIVE_PRICE\t1\n")
        else:
            sys.stdout.write("VALID_OHLC\t1\n")

        if v is not None:
            if v < 0:
                sys.stdout.write("ANOMALY_NEGATIVE_VOLUME\t1\n")
            else:
                sys.stdout.write("VALID_VOLUME\t1\n")
        else:
            sys.stdout.write("MISSING_VOLUME\t1\n")

        # Return check
        if ret_idx is not None and ret_idx < len(row) and row[ret_idx].strip():
            try:
                float(row[ret_idx])
                sys.stdout.write("VALID_RETURNS\t1\n")
            except ValueError:
                sys.stdout.write("INVALID_RETURN_VALUE\t1\n")
        else:
            sys.stdout.write("MISSING_RETURN\t1\n")


if __name__ == "__main__":
    mapper()

