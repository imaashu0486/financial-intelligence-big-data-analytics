#!/usr/bin/env python3
"""
MapReduce Job 2: Cross-Sectional Daily Market Return — Reducer.
Input: sorted date \t daily_adjusted_return
Output: date \t mean_daily_return \t constituent_count
"""

import sys


def reducer():
    current_date = None
    sum_return = 0.0
    count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        dt, ret_str = parts
        try:
            val = float(ret_str)
        except ValueError:
            continue

        if dt == current_date:
            sum_return += val
            count += 1
        else:
            if current_date is not None and count > 0:
                mean_ret = sum_return / count
                sys.stdout.write(f"{current_date}\t{mean_ret:.8f}\t{count}\n")
            current_date = dt
            sum_return = val
            count = 1

    if current_date is not None and count > 0:
        mean_ret = sum_return / count
        sys.stdout.write(f"{current_date}\t{mean_ret:.8f}\t{count}\n")


if __name__ == "__main__":
    reducer()

