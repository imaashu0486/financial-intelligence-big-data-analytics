#!/usr/bin/env python3
"""
MapReduce Job 3: Company-Level 10-Year Average Daily Return — Reducer.
Input: sorted symbol \t daily_adjusted_return
Output: symbol \t mean_daily_return \t observation_count
"""

import sys


def reducer():
    current_symbol = None
    sum_return = 0.0
    count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        sym, ret_str = parts
        try:
            val = float(ret_str)
        except ValueError:
            continue

        if sym == current_symbol:
            sum_return += val
            count += 1
        else:
            if current_symbol is not None and count > 0:
                mean_ret = sum_return / count
                sys.stdout.write(f"{current_symbol}\t{mean_ret:.8f}\t{count}\n")
            current_symbol = sym
            sum_return = val
            count = 1

    if current_symbol is not None and count > 0:
        mean_ret = sum_return / count
        sys.stdout.write(f"{current_symbol}\t{mean_ret:.8f}\t{count}\n")


if __name__ == "__main__":
    reducer()

