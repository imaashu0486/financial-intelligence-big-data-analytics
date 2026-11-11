#!/usr/bin/env python3
"""
MapReduce Job 5: Market Data Quality & Integrity Summary — Reducer.
Input: sorted metric_name \t count
Output: metric_name \t total_count
"""

import sys


def reducer():
    current_metric = None
    metric_sum = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        metric, count_str = parts
        try:
            cnt = int(count_str)
        except ValueError:
            continue

        if metric == current_metric:
            metric_sum += cnt
        else:
            if current_metric is not None:
                sys.stdout.write(f"{current_metric}\t{metric_sum}\n")
            current_metric = metric
            metric_sum = cnt

    if current_metric is not None:
        sys.stdout.write(f"{current_metric}\t{metric_sum}\n")


if __name__ == "__main__":
    reducer()

