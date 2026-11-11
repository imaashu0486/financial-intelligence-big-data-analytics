#!/usr/bin/env python3
"""
MapReduce Reducer — Stock Volume Aggregation.

Reads sorted symbol\\tvolume lines from stdin (output of volume_mapper.py
after sort), sums volumes per symbol, and emits:

    SYMBOL\ttotal_volume

Hadoop Streaming Protocol:
- Hadoop's shuffle/sort phase groups all lines with the same symbol
  together, so the reducer only needs to accumulate until the key changes.

Local testing:
    cat market.csv | python volume_mapper.py | sort | python volume_reducer.py

This script does NOT depend on Spark.
"""

import sys


def reducer():
    """
    Read sorted symbol\\tvolume lines from stdin, sum per symbol.
    """
    current_symbol = None
    current_total = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            continue

        symbol, volume_str = parts

        try:
            volume = int(volume_str)
        except ValueError:
            continue

        if symbol == current_symbol:
            current_total += volume
        else:
            if current_symbol is not None:
                print(f"{current_symbol}\t{current_total}")
            current_symbol = symbol
            current_total = volume

    # Emit the last symbol
    if current_symbol is not None:
        print(f"{current_symbol}\t{current_total}")


if __name__ == "__main__":
    reducer()
