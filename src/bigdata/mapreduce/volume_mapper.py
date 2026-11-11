#!/usr/bin/env python3
"""
MapReduce Mapper — Stock Volume Aggregation.

Reads market CSV from stdin (market_features.csv), extracts the symbol
and volume columns, and emits:

    SYMBOL\tvolume

for each row. This is the map phase of a sum-by-key aggregation.

Hadoop Streaming compatible:
    hadoop jar hadoop-streaming.jar \
        -input /financial_analytics/raw/market_features.csv \
        -output /financial_analytics/processed/volume_totals \
        -mapper "python3 volume_mapper.py" \
        -reducer "python3 volume_reducer.py"

Local testing:
    cat data/processed/market_features.csv | python src/bigdata/mapreduce/volume_mapper.py
"""

import csv
import sys


def mapper():
    """
    Read market CSV from stdin, emit symbol\\tvolume for each row.

    Hadoop Streaming Protocol:
    - Input: CSV lines from stdin
    - Output: key\\tvalue pairs to stdout
    """
    reader = csv.reader(sys.stdin)
    header = None

    for row in reader:
        if header is None:
            header = [col.strip().lower() for col in row]
            symbol_idx = header.index("symbol") if "symbol" in header else None
            volume_idx = header.index("volume") if "volume" in header else None
            if symbol_idx is None or volume_idx is None:
                print("ERROR: 'symbol' or 'volume' column not found", file=sys.stderr)
                sys.exit(1)
            continue

        if symbol_idx < len(row) and volume_idx < len(row):
            symbol = row[symbol_idx].strip().upper()
            try:
                volume = int(float(row[volume_idx]))
                if volume >= 0:
                    # Emit: symbol\tvolume
                    print(f"{symbol}\t{volume}")
            except (ValueError, IndexError):
                # Skip malformed rows
                pass


if __name__ == "__main__":
    mapper()
