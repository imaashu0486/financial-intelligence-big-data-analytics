#!/usr/bin/env python3
"""
MapReduce Reducer — Word Count on Financial News.

Reads sorted word\\t1 lines from stdin (output of news_mapper.py after sort),
aggregates counts per word, and emits:

    word\ttotal_count

Hadoop Streaming Protocol:
- Hadoop sorts mapper output by key before feeding to reducer.
- For local testing, use Unix sort:
    cat news.csv | python news_mapper.py | sort | python news_reducer.py

This script does NOT depend on Spark.
"""

import sys


def reducer():
    """
    Read sorted key\\tvalue lines from stdin, sum counts per word.

    Hadoop Streaming ensures all lines with the same key arrive
    consecutively (shuffle/sort phase). The reducer simply accumulates
    until the key changes.
    """
    current_word = None
    current_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            continue

        word, count_str = parts

        try:
            count = int(count_str)
        except ValueError:
            continue

        # Hadoop guarantees sorted input — same words are consecutive
        if word == current_word:
            current_count += count
        else:
            # Emit the previous word's total
            if current_word is not None:
                print(f"{current_word}\t{current_count}")
            current_word = word
            current_count = count

    # Emit the last word
    if current_word is not None:
        print(f"{current_word}\t{current_count}")


if __name__ == "__main__":
    reducer()
