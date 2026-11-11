#!/usr/bin/env python3
"""
MapReduce Mapper — Word Count on Financial News.

Reads CSV lines from stdin (financial news), extracts text from the
'title' and 'description' columns, tokenises, and emits:

    word\t1

for each word (lowercased, stopwords removed, min length 3).

This script is compatible with Hadoop Streaming:
    hadoop jar hadoop-streaming.jar \
        -input /financial_analytics/raw/news_cleaned.csv \
        -output /financial_analytics/processed/word_count \
        -mapper "python3 news_mapper.py" \
        -reducer "python3 news_reducer.py"

Local testing:
    cat data/processed/news_cleaned.csv | python src/bigdata/mapreduce/news_mapper.py
"""

import csv
import re
import sys

# Common English stopwords to filter out
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "has", "had", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
    "she", "too", "use", "been", "each", "have", "this", "that", "with",
    "they", "from", "what", "when", "will", "more", "also", "than", "them",
    "into", "some", "very", "just", "over", "such", "like", "only", "most",
    "then", "being", "about", "would", "which", "could", "other", "their",
    "after", "there", "these", "those", "where", "while", "should",
    "market", "nan", "none",  # data artefacts
}


def clean_word(word: str) -> str:
    """Remove non-alphabetic characters and lowercase."""
    return re.sub(r"[^a-z]", "", word.lower())


def mapper():
    """
    Read CSV from stdin, emit word\\t1 for each valid word.

    Hadoop Streaming Protocol:
    - Input: lines from stdin (CSV format)
    - Output: key\\tvalue pairs to stdout
    - Errors/logs: stderr (not captured by Hadoop)
    """
    reader = csv.reader(sys.stdin)
    header = None

    for row in reader:
        if header is None:
            # First line is the CSV header
            header = [col.strip().lower() for col in row]
            # Find title and description column indices
            title_idx = header.index("title") if "title" in header else None
            desc_idx = header.index("description") if "description" in header else None
            if title_idx is None:
                print("ERROR: 'title' column not found", file=sys.stderr)
                sys.exit(1)
            continue

        # Extract text from title and description
        text_parts = []
        if title_idx is not None and title_idx < len(row):
            text_parts.append(row[title_idx])
        if desc_idx is not None and desc_idx < len(row):
            text_parts.append(row[desc_idx])

        text = " ".join(text_parts)

        # Tokenise and emit
        for token in text.split():
            word = clean_word(token)
            if len(word) >= 3 and word not in STOPWORDS:
                # Emit: word\t1
                print(f"{word}\t1")


if __name__ == "__main__":
    mapper()
