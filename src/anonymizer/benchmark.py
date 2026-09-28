"""
Loading the benchmark dataset and recovering the "answer key" positions.

The dataset (data/benchmark.csv) has two columns:
  text  : the original sentence
  label : the same sentence, but every sensitive entity replaced by <LABEL>

Example:
  text  = "John Smith works for Apple Inc. as a software engineer."
  label = "<PERSON> works for <ORG>. as a <JOB>."

The dataset does NOT tell us where each entity starts and ends. We recover
that by lining up the two strings (see align_labels below).
"""

import re
from pathlib import Path

import pandas as pd

from anonymizer.spans import Span

# Default location of the dataset, relative to the project root.
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark.csv"

# Finds placeholders like <PERSON> or <EMAIL_ADDRESS> inside the label string.
PLACEHOLDER = re.compile(r"<([A-Z_]+)>")


def load_benchmark(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV file and return a table with the columns 'text' and 'label'."""
    return pd.read_csv(path, sep=";")


def align_labels(text: str, label: str) -> list[Span]:
    """
    Work out where each entity sits in the original text.

    How it works, in plain words:
      1. Cut the label string into pieces: normal words and <PLACEHOLDER> tags.
      2. Build a search pattern: normal words must appear exactly as they are,
         and each placeholder becomes a "capture whatever is here" slot.
      3. Match that pattern against the original text. Each slot then tells us
         the exact start and end of one entity.

    Returns the entities as a list of Span objects, in reading order.
    Raises ValueError if the label string does not line up with the text.
    """
    pattern_parts = []
    labels = []

    # re.split with a capturing group gives: [words, LABEL, words, LABEL, ..., words]
    pieces = PLACEHOLDER.split(label)
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            # Even positions are literal text: it must match character for character.
            pattern_parts.append(re.escape(piece))
        else:
            # Odd positions are label names: capture at least one character here.
            # "+?" means "as little as possible", so the slot stops as soon as the
            # following literal text can match.
            pattern_parts.append("(.+?)")
            labels.append(piece)

    pattern = "".join(pattern_parts)
    match = re.fullmatch(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Label does not line up with text:\n  text : {text}\n  label: {label}")

    # Group 1 is the first slot, group 2 the second, and so on.
    return [
        Span(start=match.start(i + 1), end=match.end(i + 1), label=lab, source="gold")
        for i, lab in enumerate(labels)
    ]


def load_gold(path: Path = DATA_PATH) -> list[tuple[str, list[Span]]]:
    """Convenience: load the dataset and return (text, gold spans) for every row."""
    df = load_benchmark(path)
    return [(row.text, align_labels(row.text, row.label)) for row in df.itertuples()]