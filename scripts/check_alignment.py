"""
Sanity check: print the recovered gold entities for every example.

Run with:  uv run python scripts/check_alignment.py

Eyeball the output: every entity should show the right words and label.
"""

from anonymizer.benchmark import load_gold

for i, (text, spans) in enumerate(load_gold(), start=1):
    print(f"--- example {i}")
    print(text)
    for span in spans:
        print(f"  {span.label:<14} {span.start:>3}-{span.end:<3} '{span.text(text)}'")
