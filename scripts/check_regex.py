"""
Sanity check: run the regex extractor on the examples from the
assignment brief and on the benchmark sentences, and print what it finds.

Run with:  uv run python scripts/check_regex.py
"""

from anonymizer.benchmark import load_benchmark
from anonymizer.extractors.regex_extractor import RegexExtractor

extractor = RegexExtractor()

# Examples straight from the assignment brief, plus a few tricky ones.
samples = [
    "Contact John.doe@ordina.be or call 015 29 58 58 before 2022-12-27 08:26:49.21.",
    "Transfer €50.000 to BE68 5390 0754 7034. Belgian SSN: 82.05.30-025.56, US SSN 123-45-6789.",
    "See https://example.com/page?id=1, or www.amazon.com. Fake IBAN: BE00 1234 5678 9012.",
    "Born in 1975, phone +1 (202) 555-0199, zip WA 98052, worth $1.65 billion or 20 euros.",
]

print("===== brief examples")
for text in samples:
    print(text)
    for span in extractor.extract(text):
        print(f"  {span.label:<14} {span.text(text)!r}")

print("\n===== benchmark sentences (only sentences where regex finds something)")
for text in load_benchmark().text:
    spans = extractor.extract(text)
    if spans:
        print(text)
        for span in spans:
            print(f"  {span.label:<14} {span.text(text)!r}")
