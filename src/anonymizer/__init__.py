"""
Command-line demo, installed as the `anonymizer` command.

Examples:
  uv run anonymizer "John Smith lives in London."
  uv run anonymizer "John Smith lives in London." --model spacy --labels PERSON
"""

import argparse

from anonymizer.anonymize import anonymize, build_pipeline
from anonymizer.extractors.registry import MODELS
from anonymizer.spans import TARGET_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymize sensitive entities in a text.")
    parser.add_argument("text", help="the text to anonymize (put it between quotes)")
    parser.add_argument("--model", choices=list(MODELS), default="gliner", help="which model to use")
    parser.add_argument("--labels", nargs="*", choices=TARGET_LABELS, help="only hide these types (default: all)")
    args = parser.parse_args()

    pipeline = build_pipeline(args.model)
    spans = pipeline.extract(args.text)
    print(anonymize(args.text, spans, set(args.labels) if args.labels else None))


if __name__ == "__main__":
    main()
