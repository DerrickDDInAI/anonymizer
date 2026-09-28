"""
Turning found spans into an anonymized text.

Example:
  text  = "John Smith works for Apple."
  spans = [PERSON 0-10, ORG 21-26]
  result = "<PERSON> works for <ORG>."
"""

from anonymizer.extractors.composite import CompositeExtractor
from anonymizer.extractors.registry import get_model
from anonymizer.spans import Span


def anonymize(text: str, spans: list[Span], labels: set[str] | None = None) -> str:
    """
    Replace each span in the text by its <LABEL> tag.

    labels: optional set of label names to anonymize. If given, spans with
            other labels are left untouched (so a user can e.g. hide names
            but keep locations). None means "anonymize everything".

    We replace from right to left: replacing "John Smith" by "<PERSON>" changes
    the length of the text, which would shift the positions of every span
    after it. Going backwards, the positions we still need never move.
    """
    if labels is not None:
        spans = [s for s in spans if s.label in labels]

    result = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        result = result[: span.start] + f"<{span.label}>" + result[span.end :]
    return result


def build_pipeline(model_key: str) -> CompositeExtractor:
    """Convenience: model by key (see registry) + regex layer, ready to use."""
    return CompositeExtractor(get_model(model_key))
