"""
Composite extractor: one ML model + the regex layer, combined into a single
list of non-overlapping spans.

Why combine? The model is good at names, companies, places and jobs. The regex
layer is better at emails, phones, IBANs and the like. Together they cover all
12 target types. The regex layer is the same for every model, so comparing
"spacy+regex" against "gliner+regex" is a fair comparison of the models.
"""

from anonymizer.extractors.base import Extractor
from anonymizer.extractors.regex_extractor import RegexExtractor
from anonymizer.spans import Span, overlaps


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """
    When two spans claim the same characters, keep only one. The rule:
      1. a regex span beats a model span (regex is exact for structured types),
      2. otherwise the longer span wins (e.g. "New York City" over "New York"),
      3. if still tied, the higher confidence score wins.

    We sort by that rule and then walk the list, keeping a span only if it
    does not overlap anything already kept ("greedy" selection).
    """
    ranked = sorted(
        spans,
        key=lambda s: (s.source == "regex", s.end - s.start, s.score),
        reverse=True,  # best first
    )
    kept: list[Span] = []
    for span in ranked:
        if not any(overlaps(span, other) for other in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


class CompositeExtractor(Extractor):
    """Runs a model and the regex layer, then removes overlaps."""

    def __init__(self, model: Extractor):
        self.model = model
        self.regex = RegexExtractor()
        self.name = f"{model.name}+regex"

    def extract(self, text: str) -> list[Span]:
        spans = self.model.extract(text) + self.regex.extract(text)
        return resolve_overlaps(spans)
