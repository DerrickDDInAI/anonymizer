"""
A "span" is one piece of sensitive text we found (or should find) in a sentence.

Example: in "John Smith works for Apple", the name "John Smith" is a span that
starts at character 0, ends at character 10, and has the label PERSON.

Every part of the project (regex, ML models, benchmark, anonymizer) speaks in
spans, so they can all be compared and combined in the same way.
"""

from dataclasses import dataclass

# Set the 12 kinds of sensitive information to detect
TARGET_LABELS = [
    "PERSON",
    "ORG",
    "JOB",
    "EMAIL_ADDRESS",
    "LOCATION",
    "AMOUNT",
    "DATE_TIME",
    "UNIVERSITY",
    "PHONE_NUMBER",
    "URL",
    "IBAN",
    "SSN",
]


@dataclass
class Span:
    """One detected entity: where it is in the text and what kind it is."""

    start: int  # index of the first character
    end: int  # index just after the last character, so text[start:end] is the entity
    label: str  # one of TARGET_LABELS, e.g. "PERSON"
    source: str = ""  # who found it: "gold" (the answer key, i.e. from the benchmark), "regex", "spacy", ...
    score: float = 1.0  # how confident the finder was (1.0 = certain), 
                        # 1.0 by default when source is regex or model doesn't provide a confidence score

    def text(self, full_text: str) -> str:
        """Return the actual words this span covers inside the full text."""
        return full_text[self.start : self.end]


def overlaps(a: Span, b: Span) -> bool:
    """True if the two spans share at least one character."""
    return a.start < b.end and b.start < a.end


def merge_adjacent(spans: list[Span], text: str) -> list[Span]:
    """
    Glue together neighbouring spans with the same label that are separated by
    spaces only. Example: a model that tags "John" as first name and "Smith"
    as last name gives two PERSON spans; we want one span "John Smith".
    """
    merged: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        last = merged[-1] if merged else None
        gap = text[last.end : span.start] if last else None
        if last and last.label == span.label and gap.strip() == "":
            # Extend the previous span instead of adding a new one.
            last.end = span.end
            last.score = min(last.score, span.score)
        else:
            merged.append(Span(span.start, span.end, span.label, span.source, span.score))
    return merged
