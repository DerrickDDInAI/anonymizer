"""
The common shape every extractor follows.

An "extractor" is anything that reads a text and returns the sensitive
entities it found, as a list of Span objects. Regex rules, spaCy, OpenMed and
GLiNER are all extractors, so the rest of the project can treat them the same.
"""

from anonymizer.spans import Span


class Extractor:
    """Base class: subclasses must set a name and implement extract()."""

    name: str = "base"

    def extract(self, text: str) -> list[Span]:
        """Return the entities found in the text, sorted by start position."""
        raise NotImplementedError
