"""
Model 1: spaCy's small English model (en_core_web_sm).

Philosophy: a fast, classic NER model with a FIXED set of general-purpose
entity types (people, organisations, places, dates, money, ...). It was not
trained for privacy, so it knows nothing about emails, IBANs, jobs or
universities. It is the quick baseline the other models are compared against.
"""

import spacy

from anonymizer.extractors.base import Extractor
from anonymizer.spans import Span

# spaCy's entity type -> our label. Types not listed here are ignored
# (e.g. CARDINAL numbers, NORP nationalities, PRODUCT).
LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",  # countries, cities, states
    "LOC": "LOCATION",  # non-political places: mountains, rivers, ...
    "FAC": "LOCATION",  # facilities: buildings, airports, streets
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "MONEY": "AMOUNT",
}


class SpacyExtractor(Extractor):
    name = "spacy"

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract(self, text: str) -> list[Span]:
        doc = self.nlp(text)
        spans = []
        for ent in doc.ents:
            label = LABEL_MAP.get(ent.label_)
            if label:  # spaCy gives no confidence score, so we use 1.0
                spans.append(Span(ent.start_char, ent.end_char, label, self.name))
        return spans
