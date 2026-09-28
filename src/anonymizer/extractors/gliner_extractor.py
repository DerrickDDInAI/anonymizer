"""
Model 3: GLiNER PII model (knowledgator/gliner-pii-base-v1.0).

Philosophy: an OPEN-schema model. Instead of a fixed list of types, we hand it
the names of the types we want, in plain English, at prediction time. That
means we can ask for "university" or "amount of money" even though nobody
trained it on exactly those. The wording of the label matters: "job title"
works better than "job", for example. Larger and slower than the other two.
"""

from gliner import GLiNER

from anonymizer.extractors.base import Extractor
from anonymizer.spans import Span

MODEL_NAME = "knowledgator/gliner-pii-base-v1.0"

# The plain-English label we ask GLiNER for -> our label.
LABEL_MAP = {
    "person": "PERSON",
    "organization": "ORG",
    "job title": "JOB",
    "location": "LOCATION",
    "date": "DATE_TIME",
    "university": "UNIVERSITY",
    "amount of money": "AMOUNT",
}


class GLiNERExtractor(Extractor):
    name = "gliner"

    def __init__(self, model_name: str = MODEL_NAME, threshold: float = 0.4):
        self.model = GLiNER.from_pretrained(model_name)
        self.threshold = threshold  # entities scored below this are dropped

    def extract(self, text: str) -> list[Span]:
        entities = self.model.predict_entities(text, list(LABEL_MAP), threshold=self.threshold)
        return [
            Span(ent["start"], ent["end"], LABEL_MAP[ent["label"]], self.name, float(ent["score"]))
            for ent in entities
        ]
