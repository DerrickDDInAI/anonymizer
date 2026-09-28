"""
Model 2: OpenMed PII model (OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1).

Philosophy: a transformer trained SPECIFICALLY to find personal information,
with a fixed list of about 50 PII types (first name, last name, email, phone,
street address, ...). It is small (44M parameters) and runs on a laptop CPU.
Because its list is fixed, it can never find types it was not trained on:
there is no IBAN, university or money type, so those score 0 by construction.
"""

from transformers import pipeline

from anonymizer.extractors.base import Extractor
from anonymizer.spans import Span, merge_adjacent

MODEL_NAME = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"

# OpenMed's type -> our label. Types not listed here are ignored.
LABEL_MAP = {
    "first_name": "PERSON",
    "last_name": "PERSON",  # first + last are glued together by merge_adjacent
    "company_name": "ORG",
    "occupation": "JOB",
    "city": "LOCATION",
    "country": "LOCATION",
    "state": "LOCATION",
    "county": "LOCATION",
    "street_address": "LOCATION",
    "postcode": "LOCATION",
    "date": "DATE_TIME",
    "date_time": "DATE_TIME",
    "time": "DATE_TIME",
    "date_of_birth": "DATE_TIME",
    "email": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER",
    "url": "URL",
    "ssn": "SSN",
}


class HFExtractor(Extractor):
    name = "openmed"

    def __init__(self, model_name: str = MODEL_NAME):
        # "simple" aggregation: the model reads word pieces, this glues the
        # pieces of one word back together into a single entity.
        self.pipe = pipeline("token-classification", model=model_name, aggregation_strategy="simple")

    def extract(self, text: str) -> list[Span]:
        spans = []
        for ent in self.pipe(text):
            label = LABEL_MAP.get(ent["entity_group"])
            if not label:
                continue
            start, end = ent["start"], ent["end"]
            # The tokenizer sometimes includes the space before a word; trim it.
            while start < end and text[start].isspace():
                start += 1
            spans.append(Span(start, end, label, self.name, float(ent["score"])))
        return merge_adjacent(spans, text)
