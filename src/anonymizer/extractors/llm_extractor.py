"""
Model 4: a small local LLM (large language model) through Ollama.

Philosophy: a GENERATIVE model. Instead of tagging words, we describe the task
in plain English and force the model to answer in a fixed JSON shape (Ollama's
"format" option guarantees the reply matches our schema). This is the most
flexible approach: it can find any type we can describe. It is also by far the
slowest (seconds per sentence instead of milliseconds) and the least
predictable on boundaries.

Setup (not done by `uv sync`, so this model is optional):
  brew install ollama
  ollama serve                       (or start the Ollama app)
  ollama pull qwen2.5:3b-instruct    (about 2 GB, fits an 8 GB Mac)

Two rules that matter for entity extraction with an LLM:
  1. We ask for the entity TEXT, never for character positions: small models
     count characters badly. We locate each text in the sentence ourselves.
  2. We only keep entities that occur word for word in the input. Models like
     to "tidy up" ("Apple Inc" for "Apple Inc."), and those are dropped.
"""

from typing import Literal

import ollama
from pydantic import BaseModel

from anonymizer.extractors.base import Extractor
from anonymizer.spans import TARGET_LABELS, Span

MODEL_NAME = "qwen2.5:3b-instruct"

# The LLM produces our labels directly, so the "map" is the identity.
LABEL_MAP = {label: label for label in TARGET_LABELS}

# The label is restricted to our 12 types, so the model cannot invent categories.
Label = Literal[
    "PERSON", "ORG", "JOB", "EMAIL_ADDRESS", "LOCATION", "AMOUNT",
    "DATE_TIME", "UNIVERSITY", "PHONE_NUMBER", "URL", "IBAN", "SSN",
]


class Entity(BaseModel):
    text: str
    label: Label


class Entities(BaseModel):
    entities: list[Entity]


SYSTEM_PROMPT = (
    "Extract every sensitive entity from the user's text. "
    "Types: PERSON (people's names), ORG (companies, organisations), JOB (job titles), "
    "EMAIL_ADDRESS, LOCATION (countries, cities, states, addresses), AMOUNT (money), "
    "DATE_TIME (dates, years, times), UNIVERSITY (schools, universities), PHONE_NUMBER, "
    "URL, IBAN (bank account numbers), SSN (social security or national numbers). "
    "Copy each entity's text verbatim, exactly as it appears in the input, with no changes. "
    "Return only entities that are present in the text."
)


def ollama_available(model_name: str = MODEL_NAME) -> bool:
    """True if the Ollama server is running and the model has been pulled."""
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        return False
    return any(n == model_name or n.startswith(model_name + ":") for n in names)


def locate(text: str, entities: Entities, source: str) -> list[Span]:
    """
    Turn the entity strings the model returned into spans with positions.
    A cursor per entity text makes sure that a name mentioned twice maps to
    two different places, not twice to the first one.
    """
    spans, cursor = [], {}
    for ent in entities.entities:
        start = text.find(ent.text, cursor.get(ent.text, 0))
        if start == -1 or not ent.text.strip():
            continue  # not in the text word for word -> hallucinated or altered
        spans.append(Span(start, start + len(ent.text), ent.label, source))
        cursor[ent.text] = start + len(ent.text)
    return sorted(spans, key=lambda s: s.start)


class LLMExtractor(Extractor):
    name = "llm"

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def extract(self, text: str) -> list[Span]:
        reply = ollama.chat(
            model=self.model_name,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            format=Entities.model_json_schema(),  # constrained decoding: valid JSON, our shape
            options={"temperature": 0, "seed": 42},  # same answer every run
        )
        return locate(text, Entities.model_validate_json(reply.message.content), self.name)
