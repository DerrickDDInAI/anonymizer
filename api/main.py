"""
Web API (FastAPI) exposing the anonymizer as a service.

Run with:  uv run uvicorn api.main:app
Then open http://127.0.0.1:8000/docs for an interactive page to try it.

Endpoints:
  GET  /models   -> the model keys you can choose from
  POST /predict  -> anonymize a text; body: {"text": "...", "model": "gliner", "labels": ["PERSON"]}

Why FastAPI rather than Flask?
  - It checks the input for us: a request with a missing text or an unknown
    model key is rejected with a clear error before our code runs (pydantic).
  - It generates the interactive documentation page (/docs) automatically,
    which doubles as a demo.
  - It is built on modern Python type hints, so the request and response
    shapes below are also the documentation.
  Flask would need extra libraries and code for all three.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anonymizer.anonymize import anonymize, build_pipeline
from anonymizer.extractors.registry import MODELS, available_models
from anonymizer.spans import TARGET_LABELS

app = FastAPI(title="Anonymizer API", description="Find and hide sensitive entities in text.")

# Loaded pipelines, kept in memory so a model is only loaded once per process.
_pipelines = {}


def get_pipeline(model_key: str):
    if model_key not in _pipelines:
        _pipelines[model_key] = build_pipeline(model_key)
    return _pipelines[model_key]


# ---- request and response shapes -------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["John Smith works for Apple Inc. in London."])
    model: str = Field("gliner", description=f"one of {list(MODELS)}")
    labels: list[str] | None = Field(None, description=f"only hide these types; default all of {TARGET_LABELS}")


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    source: str  # "regex" or the model name
    score: float


class PredictResponse(BaseModel):
    anonymized: str
    entities: list[Entity]


# ---- endpoints -------------------------------------------------------------
@app.get("/models")
def list_models() -> list[str]:
    """The model keys accepted by /predict (the llm only when Ollama is running)."""
    return available_models()


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    """Anonymize the text and return the entities that were found."""
    if request.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{request.model}'. Choose from {list(MODELS)}")
    unknown = set(request.labels or []) - set(TARGET_LABELS)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown labels {sorted(unknown)}. Choose from {TARGET_LABELS}")

    text = request.text
    spans = get_pipeline(request.model).extract(text)
    labels = set(request.labels) if request.labels else None
    kept = [s for s in spans if labels is None or s.label in labels]
    return PredictResponse(
        anonymized=anonymize(text, spans, labels),
        entities=[Entity(text=s.text(text), label=s.label, start=s.start, end=s.end, source=s.source, score=s.score) for s in kept],
    )
