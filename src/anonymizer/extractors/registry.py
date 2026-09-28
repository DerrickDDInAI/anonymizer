"""
One place that lists every model we can run.

Adding a new model = one new line here (plus its extractor file).
The imports happen inside the functions so that asking for spaCy does not
also load the heavy transformer libraries.
"""

from anonymizer.extractors.base import Extractor


def _spacy() -> Extractor:
    from anonymizer.extractors.spacy_extractor import SpacyExtractor

    return SpacyExtractor()


def _openmed() -> Extractor:
    from anonymizer.extractors.hf_extractor import HFExtractor

    return HFExtractor()


def _gliner() -> Extractor:
    from anonymizer.extractors.gliner_extractor import GLiNERExtractor

    return GLiNERExtractor()


def _llm() -> Extractor:
    from anonymizer.extractors.llm_extractor import LLMExtractor

    return LLMExtractor()


# key used in the CLI / UI / benchmark -> function that builds the extractor
MODELS = {
    "spacy": _spacy,
    "openmed": _openmed,
    "gliner": _gliner,
    "llm": _llm,  # optional: needs Ollama running, see llm_extractor.py
}


def get_model(key: str) -> Extractor:
    """Build the extractor with the given key. Raises KeyError for unknown keys."""
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Choose from: {list(MODELS)}")
    return MODELS[key]()


def available_models() -> list[str]:
    """The model keys that can actually run right now (the LLM needs Ollama)."""
    from anonymizer.extractors.llm_extractor import ollama_available

    return [key for key in MODELS if key != "llm" or ollama_available()]


def coverage() -> dict[str, set[str]]:
    """For every model, the set of our 12 labels it can produce at all."""
    from anonymizer.extractors import gliner_extractor, hf_extractor, llm_extractor, spacy_extractor

    modules = {"spacy": spacy_extractor, "openmed": hf_extractor, "gliner": gliner_extractor, "llm": llm_extractor}
    return {key: set(module.LABEL_MAP.values()) for key, module in modules.items()}
