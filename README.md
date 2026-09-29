# anonymizer

Finds sensitive entities in free text and replaces them by tags, using pre-trained NER models plus a regex layer.
Benchmarks several models on a labelled dataset. Comes with a Streamlit demo, a command-line tool and a FastAPI endpoint.

```
John Smith works for Apple Inc. as a software engineer. His email is john.smith@apple.com.
<PERSON> works for <ORG> as a <JOB>. His email is <EMAIL_ADDRESS>.
```

Twelve entity types: PERSON, ORG, JOB, EMAIL_ADDRESS, LOCATION, AMOUNT, DATE_TIME, UNIVERSITY, PHONE_NUMBER, URL, IBAN, SSN.

## Setup

Python 3.12. Two ways to install, both tested from a fresh clone.

### 1. With uv (recommended)

1. Install [uv](https://docs.astral.sh/uv/) if needed.
2. Clone the repository and go into it: `cd anonymizer`
3. Run `uv sync`. This installs Python 3.12, all dependencies and the spaCy model from the lock file.

### 2. With pip

1. Clone the repository and go into it: `cd anonymizer`
2. `python3.12 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt && pip install -e .`

The two transformer models (OpenMed, GLiNER) are downloaded from Hugging Face on first use, about 1 GB in total.

### Optional: the local LLM model

A fourth model uses a small local language model through [Ollama](https://ollama.com). It is optional: everything else works without it, and it is skipped automatically when Ollama is not running.

```bash
brew install ollama
ollama serve
ollama pull qwen2.5:3b-instruct
```

## Usage

Prefix the commands with `uv run` (uv) or activate the venv first (pip).

**Web demo** (pick one or several models, choose which types to hide, compare results side by side, read the benchmark report):

```bash
uv run streamlit run app/app.py
```

**Command line:**

```bash
uv run anonymizer "Elon Musk is the CEO of SpaceX." --model gliner --labels PERSON ORG
```

**Web API** (interactive documentation at http://127.0.0.1:8000/docs):

```bash
uv run uvicorn api.main:app
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "John Smith works for Apple Inc. in London.", "model": "gliner"}'
```

`GET /models` lists the model keys that can run. `POST /predict` takes `text`, an optional `model` (default `gliner`) and an optional list of `labels`, and returns the anonymized text plus every entity with its position, source and score.

**Benchmark** (regenerates `reports/benchmark.md`, takes a few minutes):

```bash
uv run python scripts/run_benchmark.py
```

## How it works

```
text  ->  ML model  (PERSON, ORG, LOCATION, JOB, DATE_TIME, UNIVERSITY)   \
      ->  regex layer (EMAIL, PHONE, URL, IBAN, SSN, AMOUNT, ISO dates)    ->  resolve overlaps  ->  anonymize
```

- **Regex for structured types, models for the rest.** Emails, phone numbers, IBANs and the like have a fixed shape, so a pattern finds them reliably (IBANs are also checked with the mod-97 rule). Names, companies, places and jobs need language understanding.
- **The regex layer is applied identically on top of every model**, so the benchmark compares only what the models add.
- **Overlap rule** when two spans claim the same characters: a regex span beats a model span, otherwise the longer span wins, otherwise the higher confidence score.
- **Anonymizing** replaces the kept spans right to left by `<LABEL>`, optionally only for chosen types.

Every extractor follows the same small contract (`extract(text) -> list[Span]`), and `extractors/registry.py` maps a key to a model. Adding a model is one new file plus one line in the registry.

### Folder layout

```
src/anonymizer/
  spans.py                 Span record, target labels, overlap and merge helpers
  benchmark.py             dataset loading, gold-span recovery, scoring, report writer
  anonymize.py             replace spans by tags, build a model+regex pipeline
  extractors/              base contract, regex, spacy, hf (OpenMed), gliner, llm, composite, registry
app/app.py                 Streamlit demo
api/main.py                FastAPI endpoint
scripts/                   sanity-check scripts, one per step, plus run_benchmark.py
notebooks/                 step-by-step notebooks that mirror the package code
data/                      benchmark.csv (given), extra.csv (6 sentences of our own)
reports/                   benchmark.md, presentation.pptx
```

### Recovering the gold spans

The dataset only gives each sentence and a copy with entities replaced by `<LABEL>`, no positions. `align_labels` turns the label string into a pattern where each tag captures anything and everything else must match literally, then matches it against the original sentence. The whole sentence has to match, so `$1.65 billion` is captured whole.

## Models

| key | model | philosophy | ms per sentence |
|---|---|---|---|
| `spacy` | spaCy `en_core_web_sm` | classic NER, fixed general types; no JOB or UNIVERSITY | 7 |
| `openmed` | `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | transformer trained for personal data, about 50 fixed PII types; no IBAN, university or money type | ~100 |
| `gliner` | `knowledgator/gliner-pii-base-v1.0` | open schema: we describe the types in plain English at prediction time | ~80 |
| `llm` | `qwen2.5:3b-instruct` via Ollama (optional) | generative model forced to answer in a JSON schema; asked for entity text only, positions located by us | ~4000 |

## Results

Overall F1 with the regex layer. Strict requires the exact same boundaries as the answer key, lenient only requires overlap with the right label. Full per-type tables in [reports/benchmark.md](reports/benchmark.md).

| pipeline | official data, strict | official data, lenient | extra data, strict |
|---|---|---|---|
| spacy+regex | 85% | 87% | 81% |
| openmed+regex | 76% | 79% | 93% |
| gliner+regex | 71% | 87% | 95% |
| llm+regex | 61% | 76% | 90% |

What the numbers say:

- spaCy wins strict on the official data because it is conservative and its boundaries match the answer key's style, but it can never find JOB or UNIVERSITY.
- GLiNER ties spaCy on lenient. The strict gap is boundaries (it merges "Seattle, Washington" into one span), not misses. It is the only model covering every semantic type.
- OpenMed misses organisations and years on the official data but is strong on the extra data, where jobs, addresses and structured types dominate.
- The regex layer scores 100% on every structured type for every model.
- The LLM is decent on names and jobs, loose on boundaries, and about 500 times slower than spaCy.

### Caveats

- The official dataset has 10 sentences and the extra one 6. One entity more or less moves a per-type score by tens of points. Read the numbers as indications.
- The answer key is noisy: the trailing period after "Apple Inc" is handled inconsistently, the area code "212" is tagged LOCATION, "The Eiffel Tower" is not tagged at all, and IBAN and SSN never appear. That is why strict and lenient are both reported and why `data/extra.csv` exists.

## Why FastAPI

FastAPI validates the input for us (a missing text or unknown model is rejected before our code runs), generates the interactive `/docs` page automatically, and uses type hints as the contract. Flask would need extra libraries and code for all three.

## Git workflow

One commit per roadmap step with a `type: message` prefix (feat, fix, docs, chore). Design decisions and the roadmap were written down before coding. The environment is pinned by `uv.lock`, with `requirements.txt` exported as a pip fallback.

## Future work

- A larger, cleaner labelled set with span positions.
- Automated tests (pytest) around the alignment, the regex patterns and the overlap rule. For now the `scripts/` folder holds sanity checks.
- Presidio as a prior-art comparison row.
- An optional PRONOUN/GENDER type via a word list.
- Tuning GLiNER's label wording and threshold; regex patterns for more countries.
