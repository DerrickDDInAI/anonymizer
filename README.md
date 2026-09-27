# anonymizer

Detects and anonymizes sensitive entities in text (PERSON, ORG, LOCATION, …) using pre-trained NER models + regex recognizers.
Also benchmarks the models against a labelled dataset.

## Setup (2 ways)
### 1st way (recommended; uv needs to be installed)
1. Clone repository
2. Open a terminal and navigate to repository: `cd anonymizer`
3. Run `uv sync`. This will install Python 3.12, deps, spaCy model

### 2nd way
1. Clone repository
2. Open a terminal and navigate to repository: `cd anonymizer`
3. python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

## TODO: Usage / Benchmark / Demo