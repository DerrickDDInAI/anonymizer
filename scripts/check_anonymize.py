"""
Sanity check: anonymize the benchmark sentences with each model and
show the result next to the expected answer.

Run with:  uv run python scripts/check_anonymize.py            (all models)
           uv run python scripts/check_anonymize.py gliner     (one model)
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from anonymizer.anonymize import anonymize, build_pipeline
from anonymizer.benchmark import load_benchmark
from anonymizer.extractors.registry import available_models

keys = sys.argv[1:] or available_models()
df = load_benchmark()

for key in keys:
    pipeline = build_pipeline(key)
    print(f"\n===== {pipeline.name}")
    exact = 0
    for text, gold in zip(df.text, df.label):
        result = anonymize(text, pipeline.extract(text))
        exact += result == gold
        print("expected:", gold)
        print("got     :", result)
        print()
    print(f"--> {exact}/{len(df)} sentences anonymized exactly like the answer key")

# Show the overlap rule and the label selection on one sentence with the last model.
text = "Barack Obama attended Harvard Law School. His phone number is +1 (202) 555-0199."
spans = pipeline.extract(text)
print(f"\n===== label selection ({pipeline.name})")
print("all      :", anonymize(text, spans))
print("only name:", anonymize(text, spans, labels={"PERSON"}))
