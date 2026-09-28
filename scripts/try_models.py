"""
Sanity check for step 4: run each ML model on the benchmark sentences.

Prints, per model: load time, time per sentence, what it found, and a table
of which of the 12 target types the model can produce at all.

Run with:  uv run python scripts/try_models.py            (all models)
           uv run python scripts/try_models.py spacy      (one model)
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

from anonymizer.benchmark import load_benchmark
from anonymizer.extractors.registry import available_models, coverage, get_model
from anonymizer.spans import TARGET_LABELS

# Which of our 12 labels each model can output in principle (from its label map).
CAN_PRODUCE = coverage()

keys = sys.argv[1:] or available_models()
texts = list(load_benchmark().text)

for key in keys:
    t0 = time.time()
    model = get_model(key)
    print(f"\n===== {key}  (loaded in {time.time() - t0:.1f}s)")

    t0 = time.time()
    for text in texts:
        print(text)
        for span in model.extract(text):
            print(f"  {span.label:<14} {span.score:.2f} {span.text(text)!r}")
    print(f"--> {(time.time() - t0) / len(texts) * 1000:.0f} ms per sentence")

print("\n===== coverage: can the model produce this type at all?")
print(f"{'label':<14}" + "".join(f"{k:>9}" for k in keys))
for label in TARGET_LABELS:
    print(f"{label:<14}" + "".join(f"{'yes' if label in CAN_PRODUCE[k] else '-':>9}" for k in keys))
print("(EMAIL, PHONE, URL, IBAN, SSN, AMOUNT are also covered by the regex layer for every model)")
