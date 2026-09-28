"""
Run the full benchmark and write reports/benchmark.md.

Run with:  uv run python scripts/run_benchmark.py
Takes a minute or two (the transformer models have to load).
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from anonymizer.benchmark import DATA_PATH, EXTRA_PATH, run_benchmark, write_report
from anonymizer.extractors.registry import available_models, coverage

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "benchmark.md"

keys = available_models()  # "llm" is skipped automatically if Ollama is not running
print("models:", keys)

results = {}
for path in (DATA_PATH, EXTRA_PATH):
    print(f"=== {path.name}")
    results[path.name] = run_benchmark(keys, path)
    for key, r in results[path.name].items():
        sp, sr, sf = r["overall"]["strict"]
        lp, lr, lf = r["overall"]["lenient"]
        print(f"{key:<8} strict F1 {sf:.2f}  lenient F1 {lf:.2f}  exact {r['exact']}/{r['n']}  {r['ms']:.0f} ms")

write_report(results, coverage(), REPORT_PATH)
print(f"\nreport written to {REPORT_PATH}")
