"""
Loading the benchmark dataset and recovering the "answer key" positions.

The dataset (data/benchmark.csv) has two columns:
  text  : the original sentence
  label : the same sentence, but every sensitive entity replaced by <LABEL>

Example:
  text  = "John Smith works for Apple Inc. as a software engineer."
  label = "<PERSON> works for <ORG>. as a <JOB>."

The dataset does NOT tell us where each entity starts and ends. We recover
that by lining up the two strings (see align_labels below).
"""

import re
from pathlib import Path

import pandas as pd

from anonymizer.spans import Span

# Default location of the dataset, relative to the project root.
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark.csv"

# Finds placeholders like <PERSON> or <EMAIL_ADDRESS> inside the label string.
PLACEHOLDER = re.compile(r"<([A-Z_]+)>")


def load_benchmark(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV file and return a table with the columns 'text' and 'label'."""
    return pd.read_csv(path, sep=";")


def align_labels(text: str, label: str) -> list[Span]:
    """
    Work out where each entity sits in the original text.

    How it works, in plain words:
      1. Cut the label string into pieces: normal words and <PLACEHOLDER> tags.
      2. Build a search pattern: normal words must appear exactly as they are,
         and each placeholder becomes a "capture whatever is here" slot.
      3. Match that pattern against the original text. Each slot then tells us
         the exact start and end of one entity.

    Returns the entities as a list of Span objects, in reading order.
    Raises ValueError if the label string does not line up with the text.
    """
    pattern_parts = []
    labels = []

    # re.split with a capturing group gives: [words, LABEL, words, LABEL, ..., words]
    pieces = PLACEHOLDER.split(label)
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            # Even positions are literal text: it must match character for character.
            pattern_parts.append(re.escape(piece))
        else:
            # Odd positions are label names: capture at least one character here.
            # "+?" means "as little as possible", so the slot stops as soon as the
            # following literal text can match.
            pattern_parts.append("(.+?)")
            labels.append(piece)

    pattern = "".join(pattern_parts)
    match = re.fullmatch(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Label does not line up with text:\n  text : {text}\n  label: {label}")

    # Group 1 is the first slot, group 2 the second, and so on.
    return [
        Span(start=match.start(i + 1), end=match.end(i + 1), label=lab, source="gold")
        for i, lab in enumerate(labels)
    ]


def load_gold(path: Path = DATA_PATH) -> list[tuple[str, list[Span]]]:
    """Convenience: load the dataset and return (text, gold spans) for every row."""
    df = load_benchmark(path)
    return [(row.text, align_labels(row.text, row.label)) for row in df.itertuples()]


# ---------------------------------------------------------------------------
# Scoring: how good are a model's spans compared with the gold spans?
# ---------------------------------------------------------------------------

import time

from anonymizer.anonymize import anonymize, build_pipeline
from anonymizer.spans import TARGET_LABELS, overlaps

EXTRA_PATH = DATA_PATH.parent / "extra.csv"


def count_matches(predicted: list[Span], gold: list[Span], strict: bool) -> dict[str, dict[str, int]]:
    """
    Compare the spans a model found with the gold spans of ONE text.

    For every label we count:
      tp (true positives) : predicted spans that match a gold span
      fp (false positives): predicted spans with no matching gold span
      fn (false negatives): gold spans the model did not find

    "match" means: same label and
      strict  -> exactly the same start and end
      lenient -> the two spans overlap (boundaries may differ a little,
                 e.g. "Apple Inc." versus "Apple Inc")
    Each gold span can be matched at most once.
    """
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in TARGET_LABELS}
    unmatched_gold = list(gold)

    for pred in predicted:
        if pred.label not in counts:
            continue
        hit = None
        for g in unmatched_gold:
            if g.label != pred.label:
                continue
            same = (pred.start == g.start and pred.end == g.end) if strict else overlaps(pred, g)
            if same:
                hit = g
                break
        if hit is not None:
            counts[pred.label]["tp"] += 1
            unmatched_gold.remove(hit)
        else:
            counts[pred.label]["fp"] += 1

    for g in unmatched_gold:
        counts[g.label]["fn"] += 1
    return counts


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    precision = of everything the model flagged, how much was right?
    recall    = of everything it should have flagged, how much did it find?
    f1        = one number that balances the two.
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(pipeline, examples: list[tuple[str, list[Span]]]) -> dict:
    """
    Run one pipeline (model + regex) over all examples and collect the scores.

    Returns a dict with:
      per_label : DataFrame, one row per label, strict and lenient P/R/F1 + support
      overall   : dict with strict/lenient micro P/R/F1 (all labels pooled)
      exact     : how many texts were anonymized exactly like the answer key
      ms        : average milliseconds per text
    """
    totals = {
        mode: {label: {"tp": 0, "fp": 0, "fn": 0} for label in TARGET_LABELS} for mode in ("strict", "lenient")
    }
    exact = 0
    t0 = time.time()
    for text, gold in examples:
        predicted = pipeline.extract(text)
        for mode, strict in (("strict", True), ("lenient", False)):
            for label, c in count_matches(predicted, gold, strict).items():
                for key in c:
                    totals[mode][label][key] += c[key]
        exact += anonymize(text, predicted) == anonymize(text, gold)
    ms = (time.time() - t0) / len(examples) * 1000

    rows = []
    for label in TARGET_LABELS:
        s, l = totals["strict"][label], totals["lenient"][label]
        support = s["tp"] + s["fn"]  # number of gold spans with this label
        sp, sr, sf = precision_recall_f1(**s)
        lp, lr, lf = precision_recall_f1(**l)
        rows.append(
            {"label": label, "support": support, "P_strict": sp, "R_strict": sr, "F1_strict": sf,
             "P_lenient": lp, "R_lenient": lr, "F1_lenient": lf}
        )
    per_label = pd.DataFrame(rows).set_index("label")

    overall = {}
    for mode in ("strict", "lenient"):
        pooled = {k: sum(c[k] for c in totals[mode].values()) for k in ("tp", "fp", "fn")}
        overall[mode] = precision_recall_f1(**pooled)

    return {"per_label": per_label, "overall": overall, "exact": exact, "n": len(examples), "ms": ms}


def run_benchmark(model_keys: list[str], data_path: Path = DATA_PATH) -> dict[str, dict]:
    """Evaluate every model (+ regex) on one dataset. Returns {model key: scores}."""
    examples = load_gold(data_path)
    return {key: evaluate(build_pipeline(key), examples) for key in model_keys}


# ---------------------------------------------------------------------------
# Report: turn the scores into a markdown file a human can read
# ---------------------------------------------------------------------------

LABEL_NOISE_NOTES = """
## Notes on the dataset (label noise)

The 10 official examples were labelled by hand and are not perfectly consistent.
This matters when reading the strict scores:

- **Trailing period is inconsistent.** Example 1 tags `Apple Inc` (label `<ORG>.`),
  example 7 tags `Apple Inc.` (label `<ORG> He`). A model is "wrong" on one of them
  whatever it does. The lenient score forgives this.
- **Debatable labels.** The area code `212` is tagged LOCATION; `US` in "naturalized
  US citizen" is tagged LOCATION.
- **Missing labels.** `The Eiffel Tower` in example 5 is not tagged at all, so a model
  that finds it is punished with a false positive.
- **Missing types.** IBAN and SSN never appear in the official examples. The extra
  dataset (`data/extra.csv`, 6 sentences written by us) covers them, so those columns
  are only meaningful there.
- **Tiny dataset.** With 10 sentences, one entity more or less moves a per-type score by
  a lot. Read the numbers as indications, not precise measurements.
"""


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def write_report(results_by_dataset: dict[str, dict[str, dict]], coverage: dict[str, set[str]], path: Path) -> None:
    """
    results_by_dataset: {"benchmark.csv": {model key: scores}, "extra.csv": {...}}
    coverage: {model key: set of labels the model can produce at all}
    """
    lines = ["# Benchmark report", ""]
    lines += [
        "Every model is combined with the same regex layer (email, phone, URL, IBAN, SSN,",
        "amount, ISO date), so the comparison is about what the models themselves add.",
        "",
        "The `llm` row is a small local language model (qwen2.5:3b-instruct through Ollama)",
        "asked to answer in a fixed JSON shape. It is optional: it only appears when Ollama",
        "is running with that model pulled, so a fresh clone can run the benchmark without it.",
        "",
        "**How to read the scores.** Precision = of what was flagged, how much was right.",
        "Recall = of what should have been flagged, how much was found. F1 balances both.",
        "*Strict* requires the exact same start and end as the answer key; *lenient* only",
        "requires overlap with the right label. *Exact match* counts whole sentences whose",
        "anonymized text is identical to the answer key (a very harsh secondary metric).",
        "",
    ]

    for dataset, results in results_by_dataset.items():
        keys = list(results)
        n = next(iter(results.values()))["n"]
        lines += [f"## Dataset: `{dataset}` ({n} sentences)", "", "### Overall", ""]
        lines += ["| model | strict P | strict R | strict F1 | lenient P | lenient R | lenient F1 | exact match | ms/sentence |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, r in results.items():
            sp, sr, sf = r["overall"]["strict"]
            lp, lr, lf = r["overall"]["lenient"]
            lines.append(
                f"| {key}+regex | {_pct(sp)} | {_pct(sr)} | {_pct(sf)} | {_pct(lp)} | {_pct(lr)} | {_pct(lf)} "
                f"| {r['exact']}/{r['n']} | {r['ms']:.0f} |"
            )
        lines += ["", "### F1 per entity type (strict / lenient)", ""]
        lines += ["| label | support | " + " | ".join(f"{k}+regex" for k in keys) + " |",
                  "|---|---|" + "---|" * len(keys)]
        any_table = next(iter(results.values()))["per_label"]
        for label in TARGET_LABELS:
            support = int(any_table.loc[label, "support"])
            cells = []
            for key in keys:
                row = results[key]["per_label"].loc[label]
                cells.append("n/a" if support == 0 else f"{_pct(row.F1_strict)} / {_pct(row.F1_lenient)}")
            lines.append(f"| {label} | {support} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Coverage: which types can each model produce at all?", "",
              "From the label maps. A dash means the model can never output that type. Types marked",
              "*(regex)* are always covered by the regex layer regardless of the model.", ""]
    keys = list(coverage)
    regex_types = {"EMAIL_ADDRESS", "PHONE_NUMBER", "URL", "IBAN", "SSN", "AMOUNT", "DATE_TIME"}
    lines += ["| label | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for label in TARGET_LABELS:
        name = f"{label} *(regex)*" if label in regex_types else label
        lines.append(f"| {name} | " + " | ".join("yes" if label in coverage[k] else "-" for k in keys) + " |")
    lines.append("")

    lines.append(LABEL_NOISE_NOTES.strip())
    lines.append("")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))
