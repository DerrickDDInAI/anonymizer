# Benchmark report

Every model is combined with the same regex layer (email, phone, URL, IBAN, SSN,
amount, ISO date), so the comparison is about what the models themselves add.

The `llm` row is a small local language model (qwen2.5:3b-instruct through Ollama)
asked to answer in a fixed JSON shape. It is optional: it only appears when Ollama
is running with that model pulled, so a fresh clone can run the benchmark without it.

**How to read the scores.** Precision = of what was flagged, how much was right.
Recall = of what should have been flagged, how much was found. F1 balances both.
*Strict* requires the exact same start and end as the answer key; *lenient* only
requires overlap with the right label. *Exact match* counts whole sentences whose
anonymized text is identical to the answer key (a very harsh secondary metric).

## Dataset: `benchmark.csv` (10 sentences)

### Overall

| model | strict P | strict R | strict F1 | lenient P | lenient R | lenient F1 | exact match | ms/sentence |
|---|---|---|---|---|---|---|---|---|
| spacy+regex | 89% | 80% | 85% | 91% | 82% | 87% | 3/10 | 7 |
| openmed+regex | 97% | 63% | 76% | 100% | 65% | 79% | 1/10 | 197 |
| gliner+regex | 73% | 69% | 71% | 90% | 84% | 87% | 1/10 | 88 |
| llm+regex | 68% | 55% | 61% | 85% | 69% | 76% | 3/10 | 4138 |

### F1 per entity type (strict / lenient)

| label | support | spacy+regex | openmed+regex | gliner+regex | llm+regex |
|---|---|---|---|---|---|
| PERSON | 7 | 100% / 100% | 100% / 100% | 100% / 100% | 92% / 92% |
| ORG | 8 | 62% / 75% | 0% / 0% | 67% / 89% | 59% / 71% |
| JOB | 2 | 0% / 0% | 100% / 100% | 80% / 80% | 50% / 100% |
| EMAIL_ADDRESS | 1 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| LOCATION | 20 | 89% / 89% | 89% / 95% | 36% / 73% | 39% / 71% |
| AMOUNT | 2 | 100% / 100% | 100% / 100% | 80% / 80% | 80% / 80% |
| DATE_TIME | 8 | 88% / 88% | 22% / 22% | 100% / 100% | 77% / 77% |
| UNIVERSITY | 1 | 0% / 0% | 0% / 0% | 100% / 100% | 0% / 0% |
| PHONE_NUMBER | 1 | 100% / 100% | 100% / 100% | 100% / 100% | 67% / 67% |
| URL | 1 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| IBAN | 0 | n/a | n/a | n/a | n/a |
| SSN | 0 | n/a | n/a | n/a | n/a |

## Dataset: `extra.csv` (6 sentences)

### Overall

| model | strict P | strict R | strict F1 | lenient P | lenient R | lenient F1 | exact match | ms/sentence |
|---|---|---|---|---|---|---|---|---|
| spacy+regex | 81% | 81% | 81% | 81% | 81% | 81% | 2/6 | 12 |
| openmed+regex | 95% | 90% | 93% | 95% | 90% | 93% | 5/6 | 95 |
| gliner+regex | 95% | 95% | 95% | 95% | 95% | 95% | 5/6 | 171 |
| llm+regex | 95% | 86% | 90% | 95% | 86% | 90% | 4/6 | 5417 |

### F1 per entity type (strict / lenient)

| label | support | spacy+regex | openmed+regex | gliner+regex | llm+regex |
|---|---|---|---|---|---|
| PERSON | 3 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| ORG | 1 | 0% / 0% | 0% / 0% | 0% / 0% | 67% / 67% |
| JOB | 2 | 0% / 0% | 100% / 100% | 100% / 100% | 0% / 0% |
| EMAIL_ADDRESS | 1 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| LOCATION | 1 | 67% / 67% | 67% / 67% | 100% / 100% | 100% / 100% |
| AMOUNT | 3 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| DATE_TIME | 2 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| UNIVERSITY | 1 | 0% / 0% | 0% / 0% | 67% / 67% | 0% / 0% |
| PHONE_NUMBER | 2 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| URL | 1 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| IBAN | 2 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| SSN | 2 | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |

## Coverage: which types can each model produce at all?

From the label maps. A dash means the model can never output that type. Types marked
*(regex)* are always covered by the regex layer regardless of the model.

| label | spacy | openmed | gliner | llm |
|---|---|---|---|---|
| PERSON | yes | yes | yes | yes |
| ORG | yes | yes | yes | yes |
| JOB | - | yes | yes | yes |
| EMAIL_ADDRESS *(regex)* | - | yes | - | yes |
| LOCATION | yes | yes | yes | yes |
| AMOUNT *(regex)* | yes | - | yes | yes |
| DATE_TIME *(regex)* | yes | yes | yes | yes |
| UNIVERSITY | - | - | yes | yes |
| PHONE_NUMBER *(regex)* | - | yes | - | yes |
| URL *(regex)* | - | yes | - | yes |
| IBAN *(regex)* | - | - | - | yes |
| SSN *(regex)* | - | yes | - | yes |

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
