# Encoding, comparison, and search

The inference tools use `checkpoints/glucofm-research.pt`. Every output is
research-only and descriptive. Similarity is not a diagnostic score and must not
be used for treatment, dosing, monitoring, alerts, or patient care.

## Input contract

CSV input requires an ISO-8601 `timestamp` column and an mg/dL `glucose` column.
Empty glucose cells are missing. The loader builds a five-minute grid without
interpolation, and the physical observation mask remains authoritative.

One fingerprint represents one complete 288-position window. If a CSV contains
multiple complete non-overlapping windows, pass `--window-index`. Partial final
windows are not padded or encoded.

## Fingerprint definition

The encoder first produces its raw 128-value pooled embedding. Each coordinate
is then centered and scaled using statistics fitted only on the declared
protocol 1.1 validation partition, followed by L2 normalization. The resulting
128-number fingerprint has unit length. Calibration values and their provenance
are embedded in the checkpoint.

This calibration matches the similarity space used by the frozen evaluation.
It is not clinical calibration and does not convert the output into a
probability.

## Encode

```bash
glucofm-encode day.csv \
  --checkpoint checkpoints/glucofm-research.pt \
  --output day.fingerprint.json
```

The output records the checkpoint and input checksums, selected window,
observation fraction, calibration method, and 128-number fingerprint.

For a multi-day file:

```bash
glucofm-encode several-days.csv --window-index 1
```

## Compare

```bash
glucofm-compare first-day.csv second-day.csv \
  --checkpoint checkpoints/glucofm-research.pt \
  --output comparison.json
```

The result contains cosine similarity in `[-1, 1]` and cosine distance
`1 - similarity`. The project defines no cutoff for declaring days equivalent,
normal, abnormal, or clinically related.

## Search

```bash
glucofm-search query-day.csv \
  --checkpoint checkpoints/glucofm-research.pt \
  --manifest data/processed/big_ideas/manifest.json \
  --manifest data/processed/colas/manifest.json \
  --top-k 5 \
  --output nearest-days.json
```

Search verifies every canonical file checksum, embeds candidates in batches,
and ranks them by cosine similarity. A byte-identical query file is excluded by
default; pass `--include-self` to retain it. Results include declared dataset,
participant identifier, start time, canonical path, checksum, and observed
fraction.

Search output may expose identifiers and filenames from a private manifest.
Researchers are responsible for access control and appropriate handling.

## Python API

```python
from glucofm import ResearchEncoder, cosine_similarity

encoder = ResearchEncoder.load("checkpoints/glucofm-research.pt")
first, first_metadata = encoder.encode_csv("first-day.csv")
second, second_metadata = encoder.encode_csv("second-day.csv")
similarity = cosine_similarity(first, second)
```

`ResearchEncoder.encode_tensors` accepts glucose, physical observation masks,
optional gap age, and optional circular time-of-day tensors for integration into
other research code.

## Known limitation

The released checkpoint passed its declared artificial-missingness checks, but
its embeddings strongly predict which of the two training datasets supplied a
day. Similar results can therefore reflect dataset, population, protocol,
geography, collection-era, or device artifacts rather than shared physiology.
