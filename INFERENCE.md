# Encoding, comparison, and search

The inference tools use `checkpoints/glucotrace-research.pt`. Every output is
research-only and descriptive. Similarity is not a diagnostic score and must not
be used for treatment, dosing, monitoring, alerts, or patient care.

## Input contract

CSV input requires an ISO-8601 `timestamp` column and a `glucose` column in
mg/dL. For mmol/L files, pass `--unit mmol/L` (Python: `unit="mmol/L"`). Values
are then multiplied by 18.016 and the fingerprint metadata records
`input_unit`. The loader rejects a file whose median contradicts the declared
unit (below 35 for mg/dL, 35 or above for mmol/L); it never guesses. Empty
glucose cells are missing. The loader builds a five-minute grid without
interpolation, and the physical observation mask remains authoritative.

**Time zones.** Timestamps may carry an offset (`2026-01-01T08:00:00+05:30`)
or none; a file must not mix the two. Offsets are used to place readings by
true elapsed time, and time-of-day features use each reading's own local
clock, so 08:00 in India is morning to the model. Offsets may change within a
file, for example across a daylight-saving change.

**Dexcom Clarity exports.** Pass `--format dexcom-clarity` to read a Clarity
CSV export directly. Only estimated glucose value (`EGV`) rows are used, and the
unit is read from the export's own glucose column header.

**FreeStyle Libre (LibreView) exports.** Pass `--format libreview` to read a
LibreView CSV export directly. Only automatic historic readings (record type
`0`, usually every 15 minutes) are used; scans, strip tests, food, insulin, and
notes are skipped. The unit comes from the `Historic Glucose` column header,
and decimal commas (`5,6`) are accepted. LibreView writes dates as month-day
(`03-01-2024 08:05 AM`) or day-month (`01-03-2024 08:05`) depending on region.
The order is detected from the file; if every date could be read either way,
the file is rejected rather than guessed, and you pass `--date-order mdy` or
`--date-order dmy`. Readings 15 minutes apart sit on every third position of
the five-minute grid, with the positions between marked missing, never
interpolated. The released checkpoint was trained on five-minute data; its
fingerprints passed the protocol 1.1 stability check at 15-minute cadence
(median cosine 0.993), but expect somewhat more noise than with five-minute
data.

**Readings beyond the sensor range.** Dexcom records values below 40 mg/dL as
`Low` and above 400 mg/dL as `High`, with no number. By default these are
treated as **missing**, because the true value was never measured, and a
warning reports how many there were. `--out-of-range clamp` records them at
40 and 400 mg/dL instead, which keeps the shape of an extreme but assumes a
value. The fingerprint metadata records `out_of_range_readings` and
`out_of_range_policy`. The released checkpoint never saw such readings during
training. `Low`/`High` are also accepted in plain CSVs. FreeStyle Libre writes
`LO` (below 40) and `HI` (above 500 mg/dL); with `--out-of-range clamp` these
are recorded at 40 and 500 mg/dL.

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
glucotrace encode day.csv \
  --checkpoint checkpoints/glucotrace-research.pt \
  --output day.fingerprint.json
```

The output records the checkpoint and input checksums, selected window,
observation fraction, calibration method, and 128-number fingerprint.

For a multi-day file:

```bash
glucotrace encode several-days.csv --window-index 1
```

## Compare

```bash
glucotrace compare first-day.csv second-day.csv \
  --checkpoint checkpoints/glucotrace-research.pt \
  --output comparison.json
```

The result contains cosine similarity in `[-1, 1]` and cosine distance
`1 - similarity`. The project defines no cutoff for declaring days equivalent,
normal, abnormal, or clinically related.

## Search

```bash
glucotrace search query-day.csv \
  --checkpoint checkpoints/glucotrace-research.pt \
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

## Report

```bash
glucotrace report query-day.csv \
  --manifest data/processed/big_ideas/manifest.json \
  --manifest data/processed/colas/manifest.json \
  --top-k 5 --output report.html
```

This writes one self-contained HTML file: the query day overlaid with its
nearest days, a table of similarity, mean, standard deviation, share of
readings from 70 to 180 mg/dL, and coverage, and one comparison chart per
neighbor. It uses inline SVG and CSS only and loads no scripts or network
resources. Missing readings are drawn as gaps. The report contains the same
participant identifiers and file names as `search`, so share it only when the
manifests may be shared.

## Python API

```python
from glucotrace import ResearchEncoder, cosine_similarity

encoder = ResearchEncoder.load("checkpoints/glucotrace-research.pt")
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
