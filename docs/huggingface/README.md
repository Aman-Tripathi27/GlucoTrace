---
license: mit
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - continuous-glucose-monitoring
  - cgm
  - time-series
  - representation-learning
  - self-supervised
  - research-only
---

# GlucoTrace research checkpoint

A 511k-parameter Transformer that encodes one 24-hour continuous glucose
monitor (CGM) day into a 128-number fingerprint for similarity search and
representation research.

> **Research use only.** Not a medical device. Do not use for diagnosis,
> treatment, dosing, alerts, or patient care.

## Use

```bash
pip install git+https://github.com/Aman-Tripathi27/GlucoTrace
glucotrace download-model
glucotrace encode day.csv
glucotrace report day.csv --manifest corpus/manifest.json --output report.html
```

Input: a CSV with `timestamp,glucose` in mg/dL (or pass `--unit mmol/L`) on a
five-minute grid. Missing readings stay missing; nothing is interpolated.

## Training data

Participant-disjoint training partitions of two public datasets:
BIG IDEAs Glycemic Wearable (PhysioNet 1.1.3, ODC-By 1.0) and the Colas et al.
2019 PLOS ONE supporting dataset. 363 training days from 159 participants.

## Evaluation (all pre-registered)

- Protocol 1.1 held-out test (69 days, 31 participants): passed all five
  checks for non-collapse and stability under missing data (median cosine
  0.994 after removing 30% of readings).
- Validation: predicts the mean glucose of a hidden six-hour window with a
  lower error than a summary-statistics baseline (7.71 vs 8.22 mg/dL).

## Known limitations

- The embedding strongly encodes the source dataset (linear probe about 0.98
  balanced accuracy). This is likely caused by the datasets aligning days to
  different clock times.
- Fingerprints can link days from the same person. Treat them as personal
  data.
- Two datasets, no external validation, and no subgroup or fairness analysis.

Full details: the project's `MODEL_CARD.md`, `RELEASE_RESULTS.md`, and
`PROTOCOL_1_2_RESULTS.md` on GitHub (Aman-Tripathi27/GlucoTrace).

SHA-256: `1fbeecd68d81d239fa26726b67ca2d05bf485569a8d3619cdc624f86bba8092b`
