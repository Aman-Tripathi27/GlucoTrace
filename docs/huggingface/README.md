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

# GlucoTrace 0.3 research checkpoint

A 511k-parameter Transformer that encodes one 24-hour continuous glucose
monitor (CGM) day into a 128-number fingerprint for similarity search and
representation research.

> **Research use only.** Not a medical device. Do not use for diagnosis,
> treatment, dosing, alerts, or patient care.

## Use

```bash
pip install glucotrace
glucotrace download-model
glucotrace encode day.csv                         # plain timestamp,glucose CSV
glucotrace encode export.csv --format libreview   # FreeStyle Libre export
glucotrace encode export.csv --format dexcom-clarity
```

Missing readings stay missing; nothing is interpolated.

## What changed in 0.3

The model has **no clock input**. The 0.2 model could tell which study (and
country) a day came from by its meal times; this one sees only the shape of
the glucose day. Fingerprints are not comparable with 0.2 fingerprints.

## Training data

Participant-disjoint training partitions of BIG IDEAs (USA, Dexcom G6),
Colas et al. 2019 (Spain, Medtronic iPro), and ShanghaiT1DM/T2DM (China,
FreeStyle Libre H): 1,190 training days from 237 participants.

## Evaluation (pre-registered, protocol 1.5; sealed test, 49 new people)

- Source bias fixed: margin 0.067 against summary statistics (limit 0.10).
- Stable under missing data (median cosine 0.965 after removing 30%).
- **Utility: tied with summary statistics** (20.1 vs 19.6 mg/dL hidden-window
  error). This failed the pre-registered utility gate; the owner released the
  model anyway because of the bias fix. Disclosed in `RELEASE_RESULTS_0_3.md`.

## Known limitations

- Blind to clock time: an after-breakfast and an after-dinner rise of the same
  shape look alike.
- Fingerprints link days from the same person about half the time. Treat them
  as personal data.
- Three cohorts, no external validation, no subgroup or fairness analysis.

Full details on GitHub (Aman-Tripathi27/GlucoTrace): `MODEL_CARD.md`,
`RELEASE_RESULTS_0_3.md`, `PROTOCOL_1_5_RESULTS.md`.

SHA-256: `a4ee0f8f15ab6e2d8a9f311acf5884594889300a70df35bfa599e45605845389`
