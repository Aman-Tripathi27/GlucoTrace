# GlucoTrace 0.3 checkpoint

This report describes `checkpoints/glucotrace-0.3.pt`, the default model from
version 0.3.0. It is an experimental representation model, not a medical
device or a clinically evaluated model. The 0.2 checkpoint is described in
[RELEASE_RESULTS.md](RELEASE_RESULTS.md) and remains available.

## Disclosure: released despite failing one pre-registered gate

On the protocol 1.5 sealed test, this model **failed the utility gate**: its
hidden-window error was 20.10 mg/dL against 19.60 for eleven summary
statistics (+0.50; 95% participant-bootstrap interval −1.36 to +2.12). The
protocol's rule was that a failed test is final and the model is not
released. The owner decided on 2026-10-05, after seeing that result, to
release it anyway: it removes the source-dataset bias of the 0.2 model, and
the utility gap lies within the uncertainty interval. This decision overrides
the protocol and is recorded here and in
[PROTOCOL_1_5_RESULTS.md](PROTOCOL_1_5_RESULTS.md). Read every utility claim
for this model as "statistically tied with summary statistics", not "better".

## Artifact

- checkpoint SHA-256:
  `a4ee0f8f15ab6e2d8a9f311acf5884594889300a70df35bfa599e45605845389`
- protocol 1.5, recipe C, seed 13 (the median validation source margin of the
  three seeds, as the protocol specifies for release)
- **no clock input**: the time-of-day feature is zeroed, so the model sees
  only the shape of the glucose day
- trainable encoder parameters: 511,372; explicit marker `research_only=true`
- fingerprint calibration: validation z-score (220 days) then L2 normalization

**Fingerprints from 0.3 are not comparable with fingerprints from 0.2.** Each
output records the checkpoint SHA-256, so mixed comparisons can be detected.

## Data

Participant-disjoint training partitions of three public cohorts:

| Cohort | Country | Device | Training days |
|---|---|---|---|
| BIG IDEAs 1.1.3 | USA | Dexcom G6, 5 min | 67 |
| Colas et al. 2019 | Spain | Medtronic iPro, 5 min | 281 |
| ShanghaiT1DM/T2DM | China | FreeStyle Libre H, 15 min | 842 |

Training used 24-hour windows starting at random readings, source-balanced
sampling, and 15-minute cadence dropout (probability 0.5). See
[EVALUATION_1_5.md](EVALUATION_1_5.md).

## Sealed-test results (this seed; 298 days from 49 new participants)

| Check | Result | Limit |
|---|---|---|
| Source margin (cadence-harmonized) | 0.067 | ≤ 0.10, pass |
| Hidden-window MAE minus baseline | +0.50 mg/dL | ≤ 0, **fail** |
| Effective rank | 15.36 | ≥ 8, pass |
| Random 30% removal, median cosine | 0.965 | ≥ 0.80, pass |
| Same-person top-1 retrieval | 0.475 (baseline 0.322) | reported |

The three-seed means are in [PROTOCOL_1_5_RESULTS.md](PROTOCOL_1_5_RESULTS.md).

## What this model is good for, and what it is not

- **Good for:** comparing and searching CGM days by the shape of the glucose
  curve, without the fingerprint encoding which study or country a day came
  from.
- **Not shown:** that the fingerprint carries more predictive information than
  simple summary statistics. On new people it is tied.
- **Blind to clock time:** an after-breakfast and an after-dinner rise of the
  same shape look alike to this model.
- **Personal data:** a day's nearest neighbour is the same person about half
  the time.
- Not clinically validated, not externally validated beyond these three
  cohorts, and no subgroup or fairness analysis.
