# Protocol 1.5 results: leak-free on new people, but not more useful than summary statistics

Protocol 1.5 ([EVALUATION_1_5.md](EVALUATION_1_5.md)) added a third cohort
(ShanghaiT1DM/T2DM, China) and tested three clock designs, each with seeds 7,
11, and 13. Recipe C (no clock) was the only recipe to pass all seven gates on
validation, so the test partition was opened once for it. **On the test
partition it failed the utility gate.** Under the protocol that result is
final and the checkpoint would not be released. The owner later released it
anyway as the 0.3 default, with the failure disclosed (see Decision below).

This is representation research only. None of these numbers are clinical,
diagnostic, or safety evidence.

## Validation (three-seed mean)

Summary baseline hidden-window MAE: `15.96` mg/dL.

| Recipe | Clock | Source margin (≤ 0.10) | Hidden-window MAE | Effective rank (≥ 8) | Same-person top-1 | Eligible |
|---|---|---|---|---|---|---|
| X (diagnostic) | exact | 0.063 | 16.05 | 6.84 | 0.428 | not a candidate |
| D | four day-part bins | 0.093 | **15.40** | 6.51 | 0.425 | no (rank) |
| C | none | **0.030** | 15.64 | **15.27** | 0.490 | **yes** |

Giving the model any clock narrowed the embedding below the effective-rank
minimum. Recipe D predicted best but could not be selected.

## Test (recipe C, opened once; 298 days from 49 participants)

| Gate | Test mean | Validation mean | Result |
|---|---|---|---|
| Source margin (≤ 0.10) | 0.062 | 0.030 | pass |
| Model minus baseline MAE (≤ 0) | **+0.452** | −0.322 | **fail** |
| Effective rank (≥ 8) | 16.86 | 15.27 | pass |
| Feature dispersion (≥ 0.05) | 0.978 | 0.944 | pass |
| Missing-data stability (9 criteria) | all pass | all pass | pass |

Per seed, the utility difference was +0.92, −0.06, and +0.50 mg/dL. Every
seed's 95% participant-bootstrap interval includes zero, so on new people the
embedding is statistically indistinguishable from eleven summary statistics,
not better than them.

| Test source | Days | Model MAE | Baseline MAE |
|---|---|---|---|
| BIG IDEAs (USA) | 14 | 15.54 | 14.55 |
| Colas (Spain) | 55 | 9.22 | 8.51 |
| Shanghai (China) | 229 | 22.93 | 22.58 |

## What we learned

1. **The source leak is solved by removing the clock**, and it stays solved on
   new people (margin 0.062, limit 0.10).
2. **The utility advantage did not generalize.** The validation gain came
   mostly from Shanghai and did not hold on new patients. Honestly stated: the
   fingerprint currently carries about as much predictive information as
   simple summary statistics.
3. **Fingerprints are personal.** A day's nearest neighbour comes from the
   same person 48% of the time on test (summary statistics: 32%). Treat
   fingerprints as personal data.
4. **A CGM-only meal detector was explored and dropped.** Tuned on training
   patients, it found 44% of logged Shanghai validation meals (75% of logged
   meals show a clear rise), so meal-relative time was not pursued.

## Decision

Research on this question stops here, as the owner decided. The project now
focuses on the fingerprint tool itself.

**Amendment, 2026-10-05, after the test result.** The owner decided to
release recipe C seed 13 (the median validation-margin seed) as the 0.3
default model despite the failed utility gate, judging the source-bias fix
more important than a utility gap inside its uncertainty interval. This
overrides the protocol's rule that a failed test is final. Complete reports:
`evaluations/protocol-1.5-recipe-*-validation.json` and
`evaluations/protocol-1.5-recipe-C-seed-*-test.json`.
