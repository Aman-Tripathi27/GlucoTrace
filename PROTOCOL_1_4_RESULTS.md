# Protocol 1.4 results: the clock input carries the source leak

Protocol 1.4 ([EVALUATION_1_4.md](EVALUATION_1_4.md)) trained three recipes
with three seeds each (nine models, the full budget) on random-phase training
windows, and measured them on validation only. **No recipe passed all seven
gates on its three-seed mean**, so the test partition stays closed and the
released checkpoint is unchanged. Per the declared stopping rule, tuning on
this two-source corpus pauses; the next step adds new data (protocol 1.5).

This is representation research only. None of these numbers are clinical,
diagnostic, or safety evidence.

## Results (validation, mean ± SD over seeds 7, 11, 13)

Summary baseline: linear source probe `0.744`, hidden-window MAE `7.78` mg/dL.

| Recipe | Source margin (≤ 0.10) | Hidden-window MAE | Effective rank (≥ 8.0) | Same-person top-1 | Failed gates |
|---|---|---|---|---|---|
| A random phase | 0.103 ± 0.009 | 8.48 ± 0.29 | 4.36 ± 0.27 | 0.169 | source, utility, rank |
| B A + within-source negatives | 0.103 ± 0.015 | 8.22 ± 0.13 | 3.53 ± 0.24 | 0.143 | source, utility, rank |
| C A + **no clock** | **−0.004 ± 0.021** | 8.01 ± 0.17 | **8.56 ± 0.37** | **0.238** | utility only |

All recipes passed the three missing-data stability checks (median cosine
0.948 to 0.998). Per-seed reports:
`evaluations/protocol-1.4-recipe-*-seed-*-validation.json`.

## What we learned

1. **The time-of-day input is where the source leak lives.** Without it
   (recipe C), the embedding reveals no more about the source dataset than
   eleven summary statistics, consistently across seeds. Every earlier model
   had a margin of 0.10 to 0.23.
2. **The phase-augmentation hypothesis from protocol 1.3 was wrong.** Random
   start times with the clock (A, B) kept the leakage and collapsed the
   embedding.
3. **Removing the clock costs a little utility:** 8.01 vs 7.78 mg/dL, about 3%
   worse (best seed 7.84).
4. **Plausible, untested explanation:** the two cohorts come from countries
   with different meal times (Spain and the USA), so exact clock time
   identifies the cohort. Protocol 1.5 adds a third country and meal logs to
   test this, and compares no clock with a coarse day/night clock.
