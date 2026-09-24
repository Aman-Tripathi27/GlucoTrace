# Protocol 1.3 results: the bias fix worked, with a trade-off

Protocol 1.3 ([EVALUATION_1_3.md](EVALUATION_1_3.md)) tested one change:
starting every canonical day just after local midnight, so that window start
time no longer identifies the source dataset. Gates were unchanged from
protocol 1.2. Three candidates, the full declared budget, were trained from
random weights and measured on validation only.

**No candidate passed all seven checks**, so the test partition stays closed
and the released checkpoint is unchanged. The diagnosis from protocol 1.2 was
confirmed, however: aligning days to midnight halved the source leakage.

This is representation research only. None of these numbers are clinical,
diagnostic, or safety evidence.

## Validation results

Summary baseline: linear source probe `0.744` balanced accuracy,
hidden-window MAE `7.78` mg/dL.

| Candidate | Recipe | Source probe | Source margin (≤ 0.10) | Hidden-window MAE | Effective rank (≥ 8.0) | Failed checks |
|---|---|---|---|---|---|---|
| g | plain | 0.850 | 0.106 | 8.71 | 9.10 | source, utility |
| h | within-source negatives + adversary 0.1 | 0.852 | 0.108 | 9.08 | 7.78 | source, utility, rank |
| i | within-source negatives | 0.843 | **0.099** | 8.91 | 7.99 | utility, rank |

Complete reports: `evaluations/protocol-1.3-candidate-*-validation.json`.

## Compared with protocol 1.2

| | Protocol 1.2 (mixed start times) | Protocol 1.3 (midnight) |
|---|---|---|
| Source margin, best candidate | 0.193 | **0.099** |
| Source margin, range | 0.193 to 0.233 | 0.099 to 0.108 |
| Hidden-window MAE vs baseline | 4 to 13% **better** | 12 to 17% **worse** |

## What we learned

1. **The diagnosis was right.** Window start time was the main source
   shortcut. Removing it cut the source margin roughly in half, with no
   adversarial training needed (candidate g). All three candidates are
   within 0.01 of the gate.
2. **Usefulness dropped, and not for the reason we expected.** A
   validation-only breakdown shows that on the 55 Colas validation days,
   which are nearly identical in both protocols, the hidden-window error rose
   from 6.59 (protocol 1.2 candidate a) to 8.01 mg/dL (candidate i), while the
   summary baseline stayed at 7.7 to 7.9. Giving the baseline the dataset
   label did not improve it (7.92 and 7.63). So the earlier advantage did
   **not** come from knowing the source.
3. **Likely explanation, not yet tested:** the varied BIG IDEAs start times
   were acting as accidental phase augmentation, showing the model days at
   many circadian offsets. Other contributors may be fewer BIG IDEAs training
   days (67, down from 80) and single-seed variance. The BIG IDEAs validation
   subset (13 days, 2 participants) is too small to read on its own.

## Next step (a future protocol)

Keep evaluation days aligned to midnight, but train on windows with
**random start times drawn equally from both sources**. That keeps the
diversity without letting start time identify the source. Run several seeds
per recipe so variance is visible.
