# Evaluation protocol 1.5 declaration

Protocol 1.5 was declared on 2026-09-28, before any protocol 1.5 candidate was
trained. It adds a third dataset from a third country and makes the evaluation
more rigorous. Tracked in issue #5.

This is representation evaluation only. It provides no clinical, diagnostic,
treatment, dosing, monitoring, alerting, or safety evidence.

## Why

Protocol 1.4 showed that the time-of-day input carries the source leak:
without it, leakage disappeared (margin −0.004 ± 0.021) at a ~3% utility cost.
The leading explanation is that the two cohorts come from countries with
different meal times. Protocol 1.5 adds a cohort from China, tests a coarse
day-part clock against no clock, and adds per-source results, confidence
intervals, and leave-one-dataset-out transfer.

## Data

| Source | Country | Device (cadence) | License |
|---|---|---|---|
| BIG IDEAs 1.1.3 | USA | Dexcom G6 (5 min) | ODC-By 1.0 |
| Colas et al. 2019 | Spain | Medtronic iPro (5 min) | CC BY 4.0 |
| ShanghaiT1DM/T2DM, figshare v5 (new) | China | Abbott FreeStyle Libre H (15 min) | CC BY 4.0 |

`glucotrace prepare-shanghai` (module `glucotrace.adapters.shanghai`) reads the
official ZIP (SHA-256
`59b5f5c4053a32bb6b7827844a0191597dc82228fe88ce332a189fdfc659c4cb`), merges each
patient's recording files, and builds midnight-aligned days on the canonical
five-minute grid. The two grid positions between 15-minute readings are marked
missing, never interpolated, and the observed-day threshold is 80% of the
sensor's own 96 daily readings. The T1DM and T2DM subsets are separate
manifests sharing the source label `Shanghai`: 149 and 1,074 days from 12 and
100 patients.

## Frozen partitions

BIG IDEAs and Colas keep their protocol 1.3 partitions; that test partition
has never been opened. Shanghai is split by patient with
`glucotrace split --seed 71` (70/15/15), separately per subset. Verify with
`glucotrace verify-protocol 1.5`.

| Partition | Participants | Days | BIG IDEAs | Colas | Shanghai T1DM | Shanghai T2DM |
|---|---|---|---|---|---|---|
| training | 237 | 1,190 | 12 / 67 | 147 / 281 | 8 / 90 | 70 / 752 |
| validation | 48 | 220 | 2 / 13 | 30 / 55 | 1 / 5 | 15 / 147 |
| test | 49 | 298 | 2 / 14 | 29 / 55 | 3 / 54 | 15 / 175 |

The Shanghai T1DM validation set holds one patient, so no conclusion will be
drawn about T1DM alone.

## Training

All candidates start from random weights with 40 epochs, batch size 32,
learning rate `3e-4`, weight decay `0.05`, and seeds **7, 11, and 13**.

- **Random-phase windows** for all three sources (as in protocol 1.4), drawn
  from training participants' raw recordings only. Draws per epoch alternate
  sources and equal three times the largest source's training days.
- **Cadence dropout 0.5:** each training day is thinned to 15-minute cadence
  with probability 0.5 and a random phase, so sampling cadence cannot become
  a learned source signature. Native 15-minute days are unchanged.

| Recipe | Clock input |
|---|---|
| C | none (the protocol 1.4 leak-free design) |
| D | coarse: four six-hour bins (night, morning, afternoon, evening) |

## Gates (pass on the three-seed mean)

The seven protocol 1.2 gates, with one declared change:

- **Gate 6 is cadence-harmonized.** Before both source probes are fitted, every
  training and target day is thinned to 15-minute cadence with a seeded random
  phase, so cadence cannot reveal the source. The limit is unchanged: model
  minus summary-baseline balanced accuracy of at most `0.10`. The
  unharmonized probe is still reported.

Two evaluation fixes are required by 15-minute data and change nothing for
5-minute data: the cadence stress test leaves days already at 15-minute
cadence unchanged, and hidden-window eligibility counts expected readings at
each day's own cadence.

## Reported, not gated

- Per-source hidden-window MAE (model and baseline) and same-person retrieval.
- 95% participant-level bootstrap intervals (1,000 resamples) for the
  harmonized source margin and the model-minus-baseline MAE.
- **Leave-one-dataset-out transfer:** recipe C, seeds 7, 11, and 13, trained
  on two sources and scored on the third source's validation partition, for
  all three folds (nine models).
- Descriptive meal timing from the Shanghai meal logs.

## Selection, test, and stopping rules

- Only recipes passing all seven gates on their three-seed mean are eligible;
  choose the lowest mean harmonized margin, ties within `0.01` going to the
  lower mean MAE.
- The owner reviews validation results first. The test partition is then
  opened once, for the selected recipe's three seeds; it passes if the mean
  meets every gate. That result is final. If it passes, the median-margin seed
  is calibrated on validation and released.
- If no recipe passes, the test partition stays closed and the result is
  published. Rules changed after seeing results are recorded here as dated
  amendments.

## Disclosure

Before this declaration, one-epoch smoke tests confirmed the pipeline runs.
Only completion, runtime, and row counts were inspected, and no metric was
read. The smoke checkpoints are not candidates.

## Amendments

**2026-09-29, before any protocol 1.5 candidate completed training.** An
earlier launch of the candidates stopped without writing any output (four
empty log files, no checkpoints); nothing was trained or measured. The owner
chose to run **recipe C only** (seeds 7, 11, 13) first, to isolate the effect
of adding the third cohort. Recipe D and the leave-one-dataset-out transfer
runs are deferred, not cancelled. Until they are run, only recipe C can be
selected, so any test opening would concern recipe C alone. No gate,
threshold, partition, or training setting changes.

**2026-09-29, after recipe C's validation results and before recipe D or any
diagnostic was trained.** The owner reviewed recipe C's validation results
(all seven gates pass on the three-seed mean) and chose to run recipe D
before opening the test partition, as the selection rule allows. In addition,
a **diagnostic recipe X** (exact clock; otherwise identical to C and D) is
trained on seeds 7, 11, and 13 to show what the clock contributes. Recipe X is
**not a candidate**: it cannot be selected, and the test partition will never
be opened for it. Recipe D and X runs are trained one at a time to limit
laptop heat; this does not change any training setting.

**2026-10-05, before the test partition was opened.** Recipe D failed the
effective-rank gate on its three-seed mean (6.51; minimum 8.0) and is not
eligible. Diagnostic recipe X is not a candidate. Recipe C passed all seven
gates on its three-seed mean and is the only eligible recipe, so it is
selected. Between these runs, a CGM-only meal detector was explored on
training and validation patients only (no model was trained with it); it
found 44% of logged Shanghai validation meals and was dropped. The owner
approved opening the test partition once, for recipe C seeds 7, 11, and 13.
It passes if the three-seed mean meets every gate; the result is final. If
it passes, the median-margin seed is calibrated on validation and released.
