# Evaluation protocol 1.4 declaration

Protocol 1.4 was declared on 2026-09-28, before any protocol 1.4 candidate was
trained. It changes how **training** windows are drawn. Evaluation data,
partitions, probes, and all seven gates are identical to protocol 1.3.
Tracked in issue #5.

This is representation evaluation only. It provides no clinical, diagnostic,
treatment, dosing, monitoring, alerting, or safety evidence.

## Why

Protocol 1.3 aligned every day to midnight. That halved source leakage (best
margin 0.193 to 0.099) but cost hidden-window utility: on the same Colas
validation days, error rose from 6.59 to 8.01 mg/dL, and giving the baseline
the source label did not help it. The leading hypothesis is that the varied
BIG IDEAs start times had acted as phase augmentation. Protocol 1.4 restores
that variety for **both** sources, so start time cannot identify either one.

## Training windows

`glucotrace pretrain --random-phase-raw` loads the continuous raw recordings of
the protocol 1.3 **training** participants only (159 participants, 177
continuous segments) and cuts each 24-hour window at a reading chosen at
random, freshly every epoch, with the same procedure for both sources. Draws
per epoch equal the canonical sampler's (562), alternating sources. A
pre-training check drew 800 windows: both sources covered all 24 start hours
(4 to 6% per hour), and no validation or test participant was present.

Validation and test use the midnight-aligned protocol 1.3 files unchanged.

## Frozen partitions

Identical to [EVALUATION_1_3.md](EVALUATION_1_3.md) and verifiable with
`glucotrace verify-protocol 1.3`. The protocol 1.3 test partition was never
opened (no 1.3 candidate passed validation), so it is reused as the protocol
1.4 holdout. Every candidate starts from random weights.

## Candidates

Every recipe uses the candidate 6 settings (40 epochs, batch size 32,
learning rate `3e-4`, weight decay `0.05`) and is trained with **seeds 7, 11,
and 13**: nine models in total, the full budget.

| Recipe | Random-phase windows | Within-source negatives | Clock input |
|---|:-:|:-:|:-:|
| A | yes | no | yes |
| B | yes | yes | yes |
| C | yes | no | **no** (time of day zeroed; a diagnostic) |

## Gates

The seven protocol 1.2/1.3 gates, unchanged: feature dispersion, effective
rank at least `8.0`, three missing-data stability checks, source margin at most
`0.10`, and hidden-window MAE no worse than the summary baseline.

## Pass rule and selection

- A recipe passes validation when the **mean over its three seeds** of each
  gated metric meets that gate. Per-seed results are always reported.
- Among passing recipes, choose the lowest mean validation source margin;
  ties within `0.01` go to the lower mean hidden-window MAE.
- The test partition is opened **once**, for the selected recipe only. All
  three seeds are evaluated and the recipe passes if the three-seed mean meets
  every gate. That result is final.
- If it passes, the seed with the median validation source margin is
  calibrated on validation and released as the v0.3 checkpoint.

## Stopping rule

If no recipe passes validation, the test partition stays closed, the result is
published, and **model tuning for source leakage pauses**. The problem stays
open as issue #5.

## Review note

The project owner reviews the validation results before the test partition is
opened. The rules above stand as declared. Any change made after seeing
validation results will be recorded here as a dated amendment, with its
reason.
