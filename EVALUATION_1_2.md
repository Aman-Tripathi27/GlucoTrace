# Evaluation protocol 1.2 declaration

Protocol 1.2 was declared on 2026-09-23, after the released candidate 6
checkpoint passed protocol 1.1 and before any protocol 1.2 candidate was
trained. It keeps all protocol 1.1 scenarios and thresholds and adds two gates:
one for source leakage and one for downstream usefulness.

This is representation evaluation only. It provides no clinical, diagnostic,
treatment, dosing, monitoring, alerting, or safety evidence.

## Why a new version is required

The protocol 1.1 test results were inspected when candidate 6 was released, so
they cannot be used to select another model. Protocol 1.1 also left one known
weakness open: a simple probe identified the source dataset of a held-out day
with 0.93 balanced accuracy. Protocol 1.2 makes that weakness a release gate.

## Frozen split provenance

`glucofm-reserve-holdout` was run with seed `43` and reserve fraction `0.20` on
the protocol 1.1 split of each source. The protocol 1.1 test participants move
into training. Validation participants are unchanged.

- BIG IDEAs manifest SHA-256:
  `29402437dbab6ef8cfed58882b36b937c96fb850eebc8498becd5bb03e4cbd24`
- BIG IDEAs protocol 1.2 split SHA-256:
  `ae4f34ffdbc235a394d6f65de7510c883c15a685e88abe2884f11ab0cb7f878a`
- Colas manifest SHA-256:
  `cc660486424e6c51a47671eadf06ad0bdc849cf6f6120c52722225ac7ad136ff`
- Colas protocol 1.2 split SHA-256:
  `28de2e3418dc3939b8fd24fa6995c5ba71801941425c9e30bc1b6d93a1932a5b`

The partitions contain:

- training: 361 days from 159 declared participants
- validation: 71 days from 32 declared participants (unchanged from 1.1)
- prospective test: 71 days from 31 declared participants
  (16 BIG IDEAs days from 2 participants, 55 Colas days from 29 participants)

The new test participants were training data for candidate 6. Candidate 6
therefore cannot be evaluated on this holdout, and every protocol 1.2
candidate must start from random weights. The BIG IDEAs portion has only two
participants, so the per-source test result will be imprecise.

## Probes

Every probe is fitted on the training partition and scored on the target
partition (validation during development, test once). Features are
standardized with training statistics only.

- **Linear source probe:** class-balanced L2 logistic regression
  (penalty `1e-2`, LBFGS, 300 iterations) that predicts the source dataset.
  The same probe is fitted on the 11-feature summary baseline.
- **Same-participant retrieval:** fraction of days whose nearest other day
  (cosine similarity) comes from the same participant. This is reported for
  information only. It measures personal pattern and re-identification risk
  at the same time.
- **Hidden-window utility:** the final 72 readings (6 hours) are hidden. Using
  the first 18 hours only, closed-form ridge regression (penalty `10`)
  predicts the observed mean glucose of the hidden window. Rows need at least
  half of the hidden window observed. A last-visible-hour persistence baseline
  is reported for context.

## Release gates

A candidate must pass all seven checks:

1. to 5. The five unchanged protocol 1.1 checks (feature dispersion,
   effective rank, and stability under random 30% removal, a contiguous
   60-minute gap, and 15-minute cadence).
6. **Source leakage:** the model's linear-probe balanced accuracy minus the
   summary baseline's linear-probe balanced accuracy is at most `0.10`. In
   other words, the embedding may not reveal much more about the source
   dataset than eleven basic summary statistics do.
7. **Utility:** the model's hidden-window ridge MAE (mg/dL) is no worse than
   the summary baseline's.

For reference, candidate 6 measured on validation (development data, fitted
on protocol 1.1 training) had a source margin of `0.18` (fails gate 6) and a
utility margin of `-0.51` mg/dL (passes gate 7).

## Candidate budget and selection rule

- At most six candidates may be trained and measured on validation.
- Only candidates that pass all seven checks on validation are eligible.
- Among eligible candidates, the one with the lowest validation source margin
  is selected. Ties within `0.01` go to the lower hidden-window MAE.
- The test partition is opened once, for the selected candidate only. Its
  result is final. Thresholds will not be weakened and the holdout will not
  be used to select another model.
- If no candidate is eligible, the test partition stays closed and the negative
  result is reported.
