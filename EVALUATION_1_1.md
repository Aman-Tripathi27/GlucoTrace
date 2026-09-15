# Evaluation protocol 1.1 declaration

Protocol 1.1 was declared on 2026-09-15 after candidate 1 failed protocol 1.0
and before candidate 2 was trained. It preserves every protocol 1.0 scenario,
metric, repetition count, and engineering threshold. The version changes because
it binds a newly reserved participant holdout and adds a validation-first access
rule.

This is representation evaluation only. It provides no clinical, diagnostic,
treatment, dosing, monitoring, alerting, or safety evidence.

## Why a new boundary is required

The protocol 1.0 test aggregate was inspected when candidate 1 failed. It cannot
honestly be reused to select candidate 2. For each source, protocol 1.1 reserves
20% of the former training participants using a deterministic SHA-256 ordering
with seed 29. Former protocol 1.0 test participants move into training, while
the original validation participants remain validation-only.

Candidate 2 must start from random weights and may not load candidate 1. The
newly reserved participants were included in candidate 1 pretraining, but no
metrics for them were inspected and no candidate 1 parameters are reused. This
is therefore an internal prospective holdout, not a new cohort or external
validation.

## Frozen split provenance

- BIG IDEAs manifest SHA-256:
  `29402437dbab6ef8cfed58882b36b937c96fb850eebc8498becd5bb03e4cbd24`
- BIG IDEAs protocol 1.1 split SHA-256:
  `dca6fbd5dced68f92d59d9e506e8a11ecd87e783f9d68fbb78300f27f2e5aadd`
- Colas manifest SHA-256:
  `cc660486424e6c51a47671eadf06ad0bdc849cf6f6120c52722225ac7ad136ff`
- Colas protocol 1.1 split SHA-256:
  `1dbfdf0fd29aab8fe3434127381e58a3ae6ff5da76fc532f67f55c89debcf83c`

The resulting partitions contain:

- training: 363 days from 159 declared participants
- validation: 71 days from 32 declared participants
- prospective test: 69 days from 31 declared participants

Source-specific prospective test counts are 14 BIG IDEAs days from two
participants and 55 Colas days from 29 participants.

## Validation-first access rule

Development may inspect training loss and protocol metrics calculated on the
validation partition. The prospective test partition must not be loaded until a
candidate passes all five engineering checks on validation. Once opened, its
result is final for that candidate; failed test thresholds will not be weakened
or used for another selection on the same holdout.

## Unchanged release checks

The protocol 1.0 checks remain unchanged:

- mean feature standard deviation is at least `0.05`
- effective rank is at least `8.0`
- for random 30% removal, a contiguous 60-minute gap, and 15-minute cadence:
  median cosine is at least `0.80`, 10th-percentile cosine is at least `0.50`,
  and five-neighbor overlap is at least `0.50`

Passing these checks only clears this project's internal representation sanity
gate. It cannot establish clinical utility, generalization, fairness, privacy,
or equivalence between sensors.
