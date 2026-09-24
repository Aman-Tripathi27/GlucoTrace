# Evaluation protocol 1.3 declaration

Protocol 1.3 was declared on 2026-09-24, before any protocol 1.3 candidate was
trained. It changes one thing relative to protocol 1.2: **every canonical day
now starts just after local midnight.** All gates, probes, and thresholds are
unchanged. Tracked in issue #5.

This is representation evaluation only. It provides no clinical, diagnostic,
treatment, dosing, monitoring, alerting, or safety evidence.

## Why

Protocol 1.2 found that a linear probe identifies the source dataset from the
embedding with 0.88 to 0.98 balanced accuracy. The likely cause is that BIG
IDEAs days started at the recording start time (for example 17:23), while
Colas days started at midnight. Window start time therefore identified the
dataset. Protocol 1.3 removes that difference and asks whether the leakage
falls below the unchanged gate.

## Corpus change

Both adapters gained `--anchor midnight`. Each 24-hour window starts at the
first time in the first five-minute interval after local midnight that keeps
the device's own sampling phase. For example, a sensor that reads at
`hh:m3:32` produces days starting at `00:03:32`. Every reading therefore stays
on the grid, and every day in both sources starts between `00:00` and `00:05`.

- BIG IDEAs: 94 days (was 112; partial first and last days are no longer
  complete windows). All 16 participants remain.
- Colas: 391 days, 206 participants. It was already almost entirely
  midnight-aligned; a few windows moved.

## Frozen split provenance

Partition membership is **copied exactly** from protocol 1.2 with
`glucofm-reserve-holdout --carry-membership`. No participant changes
partition and none were dropped.

The protocol 1.2 test partition was never opened: no candidate passed
validation, so under that protocol's rules it stayed closed. No protocol 1.2
decision used it. It is therefore reused as the protocol 1.3 holdout. As in
protocol 1.2, the released candidate 6 was trained on these participants and
cannot be evaluated on this holdout; every protocol 1.3 candidate starts from
random weights.

- BIG IDEAs midnight manifest SHA-256:
  `2347aa6588eb86c380267824db2956ebdef58727c7cae81185bd3e458eb59c31`
- BIG IDEAs protocol 1.3 split SHA-256:
  `c9e9cb7e01eb1025753a9f2c2ccc3e8b6d0e2fabb77647baf0192c16e143a5e0`
- Colas midnight manifest SHA-256:
  `706e9056aec882633ad4930b2995a9aefaff5fae90fba7144a76a9825a74576a`
- Colas protocol 1.3 split SHA-256:
  `cdd3955d682ee6ba23e387d11243e39e5efc213c62cd01384ea8e7751c45e4b7`

The partitions contain:

- training: 348 days from 159 participants (67 BIG IDEAs, 281 Colas)
- validation: 68 days from 32 participants (13 BIG IDEAs, 55 Colas)
- prospective test: 69 days from 31 participants (14 BIG IDEAs from 2
  participants, 55 Colas from 29 participants)

## Probes and gates

Identical to [EVALUATION_1_2.md](EVALUATION_1_2.md): the five protocol 1.1
checks, the linear source-probe gate (model minus summary baseline balanced
accuracy at most `0.10`), and the hidden-window utility gate (model ridge MAE
no worse than the summary baseline). The summary baseline is unchanged. With
every day starting at midnight, it and the model now see the same clock
information.

## Candidates, declared in advance

All use the candidate 6 recipe (40 epochs, batch size 32, seed 7,
source-balanced sampling) and start from random weights.

| Candidate | Within-source negatives | Adversary weight | Rationale |
|---|:-:|:-:|---|
| g | no | 0 | Plain recipe: does alignment alone fix the leakage? |
| h | yes | 0.1 | Best protocol 1.2 recipe (candidate c) |
| i | yes | 0 | Best protocol 1.2 utility (candidate a) |

## Selection rule

- Only candidates that pass all seven checks on validation are eligible.
- Among eligible candidates, choose the lowest validation source margin; ties
  within `0.01` go to the lower hidden-window MAE.
- Open the test partition once, for the selected candidate only. The result is
  final. Thresholds will not be weakened and the holdout will not be used to
  select another model.
- If no candidate is eligible, the test partition stays closed and the result
  is reported.
