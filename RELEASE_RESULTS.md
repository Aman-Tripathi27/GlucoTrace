# GlucoFM research checkpoint results

This report describes `checkpoints/glucofm-research.pt`. It is an experimental
representation model, not a medical device or a clinically evaluated model.

## Artifact

- checkpoint SHA-256:
  `1fbeecd68d81d239fa26726b67ca2d05bf485569a8d3619cdc624f86bba8092b`
- size: approximately 2.0 MB
- trainable encoder parameters: 511,372
- PyTorch version recorded by the checkpoint: 2.10.0
- explicit checkpoint marker: `research_only=true`
- fingerprint calibration: validation feature z-score followed by L2 normalization

The adjacent `glucofm-research.pt.sha256` file can be used to verify the binary.

## Data boundary

Protocol 1.1 was declared before this checkpoint was trained. It reserved 69
days from 31 participants from the former training partitions. Candidate 6 was
initialized randomly and did not reuse candidate 1 weights.

- training: 363 unique days from 159 declared participants
- validation: 71 days from 32 declared participants
- prospective test: 69 days from 31 declared participants
- test sources: 14 BIG IDEAs days and 55 Colas 2019 days

This is an internal prospective holdout, not an external cohort. The underlying
sources have important population and study-design differences. See
`EVALUATION_1_1.md` for the exact split lineage and checksums.

## Training

Candidate 6 used 40 epochs, batch size 32, seed 7, learning rate `3e-4`, weight
decay `0.05`, and 562 source-balanced draws per epoch. Its objective combined:

- density-weighted masked hourly-token prediction from an EMA teacher
- two mask-only views using scattered gaps, contiguous gaps, or reduced cadence
- direct pooled-embedding consistency
- pooled feature variance and correlation penalties
- symmetric in-batch contrastive separation between different days

Candidate development inspected validation data only. Candidate 5 passed the
three missingness checks but had effective rank `7.33`, below the fixed minimum
of `8.0`. The otherwise identical 40-epoch candidate 6 reached `8.88` on
validation and therefore qualified for the one-time prospective evaluation.

## Prospective protocol 1.1 result

Candidate 6 passed all five internal engineering checks on the prospective
partition:

- mean feature standard deviation: `0.7720` (minimum `0.05`)
- effective rank: `8.5469` (minimum `8.0`)
- random 30% removal: median cosine `0.9940`, 10th-percentile cosine `0.9840`,
  five-neighbor overlap `0.9432`
- contiguous 60-minute gap: median cosine `0.9987`, 10th-percentile cosine
  `0.9946`, five-neighbor overlap `0.9652`
- 15-minute cadence: median cosine `0.9929`, 10th-percentile cosine `0.9860`,
  five-neighbor overlap `0.9517`

The complete aggregate report is `evaluations/protocol-1.1.json`, with SHA-256
`0268b9b0b896b2e8628198d2df19ffe0b04e2adc7b5c2d9462d571e44985ed3a`.
It is metric-identical to the candidate-6 decision report. The released file
adds validation-only calibration metadata without changing any learned tensor.

## Important negative signal

The source probe achieved `0.9710` accuracy and `0.9286` balanced accuracy,
compared with `0.7101` and `0.6052` for the summary baseline. The checkpoint
therefore retains strong source-related information. Because source also
changes with device, population, geography, study protocol, and collection era,
this cannot identify the cause and is not a sensor-performance result.

Passing the internal gate establishes only that the representation is
non-collapsed by the declared checks and stable under the specified artificial
missingness perturbations on this holdout. It does not establish downstream
utility, external generalization, sensor equivalence, privacy, fairness, safety,
or clinical value.
