# Research pretraining

This project implements a reproducible self-supervised training path and ships
one research-only checkpoint. The objective is intended for transparent
experimentation with declared CGM sources, not clinical use.

## Input partitions

Every corpus must have a canonical `manifest.json` and a participant-disjoint
`splits.json`. Pretraining opens only the `train` partition. Validation loss uses
only the `validation` partition. The command does not load test days.

For the current local BIG IDEAs and Colas corpora:

```bash
glucofm-pretrain \
  --corpus data/processed/big_ideas/manifest.json data/processed/big_ideas/splits_protocol_1_1.json \
  --corpus data/processed/colas/manifest.json data/processed/colas/splits_protocol_1_1.json \
  --epochs 40 --batch-size 32 \
  --output checkpoints/glucofm-research.pt
```

`--corpus MANIFEST SPLITS` can be repeated for additional approved sources.
Manifest and split checksums are saved in the checkpoint.

## Source-balanced sampling

The current training partition has more Colas days than BIG IDEAs days. A plain
shuffle would therefore optimize mostly for Colas. `SourceBalancedSampler`
alternates declared dataset sources exactly; source counts differ by at most one
per epoch. Days are shuffled reproducibly inside each source. When a smaller
source is exhausted it is reshuffled and cycled, so researchers must report the
number of sampled examples as well as the number of unique days.

By default, an epoch draws `number_of_sources × largest_source_size` examples.
This visits every day in the largest source once while matching that draw count
from every smaller source. `--samples-per-epoch` can override this value.

The sampler seed is combined with the epoch number. Re-running with the same
inputs, seed, software, and hardware follows the same index sequence.

## Masked latent-prediction objective

For each 24-hour day:

1. Identify hourly patches containing at least one physical measurement.
2. Randomly select 50% of eligible patches by default, while retaining at least
   one visible patch when possible.
3. Remove their glucose values from the student input by updating both the
   values and observation mask. Student gap age is recomputed from this reduced
   visibility mask, preventing a hidden measurement from leaking through a
   derived feature.
4. Encode the complete physical record with a non-gradient teacher encoder.
5. Predict the teacher’s 128-value token for each hidden hour from the student’s
   contextual token.
6. Compute Smooth L1 error only on hidden patches, weighted by their physical
   observation density.
7. Create a second mask-only view using scattered removal, a contiguous gap, or
   10-/15-minute cadence.
8. Directly align the two pooled day embeddings with the clean teacher while
   enforcing feature variance and low cross-feature correlation.
9. Use a symmetric in-batch contrastive term so matching day views remain close
   while different days remain distinguishable.
10. After the optimizer step, update teacher parameters as an exponential moving
   average of student parameters. The default decay is 0.996.

This objective predicts representations, not clinical outcomes. The model’s
reconstruction head is not optimized by this pretraining path and must not be
treated as a glucose estimator.

## Checkpoint contents

The saved model file contains:

- EMA teacher encoder weights for later embedding extraction
- exact model, objective, and optimizer-run configurations
- source manifest and split-file paths with SHA-256 checksums
- seed, PyTorch version, and per-epoch training/validation losses
- an explicit `research_only` marker

The current command creates a final encoder checkpoint and does not yet support
resuming optimizer state. Exact release configuration and results are recorded
in [RELEASE_RESULTS.md](RELEASE_RESULTS.md).

## What this does not establish

A decreasing latent loss does not prove that embeddings are scientifically
useful, robust across sensors, private, fair, or safe. Test partitions must stay
untouched until the evaluation protocol is frozen. This software must not be
used for diagnosis, dosing, alerts, treatment, or patient care.
