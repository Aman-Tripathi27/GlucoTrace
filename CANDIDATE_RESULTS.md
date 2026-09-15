# Research candidate 1 results

This document records the first full GlucoFM training attempt and its negative
release decision. The checkpoint is a research candidate, not a released model,
and the results do not establish scientific or clinical usefulness.

## Training run

- candidate file: `checkpoints/glucofm-research-candidate.pt`
- checkpoint SHA-256:
  `d35299dfd119aa5b2f9d8b47c4a4ce54c68bf26300a51eb92782e8271202fe5d`
- architecture: default 412,684-parameter GlucoFM encoder
- objective: masked latent prediction with an exponential-moving-average teacher
- sources: BIG IDEAs and Colas 2019
- unique training days: 348
- source-balanced draws per epoch: 548
- epochs: 10
- batch size: 16
- learning rate: `3e-4`
- weight decay: `0.05`
- mask probability: `0.50`
- EMA decay: `0.996`
- variance weight: `0.05`
- seed: 7

Training loss decreased from `0.281811` to `0.039412`; validation loss decreased
from `0.137667` to `0.037280`. These objective values are optimization
diagnostics only and are not evidence that the representation is useful.

## Frozen protocol 1.0 result

The complete machine-readable report is
`evaluations/protocol-1.0-candidate.json`. It covers 84 held-out days from 35
declared participants: 22 BIG IDEAs days and 62 Colas days. The candidate failed
the predeclared all-checks release gate.

Passed checks:

- mean feature standard deviation: `0.1693` (required at least `0.05`)
- 60-minute contiguous-gap stability: median cosine `0.9215`, 10th-percentile
  cosine `0.7429`, and five-neighbor overlap `0.7200`

Failed checks:

- effective rank: `1.4331` (required at least `8.0`)
- random 30% removal: median cosine `0.4848`, 10th-percentile cosine `0.0852`,
  and five-neighbor overlap `0.3162`
- 15-minute cadence: median cosine `0.1274`, 10th-percentile cosine `-0.1931`,
  and five-neighbor overlap `0.1468`

The source probe produced `0.7143` accuracy and `0.7771` balanced accuracy. The
summary baseline produced `0.7500` and `0.8013`, respectively. These results show
that source is predictable, but BIG IDEAs and Colas differ in population,
protocol, device, geography, and collection era. They cannot be interpreted as
a sensor comparison.

## Decision and next research boundary

The candidate is not promoted to `glucofm-research.pt` because it failed three
of five fixed engineering checks. The test result will not be used to weaken
protocol 1.0 thresholds or select another model on the same test partition.

The architectural diagnosis is that patch-token prediction does not directly
constrain the pooled day embedding to remain stable under missingness, while the
small variance penalty did not preserve enough independent embedding
dimensions. A future objective can add pooled-view consistency and explicit
covariance regularization, developed using training and validation data only.
An untouched dataset or prospectively reserved partition is required before a
new release decision.
