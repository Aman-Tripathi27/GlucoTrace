# Evaluation protocol 1.0

This protocol was written before full model training or inspection of test-set
results. Its scenarios, repetitions, metrics, and engineering thresholds are
fixed under version `1.0`. Any future change must increment the version and be
reported alongside both protocols.

The protocol evaluates representation behavior only. It does not evaluate or
support diagnosis, treatment, dosing, monitoring, alerts, or patient care.

## Data boundary

The evaluator verifies that every provided manifest and split checksum matches
the checkpoint's recorded training provenance. Validation partitions are used
only to fit feature standardization and source centroids. Aggregate metrics are
then calculated on participant-disjoint test partitions. No model fitting occurs
inside the evaluator.

Run it after training with:

```bash
glucofm-evaluate checkpoints/glucofm-research.pt \
  --corpus data/processed/big_ideas/manifest.json data/processed/big_ideas/splits.json \
  --corpus data/processed/colas/manifest.json data/processed/colas/splits.json \
  --output evaluations/protocol-1.0.json
```

The JSON report records protocol version, checkpoint checksum, corpus checksums,
test counts, aggregate metrics, and all predeclared check outcomes. It contains
no per-participant embeddings or clinical variables.

## Controlled missingness

Every perturbation only changes observed values into missing values. It never
creates or interpolates glucose. Random locations and contiguous-gap starts are
derived deterministically from the protocol seed, scenario, repeat, and row.

- Randomly remove 10%, 30%, and 50% of physical observations; five repeats each.
- Remove one contiguous 30-, 60-, and 120-minute block; five locations each.
- Retain one of every two positions to simulate 10-minute cadence; two phases.
- Retain one of every three positions to simulate 15-minute cadence; three
  phases.

Cadence perturbations test sampling-density sensitivity. They do not reproduce
sensor noise, calibration, lag, or device algorithms and must not be described
as sensor-equivalence tests.

## Stability metrics

Model embeddings and summary-baseline features are standardized using validation
statistics. For every scenario the report includes:

- mean, median, and 10th-percentile cosine similarity to the clean day
- median relative L2 drift
- agreement of the nearest clean neighbor
- overlap of the five nearest clean neighbors
- mean fraction of physical observations removed

The transparent non-neural baseline contains mean, standard deviation, minimum,
quartiles, maximum, adjacent absolute-change mean and standard deviation,
observed fraction, and longest missing-run fraction. It provides context; it is
not a clinical baseline.

## Collapse diagnostics

On clean test embeddings the evaluator reports mean and minimum feature standard
deviation, covariance effective rank, and mean absolute off-diagonal cosine
similarity. These expose constant or nearly identical representations.

## Predeclared engineering checks

Protocol 1.0 marks its representation sanity gate as passed only when all of the
following hold:

- mean feature standard deviation is at least `0.05`
- effective rank is at least `8.0`
- for random 30% removal, a contiguous 60-minute gap, and 15-minute cadence:
  median cosine is at least `0.80`, 10th-percentile cosine is at least `0.50`,
  and five-neighbor overlap is at least `0.50`

These thresholds are fixed engineering checks, not validated scientific or
clinical cutoffs. Passing them cannot establish usefulness; failing them is a
clear reason not to release the checkpoint as the project model.

## Cross-source separability

A nearest-centroid source probe is fitted on validation embeddings and evaluated
on test embeddings. Accuracy, balanced accuracy, and mean centroid distance are
reported for the model and summary baseline. Strong source predictability is a
warning that embeddings may encode collection artifacts.

BIG IDEAs and Colas differ in device, population, study protocol, geography, and
collection era. Their separability cannot be attributed to sensor hardware.
Paired-sensor stability requires an approved paired dataset such as CGMacros and
is outside protocol 1.0 until that adapter and split policy are implemented.
