# Model Card: GlucoFM (research checkpoint)

## Summary

GlucoFM is a compact PyTorch Transformer encoder for regularly sampled
continuous glucose monitor time series. This repository includes source code,
tests, and one research-only checkpoint. It is not clinically evaluated.

- Version: 0.1.0
- License: MIT
- Status: experimental research checkpoint

## Intended use

The code is intended for method development, education, reproducibility
experiments, and representation-learning research with appropriately governed
CGM datasets. Suitable uses require researchers to define their own task,
dataset splits, evaluation protocol, and ethical review.

## Out-of-scope use

Do not use this software for diagnosis, screening, treatment selection, insulin
or medication dosing, alarms, patient monitoring, or any other clinical or
safety-critical decision. Do not treat model reconstructions as measured
glucose. The project is not a medical device and provides no performance or
safety assurance.

## Architecture

The input is a regularly sampled mg/dL glucose sequence plus a Boolean physical
observation mask, capped gap age, and circular time-of-day. Unobserved numerical
placeholders are removed before glucose features are calculated.

Three causal, mask-normalized trailing averages create a slow-trend bank. The
residual from the one-hour trend and changes between adjacent observed residuals
create a rapid-event stream. Each stream is projected separately from 12-reading
patches, then concatenated into 128-value hourly tokens. Fixed positional codes
and a small bidirectional Transformer contextualize the 24 tokens. Global mean,
global variation, and four six-hour segment means are projected into one
128-value day embedding.

Default configuration:

- hidden size: 128
- attention heads: 4
- Transformer layers: 3
- feed-forward size: 256
- patch size: 12 samples (one hour at five-minute sampling)
- trend windows: 3, 12, and 36 samples (15, 60, and 180 minutes)
- dropout: 0.1
- pooling segments: 4
- trainable parameters: 511,372

The causal qualifier applies to the trend filters, not Transformer attention.
The model represents a completed window and is not a real-time forecaster.

## Data and preprocessing

No dataset is bundled with the source package. The checkpoint was trained on
participant-disjoint portions of BIG IDEAs and Colas 2019. The canonical
public-data pipeline constructs 288-position days, zero-fills tensor
placeholders, retains an observation mask, records gap age, and keeps source
provenance in a manifest. The convenience CSV loader follows the same
no-interpolation rule.

The implemented adapters cover BIG IDEAs and the Colas 2019 PLOS supporting
dataset. The Colas adapter uses only per-participant CGM case files; it does not
load the accompanying clinical variables or outcomes. Its source contains clock
time without calendar dates, so generated dates are explicitly relative
placeholders.

The model uses a fixed documented numerical transformation of observed mg/dL
values: `(glucose - 120) / 40`. It does not fit normalization statistics on an
input day. Neither loader infers units or performs physiological plausibility
filtering.

The public fingerprint applies coordinate-wise validation-set centering and
scaling to the raw pooled embedding, followed by L2 normalization. These values
are stored in the checkpoint. This is representation calibration for cosine
comparison, not clinical calibration.

## Training and evaluation

The primary research objective masks complete hourly patches and trains a
student encoder to predict the corresponding full-context tokens from an
exponential-moving-average teacher. Smooth L1 token error is weighted by
physical patch density. A second mask-only view adds pooled consistency,
variance, feature-correlation, and in-batch contrastive losses. Source-balanced
sampling prevents the larger declared corpus from dominating by day count. The
reconstruction head is not trained by this objective.

Evaluation protocols 1.0 and 1.1 were fixed before their respective training
decisions and include embedding-collapse diagnostics, controlled missingness
and cadence perturbations, nearest-neighbor stability, a transparent
summary-feature baseline, and a dataset-source separability probe. Their
thresholds are engineering sanity checks, not clinical or scientific
validation. Because the two current datasets differ in population and study
design as well as device, cross-source separability cannot be interpreted as
sensor performance. See [EVALUATION.md](EVALUATION.md) and
[EVALUATION_1_1.md](EVALUATION_1_1.md).

The first full research candidate failed three of five predeclared engineering
checks and was not promoted to a released model. Its negative result and exact
decision boundary are recorded in
[CANDIDATE_RESULTS.md](CANDIDATE_RESULTS.md).

After validation-only development, the repaired checkpoint passed all five
unchanged checks on protocol 1.1's 69-day prospective partition. Source
remained highly predictable, which is a warning that collection artifacts are
retained. Exact provenance and aggregate results are in
[RELEASE_RESULTS.md](RELEASE_RESULTS.md).

No downstream benchmark, clinically validated threshold, claim of external
generalization, or state-of-the-art comparison is provided. A downstream
study should report at least data provenance, participant-level splitting,
missingness, sensor type, units, demographics where lawful and appropriate,
uncertainty, calibration, subgroup results, and external validation.

The corpus utility creates deterministic participant-disjoint train,
validation, and test partitions and binds them to the exact canonical manifest
checksum. This prevents declared participant overlap but does not establish that
unrelated identifiers across different datasets represent different people.

## Risks and limitations

- CGM devices and populations can differ systematically.
- Missingness may correlate with behavior, device failure, or outcomes.
- Zero placeholders are protected by masks in this implementation, but the
  frequency and shape of gaps can still identify a dataset, sensor, or person.
- The fixed 15/60/180-minute decomposition may not match useful timescales in
  every dataset.
- Small Transformers can memorize participants or collection artifacts.
- Similar-day search can reveal participant identifiers and filenames from a
  private manifest if its output is not access-controlled.
- Reconstructions can look plausible while being wrong.
- This implementation has no privacy protection, uncertainty estimation,
  calibration, fairness mitigation, or deployment safeguards.

## Responsible research checklist

- Confirm consent, governance, and de-identification requirements.
- Split data by participant before fitting normalization or model parameters.
- Preserve units, device metadata, timezone, and sampling information.
- Quantify performance across gap lengths, devices, sites, and relevant groups.
- Compare with simple baselines and publish negative results.
- Keep a human-reviewed boundary between experimental outputs and care.

## Contact and changes

Use the repository issue tracker for reproducible bugs or proposed changes. Any
future checkpoint must record its exact data, objective, evaluation, and
limitations.
