# Changelog

## 0.2.0 (2026-09-23)

The distribution is now named `glucotrace`. The research checkpoint is
unchanged (SHA-256 `1fbeecd6…8092b`); protocol 1.2 did not produce a
replacement.

### Added

- `glucotrace` command with thirteen subcommands (`encode`, `compare`, `search`,
  `report`, `pretrain`, `evaluate`, …). The `glucofm-*` commands still work.
- `import glucotrace` alias for the `glucofm` API.
- `glucotrace download-model`: fetches the checkpoint from the GitHub release
  and verifies its pinned SHA-256. Commands find the checkpoint via
  `$GLUCOTRACE_CHECKPOINT`, `./checkpoints`, then `~/.cache/glucotrace`.
- `glucotrace report`: an offline, self-contained HTML report of a CGM day and
  its nearest days, with no scripts and no network requests.
- `--unit mmol/L` for encode, compare, search, and report. Files whose median
  contradicts the declared unit are rejected instead of silently misread.
- Evaluation protocol 1.2 (`--protocol-version 1.2`): linear source probe,
  same-participant retrieval, hidden-window utility, and two new release gates.
- Source-invariance training options: `--within-source-negatives` and
  `--source-adversary-weight` (gradient-reversal source classifier).
- `glucofm-reserve-holdout --parent-protocol` to record which protocol's test
  partition was consumed.
- Protocol 1.2 declaration, frozen splits, six validation reports, and a
  negative-result write-up.

### Findings

- The released checkpoint beats a summary-statistics baseline at predicting a
  hidden six-hour window on validation (7.71 vs 8.22 mg/dL MAE).
- Source leakage persists (linear probe 0.88 to 0.98 balanced accuracy). The
  likely cause is that Colas days start at midnight while BIG IDEAs days do not.

## 0.1.0 (2026-09-15)

- First research release: dual-stream encoder, BIG IDEAs and Colas adapters,
  source-balanced latent pretraining, protocols 1.0 and 1.1, calibrated
  128-number fingerprints, and `encode`, `compare`, and `search`.
