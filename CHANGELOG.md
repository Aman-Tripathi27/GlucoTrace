# Changelog

## 0.3.0 (2026-10-05)

### Breaking: renamed to GlucoTrace everywhere

- The Python package is now `glucotrace` (was `glucofm`); the model classes are
  `GlucoTrace` and `GlucoTraceConfig`.
- All commands run through `glucotrace <command>`; the separate `glucofm-*`
  commands are removed.
- **New default model `glucotrace-0.3.pt`** (no clock input, trained on three
  cohorts; SHA-256 `a4ee0f8f…45389`). **Fingerprints are not comparable with
  0.2 fingerprints.** The 0.2 model stays at release v0.2.0 and as
  `checkpoints/glucotrace-research.pt`. Run `glucotrace download-model` to
  fetch the new model.
- Output kinds are `glucotrace_fingerprint`, `glucotrace_comparison`, and
  `glucotrace_search`.
- Cites the independent GlucoFM paper (arXiv:2605.30865) as related work.

### Added

- `--format libreview` reads FreeStyle Libre LibreView CSV exports directly:
  historic readings only, unit from the header, decimal commas, `LO`/`HI`,
  and automatic month-day/day-month detection that refuses ambiguous files
  (`--date-order mdy|dmy` resolves them). Closes issue #2.

- Protocol 1.4 (random-phase training windows, three seeds, a no-clock
  diagnostic) with `--random-phase-raw` and `--no-clock` for pretraining.
  Result: removing the clock input eliminated source leakage (margin
  -0.004) at a ~3% utility cost; no recipe passed all gates.
- ShanghaiT1DM/T2DM adapter (`glucotrace prepare-shanghai`, China, FreeStyle
  Libre 15-minute data) and protocol 1.5 (three cohorts, cadence-harmonized
  source gate, bootstrap intervals, `--clock-bins`, `--cadence-dropout`).
  Results: `PROTOCOL_1_5_RESULTS.md`.

## 0.2.2 (2026-09-28)

### Fixed

- **Timezone-aware timestamps now keep local time of day.** Previously
  `08:00+05:30` reached the model as 02:30 UTC, skewing fingerprints for any
  file with offsets. Files without offsets were unaffected.
- **Dexcom `Low`/`High` readings no longer crash the loader.** They are
  treated as missing by default, with a warning and a count in the output;
  `--out-of-range clamp` records them at 40/400 mg/dL instead.

### Added

- `--format dexcom-clarity` reads Dexcom Clarity CSV exports directly.
- `glucotrace verify-protocol`: checks the now-published manifests and split
  files of protocols 1.0 to 1.3 against their declared SHA-256 values.
- `.gitattributes` keeps published protocol files byte-exact on every OS.

## 0.2.1 (2026-09-24)

Available on PyPI: `pip install glucotrace`. The research checkpoint is
unchanged.

### Added

- `--anchor midnight` for the BIG IDEAs and Colas adapters: each day starts
  in the first five minutes after local midnight, on the device's own
  sampling grid.
- `glucofm-reserve-holdout --carry-membership`: copy a split's exact
  participant partitions onto a re-windowed manifest.
- Evaluation protocol 1.3 declaration, three validation reports, and results.
- PyPI publishing through GitHub Actions Trusted Publishing, and Zenodo
  archiving for a citable DOI.

### Findings

- Midnight alignment halved source leakage (best margin 0.193 to 0.099),
  confirming the protocol 1.2 diagnosis.
- Aligned candidates lost their hidden-window advantage over the baseline, so
  none passed all checks and the checkpoint is unchanged.

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
