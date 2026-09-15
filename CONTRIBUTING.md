# Contributing to GlucoFM

Thank you for helping make this small CGM representation-learning project more
readable and reproducible. Contributions are welcome from software engineers,
data stewards, researchers, and documentation reviewers.

## Project boundary

GlucoFM is research software. Contributions must not present it as a medical
device or make diagnostic, treatment, dosing, monitoring, alerting, safety, or
state-of-the-art claims. A pull request that changes this boundary will not be
accepted without a separately governed project decision.

Never commit protected health information, credentials, access tokens, private
datasets, or files whose redistribution terms are unclear.

## Ways to contribute

- Report a reproducible bug.
- Improve documentation, typing, error messages, or tests.
- Propose a public-dataset adapter with clear consent, provenance, and license.
- Improve missing-data handling without inventing glucose values.
- Add transparent baselines or reproducible representation evaluations.
- Review code for leakage, privacy, reproducibility, or accessibility problems.

For substantial architecture, objective, split-policy, or evaluation changes,
open an issue before writing a large patch. This lets maintainers agree on the
scientific boundary and avoids duplicated work.

## Development setup

Fork the repository, create a focused branch, and install it in an isolated
Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

Python 3.10 or newer is supported. Do not add a dependency when a small standard
library or existing PyTorch implementation is clear enough.

## Pull-request checklist

- Keep the change small and readable.
- Add or update tests for changed behavior.
- Run the complete `pytest` suite.
- Update relevant documentation and model/data cards.
- Preserve missing-data masks and the no-interpolation rule.
- Do not include raw datasets or intermediate candidate checkpoints.
- Record seeds, data/split checksums, and configuration for experimental work.
- State negative results and limitations as clearly as positive results.
- Confirm that no clinical, diagnostic, or SOTA claim was introduced.

## Dataset adapters

A proposed adapter must document:

- the authoritative source URL and version
- license and redistribution terms
- participant identifier and grouping policy
- device, units, timezone, cadence, and known timestamp limitations
- exact column mapping and missing-value rules
- exclusions and their machine-checkable reasons
- checksums for source and canonical artifacts
- tests built from synthetic fixtures rather than redistributed participant data

Adapters must emit the canonical timestamp/glucose/mask representation. They
must not load clinical outcomes merely because those outcomes accompany a CGM
dataset.

## Model and evaluation changes

Use training and validation partitions for development. Once a held-out result
has been inspected, do not use it to select another model or weaken a threshold.
Material evaluation changes require a new, predeclared protocol version and a
new untouched holdout. Passing an engineering gate is not clinical validation.

Changes to the public 128-number fingerprint require an output-schema or model
version decision, backward-compatibility notes, and updated encode/compare/search
tests.

## Commit and review style

Write imperative, focused commit messages such as `Add cadence masking test`.
Explain why the change is needed, what was tested, and any remaining limitation
in the pull request. Maintainers may ask for a smaller PR when unrelated changes
are combined.

By contributing, you agree that your contribution is licensed under this
repository's MIT License.
