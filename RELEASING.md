# Releasing GlucoTrace

These are the steps to publish a version. Each step uses your own accounts and
credentials, so run them yourself. Nothing here uploads data or model weights
automatically.

## 0. One-time machine fix (macOS)

If `git` prints "You have not agreed to the Xcode license agreements", run:

```bash
sudo xcodebuild -license accept
```

## 1. Verify

```bash
python -m pip install -e '.[dev]'
pytest
(cd checkpoints && shasum -a 256 -c glucofm-research.pt.sha256)
```

The printed checksum must match `CHECKPOINT_SHA256` in
`glucofm/inference.py`. `glucotrace download-model` refuses any other file.

## 2. Commit and tag

```bash
git add -A
git commit -m "GlucoTrace 0.2.0"
git tag -a v0.2.0 -m "GlucoTrace 0.2.0"
git push origin main --tags
```

## 3. GitHub release (hosts the checkpoint)

`glucotrace download-model` downloads from
`https://github.com/Aman-Tripathi27/GlucoTrace/releases/download/v0.2.0/glucofm-research.pt`.
Create that release and attach the checkpoint:

```bash
gh release create v0.2.0 checkpoints/glucofm-research.pt \
  checkpoints/glucofm-research.pt.sha256 \
  --title "GlucoTrace 0.2.0" --notes-file CHANGELOG.md
```

For later versions, update `CHECKPOINT_URL` (and `CHECKPOINT_SHA256` if the
weights change) before tagging.

## 4. PyPI

The name `glucotrace` was unclaimed on 2026-09-23. Use a PyPI API token;
Trusted Publishing from GitHub Actions is better once the project exists.

```bash
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*   # rehearse first
python -m twine upload dist/*
```

The wheel contains code only (about 65 KB). Weights come from the GitHub
release.

## 5. Optional: Hugging Face Hub

`docs/huggingface/README.md` is a ready model card. Create a model repo, then
upload the checkpoint, its `.sha256` file, and that card as `README.md`.

## Before announcing

- Keep the research-only warning in every post. Never describe the model as
  diagnostic, predictive of health outcomes, or clinically validated.
- Link the negative results (`CANDIDATE_RESULTS.md`, `PROTOCOL_1_2_RESULTS.md`).
  They are part of what makes the project credible.
- Do not post screenshots that contain private CGM data.

## Project website (GitHub Pages)

`docs/index.html` is the project landing page. In the repository settings,
open Pages, choose "Deploy from a branch", then select `main` and `/docs`.
The site will be at `https://aman-tripathi27.github.io/GlucoTrace/`.
