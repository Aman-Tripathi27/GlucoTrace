"""Fit validation-only fingerprint calibration and attach it to a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .inference import load_model_payload, resolve_device, sha256_file
from .pretrain import build_multisource_split


@torch.no_grad()
def calibrate_checkpoint(
    checkpoint_path: str | Path,
    corpus_pairs: Sequence[Sequence[str | Path]],
    output_path: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(checkpoint_path)
    destination = Path(output_path)
    resolved_device = resolve_device(device)
    model, payload = load_model_payload(source, device=resolved_device)

    expected = {
        (item["manifest_sha256"], item["split_sha256"])
        for item in payload["corpora"]
    }
    provided = {
        (sha256_file(manifest), sha256_file(split_file))
        for manifest, split_file in corpus_pairs
    }
    if provided != expected:
        raise ValueError("calibration corpora do not match checkpoint provenance")

    validation = build_multisource_split(corpus_pairs, "validation")
    loader = DataLoader(validation, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in loader:
        output = model(
            batch["glucose"].to(resolved_device),
            batch["observed_mask"].to(resolved_device),
            batch["gap_age_minutes"].to(resolved_device),
            batch["time_of_day"].to(resolved_device),
        )
        embeddings.append(output["pooled_embedding"].cpu())
    values = torch.cat(embeddings)
    center = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)

    calibrated = dict(payload)
    calibrated["embedding_calibration"] = {
        "method": "validation_zscore_then_l2",
        "partition": "validation",
        "validation_days": len(validation),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "corpora": [
            {
                "manifest_sha256": sha256_file(manifest),
                "split_sha256": sha256_file(split_file),
            }
            for manifest, split_file in corpus_pairs
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach validation-fitted fingerprint calibration."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--corpus",
        action="append",
        nargs=2,
        required=True,
        metavar=("MANIFEST", "SPLITS"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = calibrate_checkpoint(
        args.checkpoint,
        args.corpus,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
