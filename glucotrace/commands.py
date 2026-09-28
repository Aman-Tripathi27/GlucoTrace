"""Command-line interfaces for encoding, comparing, and searching CGM days."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .inference import (
    ResearchEncoder,
    cosine_similarity,
    download_checkpoint,
    search_manifests,
    sha256_file,
)


def _common_csv_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--glucose-col", default="glucose")
    parser.add_argument("--window-index", type=int)
    input_arguments(parser)


def input_arguments(parser: argparse.ArgumentParser) -> None:
    """Options describing how to read a user's CGM CSV."""

    parser.add_argument(
        "--unit",
        choices=("mg/dL", "mmol/L"),
        help="glucose unit of a plain CSV (default mg/dL); mmol/L is converted. "
        "Dexcom Clarity exports declare their own unit",
    )
    parser.add_argument(
        "--format",
        dest="input_format",
        choices=("plain", "dexcom-clarity"),
        default="plain",
        help="plain timestamp/glucose CSV, or a Dexcom Clarity CSV export",
    )
    parser.add_argument(
        "--out-of-range",
        choices=("missing", "clamp"),
        default="missing",
        help="treat Low/High readings as missing (default) or record them at "
        "the sensor limits of 40 and 400 mg/dL",
    )


def warn_out_of_range(metadata: dict[str, Any]) -> None:
    count = int(metadata.get("out_of_range_readings", 0))
    if not count:
        return
    action = (
        "treated as missing"
        if metadata["out_of_range_policy"] == "missing"
        else "recorded at 40/400 mg/dL"
    )
    print(
        f"warning: {Path(metadata['path']).name}: {count} Low/High reading(s) "
        f"were outside the sensor range and {action}",
        file=sys.stderr,
    )


def _encoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="defaults to $GLUCOTRACE_CHECKPOINT, ./checkpoints, then the "
        "download cache",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)


def build_download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the released research checkpoint and verify it."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destination (default: $GLUCOTRACE_HOME or ~/.cache/glucotrace)",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def download_main() -> None:
    args = build_download_parser().parse_args()
    path = download_checkpoint(args.output, force=args.force)
    print(f"checkpoint={path}")
    print(f"sha256={sha256_file(path)}")


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"saved={output}")


def _model_metadata(encoder: ResearchEncoder) -> dict[str, Any]:
    return {
        "checkpoint": str(encoder.checkpoint_path),
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "embedding_dimension": encoder.model.config.hidden_size,
        "calibration_method": encoder.calibration["method"],
    }


def build_encode_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode one complete CGM day into a 128-number fingerprint."
    )
    parser.add_argument("csv", type=Path)
    _common_csv_arguments(parser)
    _encoder_arguments(parser)
    return parser


def encode_main() -> None:
    args = build_encode_parser().parse_args()
    encoder = ResearchEncoder.load(args.checkpoint, device=args.device)
    fingerprint, input_metadata = encoder.encode_csv(
        args.csv,
        timestamp_col=args.timestamp_col,
        glucose_col=args.glucose_col,
        window_index=args.window_index,
        unit=args.unit,
        input_format=args.input_format,
        out_of_range=args.out_of_range,
    )
    warn_out_of_range(input_metadata)
    _emit(
        {
            "schema_version": "1.0",
            "kind": "glucotrace_fingerprint",
            "research_only": True,
            "model": _model_metadata(encoder),
            "input": input_metadata,
            "fingerprint": fingerprint.tolist(),
        },
        args.output,
    )


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two complete CGM days by fingerprint cosine similarity."
    )
    parser.add_argument("first_csv", type=Path)
    parser.add_argument("second_csv", type=Path)
    _common_csv_arguments(parser)
    _encoder_arguments(parser)
    return parser


def compare_main() -> None:
    args = build_compare_parser().parse_args()
    encoder = ResearchEncoder.load(args.checkpoint, device=args.device)
    first, first_metadata = encoder.encode_csv(
        args.first_csv,
        timestamp_col=args.timestamp_col,
        glucose_col=args.glucose_col,
        window_index=args.window_index,
        unit=args.unit,
        input_format=args.input_format,
        out_of_range=args.out_of_range,
    )
    second, second_metadata = encoder.encode_csv(
        args.second_csv,
        timestamp_col=args.timestamp_col,
        glucose_col=args.glucose_col,
        window_index=args.window_index,
        unit=args.unit,
        input_format=args.input_format,
        out_of_range=args.out_of_range,
    )
    similarity = cosine_similarity(first, second)
    warn_out_of_range(first_metadata)
    warn_out_of_range(second_metadata)
    _emit(
        {
            "schema_version": "1.0",
            "kind": "glucotrace_comparison",
            "research_only": True,
            "model": _model_metadata(encoder),
            "first_input": first_metadata,
            "second_input": second_metadata,
            "cosine_similarity": similarity,
            "cosine_distance": 1.0 - similarity,
            "interpretation_limit": (
                "Similarity is descriptive research output, not a clinical score "
                "or proof that two days have the same cause or outcome."
            ),
        },
        args.output,
    )


def build_search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find fingerprint-nearest CGM days in canonical manifests."
    )
    parser.add_argument("query_csv", type=Path)
    parser.add_argument(
        "--manifest", action="append", type=Path, required=True
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="include a byte-identical canonical query file in results",
    )
    _common_csv_arguments(parser)
    _encoder_arguments(parser)
    return parser


def search_main() -> None:
    args = build_search_parser().parse_args()
    encoder = ResearchEncoder.load(args.checkpoint, device=args.device)
    query, input_metadata = encoder.encode_csv(
        args.query_csv,
        timestamp_col=args.timestamp_col,
        glucose_col=args.glucose_col,
        window_index=args.window_index,
        unit=args.unit,
        input_format=args.input_format,
        out_of_range=args.out_of_range,
    )
    warn_out_of_range(input_metadata)
    matches = search_manifests(
        encoder,
        query,
        args.manifest,
        top_k=args.top_k,
        exclude_sha256=None if args.include_self else input_metadata["sha256"],
        batch_size=args.batch_size,
    )
    _emit(
        {
            "schema_version": "1.0",
            "kind": "glucotrace_search",
            "research_only": True,
            "model": _model_metadata(encoder),
            "query": input_metadata,
            "manifests": [str(path) for path in args.manifest],
            "top_k_requested": args.top_k,
            "result_count": len(matches),
            "results": matches,
            "interpretation_limit": (
                "Nearest days reflect this experimental embedding and may share "
                "dataset artifacts rather than physiology."
            ),
        },
        args.output,
    )
