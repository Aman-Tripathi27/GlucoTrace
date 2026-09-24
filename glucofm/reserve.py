"""Command line entry point for reserving a prospective CGM holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import SPLIT_NAMES, carry_split_membership, create_prospective_holdout_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reserve a new holdout from a previous training partition."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("parent_split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--parent-protocol",
        default="1.0",
        help="protocol whose test partition was consumed and moves into training",
    )
    parser.add_argument(
        "--carry-membership",
        action="store_true",
        help="copy the parent's exact partitions onto a re-windowed manifest "
        "instead of reserving a new holdout",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.carry_membership:
        output = carry_split_membership(
            args.manifest, args.parent_split, args.output, overwrite=args.overwrite
        )
    else:
        output = create_prospective_holdout_split(
            args.manifest,
            args.parent_split,
            args.output,
            reserve_fraction=args.reserve_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
            parent_protocol=args.parent_protocol,
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    for name in SPLIT_NAMES:
        split = payload["splits"][name]
        print(
            f"{name}: participants={split['participant_count']} "
            f"days={split['record_count']}"
        )
    print(f"split_file={output}")


if __name__ == "__main__":
    main()
