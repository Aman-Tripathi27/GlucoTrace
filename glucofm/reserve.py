"""Command line entry point for reserving a prospective CGM holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus import SPLIT_NAMES, create_prospective_holdout_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reserve a new holdout from a previous training partition."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("parent_split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = create_prospective_holdout_split(
        args.manifest,
        args.parent_split,
        args.output,
        reserve_fraction=args.reserve_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
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
