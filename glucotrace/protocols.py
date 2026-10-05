"""Verify published protocol split files against their declared checksums.

Each evaluation protocol fixed its manifests and participant splits before
training. The files are published under ``data/processed`` and their SHA-256
values are recorded here and in the declaration documents, so anyone can
confirm that the partitions used for a result are the ones declared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus import CanonicalCGMDataset

PROTOCOL_FILES: dict[str, dict[str, Any]] = {
    "1.0": {
        "declaration": "EVALUATION.md (hashes recorded in evaluations/protocol-1.0-candidate.json)",
        "files": {
            "big_ideas/manifest.json": (
                "29402437dbab6ef8cfed58882b36b937c96fb850eebc8498becd5bb03e4cbd24"
            ),
            "big_ideas/splits.json": (
                "a75398f7ec89410fbde9e062ff6a7721d70ed5f66d07deb1e7bf3308b9285294"
            ),
            "colas/manifest.json": (
                "cc660486424e6c51a47671eadf06ad0bdc849cf6f6120c52722225ac7ad136ff"
            ),
            "colas/splits.json": (
                "b03fa6f67738a0ca98495d3b4b3aa9cbdcec6f889e8fc576606cd45d18d6b0b9"
            ),
        },
    },
    "1.1": {
        "declaration": "EVALUATION_1_1.md",
        "files": {
            "big_ideas/manifest.json": (
                "29402437dbab6ef8cfed58882b36b937c96fb850eebc8498becd5bb03e4cbd24"
            ),
            "big_ideas/splits_protocol_1_1.json": (
                "dca6fbd5dced68f92d59d9e506e8a11ecd87e783f9d68fbb78300f27f2e5aadd"
            ),
            "colas/manifest.json": (
                "cc660486424e6c51a47671eadf06ad0bdc849cf6f6120c52722225ac7ad136ff"
            ),
            "colas/splits_protocol_1_1.json": (
                "1dbfdf0fd29aab8fe3434127381e58a3ae6ff5da76fc532f67f55c89debcf83c"
            ),
        },
    },
    "1.2": {
        "declaration": "EVALUATION_1_2.md",
        "files": {
            "big_ideas/manifest.json": (
                "29402437dbab6ef8cfed58882b36b937c96fb850eebc8498becd5bb03e4cbd24"
            ),
            "big_ideas/splits_protocol_1_2.json": (
                "ae4f34ffdbc235a394d6f65de7510c883c15a685e88abe2884f11ab0cb7f878a"
            ),
            "colas/manifest.json": (
                "cc660486424e6c51a47671eadf06ad0bdc849cf6f6120c52722225ac7ad136ff"
            ),
            "colas/splits_protocol_1_2.json": (
                "28de2e3418dc3939b8fd24fa6995c5ba71801941425c9e30bc1b6d93a1932a5b"
            ),
        },
    },
    "1.3": {
        "declaration": "EVALUATION_1_3.md",
        "files": {
            "big_ideas_midnight/manifest.json": (
                "2347aa6588eb86c380267824db2956ebdef58727c7cae81185bd3e458eb59c31"
            ),
            "big_ideas_midnight/splits_protocol_1_3.json": (
                "c9e9cb7e01eb1025753a9f2c2ccc3e8b6d0e2fabb77647baf0192c16e143a5e0"
            ),
            "colas_midnight/manifest.json": (
                "706e9056aec882633ad4930b2995a9aefaff5fae90fba7144a76a9825a74576a"
            ),
            "colas_midnight/splits_protocol_1_3.json": (
                "cdd3955d682ee6ba23e387d11243e39e5efc213c62cd01384ea8e7751c45e4b7"
            ),
        },
    },
    "1.5": {
        "declaration": "EVALUATION_1_5.md",
        "files": {
            "big_ideas_midnight/manifest.json": (
                "2347aa6588eb86c380267824db2956ebdef58727c7cae81185bd3e458eb59c31"
            ),
            "big_ideas_midnight/splits_protocol_1_3.json": (
                "c9e9cb7e01eb1025753a9f2c2ccc3e8b6d0e2fabb77647baf0192c16e143a5e0"
            ),
            "colas_midnight/manifest.json": (
                "706e9056aec882633ad4930b2995a9aefaff5fae90fba7144a76a9825a74576a"
            ),
            "colas_midnight/splits_protocol_1_3.json": (
                "cdd3955d682ee6ba23e387d11243e39e5efc213c62cd01384ea8e7751c45e4b7"
            ),
            "shanghai_t1dm_midnight/manifest.json": (
                "de8435efe6cf7d0468e47aace2fdbe3e71cf7bd7b848613f4f16740650dc9d50"
            ),
            "shanghai_t1dm_midnight/splits_protocol_1_5.json": (
                "9c3a02fd43c6509e38000472c58611df809fdcc76b53123f1e59380dae7f7290"
            ),
            "shanghai_t2dm_midnight/manifest.json": (
                "3458d20f352f3b59c4d34c680c046c7baf5cf58a9c113dcec022365c043154d7"
            ),
            "shanghai_t2dm_midnight/splits_protocol_1_5.json": (
                "ac586243849a7786d52103ea8c76c00273cd8dd0dad9629d2a38f46edca33038"
            ),
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol(
    version: str, data_root: str | Path = "data/processed", *, check_days: bool = False
) -> dict[str, Any]:
    """Check every declared file for ``version``; optionally check day files.

    With ``check_days``, every canonical day CSV listed in the manifests must
    exist locally (after running the dataset adapters) and match its SHA-256.
    """

    if version not in PROTOCOL_FILES:
        raise ValueError(f"unknown protocol {version!r}; known: {sorted(PROTOCOL_FILES)}")
    root = Path(data_root)
    entry = PROTOCOL_FILES[version]
    files = []
    for relative, expected in entry["files"].items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else None
        files.append(
            {
                "file": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": "missing" if actual is None
                else ("ok" if actual == expected else "MISMATCH"),
            }
        )
    days: dict[str, Any] | None = None
    if check_days:
        checked, problems = 0, []
        for relative in entry["files"]:
            if not relative.endswith("manifest.json"):
                continue
            try:
                checked += len(CanonicalCGMDataset(root / relative))
            except (FileNotFoundError, ValueError) as exc:
                problems.append(f"{relative}: {exc}")
        days = {"days_verified": checked, "problems": problems}
    passed = all(item["status"] == "ok" for item in files) and (
        days is None or not days["problems"]
    )
    return {
        "protocol": version,
        "declaration": entry["declaration"],
        "passed": passed,
        "files": files,
        "canonical_days": days,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a protocol's published manifests and splits by SHA-256."
    )
    parser.add_argument("protocol", nargs="?", choices=sorted(PROTOCOL_FILES))
    parser.add_argument("--all", action="store_true", help="verify every protocol")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--check-days",
        action="store_true",
        help="also verify every canonical day CSV (run the dataset adapters first)",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.all and args.protocol is None:
        raise ValueError("name a protocol (for example 1.3) or pass --all")
    versions = sorted(PROTOCOL_FILES) if args.all else [args.protocol]
    reports = [
        verify_protocol(version, args.data_root, check_days=args.check_days)
        for version in versions
    ]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            print(f"protocol {report['protocol']} ({report['declaration']}):")
            for item in report["files"]:
                print(f"  {item['status']:<8} {item['file']}")
            if report["canonical_days"] is not None:
                days = report["canonical_days"]
                print(f"  days verified: {days['days_verified']}")
                for problem in days["problems"]:
                    print(f"  PROBLEM  {problem}")
            print("  PASSED" if report["passed"] else "  FAILED")
    if not all(report["passed"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
