"""Adapter for the PhysioNet BIG IDEAs CGM dataset.

Dataset page: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/
The adapter reads only ``Dexcom_<participant>.csv`` files. It ignores the
wearable and food files because this project currently models CGM alone.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from glucofm.canonical import (
    CGMDay,
    CGMReading,
    SourceProvenance,
    build_24h_windows,
    write_canonical_corpus,
)
from glucofm.data import parse_timestamp

DATASET_NAME = "BIG_IDEAs"
DATASET_VERSION = "1.1.3"
SOURCE_URL = (
    "https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/"
)
LICENSE_NAME = "Open Data Commons Attribution License v1.0"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/"
TIMESTAMP_COLUMN = "Timestamp (YYYY-MM-DDThh:mm:ss)"
GLUCOSE_COLUMN = "Glucose Value (mg/dL)"
EVENT_COLUMN = "Event Type"


def read_big_ideas_file(path: str | Path) -> tuple[str, list[CGMReading]]:
    """Read physical EGV observations from one native BIG IDEAs Dexcom CSV."""

    csv_path = Path(path)
    participant_id = csv_path.stem.removeprefix("Dexcom_")
    if not participant_id or participant_id == csv_path.stem:
        raise ValueError(f"expected a Dexcom_<participant>.csv file, got {csv_path.name}")

    readings: list[CGMReading] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {TIMESTAMP_COLUMN, GLUCOSE_COLUMN, EVENT_COLUMN}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing BIG IDEAs column(s): {names}")

        for line_number, row in enumerate(reader, start=2):
            if row[EVENT_COLUMN].strip() != "EGV":
                continue
            timestamp_text = row[TIMESTAMP_COLUMN].strip()
            glucose_text = row[GLUCOSE_COLUMN].strip()
            if not timestamp_text or not glucose_text:
                continue
            try:
                timestamp = parse_timestamp(timestamp_text)
                glucose = float(glucose_text)
            except ValueError as exc:
                raise ValueError(f"{csv_path}: line {line_number}: {exc}") from exc
            if not math.isfinite(glucose):
                raise ValueError(f"{csv_path}: line {line_number}: non-finite glucose")
            readings.append(CGMReading(timestamp, glucose))

    if not readings:
        raise ValueError(f"{csv_path} contains no EGV glucose observations")
    return participant_id, sorted(readings)


def load_big_ideas(
    dataset_root: str | Path,
    *,
    version: str = DATASET_VERSION,
    stride_hours: int = 24,
    min_observed_fraction: float = 0.8,
) -> list[CGMDay]:
    """Convert all discovered participant Dexcom files to canonical CGM days."""

    root = Path(dataset_root)
    paths = sorted(root.glob("[0-9][0-9][0-9]/Dexcom_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no participant Dexcom files found under {root}; expected "
            "<root>/001/Dexcom_001.csv"
        )

    days: list[CGMDay] = []
    for path in paths:
        participant_id, readings = read_big_ideas_file(path)
        if path.parent.name != participant_id:
            raise ValueError(
                f"participant folder {path.parent.name!r} does not match "
                f"file participant {participant_id!r}"
            )
        provenance = SourceProvenance(
            dataset=DATASET_NAME,
            version=version,
            participant_id=participant_id,
            device="Dexcom G6",
            unit="mg/dL",
            source_url=SOURCE_URL,
            license_name=LICENSE_NAME,
            license_url=LICENSE_URL,
        )
        days.extend(
            build_24h_windows(
                readings,
                provenance,
                stride_hours=stride_hours,
                min_observed_fraction=min_observed_fraction,
            )
        )
    if not days:
        raise ValueError(
            "BIG IDEAs files were read, but no complete 24-hour windows met "
            "the observed-data threshold"
        )
    return days


def prepare_big_ideas(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    version: str = DATASET_VERSION,
    stride_hours: int = 24,
    min_observed_fraction: float = 0.8,
    overwrite: bool = False,
) -> Path:
    """Build canonical CSVs and return the written manifest path."""

    days = load_big_ideas(
        dataset_root,
        version=version,
        stride_hours=stride_hours,
        min_observed_fraction=min_observed_fraction,
    )
    return write_canonical_corpus(days, output_dir, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare PhysioNet BIG IDEAs Dexcom files for GlucoTrace research."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--version", default=DATASET_VERSION)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--min-observed-fraction", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    days = load_big_ideas(
        args.dataset_root,
        version=args.version,
        stride_hours=args.stride_hours,
        min_observed_fraction=args.min_observed_fraction,
    )
    manifest = write_canonical_corpus(
        days, args.output_dir, overwrite=args.overwrite
    )
    participants = {day.provenance.participant_id for day in days}
    observed = sum(int(day.observed_mask.sum()) for day in days)
    total = sum(day.observed_mask.numel() for day in days)
    print(f"participants={len(participants)}")
    print(f"days={len(days)}")
    print(f"observed_fraction={observed / total:.6f}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
