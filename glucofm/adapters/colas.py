"""Adapter for the Colas et al. 2019 PLOS supporting dataset.

Authoritative release:
https://doi.org/10.1371/journal.pone.0225817.s001

The source provides clock times but no calendar dates. This adapter preserves
the first clock time and reconstructs exact five-minute relative positions from
row order, using 2000-01-01 only as a documented placeholder date.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, TextIO

from glucofm.canonical import (
    CGMDay,
    CGMReading,
    SourceProvenance,
    build_24h_windows,
    write_canonical_corpus,
)

DATASET_NAME = "Colas2019"
DATASET_VERSION = "10.1371-journal.pone.0225817.s001"
SOURCE_URL = "https://doi.org/10.1371/journal.pone.0225817.s001"
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEVICE = "Medtronic MiniMed iPro CGMS"
INTERVAL_MINUTES = 5
POSITIONS_PER_DAY = 288
PLACEHOLDER_DATE = datetime(2000, 1, 1)
MAX_CLOCK_DRIFT_SECONDS = 300.0

INDEX_COLUMN = ""
TIME_COLUMN = "hora"
GLUCOSE_COLUMN = "glucemia"
CASE_PATTERN = re.compile(r"case\s+(\d+)\.csv", re.IGNORECASE)


def _case_number(filename: str) -> int:
    match = CASE_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"expected a Colas 'case <number>.csv' file, got {filename}")
    number = int(match.group(1))
    if number <= 0:
        raise ValueError("Colas participant number must be positive")
    return number


def _participant_id(filename: str) -> str:
    number = _case_number(filename)
    return f"{number:03d}"


def _clock_seconds(value: str) -> int:
    try:
        parsed = datetime.strptime(value.strip(), "%H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"invalid Colas clock time {value!r}") from exc
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def _circular_clock_error(actual_seconds: int, expected_seconds: int) -> float:
    difference = (actual_seconds - expected_seconds + 43_200) % 86_400 - 43_200
    return abs(float(difference))


def _read_case(handle: TextIO, filename: str) -> tuple[str, list[CGMReading]]:
    participant_id = _participant_id(filename)
    reader = csv.DictReader(handle)
    required = {INDEX_COLUMN, TIME_COLUMN, GLUCOSE_COLUMN}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        shown = ["row_index" if name == "" else name for name in sorted(missing)]
        raise ValueError(f"{filename} is missing Colas column(s): {', '.join(shown)}")

    rows = list(reader)
    if len(rows) < POSITIONS_PER_DAY or len(rows) % POSITIONS_PER_DAY:
        raise ValueError(
            f"{filename} must contain a whole number of {POSITIONS_PER_DAY}-slot days"
        )

    try:
        first_index = int(rows[0][INDEX_COLUMN].strip())
        first_clock_seconds = _clock_seconds(rows[0][TIME_COLUMN])
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{filename}: invalid first row: {exc}") from exc

    anchor = PLACEHOLDER_DATE + timedelta(seconds=first_clock_seconds)
    readings: list[CGMReading] = []
    for position, row in enumerate(rows):
        line_number = position + 2
        try:
            row_index = int(row[INDEX_COLUMN].strip())
            if row_index != first_index + position:
                raise ValueError(
                    f"row index {row_index} is not consecutive from {first_index}"
                )
            actual_clock = _clock_seconds(row[TIME_COLUMN])
            expected_clock = (
                first_clock_seconds + position * INTERVAL_MINUTES * 60
            ) % 86_400
            clock_error = _circular_clock_error(actual_clock, expected_clock)
            if clock_error > MAX_CLOCK_DRIFT_SECONDS:
                raise ValueError(
                    f"clock differs from nominal grid by {clock_error:.1f} seconds"
                )

            glucose_text = row[GLUCOSE_COLUMN].strip()
            if glucose_text in {"", "NA"}:
                continue
            glucose = float(glucose_text)
            if not math.isfinite(glucose):
                raise ValueError("glucose must be finite, numeric, or NA")
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"{filename}: line {line_number}: {exc}") from exc

        timestamp = anchor + timedelta(minutes=INTERVAL_MINUTES * position)
        readings.append(CGMReading(timestamp, glucose))

    if not readings:
        raise ValueError(f"{filename} contains no observed glucose")
    return participant_id, readings


def read_colas_case(path: str | Path) -> tuple[str, list[CGMReading]]:
    """Read one extracted native Colas case CSV."""

    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        return _read_case(handle, csv_path.name)


def _iter_cases(source: Path) -> Iterable[tuple[str, str]]:
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = [
                name
                for name in archive.namelist()
                if CASE_PATTERN.fullmatch(Path(name).name)
            ]
            for name in sorted(names, key=_case_number):
                with archive.open(name) as raw:
                    yield Path(name).name, raw.read().decode("utf-8-sig")
        return

    if source.is_dir():
        paths = [
            path
            for path in source.rglob("case*.csv")
            if CASE_PATTERN.fullmatch(path.name)
        ]
        for path in sorted(paths, key=lambda item: _case_number(item.name)):
            yield path.name, path.read_text(encoding="utf-8-sig")
        return
    raise FileNotFoundError(
        f"expected the official Colas ZIP or an extracted directory: {source}"
    )


def load_colas(
    source: str | Path,
    *,
    stride_hours: int = 24,
    min_observed_fraction: float = 0.8,
) -> list[CGMDay]:
    """Convert all discovered Colas cases into canonical CGM days."""

    days: list[CGMDay] = []
    seen_participants: set[str] = set()
    for filename, contents in _iter_cases(Path(source)):
        participant_id, readings = _read_case(io.StringIO(contents), filename)
        if participant_id in seen_participants:
            raise ValueError(f"duplicate Colas participant {participant_id}")
        seen_participants.add(participant_id)
        provenance = SourceProvenance(
            dataset=DATASET_NAME,
            version=DATASET_VERSION,
            participant_id=participant_id,
            device=DEVICE,
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
    if not seen_participants:
        raise FileNotFoundError(f"no Colas case CSV files found in {source}")
    if not days:
        raise ValueError(
            "Colas files were read, but no 24-hour windows met the coverage threshold"
        )
    return days


def prepare_colas(
    source: str | Path,
    output_dir: str | Path,
    *,
    stride_hours: int = 24,
    min_observed_fraction: float = 0.8,
    overwrite: bool = False,
) -> Path:
    """Build canonical Colas CSVs and return the manifest path."""

    days = load_colas(
        source,
        stride_hours=stride_hours,
        min_observed_fraction=min_observed_fraction,
    )
    return write_canonical_corpus(days, output_dir, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the official Colas 2019 PLOS CGM supporting data."
    )
    parser.add_argument("source", type=Path, help="official S1 ZIP or extracted folder")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--min-observed-fraction", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    days = load_colas(
        args.source,
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
