"""Adapter for the ShanghaiT1DM and ShanghaiT2DM datasets.

Authoritative release (CC BY 4.0):
https://doi.org/10.6084/m9.figshare.21600933.v5

Zhao Q. et al., "Chinese diabetes datasets for data-driven machine learning",
Scientific Data 10, 35 (2023).

Each Excel file is one recording period of one patient; the file name starts
with the patient number (for example ``2001_1_20201117``), and one patient can
have several files. Every reading is placed on the canonical five-minute grid.
The FreeStyle Libre H records every 15 minutes, so the two positions between
readings are marked missing rather than interpolated, and the observed-data
threshold is scaled to that cadence. Only the CGM column is used for canonical
days; meal times are read separately for descriptive analysis.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from glucotrace.canonical import (
    CGMDay,
    CGMReading,
    SourceProvenance,
    build_24h_windows,
    write_canonical_corpus,
)

DATASET_NAME = "Shanghai"
DATASET_VERSION = "figshare-21600933-v5"
SOURCE_URL = "https://doi.org/10.6084/m9.figshare.21600933.v5"
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DEVICE = "Abbott FreeStyle Libre H"
SUBSETS = ("T1DM", "T2DM")
SENSOR_INTERVAL_MINUTES = 15
GRID_INTERVAL_MINUTES = 5
FILE_PATTERN = re.compile(r"(\d+)_(\d+)_(\d{8})\.xlsx?$", re.IGNORECASE)
NO_RECORD = {"data not available", "未记录", ""}


@dataclass(frozen=True)
class Meal:
    participant_id: str
    subset: str
    timestamp: datetime
    description: str


def _pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the Shanghai adapter needs the 'shanghai' extra: "
            "pip install 'glucotrace[shanghai]'"
        ) from exc
    return pandas


def _iter_files(source: Path, subset: str) -> Iterable[tuple[str, bytes]]:
    """Yield (file name, bytes) for one subset from the ZIP or a directory."""

    folder = f"Shanghai_{subset}"
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if not name.startswith("__MACOSX/")
                and Path(name).parent.name == folder
                and FILE_PATTERN.fullmatch(Path(name).name)
            )
            for name in names:
                yield Path(name).name, archive.read(name)
        return
    directory = source / folder if (source / folder).is_dir() else source
    if not directory.is_dir():
        raise FileNotFoundError(f"expected the official ZIP or an extracted folder: {source}")
    for path in sorted(directory.iterdir()):
        if FILE_PATTERN.fullmatch(path.name):
            yield path.name, path.read_bytes()


def _read_frame(name: str, content: bytes) -> Any:
    pandas = _pandas()
    engine = "xlrd" if name.lower().endswith(".xls") else "openpyxl"
    frame = pandas.read_excel(io.BytesIO(content), engine=engine)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _cgm_column(frame: Any, name: str) -> str:
    matches = [column for column in frame.columns if column.startswith("CGM")]
    if len(matches) != 1 or "Date" not in frame.columns:
        raise ValueError(f"{name}: expected one 'CGM...' column and a 'Date' column")
    return matches[0]


def read_shanghai_file(name: str, content: bytes) -> tuple[str, list[CGMReading], list[datetime]]:
    """Return (patient id, CGM readings, meal timestamps) for one recording."""

    match = FILE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected Shanghai file name: {name}")
    participant_id = match.group(1)
    frame = _read_frame(name, content)
    cgm = _cgm_column(frame, name)
    pandas = _pandas()
    readings: list[CGMReading] = []
    meals: list[datetime] = []
    diet = "Dietary intake" if "Dietary intake" in frame.columns else None
    for line, row in enumerate(frame.itertuples(index=False), start=2):
        values = dict(zip(frame.columns, row))
        stamp = pandas.to_datetime(values["Date"], errors="coerce")
        if pandas.isna(stamp):
            continue
        timestamp = stamp.to_pydatetime().replace(tzinfo=None)
        raw = values[cgm]
        if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
            try:
                glucose = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name}: row {line}: non-numeric CGM value {raw!r}") from exc
            if not math.isfinite(glucose) or glucose <= 0:
                raise ValueError(f"{name}: row {line}: invalid CGM value {raw!r}")
            readings.append(CGMReading(timestamp, glucose))
        if diet is not None:
            text = values[diet]
            if isinstance(text, str) and text.strip().lower() not in NO_RECORD:
                meals.append(timestamp)
    if not readings:
        raise ValueError(f"{name} contains no CGM readings")
    return participant_id, readings, meals


def _provenance(participant_id: str, subset: str) -> SourceProvenance:
    return SourceProvenance(
        dataset=DATASET_NAME,
        version=f"{DATASET_VERSION}/{subset}",
        participant_id=participant_id,
        device=DEVICE,
        unit="mg/dL",
        source_url=SOURCE_URL,
        license_name=LICENSE_NAME,
        license_url=LICENSE_URL,
    )


def load_participant_readings(
    source: str | Path, subset: str
) -> dict[str, list[CGMReading]]:
    """All CGM readings per patient, merging a patient's recording files."""

    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    merged: dict[str, dict[datetime, CGMReading]] = {}
    for name, content in _iter_files(Path(source), subset):
        participant_id, readings, _ = read_shanghai_file(name, content)
        by_time = merged.setdefault(participant_id, {})
        for reading in readings:
            by_time.setdefault(reading.timestamp, reading)
    if not merged:
        raise FileNotFoundError(f"no Shanghai_{subset} recordings found under {source}")
    return {pid: sorted(by_time.values()) for pid, by_time in sorted(merged.items())}


def load_meals(source: str | Path) -> list[Meal]:
    """Logged meal times from both subsets (for descriptive analysis only)."""

    meals: list[Meal] = []
    for subset in SUBSETS:
        for name, content in _iter_files(Path(source), subset):
            participant_id, _, stamps = read_shanghai_file(name, content)
            meals.extend(Meal(participant_id, subset, stamp, "") for stamp in stamps)
    return meals


def load_shanghai(
    source: str | Path,
    subset: str,
    *,
    min_observed_fraction: float = 0.8,
    anchor: str = "segment_start",
) -> list[CGMDay]:
    """Build canonical five-minute-grid days for one subset.

    ``min_observed_fraction`` is expressed relative to the sensor's own cadence:
    0.8 means at least 80% of the 96 expected 15-minute readings.
    """

    scale = GRID_INTERVAL_MINUTES / SENSOR_INTERVAL_MINUTES
    days: list[CGMDay] = []
    for participant_id, readings in load_participant_readings(source, subset).items():
        days.extend(
            build_24h_windows(
                readings,
                _provenance(participant_id, subset),
                interval_minutes=GRID_INTERVAL_MINUTES,
                min_observed_fraction=min_observed_fraction * scale,
                anchor=anchor,
            )
        )
    if not days:
        raise ValueError(f"no complete 24-hour Shanghai_{subset} days were found")
    return days


def prepare_shanghai(
    source: str | Path,
    output_dir: str | Path,
    subset: str,
    *,
    min_observed_fraction: float = 0.8,
    anchor: str = "segment_start",
    overwrite: bool = False,
) -> Path:
    days = load_shanghai(
        source, subset, min_observed_fraction=min_observed_fraction, anchor=anchor
    )
    return write_canonical_corpus(days, output_dir, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the ShanghaiT1DM/T2DM datasets for GlucoTrace research."
    )
    parser.add_argument("source", type=Path, help="diabetes_datasets.zip or its extracted folder")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--min-observed-fraction", type=float, default=0.8)
    parser.add_argument(
        "--anchor",
        choices=("segment_start", "midnight"),
        default="segment_start",
        help="start each day at the recording start or just after midnight",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    days = load_shanghai(
        args.source,
        args.subset,
        min_observed_fraction=args.min_observed_fraction,
        anchor=args.anchor,
    )
    manifest = write_canonical_corpus(days, args.output_dir, overwrite=args.overwrite)
    participants = {day.provenance.participant_id for day in days}
    print(f"participants={len(participants)}")
    print(f"days={len(days)}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
