"""Canonical, provenance-aware 24-hour CGM records.

This module is deliberately independent of any one public dataset. Source
adapters convert their native files to :class:`CGMReading` objects and this
module applies the shared segmentation and grid-building rules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True, order=True)
class CGMReading:
    """One physically observed interstitial-glucose measurement in mg/dL."""

    timestamp: datetime
    glucose_mg_dl: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.glucose_mg_dl):
            raise ValueError("glucose_mg_dl must be finite")


@dataclass(frozen=True)
class SourceProvenance:
    """Dataset information that must remain attached to every derived day."""

    dataset: str
    version: str
    participant_id: str
    device: str
    unit: str
    source_url: str
    license_name: str
    license_url: str

    def __post_init__(self) -> None:
        if self.unit != "mg/dL":
            raise ValueError("canonical glucose unit must be mg/dL")
        for field_name, value in asdict(self).items():
            if not value:
                raise ValueError(f"provenance field {field_name} cannot be empty")


@dataclass(frozen=True)
class CGMDay:
    """One model-ready 24-hour window on a five-minute grid.

    Missing glucose positions are zero-filled only to create a finite tensor.
    ``observed_mask`` is the authority on whether a value was measured.
    ``gap_age_minutes`` records time since the latest physical measurement.
    """

    start_time: datetime
    timestamps: tuple[datetime, ...]
    glucose: torch.Tensor
    observed_mask: torch.Tensor
    gap_age_minutes: torch.Tensor
    time_of_day: torch.Tensor
    provenance: SourceProvenance
    interval_minutes: int = 5

    def __post_init__(self) -> None:
        expected = 24 * 60 // self.interval_minutes
        if len(self.timestamps) != expected:
            raise ValueError(f"a CGM day must contain {expected} timestamps")
        if self.glucose.shape != (expected,):
            raise ValueError(f"glucose must have shape [{expected}]")
        if self.observed_mask.shape != (expected,):
            raise ValueError(f"observed_mask must have shape [{expected}]")
        if self.gap_age_minutes.shape != (expected,):
            raise ValueError(f"gap_age_minutes must have shape [{expected}]")
        if self.time_of_day.shape != (expected, 2):
            raise ValueError(f"time_of_day must have shape [{expected}, 2]")
        if self.observed_mask.dtype != torch.bool:
            raise TypeError("observed_mask must be Boolean")
        if not torch.isfinite(self.glucose).all():
            raise ValueError("glucose tensor must be finite")
        if not torch.isfinite(self.gap_age_minutes).all():
            raise ValueError("gap_age_minutes tensor must be finite")
        if not torch.isfinite(self.time_of_day).all():
            raise ValueError("time_of_day tensor must be finite")
        if (self.glucose[~self.observed_mask] != 0).any():
            raise ValueError("missing glucose positions must be zero-filled")

    @property
    def observed_fraction(self) -> float:
        return float(self.observed_mask.float().mean())


def split_continuous_segments(
    readings: Iterable[CGMReading], *, max_gap_minutes: int = 60
) -> list[list[CGMReading]]:
    """Split a trace whenever consecutive observations are over an hour apart."""

    if max_gap_minutes <= 0:
        raise ValueError("max_gap_minutes must be positive")
    ordered = sorted(readings)
    if not ordered:
        return []

    segments: list[list[CGMReading]] = [[ordered[0]]]
    maximum_gap = timedelta(minutes=max_gap_minutes)
    for reading in ordered[1:]:
        gap = reading.timestamp - segments[-1][-1].timestamp
        if gap.total_seconds() < 0:
            raise AssertionError("readings were not sorted")
        if gap > maximum_gap:
            segments.append([reading])
        else:
            segments[-1].append(reading)
    return segments


def _time_features(timestamps: Sequence[datetime]) -> torch.Tensor:
    phases = []
    for timestamp in timestamps:
        minutes = (
            timestamp.hour * 60
            + timestamp.minute
            + timestamp.second / 60
            + timestamp.microsecond / 60_000_000
        )
        angle = 2 * math.pi * minutes / (24 * 60)
        phases.append((math.sin(angle), math.cos(angle)))
    return torch.tensor(phases, dtype=torch.float32)


def _gap_age(mask: torch.Tensor, interval_minutes: int, cap_minutes: int) -> torch.Tensor:
    ages = torch.empty(mask.shape, dtype=torch.float32)
    age = float(cap_minutes)
    for index, observed in enumerate(mask.tolist()):
        if observed:
            age = 0.0
        else:
            age = min(float(cap_minutes), age + interval_minutes)
        ages[index] = age
    return ages


def _build_window(
    readings: Sequence[CGMReading],
    *,
    start: datetime,
    provenance: SourceProvenance,
    interval_minutes: int,
    max_gap_minutes: int,
    alignment_tolerance_seconds: float,
) -> CGMDay:
    slots = 24 * 60 // interval_minutes
    step_seconds = interval_minutes * 60
    sums = torch.zeros(slots, dtype=torch.float64)
    counts = torch.zeros(slots, dtype=torch.int64)
    end = start + timedelta(hours=24)

    for reading in readings:
        if reading.timestamp < start or reading.timestamp >= end:
            continue
        offset = (reading.timestamp - start).total_seconds()
        index = int(round(offset / step_seconds))
        if index >= slots:
            continue
        error = abs(offset - index * step_seconds)
        if error > alignment_tolerance_seconds:
            raise ValueError(
                f"timestamp {reading.timestamp.isoformat()} is {error:.1f}s "
                "from its nearest five-minute grid position"
            )
        sums[index] += reading.glucose_mg_dl
        counts[index] += 1

    observed = counts > 0
    glucose = torch.zeros(slots, dtype=torch.float32)
    glucose[observed] = (sums[observed] / counts[observed]).float()
    timestamps = tuple(
        start + timedelta(minutes=index * interval_minutes) for index in range(slots)
    )
    return CGMDay(
        start_time=start,
        timestamps=timestamps,
        glucose=glucose,
        observed_mask=observed,
        gap_age_minutes=_gap_age(observed, interval_minutes, max_gap_minutes),
        time_of_day=_time_features(timestamps),
        provenance=provenance,
        interval_minutes=interval_minutes,
    )


WINDOW_ANCHORS = ("segment_start", "midnight")


def build_24h_windows(
    readings: Iterable[CGMReading],
    provenance: SourceProvenance,
    *,
    interval_minutes: int = 5,
    stride_hours: int = 24,
    min_observed_fraction: float = 0.8,
    max_gap_minutes: int = 60,
    alignment_tolerance_seconds: float | None = None,
    anchor: str = "segment_start",
) -> list[CGMDay]:
    """Create non-partial 24-hour windows from continuous recording segments.

    With ``anchor="segment_start"`` each segment's windows begin at its first
    real measurement. With ``anchor="midnight"`` they begin at the first
    grid-consistent time in the first sampling interval after local midnight,
    so every day covers the same clock hours and window start time carries no
    information about the source. By default, windows do not overlap.
    """

    if interval_minutes <= 0 or 24 * 60 % interval_minutes:
        raise ValueError("interval_minutes must evenly divide 24 hours")
    if stride_hours <= 0:
        raise ValueError("stride_hours must be positive")
    if anchor not in WINDOW_ANCHORS:
        raise ValueError(f"anchor must be one of {WINDOW_ANCHORS}")
    if not 0.0 <= min_observed_fraction <= 1.0:
        raise ValueError("min_observed_fraction must be in [0, 1]")
    tolerance = (
        0.4 * interval_minutes * 60
        if alignment_tolerance_seconds is None
        else float(alignment_tolerance_seconds)
    )
    if tolerance < 0 or tolerance >= interval_minutes * 30:
        raise ValueError("alignment tolerance must be in [0, half the interval)")

    slots = 24 * 60 // interval_minutes
    final_offset = timedelta(minutes=(slots - 1) * interval_minutes)
    stride = timedelta(hours=stride_hours)
    days: list[CGMDay] = []
    for segment in split_continuous_segments(
        readings, max_gap_minutes=max_gap_minutes
    ):
        start = segment[0].timestamp
        if anchor == "midnight":
            start = _first_start_after_midnight(start, interval_minutes)
        while start + final_offset <= segment[-1].timestamp:
            day = _build_window(
                segment,
                start=start,
                provenance=provenance,
                interval_minutes=interval_minutes,
                max_gap_minutes=max_gap_minutes,
                alignment_tolerance_seconds=tolerance,
            )
            if day.observed_fraction >= min_observed_fraction:
                days.append(day)
            start += stride
    return days


def _first_start_after_midnight(first: datetime, interval_minutes: int) -> datetime:
    """Return the earliest time >= ``first`` that falls in [00:00, 00:00 + interval)
    and keeps ``first``'s phase on the sampling grid."""

    step = timedelta(minutes=interval_minutes)
    midnight = first.replace(hour=0, minute=0, second=0, microsecond=0)
    start = midnight + (first - midnight) % step
    if start < first:
        start += timedelta(days=1)
    return start


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_corpus(
    days: Sequence[CGMDay], output_dir: str | Path, *, overwrite: bool = False
) -> Path:
    """Write canonical day CSVs and a provenance manifest.

    The function never includes source data in the Python package. Researchers
    run it locally after obtaining a dataset under its own license.
    """

    if not days:
        raise ValueError("cannot write an empty canonical corpus")
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass overwrite=True to replace it"
        )
    day_dir = destination / "days"
    day_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for day in days:
        stamp = day.start_time.strftime("%Y%m%dT%H%M%S")
        filename = "__".join(
            (
                _safe_name(day.provenance.dataset),
                _safe_name(day.provenance.participant_id),
                stamp,
            )
        ) + ".csv"
        path = day_dir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ("timestamp", "glucose", "observed_mask", "gap_age_minutes")
            )
            for index, timestamp in enumerate(day.timestamps):
                observed = bool(day.observed_mask[index])
                writer.writerow(
                    (
                        timestamp.isoformat(),
                        float(day.glucose[index]) if observed else "",
                        int(observed),
                        float(day.gap_age_minutes[index]),
                    )
                )
        records.append(
            {
                "file": str(path.relative_to(destination)),
                "sha256": _sha256(path),
                "start_time": day.start_time.isoformat(),
                "participant_id": day.provenance.participant_id,
                "observed_count": int(day.observed_mask.sum()),
                "observed_fraction": day.observed_fraction,
                "provenance": asdict(day.provenance),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "canonical_unit": "mg/dL",
        "interval_minutes": days[0].interval_minutes,
        "window_positions": len(days[0].timestamps),
        "records": records,
    }
    destination.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path
