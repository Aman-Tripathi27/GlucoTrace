"""CSV loading and windowing for regularly sampled CGM sequences.

Missing positions are zero-filled only to make finite tensors. The observation
mask remains authoritative; the loader does not invent glucose by interpolation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp cannot be empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


@dataclass(frozen=True)
class CGMSeries:
    """A regularized CGM sequence in mg/dL.

    ``glucose`` contains physical values at observed positions and zero
    placeholders elsewhere. ``observed_mask`` is true only where the source CSV
    contained a measurement.
    """

    timestamps: tuple[datetime, ...]
    glucose: torch.Tensor
    observed_mask: torch.Tensor
    gap_age_minutes: torch.Tensor
    time_of_day: torch.Tensor
    mean: float
    std: float
    interval_minutes: int


def _gap_age(
    observed_mask: np.ndarray, interval_minutes: int, cap_minutes: int = 60
) -> np.ndarray:
    ages = np.empty(observed_mask.shape, dtype=np.float32)
    age = float(cap_minutes)
    for index, observed in enumerate(observed_mask):
        age = 0.0 if observed else min(float(cap_minutes), age + interval_minutes)
        ages[index] = age
    return ages


def _time_of_day(timestamps: tuple[datetime, ...]) -> np.ndarray:
    features = np.empty((len(timestamps), 2), dtype=np.float32)
    for index, timestamp in enumerate(timestamps):
        minutes = (
            timestamp.hour * 60
            + timestamp.minute
            + timestamp.second / 60
            + timestamp.microsecond / 60_000_000
        )
        angle = 2.0 * math.pi * minutes / (24.0 * 60.0)
        features[index] = (math.sin(angle), math.cos(angle))
    return features


def load_cgm_csv(
    path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    glucose_col: str = "glucose",
    interval_minutes: int = 5,
    alignment_tolerance_seconds: float | None = None,
) -> CGMSeries:
    """Load a timestamp-and-glucose CSV onto a regular time grid.

    The earliest timestamp anchors the grid. Later timestamps must fall within
    ``alignment_tolerance_seconds`` of a grid point; the default is 40% of the
    sampling interval. Duplicate grid positions use the last non-empty value.
    Empty glucose cells and absent time points are marked missing.
    """

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")

    csv_path = Path(path)
    rows: list[tuple[datetime, float | None]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing_columns = {timestamp_col, glucose_col}.difference(fields)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required column(s): {missing}")

        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = parse_timestamp(row[timestamp_col])
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc

            cell = row[glucose_col].strip()
            if not cell:
                value = None
            else:
                try:
                    value = float(cell)
                except ValueError as exc:
                    raise ValueError(
                        f"line {line_number}: glucose must be numeric or empty"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"line {line_number}: glucose must be finite")
            rows.append((timestamp, value))

    if not rows:
        raise ValueError("CSV contains no data rows")

    aware = {timestamp.tzinfo is not None for timestamp, _ in rows}
    if len(aware) > 1:
        raise ValueError("timestamps must be either all timezone-aware or all naive")
    rows.sort(key=lambda item: item[0])

    step_seconds = float(interval_minutes * 60)
    tolerance = (
        0.4 * step_seconds
        if alignment_tolerance_seconds is None
        else float(alignment_tolerance_seconds)
    )
    if tolerance < 0 or tolerance >= 0.5 * step_seconds:
        raise ValueError("alignment tolerance must be in [0, half the interval)")

    start = rows[0][0]
    final_offset = (rows[-1][0] - start).total_seconds()
    length = int(round(final_offset / step_seconds)) + 1
    values = np.full(length, np.nan, dtype=np.float32)

    for timestamp, value in rows:
        offset = (timestamp - start).total_seconds()
        index = int(round(offset / step_seconds))
        alignment_error = abs(offset - index * step_seconds)
        if alignment_error > tolerance:
            raise ValueError(
                f"timestamp {timestamp.isoformat()} is {alignment_error:.1f}s "
                "from the nearest grid point"
            )
        if value is not None:
            values[index] = value

    observed = np.isfinite(values)
    if not observed.any():
        raise ValueError("CSV contains no observed glucose values")

    observed_values = values[observed].astype(np.float64)
    mean = float(observed_values.mean())
    std = float(observed_values.std())
    if std < 1e-6:
        std = 1.0

    filled = np.zeros(length, dtype=np.float32)
    filled[observed] = values[observed]
    timestamps = tuple(
        start + timedelta(seconds=index * step_seconds) for index in range(length)
    )

    return CGMSeries(
        timestamps=timestamps,
        glucose=torch.from_numpy(filled),
        observed_mask=torch.from_numpy(observed),
        gap_age_minutes=torch.from_numpy(
            _gap_age(observed, interval_minutes=interval_minutes)
        ),
        time_of_day=torch.from_numpy(_time_of_day(timestamps)),
        mean=mean,
        std=std,
        interval_minutes=interval_minutes,
    )


class CGMWindowDataset(Dataset[dict[str, Any]]):
    """Create fixed-size windows from a :class:`CGMSeries`."""

    def __init__(
        self,
        series: CGMSeries,
        *,
        window_size: int = 288,
        stride: int = 72,
        min_observed: int = 1,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if min_observed < 0 or min_observed > window_size:
            raise ValueError("min_observed must be between 0 and window_size")

        self.series = series
        self.window_size = window_size
        last_start = len(series.glucose) - window_size
        candidates = range(0, last_start + 1, stride) if last_start >= 0 else ()
        self.starts = [
            start
            for start in candidates
            if int(series.observed_mask[start : start + window_size].sum())
            >= min_observed
        ]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = self.starts[index]
        stop = start + self.window_size
        return {
            "glucose": self.series.glucose[start:stop],
            "observed_mask": self.series.observed_mask[start:stop],
            "gap_age_minutes": self.series.gap_age_minutes[start:stop],
            "time_of_day": self.series.time_of_day[start:stop],
            "start_index": start,
        }
