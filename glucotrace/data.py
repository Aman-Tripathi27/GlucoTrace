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

MG_DL_PER_MMOL_L = 18.016
SUPPORTED_UNITS = ("mg/dL", "mmol/L")
# No plausible day of mg/dL readings has a median this low, and no plausible
# day of mmol/L readings has a median this high.
_UNIT_MEDIAN_BOUNDARY = 35.0
INPUT_FORMATS = ("plain", "dexcom-clarity", "libreview")
OUT_OF_RANGE_POLICIES = ("missing", "clamp")
DATE_ORDERS = ("auto", "mdy", "dmy")
# Text a sensor writes instead of a number when a reading is out of range.
OUT_OF_RANGE_TEXT = {"low": "low", "high": "high", "lo": "low", "hi": "high"}
# Dexcom G6/G7 report readings beyond these limits as "Low"/"High".
SENSOR_LOW_MG_DL = 40.0
SENSOR_HIGH_MG_DL = 400.0
# FreeStyle Libre reports "LO" below 40 and "HI" above 500 mg/dL.
SENSOR_LIMITS_MG_DL = {
    "plain": (SENSOR_LOW_MG_DL, SENSOR_HIGH_MG_DL),
    "dexcom-clarity": (SENSOR_LOW_MG_DL, SENSOR_HIGH_MG_DL),
    "libreview": (40.0, 500.0),
}


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
    input_unit: str = "mg/dL"
    out_of_range_count: int = 0
    out_of_range_policy: str = "missing"


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


def _parse_instant(value: str) -> tuple[datetime, timedelta | None]:
    """Return an instant for gridding plus the timestamp's own UTC offset.

    Aware timestamps are normalized to UTC so readings are placed on the grid
    by true elapsed time; the original offset is kept so time-of-day can use
    the wearer's local clock. Naive timestamps are local clock time already.
    """

    parsed = parse_timestamp(value)
    if parsed.tzinfo is None:
        return parsed, None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return parsed, datetime.fromisoformat(text).utcoffset()


def _parse_glucose_cell(cell: str, line_number: int) -> tuple[float | None, str | None]:
    """Return (value, out-of-range flag). ``Low``/``High`` text has no value."""

    text = cell.strip()
    if not text:
        return None, None
    lowered = text.lower()
    if lowered in OUT_OF_RANGE_TEXT:
        return None, OUT_OF_RANGE_TEXT[lowered]
    if "," in text and "." not in text:
        text = text.replace(",", ".")  # decimal comma, as in many European exports
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"line {line_number}: glucose must be numeric, empty, Low, or High"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"line {line_number}: glucose must be finite")
    return value, None


_Row = tuple[datetime, "timedelta | None", "float | None", "str | None"]


def _read_plain_rows(
    handle: Any, timestamp_col: str, glucose_col: str
) -> list[_Row]:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames or []
    missing_columns = {timestamp_col, glucose_col}.difference(fields)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")
    rows: list[_Row] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            instant, offset = _parse_instant(row[timestamp_col])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        value, flag = _parse_glucose_cell(row[glucose_col], line_number)
        rows.append((instant, offset, value, flag))
    return rows


def _read_dexcom_clarity_rows(handle: Any) -> tuple[list[_Row], str]:
    """Read glucose (EGV) rows and the unit from a Dexcom Clarity CSV export.

    Clarity exports start with patient and device rows that have no
    timestamp, and mix in alerts, calibrations, insulin, and carbs. Only
    estimated glucose value (EGV) rows are glucose readings.
    """

    reader = csv.DictReader(handle)
    fields = reader.fieldnames or []
    timestamp_col = next((f for f in fields if f.startswith("Timestamp")), None)
    glucose_col = next((f for f in fields if f.startswith("Glucose Value")), None)
    if timestamp_col is None or glucose_col is None or "Event Type" not in fields:
        raise ValueError(
            "not a Dexcom Clarity export: expected 'Timestamp (...)', "
            "'Event Type', and 'Glucose Value (...)' columns"
        )
    if "mmol/L" in glucose_col:
        unit = "mmol/L"
    elif "mg/dL" in glucose_col:
        unit = "mg/dL"
    else:
        raise ValueError(f"cannot read the glucose unit from column {glucose_col!r}")
    rows: list[_Row] = []
    for line_number, row in enumerate(reader, start=2):
        if (row.get("Event Type") or "").strip() != "EGV":
            continue
        try:
            instant, offset = _parse_instant(row[timestamp_col] or "")
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        value, flag = _parse_glucose_cell(row[glucose_col] or "", line_number)
        rows.append((instant, offset, value, flag))
    return rows, unit


_LIBRE_TIMESTAMP = "Device Timestamp"
_LIBRE_HISTORIC = "0"  # record type of automatic sensor readings


def _split_libre_stamp(text: str) -> tuple[list[int], str]:
    """Split ``03-01-2024 08:05 AM`` into ([3, 1, 2024], "08:05 AM")."""

    date_part, _, time_part = text.strip().partition(" ")
    pieces = date_part.replace("/", "-").replace(".", "-").split("-")
    if len(pieces) != 3 or not all(piece.isdigit() for piece in pieces):
        raise ValueError(f"unrecognized LibreView timestamp {text!r}")
    return [int(piece) for piece in pieces], time_part.strip()


def _libre_date_order(stamps: list[str]) -> str:
    """Decide whether dates are month-day-year or day-month-year.

    Year-first dates are unambiguous. Otherwise a first field above 12 means
    day-month, a second field above 12 means month-day, and an AM/PM clock is
    the US month-day style. If nothing decides it, the file is rejected rather
    than guessed, because swapping days and months silently corrupts every day.
    """

    first_above = second_above = ampm = False
    for stamp in stamps:
        fields, clock = _split_libre_stamp(stamp)
        if fields[0] > 31:
            return "ymd"
        first_above |= fields[0] > 12
        second_above |= fields[1] > 12
        ampm |= clock.upper().endswith(("AM", "PM"))
    if first_above and second_above:
        raise ValueError("LibreView dates are inconsistent (neither MM-DD nor DD-MM)")
    if first_above:
        return "dmy"
    if second_above or ampm:
        return "mdy"
    raise ValueError(
        "cannot tell whether LibreView dates are month-day or day-month; "
        "pass --date-order mdy or --date-order dmy"
    )


def _parse_libre_stamp(text: str, order: str) -> datetime:
    fields, clock = _split_libre_stamp(text)
    if order == "ymd":
        year, month, day = fields
    elif order == "mdy":
        month, day, year = fields
    else:
        day, month, year = fields
    upper = clock.upper()
    meridiem = upper[-2:] if upper.endswith(("AM", "PM")) else None
    hours_text, _, minutes_text = (upper[:-2] if meridiem else upper).strip().partition(":")
    try:
        hour, minute = int(hours_text), int(minutes_text[:2])
    except ValueError as exc:
        raise ValueError(f"unrecognized LibreView time {text!r}") from exc
    if meridiem:
        if not 1 <= hour <= 12:
            raise ValueError(f"unrecognized LibreView time {text!r}")
        hour = hour % 12 + (12 if meridiem == "PM" else 0)
    return datetime(year, month, day, hour, minute)


def _read_libreview_rows(handle: Any, date_order: str) -> tuple[list[_Row], str]:
    """Read historic glucose rows and the unit from a LibreView CSV export.

    LibreView exports start with a title line, then a header that includes
    ``Device Timestamp``, ``Record Type``, and ``Historic Glucose <unit>``.
    Only record type 0 (automatic historic readings, usually every 15
    minutes) is used; scans, strips, food, insulin, and notes are skipped.
    Timestamps are the device's local clock.
    """

    lines = list(csv.reader(handle))
    header_index = next(
        (i for i, line in enumerate(lines) if _LIBRE_TIMESTAMP in line), None
    )
    if header_index is None:
        raise ValueError(
            "not a LibreView export: expected a 'Device Timestamp' column"
        )
    header = lines[header_index]
    glucose_col = next((c for c in header if c.startswith("Historic Glucose")), None)
    if glucose_col is None or "Record Type" not in header:
        raise ValueError(
            "not a LibreView export: expected 'Record Type' and "
            "'Historic Glucose ...' columns"
        )
    if "mmol/L" in glucose_col:
        unit = "mmol/L"
    elif "mg/dL" in glucose_col:
        unit = "mg/dL"
    else:
        raise ValueError(f"cannot read the glucose unit from column {glucose_col!r}")
    stamp_at = header.index(_LIBRE_TIMESTAMP)
    type_at = header.index("Record Type")
    glucose_at = header.index(glucose_col)
    historic = [
        (line_number, line)
        for line_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2)
        if len(line) > max(stamp_at, type_at, glucose_at)
        and line[type_at].strip() == _LIBRE_HISTORIC
    ]
    if not historic:
        return [], unit
    order = (
        _libre_date_order([line[stamp_at] for _, line in historic])
        if date_order == "auto"
        else date_order
    )
    rows: list[_Row] = []
    for line_number, line in historic:
        try:
            instant = _parse_libre_stamp(line[stamp_at], order)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        value, flag = _parse_glucose_cell(line[glucose_at], line_number)
        rows.append((instant, None, value, flag))
    return rows, unit


def load_cgm_csv(
    path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    glucose_col: str = "glucose",
    interval_minutes: int = 5,
    alignment_tolerance_seconds: float | None = None,
    unit: str | None = None,
    input_format: str = "plain",
    out_of_range: str = "missing",
    date_order: str = "auto",
) -> CGMSeries:
    """Load a CGM CSV onto a regular time grid.

    The earliest reading anchors the grid. Later readings must fall within
    ``alignment_tolerance_seconds`` of a grid point; the default is 40% of the
    sampling interval. Duplicate grid positions use the last non-empty value.
    Empty glucose cells and absent time points are marked missing.

    Timezone-aware timestamps are placed on the grid by true elapsed time,
    while time-of-day features use each reading's own local clock, so
    ``08:00+05:30`` is morning, not 02:30 UTC. Files may mix offsets (for
    example across a daylight-saving change) but not aware and naive stamps.

    ``input_format`` is ``"plain"`` (``timestamp_col``/``glucose_col``),
    ``"dexcom-clarity"`` (a Clarity CSV export), or ``"libreview"`` (a
    FreeStyle Libre LibreView CSV export). Both exports declare their unit.
    LibreView dates are read as month-day or day-month automatically when the
    file makes it clear; otherwise pass ``date_order`` ("mdy" or "dmy"). ``unit`` declares a plain file's unit (default mg/dL); mmol/L is
    converted to mg/dL. A file whose median contradicts the unit is rejected.

    Readings recorded as ``Low``/``High`` (beyond the sensor's range) have no
    measured value. ``out_of_range="missing"`` marks them missing;
    ``"clamp"`` records them at the sensor limits (Dexcom 40 and 400 mg/dL,
    FreeStyle Libre 40 and 500 mg/dL). The
    count is returned as ``out_of_range_count``.
    """

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if unit is not None and unit not in SUPPORTED_UNITS:
        raise ValueError(f"unit must be one of {SUPPORTED_UNITS}")
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"input_format must be one of {INPUT_FORMATS}")
    if out_of_range not in OUT_OF_RANGE_POLICIES:
        raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE_POLICIES}")
    if date_order not in DATE_ORDERS:
        raise ValueError(f"date_order must be one of {DATE_ORDERS}")

    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        if input_format in ("dexcom-clarity", "libreview"):
            if input_format == "dexcom-clarity":
                rows, declared_unit = _read_dexcom_clarity_rows(handle)
            else:
                rows, declared_unit = _read_libreview_rows(handle, date_order)
            if unit is not None and unit != declared_unit:
                raise ValueError(
                    f"the {input_format} export declares {declared_unit}, not {unit}"
                )
            unit = declared_unit
        else:
            rows = _read_plain_rows(handle, timestamp_col, glucose_col)
    unit = unit or "mg/dL"

    if not rows:
        raise ValueError("CSV contains no data rows")

    aware = {instant.tzinfo is not None for instant, _, _, _ in rows}
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
    flags: list[str | None] = [None] * length
    offsets: list[timedelta | None] = [None] * length

    for instant, utc_offset, value, flag in rows:
        offset = (instant - start).total_seconds()
        index = int(round(offset / step_seconds))
        alignment_error = abs(offset - index * step_seconds)
        if alignment_error > tolerance:
            raise ValueError(
                f"timestamp {instant.isoformat()} is {alignment_error:.1f}s "
                "from the nearest grid point"
            )
        offsets[index] = utc_offset
        if value is not None:
            values[index] = value
            flags[index] = None
        elif flag is not None and not np.isfinite(values[index]):
            flags[index] = flag

    out_of_range_positions = [
        (index, flag) for index, flag in enumerate(flags) if flag is not None
    ]
    measured = np.isfinite(values)
    if measured.any():
        median = float(np.median(values[measured]))
        if unit == "mg/dL" and median < _UNIT_MEDIAN_BOUNDARY:
            raise ValueError(
                f"median glucose {median:g} is implausible for mg/dL; the file "
                "looks like mmol/L (pass unit='mmol/L' or --unit mmol/L)"
            )
        if unit == "mmol/L" and median >= _UNIT_MEDIAN_BOUNDARY:
            raise ValueError(
                f"median glucose {median:g} is implausible for mmol/L; the file "
                "looks like mg/dL"
            )
    if unit == "mmol/L":
        values = values * np.float32(MG_DL_PER_MMOL_L)
    if out_of_range == "clamp":
        low, high = SENSOR_LIMITS_MG_DL[input_format]
        for index, flag in out_of_range_positions:
            values[index] = low if flag == "low" else high

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
    grid = [start + timedelta(seconds=index * step_seconds) for index in range(length)]
    if start.tzinfo is not None:
        # Carry each reading's offset to the empty positions that follow it.
        current = next(offset for offset in offsets if offset is not None)
        local = []
        for instant, utc_offset in zip(grid, offsets):
            current = utc_offset if utc_offset is not None else current
            local.append(instant.astimezone(timezone(current)))
        grid = local
    timestamps = tuple(grid)

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
        input_unit=unit,
        out_of_range_count=len(out_of_range_positions),
        out_of_range_policy=out_of_range,
    )


def thin_to_cadence(
    glucose: torch.Tensor,
    observed_mask: torch.Tensor,
    step: int = 3,
    *,
    generator: torch.Generator | None = None,
    rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep every ``step``-th grid position of each selected row.

    Each thinned row keeps positions with one residue modulo ``step``, chosen at
    random so the phase carries no information. Rows whose observations
    already share one residue (for example 15-minute sensors on a five-minute
    grid) are left unchanged. ``rows`` optionally selects which rows to thin.
    """

    if glucose.ndim != 2 or observed_mask.shape != glucose.shape:
        raise ValueError("glucose and observed_mask must have shape [batch, time]")
    if step < 2:
        raise ValueError("step must be at least 2")
    mask = observed_mask.bool().clone()
    positions = torch.arange(mask.shape[1])
    selected = torch.ones(mask.shape[0], dtype=torch.bool) if rows is None else rows.bool()
    for row in range(mask.shape[0]):
        observed = positions[mask[row]]
        if not selected[row] or observed.numel() == 0:
            continue
        if torch.unique(observed.remainder(step)).numel() == 1:
            continue
        offset = int(torch.randint(step, (1,), generator=generator))
        kept = mask[row] & positions.remainder(step).eq(offset)
        if kept.any():
            mask[row] = kept
    return glucose.masked_fill(~mask, 0.0), mask


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
