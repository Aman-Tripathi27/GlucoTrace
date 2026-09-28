"""Timezone-aware time of day, Dexcom Clarity exports, and Low/High readings."""

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from glucofm.commands import encode_main
from glucofm.data import SENSOR_HIGH_MG_DL, SENSOR_LOW_MG_DL, load_cgm_csv
from glucofm.inference import ResearchEncoder
from test_inference import make_checkpoint

CLARITY_HEADER = [
    "Index",
    "Timestamp (YYYY-MM-DDThh:mm:ss)",
    "Event Type",
    "Event Subtype",
    "Patient Info",
    "Device Info",
    "Source Device ID",
    "Glucose Value ({unit})",
    "Insulin Value (u)",
    "Carb Value (grams)",
    "Duration (hh:mm:ss)",
    "Glucose Rate of Change ({unit}/min)",
    "Transmitter Time (Long Integer)",
    "Transmitter ID",
]


def write_plain(path: Path, stamps: list[str], values: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "glucose"))
        writer.writerows(zip(stamps, values))
    return path


def local_stamps(start: datetime, count: int, suffix: str = "") -> list[str]:
    return [
        (start + timedelta(minutes=5 * index)).isoformat() + suffix
        for index in range(count)
    ]


def write_clarity(path: Path, values: list[str], *, unit: str = "mg/dL") -> Path:
    """A synthetic Clarity-style export: metadata rows, events, and EGVs."""

    header = [column.format(unit=unit) for column in CLARITY_HEADER]
    start = datetime(2026, 3, 1, 8, 1, 7)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        blank = [""] * len(header)
        for index, (field, value) in enumerate(
            [("FirstName", "Test"), ("LastName", "Person"), ("Device", "G6")], start=1
        ):
            row = list(blank)
            row[0], row[2], row[4] = str(index), field, value
            writer.writerow(row)
        row = list(blank)
        row[0], row[1], row[2], row[3] = "4", start.isoformat(), "Alert", "High"
        writer.writerow(row)
        for index, value in enumerate(values):
            row = list(blank)
            row[0] = str(index + 5)
            row[1] = (start + timedelta(minutes=5 * index)).isoformat()
            row[2] = "EGV"
            row[7] = value
            writer.writerow(row)
        row = list(blank)
        row[0], row[1], row[2], row[7] = "999", start.isoformat(), "Calibration", "999"
        writer.writerow(row)
    return path


# Bug 1: timezone-aware timestamps keep the wearer's local time of day.


def test_offset_timestamps_use_local_time_of_day(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, 8, 0)
    values = [str(100 + index % 7) for index in range(24)]
    naive = load_cgm_csv(write_plain(tmp_path / "n.csv", local_stamps(start, 24), values))
    india = load_cgm_csv(
        write_plain(tmp_path / "a.csv", local_stamps(start, 24, "+05:30"), values)
    )
    assert india.timestamps[0].strftime("%H:%M") == "08:00"
    assert india.timestamps[0].utcoffset() == timedelta(hours=5, minutes=30)
    assert torch.equal(india.time_of_day, naive.time_of_day)
    assert torch.equal(india.glucose, naive.glucose)


def test_utc_z_suffix_uses_utc_clock(tmp_path: Path) -> None:
    stamps = local_stamps(datetime(2026, 1, 1, 23, 50), 4, "Z")
    series = load_cgm_csv(write_plain(tmp_path / "z.csv", stamps, ["100"] * 4))
    assert [stamp.strftime("%H:%M") for stamp in series.timestamps] == [
        "23:50", "23:55", "00:00", "00:05"
    ]


def test_daylight_saving_change_follows_local_clock(tmp_path: Path) -> None:
    # US fall-back: 01:55-04:00 is followed five minutes later by 01:00-05:00.
    edt, est = timezone(timedelta(hours=-4)), timezone(timedelta(hours=-5))
    before = [datetime(2026, 11, 1, 1, 50, tzinfo=edt), datetime(2026, 11, 1, 1, 55, tzinfo=edt)]
    after = [datetime(2026, 11, 1, 1, 0, tzinfo=est), datetime(2026, 11, 1, 1, 5, tzinfo=est)]
    stamps = [stamp.isoformat() for stamp in before + after]
    series = load_cgm_csv(write_plain(tmp_path / "dst.csv", stamps, ["100"] * 4))
    assert len(series.glucose) == 4
    assert [stamp.strftime("%H:%M") for stamp in series.timestamps] == [
        "01:50", "01:55", "01:00", "01:05"
    ]


def test_fingerprint_is_identical_with_or_without_offsets(tmp_path: Path) -> None:
    encoder = ResearchEncoder.load(make_checkpoint(tmp_path / "model.pt"))
    start = datetime(2026, 1, 1, 7, 0)
    values = [str(95 + (index * 7) % 30) for index in range(12)]
    naive = write_plain(tmp_path / "n.csv", local_stamps(start, 12), values)
    aware = write_plain(tmp_path / "a.csv", local_stamps(start, 12, "+05:30"), values)
    first, _ = encoder.encode_csv(naive)
    second, metadata = encoder.encode_csv(aware)
    assert torch.equal(first, second)
    assert metadata["start_time"].startswith("2026-01-01T07:00:00+05:30")


# Bug 3: Dexcom Clarity exports and Low/High readings.


def test_clarity_export_reads_only_glucose_readings(tmp_path: Path) -> None:
    path = write_clarity(tmp_path / "clarity.csv", ["101", "104", "", "110"])
    series = load_cgm_csv(path, input_format="dexcom-clarity")
    assert series.input_unit == "mg/dL"
    assert series.glucose.tolist() == [101.0, 104.0, 0.0, 110.0]
    assert series.observed_mask.tolist() == [True, True, False, True]
    assert series.timestamps[0] == datetime(2026, 3, 1, 8, 1, 7)


def test_clarity_mmol_export_is_converted_and_checked(tmp_path: Path) -> None:
    path = write_clarity(tmp_path / "mmol.csv", ["5.5", "6.0"], unit="mmol/L")
    series = load_cgm_csv(path, input_format="dexcom-clarity")
    assert series.input_unit == "mmol/L"
    assert series.glucose[0].item() == pytest.approx(5.5 * 18.016)
    with pytest.raises(ValueError, match="declares mmol/L"):
        load_cgm_csv(path, input_format="dexcom-clarity", unit="mg/dL")


def test_plain_file_is_not_mistaken_for_clarity(tmp_path: Path) -> None:
    path = write_plain(tmp_path / "p.csv", local_stamps(datetime(2026, 1, 1), 2), ["1", "2"])
    with pytest.raises(ValueError, match="not a Dexcom Clarity export"):
        load_cgm_csv(path, input_format="dexcom-clarity")


def test_low_and_high_are_missing_by_default(tmp_path: Path) -> None:
    path = write_clarity(tmp_path / "lh.csv", ["Low", "90", "High", "120", "low"])
    series = load_cgm_csv(path, input_format="dexcom-clarity")
    assert series.out_of_range_count == 3
    assert series.out_of_range_policy == "missing"
    assert series.observed_mask.tolist() == [False, True, False, True, False]
    assert series.glucose.tolist() == [0.0, 90.0, 0.0, 120.0, 0.0]


def test_low_and_high_can_be_clamped_to_sensor_limits(tmp_path: Path) -> None:
    path = write_clarity(tmp_path / "lh.csv", ["Low", "90", "High"])
    series = load_cgm_csv(path, input_format="dexcom-clarity", out_of_range="clamp")
    assert series.glucose.tolist() == [SENSOR_LOW_MG_DL, 90.0, SENSOR_HIGH_MG_DL]
    assert series.observed_mask.all()

    mmol = write_clarity(tmp_path / "mmol.csv", ["Low", "6.0", "High"], unit="mmol/L")
    series = load_cgm_csv(mmol, input_format="dexcom-clarity", out_of_range="clamp")
    assert series.glucose[0].item() == SENSOR_LOW_MG_DL
    assert series.glucose[2].item() == SENSOR_HIGH_MG_DL


def test_plain_csv_accepts_low_high_and_rejects_other_text(tmp_path: Path) -> None:
    stamps = local_stamps(datetime(2026, 1, 1), 3)
    series = load_cgm_csv(write_plain(tmp_path / "ok.csv", stamps, ["HIGH", "100", "105"]))
    assert series.out_of_range_count == 1
    with pytest.raises(ValueError, match="numeric, empty, Low, or High"):
        load_cgm_csv(write_plain(tmp_path / "bad.csv", stamps, ["n/a", "100", "105"]))
    with pytest.raises(ValueError, match="out_of_range"):
        load_cgm_csv(tmp_path / "ok.csv", out_of_range="drop")


def test_encode_warns_about_out_of_range_readings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    values = ["Low"] + [str(100 + index) for index in range(11)]
    path = write_clarity(tmp_path / "day.csv", values)
    argv = sys.argv
    sys.argv = [
        "glucofm-encode", str(path), "--format", "dexcom-clarity",
        "--checkpoint", str(checkpoint),
    ]
    try:
        encode_main()
    finally:
        sys.argv = argv
    captured = capsys.readouterr()
    assert "1 Low/High reading(s)" in captured.err
    assert '"out_of_range_readings": 1' in captured.out
