import csv
from datetime import datetime, timedelta
from pathlib import Path

from glucofm.adapters.big_ideas import (
    EVENT_COLUMN,
    GLUCOSE_COLUMN,
    TIMESTAMP_COLUMN,
    load_big_ideas,
    prepare_big_ideas,
    read_big_ideas_file,
)


FIELDNAMES = [
    "Index",
    TIMESTAMP_COLUMN,
    EVENT_COLUMN,
    "Event Subtype",
    "Patient Info",
    "Device Info",
    "Source Device ID",
    GLUCOSE_COLUMN,
    "Insulin Value (u)",
    "Carb Value (grams)",
    "Duration (hh:mm:ss)",
    "Glucose Rate of Change (mg/dL/min)",
    "Transmitter Time (Long Integer)",
]


def write_source_shaped_fixture(root: Path) -> Path:
    participant_dir = root / "001"
    participant_dir.mkdir(parents=True)
    path = participant_dir / "Dexcom_001.csv"
    start = datetime(2020, 2, 13, 17, 23, 32)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {"Index": 1, EVENT_COLUMN: "FirstName", "Patient Info": "2019"}
        )
        writer.writerow(
            {"Index": 2, EVENT_COLUMN: "Alert", "Event Subtype": "High"}
        )
        for index in range(288):
            if index == 20:
                continue
            writer.writerow(
                {
                    "Index": index + 13,
                    TIMESTAMP_COLUMN: (
                        start + timedelta(minutes=5 * index)
                    ).isoformat(sep=" "),
                    EVENT_COLUMN: "EGV",
                    GLUCOSE_COLUMN: str(80 + index % 40),
                }
            )
    return path


def test_reads_only_egv_rows_from_native_layout(tmp_path: Path) -> None:
    path = write_source_shaped_fixture(tmp_path)

    participant_id, readings = read_big_ideas_file(path)

    assert participant_id == "001"
    assert len(readings) == 287
    assert readings[0].glucose_mg_dl == 80


def test_loads_canonical_big_ideas_day(tmp_path: Path) -> None:
    write_source_shaped_fixture(tmp_path)

    days = load_big_ideas(tmp_path, min_observed_fraction=0.9)

    assert len(days) == 1
    assert days[0].provenance.dataset == "BIG_IDEAs"
    assert days[0].provenance.device == "Dexcom G6"
    assert days[0].observed_mask.sum() == 287
    assert days[0].gap_age_minutes[20] == 5


def test_prepare_writes_local_canonical_corpus(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_source_shaped_fixture(source)

    manifest = prepare_big_ideas(source, tmp_path / "prepared")

    assert manifest.name == "manifest.json"
    assert len(list((manifest.parent / "days").glob("*.csv"))) == 1
