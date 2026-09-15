import csv
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from glucofm.adapters.colas import (
    GLUCOSE_COLUMN,
    INDEX_COLUMN,
    TIME_COLUMN,
    load_colas,
    prepare_colas,
    read_colas_case,
)


def write_colas_case(
    root: Path,
    *,
    participant: int = 1,
    days: int = 2,
    first_index: int = 1,
    missing_position: int | None = 20,
    clock_shift_at: int | None = None,
    clock_shift_seconds: int = 0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case  {participant}.csv"
    clock_anchor = datetime(2000, 1, 1, 0, 0, 14)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[INDEX_COLUMN, TIME_COLUMN, GLUCOSE_COLUMN]
        )
        writer.writeheader()
        for position in range(days * 288):
            clock = clock_anchor + timedelta(minutes=5 * position)
            if clock_shift_at is not None and position >= clock_shift_at:
                clock += timedelta(seconds=clock_shift_seconds)
            writer.writerow(
                {
                    INDEX_COLUMN: first_index + position,
                    TIME_COLUMN: clock.strftime("%H:%M:%S"),
                    GLUCOSE_COLUMN: (
                        "NA"
                        if position == missing_position
                        else 90 + position % 30
                    ),
                }
            )
    return path


def test_reads_time_only_case_on_relative_five_minute_grid(tmp_path: Path) -> None:
    path = write_colas_case(tmp_path, days=1)

    participant_id, readings = read_colas_case(path)

    assert participant_id == "001"
    assert len(readings) == 287
    assert readings[0].timestamp == datetime(2000, 1, 1, 0, 0, 14)
    assert readings[20].timestamp == datetime(2000, 1, 1, 1, 45, 14)
    assert readings[20].glucose_mg_dl == 111


def test_accepts_source_index_starting_on_second_day(tmp_path: Path) -> None:
    path = write_colas_case(
        tmp_path, participant=197, days=1, first_index=289
    )

    participant_id, readings = read_colas_case(path)

    assert participant_id == "197"
    assert readings[0].timestamp.year == 2000


def test_rejects_clock_drift_over_one_sampling_interval(tmp_path: Path) -> None:
    path = write_colas_case(
        tmp_path,
        days=1,
        clock_shift_at=100,
        clock_shift_seconds=301,
    )

    with pytest.raises(ValueError, match="clock differs"):
        read_colas_case(path)


def test_loads_official_shaped_zip_and_writes_manifest(tmp_path: Path) -> None:
    case = write_colas_case(tmp_path / "source", days=2)
    archive = tmp_path / "pone.0225817.s001.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(case, arcname=f"S1/{case.name}")

    days = load_colas(archive)
    manifest_path = prepare_colas(archive, tmp_path / "prepared")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(days) == 2
    assert days[0].provenance.dataset == "Colas2019"
    assert days[0].provenance.device == "Medtronic MiniMed iPro CGMS"
    assert days[0].observed_mask.sum() == 287
    assert days[0].gap_age_minutes[20] == 5
    assert len(manifest["records"]) == 2
    assert manifest["records"][0]["provenance"]["license_name"].startswith(
        "Creative Commons Attribution"
    )


def test_rejects_partial_native_day(tmp_path: Path) -> None:
    path = write_colas_case(tmp_path, days=1)
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="whole number"):
        read_colas_case(path)
