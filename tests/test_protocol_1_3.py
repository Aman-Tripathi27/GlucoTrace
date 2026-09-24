import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from glucofm.canonical import CGMReading, build_24h_windows
from glucofm.corpus import carry_split_membership, create_participant_split
from test_canonical import provenance
from test_inference import make_manifest


def readings(start: datetime, hours: int) -> list[CGMReading]:
    return [
        CGMReading(start + timedelta(minutes=5 * index), 100.0 + index % 30)
        for index in range(hours * 12)
    ]


def test_midnight_anchor_starts_after_midnight_on_the_device_grid() -> None:
    # Device phase is 2 min 30 s past each five-minute mark: a strict 00:00
    # grid would put every reading 150 s off-grid.
    trace = readings(datetime(2026, 1, 1, 17, 22, 30), hours=60)
    days = build_24h_windows(trace, provenance(), anchor="midnight")
    assert [day.start_time for day in days] == [
        datetime(2026, 1, 2, 0, 2, 30),
        datetime(2026, 1, 3, 0, 2, 30),
    ]
    assert all(day.observed_fraction == 1.0 for day in days)

    default = build_24h_windows(trace, provenance())
    assert default[0].start_time == datetime(2026, 1, 1, 17, 22, 30)


def test_midnight_anchor_keeps_a_reading_exactly_at_midnight() -> None:
    trace = readings(datetime(2026, 1, 1, 0, 0), hours=24)
    days = build_24h_windows(trace, provenance(), anchor="midnight")
    assert [day.start_time for day in days] == [datetime(2026, 1, 1)]


def test_unknown_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="anchor"):
        build_24h_windows(readings(datetime(2026, 1, 1), 24), provenance(), anchor="noon")


def test_carried_split_keeps_exact_membership(tmp_path: Path) -> None:
    parent_manifest = make_manifest(tmp_path / "old", participant_count=6)
    parent_split = create_participant_split(parent_manifest, seed=3)
    # A re-windowed corpus of the same participants, one of whom lost all days.
    new_manifest = make_manifest(tmp_path / "new", participant_count=6)
    payload = json.loads(new_manifest.read_text(encoding="utf-8"))
    parent = json.loads(parent_split.read_text(encoding="utf-8"))
    removed = parent["splits"]["train"]["participants"][0]["participant_id"]
    payload["records"] = [r for r in payload["records"] if r["participant_id"] != removed]
    new_manifest.write_text(json.dumps(payload), encoding="utf-8")

    carried = carry_split_membership(new_manifest, parent_split, tmp_path / "carried.json")
    result = json.loads(carried.read_text(encoding="utf-8"))
    for name in ("validation", "test"):
        assert result["splits"][name]["participants"] == parent["splits"][name]["participants"]
    assert result["dropped_participants"] == [
        {"dataset": "tiny", "participant_id": removed, "split": "train"}
    ]
    with pytest.raises(FileExistsError):
        carry_split_membership(new_manifest, parent_split, carried)


def test_carried_split_rejects_new_participants(tmp_path: Path) -> None:
    parent_manifest = make_manifest(tmp_path / "old", participant_count=4)
    parent_split = create_participant_split(parent_manifest, seed=3)
    bigger = make_manifest(tmp_path / "new", participant_count=5)
    with pytest.raises(ValueError, match="absent from the parent split"):
        carry_split_membership(bigger, parent_split, tmp_path / "out.json")
