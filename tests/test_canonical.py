import json
from datetime import datetime, timedelta
from pathlib import Path

import torch

from glucofm.canonical import (
    CGMReading,
    SourceProvenance,
    build_24h_windows,
    split_continuous_segments,
    write_canonical_corpus,
)


def provenance() -> SourceProvenance:
    return SourceProvenance(
        dataset="test-source",
        version="1.0",
        participant_id="p001",
        device="test-device",
        unit="mg/dL",
        source_url="https://example.test/data",
        license_name="test license",
        license_url="https://example.test/license",
    )


def full_day(*, missing_index: int | None = None) -> list[CGMReading]:
    start = datetime(2026, 1, 1, 6, 2, 30)
    return [
        CGMReading(start + timedelta(minutes=5 * index), 90.0 + index % 20)
        for index in range(288)
        if index != missing_index
    ]


def test_build_day_keeps_physical_mask_and_gap_age() -> None:
    days = build_24h_windows(
        full_day(missing_index=10), provenance(), min_observed_fraction=0.9
    )

    assert len(days) == 1
    day = days[0]
    assert day.glucose.shape == (288,)
    assert day.time_of_day.shape == (288, 2)
    assert day.observed_mask.sum() == 287
    assert not day.observed_mask[10]
    assert day.glucose[10] == 0
    assert day.gap_age_minutes[10] == 5
    assert day.gap_age_minutes[11] == 0
    assert torch.allclose(
        day.time_of_day.square().sum(dim=1), torch.ones(288), atol=1e-6
    )


def test_long_gap_splits_segments_and_prevents_spanning_window() -> None:
    start = datetime(2026, 1, 1)
    readings = [
        CGMReading(start + timedelta(minutes=5 * index), 100.0)
        for index in range(100)
    ]
    readings += [
        CGMReading(start + timedelta(hours=10, minutes=5 * index), 105.0)
        for index in range(100)
    ]

    assert len(split_continuous_segments(readings)) == 2
    assert build_24h_windows(readings, provenance()) == []


def test_write_canonical_corpus_records_checksum_and_provenance(
    tmp_path: Path,
) -> None:
    day = build_24h_windows(full_day(), provenance())[0]

    manifest_path = write_canonical_corpus([day], tmp_path / "canonical")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["records"][0]

    assert manifest["schema_version"] == "1.0"
    assert manifest["window_positions"] == 288
    assert record["provenance"]["participant_id"] == "p001"
    assert len(record["sha256"]) == 64
    assert (manifest_path.parent / record["file"]).exists()
