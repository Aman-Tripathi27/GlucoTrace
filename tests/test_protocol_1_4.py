"""Random-phase training windows and the no-clock model option."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

from glucotrace import phase
from glucotrace.canonical import CGMReading, SourceProvenance
from glucotrace.model import GlucoTrace, GlucoTraceConfig
from glucotrace.phase import RandomPhaseCGMDataset, Recording


def recording(dataset: str, participant: str, start: datetime, hours: int) -> Recording:
    readings = tuple(
        CGMReading(start + timedelta(minutes=5 * index), 100.0 + index % 40)
        for index in range(hours * 12)
    )
    return Recording(
        dataset=dataset,
        participant_id=participant,
        readings=readings,
        timestamps=tuple(item.timestamp for item in readings),
        provenance=SourceProvenance(
            dataset=dataset, version="1", participant_id=participant, device="d",
            unit="mg/dL", source_url="https://example.test", license_name="l",
            license_url="https://example.test/l",
        ),
    )


def dataset(seed: int = 7) -> RandomPhaseCGMDataset:
    return RandomPhaseCGMDataset(
        [
            recording("A", "a1", datetime(2026, 1, 1, 0, 0), 48),
            recording("B", "b1", datetime(2026, 1, 1, 0, 2), 48),
        ],
        samples_per_epoch=40,
        seed=seed,
    )


def test_windows_are_full_days_alternating_sources() -> None:
    data = dataset()
    items = [data[index] for index in range(6)]
    assert [item["dataset"] for item in items] == ["A", "B"] * 3
    assert all(item["glucose"].shape == (288,) for item in items)
    assert all(item["observed_mask"].all() for item in items)


def test_draws_are_reproducible_and_change_each_epoch() -> None:
    first, second = dataset(), dataset()
    assert torch.equal(first[3]["glucose"], second[3]["glucose"])
    phases = {tuple(first[index]["time_of_day"][0].tolist()) for index in range(40)}
    assert len(phases) > 10
    before = first[3]["time_of_day"][0].clone()
    first.set_epoch(1)
    assert not torch.equal(before, first[3]["time_of_day"][0])


def test_short_recordings_are_rejected() -> None:
    with pytest.raises(ValueError, match="long enough"):
        RandomPhaseCGMDataset(
            [recording("A", "a1", datetime(2026, 1, 1), 12)], samples_per_epoch=4
        )


def test_only_training_participants_are_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"splits": {
        "train": {"participants": [{"dataset": "A", "participant_id": "keep"}]},
        "validation": {"participants": [{"dataset": "A", "participant_id": "val"}]},
        "test": {"participants": [{"dataset": "A", "participant_id": "test"}]},
    }}), encoding="utf-8")

    def fake_reader(dataset_name: str, raw_path: Path):
        return [
            (pid, list(recording("A", pid, datetime(2026, 1, 1), 30).readings),
             recording("A", pid, datetime(2026, 1, 1), 30).provenance)
            for pid in ("keep", "val", "test")
        ]

    monkeypatch.setattr(phase, "_read_participants", fake_reader)
    loaded = phase.load_training_recordings([("A", tmp_path)], [split])
    assert {item.participant_id for item in loaded} == {"keep"}


def test_no_clock_model_ignores_time_of_day() -> None:
    torch.manual_seed(0)
    config = GlucoTraceConfig(
        hidden_size=16, num_layers=1, num_heads=4, feedforward_size=32, dropout=0.0,
        patch_size=3, trend_windows=(1, 3, 6), event_trend_window=3, max_patches=8,
        use_clock=False,
    )
    model = GlucoTrace(config).eval()
    glucose = 100 + 20 * torch.rand(2, 24)
    mask = torch.ones(2, 24, dtype=torch.bool)
    morning = torch.zeros(2, 24, 2)
    evening = torch.randn(2, 24, 2)
    with torch.no_grad():
        first = model(glucose, mask, time_of_day=morning)["pooled_embedding"]
        second = model(glucose, mask, time_of_day=evening)["pooled_embedding"]
    assert torch.equal(first, second)
    assert GlucoTraceConfig().use_clock is True
