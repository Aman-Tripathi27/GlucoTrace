from pathlib import Path

import pytest
import torch

from glucofm.data import CGMWindowDataset, load_cgm_csv


def write_csv(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_regularizes_gap_and_keeps_mask(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "cgm.csv",
        "timestamp,glucose\n"
        "2026-01-01T00:00:00,100\n"
        "2026-01-01T00:10:00,120\n",
    )

    series = load_cgm_csv(path, interval_minutes=5)

    assert len(series.timestamps) == 3
    assert series.observed_mask.tolist() == [True, False, True]
    assert torch.equal(series.glucose, torch.tensor([100.0, 0.0, 120.0]))
    assert torch.equal(series.gap_age_minutes, torch.tensor([0.0, 5.0, 0.0]))
    assert series.time_of_day.shape == (3, 2)


def test_empty_cell_is_missing_and_duplicate_uses_last_value(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "cgm.csv",
        "timestamp,glucose\n"
        "2026-01-01T00:00:00,90\n"
        "2026-01-01T00:05:00,\n"
        "2026-01-01T00:05:00,100\n",
    )

    series = load_cgm_csv(path)

    assert series.observed_mask.tolist() == [True, True]
    assert torch.equal(series.glucose, torch.tensor([90.0, 100.0]))


def test_rejects_off_grid_timestamp(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "cgm.csv",
        "timestamp,glucose\n"
        "2026-01-01T00:00:00,100\n"
        "2026-01-01T00:02:30,105\n",
    )

    with pytest.raises(ValueError, match="nearest grid point"):
        load_cgm_csv(path, interval_minutes=5)


def test_window_dataset_filters_and_shapes(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "cgm.csv",
        "timestamp,glucose\n"
        "2026-01-01T00:00:00,100\n"
        "2026-01-01T00:05:00,101\n"
        "2026-01-01T00:10:00,102\n"
        "2026-01-01T00:15:00,103\n",
    )
    dataset = CGMWindowDataset(
        load_cgm_csv(path), window_size=3, stride=1, min_observed=3
    )

    assert len(dataset) == 2
    assert dataset[0]["glucose"].shape == (3,)
    assert dataset[0]["gap_age_minutes"].shape == (3,)
    assert dataset[0]["time_of_day"].shape == (3, 2)
    assert dataset[1]["start_index"] == 1
