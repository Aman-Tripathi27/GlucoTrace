"""Shanghai adapter, cadence handling, coarse clock, and protocol 1.5 rigor."""

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from glucotrace import rigor
from glucotrace.adapters import shanghai
from glucotrace.data import thin_to_cadence
from glucotrace.model import GlucoTrace, GlucoTraceConfig


def workbook(start: datetime, hours: int, *, meal_every: int = 24) -> bytes:
    rows = []
    for index in range(hours * 4):
        stamp = start + timedelta(minutes=15 * index)
        rows.append(
            {
                "Date": stamp,
                "CGM (mg / dl)": 100.0 + index % 12,
                "Dietary intake": "Rice 100 g" if index % meal_every == 0 else None,
            }
        )
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def make_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Shanghai_T2DM/2001_0_20210101.xlsx", workbook(datetime(2021, 1, 1, 16, 43), 50))
        archive.writestr("Shanghai_T2DM/2001_1_20210110.xlsx", workbook(datetime(2021, 1, 10, 9, 13), 30))
        archive.writestr("Shanghai_T2DM/2002_0_20210201.xlsx", workbook(datetime(2021, 2, 1, 7, 58), 30))
        archive.writestr("__MACOSX/Shanghai_T2DM/._2002_0_20210201.xlsx", b"junk")
    return path


def test_shanghai_groups_files_by_patient_and_aligns_to_midnight(tmp_path: Path) -> None:
    source = make_zip(tmp_path / "diabetes_datasets.zip")
    readings = shanghai.load_participant_readings(source, "T2DM")
    assert sorted(readings) == ["2001", "2002"]
    assert len(readings["2001"]) == (50 + 30) * 4

    days = shanghai.load_shanghai(source, "T2DM", anchor="midnight")
    assert days and all(day.start_time.hour == 0 and day.start_time.minute < 5 for day in days)
    assert all(day.provenance.dataset == "Shanghai" for day in days)
    fractions = {round(day.observed_fraction, 3) for day in days}
    assert fractions == {0.333}  # 96 readings on a 288-position grid


def test_shanghai_meals_skip_unrecorded_entries(tmp_path: Path) -> None:
    source = make_zip(tmp_path / "diabetes_datasets.zip")
    meals = shanghai.load_meals(source)
    assert meals and all(meal.subset == "T2DM" for meal in meals)
    assert {meal.participant_id for meal in meals} == {"2001", "2002"}


def test_thin_to_cadence_keeps_native_coarse_rows() -> None:
    glucose = torch.full((2, 12), 100.0)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1] = torch.arange(12).remainder(3).eq(1)
    _, thinned = thin_to_cadence(glucose, mask, 3, generator=torch.Generator().manual_seed(0))
    assert int(thinned[0].sum()) == 4
    assert torch.unique(torch.arange(12)[thinned[0]].remainder(3)).numel() == 1
    assert torch.equal(thinned[1], mask[1])


def test_coarse_clock_maps_times_within_a_bin_to_one_value() -> None:
    torch.manual_seed(0)
    config = GlucoTraceConfig(
        hidden_size=16, num_layers=1, num_heads=4, feedforward_size=32, dropout=0.0,
        patch_size=3, trend_windows=(1, 3, 6), event_trend_window=3, max_patches=8,
        clock_bins=4,
    )
    model = GlucoTrace(config).eval()
    glucose = 100 + 20 * torch.rand(1, 24)
    mask = torch.ones(1, 24, dtype=torch.bool)

    def clock(hour: float) -> torch.Tensor:
        angle = torch.full((1, 24), 2 * torch.pi * hour / 24)
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    with torch.no_grad():
        morning_a = model(glucose, mask, time_of_day=clock(7.0))["pooled_embedding"]
        morning_b = model(glucose, mask, time_of_day=clock(10.5))["pooled_embedding"]
        evening = model(glucose, mask, time_of_day=clock(20.0))["pooled_embedding"]
    assert torch.allclose(morning_a, morning_b)
    assert not torch.allclose(morning_a, evening)
    with pytest.raises(ValueError, match="clock_bins"):
        GlucoTraceConfig(use_clock=False, clock_bins=4)


def test_cluster_bootstrap_resamples_whole_participants() -> None:
    participants = ["a"] * 5 + ["b"] * 5
    values = torch.tensor([1.0] * 5 + [3.0] * 5)
    interval = rigor.cluster_bootstrap(
        participants, lambda rows: float(values[rows].mean()), replicates=200
    )
    assert 1.0 <= interval["ci95_low"] <= 2.0 <= interval["ci95_high"] <= 3.0
    assert interval["unit"] == "participant"
