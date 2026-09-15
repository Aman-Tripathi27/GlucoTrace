import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

from glucofm.canonical import (
    CGMReading,
    SourceProvenance,
    build_24h_windows,
    write_canonical_corpus,
)
from glucofm.corpus import (
    CanonicalCGMDataset,
    create_participant_split,
    create_prospective_holdout_split,
)


def make_corpus(root: Path, participant_count: int = 6) -> Path:
    days = []
    start = datetime(2026, 1, 1, 6)
    for participant_index in range(participant_count):
        participant_id = f"p{participant_index:03d}"
        provenance = SourceProvenance(
            dataset="test-source",
            version="1.0",
            participant_id=participant_id,
            device="test-device",
            unit="mg/dL",
            source_url="https://example.test/data",
            license_name="test license",
            license_url="https://example.test/license",
        )
        readings = [
            CGMReading(
                start + timedelta(minutes=5 * reading_index),
                90.0 + participant_index + reading_index % 10,
            )
            for reading_index in range(288)
            if reading_index != 20 + participant_index
        ]
        days.append(
            build_24h_windows(
                readings, provenance, min_observed_fraction=0.9
            )[0]
        )
    return write_canonical_corpus(days, root)


def participant_members(payload: dict, split: str) -> set[tuple[str, str]]:
    return {
        (member["dataset"], member["participant_id"])
        for member in payload["splits"][split]["participants"]
    }


def test_split_is_deterministic_and_participant_disjoint(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus")
    first_path = create_participant_split(manifest, tmp_path / "first.json")
    second_path = create_participant_split(manifest, tmp_path / "second.json")
    third_path = create_participant_split(
        manifest, tmp_path / "third.json", seed=8
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    third = json.loads(third_path.read_text(encoding="utf-8"))

    train = participant_members(first, "train")
    validation = participant_members(first, "validation")
    test = participant_members(first, "test")

    assert first["splits"] == second["splits"]
    assert first["splits"]["train"] != third["splits"]["train"]
    assert not train.intersection(validation)
    assert not train.intersection(test)
    assert not validation.intersection(test)
    assert len(train | validation | test) == 6
    assert sum(first["splits"][name]["record_count"] for name in first["splits"]) == 6


def test_canonical_dataset_loads_model_ready_tensors(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus")
    dataset = CanonicalCGMDataset(manifest)

    sample = dataset[0]

    assert len(dataset) == 6
    assert sample["glucose"].shape == (288,)
    assert sample["observed_mask"].dtype == torch.bool
    assert sample["gap_age_minutes"].shape == (288,)
    assert sample["time_of_day"].shape == (288, 2)
    assert sample["glucose"][~sample["observed_mask"]].eq(0).all()
    assert sample["dataset"] == "test-source"


def test_dataset_selects_only_requested_split(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus")
    split_path = create_participant_split(manifest, tmp_path / "split.json")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))

    datasets = {
        name: CanonicalCGMDataset(
            manifest, split_path=split_path, split=name
        )
        for name in ("train", "validation", "test")
    }

    for name, dataset in datasets.items():
        expected = participant_members(split_payload, name)
        actual = {
            (sample["dataset"], sample["participant_id"])
            for sample in dataset
        }
        assert actual == expected


def test_split_rejects_changed_manifest(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus")
    split_path = create_participant_split(manifest, tmp_path / "split.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["observed_count"] -= 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        CanonicalCGMDataset(
            manifest, split_path=split_path, split="train"
        )


def test_dataset_rejects_participant_overlap_in_split_file(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus")
    split_path = create_participant_split(manifest, tmp_path / "split.json")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["splits"]["validation"]["participants"].append(
        payload["splits"]["train"]["participants"][0]
    )
    split_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="participant overlap"):
        CanonicalCGMDataset(
            manifest, split_path=split_path, split="validation"
        )


def test_dataset_detects_changed_day_file(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    day_path = manifest.parent / payload["records"][0]["file"]
    day_path.write_text(day_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        CanonicalCGMDataset(manifest)


def test_split_requires_enough_participants(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus", participant_count=2)

    with pytest.raises(ValueError, match="at least three"):
        create_participant_split(manifest, tmp_path / "split.json")


def test_prospective_holdout_comes_only_from_parent_training(tmp_path: Path) -> None:
    manifest = make_corpus(tmp_path / "corpus", participant_count=10)
    parent_path = create_participant_split(manifest, tmp_path / "parent.json")
    output = create_prospective_holdout_split(
        manifest,
        parent_path,
        tmp_path / "prospective.json",
        reserve_fraction=0.30,
        seed=29,
    )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    prospective = json.loads(output.read_text(encoding="utf-8"))

    parent_train = participant_members(parent, "train")
    parent_validation = participant_members(parent, "validation")
    parent_test = participant_members(parent, "test")
    new_train = participant_members(prospective, "train")
    new_validation = participant_members(prospective, "validation")
    new_test = participant_members(prospective, "test")

    assert new_test
    assert new_test < parent_train
    assert new_validation == parent_validation
    assert parent_test <= new_train
    assert not new_train.intersection(new_validation | new_test)
    assert new_train | new_validation | new_test == (
        parent_train | parent_validation | parent_test
    )
