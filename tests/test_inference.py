import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

from glucofm.calibrate import calibrate_checkpoint
from glucofm.commands import compare_main, encode_main, search_main
from glucofm.corpus import create_participant_split
from glucofm.inference import (
    ResearchEncoder,
    cosine_similarity,
    search_manifests,
    sha256_file,
)
from glucofm.model import GlucoFM, GlucoFMConfig


def tiny_config() -> GlucoFMConfig:
    return GlucoFMConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        feedforward_size=32,
        dropout=0.0,
        patch_size=3,
        trend_windows=(1, 3, 6),
        event_trend_window=3,
        max_patches=4,
        pool_segments=4,
    )


def write_day(path: Path, *, start: datetime, offset: float = 0.0) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "timestamp",
                "glucose",
                "observed_mask",
                "gap_age_minutes",
            ),
        )
        writer.writeheader()
        for index in range(12):
            writer.writerow(
                {
                    "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
                    "glucose": 90.0 + offset + index,
                    "observed_mask": 1,
                    "gap_age_minutes": 0,
                }
            )


def make_manifest(root: Path, participant_count: int = 6) -> Path:
    root.mkdir()
    day_root = root / "days"
    day_root.mkdir()
    records = []
    for index in range(participant_count):
        participant = f"p{index}"
        day_path = day_root / f"{participant}.csv"
        start = datetime(2026, 1, index + 1)
        write_day(day_path, start=start, offset=10.0 * index)
        records.append(
            {
                "file": f"days/{participant}.csv",
                "sha256": sha256_file(day_path),
                "participant_id": participant,
                "start_time": start.isoformat(),
                "observed_count": 12,
                "observed_fraction": 1.0,
                "provenance": {
                    "dataset": "tiny",
                    "version": "1.0",
                    "participant_id": participant,
                    "device": "test",
                    "unit": "mg/dL",
                    "source_url": "https://example.test",
                    "license_name": "test",
                    "license_url": "https://example.test/license",
                },
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "canonical_unit": "mg/dL",
                "interval_minutes": 5,
                "window_positions": 12,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def make_checkpoint(
    path: Path,
    *,
    calibrated: bool = True,
    corpora: list[dict] | None = None,
) -> Path:
    torch.manual_seed(3)
    config = tiny_config()
    payload = {
        "model_state_dict": GlucoFM(config).state_dict(),
        "config": asdict(config),
        "research_only": True,
        "corpora": corpora or [],
    }
    if calibrated:
        payload["embedding_calibration"] = {
            "method": "validation_zscore_then_l2",
            "partition": "validation",
            "center": [0.0] * config.hidden_size,
            "scale": [1.0] * config.hidden_size,
        }
    torch.save(payload, path)
    return path


def test_encoder_is_deterministic_calibrated_and_unit_length(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    day = tmp_path / "day.csv"
    write_day(day, start=datetime(2026, 1, 1))
    encoder = ResearchEncoder.load(checkpoint)

    first, metadata = encoder.encode_csv(day)
    second, _ = encoder.encode_csv(day)

    assert first.shape == (16,)
    assert torch.equal(first, second)
    assert torch.linalg.vector_norm(first) == pytest.approx(1.0)
    assert cosine_similarity(first, second) == pytest.approx(1.0)
    assert metadata["observed_count"] == 12
    assert metadata["sha256"] == sha256_file(day)


def test_encoder_rejects_checkpoint_without_calibration(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "model.pt", calibrated=False)

    with pytest.raises(ValueError, match="glucofm-calibrate"):
        ResearchEncoder.load(checkpoint)


def test_search_ranks_identical_canonical_day_first(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path / "corpus")
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    encoder = ResearchEncoder.load(checkpoint)
    query_path = manifest.parent / "days/p0.csv"
    query, metadata = encoder.encode_csv(query_path)

    included = search_manifests(
        encoder, query, [manifest], top_k=3, batch_size=6
    )
    excluded = search_manifests(
        encoder,
        query,
        [manifest],
        top_k=3,
        exclude_sha256=metadata["sha256"],
        batch_size=6,
    )

    assert included[0]["participant_id"] == "p0"
    assert included[0]["similarity"] == pytest.approx(1.0)
    assert all(match["participant_id"] != "p0" for match in excluded)
    assert [match["rank"] for match in excluded] == [1, 2, 3]


def test_calibration_uses_declared_validation_partition(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path / "corpus")
    split = create_participant_split(manifest, tmp_path / "split.json")
    corpora = [
        {
            "manifest_sha256": sha256_file(manifest),
            "split_sha256": sha256_file(split),
        }
    ]
    source = make_checkpoint(
        tmp_path / "uncalibrated.pt", calibrated=False, corpora=corpora
    )

    output = calibrate_checkpoint(
        source, [(manifest, split)], tmp_path / "calibrated.pt", batch_size=2
    )
    encoder = ResearchEncoder.load(output)

    assert encoder.calibration["partition"] == "validation"
    assert encoder.calibration["validation_days"] > 0


def test_encode_command_writes_versioned_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    day = tmp_path / "day.csv"
    output = tmp_path / "fingerprint.json"
    write_day(day, start=datetime(2026, 1, 1))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glucofm-encode",
            str(day),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )

    encode_main()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["research_only"] is True
    assert len(payload["fingerprint"]) == 16
    assert payload["input"]["sha256"] == sha256_file(day)


def test_compare_and_search_commands_write_descriptive_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = make_manifest(tmp_path / "corpus")
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    first = manifest.parent / "days/p0.csv"
    second = manifest.parent / "days/p1.csv"
    comparison_path = tmp_path / "comparison.json"
    search_path = tmp_path / "search.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glucofm-compare",
            str(first),
            str(second),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(comparison_path),
        ],
    )
    compare_main()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glucofm-search",
            str(first),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--top-k",
            "2",
            "--batch-size",
            "6",
            "--output",
            str(search_path),
        ],
    )
    search_main()
    search = json.loads(search_path.read_text(encoding="utf-8"))

    assert -1.0 <= comparison["cosine_similarity"] <= 1.0
    assert "clinical score" in comparison["interpretation_limit"]
    assert search["result_count"] == 2
    assert [result["rank"] for result in search["results"]] == [1, 2]
    assert "dataset artifacts" in search["interpretation_limit"]
