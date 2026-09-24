"""Manifest-backed CGM datasets and participant-disjoint data splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .data import parse_timestamp

SplitName = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical manifest {path}: {exc}") from exc

    required = {
        "schema_version",
        "canonical_unit",
        "interval_minutes",
        "window_positions",
        "records",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"manifest is missing field(s): {', '.join(sorted(missing))}")
    if manifest["schema_version"] != "1.0":
        raise ValueError("only canonical manifest schema 1.0 is supported")
    if manifest["canonical_unit"] != "mg/dL":
        raise ValueError("canonical manifest unit must be mg/dL")
    if not isinstance(manifest["records"], list) or not manifest["records"]:
        raise ValueError("canonical manifest must contain records")
    if int(manifest["interval_minutes"]) <= 0:
        raise ValueError("manifest interval_minutes must be positive")
    if int(manifest["window_positions"]) <= 0:
        raise ValueError("manifest window_positions must be positive")
    return manifest


def _participant_key(record: dict[str, Any]) -> tuple[str, str]:
    try:
        provenance = record["provenance"]
        dataset = str(provenance["dataset"])
        participant_id = str(record["participant_id"])
    except (KeyError, TypeError) as exc:
        raise ValueError("manifest record is missing participant provenance") from exc
    if str(provenance.get("participant_id")) != participant_id:
        raise ValueError("record participant_id does not match its provenance")
    if not dataset or not participant_id:
        raise ValueError("dataset and participant_id cannot be empty")
    return dataset, participant_id


def _split_counts(
    participant_count: int, train_fraction: float, validation_fraction: float
) -> tuple[int, int, int]:
    test_fraction = round(1.0 - train_fraction - validation_fraction, 12)
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("train, validation, and test fractions must all be positive")
    if participant_count < 3:
        raise ValueError("at least three participants are required for three splits")

    train_count = max(1, int(participant_count * train_fraction))
    validation_count = max(1, int(participant_count * validation_fraction))
    test_count = participant_count - train_count - validation_count
    while test_count < 1:
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("cannot allocate non-empty participant splits")
        test_count = participant_count - train_count - validation_count
    return train_count, validation_count, test_count


def create_participant_split(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    *,
    seed: int = 7,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    overwrite: bool = False,
) -> Path:
    """Write a deterministic split in which participant groups never overlap."""

    source = Path(manifest_path)
    manifest = _read_manifest(source)
    destination = (
        Path(output_path) if output_path is not None else source.with_name("splits.json")
    )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass overwrite=True to replace it"
        )

    participants = sorted({_participant_key(record) for record in manifest["records"]})
    train_count, validation_count, _ = _split_counts(
        len(participants), train_fraction, validation_fraction
    )
    participants.sort(
        key=lambda member: (
            hashlib.sha256(
                f"{seed}\x1f{member[0]}\x1f{member[1]}".encode("utf-8")
            ).digest(),
            member,
        )
    )
    members = {
        "train": participants[:train_count],
        "validation": participants[
            train_count : train_count + validation_count
        ],
        "test": participants[train_count + validation_count :],
    }

    records_per_participant: dict[tuple[str, str], int] = {}
    for record in manifest["records"]:
        key = _participant_key(record)
        records_per_participant[key] = records_per_participant.get(key, 0) + 1

    split_payload: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        split_members = members[name]
        split_payload[name] = {
            "participants": [
                {"dataset": dataset, "participant_id": participant_id}
                for dataset, participant_id in split_members
            ],
            "participant_count": len(split_members),
            "record_count": sum(
                records_per_participant[member] for member in split_members
            ),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_manifest = os.path.relpath(source.resolve(), destination.parent.resolve())
    payload = {
        "schema_version": "1.0",
        "strategy": "participant_disjoint_seeded_sha256_order",
        "participant_key": ["dataset", "participant_id"],
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": round(1.0 - train_fraction - validation_fraction, 12),
        },
        "source_manifest": relative_manifest,
        "source_manifest_sha256": _sha256(source),
        "splits": split_payload,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def create_prospective_holdout_split(
    manifest_path: str | Path,
    parent_split_path: str | Path,
    output_path: str | Path,
    *,
    reserve_fraction: float = 0.20,
    seed: int = 29,
    overwrite: bool = False,
    parent_protocol: str = "1.0",
) -> Path:
    """Reserve an unseen holdout from a parent split's former training members.

    The parent validation members remain validation members. Its consumed test
    members move into training, and only deterministically selected former
    training members become the new prospective test set.
    """

    if not 0.0 < reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in (0, 1)")
    source = Path(manifest_path)
    parent = Path(parent_split_path)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass overwrite=True to replace it"
        )

    manifest = _read_manifest(source)
    manifest_members = {_participant_key(record) for record in manifest["records"]}
    parent_members: dict[SplitName, set[tuple[str, str]]] = {}
    for split_name in SPLIT_NAMES:
        members, manifest_hash, declared_members = _load_split_members(
            parent, split_name
        )
        if manifest_hash != _sha256(source):
            raise ValueError("parent split does not match the canonical manifest")
        if declared_members != manifest_members:
            raise ValueError(
                "parent split participants do not match canonical manifest participants"
            )
        parent_members[split_name] = members

    former_train = parent_members["train"]
    if len(former_train) < 2:
        raise ValueError("parent training split needs at least two participants")
    ordered_train = sorted(
        former_train,
        key=lambda member: (
            hashlib.sha256(
                f"{seed}\x1f{member[0]}\x1f{member[1]}".encode("utf-8")
            ).digest(),
            member,
        ),
    )
    reserve_count = min(
        len(former_train) - 1,
        max(1, round(len(former_train) * reserve_fraction)),
    )
    prospective_test = set(ordered_train[:reserve_count])
    members_by_split: dict[SplitName, set[tuple[str, str]]] = {
        "train": (
            former_train.difference(prospective_test) | parent_members["test"]
        ),
        "validation": parent_members["validation"],
        "test": prospective_test,
    }

    records_per_participant: dict[tuple[str, str], int] = {}
    for record in manifest["records"]:
        key = _participant_key(record)
        records_per_participant[key] = records_per_participant.get(key, 0) + 1
    split_payload: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_members = sorted(members_by_split[split_name])
        split_payload[split_name] = {
            "participants": [
                {"dataset": dataset, "participant_id": participant_id}
                for dataset, participant_id in split_members
            ],
            "participant_count": len(split_members),
            "record_count": sum(
                records_per_participant[member] for member in split_members
            ),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "strategy": "prospective_holdout_from_parent_training_v1",
        "participant_key": ["dataset", "participant_id"],
        "seed": seed,
        "reserve_fraction_of_parent_training": reserve_fraction,
        "source_manifest": os.path.relpath(
            source.resolve(), destination.parent.resolve()
        ),
        "source_manifest_sha256": _sha256(source),
        "parent_split": os.path.relpath(
            parent.resolve(), destination.parent.resolve()
        ),
        "parent_split_sha256": _sha256(parent),
        "parent_test_disposition": (
            f"moved_to_train_after_protocol_{parent_protocol}_consumption"
        ),
        "prospective_test_origin": "parent_train",
        "splits": split_payload,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def carry_split_membership(
    manifest_path: str | Path,
    parent_split_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Bind a parent split's exact participant membership to a new manifest.

    Use this when the same participants are re-windowed (for example aligned to
    midnight) so no participant changes partition. Every participant in the new
    manifest must belong to the parent split; participants who no longer have
    any day are listed as dropped rather than reassigned.
    """

    source = Path(manifest_path)
    parent = Path(parent_split_path)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass overwrite=True to replace it"
        )
    manifest = _read_manifest(source)
    manifest_members = {_participant_key(record) for record in manifest["records"]}
    parent_members = {
        split_name: _load_split_members(parent, split_name)[0]
        for split_name in SPLIT_NAMES
    }
    declared = set().union(*parent_members.values())
    unknown = manifest_members.difference(declared)
    if unknown:
        raise ValueError(
            "manifest contains participants absent from the parent split: "
            + ", ".join(f"{dataset}/{pid}" for dataset, pid in sorted(unknown))
        )

    records_per_participant: dict[tuple[str, str], int] = {}
    for record in manifest["records"]:
        key = _participant_key(record)
        records_per_participant[key] = records_per_participant.get(key, 0) + 1
    split_payload: dict[str, Any] = {}
    dropped: list[dict[str, str]] = []
    for split_name in SPLIT_NAMES:
        kept = sorted(parent_members[split_name] & manifest_members)
        dropped.extend(
            {"dataset": dataset, "participant_id": pid, "split": split_name}
            for dataset, pid in sorted(parent_members[split_name] - manifest_members)
        )
        if not kept:
            raise ValueError(f"split {split_name!r} has no participants left")
        split_payload[split_name] = {
            "participants": [
                {"dataset": dataset, "participant_id": pid} for dataset, pid in kept
            ],
            "participant_count": len(kept),
            "record_count": sum(records_per_participant[member] for member in kept),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "strategy": "membership_carried_from_parent_v1",
        "participant_key": ["dataset", "participant_id"],
        "source_manifest": os.path.relpath(
            source.resolve(), destination.parent.resolve()
        ),
        "source_manifest_sha256": _sha256(source),
        "parent_split": os.path.relpath(
            parent.resolve(), destination.parent.resolve()
        ),
        "parent_split_sha256": _sha256(parent),
        "dropped_participants": dropped,
        "splits": split_payload,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _load_split_members(
    split_path: Path, split: SplitName
) -> tuple[set[tuple[str, str]], str, set[tuple[str, str]]]:
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        manifest_hash = str(payload["source_manifest_sha256"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read split file {split_path}: {exc}") from exc

    members_by_split: dict[SplitName, set[tuple[str, str]]] = {}
    all_members: set[tuple[str, str]] = set()
    try:
        for name in SPLIT_NAMES:
            raw_members = payload["splits"][name]["participants"]
            members = {
                (str(member["dataset"]), str(member["participant_id"]))
                for member in raw_members
            }
            if len(members) != len(raw_members):
                raise ValueError(f"split {name!r} contains duplicate participants")
            if all_members.intersection(members):
                raise ValueError("split file contains participant overlap")
            members_by_split[name] = members
            all_members.update(members)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"split file {split_path} has invalid participants") from exc
    return members_by_split[split], manifest_hash, all_members


def _time_of_day(timestamps: list[datetime]) -> torch.Tensor:
    features = []
    for timestamp in timestamps:
        minutes = (
            timestamp.hour * 60
            + timestamp.minute
            + timestamp.second / 60
            + timestamp.microsecond / 60_000_000
        )
        angle = 2.0 * math.pi * minutes / (24.0 * 60.0)
        features.append((math.sin(angle), math.cos(angle)))
    return torch.tensor(features, dtype=torch.float32)


class CanonicalCGMDataset(Dataset[dict[str, Any]]):
    """Load canonical day CSVs selected by an optional participant split."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split_path: str | Path | None = None,
        split: SplitName | None = None,
        verify_checksums: bool = True,
    ) -> None:
        if (split_path is None) != (split is None):
            raise ValueError("split_path and split must be provided together")
        if split is not None and split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")

        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent.resolve()
        self.manifest = _read_manifest(self.manifest_path)
        records = self.manifest["records"]

        if split_path is not None and split is not None:
            members, expected_manifest_hash, declared_members = _load_split_members(
                Path(split_path), split
            )
            actual_manifest_hash = _sha256(self.manifest_path)
            if actual_manifest_hash != expected_manifest_hash:
                raise ValueError("split file does not match the canonical manifest")
            manifest_members = {_participant_key(record) for record in records}
            if declared_members != manifest_members:
                raise ValueError(
                    "split participants do not match canonical manifest participants"
                )
            records = [
                record for record in records if _participant_key(record) in members
            ]
            if not records:
                raise ValueError(f"split {split!r} contains no canonical records")

        self.records = records
        seen_paths: set[Path] = set()
        for record in self.records:
            _participant_key(record)
            try:
                path = (self.root / record["file"]).resolve()
            except (KeyError, TypeError) as exc:
                raise ValueError("manifest record has no valid file path") from exc
            if not path.is_relative_to(self.root):
                raise ValueError(f"manifest file escapes corpus root: {path}")
            if path in seen_paths:
                raise ValueError(f"manifest contains duplicate file: {path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_checksums and _sha256(path) != record.get("sha256"):
                raise ValueError(f"checksum mismatch for {path}")
            seen_paths.add(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = (self.root / record["file"]).resolve()
        timestamps: list[datetime] = []
        glucose: list[float] = []
        observed: list[bool] = []
        gap_age: list[float] = []

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "timestamp",
                "glucose",
                "observed_mask",
                "gap_age_minutes",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing canonical column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    timestamp = parse_timestamp(row["timestamp"])
                    mask_text = row["observed_mask"].strip()
                    if mask_text not in {"0", "1"}:
                        raise ValueError("observed_mask must be 0 or 1")
                    is_observed = mask_text == "1"
                    glucose_text = row["glucose"].strip()
                    if is_observed and not glucose_text:
                        raise ValueError("observed glucose cannot be empty")
                    if not is_observed and glucose_text:
                        raise ValueError("missing glucose must be empty")
                    value = float(glucose_text) if is_observed else 0.0
                    age = float(row["gap_age_minutes"])
                    if not math.isfinite(value) or not math.isfinite(age) or age < 0:
                        raise ValueError("glucose and gap age must be finite")
                except ValueError as exc:
                    raise ValueError(f"{path}: line {line_number}: {exc}") from exc
                timestamps.append(timestamp)
                glucose.append(value)
                observed.append(is_observed)
                gap_age.append(age)

        expected_positions = int(self.manifest["window_positions"])
        if len(timestamps) != expected_positions:
            raise ValueError(
                f"{path} contains {len(timestamps)} positions; "
                f"expected {expected_positions}"
            )
        interval_seconds = int(self.manifest["interval_minutes"]) * 60
        for previous, current in zip(timestamps, timestamps[1:]):
            if (current - previous).total_seconds() != interval_seconds:
                raise ValueError(f"{path} does not have a regular timestamp grid")
        if sum(observed) != int(record["observed_count"]):
            raise ValueError(f"{path} observed count does not match the manifest")

        dataset, participant_id = _participant_key(record)
        return {
            "glucose": torch.tensor(glucose, dtype=torch.float32),
            "observed_mask": torch.tensor(observed, dtype=torch.bool),
            "gap_age_minutes": torch.tensor(gap_age, dtype=torch.float32),
            "time_of_day": _time_of_day(timestamps),
            "dataset": dataset,
            "participant_id": participant_id,
            "start_time": timestamps[0].isoformat(),
            "source_file": record["file"],
        }


class MultiSourceCGMDataset(Dataset[dict[str, Any]]):
    """Present several canonical corpora as one indexable dataset."""

    def __init__(self, datasets: Sequence[CanonicalCGMDataset]) -> None:
        if not datasets:
            raise ValueError("at least one canonical dataset is required")
        self.datasets = tuple(datasets)
        self.entries: list[tuple[int, int, str]] = []
        source_indices: dict[str, list[int]] = {}
        for dataset_index, dataset in enumerate(self.datasets):
            for local_index, record in enumerate(dataset.records):
                source_name, _ = _participant_key(record)
                global_index = len(self.entries)
                self.entries.append((dataset_index, local_index, source_name))
                source_indices.setdefault(source_name, []).append(global_index)
        if not self.entries:
            raise ValueError("canonical datasets contain no records")
        self.source_indices = {
            source: tuple(indices) for source, indices in sorted(source_indices.items())
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index, local_index, _ = self.entries[index]
        return self.datasets[dataset_index][local_index]


class SourceBalancedSampler(Sampler[int]):
    """Sample sources equally while shuffling days reproducibly within each source.

    Smaller sources are cycled with a fresh permutation when exhausted. Each
    epoch therefore contains source counts that differ by at most one.
    """

    def __init__(
        self,
        dataset: MultiSourceCGMDataset,
        *,
        num_samples: int | None = None,
        seed: int = 7,
    ) -> None:
        self.dataset = dataset
        default_samples = len(dataset.source_indices) * max(
            len(indices) for indices in dataset.source_indices.values()
        )
        self.num_samples = default_samples if num_samples is None else num_samples
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sources = tuple(self.dataset.source_indices)
        orders: dict[str, list[int]] = {source: [] for source in sources}
        positions = {source: 0 for source in sources}

        def reshuffle(source: str) -> None:
            indices = self.dataset.source_indices[source]
            permutation = torch.randperm(len(indices), generator=generator).tolist()
            orders[source] = [indices[position] for position in permutation]
            positions[source] = 0

        for step in range(self.num_samples):
            source = sources[step % len(sources)]
            if positions[source] >= len(orders[source]):
                reshuffle(source)
            yield orders[source][positions[source]]
            positions[source] += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic participant-disjoint CGM corpus splits."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = create_participant_split(
        args.manifest,
        args.output,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        overwrite=args.overwrite,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    for name in SPLIT_NAMES:
        split = payload["splits"][name]
        print(
            f"{name}: participants={split['participant_count']} "
            f"days={split['record_count']}"
        )
    print(f"split_file={output}")


if __name__ == "__main__":
    main()
