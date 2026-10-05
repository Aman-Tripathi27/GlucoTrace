"""Random-phase training windows from continuous CGM recordings (protocol 1.4).

Protocol 1.3 aligned every day to midnight, which removed the start-time cue
that identified the source dataset but also removed the variety of circadian
offsets the model had been learning from. This dataset restores that variety
without the cue: every training draw cuts a 24-hour window at a random
reading of a continuous recording, with the same procedure for every source.

Only participants in a split's training partition are loaded, so validation
and test participants can never enter training through this path.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .adapters import big_ideas, colas, shanghai
from .canonical import (
    CGMReading,
    SourceProvenance,
    _build_window,
    split_continuous_segments,
)

SOURCE_READERS = ("BIG_IDEAs", "Colas2019", "Shanghai")


@dataclass(frozen=True)
class Recording:
    """One continuous segment of one training participant."""

    dataset: str
    participant_id: str
    readings: tuple[CGMReading, ...]
    timestamps: tuple[Any, ...]
    provenance: SourceProvenance

    @property
    def cadence_minutes(self) -> float:
        """Median spacing of readings (5 for Dexcom/iPro, 15 for Libre)."""

        gaps = sorted(
            (later - earlier).total_seconds() / 60
            for earlier, later in zip(self.timestamps, self.timestamps[1:])
        )
        return gaps[len(gaps) // 2] if gaps else 5.0


def _training_members(split_path: Path) -> set[tuple[str, str]]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        (str(member["dataset"]), str(member["participant_id"]))
        for member in payload["splits"]["train"]["participants"]
    }


def _provenance(module: Any, participant_id: str, device: str) -> SourceProvenance:
    return SourceProvenance(
        dataset=module.DATASET_NAME,
        version=module.DATASET_VERSION,
        participant_id=participant_id,
        device=device,
        unit="mg/dL",
        source_url=module.SOURCE_URL,
        license_name=module.LICENSE_NAME,
        license_url=module.LICENSE_URL,
    )


def _read_participants(dataset: str, raw_path: Path) -> list[tuple[str, list[CGMReading], SourceProvenance]]:
    if dataset == "BIG_IDEAs":
        participants = []
        for path in sorted(raw_path.glob("[0-9][0-9][0-9]/Dexcom_*.csv")):
            participant_id, readings = big_ideas.read_big_ideas_file(path)
            participants.append(
                (participant_id, readings, _provenance(big_ideas, participant_id, "Dexcom G6"))
            )
        return participants
    if dataset == "Colas2019":
        participants = []
        for filename, contents in colas._iter_cases(raw_path):
            participant_id, readings = colas._read_case(io.StringIO(contents), filename)
            participants.append(
                (participant_id, readings, _provenance(colas, participant_id, colas.DEVICE))
            )
        return participants
    if dataset == "Shanghai":
        participants = []
        for subset in shanghai.SUBSETS:
            for participant_id, readings in shanghai.load_participant_readings(
                raw_path, subset
            ).items():
                participants.append(
                    (participant_id, readings, shanghai._provenance(participant_id, subset))
                )
        return participants
    raise ValueError(f"no raw reader for dataset {dataset!r}; known: {SOURCE_READERS}")


def load_training_recordings(
    raw_sources: Sequence[tuple[str, str | Path]],
    split_paths: Sequence[str | Path],
    *,
    max_gap_minutes: int = 60,
) -> list[Recording]:
    """Load continuous segments for training participants only."""

    members: set[tuple[str, str]] = set()
    for split_path in split_paths:
        members |= _training_members(Path(split_path))
    recordings: list[Recording] = []
    for dataset, raw_path in raw_sources:
        for participant_id, readings, provenance in _read_participants(dataset, Path(raw_path)):
            if (dataset, participant_id) not in members:
                continue
            for segment in split_continuous_segments(readings, max_gap_minutes=max_gap_minutes):
                recordings.append(
                    Recording(
                        dataset=dataset,
                        participant_id=participant_id,
                        readings=tuple(segment),
                        timestamps=tuple(reading.timestamp for reading in segment),
                        provenance=provenance,
                    )
                )
    loaded = {(item.dataset, item.participant_id) for item in recordings}
    missing = {member for member in members if member[0] in dict(raw_sources)} - loaded
    if missing:
        raise ValueError(
            f"{len(missing)} training participant(s) were not found in the raw data"
        )
    return recordings


class RandomPhaseCGMDataset(Dataset[dict[str, Any]]):
    """Source-balanced 24-hour windows cut at random readings.

    Item ``i`` draws from source ``i % n_sources``; within a source every valid
    start reading is equally likely, so participants contribute in proportion to
    their recording length, as with fixed canonical days. Draws are reproducible
    from ``(seed, epoch, index)``.
    """

    def __init__(
        self,
        recordings: Sequence[Recording],
        *,
        samples_per_epoch: int,
        seed: int = 7,
        interval_minutes: int = 5,
        min_observed_fraction: float = 0.8,
        max_gap_minutes: int = 60,
        max_attempts: int = 50,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.recordings = tuple(recordings)
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        self.interval_minutes = interval_minutes
        self.min_observed_fraction = min_observed_fraction
        self.max_gap_minutes = max_gap_minutes
        self.max_attempts = max_attempts
        self.span = timedelta(minutes=(24 * 60 // interval_minutes - 1) * interval_minutes)
        starts: dict[str, list[tuple[int, int]]] = {}
        for index, recording in enumerate(self.recordings):
            last = recording.timestamps[-1]
            for position, stamp in enumerate(recording.timestamps):
                if stamp + self.span > last:
                    break
                starts.setdefault(recording.dataset, []).append((index, position))
        if len(starts) < 1:
            raise ValueError("no recording is long enough for a 24-hour window")
        self.starts = {source: tuple(values) for source, values in sorted(starts.items())}
        self.sources = tuple(self.starts)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _generator(self, index: int) -> random.Random:
        payload = f"{self.seed}\x1f{self.epoch}\x1f{index}".encode("utf-8")
        return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.sources[index % len(self.sources)]
        generator = self._generator(index)
        tolerance = 0.4 * self.interval_minutes * 60
        for _ in range(self.max_attempts):
            recording_index, position = generator.choice(self.starts[source])
            recording = self.recordings[recording_index]
            start = recording.timestamps[position]
            stop = bisect.bisect_right(recording.timestamps, start + timedelta(hours=24))
            try:
                day = _build_window(
                    recording.readings[position:stop],
                    start=start,
                    provenance=recording.provenance,
                    interval_minutes=self.interval_minutes,
                    max_gap_minutes=self.max_gap_minutes,
                    alignment_tolerance_seconds=tolerance,
                )
            except ValueError:
                continue
            cadence_scale = min(1.0, self.interval_minutes / recording.cadence_minutes)
            if day.observed_fraction >= self.min_observed_fraction * cadence_scale:
                return {
                    "glucose": day.glucose,
                    "observed_mask": day.observed_mask,
                    "gap_age_minutes": day.gap_age_minutes,
                    "time_of_day": day.time_of_day,
                    "dataset": recording.dataset,
                    "participant_id": recording.participant_id,
                }
        raise RuntimeError(f"could not draw a valid window for source {source!r}")


def start_hour_histogram(dataset: RandomPhaseCGMDataset, draws: int = 400) -> dict[str, list[int]]:
    """Count window start hours per source (a leakage diagnostic)."""

    counts = {source: [0] * 24 for source in dataset.sources}
    for index in range(draws):
        item = dataset[index]
        sin, cos = item["time_of_day"][0].tolist()
        minutes = (torch.atan2(torch.tensor(sin), torch.tensor(cos)).item() / (2 * 3.141592653589793)) * 1440
        counts[item["dataset"]][int((minutes % 1440) // 60)] += 1
    return counts
