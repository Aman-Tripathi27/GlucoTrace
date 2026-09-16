"""Reusable research-only encoding and retrieval helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .corpus import CanonicalCGMDataset
from .data import CGMWindowDataset, load_cgm_csv
from .model import GlucoFM, GlucoFMConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model_payload(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[GlucoFM, dict[str, Any]]:
    """Load a marked research checkpoint without executing pickled code."""

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("research_only") is not True:
        raise ValueError("checkpoint lacks the required research_only marker")
    try:
        config_values = dict(checkpoint["config"])
        config_values["trend_windows"] = tuple(config_values["trend_windows"])
        config_values.setdefault("pool_segments", 1)
        model = GlucoFM(GlucoFMConfig(**config_values))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid GlucoTrace checkpoint: {exc}") from exc
    return model.to(device).eval(), checkpoint


@dataclass
class ResearchEncoder:
    """A loaded model plus its validation-fitted fingerprint calibration."""

    model: GlucoFM
    checkpoint_path: Path
    checkpoint_sha256: str
    center: torch.Tensor
    scale: torch.Tensor
    device: torch.device
    calibration: dict[str, Any]

    @classmethod
    def load(
        cls, checkpoint_path: str | Path, *, device: str = "cpu"
    ) -> "ResearchEncoder":
        path = Path(checkpoint_path)
        resolved_device = resolve_device(device)
        model, payload = load_model_payload(path, device=resolved_device)
        try:
            calibration = dict(payload["embedding_calibration"])
            if calibration["method"] != "validation_zscore_then_l2":
                raise ValueError("unsupported embedding calibration method")
            center = torch.as_tensor(calibration["center"], dtype=torch.float32)
            scale = torch.as_tensor(calibration["scale"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "checkpoint has no valid embedding calibration; run "
                "glucofm-calibrate first"
            ) from exc
        expected = model.config.hidden_size
        if center.shape != (expected,) or scale.shape != (expected,):
            raise ValueError("embedding calibration dimension does not match model")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("embedding calibration must contain finite values")
        if (scale <= 0).any():
            raise ValueError("embedding calibration scale must be positive")
        return cls(
            model=model,
            checkpoint_path=path,
            checkpoint_sha256=sha256_file(path),
            center=center.to(resolved_device),
            scale=scale.to(resolved_device),
            device=resolved_device,
            calibration=calibration,
        )

    @torch.no_grad()
    def encode_tensors(
        self,
        glucose: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        gap_age_minutes: torch.Tensor | None = None,
        time_of_day: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return validation-standardized, unit-length fingerprints."""

        unbatched = glucose.ndim == 1
        output = self.model(
            glucose.to(self.device),
            observed_mask.to(self.device),
            None if gap_age_minutes is None else gap_age_minutes.to(self.device),
            None if time_of_day is None else time_of_day.to(self.device),
        )["pooled_embedding"]
        if unbatched:
            output = output.unsqueeze(0)
        fingerprint = F.normalize((output - self.center) / self.scale, dim=-1)
        fingerprint = fingerprint.cpu()
        return fingerprint[0] if unbatched else fingerprint

    def encode_csv(
        self,
        path: str | Path,
        *,
        timestamp_col: str = "timestamp",
        glucose_col: str = "glucose",
        window_index: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Encode exactly one full model window selected from a CSV."""

        csv_path = Path(path)
        interval = self.model.config.interval_minutes
        window_size = self.model.config.patch_size * self.model.config.max_patches
        series = load_cgm_csv(
            csv_path,
            timestamp_col=timestamp_col,
            glucose_col=glucose_col,
            interval_minutes=interval,
        )
        windows = CGMWindowDataset(
            series,
            window_size=window_size,
            stride=window_size,
            min_observed=1,
        )
        if not windows:
            hours = window_size * interval / 60
            raise ValueError(f"CSV does not contain a complete {hours:g}-hour window")
        if window_index is None:
            if len(windows) != 1:
                raise ValueError(
                    f"CSV contains {len(windows)} complete windows; select one with "
                    "--window-index"
                )
            selected = 0
        else:
            selected = window_index
        if selected < 0 or selected >= len(windows):
            raise ValueError(f"window_index must be between 0 and {len(windows) - 1}")
        sample = windows[selected]
        fingerprint = self.encode_tensors(
            sample["glucose"],
            sample["observed_mask"],
            time_of_day=sample["time_of_day"],
        )
        start = int(sample["start_index"])
        observed_count = int(sample["observed_mask"].sum())
        metadata = {
            "path": str(csv_path),
            "sha256": sha256_file(csv_path),
            "window_index": selected,
            "available_windows": len(windows),
            "start_time": series.timestamps[start].isoformat(),
            "positions": window_size,
            "observed_count": observed_count,
            "observed_fraction": observed_count / window_size,
            "interval_minutes": interval,
        }
        return fingerprint, metadata


def cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("fingerprints must be one-dimensional and share a shape")
    return float(F.cosine_similarity(first, second, dim=0))


@torch.no_grad()
def search_manifests(
    encoder: ResearchEncoder,
    query: torch.Tensor,
    manifest_paths: Sequence[str | Path],
    *,
    top_k: int = 5,
    exclude_sha256: str | None = None,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Return the most cosine-similar canonical days from approved manifests."""

    if top_k <= 0 or batch_size <= 0:
        raise ValueError("top_k and batch_size must be positive")
    if query.ndim != 1 or query.shape[0] != encoder.model.config.hidden_size:
        raise ValueError("query fingerprint dimension does not match model")
    if not manifest_paths:
        raise ValueError("at least one manifest is required")

    matches = []
    for manifest_path in manifest_paths:
        dataset = CanonicalCGMDataset(manifest_path)
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            samples = [dataset[index] for index in range(start, stop)]
            glucose = torch.stack([sample["glucose"] for sample in samples])
            observed = torch.stack([sample["observed_mask"] for sample in samples])
            gap_age = torch.stack([sample["gap_age_minutes"] for sample in samples])
            time_of_day = torch.stack([sample["time_of_day"] for sample in samples])
            fingerprints = encoder.encode_tensors(
                glucose,
                observed,
                gap_age_minutes=gap_age,
                time_of_day=time_of_day,
            )
            similarities = fingerprints @ query
            for offset, (sample, similarity) in enumerate(
                zip(samples, similarities.tolist())
            ):
                record = dataset.records[start + offset]
                record_sha = str(record["sha256"])
                if exclude_sha256 is not None and record_sha == exclude_sha256:
                    continue
                matches.append(
                    {
                        "similarity": float(similarity),
                        "dataset": sample["dataset"],
                        "participant_id": sample["participant_id"],
                        "start_time": sample["start_time"],
                        "source_file": sample["source_file"],
                        "day_sha256": record_sha,
                        "observed_fraction": float(record["observed_fraction"]),
                        "manifest": str(Path(manifest_path)),
                    }
                )
    matches.sort(
        key=lambda match: (
            -match["similarity"],
            match["dataset"],
            match["participant_id"],
            match["start_time"],
            match["source_file"],
        )
    )
    selected = matches[:top_k]
    for rank, match in enumerate(selected, start=1):
        match["rank"] = rank
    return selected
