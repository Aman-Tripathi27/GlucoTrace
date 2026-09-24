"""Frozen research evaluation protocol for learned CGM embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from . import probes
from .model import GlucoFM, GlucoFMConfig
from .pretrain import build_multisource_split

PROTOCOL_VERSION = "1.0"
PROTOCOL_VERSIONS = ("1.0", "1.1", "1.2", "1.3")
# Protocol 1.3 reuses the protocol 1.2 probes and gates on midnight-aligned days.
PROBE_PROTOCOLS = ("1.2", "1.3")
# Protocol 1.2: largest allowed excess of model over summary-baseline source
# probe balanced accuracy. Declared in EVALUATION_1_2.md before training.
SOURCE_PROBE_MARGIN = 0.10


@dataclass(frozen=True)
class MissingnessScenario:
    """One predeclared perturbation in evaluation protocol 1.0."""

    name: str
    kind: str
    amount: float
    repeats: int


PROTOCOL_SCENARIOS = (
    MissingnessScenario("random_10_percent", "random", 0.10, 5),
    MissingnessScenario("random_30_percent", "random", 0.30, 5),
    MissingnessScenario("random_50_percent", "random", 0.50, 5),
    MissingnessScenario("contiguous_30_minutes", "block", 6, 5),
    MissingnessScenario("contiguous_60_minutes", "block", 12, 5),
    MissingnessScenario("contiguous_120_minutes", "block", 24, 5),
    MissingnessScenario("cadence_10_minutes", "cadence", 2, 2),
    MissingnessScenario("cadence_15_minutes", "cadence", 3, 3),
)


def _stable_seed(seed: int, scenario: str, repeat: int, row: int) -> int:
    payload = f"{seed}\x1f{scenario}\x1f{repeat}\x1f{row}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def perturb_missingness(
    glucose: torch.Tensor,
    observed_mask: torch.Tensor,
    scenario: MissingnessScenario,
    *,
    repeat: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a deterministic perturbation without creating glucose values."""

    if glucose.ndim != 2 or observed_mask.shape != glucose.shape:
        raise ValueError("glucose and observed_mask must have shape [batch, time]")
    if repeat < 0 or repeat >= scenario.repeats:
        raise ValueError("repeat is outside the scenario definition")
    if glucose.device.type != "cpu":
        raise ValueError("missingness perturbations are generated on CPU")

    original = observed_mask.bool()
    perturbed = original.clone()
    batch, length = original.shape
    positions = torch.arange(length)
    for row in range(batch):
        generator = torch.Generator().manual_seed(
            _stable_seed(seed, scenario.name, repeat, row)
        )
        if scenario.kind == "random":
            candidates = original[row].nonzero(as_tuple=False).flatten()
            remove_count = max(1, int(round(candidates.numel() * scenario.amount)))
            remove_count = min(remove_count, max(0, candidates.numel() - 1))
            order = torch.randperm(candidates.numel(), generator=generator)
            perturbed[row, candidates[order[:remove_count]]] = False
        elif scenario.kind == "block":
            block_size = int(scenario.amount)
            if block_size <= 0 or block_size >= length:
                raise ValueError("block scenario must be shorter than the sequence")
            start = int(
                torch.randint(length - block_size + 1, (1,), generator=generator)
            )
            perturbed[row, start : start + block_size] = False
        elif scenario.kind == "cadence":
            step = int(scenario.amount)
            if step <= 1:
                raise ValueError("cadence step must be greater than one")
            offset = repeat % step
            perturbed[row] &= positions.remainder(step).eq(offset)
        else:
            raise ValueError(f"unknown missingness scenario kind {scenario.kind!r}")

    removed = original & ~perturbed
    observed_count = original.sum(dim=1).clamp_min(1)
    removed_fraction = removed.sum(dim=1).float() / observed_count.float()
    corrupted = glucose.masked_fill(~perturbed, 0.0)
    return corrupted, perturbed, removed_fraction


def summary_baseline(
    glucose: torch.Tensor, observed_mask: torch.Tensor
) -> torch.Tensor:
    """Return transparent distribution, change, and missingness features."""

    if glucose.ndim != 2 or observed_mask.shape != glucose.shape:
        raise ValueError("glucose and observed_mask must have shape [batch, time]")
    rows = []
    length = glucose.shape[1]
    for values, mask in zip(glucose, observed_mask.bool()):
        observed = values[mask]
        if observed.numel() == 0:
            raise ValueError("baseline requires at least one observed value per day")
        adjacent = mask[1:] & mask[:-1]
        changes = (values[1:] - values[:-1])[adjacent].abs()
        if changes.numel() == 0:
            changes = values.new_zeros(1)
        missing_runs = []
        run = 0
        for is_observed in mask.tolist():
            if is_observed:
                missing_runs.append(run)
                run = 0
            else:
                run += 1
        missing_runs.append(run)
        quantiles = torch.quantile(
            observed, torch.tensor([0.25, 0.5, 0.75], device=observed.device)
        )
        rows.append(
            torch.stack(
                (
                    observed.mean(),
                    observed.std(unbiased=False),
                    observed.min(),
                    quantiles[0],
                    quantiles[1],
                    quantiles[2],
                    observed.max(),
                    changes.mean(),
                    changes.std(unbiased=False),
                    mask.float().mean(),
                    values.new_tensor(max(missing_runs) / length),
                )
            )
        )
    return torch.stack(rows)


def fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("standardizer requires at least two feature rows")
    center = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return center, scale


def embedding_diagnostics(features: torch.Tensor) -> dict[str, float]:
    """Measure representation dispersion without attaching a quality claim."""

    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("embedding diagnostics require at least two rows")
    centered = features - features.mean(dim=0)
    feature_std = centered.std(dim=0, unbiased=False)
    singular_values = torch.linalg.svdvals(centered)
    power = singular_values.square()
    probabilities = power / power.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    normalized = F.normalize(centered, dim=1)
    cosine = normalized @ normalized.T
    off_diagonal = ~torch.eye(
        cosine.shape[0], dtype=torch.bool, device=cosine.device
    )
    return {
        "mean_feature_std": float(feature_std.mean()),
        "minimum_feature_std": float(feature_std.min()),
        "effective_rank": float(effective_rank),
        "mean_absolute_off_diagonal_cosine": float(
            cosine[off_diagonal].abs().mean()
        ),
    }


def source_separability(
    validation_features: torch.Tensor,
    validation_sources: Sequence[str],
    test_features: torch.Tensor,
    test_sources: Sequence[str],
) -> dict[str, float]:
    """Fit validation centroids and report source prediction on held-out days."""

    center, scale = fit_standardizer(validation_features)
    validation = (validation_features - center) / scale
    test = (test_features - center) / scale
    source_names = sorted(set(validation_sources))
    if len(source_names) < 2 or not set(test_sources).issubset(source_names):
        raise ValueError("source evaluation requires matching multiple sources")
    centroids = torch.stack(
        [
            validation[
                torch.tensor([label == source for label in validation_sources])
            ].mean(dim=0)
            for source in source_names
        ]
    )
    distances = torch.cdist(test, centroids)
    predictions = distances.argmin(dim=1)
    targets = torch.tensor([source_names.index(label) for label in test_sources])
    accuracy = (predictions == targets).float().mean()
    recalls = []
    for source_index in range(len(source_names)):
        source_rows = targets == source_index
        recalls.append((predictions[source_rows] == source_index).float().mean())
    pairwise = torch.pdist(centroids)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(torch.stack(recalls).mean()),
        "mean_centroid_distance": float(pairwise.mean()),
    }


def robustness_metrics(
    clean: torch.Tensor, corrupted_repeats: Sequence[torch.Tensor]
) -> dict[str, float]:
    """Compare corrupted queries with their clean vectors and neighborhoods."""

    if clean.ndim != 2 or clean.shape[0] < 2 or not corrupted_repeats:
        raise ValueError("robustness metrics require multiple rows and repeats")
    clean_unit = F.normalize(clean, dim=1)
    clean_similarity = clean_unit @ clean_unit.T
    clean_similarity.fill_diagonal_(-torch.inf)
    neighbor_count = min(5, clean.shape[0] - 1)
    clean_neighbors = clean_similarity.topk(neighbor_count, dim=1).indices

    cosines = []
    relative_distances = []
    top1_agreements = []
    top5_overlaps = []
    row_index = torch.arange(clean.shape[0])
    for corrupted in corrupted_repeats:
        if corrupted.shape != clean.shape:
            raise ValueError("corrupted and clean feature shapes must match")
        cosines.append(F.cosine_similarity(clean, corrupted, dim=1))
        relative_distances.append(
            (corrupted - clean).norm(dim=1) / clean.norm(dim=1).clamp_min(1e-6)
        )
        query_similarity = F.normalize(corrupted, dim=1) @ clean_unit.T
        query_similarity[row_index, row_index] = -torch.inf
        query_neighbors = query_similarity.topk(neighbor_count, dim=1).indices
        top1_agreements.append(
            (query_neighbors[:, 0] == clean_neighbors[:, 0]).float()
        )
        overlap = []
        for clean_row, query_row in zip(clean_neighbors, query_neighbors):
            matches = (clean_row[:, None] == query_row[None, :]).any(dim=1)
            overlap.append(matches.float().mean())
        top5_overlaps.append(torch.stack(overlap))

    cosine = torch.cat(cosines)
    relative = torch.cat(relative_distances)
    return {
        "mean_cosine_similarity": float(cosine.mean()),
        "median_cosine_similarity": float(cosine.median()),
        "p10_cosine_similarity": float(torch.quantile(cosine, 0.10)),
        "median_relative_l2_drift": float(relative.median()),
        "top1_neighbor_agreement": float(torch.cat(top1_agreements).mean()),
        "top5_neighbor_overlap": float(torch.cat(top5_overlaps).mean()),
    }


def predeclared_engineering_checks(
    diagnostics: dict[str, float],
    missingness_report: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the fixed protocol 1.0 representation sanity gates."""

    checks: dict[str, dict[str, Any]] = {
        "noncollapsed_feature_dispersion": {
            "passed": diagnostics["mean_feature_std"] >= 0.05,
            "criterion": "mean_feature_std >= 0.05",
            "value": diagnostics["mean_feature_std"],
        },
        "noncollapsed_effective_rank": {
            "passed": diagnostics["effective_rank"] >= 8.0,
            "criterion": "effective_rank >= 8.0",
            "value": diagnostics["effective_rank"],
        },
    }
    primary_scenarios = (
        "random_30_percent",
        "contiguous_60_minutes",
        "cadence_15_minutes",
    )
    for scenario_name in primary_scenarios:
        metrics = missingness_report[scenario_name]["model"]
        passed = (
            metrics["median_cosine_similarity"] >= 0.80
            and metrics["p10_cosine_similarity"] >= 0.50
            and metrics["top5_neighbor_overlap"] >= 0.50
        )
        checks[f"stability_{scenario_name}"] = {
            "passed": passed,
            "criterion": (
                "median_cosine_similarity >= 0.80; "
                "p10_cosine_similarity >= 0.50; top5_neighbor_overlap >= 0.50"
            ),
            "values": {
                "median_cosine_similarity": metrics["median_cosine_similarity"],
                "p10_cosine_similarity": metrics["p10_cosine_similarity"],
                "top5_neighbor_overlap": metrics["top5_neighbor_overlap"],
            },
        }
    return {
        "all_passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "interpretation_limit": (
            "These fixed thresholds are representation sanity checks, not "
            "scientific or clinical validation."
        ),
    }


def _collect(dataset: Any) -> dict[str, Any]:
    samples = [dataset[index] for index in range(len(dataset))]
    return {
        "glucose": torch.stack([sample["glucose"] for sample in samples]),
        "observed_mask": torch.stack(
            [sample["observed_mask"] for sample in samples]
        ),
        "gap_age_minutes": torch.stack(
            [sample["gap_age_minutes"] for sample in samples]
        ),
        "time_of_day": torch.stack([sample["time_of_day"] for sample in samples]),
        "sources": [sample["dataset"] for sample in samples],
        "participants": [
            (sample["dataset"], sample["participant_id"]) for sample in samples
        ],
    }


@torch.no_grad()
def _encode(
    model: GlucoFM,
    data: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    physical_gap_age: bool,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, data["glucose"].shape[0], batch_size):
        stop = start + batch_size
        gap_age = (
            data["gap_age_minutes"][start:stop].to(device)
            if physical_gap_age
            else None
        )
        output = model(
            data["glucose"][start:stop].to(device),
            data["observed_mask"][start:stop].to(device),
            gap_age,
            data["time_of_day"][start:stop].to(device),
        )
        embeddings.append(output["pooled_embedding"].cpu())
    return torch.cat(embeddings)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path, device: torch.device) -> tuple[GlucoFM, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("research_only") is not True:
        raise ValueError("checkpoint lacks the required research_only marker")
    config_values = dict(checkpoint["config"])
    config_values["trend_windows"] = tuple(config_values["trend_windows"])
    config_values.setdefault("pool_segments", 1)
    model = GlucoFM(GlucoFMConfig(**config_values))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def _verify_checkpoint_corpora(
    checkpoint: dict, corpus_pairs: Sequence[Sequence[str | Path]]
) -> list[dict[str, str]]:
    provided = []
    for manifest, split_file in corpus_pairs:
        manifest_path = Path(manifest)
        split_path = Path(split_file)
        provided.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "split_file": str(split_path),
                "split_sha256": _sha256(split_path),
            }
        )
    expected_hashes = {
        (item["manifest_sha256"], item["split_sha256"])
        for item in checkpoint["corpora"]
    }
    provided_hashes = {
        (item["manifest_sha256"], item["split_sha256"]) for item in provided
    }
    if provided_hashes != expected_hashes:
        raise ValueError("evaluation corpora do not match checkpoint provenance")
    return provided


def _evaluate_missingness(
    model: GlucoFM,
    data: dict[str, Any],
    clean_embeddings: torch.Tensor,
    clean_baseline: torch.Tensor,
    *,
    embedding_center: torch.Tensor,
    embedding_scale: torch.Tensor,
    baseline_center: torch.Tensor,
    baseline_scale: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    standardized_clean = (clean_embeddings - embedding_center) / embedding_scale
    standardized_baseline = (clean_baseline - baseline_center) / baseline_scale
    report: dict[str, dict[str, Any]] = {}
    for scenario in PROTOCOL_SCENARIOS:
        corrupted_embeddings = []
        corrupted_baselines = []
        removed_fractions = []
        for repeat in range(scenario.repeats):
            glucose, mask, removed = perturb_missingness(
                data["glucose"],
                data["observed_mask"],
                scenario,
                repeat=repeat,
                seed=seed,
            )
            corrupted_data = dict(data)
            corrupted_data["glucose"] = glucose
            corrupted_data["observed_mask"] = mask
            embeddings = _encode(
                model,
                corrupted_data,
                device=device,
                batch_size=batch_size,
                physical_gap_age=False,
            )
            baseline = summary_baseline(glucose, mask)
            corrupted_embeddings.append(
                (embeddings - embedding_center) / embedding_scale
            )
            corrupted_baselines.append((baseline - baseline_center) / baseline_scale)
            removed_fractions.append(removed)
        report[scenario.name] = {
            "kind": scenario.kind,
            "amount": scenario.amount,
            "repeats": scenario.repeats,
            "mean_removed_observation_fraction": float(
                torch.cat(removed_fractions).mean()
            ),
            "model": robustness_metrics(
                standardized_clean, corrupted_embeddings
            ),
            "summary_baseline": robustness_metrics(
                standardized_baseline, corrupted_baselines
            ),
        }
    return report


def protocol_1_2_probes(
    model: GlucoFM,
    train: dict[str, Any],
    train_embeddings: torch.Tensor,
    target: dict[str, Any],
    target_embeddings: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Run the protocol 1.2 probes fitted on training and scored on ``target``."""

    train_baseline = summary_baseline(train["glucose"], train["observed_mask"])
    target_baseline = summary_baseline(target["glucose"], target["observed_mask"])

    def hidden(data: dict[str, Any]) -> dict[str, Any]:
        glucose, mask, values, eligible = probes.hide_final_window(
            data["glucose"], data["observed_mask"]
        )
        visible = {
            key: (value[eligible] if isinstance(value, torch.Tensor) else value)
            for key, value in data.items()
        }
        visible["glucose"] = glucose[eligible]
        visible["observed_mask"] = mask[eligible]
        embeddings = _encode(
            model, visible, device=device, batch_size=batch_size, physical_gap_age=False
        )
        return {
            "embeddings": embeddings,
            "baseline": summary_baseline(visible["glucose"], visible["observed_mask"]),
            "persistence": probes.last_visible_hour_mean(
                visible["glucose"], visible["observed_mask"]
            ),
            "targets": values[eligible],
        }

    train_hidden = hidden(train)
    target_hidden = hidden(target)
    utility = {
        "task": "predict mean glucose of the final 6 hours from the first 18 hours",
        "train_days": int(train_hidden["targets"].numel()),
        "target_days": int(target_hidden["targets"].numel()),
        "model": probes.ridge_regression(
            train_hidden["embeddings"],
            train_hidden["targets"],
            target_hidden["embeddings"],
            target_hidden["targets"],
        ),
        "summary_baseline": probes.ridge_regression(
            train_hidden["baseline"],
            train_hidden["targets"],
            target_hidden["baseline"],
            target_hidden["targets"],
        ),
        "last_visible_hour_persistence": probes.persistence_metrics(
            target_hidden["persistence"], target_hidden["targets"]
        ),
    }
    utility["comparison"] = probes.summarize_utility(utility)
    return {
        "linear_source_probe": {
            "model": probes.linear_source_probe(
                train_embeddings,
                train["sources"],
                target_embeddings,
                target["sources"],
            ),
            "summary_baseline": probes.linear_source_probe(
                train_baseline,
                train["sources"],
                target_baseline,
                target["sources"],
            ),
        },
        "same_participant_retrieval": {
            "model": probes.same_participant_retrieval(
                train_embeddings, target_embeddings, target["participants"]
            ),
            "summary_baseline": probes.same_participant_retrieval(
                train_baseline, target_baseline, target["participants"]
            ),
            "interpretation_limit": (
                "High same-participant retrieval is a personal-pattern signal and "
                "equally a re-identification risk for shared fingerprints."
            ),
        },
        "hidden_window_utility": utility,
    }


def protocol_1_2_checks(
    diagnostics: dict[str, float],
    missingness_report: dict[str, dict[str, Any]],
    probe_report: dict[str, Any],
) -> dict[str, Any]:
    """Protocol 1.1 checks plus the declared source and utility gates."""

    result = predeclared_engineering_checks(diagnostics, missingness_report)
    checks = dict(result["checks"])
    source = probe_report["linear_source_probe"]
    source_margin = (
        source["model"]["balanced_accuracy"]
        - source["summary_baseline"]["balanced_accuracy"]
    )
    checks["source_leakage_linear_probe"] = {
        "passed": source_margin <= SOURCE_PROBE_MARGIN,
        "criterion": (
            "model linear-probe balanced accuracy - summary-baseline linear-probe "
            f"balanced accuracy <= {SOURCE_PROBE_MARGIN}"
        ),
        "value": source_margin,
    }
    utility = probe_report["hidden_window_utility"]["comparison"]
    checks["hidden_window_utility"] = {
        "passed": utility["model_minus_baseline_mae_mg_dl"] <= 0.0,
        "criterion": "model ridge MAE <= summary-baseline ridge MAE (mg/dL)",
        "value": utility["model_minus_baseline_mae_mg_dl"],
    }
    result["checks"] = checks
    result["all_passed"] = all(bool(check["passed"]) for check in checks.values())
    return result


def _protocol_1_2_section(
    model: GlucoFM,
    corpus_pairs: Sequence[Sequence[str | Path]],
    target: dict[str, Any],
    target_embeddings: torch.Tensor,
    diagnostics: dict[str, float],
    missingness_report: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    train = _collect(build_multisource_split(corpus_pairs, "train"))
    train_embeddings = _encode(
        model, train, device=device, batch_size=batch_size, physical_gap_age=True
    )
    probe_report = protocol_1_2_probes(
        model,
        train,
        train_embeddings,
        target,
        target_embeddings,
        device=device,
        batch_size=batch_size,
    )
    return {
        "probes": probe_report,
        "predeclared_engineering_checks": protocol_1_2_checks(
            diagnostics, missingness_report, probe_report
        ),
    }


def evaluate_validation_checkpoint(
    checkpoint_path: str | Path,
    corpus_pairs: Sequence[Sequence[str | Path]],
    *,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 7,
    protocol_version: str = "1.1",
) -> dict[str, Any]:
    """Run development diagnostics without constructing a test dataset."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint_path = Path(checkpoint_path)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    corpus_provenance = _verify_checkpoint_corpora(checkpoint, corpus_pairs)
    validation = _collect(build_multisource_split(corpus_pairs, "validation"))
    clean_embeddings = _encode(
        model,
        validation,
        device=device,
        batch_size=batch_size,
        physical_gap_age=True,
    )
    clean_baseline = summary_baseline(
        validation["glucose"], validation["observed_mask"]
    )
    embedding_center, embedding_scale = fit_standardizer(clean_embeddings)
    baseline_center, baseline_scale = fit_standardizer(clean_baseline)
    missingness_report = _evaluate_missingness(
        model,
        validation,
        clean_embeddings,
        clean_baseline,
        embedding_center=embedding_center,
        embedding_scale=embedding_scale,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    diagnostics = embedding_diagnostics(clean_embeddings)
    source_counts = {
        source: validation["sources"].count(source)
        for source in sorted(set(validation["sources"]))
    }
    checks = predeclared_engineering_checks(diagnostics, missingness_report)
    extra: dict[str, Any] = {}
    if protocol_version in PROBE_PROTOCOLS:
        extra = _protocol_1_2_section(
            model,
            corpus_pairs,
            validation,
            clean_embeddings,
            diagnostics,
            missingness_report,
            device=device,
            batch_size=batch_size,
        )
        checks = extra.pop("predeclared_engineering_checks")
    return {
        **extra,
        "protocol_version": protocol_version,
        "evaluation_partition": "validation_development",
        "research_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "corpora": corpus_provenance,
        "validation_days": len(validation["sources"]),
        "validation_participants": len(set(validation["participants"])),
        "validation_days_by_source": source_counts,
        "embedding_diagnostics": diagnostics,
        "missingness_robustness": missingness_report,
        "predeclared_engineering_checks": checks,
        "interpretation_limit": (
            "Standardization and diagnostics use the same development partition; "
            "this report is for candidate selection, not held-out evaluation."
        ),
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    corpus_pairs: Sequence[Sequence[str | Path]],
    *,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 7,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Run a frozen protocol on validation-calibrated held-out test data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint_path = Path(checkpoint_path)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    corpus_provenance = _verify_checkpoint_corpora(checkpoint, corpus_pairs)
    validation = _collect(build_multisource_split(corpus_pairs, "validation"))
    test = _collect(build_multisource_split(corpus_pairs, "test"))

    validation_embeddings = _encode(
        model,
        validation,
        device=device,
        batch_size=batch_size,
        physical_gap_age=True,
    )
    clean_embeddings = _encode(
        model,
        test,
        device=device,
        batch_size=batch_size,
        physical_gap_age=True,
    )
    validation_baseline = summary_baseline(
        validation["glucose"], validation["observed_mask"]
    )
    clean_baseline = summary_baseline(test["glucose"], test["observed_mask"])
    embedding_center, embedding_scale = fit_standardizer(validation_embeddings)
    baseline_center, baseline_scale = fit_standardizer(validation_baseline)
    missingness_report = _evaluate_missingness(
        model,
        test,
        clean_embeddings,
        clean_baseline,
        embedding_center=embedding_center,
        embedding_scale=embedding_scale,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )

    source_counts = {
        source: test["sources"].count(source) for source in sorted(set(test["sources"]))
    }
    diagnostics = embedding_diagnostics(clean_embeddings)
    checks = predeclared_engineering_checks(diagnostics, missingness_report)
    extra: dict[str, Any] = {}
    if protocol_version in PROBE_PROTOCOLS:
        extra = _protocol_1_2_section(
            model,
            corpus_pairs,
            test,
            clean_embeddings,
            diagnostics,
            missingness_report,
            device=device,
            batch_size=batch_size,
        )
        checks = extra.pop("predeclared_engineering_checks")
    return {
        **extra,
        "protocol_version": protocol_version,
        "evaluation_partition": "prospective_test",
        "research_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "corpora": corpus_provenance,
        "validation_days": len(validation["sources"]),
        "test_days": len(test["sources"]),
        "test_participants": len(set(test["participants"])),
        "test_days_by_source": source_counts,
        "embedding_diagnostics": diagnostics,
        "source_separability": {
            "model": source_separability(
                validation_embeddings,
                validation["sources"],
                clean_embeddings,
                test["sources"],
            ),
            "summary_baseline": source_separability(
                validation_baseline,
                validation["sources"],
                clean_baseline,
                test["sources"],
            ),
            "interpretation_limit": (
                "Sources differ in device, population, protocol, and collection era; "
                "separability cannot be attributed to sensor hardware alone."
            ),
        },
        "missingness_robustness": missingness_report,
        "predeclared_engineering_checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen research embedding evaluation."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--corpus",
        action="append",
        nargs=2,
        required=True,
        metavar=("MANIFEST", "SPLITS"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--protocol-version", choices=PROTOCOL_VERSIONS, default="1.0"
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="run development checks without constructing the test partition",
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    evaluation = (
        evaluate_validation_checkpoint if args.validation_only else evaluate_checkpoint
    )
    report = evaluation(
        args.checkpoint,
        args.corpus,
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        protocol_version=args.protocol_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"protocol={report['protocol_version']}")
    print(f"partition={report['evaluation_partition']}")
    print(f"validation_days={report['validation_days']}")
    if "test_days" in report:
        print(f"test_days={report['test_days']}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
