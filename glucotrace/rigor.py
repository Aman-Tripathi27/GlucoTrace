"""Protocol 1.5 analyses: cadence-harmonized leakage, per-source results,
participant-level bootstrap confidence intervals, and leave-one-dataset-out
transfer.

Sources now differ in sensor cadence (5 versus 15 minutes). A missing-data
pattern that reveals the cadence would identify the source trivially, so the
protocol 1.5 leakage gate thins every day to 15-minute cadence, with a random
phase, before fitting either probe. Confidence intervals resample whole
participants, because days from one person are not independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.nn import functional as F

from . import probes
from .data import thin_to_cadence

HARMONIZE_STEP = 3
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 1905
HARMONIZE_SEED = 1506


def harmonize(data: dict[str, Any], seed: int) -> dict[str, Any]:
    """Thin every day to 15-minute cadence with a seeded random phase."""

    generator = torch.Generator().manual_seed(seed)
    glucose, mask = thin_to_cadence(
        data["glucose"], data["observed_mask"], HARMONIZE_STEP, generator=generator
    )
    thinned = dict(data)
    thinned["glucose"] = glucose
    thinned["observed_mask"] = mask
    return thinned


def cluster_bootstrap(
    participants: Sequence[Any],
    statistic: Callable[[torch.Tensor], float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Percentile 95% interval from resampling participants with replacement."""

    groups: dict[Any, list[int]] = {}
    for row, participant in enumerate(participants):
        groups.setdefault(participant, []).append(row)
    members = list(groups.values())
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(replicates):
        picks = torch.randint(len(members), (len(members),), generator=generator)
        rows = torch.tensor([row for pick in picks.tolist() for row in members[pick]])
        values.append(statistic(rows))
    ordered = torch.tensor(values, dtype=torch.float64)
    return {
        "ci95_low": float(torch.quantile(ordered, 0.025)),
        "ci95_high": float(torch.quantile(ordered, 0.975)),
        "replicates": replicates,
        "unit": "participant",
    }


def _hidden_window(
    model: Any, data: dict[str, Any], encode: Callable, baseline: Callable,
    device: torch.device, batch_size: int,
) -> dict[str, Any]:
    glucose, mask, values, eligible = probes.hide_final_window(
        data["glucose"], data["observed_mask"]
    )
    visible = {
        key: (value[eligible] if isinstance(value, torch.Tensor) else value)
        for key, value in data.items()
    }
    visible["glucose"] = glucose[eligible]
    visible["observed_mask"] = mask[eligible]
    keep = eligible.tolist()
    return {
        "embeddings": encode(
            model, visible, device=device, batch_size=batch_size, physical_gap_age=False
        ),
        "baseline": baseline(visible["glucose"], visible["observed_mask"]),
        "targets": values[eligible].double(),
        "sources": [s for s, k in zip(data["sources"], keep) if k],
        "participants": [p for p, k in zip(data["participants"], keep) if k],
    }


def _utility(train_hidden: dict[str, Any], target_hidden: dict[str, Any]) -> dict[str, Any]:
    model_pred = probes.ridge_predictions(
        train_hidden["embeddings"], train_hidden["targets"], target_hidden["embeddings"]
    )
    base_pred = probes.ridge_predictions(
        train_hidden["baseline"], train_hidden["targets"], target_hidden["baseline"]
    )
    truth = target_hidden["targets"]
    model_error = (model_pred - truth).abs()
    base_error = (base_pred - truth).abs()
    per_source = {}
    for source in sorted(set(target_hidden["sources"])):
        rows = torch.tensor([s == source for s in target_hidden["sources"]])
        per_source[source] = {
            "days": int(rows.sum()),
            "model_mae_mg_dl": float(model_error[rows].mean()),
            "summary_baseline_mae_mg_dl": float(base_error[rows].mean()),
        }
    difference = cluster_bootstrap(
        target_hidden["participants"],
        lambda rows: float(model_error[rows].mean() - base_error[rows].mean()),
    )
    return {
        "model_mae_mg_dl": float(model_error.mean()),
        "summary_baseline_mae_mg_dl": float(base_error.mean()),
        "model_minus_baseline_mae_mg_dl": float(model_error.mean() - base_error.mean()),
        "model_minus_baseline_ci95": difference,
        "per_source": per_source,
    }


def protocol_1_5_extras(
    model: Any,
    train: dict[str, Any],
    target: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Cadence-harmonized source probe, per-source utility, and bootstrap CIs."""

    from .evaluate import _encode, summary_baseline

    train_h = harmonize(train, HARMONIZE_SEED)
    target_h = harmonize(target, HARMONIZE_SEED + 1)
    encode = lambda data: _encode(  # noqa: E731
        model, data, device=device, batch_size=batch_size, physical_gap_age=False
    )
    model_pred, labels, names = probes.linear_source_predictions(
        encode(train_h), train["sources"], encode(target_h), target["sources"]
    )
    base_pred, _, _ = probes.linear_source_predictions(
        summary_baseline(train_h["glucose"], train_h["observed_mask"]),
        train["sources"],
        summary_baseline(target_h["glucose"], target_h["observed_mask"]),
        target["sources"],
    )
    model_ba = probes.balanced_accuracy(model_pred, labels)
    base_ba = probes.balanced_accuracy(base_pred, labels)
    margin_ci = cluster_bootstrap(
        target["participants"],
        lambda rows: probes.balanced_accuracy(model_pred[rows], labels[rows])
        - probes.balanced_accuracy(base_pred[rows], labels[rows]),
    )
    harmonized = {
        "sources": names,
        "model_balanced_accuracy": model_ba,
        "summary_baseline_balanced_accuracy": base_ba,
        "margin": model_ba - base_ba,
        "margin_ci95": margin_ci,
        "chance_balanced_accuracy": 1.0 / len(names),
    }
    utility = _utility(
        _hidden_window(model, train, _encode, summary_baseline, device, batch_size),
        _hidden_window(model, target, _encode, summary_baseline, device, batch_size),
    )
    target_embeddings = encode(target)
    train_embeddings = encode(train)
    retrieval = {}
    for source in sorted(set(target["sources"])):
        rows = torch.tensor([s == source for s in target["sources"]])
        people = [p for p, keep in zip(target["participants"], rows.tolist()) if keep]
        try:
            retrieval[source] = probes.same_participant_retrieval(
                train_embeddings, target_embeddings[rows], people
            )
        except ValueError:
            retrieval[source] = None
    return {
        "cadence_harmonized_source_probe": harmonized,
        "hidden_window_utility_detail": utility,
        "same_participant_retrieval_by_source": retrieval,
    }


def apply_protocol_1_5_gate(checks: dict[str, Any], extras: dict[str, Any], margin_limit: float) -> dict[str, Any]:
    """Replace gate 6 with its cadence-harmonized version and recompute."""

    harmonized = extras["cadence_harmonized_source_probe"]
    updated = dict(checks["checks"])
    updated["source_leakage_linear_probe"] = {
        "passed": harmonized["margin"] <= margin_limit,
        "criterion": (
            "cadence-harmonized (15-minute) model linear-probe balanced accuracy "
            f"- summary-baseline balanced accuracy <= {margin_limit}"
        ),
        "value": harmonized["margin"],
    }
    result = dict(checks)
    result["checks"] = updated
    result["all_passed"] = all(bool(check["passed"]) for check in updated.values())
    return result


def evaluate_transfer(
    checkpoint_path: str | Path,
    train_pairs: Sequence[Sequence[str | Path]],
    heldout_pairs: Sequence[Sequence[str | Path]],
    *,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 7,
) -> dict[str, Any]:
    """Leave-one-dataset-out: score a model on a source it never trained on.

    Ridge and standardization use only the training sources; the held-out
    source contributes only its validation partition as the target.
    """

    from .evaluate import (
        _collect,
        _encode,
        _evaluate_missingness,
        _load_checkpoint,
        _sha256,
        _verify_checkpoint_corpora,
        embedding_diagnostics,
        fit_standardizer,
        summary_baseline,
    )
    from .pretrain import build_multisource_split

    checkpoint_path = Path(checkpoint_path)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    _verify_checkpoint_corpora(checkpoint, train_pairs)
    train = _collect(build_multisource_split(train_pairs, "train"))
    reference = _collect(build_multisource_split(train_pairs, "validation"))
    heldout = _collect(build_multisource_split(heldout_pairs, "validation"))
    if set(heldout["sources"]) & set(train["sources"]):
        raise ValueError("held-out source was part of training")

    utility = _utility(
        _hidden_window(model, train, _encode, summary_baseline, device, batch_size),
        _hidden_window(model, heldout, _encode, summary_baseline, device, batch_size),
    )
    reference_embeddings = _encode(
        model, reference, device=device, batch_size=batch_size, physical_gap_age=True
    )
    heldout_embeddings = _encode(
        model, heldout, device=device, batch_size=batch_size, physical_gap_age=True
    )
    reference_baseline = summary_baseline(reference["glucose"], reference["observed_mask"])
    heldout_baseline = summary_baseline(heldout["glucose"], heldout["observed_mask"])
    embedding_center, embedding_scale = fit_standardizer(reference_embeddings)
    baseline_center, baseline_scale = fit_standardizer(reference_baseline)
    missingness = _evaluate_missingness(
        model,
        heldout,
        heldout_embeddings,
        heldout_baseline,
        embedding_center=embedding_center,
        embedding_scale=embedding_scale,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    primary = ("random_30_percent", "contiguous_60_minutes", "cadence_15_minutes")
    return {
        "analysis": "leave_one_dataset_out",
        "research_only": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_sources": sorted(set(train["sources"])),
        "heldout_sources": sorted(set(heldout["sources"])),
        "heldout_days": len(heldout["sources"]),
        "heldout_participants": len(set(heldout["participants"])),
        "embedding_diagnostics": embedding_diagnostics(heldout_embeddings),
        "hidden_window_utility": utility,
        "missingness_robustness": {name: missingness[name] for name in primary},
        "interpretation_limit": (
            "Transfer to one unseen public cohort; not external clinical validation."
        ),
    }
