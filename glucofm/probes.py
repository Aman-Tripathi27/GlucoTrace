"""Protocol 1.2 probes: source leakage, day matching, and hidden-window utility.

Every probe is fitted on the training partition and scored on a separate
partition. Features are standardized with training statistics only. These are
representation-research measurements, not clinical performance claims.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.nn import functional as F

HIDDEN_WINDOW_POSITIONS = 72
RIDGE_PENALTY = 10.0
LOGISTIC_PENALTY = 1e-2
LOGISTIC_STEPS = 300


def _standardize(
    train: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    center = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-6)
    return (train - center) / scale, (target - center) / scale


def linear_source_probe(
    train_features: torch.Tensor,
    train_sources: Sequence[str],
    target_features: torch.Tensor,
    target_sources: Sequence[str],
) -> dict[str, float]:
    """Fit a class-balanced L2 logistic regression and report held-out accuracy.

    A linear probe is a stronger leakage test than nearest centroids: a model
    can move centroids together while still encoding source along one axis.
    """

    source_names = sorted(set(train_sources))
    if len(source_names) < 2 or not set(target_sources).issubset(source_names):
        raise ValueError("source probe requires matching multiple sources")
    train, target = _standardize(train_features.double(), target_features.double())
    train_labels = torch.tensor([source_names.index(s) for s in train_sources])
    target_labels = torch.tensor([source_names.index(s) for s in target_sources])
    counts = torch.bincount(train_labels, minlength=len(source_names)).double()
    class_weights = counts.sum() / (len(source_names) * counts)

    weight = torch.zeros(
        train.shape[1], len(source_names), dtype=torch.float64, requires_grad=True
    )
    bias = torch.zeros(len(source_names), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=LOGISTIC_STEPS, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = train @ weight + bias
        loss = F.cross_entropy(logits, train_labels, weight=class_weights)
        loss = loss + LOGISTIC_PENALTY * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        predictions = (target @ weight + bias).argmax(dim=1)
    recalls = [
        (predictions[target_labels == index] == index).double().mean()
        for index in range(len(source_names))
        if (target_labels == index).any()
    ]
    return {
        "accuracy": float((predictions == target_labels).double().mean()),
        "balanced_accuracy": float(torch.stack(recalls).mean()),
        "chance_balanced_accuracy": 1.0 / len(source_names),
    }


def same_participant_retrieval(
    train_features: torch.Tensor,
    target_features: torch.Tensor,
    target_participants: Sequence[tuple[str, str]],
) -> dict[str, float]:
    """Measure how often a day's nearest other day comes from the same person.

    This is reported both as a personal-pattern signal and as a
    re-identification risk: a high value means fingerprints link days.
    """

    _, target = _standardize(train_features, target_features)
    unit = F.normalize(target, dim=1)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(-torch.inf)
    nearest = similarity.argmax(dim=1)
    hits = []
    chance = []
    count = len(target_participants)
    for row, participant in enumerate(target_participants):
        same = sum(1 for other in target_participants if other == participant) - 1
        if same == 0:
            continue
        hits.append(float(target_participants[int(nearest[row])] == participant))
        chance.append(same / (count - 1))
    if not hits:
        raise ValueError("retrieval probe needs participants with at least two days")
    return {
        "eligible_days": len(hits),
        "top1_same_participant": sum(hits) / len(hits),
        "chance_top1_same_participant": sum(chance) / len(chance),
    }


def hide_final_window(
    glucose: torch.Tensor,
    observed_mask: torch.Tensor,
    positions: int = HIDDEN_WINDOW_POSITIONS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hide the final ``positions`` readings and return the hidden-window mean.

    Returns visible glucose, visible mask, the observed mean glucose of the
    hidden window, and a Boolean row filter requiring at least half of the
    hidden window and one visible reading to be physically observed.
    """

    if glucose.ndim != 2 or observed_mask.shape != glucose.shape:
        raise ValueError("glucose and observed_mask must have shape [batch, time]")
    if not 0 < positions < glucose.shape[1]:
        raise ValueError("hidden window must be shorter than the sequence")
    observed = observed_mask.bool()
    hidden = observed[:, -positions:]
    hidden_count = hidden.sum(dim=1)
    hidden_sum = (glucose[:, -positions:] * hidden).sum(dim=1)
    target = hidden_sum / hidden_count.clamp_min(1)
    visible = observed.clone()
    visible[:, -positions:] = False
    eligible = (hidden_count >= positions // 2) & visible.any(dim=1)
    return glucose.masked_fill(~visible, 0.0), visible, target, eligible


def last_visible_hour_mean(
    glucose: torch.Tensor, visible_mask: torch.Tensor, positions: int = 12
) -> torch.Tensor:
    """Persistence baseline: mean of the final visible hour, else all visible."""

    rows = []
    for values, mask in zip(glucose, visible_mask.bool()):
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("persistence baseline requires a visible reading")
        recent = indices[indices > indices[-1] - positions]
        rows.append(values[recent].mean())
    return torch.stack(rows)


def ridge_regression(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    target_features: torch.Tensor,
    target_targets: torch.Tensor,
    *,
    penalty: float = RIDGE_PENALTY,
) -> dict[str, float]:
    """Closed-form ridge fitted on training rows; returns MAE and R squared."""

    train, target = _standardize(train_features.double(), target_features.double())
    offset = train_targets.double().mean()
    gram = train.T @ train + penalty * torch.eye(train.shape[1], dtype=torch.float64)
    coefficients = torch.linalg.solve(gram, train.T @ (train_targets.double() - offset))
    predictions = target @ coefficients + offset
    return _regression_metrics(predictions, target_targets.double())


def _regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    residual = predictions - targets
    total = (targets - targets.mean()).square().sum().clamp_min(1e-12)
    return {
        "mae_mg_dl": float(residual.abs().mean()),
        "r_squared": float(1.0 - residual.square().sum() / total),
    }


def persistence_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    return _regression_metrics(predictions.double(), targets.double())


def summarize_utility(report: dict[str, Any]) -> dict[str, Any]:
    """Return the model-versus-baseline comparison used by the release gate."""

    model = report["model"]["mae_mg_dl"]
    baseline = report["summary_baseline"]["mae_mg_dl"]
    return {
        "model_mae_mg_dl": model,
        "summary_baseline_mae_mg_dl": baseline,
        "model_minus_baseline_mae_mg_dl": model - baseline,
    }
