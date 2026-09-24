"""Source-balanced masked latent pretraining for the CGM encoder."""

from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .corpus import (
    CanonicalCGMDataset,
    MultiSourceCGMDataset,
    SourceBalancedSampler,
)
from .model import GlucoFM, GlucoFMConfig


@dataclass(frozen=True)
class PretrainingConfig:
    """Settings for the research-only latent prediction objective."""

    mask_probability: float = 0.50
    ema_decay: float = 0.996
    variance_weight: float = 0.05
    pooled_consistency_weight: float = 1.0
    pooled_variance_weight: float = 0.25
    pooled_covariance_weight: float = 0.05
    pooled_contrastive_weight: float = 0.25
    contrastive_temperature: float = 0.20
    within_source_negatives: bool = False
    source_adversary_weight: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_probability < 1.0:
            raise ValueError("mask_probability must be in (0, 1)")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        weights = (
            self.variance_weight,
            self.pooled_consistency_weight,
            self.pooled_variance_weight,
            self.pooled_covariance_weight,
            self.pooled_contrastive_weight,
            self.source_adversary_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("objective weights cannot be negative")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive")


def mask_cgm_patches(
    glucose: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    patch_size: int,
    mask_probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hide whole observed patches and return glucose, visible mask, patch mask."""

    if glucose.ndim != 2 or observed_mask.shape != glucose.shape:
        raise ValueError("glucose and observed_mask must have shape [batch, time]")
    if patch_size <= 0 or glucose.shape[1] % patch_size:
        raise ValueError("time length must be divisible by a positive patch_size")
    if not 0.0 < mask_probability < 1.0:
        raise ValueError("mask_probability must be in (0, 1)")

    observed = observed_mask.to(device=glucose.device, dtype=torch.bool)
    batch, length = observed.shape
    patch_count = length // patch_size
    eligible = observed.reshape(batch, patch_count, patch_size).any(dim=-1)
    random_values = torch.rand(
        (batch, patch_count), device=glucose.device, generator=generator
    )
    patch_mask = eligible & (random_values < mask_probability)

    for batch_index in range(batch):
        candidates = eligible[batch_index].nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        if not patch_mask[batch_index].any():
            choice = torch.randint(
                candidates.numel(), (1,), device=glucose.device, generator=generator
            )
            patch_mask[batch_index, candidates[choice]] = True
        if candidates.numel() > 1 and patch_mask[batch_index, candidates].all():
            patch_mask[batch_index, candidates[0]] = False

    hidden_readings = (
        patch_mask.unsqueeze(-1).expand(-1, -1, patch_size).reshape(batch, length)
        & observed
    )
    visible_mask = observed & ~hidden_readings
    corrupted = glucose.masked_fill(hidden_readings, 0.0)
    return corrupted, visible_mask, patch_mask


def augment_cgm_view(
    glucose: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one mask-only scattered, contiguous, or cadence view per row.

    Strategy identifiers are 0 for scattered removal, 1 for one contiguous
    block, and 2 for reduced cadence. At least one physical observation remains
    visible whenever the input row contains an observation.
    """

    if glucose.ndim != 2 or observed_mask.shape != glucose.shape:
        raise ValueError("glucose and observed_mask must have shape [batch, time]")
    observed = observed_mask.to(device=glucose.device, dtype=torch.bool)
    visible = observed.clone()
    batch, length = visible.shape
    strategies = torch.randint(
        3, (batch,), device=glucose.device, generator=generator
    )
    positions = torch.arange(length, device=glucose.device)

    for row in range(batch):
        candidates = observed[row].nonzero(as_tuple=False).flatten()
        if candidates.numel() <= 1:
            continue
        strategy = int(strategies[row])
        if strategy == 0:
            removal_fraction = 0.10 + 0.40 * float(
                torch.rand((), device=glucose.device, generator=generator)
            )
            remove_count = max(1, round(candidates.numel() * removal_fraction))
            remove_count = min(remove_count, candidates.numel() - 1)
            order = torch.randperm(
                candidates.numel(), device=glucose.device, generator=generator
            )
            visible[row, candidates[order[:remove_count]]] = False
        elif strategy == 1:
            maximum = min(24, length - 1)
            minimum = min(6, maximum)
            block_size = int(
                torch.randint(
                    minimum,
                    maximum + 1,
                    (1,),
                    device=glucose.device,
                    generator=generator,
                )
            )
            start = int(
                torch.randint(
                    length - block_size + 1,
                    (1,),
                    device=glucose.device,
                    generator=generator,
                )
            )
            visible[row, start : start + block_size] = False
        else:
            step = int(
                torch.randint(
                    2, 4, (1,), device=glucose.device, generator=generator
                )
            )
            offset = int(
                torch.randint(
                    step, (1,), device=glucose.device, generator=generator
                )
            )
            visible[row] &= positions.remainder(step).eq(offset)

        if not visible[row].any():
            restore_index = int(
                torch.randint(
                    candidates.numel(),
                    (1,),
                    device=glucose.device,
                    generator=generator,
                )
            )
            visible[row, candidates[restore_index]] = True

    corrupted = glucose.masked_fill(~visible, 0.0)
    return corrupted, visible, strategies


def variance_covariance_losses(
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discourage constant and correlated embedding coordinates."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [rows, features]")
    if embeddings.shape[0] < 2:
        zero = embeddings.new_zeros(())
        return zero, zero
    feature_std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(1.0 - feature_std).mean()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    standardized = centered / feature_std.clamp_min(1e-4)
    correlation = standardized.T @ standardized / embeddings.shape[0]
    off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
    covariance_loss = off_diagonal.square().mean()
    return variance_loss, covariance_loss


def symmetric_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    temperature: float,
    source_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Identify matching day views among the other days in a batch.

    With ``source_labels``, negatives are restricted to days from the same
    source, so dataset identity cannot help tell two days apart.
    """

    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError("contrastive views must share shape [batch, features]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if first.shape[0] < 2:
        return first.new_zeros(())
    logits = F.normalize(first, dim=-1) @ F.normalize(second, dim=-1).T
    logits = logits / temperature
    if source_labels is not None:
        same_source = source_labels[:, None] == source_labels[None, :]
        logits = logits.masked_fill(~same_source, -torch.inf)
    targets = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return -grad


def gradient_reversal(inputs: torch.Tensor) -> torch.Tensor:
    """Identity forward; negated gradient backward (domain-adversarial training)."""

    return _GradientReversal.apply(inputs)


class LatentPretrainer(nn.Module):
    """Student encoder trained to predict full-context EMA teacher tokens."""

    def __init__(
        self,
        student: GlucoFM | None = None,
        config: PretrainingConfig | None = None,
        *,
        num_sources: int = 0,
    ) -> None:
        super().__init__()
        self.student = student or GlucoFM()
        self.teacher = copy.deepcopy(self.student)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self.config = config or PretrainingConfig()
        hidden_size = self.student.config.hidden_size
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Training-only source classifier; it is never saved with the encoder.
        self.source_adversary: nn.Module | None = None
        if self.config.source_adversary_weight > 0.0:
            if num_sources < 2:
                raise ValueError("source adversary requires at least two sources")
            self.source_adversary = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, num_sources),
            )

    def train(self, mode: bool = True) -> "LatentPretrainer":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        glucose: torch.Tensor,
        observed_mask: torch.Tensor,
        gap_age_minutes: torch.Tensor,
        time_of_day: torch.Tensor,
        *,
        source_labels: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        needs_sources = (
            self.config.within_source_negatives or self.source_adversary is not None
        )
        if needs_sources and source_labels is None:
            raise ValueError("this objective requires source_labels")
        corrupted, visible_mask, patch_mask = mask_cgm_patches(
            glucose,
            observed_mask,
            patch_size=self.student.config.patch_size,
            mask_probability=self.config.mask_probability,
            generator=generator,
        )
        with torch.no_grad():
            target_output = self.teacher(
                glucose, observed_mask, gap_age_minutes, time_of_day
            )
        student_output = self.student(
            corrupted,
            visible_mask,
            time_of_day=time_of_day,
        )
        augmented, augmented_mask, augmentation_strategy = augment_cgm_view(
            glucose, observed_mask, generator=generator
        )
        augmented_output = self.student(
            augmented,
            augmented_mask,
            time_of_day=time_of_day,
        )
        predicted = self.predictor(student_output["embedding"])
        target = target_output["embedding"].detach()

        token_error = F.smooth_l1_loss(predicted, target, reduction="none").mean(
            dim=-1
        )
        density = target_output["patch_observed_fraction"].detach()
        weights = patch_mask.to(dtype=token_error.dtype) * density
        if not (weights > 0).any():
            raise ValueError("batch contains no observed patch that can be masked")
        latent_loss = (token_error * weights).sum() / weights.sum()

        selected = predicted[patch_mask]
        if selected.shape[0] > 1:
            feature_std = torch.sqrt(
                selected.var(dim=0, unbiased=False) + 1e-4
            )
            variance_loss = F.relu(1.0 - feature_std).mean()
        else:
            variance_loss = predicted.new_zeros(())

        pooled = student_output["pooled_embedding"]
        augmented_pooled = augmented_output["pooled_embedding"]
        target_pooled = target_output["pooled_embedding"].detach()
        teacher_consistency = 0.5 * (
            F.smooth_l1_loss(pooled, target_pooled)
            + F.smooth_l1_loss(augmented_pooled, target_pooled)
        )
        view_consistency = (
            1.0 - F.cosine_similarity(pooled, augmented_pooled, dim=-1)
        ).mean()
        pooled_consistency_loss = teacher_consistency + view_consistency
        pooled_variance_loss, pooled_covariance_loss = variance_covariance_losses(
            torch.cat((pooled, augmented_pooled), dim=0)
        )
        pooled_contrastive_loss = symmetric_contrastive_loss(
            pooled,
            augmented_pooled,
            temperature=self.config.contrastive_temperature,
            source_labels=(
                source_labels if self.config.within_source_negatives else None
            ),
        )
        source_adversary_loss = pooled.new_zeros(())
        source_adversary_accuracy = pooled.new_zeros(())
        if self.source_adversary is not None:
            views = torch.cat((pooled, augmented_pooled), dim=0)
            labels = torch.cat((source_labels, source_labels), dim=0)
            logits = self.source_adversary(gradient_reversal(views))
            source_adversary_loss = F.cross_entropy(logits, labels)
            source_adversary_accuracy = (
                (logits.argmax(dim=1) == labels).float().mean().detach()
            )

        loss = (
            latent_loss
            + self.config.variance_weight * variance_loss
            + self.config.pooled_consistency_weight * pooled_consistency_loss
            + self.config.pooled_variance_weight * pooled_variance_loss
            + self.config.pooled_covariance_weight * pooled_covariance_loss
            + self.config.pooled_contrastive_weight * pooled_contrastive_loss
            + self.config.source_adversary_weight * source_adversary_loss
        )
        return {
            "loss": loss,
            "latent_loss": latent_loss,
            "variance_loss": variance_loss,
            "pooled_consistency_loss": pooled_consistency_loss,
            "pooled_variance_loss": pooled_variance_loss,
            "pooled_covariance_loss": pooled_covariance_loss,
            "pooled_contrastive_loss": pooled_contrastive_loss,
            "source_adversary_loss": source_adversary_loss,
            "source_adversary_accuracy": source_adversary_accuracy,
            "patch_mask": patch_mask,
            "masked_patch_fraction": patch_mask.float().mean(),
            "augmentation_strategy": augmentation_strategy,
        }

    @torch.no_grad()
    def update_teacher(self) -> None:
        decay = self.config.ema_decay
        for teacher_parameter, student_parameter in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_parameter.mul_(decay).add_(
                student_parameter, alpha=1.0 - decay
            )
        for teacher_buffer, student_buffer in zip(
            self.teacher.buffers(), self.student.buffers()
        ):
            teacher_buffer.copy_(student_buffer)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_multisource_split(
    corpus_pairs: Sequence[Sequence[str | Path]], split: str
) -> MultiSourceCGMDataset:
    datasets = [
        CanonicalCGMDataset(
            Path(manifest), split_path=Path(split_file), split=split
        )
        for manifest, split_file in corpus_pairs
    ]
    return MultiSourceCGMDataset(datasets)


def run_epoch(
    pretrainer: LatentPretrainer,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    source_names: Sequence[str] = (),
) -> dict[str, float]:
    training = optimizer is not None
    pretrainer.train(training)
    totals = {
        "loss": 0.0,
        "latent_loss": 0.0,
        "variance_loss": 0.0,
        "pooled_consistency_loss": 0.0,
        "pooled_variance_loss": 0.0,
        "pooled_covariance_loss": 0.0,
        "pooled_contrastive_loss": 0.0,
        "source_adversary_loss": 0.0,
        "source_adversary_accuracy": 0.0,
    }
    batches = 0
    for batch in loader:
        source_labels = None
        if source_names:
            source_labels = torch.tensor(
                [source_names.index(name) for name in batch["dataset"]],
                device=device,
            )
        with torch.set_grad_enabled(training):
            output = pretrainer(
                batch["glucose"].to(device),
                batch["observed_mask"].to(device),
                batch["gap_age_minutes"].to(device),
                batch["time_of_day"].to(device),
                source_labels=source_labels,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                output["loss"].backward()
                optimizer.step()
                pretrainer.update_teacher()
        for name in totals:
            totals[name] += float(output[name].detach())
        batches += 1
    if batches == 0:
        raise ValueError("data loader produced no batches")
    return {name: total / batches for name, total in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-only source-balanced latent pretraining for GlucoTrace."
    )
    parser.add_argument(
        "--corpus",
        action="append",
        nargs=2,
        required=True,
        metavar=("MANIFEST", "SPLITS"),
        help="repeat once per canonical source",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-probability", type=float, default=0.50)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--pooled-consistency-weight", type=float, default=1.0)
    parser.add_argument("--pooled-variance-weight", type=float, default=0.25)
    parser.add_argument("--pooled-covariance-weight", type=float, default=0.05)
    parser.add_argument("--pooled-contrastive-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-temperature", type=float, default=0.20)
    parser.add_argument(
        "--within-source-negatives",
        action="store_true",
        help="contrast each day only against days from the same source",
    )
    parser.add_argument(
        "--source-adversary-weight",
        type=float,
        default=0.0,
        help="weight of a gradient-reversal source classifier (0 disables it)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pool-segments", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("glucofm-pretrained.pt"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = build_multisource_split(args.corpus, "train")
    validation_data = build_multisource_split(args.corpus, "validation")
    sampler = SourceBalancedSampler(
        train_data, num_samples=args.samples_per_epoch, seed=args.seed
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False
    )

    model_config = GlucoFMConfig(pool_segments=args.pool_segments)
    objective_config = PretrainingConfig(
        mask_probability=args.mask_probability,
        ema_decay=args.ema_decay,
        variance_weight=args.variance_weight,
        pooled_consistency_weight=args.pooled_consistency_weight,
        pooled_variance_weight=args.pooled_variance_weight,
        pooled_covariance_weight=args.pooled_covariance_weight,
        pooled_contrastive_weight=args.pooled_contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        within_source_negatives=args.within_source_negatives,
        source_adversary_weight=args.source_adversary_weight,
    )
    source_names = tuple(train_data.source_indices)
    needs_sources = (
        objective_config.within_source_negatives
        or objective_config.source_adversary_weight > 0.0
    )
    pretrainer = LatentPretrainer(
        GlucoFM(model_config), objective_config, num_sources=len(source_names)
    ).to(device)
    trainable_parameters = [
        parameter for parameter in pretrainer.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            pretrainer,
            train_loader,
            device=device,
            optimizer=optimizer,
            source_names=source_names if needs_sources else (),
        )
        validation_metrics = run_epoch(
            pretrainer,
            validation_loader,
            device=device,
            optimizer=None,
            source_names=source_names if needs_sources else (),
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"epoch={epoch + 1} train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f}"
        )

    corpus_provenance = []
    for manifest, split_file in args.corpus:
        manifest_path = Path(manifest)
        split_path = Path(split_file)
        corpus_provenance.append(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "split_file": str(split_path),
                "split_sha256": _sha256(split_path),
            }
        )
    checkpoint = {
        "model_state_dict": pretrainer.teacher.state_dict(),
        "config": asdict(model_config),
        "objective": asdict(objective_config),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "samples_per_epoch": len(sampler),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "corpora": corpus_provenance,
        "source_names": list(source_names),
        "history": history,
        "seed": args.seed,
        "torch_version": str(torch.__version__),
        "research_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
