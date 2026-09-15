from collections import Counter

import pytest
import torch
from torch.utils.data import Dataset

from glucofm.corpus import MultiSourceCGMDataset, SourceBalancedSampler
from glucofm.model import GlucoFM, GlucoFMConfig
from glucofm.pretrain import (
    LatentPretrainer,
    PretrainingConfig,
    augment_cgm_view,
    mask_cgm_patches,
    symmetric_contrastive_loss,
    variance_covariance_losses,
)


class TinySource(Dataset):
    def __init__(self, source: str, size: int) -> None:
        self.records = [
            {
                "participant_id": f"p{index}",
                "provenance": {
                    "dataset": source,
                    "participant_id": f"p{index}",
                },
            }
            for index in range(size)
        ]
        self.source = source

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"dataset": self.source, "item": str(index)}


def small_encoder() -> GlucoFM:
    return GlucoFM(
        GlucoFMConfig(
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            feedforward_size=32,
            dropout=0.0,
            patch_size=3,
            trend_windows=(1, 3, 6),
            event_trend_window=3,
            max_patches=8,
        )
    )


def test_source_balanced_sampler_is_exact_and_reproducible() -> None:
    dataset = MultiSourceCGMDataset(
        [TinySource("small", 2), TinySource("large", 6)]
    )
    first = SourceBalancedSampler(dataset, num_samples=20, seed=11)
    second = SourceBalancedSampler(dataset, num_samples=20, seed=11)

    first_indices = list(first)
    second_indices = list(second)
    sources = Counter(dataset[index]["dataset"] for index in first_indices)

    assert first_indices == second_indices
    assert sources == {"large": 10, "small": 10}
    first.set_epoch(1)
    assert list(first) != first_indices

    default_sampler = SourceBalancedSampler(dataset, seed=11)
    default_sources = Counter(
        dataset[index]["dataset"] for index in default_sampler
    )
    assert len(default_sampler) == 12
    assert default_sources == {"large": 6, "small": 6}


def test_patch_masking_hides_only_physical_observations() -> None:
    glucose = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    observed = torch.ones(2, 12, dtype=torch.bool)
    observed[0, 4] = False
    observed[1, 8:10] = False
    generator = torch.Generator().manual_seed(3)

    corrupted, visible, patch_mask = mask_cgm_patches(
        glucose,
        observed,
        patch_size=3,
        mask_probability=0.5,
        generator=generator,
    )

    hidden = observed & ~visible
    assert patch_mask.shape == (2, 4)
    assert patch_mask.any(dim=1).all()
    assert not patch_mask.all(dim=1).any()
    assert not (visible & ~observed).any()
    assert torch.equal(corrupted[hidden], torch.zeros_like(corrupted[hidden]))
    assert torch.equal(corrupted[visible], glucose[visible])


def test_augmented_view_is_deterministic_mask_only_and_nonempty() -> None:
    glucose = torch.arange(48, dtype=torch.float32).reshape(4, 12)
    observed = torch.ones(4, 12, dtype=torch.bool)
    observed[0, 4] = False

    first = augment_cgm_view(
        glucose,
        observed,
        generator=torch.Generator().manual_seed(13),
    )
    second = augment_cgm_view(
        glucose,
        observed,
        generator=torch.Generator().manual_seed(13),
    )
    corrupted, visible, strategies = first

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert not (visible & ~observed).any()
    assert visible.any(dim=1).all()
    assert torch.equal(corrupted[~visible], torch.zeros_like(corrupted[~visible]))
    assert strategies.min() >= 0
    assert strategies.max() <= 2


def test_variance_covariance_losses_detect_collapse_and_redundancy() -> None:
    collapsed = torch.ones(8, 4)
    generator = torch.Generator().manual_seed(5)
    independent = torch.randn(256, 4, generator=generator)
    shared = torch.randn(256, 1, generator=generator)
    redundant = shared.repeat(1, 4)

    collapsed_variance, _ = variance_covariance_losses(collapsed)
    independent_variance, independent_covariance = variance_covariance_losses(
        independent
    )
    _, redundant_covariance = variance_covariance_losses(redundant)

    assert collapsed_variance > independent_variance
    assert redundant_covariance > independent_covariance


def test_contrastive_loss_rewards_matching_views() -> None:
    first = torch.eye(4)
    matching = first.clone()
    mismatched = first.roll(1, dims=0)

    matching_loss = symmetric_contrastive_loss(
        first, matching, temperature=0.20
    )
    mismatched_loss = symmetric_contrastive_loss(
        first, mismatched, temperature=0.20
    )

    assert matching_loss < mismatched_loss


def test_latent_objective_backpropagates_without_teacher_gradients() -> None:
    pretrainer = LatentPretrainer(
        small_encoder(),
        PretrainingConfig(
            mask_probability=0.5,
            ema_decay=0.9,
            variance_weight=0.05,
        ),
    )
    pretrainer.train()
    glucose = torch.linspace(80.0, 150.0, 24).reshape(2, 12)
    observed = torch.ones(2, 12, dtype=torch.bool)
    gap_age = torch.zeros(2, 12)
    angle = torch.arange(12) * (2.0 * torch.pi / 288.0)
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    clock = clock.unsqueeze(0).expand(2, -1, -1)

    output = pretrainer(
        glucose,
        observed,
        gap_age,
        clock,
        generator=torch.Generator().manual_seed(5),
    )
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert output["patch_mask"].shape == (2, 4)
    assert any(parameter.grad is not None for parameter in pretrainer.student.parameters())
    assert any(parameter.grad is not None for parameter in pretrainer.predictor.parameters())
    assert all(parameter.grad is None for parameter in pretrainer.teacher.parameters())
    assert not pretrainer.teacher.training
    assert torch.isfinite(output["pooled_consistency_loss"])
    assert torch.isfinite(output["pooled_variance_loss"])
    assert torch.isfinite(output["pooled_covariance_loss"])
    assert torch.isfinite(output["pooled_contrastive_loss"])


def test_teacher_update_is_exponential_moving_average() -> None:
    pretrainer = LatentPretrainer(
        small_encoder(), PretrainingConfig(ema_decay=0.9)
    )
    student_parameter = next(pretrainer.student.parameters())
    teacher_parameter = next(pretrainer.teacher.parameters())
    before = teacher_parameter.detach().clone()
    with torch.no_grad():
        student_parameter.add_(1.0)

    pretrainer.update_teacher()

    assert torch.allclose(teacher_parameter, before + 0.1)


def test_pretraining_config_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="mask_probability"):
        PretrainingConfig(mask_probability=1.0)
