import pytest
import torch

from glucofm.model import GlucoFM, GlucoFMConfig, causal_masked_average
from glucofm.train import make_masked_batch


def small_model() -> GlucoFM:
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


def test_default_model_is_a_128_value_day_encoder() -> None:
    model = GlucoFM().eval()
    glucose = torch.full((288,), 110.0)
    observed = torch.ones(288, dtype=torch.bool)

    with torch.no_grad():
        output = model(glucose, observed)

    assert output["embedding"].shape == (24, 128)
    assert output["pooled_embedding"].shape == (128,)
    assert output["reconstruction"].shape == (288,)


def test_statistical_pool_uses_global_and_segment_features() -> None:
    model = small_model().eval()
    assert model.config.pool_segments == 4
    assert model.pool_projection[0].in_features == 6 * 16

    glucose = torch.linspace(80.0, 150.0, 12).unsqueeze(0)
    observed = torch.ones(1, 12, dtype=torch.bool)
    observed[:, :3] = False

    with torch.no_grad():
        pooled = model(glucose, observed)["pooled_embedding"]

    assert pooled.shape == (1, 16)
    assert torch.isfinite(pooled).all()


def test_model_shapes_streams_and_backward() -> None:
    model = small_model()
    glucose = torch.linspace(80.0, 150.0, 24).reshape(2, 12)
    observed = torch.ones(2, 12, dtype=torch.bool)
    observed[0, 4:6] = False

    output = model(glucose, observed)
    output["reconstruction"].square().mean().backward()

    assert output["reconstruction"].shape == (2, 12)
    assert output["embedding"].shape == (2, 4, 16)
    assert output["pooled_embedding"].shape == (2, 16)
    assert output["slow_trends"].shape == (2, 12, 3)
    assert output["rapid_events"].shape == (2, 12)
    assert output["patch_observed_fraction"].shape == (2, 4)
    assert torch.equal(
        output["rapid_events"][0, 4:6], torch.zeros(2)
    )
    assert all(torch.isfinite(value).all() for value in output.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_unbatched_optional_features() -> None:
    model = small_model().eval()
    glucose = torch.linspace(90.0, 120.0, 12)
    observed = torch.ones(12, dtype=torch.bool)
    gap_age = torch.zeros(12)
    angle = torch.arange(12) * (2.0 * torch.pi / 288.0)
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    output = model(glucose, observed, gap_age, clock)

    assert output["embedding"].shape == (4, 16)
    assert output["pooled_embedding"].shape == (16,)


def test_causal_trends_do_not_read_future_values() -> None:
    observed = torch.ones(1, 12, dtype=torch.bool)
    original = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    changed_future = original.clone()
    changed_future[:, 8:] = 10_000.0

    first, first_density = causal_masked_average(original, observed, window=6)
    second, second_density = causal_masked_average(
        changed_future, observed, window=6
    )

    assert torch.equal(first[:, :8], second[:, :8])
    assert torch.equal(first_density, second_density)


def test_missing_placeholder_values_cannot_change_the_encoding() -> None:
    model = small_model().eval()
    observed = torch.ones(1, 12, dtype=torch.bool)
    observed[:, 4:7] = False
    first = torch.full((1, 12), 100.0)
    second = first.clone()
    second[:, 4:7] = torch.tensor([-1_000.0, 50_000.0, -20.0])

    with torch.no_grad():
        first_output = model(first, observed)
        second_output = model(second, observed)

    for key in first_output:
        assert torch.allclose(first_output[key], second_output[key])


def test_masked_batch_only_hides_observed_values() -> None:
    torch.manual_seed(0)
    glucose = torch.randn(2, 8)
    observed = torch.tensor(
        [[True, True, False, True, True, True, False, True]] * 2
    )

    corrupted, visible, hidden = make_masked_batch(
        glucose, observed, mask_probability=0.5
    )

    assert hidden.any(dim=1).all()
    assert not (hidden & ~observed).any()
    assert torch.equal(visible, observed & ~hidden)
    assert torch.equal(corrupted[hidden], torch.zeros_like(corrupted[hidden]))


def test_config_and_length_validation() -> None:
    with pytest.raises(ValueError, match="event_trend_window"):
        GlucoFMConfig(trend_windows=(3, 6), event_trend_window=12)

    with pytest.raises(ValueError, match="divisible by patch_size"):
        small_model()(torch.ones(10), torch.ones(10, dtype=torch.bool))
