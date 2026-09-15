"""A small, inspectable dual-stream Transformer encoder for CGM days."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GlucoFMConfig:
    """Architecture settings for five-minute, 24-hour CGM windows."""

    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 4
    feedforward_size: int = 256
    dropout: float = 0.1
    patch_size: int = 12
    trend_windows: tuple[int, ...] = (3, 12, 36)
    event_trend_window: int = 12
    interval_minutes: int = 5
    max_patches: int = 24
    glucose_center_mg_dl: float = 120.0
    glucose_scale_mg_dl: float = 40.0
    gap_age_cap_minutes: float = 60.0
    pool_segments: int = 4

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.hidden_size % 2:
            raise ValueError("hidden_size must be a positive even number")
        if self.num_heads <= 0 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers <= 0 or self.feedforward_size <= 0:
            raise ValueError("num_layers and feedforward_size must be positive")
        if self.patch_size <= 0 or self.interval_minutes <= 0:
            raise ValueError("patch_size and interval_minutes must be positive")
        if not self.trend_windows or any(
            width <= 0 for width in self.trend_windows
        ):
            raise ValueError("trend_windows must contain positive widths")
        if self.event_trend_window not in self.trend_windows:
            raise ValueError("event_trend_window must be present in trend_windows")
        if self.max_patches <= 0:
            raise ValueError("max_patches must be positive")
        if self.pool_segments <= 0 or self.pool_segments > self.max_patches:
            raise ValueError("pool_segments must be between 1 and max_patches")
        if self.glucose_scale_mg_dl <= 0 or self.gap_age_cap_minutes <= 0:
            raise ValueError("glucose scale and gap-age cap must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class SinusoidalPositionEncoding(nn.Module):
    """Add a fixed position code to patch tokens."""

    def __init__(self, hidden_size: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / hidden_size)
        )
        encoding = torch.zeros(max_length, hidden_size)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.shape[1]
        if length > self.encoding.shape[0]:
            raise ValueError(
                f"patch count {length} exceeds configured maximum "
                f"{self.encoding.shape[0]}"
            )
        return inputs + self.encoding[:length].to(
            device=inputs.device, dtype=inputs.dtype
        )


def causal_masked_average(
    values: torch.Tensor, observed_mask: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average current and past observed values without using the future.

    Returns the masked average and the observed fraction inside each trailing
    window. A location with no observed history receives a zero placeholder.
    """

    if values.ndim != 2 or observed_mask.shape != values.shape:
        raise ValueError("values and observed_mask must have shape [batch, time]")
    if window <= 0:
        raise ValueError("window must be positive")

    mask = observed_mask.to(device=values.device, dtype=values.dtype)
    left_padding = (window - 1, 0)
    numerator = F.avg_pool1d(
        F.pad((values * mask).unsqueeze(1), left_padding),
        kernel_size=window,
        stride=1,
    ).squeeze(1) * window
    count = F.avg_pool1d(
        F.pad(mask.unsqueeze(1), left_padding),
        kernel_size=window,
        stride=1,
    ).squeeze(1) * window
    average = numerator / count.clamp_min(1.0)
    average = torch.where(count > 0, average, torch.zeros_like(average))
    return average, count / float(window)


class GlucoFM(nn.Module):
    """Encode a CGM window into hourly tokens and one 128-value embedding.

    ``glucose`` is expressed in mg/dL and has shape ``[batch, time]``.
    ``observed_mask`` is true only for physical measurements. Optional gap-age
    and circular time-of-day features can be supplied by the CSV loader; safe
    defaults are derived when they are omitted.

    The trend bank is causal, but Transformer attention is bidirectional. This
    encoder represents a complete window and is not a real-time forecaster.
    """

    def __init__(self, config: GlucoFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or GlucoFMConfig()
        branch_size = self.config.hidden_size // 2
        trend_count = len(self.config.trend_windows)

        slow_feature_count = 2 * trend_count + 4
        rapid_feature_count = 7
        self.slow_projection = nn.Sequential(
            nn.Linear(self.config.patch_size * slow_feature_count, branch_size),
            nn.GELU(),
            nn.LayerNorm(branch_size),
        )
        self.rapid_projection = nn.Sequential(
            nn.Linear(self.config.patch_size * rapid_feature_count, branch_size),
            nn.GELU(),
            nn.LayerNorm(branch_size),
        )
        self.position = SinusoidalPositionEncoding(
            self.config.hidden_size, self.config.max_patches
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_size,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.config.num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(self.config.hidden_size)
        self.reconstruction_head = nn.Linear(
            self.config.hidden_size, self.config.patch_size
        )
        if self.config.pool_segments == 1:
            self.pool_projection: nn.Module = nn.Identity()
        else:
            pooled_feature_count = self.config.pool_segments + 2
            self.pool_projection = nn.Sequential(
                nn.Linear(
                    pooled_feature_count * self.config.hidden_size,
                    self.config.hidden_size,
                ),
                nn.GELU(),
                nn.LayerNorm(self.config.hidden_size),
            )

    @staticmethod
    def _masked_pool(
        tokens: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        weights = density.unsqueeze(-1)
        total_weight = weights.sum(dim=1)
        weighted = (tokens * weights).sum(dim=1) / total_weight.clamp_min(1.0)
        return torch.where(total_weight > 0, weighted, tokens.mean(dim=1))

    def _default_gap_age(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch, length = mask.shape
        gap_age = torch.empty((batch, length), device=mask.device, dtype=dtype)
        age = torch.full(
            (batch,),
            self.config.gap_age_cap_minutes,
            device=mask.device,
            dtype=dtype,
        )
        for index in range(length):
            age = torch.where(
                mask[:, index],
                torch.zeros_like(age),
                (age + self.config.interval_minutes).clamp_max(
                    self.config.gap_age_cap_minutes
                ),
            )
            gap_age[:, index] = age
        return gap_age

    def _default_time_of_day(
        self, batch: int, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        minutes = torch.arange(length, device=device, dtype=dtype)
        minutes = minutes * self.config.interval_minutes
        angle = 2.0 * math.pi * minutes / (24.0 * 60.0)
        features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return features.unsqueeze(0).expand(batch, -1, -1)

    @staticmethod
    def _as_batched_feature(
        value: torch.Tensor | None,
        *,
        unbatched: bool,
        expected_shape: tuple[int, ...],
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if unbatched:
            value = value.unsqueeze(0)
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        value = value.to(device=device, dtype=dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        return value

    def forward(
        self,
        glucose: torch.Tensor,
        observed_mask: torch.Tensor,
        gap_age_minutes: torch.Tensor | None = None,
        time_of_day: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return reconstructions, patch tokens, streams, and a pooled embedding."""

        unbatched = glucose.ndim == 1
        if unbatched:
            glucose = glucose.unsqueeze(0)
            observed_mask = observed_mask.unsqueeze(0)
        if glucose.ndim != 2 or observed_mask.shape != glucose.shape:
            raise ValueError("glucose and observed_mask must have shape [batch, time]")
        if not glucose.is_floating_point():
            raise TypeError("glucose must be a floating-point tensor")
        if not torch.isfinite(glucose).all():
            raise ValueError("glucose must contain only finite values")

        batch, length = glucose.shape
        if length % self.config.patch_size:
            raise ValueError(
                f"time length {length} must be divisible by patch_size "
                f"{self.config.patch_size}"
            )
        patch_count = length // self.config.patch_size
        if patch_count > self.config.max_patches:
            raise ValueError(
                f"patch count {patch_count} exceeds configured maximum "
                f"{self.config.max_patches}"
            )

        mask = observed_mask.to(device=glucose.device, dtype=torch.bool)
        mask_feature = mask.to(dtype=glucose.dtype)
        normalized = (glucose - self.config.glucose_center_mg_dl) / (
            self.config.glucose_scale_mg_dl
        )
        # Missing placeholders must never influence either stream.
        normalized = torch.where(mask, normalized, torch.zeros_like(normalized))

        expected_gap_shape = (batch, length)
        gap_age = self._as_batched_feature(
            gap_age_minutes,
            unbatched=unbatched,
            expected_shape=expected_gap_shape,
            name="gap_age_minutes",
            device=glucose.device,
            dtype=glucose.dtype,
        )
        if gap_age is None:
            gap_age = self._default_gap_age(mask, glucose.dtype)
        if (gap_age < 0).any():
            raise ValueError("gap_age_minutes cannot be negative")
        gap_feature = gap_age.clamp_max(self.config.gap_age_cap_minutes) / (
            self.config.gap_age_cap_minutes
        )

        expected_time_shape = (batch, length, 2)
        clock = self._as_batched_feature(
            time_of_day,
            unbatched=unbatched,
            expected_shape=expected_time_shape,
            name="time_of_day",
            device=glucose.device,
            dtype=glucose.dtype,
        )
        if clock is None:
            clock = self._default_time_of_day(
                batch, length, glucose.device, glucose.dtype
            )

        trend_values = []
        trend_densities = []
        for width in self.config.trend_windows:
            trend, density = causal_masked_average(normalized, mask, width)
            trend_values.append(trend)
            trend_densities.append(density)
        trend_bank = torch.stack(trend_values, dim=-1)
        trend_density = torch.stack(trend_densities, dim=-1)

        event_index = self.config.trend_windows.index(self.config.event_trend_window)
        event_baseline = trend_bank[..., event_index]
        rapid = torch.where(
            mask, normalized - event_baseline, torch.zeros_like(normalized)
        )
        adjacent_observed = mask[:, 1:] & mask[:, :-1]
        change = rapid[:, 1:] - rapid[:, :-1]
        valid_change = torch.where(
            adjacent_observed, change, torch.zeros_like(change)
        )
        delta = F.pad(valid_change, (1, 0))

        slow_features = torch.cat(
            (
                trend_bank,
                trend_density,
                mask_feature.unsqueeze(-1),
                gap_feature.unsqueeze(-1),
                clock,
            ),
            dim=-1,
        )
        rapid_features = torch.cat(
            (
                rapid.unsqueeze(-1),
                delta.unsqueeze(-1),
                mask_feature.unsqueeze(-1),
                trend_density[..., event_index].unsqueeze(-1),
                gap_feature.unsqueeze(-1),
                clock,
            ),
            dim=-1,
        )

        slow_patches = slow_features.reshape(batch, patch_count, -1)
        rapid_patches = rapid_features.reshape(batch, patch_count, -1)
        tokens = torch.cat(
            (
                self.slow_projection(slow_patches),
                self.rapid_projection(rapid_patches),
            ),
            dim=-1,
        )
        encoded = self.output_norm(self.encoder(self.position(tokens)))

        reconstruction_normalized = self.reconstruction_head(encoded).reshape(
            batch, length
        )
        reconstruction = (
            reconstruction_normalized * self.config.glucose_scale_mg_dl
            + self.config.glucose_center_mg_dl
        )

        patch_density = mask_feature.reshape(
            batch, patch_count, self.config.patch_size
        ).mean(dim=-1)
        global_mean = self._masked_pool(encoded, patch_density)
        if self.config.pool_segments == 1:
            pooled = global_mean
        else:
            weights = patch_density.unsqueeze(-1)
            total_weight = weights.sum(dim=1).clamp_min(1.0)
            weighted_variance = (
                (encoded - global_mean.unsqueeze(1)).square() * weights
            ).sum(dim=1) / total_weight
            global_std = torch.sqrt(weighted_variance + 1e-4)
            segment_means = []
            for segment in range(self.config.pool_segments):
                start = segment * patch_count // self.config.pool_segments
                stop = (segment + 1) * patch_count // self.config.pool_segments
                if start == stop:
                    segment_means.append(torch.zeros_like(global_mean))
                else:
                    segment_means.append(
                        self._masked_pool(
                            encoded[:, start:stop], patch_density[:, start:stop]
                        )
                    )
            pooled_features = torch.cat(
                (global_mean, global_std, *segment_means), dim=-1
            )
            pooled = self.pool_projection(pooled_features)

        output = {
            "reconstruction": reconstruction,
            "embedding": encoded,
            "pooled_embedding": pooled,
            "slow_trends": trend_bank,
            "slow_trend": event_baseline,
            "trend_density": trend_density,
            "rapid_events": rapid,
            "patch_observed_fraction": patch_density,
        }
        if unbatched:
            output = {key: value.squeeze(0) for key, value in output.items()}
        return output
