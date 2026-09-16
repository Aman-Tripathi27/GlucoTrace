"""Minimal masked-reconstruction example for the dual-stream encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CGMWindowDataset, load_cgm_csv
from .model import GlucoFM, GlucoFMConfig


def make_masked_batch(
    glucose: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    mask_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hide a random subset of real measurements for reconstruction training."""

    if not 0.0 < mask_probability <= 1.0:
        raise ValueError("mask_probability must be in (0, 1]")
    hidden = observed_mask.bool() & (
        torch.rand(observed_mask.shape, device=glucose.device) < mask_probability
    )

    # Give every non-empty sequence at least one supervised position.
    for batch_index in range(glucose.shape[0]):
        if not hidden[batch_index].any() and observed_mask[batch_index].any():
            first = observed_mask[batch_index].nonzero(as_tuple=False)[0, 0]
            hidden[batch_index, first] = True

    visible = observed_mask.bool() & ~hidden
    corrupted = glucose.masked_fill(hidden, 0.0)
    return corrupted, visible, hidden


def train_epoch(
    model: GlucoFM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mask_probability: float,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        glucose = batch["glucose"].to(device)
        observed = batch["observed_mask"].to(device)
        corrupted, visible, hidden = make_masked_batch(
            glucose, observed, mask_probability=mask_probability
        )
        # Gap age is intentionally recomputed from ``visible`` so a hidden
        # target cannot leak through the physical-mask-derived gap feature.
        prediction = model(
            corrupted,
            visible,
            time_of_day=batch["time_of_day"].to(device),
        )["reconstruction"]
        if not hidden.any():
            continue
        loss = torch.nn.functional.mse_loss(prediction[hidden], glucose[hidden])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
    if batches == 0:
        raise ValueError("no trainable batches; check the CSV and window settings")
    return total_loss / batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the small GlucoTrace encoder with masked reconstruction."
    )
    parser.add_argument("csv", type=Path, help="CSV with timestamp and glucose columns")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--glucose-col", default="glucose")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=288)
    parser.add_argument("--stride", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("glucofm.pt"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    series = load_cgm_csv(
        args.csv,
        timestamp_col=args.timestamp_col,
        glucose_col=args.glucose_col,
        interval_minutes=args.interval_minutes,
    )
    dataset = CGMWindowDataset(
        series, window_size=args.window_size, stride=args.stride
    )
    if not dataset:
        raise ValueError("no windows produced; reduce --window-size or add more data")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.window_size % 12:
        raise ValueError("--window-size must be divisible by the 12-reading patch")
    config = GlucoFMConfig(max_patches=max(24, args.window_size // 12))
    model = GlucoFM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            mask_probability=args.mask_probability,
        )
        print(f"epoch={epoch} loss={loss:.6f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "input_unit": "mg/dL",
        "source_summary": {"mean_mg_dl": series.mean, "std_mg_dl": series.std},
        "interval_minutes": series.interval_minutes,
    }
    torch.save(checkpoint, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
