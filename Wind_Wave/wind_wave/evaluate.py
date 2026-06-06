from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .dataset import NormalizationStats
from .indexing import chronological_split
from .model import ConvLSTMWindWaveModel
from .train import (
    _device_from_arg,
    _evaluate_loader,
    _make_loader,
    _preload_spatial_datasets,
    _prepare_datasets,
    _write_csv,
    build_arg_parser as build_train_arg_parser,
    parse_lead_hours,
)
from .config import outputs_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_train_arg_parser()
    parser.description = "Evaluate a saved ConvLSTM wind-to-wave seq2seq model."
    parser.add_argument("--checkpoint", required=True)
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = _device_from_arg(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    stats = NormalizationStats.from_dict(checkpoint["stats"])
    lead_hours = tuple(int(value) for value in checkpoint.get("lead_hours", parse_lead_hours(args.lead_hours)))

    wind, wave, initialization_times, _ = _prepare_datasets(args)
    _, _, test_times = chronological_split(initialization_times)
    data_args = args
    if args.preload_spatial:
        wind, wave, data_args = _preload_spatial_datasets(wind, wave, args)
    loader = _make_loader(wind, wave, test_times, stats, data_args, lead_hours, shuffle=False)
    model = ConvLSTMWindWaveModel(
        input_channels=2,
        hidden_channels=int(checkpoint["hidden_channels"]),
        lead_count=len(lead_hours),
        target_channels=int(checkpoint.get("target_channels", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss, metric_rows = _evaluate_loader(model, loader, stats, device, lead_hours)
    rows = [{"split": "test", **row} for row in metric_rows]
    _write_csv(outputs_dir() / "logs" / "test_metrics_by_lead.csv", rows)
    return {"test_loss": loss}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
