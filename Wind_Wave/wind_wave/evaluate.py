from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import torch

from .dataset import NormalizationStats
from .indexing import chronological_split
from .train import (
    _device_from_arg,
    _evaluate_loader,
    _build_model,
    _make_loader,
    _output_dir_from_args,
    _preload_spatial_datasets,
    _prepare_datasets,
    _write_csv,
    build_arg_parser as build_train_arg_parser,
    parse_lead_hours,
)


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
    args.model_variant = checkpoint.get("model_variant", args.model_variant)

    wind, wave, initialization_times, _ = _prepare_datasets(args)
    _, _, test_times = chronological_split(initialization_times)
    data_args = args
    if args.preload_spatial:
        wind, wave, data_args = _preload_spatial_datasets(wind, wave, args)
    loader = _make_loader(wind, wave, test_times, stats, data_args, lead_hours, shuffle=False)
    model = _build_model(args, len(lead_hours)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loss, metric_rows = _evaluate_loader(
        model,
        loader,
        stats,
        device,
        lead_hours,
        args.model_variant,
    )
    rows = [{"split": "test", **row} for row in metric_rows]
    _write_csv(_output_dir_from_args(args) / "logs" / "test_metrics_by_lead.csv", rows)
    return {"test_loss": loss}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
