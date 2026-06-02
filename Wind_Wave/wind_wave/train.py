from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_HISTORY_HOURS,
    DEFAULT_LEAD_HOURS,
    extracted_data_dir,
    outputs_dir,
    raw_data_dir,
)
from .dataset import (
    NormalizationStats,
    WindWaveSeq2SeqDataset,
    compute_normalization_stats,
    open_dataset,
)
from .extract import ExtractedPair, extract_archives
from .indexing import build_valid_initialization_times, chronological_split
from .losses import masked_mse_loss
from .metrics import circular_mae_degrees, rmse
from .model import ConvLSTMWindWaveModel


def parse_lead_hours(value: str) -> tuple[int, ...]:
    leads = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not leads:
        raise argparse.ArgumentTypeError("lead hours cannot be empty")
    return leads


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ConvLSTM wind-to-wave seq2seq model.")
    parser.add_argument("--year", default="2025")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--history-hours", type=int, default=DEFAULT_HISTORY_HOURS)
    parser.add_argument(
        "--lead-hours",
        default=",".join(str(value) for value in DEFAULT_LEAD_HOURS),
    )
    parser.add_argument("--hidden-channels", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--spatial-stride", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    times = pd.to_datetime(ds["time"].values)
    _, first_indices = np.unique(times, return_index=True)
    return ds.isel(time=sorted(first_indices)).sortby("time")


def _open_pairs(pairs: list[ExtractedPair]) -> tuple[xr.Dataset, xr.Dataset]:
    wind_parts = [open_dataset(pair.oper_nc) for pair in pairs]
    wave_parts = [open_dataset(pair.wave_nc) for pair in pairs]
    wind = _deduplicate_time(xr.concat(wind_parts, dim="time")).sortby("time")
    wave = _deduplicate_time(xr.concat(wave_parts, dim="time")).sortby("time")

    if len(wind["latitude"]) != len(wave["latitude"]) or len(wind["longitude"]) != len(wave["longitude"]):
        raise ValueError("Wind and wave latitude/longitude grids have different shapes")

    return wind, wave


def _prepare_datasets(args: argparse.Namespace) -> tuple[xr.Dataset, xr.Dataset, list[pd.Timestamp], tuple[int, ...]]:
    lead_hours = parse_lead_hours(args.lead_hours)
    archive_limit = args.max_archives
    if archive_limit is None and args.max_samples is not None:
        archive_limit = 1

    pairs = extract_archives(
        raw_dir=raw_data_dir(args.year),
        extracted_root=extracted_data_dir(args.year),
        limit=archive_limit,
    )
    wind, wave = _open_pairs(pairs)
    initialization_times = build_valid_initialization_times(
        wind_times=pd.to_datetime(wind["time"].values),
        wave_times=pd.to_datetime(wave["time"].values),
        history_hours=args.history_hours,
        lead_hours=lead_hours,
    )
    if args.max_samples is not None:
        initialization_times = initialization_times[: args.max_samples]
    if not initialization_times:
        raise ValueError("No valid initialization times remain after applying history and leads")
    return wind, wave, initialization_times, lead_hours


def _make_loader(
    wind: xr.Dataset,
    wave: xr.Dataset,
    times: list[pd.Timestamp],
    stats: NormalizationStats,
    args: argparse.Namespace,
    lead_hours: tuple[int, ...],
    shuffle: bool,
) -> DataLoader:
    dataset = WindWaveSeq2SeqDataset(
        wind_ds=wind,
        wave_ds=wave,
        initialization_times=times,
        stats=stats,
        history_hours=args.history_hours,
        lead_hours=lead_hours,
        spatial_stride=args.spatial_stride,
        crop_size=args.crop_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def _denormalize_targets(tensor: torch.Tensor, stats: NormalizationStats, device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(stats.target_mean, dtype=tensor.dtype, device=device).view(1, 1, -1, 1, 1)
    std = torch.as_tensor(stats.target_std, dtype=tensor.dtype, device=device).view(1, 1, -1, 1, 1)
    return tensor * std + mean


def _evaluate_loader(
    model: ConvLSTMWindWaveModel,
    loader: DataLoader,
    stats: NormalizationStats,
    device: torch.device,
    lead_hours: tuple[int, ...],
) -> tuple[float, list[dict[str, float | int]]]:
    model.eval()
    losses = []
    metric_rows = []
    per_lead: dict[int, dict[str, list[float]]] = {
        lead: {"swh": [], "mwp": [], "pp1d": [], "mwd": []} for lead in lead_hours
    }
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            predictions = model(inputs)
            loss = masked_mse_loss(predictions, targets)
            losses.append(float(loss.detach().cpu()))

            pred_raw = _denormalize_targets(predictions, stats, device)
            target_raw = _denormalize_targets(targets, stats, device)
            for lead_index, lead in enumerate(lead_hours):
                per_lead[lead]["swh"].append(float(rmse(pred_raw[:, lead_index, 0], target_raw[:, lead_index, 0]).cpu()))
                per_lead[lead]["mwp"].append(float(rmse(pred_raw[:, lead_index, 1], target_raw[:, lead_index, 1]).cpu()))
                per_lead[lead]["pp1d"].append(float(rmse(pred_raw[:, lead_index, 2], target_raw[:, lead_index, 2]).cpu()))
                per_lead[lead]["mwd"].append(
                    float(
                        circular_mae_degrees(
                            pred_raw[:, lead_index, 3],
                            pred_raw[:, lead_index, 4],
                            target_raw[:, lead_index, 3],
                            target_raw[:, lead_index, 4],
                        ).cpu()
                    )
                )

    for lead, values in per_lead.items():
        metric_rows.append(
            {
                "lead_hour": lead,
                "rmse_swh": float(np.mean(values["swh"])),
                "rmse_mwp": float(np.mean(values["mwp"])),
                "rmse_pp1d": float(np.mean(values["pp1d"])),
                "mae_mwd_degrees": float(np.mean(values["mwd"])),
            }
        )

    return float(np.mean(losses)), metric_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_preview(path: Path, metadata_path: Path, predictions: torch.Tensor, targets: torch.Tensor, batch: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predictions=predictions.detach().cpu().numpy(),
        targets=targets.detach().cpu().numpy(),
    )
    rows = [{"t0": value} for value in batch["t0"]]
    _write_csv(metadata_path, rows)


def train(args: argparse.Namespace) -> dict[str, float]:
    lead_hours = parse_lead_hours(args.lead_hours)
    out_dir = outputs_dir()
    checkpoint_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    sample_dir = out_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    wind, wave, initialization_times, lead_hours = _prepare_datasets(args)
    train_times, val_times, test_times = chronological_split(initialization_times)
    stats = compute_normalization_stats(
        wind,
        wave,
        train_times,
        spatial_stride=args.spatial_stride,
        crop_size=args.crop_size,
        history_hours=args.history_hours,
        lead_hours=lead_hours,
    )

    with (out_dir / "normalization.json").open("w", encoding="utf-8") as file:
        json.dump(stats.to_dict(), file, indent=2)

    train_loader = _make_loader(wind, wave, train_times, stats, args, lead_hours, shuffle=True)
    val_loader = _make_loader(wind, wave, val_times, stats, args, lead_hours, shuffle=False)
    device = _device_from_arg(args.device)
    model = ConvLSTMWindWaveModel(
        input_channels=2,
        hidden_channels=args.hidden_channels,
        lead_count=len(lead_hours),
        target_channels=5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_log = []
    metrics_rows = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = masked_mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_loss, val_metrics = _evaluate_loader(model, val_loader, stats, device, lead_hours)
        train_loss = float(np.mean(epoch_losses))
        train_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        for row in val_metrics:
            metrics_rows.append({"epoch": epoch, "split": "validation", **row})

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "stats": stats.to_dict(),
            "lead_hours": lead_hours,
            "history_hours": args.history_hours,
            "hidden_channels": args.hidden_channels,
        }
        torch.save(checkpoint, checkpoint_dir / "seq2seq_convlstm_latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, checkpoint_dir / "seq2seq_convlstm_best.pt")

    _write_csv(log_dir / "train_log.csv", train_log)
    _write_csv(log_dir / "metrics_by_lead.csv", metrics_rows)

    preview_loader = _make_loader(wind, wave, test_times[:1], stats, args, lead_hours, shuffle=False)
    preview_batch = next(iter(preview_loader))
    model.eval()
    with torch.no_grad():
        preview_pred = model(preview_batch["inputs"].to(device))
    _write_preview(
        sample_dir / "predictions_preview.npz",
        sample_dir / "sample_metadata.csv",
        preview_pred,
        preview_batch["targets"],
        preview_batch,
    )

    return {"train_loss": train_log[-1]["train_loss"], "val_loss": train_log[-1]["val_loss"]}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = train(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
