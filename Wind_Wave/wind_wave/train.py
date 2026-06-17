from __future__ import annotations

import argparse
import copy
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
    DEFAULT_INPUT_REGION,
    DEFAULT_LEAD_HOURS,
    DEFAULT_OUTPUT_REGION,
    extracted_data_dir,
    outputs_dir,
    raw_data_dir,
)
from .dataset import (
    NormalizationStats,
    WindWaveSeq2SeqDataset,
    _spatial_indexer,
    compute_normalization_stats,
    open_dataset,
)
from .era5 import (
    drop_extra_dims,
    normalize_spatial_coords,
    normalize_time_coord,
    parse_region,
)
from .extract import ExtractedPair, extract_archives
from .indexing import build_valid_initialization_times, chronological_split
from .losses import masked_mse_loss
from .metrics import circular_mae_degrees, rmse
from .model import ConvLSTMWindWaveModel
from .model import WindWaveV2Model


MODEL_VARIANTS = ("m1", "m2-direct", "m2-wave0-direct", "m2-wave0-residual")


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
    parser.add_argument("--input-region", default=DEFAULT_INPUT_REGION)
    parser.add_argument("--output-region", default=DEFAULT_OUTPUT_REGION)
    parser.add_argument("--model-variant", default="m1", choices=MODEL_VARIANTS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--preload-spatial", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _output_dir_from_args(args: argparse.Namespace) -> Path:
    return outputs_dir() / args.run_name if args.run_name else outputs_dir()


def _preprocess_multifile_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = normalize_time_coord(ds)
    ds = normalize_spatial_coords(ds)
    return drop_extra_dims(ds)


def _open_many(paths: Sequence[Path]) -> xr.Dataset:
    if not paths:
        raise ValueError("At least one dataset path is required")
    if len(paths) == 1:
        return open_dataset(paths[0])

    combined = xr.open_mfdataset(
        paths,
        engine="netcdf4",
        combine="nested",
        concat_dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        chunks={},
        preprocess=_preprocess_multifile_dataset,
    )
    times = pd.Index(pd.to_datetime(combined["time"].values))
    if times.has_duplicates:
        keep = ~times.duplicated(keep="first")
        combined = combined.isel(time=keep)
        times = times[keep]
    if not times.is_monotonic_increasing:
        combined = combined.sortby("time")
    return combined


def _open_pairs(pairs: list[ExtractedPair]) -> tuple[xr.Dataset, xr.Dataset]:
    wind = _open_many([pair.oper_nc for pair in pairs])
    wave = _open_many([pair.wave_nc for pair in pairs])
    return wind, wave


def _preload_spatial_datasets(
    wind: xr.Dataset,
    wave: xr.Dataset,
    args: argparse.Namespace,
) -> tuple[xr.Dataset, xr.Dataset, argparse.Namespace]:
    wind = _spatial_indexer(
        wind,
        args.spatial_stride,
        args.crop_size,
        parse_region(args.input_region),
    ).load()
    wave = _spatial_indexer(
        wave,
        args.spatial_stride,
        args.crop_size,
        parse_region(args.output_region),
    ).load()
    loaded_args = copy.copy(args)
    loaded_args.spatial_stride = 1
    loaded_args.crop_size = None
    return wind, wave, loaded_args


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
        input_region=parse_region(args.input_region),
        output_region=parse_region(args.output_region),
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


def _empty_per_lead_metrics(lead_hours: tuple[int, ...]) -> dict[int, dict[str, list[float]]]:
    return {lead: {"swh": [], "mwp": [], "mwd": []} for lead in lead_hours}


def _metric_or_nan(metric_fn, *args: torch.Tensor) -> float:
    try:
        return float(metric_fn(*args).cpu())
    except ValueError as exc:
        if "finite" not in str(exc):
            raise
        return float("nan")


def _nanmean_or_nan(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


def _append_per_lead_metrics(
    per_lead: dict[int, dict[str, list[float]]],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lead_hours: tuple[int, ...],
) -> None:
    for lead_index, lead in enumerate(lead_hours):
        per_lead[lead]["swh"].append(
            _metric_or_nan(rmse, predictions[:, lead_index, 0], targets[:, lead_index, 0])
        )
        per_lead[lead]["mwp"].append(
            _metric_or_nan(rmse, predictions[:, lead_index, 1], targets[:, lead_index, 1])
        )
        per_lead[lead]["mwd"].append(
            _metric_or_nan(
                circular_mae_degrees,
                predictions[:, lead_index, 3],
                predictions[:, lead_index, 2],
                targets[:, lead_index, 3],
                targets[:, lead_index, 2],
            )
        )


def _finalize_per_lead_metrics(
    per_lead: dict[int, dict[str, list[float]]],
) -> list[dict[str, float | int]]:
    return [
        {
            "lead_hour": lead,
            "rmse_swh": _nanmean_or_nan(values["swh"]),
            "rmse_mwp": _nanmean_or_nan(values["mwp"]),
            "mae_mwd_degrees": _nanmean_or_nan(values["mwd"]),
        }
        for lead, values in per_lead.items()
    ]


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: NormalizationStats,
    device: torch.device,
    lead_hours: tuple[int, ...],
    model_variant: str = "m1",
) -> tuple[float, list[dict[str, float | int]]]:
    model.eval()
    losses = []
    per_lead = _empty_per_lead_metrics(lead_hours)
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            predictions = _predict_batch(model, batch, device, model_variant)
            loss = masked_mse_loss(predictions, targets)
            losses.append(float(loss.detach().cpu()))

            pred_raw = _denormalize_targets(predictions, stats, device)
            target_raw = _denormalize_targets(targets, stats, device)
            _append_per_lead_metrics(per_lead, pred_raw, target_raw, lead_hours)

    return float(np.mean(losses)), _finalize_per_lead_metrics(per_lead)


def _build_model(args: argparse.Namespace, lead_count: int) -> torch.nn.Module:
    if args.model_variant == "m1":
        return ConvLSTMWindWaveModel(
            input_channels=2,
            hidden_channels=args.hidden_channels,
            lead_count=lead_count,
            target_channels=4,
        )
    return WindWaveV2Model(
        hidden_channels=args.hidden_channels,
        lead_count=lead_count,
        target_channels=4,
        use_wave0=args.model_variant in {"m2-wave0-direct", "m2-wave0-residual"},
        residual=args.model_variant == "m2-wave0-residual",
    )


def _predict_batch(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    model_variant: str,
) -> torch.Tensor:
    targets = batch["targets"].to(device)
    inputs = batch["inputs"].to(device)
    if model_variant == "m1":
        return model(inputs, output_size=tuple(targets.shape[-2:]))
    future_wind = batch["future_wind"].to(device)
    wave0 = batch["wave0"].to(device) if "wave0" in batch else None
    return model(
        inputs,
        future_wind=future_wind,
        wave0=wave0,
        output_size=tuple(targets.shape[-2:]),
    )


def _evaluate_persistence_loader(
    loader: DataLoader,
    stats: NormalizationStats,
    device: torch.device,
    lead_hours: tuple[int, ...],
) -> tuple[float, list[dict[str, float | int]]]:
    losses = []
    per_lead = _empty_per_lead_metrics(lead_hours)
    for batch in loader:
        predictions = batch["persistence"].to(device)
        targets = batch["targets"].to(device)
        losses.append(float(masked_mse_loss(predictions, targets).cpu()))
        pred_raw = _denormalize_targets(predictions, stats, device)
        target_raw = _denormalize_targets(targets, stats, device)
        _append_per_lead_metrics(per_lead, pred_raw, target_raw, lead_hours)
    return float(np.mean(losses)), _finalize_per_lead_metrics(per_lead)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1000, 600
    left, top, right, bottom = 90, 65, 950, 515
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=np.float64)
    train_loss = np.asarray([float(row["train_loss"]) for row in rows], dtype=np.float64)
    val_loss = np.asarray([float(row["val_loss"]) for row in rows], dtype=np.float64)
    all_losses = np.concatenate([train_loss, val_loss])
    loss_min = float(all_losses.min())
    loss_max = float(all_losses.max())
    loss_padding = max((loss_max - loss_min) * 0.1, 1e-6)
    loss_min -= loss_padding
    loss_max += loss_padding
    epoch_min = float(epochs.min())
    epoch_max = float(epochs.max())

    def point(epoch: float, loss: float) -> tuple[int, int]:
        x_fraction = 0.5 if epoch_max == epoch_min else (epoch - epoch_min) / (epoch_max - epoch_min)
        y_fraction = (loss - loss_min) / (loss_max - loss_min)
        return (
            int(left + x_fraction * (right - left)),
            int(bottom - y_fraction * (bottom - top)),
        )

    for grid_index in range(6):
        fraction = grid_index / 5
        y = int(bottom - fraction * (bottom - top))
        value = loss_min + fraction * (loss_max - loss_min)
        draw.line((left, y, right, y), fill="#DDDDDD", width=1)
        draw.text((12, y - 7), f"{value:.4f}", fill="#333333")

    draw.line((left, top, left, bottom), fill="#222222", width=2)
    draw.line((left, bottom, right, bottom), fill="#222222", width=2)
    train_points = [point(epoch, loss) for epoch, loss in zip(epochs, train_loss)]
    val_points = [point(epoch, loss) for epoch, loss in zip(epochs, val_loss)]
    if len(train_points) > 1:
        draw.line(train_points, fill="#2166AC", width=4)
        draw.line(val_points, fill="#B2182B", width=4)
    for train_point, val_point in zip(train_points, val_points):
        draw.ellipse((*np.subtract(train_point, 5), *np.add(train_point, 5)), fill="#2166AC")
        draw.ellipse((*np.subtract(val_point, 5), *np.add(val_point, 5)), fill="#B2182B")

    draw.text((left, 20), "Training and Validation Loss", fill="#111111")
    draw.text((left, 545), f"Epoch {int(epoch_min)} to {int(epoch_max)}", fill="#333333")
    draw.line((720, 30, 755, 30), fill="#2166AC", width=4)
    draw.text((765, 22), "Train", fill="#333333")
    draw.line((840, 30, 875, 30), fill="#B2182B", width=4)
    draw.text((885, 22), "Validation", fill="#333333")
    image.save(path, format="PNG")


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
    out_dir = _output_dir_from_args(args)
    checkpoint_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    sample_dir = out_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    wind, wave, initialization_times, lead_hours = _prepare_datasets(args)
    train_times, val_times, test_times = chronological_split(initialization_times)
    data_args = args
    if args.preload_spatial:
        wind, wave, data_args = _preload_spatial_datasets(wind, wave, args)
    stats = compute_normalization_stats(
        wind,
        wave,
        train_times,
        spatial_stride=data_args.spatial_stride,
        crop_size=data_args.crop_size,
        input_region=parse_region(data_args.input_region),
        output_region=parse_region(data_args.output_region),
        history_hours=data_args.history_hours,
        lead_hours=lead_hours,
    )

    with (out_dir / "normalization.json").open("w", encoding="utf-8") as file:
        json.dump(stats.to_dict(), file, indent=2)

    train_loader = _make_loader(wind, wave, train_times, stats, data_args, lead_hours, shuffle=True)
    val_loader = _make_loader(wind, wave, val_times, stats, data_args, lead_hours, shuffle=False)
    test_loader = _make_loader(wind, wave, test_times, stats, data_args, lead_hours, shuffle=False)
    device = _device_from_arg(args.device)
    model = _build_model(args, len(lead_hours)).to(device)
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
            predictions = _predict_batch(model, batch, device, args.model_variant)
            loss = masked_mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_loss, val_metrics = _evaluate_loader(
            model,
            val_loader,
            stats,
            device,
            lead_hours,
            args.model_variant,
        )
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
            "target_channels": 4,
            "input_region": args.input_region,
            "output_region": args.output_region,
            "model_variant": args.model_variant,
        }
        torch.save(checkpoint, checkpoint_dir / "seq2seq_convlstm_latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, checkpoint_dir / "seq2seq_convlstm_best.pt")

    _write_csv(log_dir / "train_log.csv", train_log)
    _write_csv(log_dir / "metrics_by_lead.csv", metrics_rows)
    _plot_training_curve(log_dir / "training_curve.png", train_log)

    val_persistence_loss, val_persistence_metrics = _evaluate_persistence_loader(
        val_loader, stats, device, lead_hours
    )
    test_persistence_loss, test_persistence_metrics = _evaluate_persistence_loader(
        test_loader, stats, device, lead_hours
    )
    baseline_rows = [
        *({"split": "validation", **row} for row in val_persistence_metrics),
        *({"split": "test", **row} for row in test_persistence_metrics),
    ]
    _write_csv(log_dir / "baseline_metrics_by_lead.csv", baseline_rows)

    preview_loader = _make_loader(wind, wave, test_times[:1], stats, data_args, lead_hours, shuffle=False)
    preview_batch = next(iter(preview_loader))
    model.eval()
    with torch.no_grad():
        preview_targets = preview_batch["targets"]
        preview_pred = _predict_batch(model, preview_batch, device, args.model_variant)
    _write_preview(
        sample_dir / "predictions_preview.npz",
        sample_dir / "sample_metadata.csv",
        preview_pred,
        preview_batch["targets"],
        preview_batch,
    )

    return {
        "train_loss": train_log[-1]["train_loss"],
        "val_loss": train_log[-1]["val_loss"],
        "val_persistence_loss": val_persistence_loss,
        "test_persistence_loss": test_persistence_loss,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = train(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
