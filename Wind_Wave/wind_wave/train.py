from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections.abc import Sequence
from contextlib import nullcontext
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
    converted_data_dir,
    data_dir,
    extracted_data_dir,
    outputs_dir,
    raw_data_dir,
)
from .convert_grib import parse_years
from .dataset import (
    FastInMemoryWindWaveCache,
    FastInMemoryWindWaveDataset,
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
    parser.add_argument("--years", default=None)
    parser.add_argument("--data-source", default="zip", choices=("zip", "converted", "zarr"))
    parser.add_argument("--converted-dir", type=Path, default=converted_data_dir())
    parser.add_argument("--zarr-dir", type=Path, default=data_dir() / "zarr")
    parser.add_argument("--metadata-dir", type=Path, default=data_dir() / "metadata")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", "--patience", dest="early_stopping_patience", type=int, default=0)
    parser.add_argument("--precision", default="tf32", choices=("fp32", "tf32", "bf16", "fp16"))
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
    parser.add_argument("--future-wind-mode", default="target", choices=("target", "continuous72"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--preload-spatial", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--fast-in-memory-dataset", action="store_true")
    parser.add_argument("--epoch-pause-seconds", type=float, default=0.0)
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


def _discover_converted_pairs(converted_root: Path, years: Sequence[int]) -> list[ExtractedPair]:
    pairs = []
    for year in years:
        year_dir = Path(converted_root) / str(year)
        wind_nc = year_dir / "wind.nc"
        wave_nc = year_dir / "wave.nc"
        missing = [str(path) for path in (wind_nc, wave_nc) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Converted NetCDF files are missing for year {year}: {', '.join(missing)}"
            )
        pairs.append(
            ExtractedPair(
                archive=year_dir,
                extract_dir=year_dir,
                oper_nc=wind_nc,
                wave_nc=wave_nc,
            )
        )
    return pairs


def _zarr_store_paths(zarr_dir: Path) -> tuple[Path, Path]:
    zarr_dir = Path(zarr_dir)
    return (
        zarr_dir / "era5_wind_025_5N45N_95E150E.zarr",
        zarr_dir / "era5_wave_050_5N45N_95E150E.zarr",
    )


def _open_zarr_pair(zarr_dir: Path) -> tuple[xr.Dataset, xr.Dataset]:
    wind_path, wave_path = _zarr_store_paths(zarr_dir)
    missing = [str(path) for path in (wind_path, wave_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Zarr cache files are missing: {', '.join(missing)}")
    wind = xr.open_zarr(wind_path, chunks=None, consolidated=True)
    wave = xr.open_zarr(wave_path, chunks=None, consolidated=True)
    return wind, wave


def _load_common_times(metadata_dir: Path) -> pd.DatetimeIndex:
    path = Path(metadata_dir) / "common_times.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing Zarr common time index: {path}")
    return pd.DatetimeIndex(pd.to_datetime(np.load(path)))


def _load_times_from_indices(metadata_dir: Path, filename: str) -> list[pd.Timestamp]:
    metadata_dir = Path(metadata_dir)
    index_path = metadata_dir / filename
    if not index_path.exists():
        raise FileNotFoundError(f"Missing Zarr index file: {index_path}")
    common_times = _load_common_times(metadata_dir)
    indices = np.load(index_path).astype(np.int64)
    return [pd.Timestamp(common_times[int(index)]) for index in indices]


def _load_zarr_initialization_times(metadata_dir: Path) -> list[pd.Timestamp]:
    return _load_times_from_indices(metadata_dir, "sample_t0_indices.npy")


def _load_zarr_split_times(metadata_dir: Path) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    return (
        _load_times_from_indices(metadata_dir, "train_indices.npy"),
        _load_times_from_indices(metadata_dir, "val_indices.npy"),
        _load_times_from_indices(metadata_dir, "test_indices.npy"),
    )


def _stat_pair(payload: dict[str, object], group: str, name: str) -> tuple[float, float]:
    section = payload[group][name]
    mean = float(section["mean"])
    std = float(section["std"])
    return mean, std if std > 0 else 1.0


def _load_zarr_normalization_stats(metadata_dir: Path) -> NormalizationStats:
    path = Path(metadata_dir) / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Zarr normalization stats: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    input_pairs = [_stat_pair(payload, "wind", name) for name in ("u10", "v10")]
    target_pairs = [_stat_pair(payload, "wave", name) for name in ("swh", "mwp", "mwd_cos", "mwd_sin")]
    return NormalizationStats(
        input_mean=np.asarray([mean for mean, _ in input_pairs], dtype=np.float32),
        input_std=np.asarray([std for _, std in input_pairs], dtype=np.float32),
        target_mean=np.asarray([mean for mean, _ in target_pairs], dtype=np.float32),
        target_std=np.asarray([std for _, std in target_pairs], dtype=np.float32),
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "cos_mwd", "sin_mwd"),
    )


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
    if args.data_source == "zarr":
        wind, wave = _open_zarr_pair(args.zarr_dir)
        try:
            initialization_times = _load_zarr_initialization_times(args.metadata_dir)
        except FileNotFoundError:
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

    if args.data_source == "converted":
        years = parse_years(args.years or args.year)
        pairs = _discover_converted_pairs(args.converted_dir, years)
        if args.max_archives is not None:
            pairs = pairs[: args.max_archives]
    else:
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
    fast_cache: FastInMemoryWindWaveCache | None = None,
) -> DataLoader:
    if args.fast_in_memory_dataset:
        dataset = FastInMemoryWindWaveDataset(
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
            future_wind_mode=args.future_wind_mode,
            cache=fast_cache,
        )
    else:
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
            future_wind_mode=args.future_wind_mode,
        )
    return DataLoader(dataset, **_build_loader_kwargs(args, shuffle=shuffle))


def _build_loader_kwargs(args: argparse.Namespace, shuffle: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _split_times_for_training(
    args: argparse.Namespace,
    initialization_times: list[pd.Timestamp],
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    if args.data_source == "zarr" and args.max_samples is None:
        try:
            return _load_zarr_split_times(args.metadata_dir)
        except FileNotFoundError:
            pass
    return chronological_split(initialization_times)


def _normalization_stats_for_training(
    wind: xr.Dataset,
    wave: xr.Dataset,
    train_times: list[pd.Timestamp],
    data_args: argparse.Namespace,
    lead_hours: tuple[int, ...],
) -> NormalizationStats:
    if data_args.data_source == "zarr":
        try:
            return _load_zarr_normalization_stats(data_args.metadata_dir)
        except FileNotFoundError:
            pass
    return compute_normalization_stats(
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
    precision: str = "fp32",
    non_blocking: bool = False,
) -> tuple[float, list[dict[str, float | int]]]:
    model.eval()
    losses = []
    per_lead = _empty_per_lead_metrics(lead_hours)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            targets = batch["targets"]
            with _autocast_context(precision, device):
                predictions = _predict_batch(model, batch, device, model_variant)
                loss = masked_mse_loss(predictions, targets)
            losses.append(float(loss.detach().cpu()))

            pred_raw = _denormalize_targets(predictions.float(), stats, device)
            target_raw = _denormalize_targets(targets.float(), stats, device)
            _append_per_lead_metrics(per_lead, pred_raw, target_raw, lead_hours)

    return float(np.mean(losses)), _finalize_per_lead_metrics(per_lead)


def _build_model(
    args: argparse.Namespace,
    lead_count: int,
    stats: NormalizationStats | None = None,
) -> torch.nn.Module:
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
        dropout=args.dropout,
        future_wind_mode=getattr(args, "future_wind_mode", "target"),
        target_mean=None if stats is None else stats.target_mean,
        target_std=None if stats is None else stats.target_std,
    )


def _build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def _should_stop_early(epochs_without_improvement: int, patience: int) -> bool:
    return patience > 0 and epochs_without_improvement >= patience


def _configure_torch_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    allow_tf32 = args.precision in {"tf32", "bf16", "fp16"}
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _autocast_context(precision: str, device: torch.device):
    dtype = _autocast_dtype(precision, device)
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def _build_grad_scaler(precision: str, device: torch.device):
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def _maybe_compile_model(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if not args.compile_model or device.type != "cuda":
        return model
    if not hasattr(torch, "compile"):
        return model
    if not _triton_available():
        print("torch.compile skipped: Triton is not available in this environment.", flush=True)
        return model
    return torch.compile(model)


def _checkpoint_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw_model = getattr(model, "_orig_mod", model)
    return raw_model.state_dict()


def _move_batch_to_device(
    batch: dict[str, object],
    device: torch.device,
    non_blocking: bool,
) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pause_after_epoch(
    epoch: int,
    total_epochs: int,
    seconds: float,
    sleeper=time.sleep,
    logger=print,
) -> None:
    pause_seconds = float(seconds)
    if pause_seconds <= 0 or epoch >= total_epochs:
        return
    logger(f"epoch={epoch} pause_seconds={pause_seconds:.1f}")
    sleeper(pause_seconds)


def _predict_batch(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    model_variant: str,
) -> torch.Tensor:
    targets = batch["targets"]
    inputs = batch["inputs"]
    if model_variant == "m1":
        return model(inputs, output_size=tuple(targets.shape[-2:]))
    future_wind = batch["future_wind"]
    wave0 = batch["wave0"] if "wave0" in batch else None
    future_wind_offsets = batch.get("future_wind_offsets")
    return model(
        inputs,
        future_wind=future_wind,
        wave0=wave0,
        future_wind_offsets=future_wind_offsets,
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
        batch = _move_batch_to_device(batch, device, non_blocking=False)
        predictions = batch["persistence"]
        targets = batch["targets"]
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
        predictions=predictions.detach().float().cpu().numpy(),
        targets=targets.detach().float().cpu().numpy(),
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
    train_times, val_times, test_times = _split_times_for_training(args, initialization_times)
    data_args = args
    if args.preload_spatial:
        wind, wave, data_args = _preload_spatial_datasets(wind, wave, args)
    stats = _normalization_stats_for_training(wind, wave, train_times, data_args, lead_hours)

    with (out_dir / "normalization.json").open("w", encoding="utf-8") as file:
        json.dump(stats.to_dict(), file, indent=2)

    fast_cache = None
    if data_args.fast_in_memory_dataset:
        fast_cache = FastInMemoryWindWaveCache.from_datasets(
            wind,
            wave,
            stats=stats,
            spatial_stride=data_args.spatial_stride,
            crop_size=data_args.crop_size,
            input_region=parse_region(data_args.input_region),
            output_region=parse_region(data_args.output_region),
        )

    train_loader = _make_loader(wind, wave, train_times, stats, data_args, lead_hours, shuffle=True, fast_cache=fast_cache)
    val_loader = _make_loader(wind, wave, val_times, stats, data_args, lead_hours, shuffle=False, fast_cache=fast_cache)
    test_loader = _make_loader(wind, wave, test_times, stats, data_args, lead_hours, shuffle=False, fast_cache=fast_cache)
    device = _device_from_arg(args.device)
    _configure_torch_runtime(args, device)
    model = _build_model(args, len(lead_hours), stats=stats).to(device)
    model = _maybe_compile_model(model, args, device)
    optimizer = _build_optimizer(model, args)
    scaler = _build_grad_scaler(args.precision, device)
    non_blocking = bool(args.pin_memory and device.type == "cuda")

    train_log = []
    metrics_rows = []
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        model.train()
        epoch_losses = []
        train_samples = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            targets = batch["targets"]
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(args.precision, device):
                predictions = _predict_batch(model, batch, device, args.model_variant)
                loss = masked_mse_loss(predictions, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            train_samples += int(targets.shape[0])
            if args.log_every > 0 and step % args.log_every == 0:
                elapsed = time.perf_counter() - train_start
                samples_per_second = train_samples / max(elapsed, 1e-9)
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={epoch_losses[-1]:.6f} samples_per_second={samples_per_second:.1f}",
                    flush=True,
                )

        _sync_cuda(device)
        train_seconds = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_loss, val_metrics = _evaluate_loader(
            model,
            val_loader,
            stats,
            device,
            lead_hours,
            args.model_variant,
            precision=args.precision,
            non_blocking=non_blocking,
        )
        _sync_cuda(device)
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start
        train_loss = float(np.mean(epoch_losses))
        samples_per_second = train_samples / max(train_seconds, 1e-9)
        train_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": epoch_seconds,
                "samples_per_second": samples_per_second,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_seconds={train_seconds:.2f} val_seconds={val_seconds:.2f} "
            f"samples_per_second={samples_per_second:.1f}",
            flush=True,
        )
        for row in val_metrics:
            metrics_rows.append({"epoch": epoch, "split": "validation", **row})

        checkpoint = {
            "model_state_dict": _checkpoint_state_dict(model),
            "stats": stats.to_dict(),
            "lead_hours": lead_hours,
            "history_hours": args.history_hours,
            "hidden_channels": args.hidden_channels,
            "target_channels": 4,
            "input_region": args.input_region,
            "output_region": args.output_region,
            "model_variant": args.model_variant,
            "future_wind_mode": args.future_wind_mode,
            "precision": args.precision,
            "dropout": args.dropout,
        }
        torch.save(checkpoint, checkpoint_dir / "seq2seq_convlstm_latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, checkpoint_dir / "seq2seq_convlstm_best.pt")
        else:
            epochs_without_improvement += 1
            if _should_stop_early(epochs_without_improvement, args.early_stopping_patience):
                break
        _pause_after_epoch(epoch, args.epochs, args.epoch_pause_seconds)

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

    preview_loader = _make_loader(
        wind,
        wave,
        test_times[:1],
        stats,
        data_args,
        lead_hours,
        shuffle=False,
        fast_cache=fast_cache,
    )
    preview_batch = next(iter(preview_loader))
    model.eval()
    with torch.no_grad():
        preview_targets = preview_batch["targets"]
        preview_batch = _move_batch_to_device(preview_batch, device, non_blocking=non_blocking)
        with _autocast_context(args.precision, device):
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
