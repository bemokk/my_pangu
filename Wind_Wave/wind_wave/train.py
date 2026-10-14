from __future__ import annotations

import argparse
import copy
import csv
import json
import random
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
from .losses import (
    LOSS_MODES,
    circular_cosine_sum_and_count,
    masked_mse_sum_and_count,
    wind_wave_loss,
)
from .metrics import (
    circular_absolute_error_sum_and_count,
    squared_error_sum_and_count,
)
from .model import ConvLSTMWindWaveModel
from .model import WindWaveV2Model


MODEL_VARIANTS = (
    "m1",
    "m2-direct",
    "m2-wave0-direct",
    "m2-wave0-direct-unitdir",
    "m2-wave0-residual",
)
CHECKPOINT_VERSION = 2
RESUME_CONFIG_FIELDS = (
    "data_source",
    "year",
    "years",
    "converted_dir",
    "zarr_dir",
    "metadata_dir",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "precision",
    "history_hours",
    "lead_hours",
    "hidden_channels",
    "max_samples",
    "max_archives",
    "spatial_stride",
    "crop_size",
    "input_region",
    "output_region",
    "model_variant",
    "future_wind_mode",
    "split_embargo_hours",
    "loss_mode",
    "direction_loss_weight",
    "preload_spatial",
    "fast_in_memory_dataset",
    "num_workers",
    "seed",
)


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
    parser.add_argument("--split-embargo-hours", type=int, default=0)
    parser.add_argument("--loss-mode", default="mse", choices=LOSS_MODES)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
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
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--step-pause-every", type=int, default=0)
    parser.add_argument("--step-pause-seconds", type=float, default=0.0)
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
    generator: torch.Generator | None = None,
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
    return DataLoader(dataset, **_build_loader_kwargs(args, shuffle=shuffle, generator=generator))


def _build_loader_kwargs(
    args: argparse.Namespace,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _split_times_for_training(
    args: argparse.Namespace,
    initialization_times: list[pd.Timestamp],
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    embargo_hours = int(getattr(args, "split_embargo_hours", 0))
    if args.data_source == "zarr" and args.max_samples is None and embargo_hours == 0:
        try:
            return _load_zarr_split_times(args.metadata_dir)
        except FileNotFoundError:
            pass
    return chronological_split(initialization_times, embargo_hours=embargo_hours)


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


def _empty_per_lead_metrics(lead_hours: tuple[int, ...]) -> dict[int, dict[str, float | int]]:
    return {
        lead: {
            "swh_squared_error_sum": 0.0,
            "swh_count": 0,
            "mwp_squared_error_sum": 0.0,
            "mwp_count": 0,
            "mwd_absolute_error_sum": 0.0,
            "mwd_count": 0,
        }
        for lead in lead_hours
    }


def _append_per_lead_metrics(
    per_lead: dict[int, dict[str, float | int]],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    lead_hours: tuple[int, ...],
) -> None:
    for lead_index, lead in enumerate(lead_hours):
        values = per_lead[lead]
        swh_sum, swh_count = squared_error_sum_and_count(
            predictions[:, lead_index, 0], targets[:, lead_index, 0]
        )
        mwp_sum, mwp_count = squared_error_sum_and_count(
            predictions[:, lead_index, 1], targets[:, lead_index, 1]
        )
        mwd_sum, mwd_count = circular_absolute_error_sum_and_count(
            predictions[:, lead_index, 3],
            predictions[:, lead_index, 2],
            targets[:, lead_index, 3],
            targets[:, lead_index, 2],
        )
        values["swh_squared_error_sum"] += float(swh_sum.cpu())
        values["swh_count"] += swh_count
        values["mwp_squared_error_sum"] += float(mwp_sum.cpu())
        values["mwp_count"] += mwp_count
        values["mwd_absolute_error_sum"] += float(mwd_sum.cpu())
        values["mwd_count"] += mwd_count


def _finalize_per_lead_metrics(
    per_lead: dict[int, dict[str, float | int]],
) -> list[dict[str, float | int]]:
    rows = []
    for lead, values in per_lead.items():
        swh_count = int(values["swh_count"])
        mwp_count = int(values["mwp_count"])
        mwd_count = int(values["mwd_count"])
        rows.append(
            {
                "lead_hour": lead,
                "rmse_swh": (
                    float(np.sqrt(float(values["swh_squared_error_sum"]) / swh_count))
                    if swh_count
                    else float("nan")
                ),
                "rmse_mwp": (
                    float(np.sqrt(float(values["mwp_squared_error_sum"]) / mwp_count))
                    if mwp_count
                    else float("nan")
                ),
                "mae_mwd_degrees": (
                    float(values["mwd_absolute_error_sum"]) / mwd_count
                    if mwd_count
                    else float("nan")
                ),
                "valid_swh": swh_count,
                "valid_mwp": mwp_count,
                "valid_mwd": mwd_count,
            }
        )
    return rows


def _empty_loss_totals(loss_mode: str) -> dict[str, float | int | str]:
    return {
        "mode": loss_mode,
        "scalar_sum": 0.0,
        "scalar_count": 0,
        "direction_sum": 0.0,
        "direction_count": 0,
    }


def _append_loss_totals(
    totals: dict[str, float | int | str],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    stats: NormalizationStats,
    *,
    require_finite_prediction: bool,
) -> None:
    mode = str(totals["mode"])
    if mode == "mse":
        value_sum, count = masked_mse_sum_and_count(
            predictions.float(),
            targets.float(),
            require_finite_prediction=require_finite_prediction,
        )
        totals["scalar_sum"] += float(value_sum.cpu())
        totals["scalar_count"] += count
        return

    scalar_sum, scalar_count = masked_mse_sum_and_count(
        predictions[:, :, :2].float(),
        targets[:, :, :2].float(),
        require_finite_prediction=require_finite_prediction,
    )
    prediction_raw = _denormalize_targets(predictions.float(), stats, predictions.device)
    target_raw = _denormalize_targets(targets.float(), stats, targets.device)
    direction_sum, direction_count = circular_cosine_sum_and_count(
        prediction_raw[:, :, 2],
        prediction_raw[:, :, 3],
        target_raw[:, :, 2],
        target_raw[:, :, 3],
        require_finite_prediction=require_finite_prediction,
    )
    totals["scalar_sum"] += float(scalar_sum.cpu())
    totals["scalar_count"] += scalar_count
    totals["direction_sum"] += float(direction_sum.cpu())
    totals["direction_count"] += direction_count


def _finalize_loss_totals(
    totals: dict[str, float | int | str],
    direction_loss_weight: float,
) -> float:
    scalar_count = int(totals["scalar_count"])
    if scalar_count == 0:
        return float("nan")
    scalar_loss = float(totals["scalar_sum"]) / scalar_count
    if totals["mode"] == "mse":
        return scalar_loss
    direction_count = int(totals["direction_count"])
    if direction_count == 0:
        return float("nan")
    direction_loss = float(totals["direction_sum"]) / direction_count
    return scalar_loss + float(direction_loss_weight) * direction_loss


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: NormalizationStats,
    device: torch.device,
    lead_hours: tuple[int, ...],
    model_variant: str = "m1",
    precision: str = "fp32",
    non_blocking: bool = False,
    loss_mode: str = "mse",
    direction_loss_weight: float = 1.0,
) -> tuple[float, list[dict[str, float | int]]]:
    model.eval()
    loss_totals = _empty_loss_totals(loss_mode)
    per_lead = _empty_per_lead_metrics(lead_hours)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            targets = batch["targets"]
            with _autocast_context(precision, device):
                predictions = _predict_batch(model, batch, device, model_variant)
            _append_loss_totals(
                loss_totals,
                predictions,
                targets,
                stats,
                require_finite_prediction=False,
            )

            pred_raw = _denormalize_targets(predictions.float(), stats, device)
            target_raw = _denormalize_targets(targets.float(), stats, device)
            _append_per_lead_metrics(per_lead, pred_raw, target_raw, lead_hours)

    return (
        _finalize_loss_totals(loss_totals, direction_loss_weight),
        _finalize_per_lead_metrics(per_lead),
    )


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
        use_wave0=args.model_variant
        in {"m2-wave0-direct", "m2-wave0-direct-unitdir", "m2-wave0-residual"},
        residual=args.model_variant == "m2-wave0-residual",
        normalize_direction_output=args.model_variant == "m2-wave0-direct-unitdir",
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _epoch_data_generator(seed: int, epoch: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    return generator


def _checkpoint_training_config(
    args: argparse.Namespace,
    lead_hours: tuple[int, ...],
) -> dict[str, object]:
    config = {}
    for field in RESUME_CONFIG_FIELDS:
        if field == "lead_hours":
            continue
        value = getattr(args, field)
        config[field] = str(value.resolve()) if isinstance(value, Path) else value
    config["lead_hours"] = list(lead_hours)
    return config


def _validate_resume_checkpoint(
    checkpoint: dict[str, object],
    args: argparse.Namespace,
    lead_hours: tuple[int, ...],
) -> None:
    version = int(checkpoint.get("checkpoint_version", 0))
    required = (
        "optimizer_state_dict",
        "scaler_state_dict",
        "progress",
        "rng_state",
        "training_config",
    )
    missing = [name for name in required if name not in checkpoint]
    if version < CHECKPOINT_VERSION or missing:
        details = ", ".join(missing) if missing else f"version={version}"
        raise ValueError(
            "This is a legacy weights-only checkpoint and cannot be resumed losslessly "
            f"({details}). Start a new run or use it only for model initialization."
        )

    expected = _checkpoint_training_config(args, lead_hours)
    saved = checkpoint["training_config"]
    mismatches = []
    for field in RESUME_CONFIG_FIELDS:
        if saved.get(field) != expected.get(field):
            mismatches.append(f"{field}: saved={saved.get(field)!r}, current={expected.get(field)!r}")
    if mismatches:
        raise ValueError("Resume configuration mismatch: " + "; ".join(mismatches))


def _load_checkpoint(path: Path | str) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _build_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    stats: NormalizationStats,
    lead_hours: tuple[int, ...],
    args: argparse.Namespace,
    epoch: int,
    step_in_epoch: int,
    epoch_complete: bool,
    epoch_loss_sum: float,
    epoch_loss_count: int,
    train_samples: int,
    train_seconds: float,
    train_log: list[dict[str, object]],
    metrics_rows: list[dict[str, object]],
    best_val: float,
    epochs_without_improvement: int,
) -> dict[str, object]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": _checkpoint_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": _capture_rng_state(),
        "progress": {
            "epoch": int(epoch),
            "step_in_epoch": int(step_in_epoch),
            "epoch_complete": bool(epoch_complete),
            "epoch_loss_sum": float(epoch_loss_sum),
            "epoch_loss_count": int(epoch_loss_count),
            "train_samples": int(train_samples),
            "train_seconds": float(train_seconds),
        },
        "train_log": list(train_log),
        "metrics_rows": list(metrics_rows),
        "best_val": float(best_val),
        "epochs_without_improvement": int(epochs_without_improvement),
        "training_config": _checkpoint_training_config(args, lead_hours),
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
        "split_embargo_hours": args.split_embargo_hours,
        "loss_mode": args.loss_mode,
        "direction_loss_weight": args.direction_loss_weight,
    }


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


def _pause_after_step(
    step: int,
    total_steps: int,
    every_steps: int,
    seconds: float,
    sleeper=time.sleep,
    logger=print,
) -> None:
    interval = int(every_steps)
    pause_seconds = float(seconds)
    if interval <= 0 or pause_seconds <= 0 or step >= total_steps or step % interval != 0:
        return
    logger(f"step={step}/{total_steps} pause_seconds={pause_seconds:.1f}")
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
    loss_mode: str = "mse",
    direction_loss_weight: float = 1.0,
) -> tuple[float, list[dict[str, float | int]]]:
    loss_totals = _empty_loss_totals(loss_mode)
    per_lead = _empty_per_lead_metrics(lead_hours)
    for batch in loader:
        batch = _move_batch_to_device(batch, device, non_blocking=False)
        predictions = batch["persistence"]
        targets = batch["targets"]
        _append_loss_totals(
            loss_totals,
            predictions,
            targets,
            stats,
            require_finite_prediction=True,
        )
        pred_raw = _denormalize_targets(predictions, stats, device)
        target_raw = _denormalize_targets(targets, stats, device)
        _append_per_lead_metrics(per_lead, pred_raw, target_raw, lead_hours)
    return (
        _finalize_loss_totals(loss_totals, direction_loss_weight),
        _finalize_per_lead_metrics(per_lead),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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
    if args.checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be non-negative")
    if args.step_pause_every < 0 or args.step_pause_seconds < 0:
        raise ValueError("step pause settings must be non-negative")
    if args.split_embargo_hours < 0:
        raise ValueError("split_embargo_hours must be non-negative")
    if args.direction_loss_weight < 0:
        raise ValueError("direction_loss_weight must be non-negative")

    _seed_everything(args.seed)
    lead_hours = parse_lead_hours(args.lead_hours)
    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = _load_checkpoint(args.resume_checkpoint)
        _validate_resume_checkpoint(resume_checkpoint, args, lead_hours)

    out_dir = _output_dir_from_args(args)
    checkpoint_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    sample_dir = out_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    wind, wave, initialization_times, lead_hours = _prepare_datasets(args)
    train_times, val_times, test_times = _split_times_for_training(args, initialization_times)
    print(
        "split "
        f"train={len(train_times)} [{train_times[0]} .. {train_times[-1]}] "
        f"validation={len(val_times)} [{val_times[0]} .. {val_times[-1]}] "
        f"test={len(test_times)} [{test_times[0]} .. {test_times[-1]}] "
        f"embargo_hours={args.split_embargo_hours}",
        flush=True,
    )
    data_args = args
    if args.preload_spatial:
        wind, wave, data_args = _preload_spatial_datasets(wind, wave, args)
    if resume_checkpoint is None:
        stats = _normalization_stats_for_training(wind, wave, train_times, data_args, lead_hours)
    else:
        stats = NormalizationStats.from_dict(resume_checkpoint["stats"])

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

    train_generator = _epoch_data_generator(args.seed, 1)
    train_loader = _make_loader(
        wind,
        wave,
        train_times,
        stats,
        data_args,
        lead_hours,
        shuffle=True,
        fast_cache=fast_cache,
        generator=train_generator,
    )
    val_loader = _make_loader(wind, wave, val_times, stats, data_args, lead_hours, shuffle=False, fast_cache=fast_cache)
    test_loader = _make_loader(wind, wave, test_times, stats, data_args, lead_hours, shuffle=False, fast_cache=fast_cache)
    device = _device_from_arg(args.device)
    _configure_torch_runtime(args, device)
    model = _build_model(args, len(lead_hours), stats=stats).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
    model = _maybe_compile_model(model, args, device)
    optimizer = _build_optimizer(model, args)
    scaler = _build_grad_scaler(args.precision, device)
    non_blocking = bool(args.pin_memory and device.type == "cuda")

    train_log: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    best_val = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    resume_step = 0
    resume_epoch_loss_sum = 0.0
    resume_epoch_loss_count = 0
    resume_train_samples = 0
    resume_train_seconds = 0.0
    resume_rng_state = None

    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        train_log = [dict(row) for row in resume_checkpoint.get("train_log", [])]
        metrics_rows = [dict(row) for row in resume_checkpoint.get("metrics_rows", [])]
        best_val = float(resume_checkpoint.get("best_val", float("inf")))
        epochs_without_improvement = int(resume_checkpoint.get("epochs_without_improvement", 0))
        progress = resume_checkpoint["progress"]
        checkpoint_epoch = int(progress["epoch"])
        if bool(progress["epoch_complete"]):
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = checkpoint_epoch
            resume_step = int(progress["step_in_epoch"])
            resume_epoch_loss_sum = float(progress["epoch_loss_sum"])
            resume_epoch_loss_count = int(progress["epoch_loss_count"])
            resume_train_samples = int(progress["train_samples"])
            resume_train_seconds = float(progress["train_seconds"])
        resume_rng_state = resume_checkpoint["rng_state"]
        print(
            f"resume checkpoint={args.resume_checkpoint} start_epoch={start_epoch} "
            f"completed_step={resume_step} target_epochs={args.epochs}",
            flush=True,
        )

    latest_checkpoint_path = checkpoint_dir / "seq2seq_convlstm_latest.pt"
    best_checkpoint_path = checkpoint_dir / "seq2seq_convlstm_best.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        continuing_partial_epoch = epoch == start_epoch and resume_step > 0
        epoch_loss_sum = resume_epoch_loss_sum if continuing_partial_epoch else 0.0
        epoch_loss_count = resume_epoch_loss_count if continuing_partial_epoch else 0
        train_samples = resume_train_samples if continuing_partial_epoch else 0
        previous_train_seconds = resume_train_seconds if continuing_partial_epoch else 0.0

        train_generator.manual_seed(int(args.seed) + epoch)
        train_iterator = iter(train_loader)
        if continuing_partial_epoch:
            for _ in range(resume_step):
                try:
                    next(train_iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"Checkpoint step {resume_step} exceeds epoch length {len(train_loader)}"
                    ) from exc
        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state)
            resume_rng_state = None

        train_start = time.perf_counter()
        model.train()
        completed_step = resume_step if continuing_partial_epoch else 0
        for step, batch in enumerate(train_iterator, start=completed_step + 1):
            batch = _move_batch_to_device(batch, device, non_blocking=non_blocking)
            targets = batch["targets"]
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(args.precision, device):
                predictions = _predict_batch(model, batch, device, args.model_variant)
                loss = wind_wave_loss(
                    predictions,
                    targets,
                    mode=args.loss_mode,
                    target_mean=stats.target_mean,
                    target_std=stats.target_std,
                    direction_weight=args.direction_loss_weight,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_value = float(loss.detach().cpu())
            epoch_loss_sum += loss_value
            epoch_loss_count += 1
            train_samples += int(targets.shape[0])
            completed_step = step
            if args.log_every > 0 and step % args.log_every == 0:
                elapsed = previous_train_seconds + (time.perf_counter() - train_start)
                samples_per_second = train_samples / max(elapsed, 1e-9)
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss_value:.6f} samples_per_second={samples_per_second:.1f}",
                    flush=True,
                )
            if args.checkpoint_every_steps > 0 and step % args.checkpoint_every_steps == 0:
                _sync_cuda(device)
                elapsed = previous_train_seconds + (time.perf_counter() - train_start)
                checkpoint = _build_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    stats=stats,
                    lead_hours=lead_hours,
                    args=args,
                    epoch=epoch,
                    step_in_epoch=step,
                    epoch_complete=False,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_loss_count=epoch_loss_count,
                    train_samples=train_samples,
                    train_seconds=elapsed,
                    train_log=train_log,
                    metrics_rows=metrics_rows,
                    best_val=best_val,
                    epochs_without_improvement=epochs_without_improvement,
                )
                _atomic_torch_save(checkpoint, latest_checkpoint_path)
                print(f"checkpoint epoch={epoch} step={step}/{len(train_loader)}", flush=True)
            _pause_after_step(
                step,
                len(train_loader),
                args.step_pause_every,
                args.step_pause_seconds,
            )

        _sync_cuda(device)
        train_seconds = previous_train_seconds + (time.perf_counter() - train_start)

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
            loss_mode=args.loss_mode,
            direction_loss_weight=args.direction_loss_weight,
        )
        _sync_cuda(device)
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = train_seconds + val_seconds
        train_loss = epoch_loss_sum / max(epoch_loss_count, 1)
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

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = _build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            stats=stats,
            lead_hours=lead_hours,
            args=args,
            epoch=epoch,
            step_in_epoch=len(train_loader),
            epoch_complete=True,
            epoch_loss_sum=epoch_loss_sum,
            epoch_loss_count=epoch_loss_count,
            train_samples=train_samples,
            train_seconds=train_seconds,
            train_log=train_log,
            metrics_rows=metrics_rows,
            best_val=best_val,
            epochs_without_improvement=epochs_without_improvement,
        )
        _atomic_torch_save(checkpoint, latest_checkpoint_path)
        if improved:
            _atomic_torch_save(checkpoint, best_checkpoint_path)
        _write_csv(log_dir / "train_log.csv", train_log)
        _write_csv(log_dir / "metrics_by_lead.csv", metrics_rows)
        _plot_training_curve(log_dir / "training_curve.png", train_log)
        print(f"checkpoint epoch={epoch} complete=true best_val={best_val:.6f}", flush=True)

        resume_step = 0
        resume_epoch_loss_sum = 0.0
        resume_epoch_loss_count = 0
        resume_train_samples = 0
        resume_train_seconds = 0.0
        if _should_stop_early(epochs_without_improvement, args.early_stopping_patience):
            break
        _pause_after_epoch(epoch, args.epochs, args.epoch_pause_seconds)

    _write_csv(log_dir / "train_log.csv", train_log)
    _write_csv(log_dir / "metrics_by_lead.csv", metrics_rows)
    _plot_training_curve(log_dir / "training_curve.png", train_log)

    val_persistence_loss, val_persistence_metrics = _evaluate_persistence_loader(
        val_loader,
        stats,
        device,
        lead_hours,
        loss_mode=args.loss_mode,
        direction_loss_weight=args.direction_loss_weight,
    )
    test_persistence_loss, test_persistence_metrics = _evaluate_persistence_loader(
        test_loader,
        stats,
        device,
        lead_hours,
        loss_mode=args.loss_mode,
        direction_loss_weight=args.direction_loss_weight,
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
