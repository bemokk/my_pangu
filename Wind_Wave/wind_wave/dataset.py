from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from .config import DEFAULT_HISTORY_HOURS, DEFAULT_LEAD_HOURS
from .era5 import (
    Region,
    direction_degrees_to_unit,
    drop_extra_dims,
    find_data_var,
    normalize_spatial_coords,
    normalize_time_coord,
    select_region,
)


WIND_CANDIDATES = {
    "u10": ("u10", "10m_u_component_of_wind"),
    "v10": ("v10", "10m_v_component_of_wind"),
}

WAVE_CANDIDATES = {
    "swh": ("swh", "significant_height_of_combined_wind_waves_and_swell"),
    "mwp": ("mwp", "mean_wave_period"),
    "mwd": ("mwd", "mean_wave_direction"),
}


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    input_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @classmethod
    def identity(
        cls,
        input_names: tuple[str, ...],
        target_names: tuple[str, ...],
    ) -> "NormalizationStats":
        return cls(
            input_mean=np.zeros(len(input_names), dtype=np.float32),
            input_std=np.ones(len(input_names), dtype=np.float32),
            target_mean=np.zeros(len(target_names), dtype=np.float32),
            target_std=np.ones(len(target_names), dtype=np.float32),
            input_names=input_names,
            target_names=target_names,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "input_names": list(self.input_names),
            "target_names": list(self.target_names),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationStats":
        return cls(
            input_mean=np.asarray(data["input_mean"], dtype=np.float32),
            input_std=np.asarray(data["input_std"], dtype=np.float32),
            target_mean=np.asarray(data["target_mean"], dtype=np.float32),
            target_std=np.asarray(data["target_std"], dtype=np.float32),
            input_names=tuple(data["input_names"]),
            target_names=tuple(data["target_names"]),
        )


def open_dataset(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path, engine="netcdf4")
    ds = normalize_time_coord(ds)
    ds = normalize_spatial_coords(ds)
    ds = drop_extra_dims(ds)
    return ds


def _spatial_indexer(
    ds: xr.Dataset,
    spatial_stride: int,
    crop_size: int | None,
    region: Region | None,
) -> xr.Dataset:
    if spatial_stride < 1:
        raise ValueError("spatial_stride must be >= 1")

    if region is not None:
        ds = select_region(ds, region)

    ds = ds.isel(
        latitude=slice(None, None, spatial_stride),
        longitude=slice(None, None, spatial_stride),
    )

    if crop_size is not None:
        if crop_size < 1:
            raise ValueError("crop_size must be >= 1")
        ds = ds.isel(latitude=slice(0, crop_size), longitude=slice(0, crop_size))

    return ds


def _select_wind_array(
    wind_ds: xr.Dataset,
    times: list[pd.Timestamp],
    spatial_stride: int,
    crop_size: int | None,
    region: Region | None,
) -> np.ndarray:
    names = [find_data_var(wind_ds, WIND_CANDIDATES[key]) for key in ("u10", "v10")]
    sliced = _spatial_indexer(wind_ds[names].sel(time=times), spatial_stride, crop_size, region)
    arrays = [
        sliced[name].transpose("time", "latitude", "longitude").values.astype(np.float32)
        for name in names
    ]
    return np.stack(arrays, axis=1)


def _select_wave_array(
    wave_ds: xr.Dataset,
    times: list[pd.Timestamp],
    spatial_stride: int,
    crop_size: int | None,
    region: Region | None,
) -> np.ndarray:
    names = {
        key: find_data_var(wave_ds, candidates)
        for key, candidates in WAVE_CANDIDATES.items()
    }
    sliced = _spatial_indexer(wave_ds[list(names.values())].sel(time=times), spatial_stride, crop_size, region)

    swh = sliced[names["swh"]].transpose("time", "latitude", "longitude").values.astype(np.float32)
    mwp = sliced[names["mwp"]].transpose("time", "latitude", "longitude").values.astype(np.float32)
    mwd = sliced[names["mwd"]].transpose("time", "latitude", "longitude").values.astype(np.float32)
    sin_mwd, cos_mwd = direction_degrees_to_unit(mwd)
    return np.stack([swh, mwp, cos_mwd, sin_mwd], axis=1)


def _history_times(t0: pd.Timestamp, history_hours: int) -> list[pd.Timestamp]:
    start = history_hours - 1
    return [t0 - pd.Timedelta(hours=offset) for offset in range(start, -1, -1)]


def _lead_times(t0: pd.Timestamp, lead_hours: tuple[int, ...]) -> list[pd.Timestamp]:
    return [t0 + pd.Timedelta(hours=lead) for lead in lead_hours]


def _accumulate(sum_v: np.ndarray, sumsq_v: np.ndarray, count_v: np.ndarray, arr: np.ndarray) -> None:
    channels = arr.shape[1]
    flat = np.moveaxis(arr, 1, 0).reshape(channels, -1)
    finite = np.isfinite(flat)
    values = np.where(finite, flat, 0.0)
    sum_v += values.sum(axis=1)
    sumsq_v += (values * values).sum(axis=1)
    count_v += finite.sum(axis=1)


def _accumulate_weighted(
    sum_v: np.ndarray,
    sumsq_v: np.ndarray,
    count_v: np.ndarray,
    arr: np.ndarray,
    time_weights: np.ndarray,
) -> None:
    weights = np.asarray(time_weights, dtype=np.int64)[:, None, None, None]
    finite = np.isfinite(arr)
    values = np.where(finite, arr, 0.0)
    sum_v += (values * weights).sum(axis=(0, 2, 3))
    sumsq_v += (values * values * weights).sum(axis=(0, 2, 3))
    count_v += (finite * weights).sum(axis=(0, 2, 3))


def _unique_times_and_counts(times: list[pd.Timestamp]) -> tuple[list[pd.Timestamp], np.ndarray]:
    counts = Counter(pd.Timestamp(value) for value in times)
    unique_times = sorted(counts)
    weights = np.asarray([counts[value] for value in unique_times], dtype=np.int64)
    return unique_times, weights


def _finalize_stats(sum_v: np.ndarray, sumsq_v: np.ndarray, count_v: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    if np.any(count_v == 0):
        raise ValueError(f"Cannot compute {label} normalization without finite values")
    mean = sum_v / count_v
    variance = np.maximum(sumsq_v / count_v - mean * mean, 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_normalization_stats(
    wind_ds: xr.Dataset,
    wave_ds: xr.Dataset,
    initialization_times: list[pd.Timestamp],
    spatial_stride: int = 1,
    crop_size: int | None = None,
    input_region: Region | None = None,
    output_region: Region | None = None,
    history_hours: int = DEFAULT_HISTORY_HOURS,
    lead_hours: tuple[int, ...] = DEFAULT_LEAD_HOURS,
) -> NormalizationStats:
    input_sum = np.zeros(2, dtype=np.float64)
    input_sumsq = np.zeros(2, dtype=np.float64)
    input_count = np.zeros(2, dtype=np.int64)
    target_sum = np.zeros(4, dtype=np.float64)
    target_sumsq = np.zeros(4, dtype=np.float64)
    target_count = np.zeros(4, dtype=np.int64)

    input_times = []
    target_times = []
    for raw_t0 in initialization_times:
        t0 = pd.Timestamp(raw_t0)
        input_times.extend(_history_times(t0, history_hours))
        target_times.extend(_lead_times(t0, lead_hours))

    unique_input_times, input_weights = _unique_times_and_counts(input_times)
    unique_target_times, target_weights = _unique_times_and_counts(target_times)
    time_chunk_size = 256
    for start in range(0, len(unique_input_times), time_chunk_size):
        end = start + time_chunk_size
        inputs = _select_wind_array(
            wind_ds,
            unique_input_times[start:end],
            spatial_stride,
            crop_size,
            input_region,
        )
        _accumulate_weighted(
            input_sum,
            input_sumsq,
            input_count,
            inputs,
            input_weights[start:end],
        )

    for start in range(0, len(unique_target_times), time_chunk_size):
        end = start + time_chunk_size
        targets = _select_wave_array(
            wave_ds,
            unique_target_times[start:end],
            spatial_stride,
            crop_size,
            output_region,
        )
        _accumulate_weighted(
            target_sum,
            target_sumsq,
            target_count,
            targets,
            target_weights[start:end],
        )

    input_mean, input_std = _finalize_stats(input_sum, input_sumsq, input_count, "input")
    target_mean, target_std = _finalize_stats(target_sum, target_sumsq, target_count, "target")

    return NormalizationStats(
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "cos_mwd", "sin_mwd"),
    )


class WindWaveSeq2SeqDataset(Dataset):
    def __init__(
        self,
        wind_ds: xr.Dataset,
        wave_ds: xr.Dataset,
        initialization_times: list[pd.Timestamp],
        stats: NormalizationStats,
        history_hours: int = DEFAULT_HISTORY_HOURS,
        lead_hours: tuple[int, ...] = DEFAULT_LEAD_HOURS,
        spatial_stride: int = 1,
        crop_size: int | None = None,
        input_region: Region | None = None,
        output_region: Region | None = None,
    ) -> None:
        self.wind_ds = wind_ds
        self.wave_ds = wave_ds
        self.initialization_times = [pd.Timestamp(value) for value in initialization_times]
        self.stats = stats
        self.history_hours = history_hours
        self.lead_hours = tuple(lead_hours)
        self.spatial_stride = spatial_stride
        self.crop_size = crop_size
        self.input_region = input_region
        self.output_region = output_region

    def __len__(self) -> int:
        return len(self.initialization_times)

    def __getitem__(self, index: int) -> dict[str, Any]:
        t0 = self.initialization_times[index]
        input_times = _history_times(t0, self.history_hours)
        target_times = _lead_times(t0, self.lead_hours)

        inputs = _select_wind_array(
            self.wind_ds,
            input_times,
            self.spatial_stride,
            self.crop_size,
            self.input_region,
        )
        targets = _select_wave_array(
            self.wave_ds,
            target_times,
            self.spatial_stride,
            self.crop_size,
            self.output_region,
        )
        persistence = _select_wave_array(
            self.wave_ds,
            [t0],
            self.spatial_stride,
            self.crop_size,
            self.output_region,
        )
        persistence = np.repeat(persistence, len(self.lead_hours), axis=0)

        inputs = (inputs - self.stats.input_mean[None, :, None, None]) / self.stats.input_std[
            None, :, None, None
        ]
        targets = (targets - self.stats.target_mean[None, :, None, None]) / self.stats.target_std[
            None, :, None, None
        ]
        persistence = (
            persistence - self.stats.target_mean[None, :, None, None]
        ) / self.stats.target_std[None, :, None, None]

        return {
            "inputs": torch.from_numpy(inputs.astype(np.float32)),
            "targets": torch.from_numpy(targets.astype(np.float32)),
            "persistence": torch.from_numpy(persistence.astype(np.float32)),
            "t0": t0.isoformat(),
            "input_times": [time.isoformat() for time in input_times],
            "target_times": [time.isoformat() for time in target_times],
        }
