from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import DEFAULT_HISTORY_HOURS, DEFAULT_INPUT_REGION, DEFAULT_LEAD_HOURS, data_dir, grib_raw_data_dir
from .convert_zarr import (
    OUTPUT_VAR_NAMES,
    WAVE_SHORT_NAMES,
    WIND_SHORT_NAMES,
    _open_grib_variable,
    _preprocess_grib_dataset,
)


WIND_STORE_NAME = "era5_wind_025_5N45N_95E150E.zarr"
WAVE_STORE_NAME = "era5_wave_050_5N45N_95E150E.zarr"
WAVE_MODEL_VARIABLES = ("swh", "mwp", "mwd_cos", "mwd_sin")


@dataclass(frozen=True)
class ZarrCachePaths:
    wind_store: Path
    wave_store: Path
    metadata_dir: Path


SingleFileWorker = Callable[[Path, Path, str, int, bool, bool], None]


def discover_grib_files(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"GRIB raw directory does not exist: {raw_dir}")
    files = sorted(
        path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in {".grib", ".grib2"}
    )
    if not files:
        raise FileNotFoundError(f"No GRIB files found in: {raw_dir}")
    return files


def _first_grib_time(path: Path, short_name: str = "10u") -> pd.Timestamp | None:
    try:
        import pygrib
    except Exception:
        return None
    try:
        with pygrib.open(str(path)) as messages:
            for message in messages:
                if getattr(message, "shortName", None) == short_name:
                    return pd.Timestamp(message.validDate)
    except Exception:
        return None
    return None


def _sort_grib_files_by_time(grib_files: Sequence[Path]) -> list[Path]:
    return sorted(grib_files, key=lambda path: (_first_grib_time(path) or pd.Timestamp.max, path.name))


def _normalize_output_variable_name(ds: xr.Dataset, short_name: str) -> xr.Dataset:
    output_name = OUTPUT_VAR_NAMES.get(short_name)
    if output_name is None or output_name in ds.data_vars:
        return ds
    if short_name in ds.data_vars:
        return ds.rename({short_name: output_name})
    if len(ds.data_vars) == 1:
        return ds.rename({next(iter(ds.data_vars)): output_name})
    return ds


def _ensure_latitude_ascending(ds: xr.Dataset) -> xr.Dataset:
    if ds.sizes.get("latitude", 0) > 1 and ds["latitude"].values[0] > ds["latitude"].values[-1]:
        return ds.sortby("latitude")
    return ds


def _drop_non_core_coords(ds: xr.Dataset) -> xr.Dataset:
    return ds.reset_coords(drop=True)


def _deduplicate_and_sort_time(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.sortby("time")
    times = pd.Index(pd.to_datetime(ds["time"].values))
    if times.has_duplicates:
        ds = ds.isel(time=~times.duplicated(keep="first"))
    return ds


def _prepare_variable_dataset(ds: xr.Dataset, short_name: str, region_text: str | None) -> xr.Dataset:
    ds = _normalize_output_variable_name(ds, short_name)
    ds = _preprocess_grib_dataset(ds, region_text)
    ds = _ensure_latitude_ascending(ds)
    ds = _drop_non_core_coords(ds)
    ds = _deduplicate_and_sort_time(ds)
    return ds.astype({name: "float32" for name in ds.data_vars})


def _merge_variables_for_file(
    grib_file: Path,
    short_names: Sequence[str],
    region_text: str | None,
    opener: Callable[[Path, str], xr.Dataset],
) -> xr.Dataset:
    datasets = []
    for short_name in short_names:
        print(f"Reading {short_name} from {grib_file.name}", flush=True)
        ds = opener(grib_file, short_name)
        if not ds.data_vars:
            raise ValueError(f"No data variables found for shortName={short_name}")
        datasets.append(_prepare_variable_dataset(ds, short_name, region_text))
    merged = xr.merge(datasets, compat="override", join="exact")
    return _deduplicate_and_sort_time(merged)


def _add_wave_direction_components(ds: xr.Dataset) -> xr.Dataset:
    if "mwd" not in ds:
        raise KeyError("wave dataset must contain mwd before deriving direction components")
    radians = np.deg2rad(ds["mwd"].astype("float32"))
    ds["mwd_cos"] = np.cos(radians).astype("float32")
    ds["mwd_sin"] = np.sin(radians).astype("float32")
    return ds


def _chunk_map(ds: xr.Dataset, time_chunk: int) -> dict[str, int]:
    chunks = {}
    for dim, size in ds.sizes.items():
        chunks[dim] = min(int(time_chunk), int(size)) if dim == "time" else int(size)
    return chunks


def _encoding_for_dataset(ds: xr.Dataset, time_chunk: int) -> dict[str, dict[str, tuple[int, ...]]]:
    chunks = _chunk_map(ds, time_chunk)
    return {
        name: {"chunks": tuple(chunks[dim] for dim in data_array.dims)}
        for name, data_array in ds.data_vars.items()
    }


def _to_zarr_with_local_scheduler(
    ds: xr.Dataset,
    store: Path,
    mode: str,
    time_chunk: int,
    append_dim: str | None = None,
) -> None:
    ds = ds.chunk(_chunk_map(ds, time_chunk))
    kwargs = {
        "store": store,
        "mode": mode,
        "consolidated": True,
        "zarr_format": 2,
        "safe_chunks": False,
    }
    if append_dim is None:
        kwargs["encoding"] = _encoding_for_dataset(ds, time_chunk)
    else:
        kwargs["append_dim"] = append_dim

    try:
        import dask
        import dask.base
    except Exception:
        ds.to_zarr(**kwargs)
        return
    dask.base._DISTRIBUTED_AVAILABLE = False
    with dask.config.set(scheduler="synchronous"):
        ds.to_zarr(**kwargs)


def _prepare_output_paths(zarr_dir: Path, metadata_dir: Path, overwrite: bool) -> ZarrCachePaths:
    zarr_dir = Path(zarr_dir)
    metadata_dir = Path(metadata_dir)
    paths = ZarrCachePaths(
        wind_store=zarr_dir / WIND_STORE_NAME,
        wave_store=zarr_dir / WAVE_STORE_NAME,
        metadata_dir=metadata_dir,
    )
    for store in (paths.wind_store, paths.wave_store):
        if not store.exists():
            continue
        if not overwrite:
            raise FileExistsError(f"Zarr store already exists: {store}")
        shutil.rmtree(store)
    if metadata_dir.exists() and overwrite:
        for path in metadata_dir.iterdir():
            if path.is_file():
                path.unlink()
    zarr_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _paths_for_zarr_dir(zarr_dir: Path, metadata_dir: Path | None = None) -> ZarrCachePaths:
    zarr_dir = Path(zarr_dir)
    return ZarrCachePaths(
        wind_store=zarr_dir / WIND_STORE_NAME,
        wave_store=zarr_dir / WAVE_STORE_NAME,
        metadata_dir=Path(metadata_dir) if metadata_dir is not None else zarr_dir.parent / "metadata",
    )


def _write_single_file_dataset_to_store(
    ds: xr.Dataset,
    store: Path,
    time_chunk: int,
    overwrite: bool,
    append: bool,
) -> None:
    if overwrite and append:
        raise ValueError("overwrite and append cannot both be true")
    if store.exists():
        if overwrite:
            shutil.rmtree(store)
            mode = "w"
            append_dim = None
        elif append:
            mode = "a"
            append_dim = "time"
        else:
            raise FileExistsError(f"Zarr store already exists: {store}")
    else:
        mode = "w"
        append_dim = None
    store.parent.mkdir(parents=True, exist_ok=True)
    _to_zarr_with_local_scheduler(ds, store=store, mode=mode, time_chunk=time_chunk, append_dim=append_dim)


def process_single_grib_to_zarr(
    grib_path: Path,
    zarr_dir: Path,
    region_text: str = DEFAULT_INPUT_REGION,
    time_chunk: int = 168,
    overwrite: bool = False,
    append: bool = False,
    opener: Callable[[Path, str], xr.Dataset] = _open_grib_variable,
) -> ZarrCachePaths:
    grib_path = Path(grib_path)
    if not grib_path.exists():
        raise FileNotFoundError(f"GRIB file does not exist: {grib_path}")
    if time_chunk <= 0:
        raise ValueError("time_chunk must be positive")

    paths = _paths_for_zarr_dir(zarr_dir)
    print(f"Processing one GRIB file: {grib_path}", flush=True)

    print(f"Writing wind from {grib_path.name}: {paths.wind_store}", flush=True)
    wind = _merge_variables_for_file(grib_path, WIND_SHORT_NAMES, region_text, opener)
    try:
        _write_single_file_dataset_to_store(
            wind,
            store=paths.wind_store,
            time_chunk=time_chunk,
            overwrite=overwrite,
            append=append,
        )
    finally:
        wind.close()

    print(f"Writing wave from {grib_path.name}: {paths.wave_store}", flush=True)
    wave = _merge_variables_for_file(grib_path, WAVE_SHORT_NAMES, region_text, opener)
    wave = _add_wave_direction_components(wave)
    try:
        _write_single_file_dataset_to_store(
            wave,
            store=paths.wave_store,
            time_chunk=time_chunk,
            overwrite=overwrite,
            append=append,
        )
    finally:
        wave.close()

    print(f"Finished one GRIB file: {grib_path}", flush=True)
    return paths


def _open_zarr_store(path: Path) -> xr.Dataset:
    return xr.open_zarr(path, chunks=None, consolidated=True)


def _split_sample_indices(sample_t0_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(sample_t0_indices)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)
    return sample_t0_indices[:train_end], sample_t0_indices[train_end:val_end], sample_t0_indices[val_end:]


def _safe_float(value: object) -> float:
    numeric = float(value)
    if np.isfinite(numeric):
        return numeric
    return 0.0


def _compute_stats(ds: xr.Dataset, variables: Sequence[str], time_indices: np.ndarray) -> dict[str, dict[str, float]]:
    if len(time_indices) == 0:
        return {name: {"mean": 0.0, "std": 1.0} for name in variables}
    subset = ds[list(variables)].isel(time=time_indices)
    stats: dict[str, dict[str, float]] = {}
    try:
        import dask
        import dask.base
    except Exception:
        context = None
    else:
        dask.base._DISTRIBUTED_AVAILABLE = False
        context = dask.config.set(scheduler="synchronous")
    with context or nullcontext():
        for name in variables:
            mean = _safe_float(subset[name].mean(skipna=True).values)
            std = _safe_float(subset[name].std(skipna=True).values)
            stats[name] = {"mean": mean, "std": std if std > 0 else 1.0}
    return stats


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_grid_metadata(path: Path, ds: xr.Dataset) -> None:
    payload = {
        "time_size": int(ds.sizes["time"]),
        "latitude_size": int(ds.sizes["latitude"]),
        "longitude_size": int(ds.sizes["longitude"]),
        "latitude_min": float(ds["latitude"].min().values),
        "latitude_max": float(ds["latitude"].max().values),
        "longitude_min": float(ds["longitude"].min().values),
        "longitude_max": float(ds["longitude"].max().values),
    }
    _write_json(path, payload)


def _write_metadata(
    paths: ZarrCachePaths,
    history_hours: int,
    lead_hours: Sequence[int],
) -> None:
    wind = _open_zarr_store(paths.wind_store)
    wave = _open_zarr_store(paths.wave_store)
    try:
        wind_times = pd.DatetimeIndex(pd.to_datetime(wind["time"].values))
        wave_times = pd.DatetimeIndex(pd.to_datetime(wave["time"].values))
        common_times = np.intersect1d(wind_times.values.astype("datetime64[ns]"), wave_times.values.astype("datetime64[ns]"))
        np.save(paths.metadata_dir / "common_times.npy", common_times)

        max_lead = max(int(lead) for lead in lead_hours)
        sample_t0_indices = np.arange(int(history_hours) - 1, len(common_times) - max_lead, dtype=np.int64)
        np.save(paths.metadata_dir / "sample_t0_indices.npy", sample_t0_indices)

        train, val, test = _split_sample_indices(sample_t0_indices)
        np.save(paths.metadata_dir / "train_indices.npy", train)
        np.save(paths.metadata_dir / "val_indices.npy", val)
        np.save(paths.metadata_dir / "test_indices.npy", test)

        normalization = {
            "wind": _compute_stats(wind, ("u10", "v10"), train),
            "wave": _compute_stats(wave, WAVE_MODEL_VARIABLES, train),
        }
        _write_json(paths.metadata_dir / "normalization.json", normalization)
        _write_grid_metadata(paths.metadata_dir / "grid_wind_025.json", wind)
        _write_grid_metadata(paths.metadata_dir / "grid_wave_050.json", wave)
    finally:
        wind.close()
        wave.close()


def build_zarr_cache(
    raw_dir: Path,
    zarr_dir: Path,
    metadata_dir: Path,
    region_text: str = DEFAULT_INPUT_REGION,
    history_hours: int = DEFAULT_HISTORY_HOURS,
    lead_hours: Sequence[int] = DEFAULT_LEAD_HOURS,
    time_chunk: int = 168,
    overwrite: bool = False,
    worker_runner: SingleFileWorker | None = None,
) -> ZarrCachePaths:
    if history_hours <= 0:
        raise ValueError("history_hours must be positive")
    if not lead_hours or any(int(lead) <= 0 for lead in lead_hours):
        raise ValueError("lead_hours must contain positive lead times")
    if time_chunk <= 0:
        raise ValueError("time_chunk must be positive")

    grib_files = _sort_grib_files_by_time(discover_grib_files(raw_dir))
    paths = _prepare_output_paths(zarr_dir, metadata_dir, overwrite=overwrite)
    runner = worker_runner or run_single_grib_worker

    for index, grib_file in enumerate(grib_files):
        runner(
            grib_file,
            Path(zarr_dir),
            region_text,
            time_chunk,
            overwrite and index == 0,
            index > 0,
        )

    print(f"Writing metadata: {paths.metadata_dir}", flush=True)
    _write_metadata(paths, history_hours=history_hours, lead_hours=lead_hours)
    return paths


def run_single_grib_worker(
    grib_path: Path,
    zarr_dir: Path,
    region_text: str,
    time_chunk: int,
    overwrite: bool,
    append: bool,
) -> None:
    worker_script = Path(__file__).resolve().parents[1] / "process_one_grib_to_zarr.py"
    command = [
        sys.executable,
        str(worker_script),
        "--grib-path",
        str(grib_path),
        "--zarr-dir",
        str(zarr_dir),
        "--region",
        region_text,
        "--time-chunk",
        str(time_chunk),
    ]
    if overwrite:
        command.append("--overwrite")
    if append:
        command.append("--append")
    print(f"Starting worker: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build formal ERA5 wind/wave Zarr caches from GRIB files.")
    parser.add_argument("--raw-dir", type=Path, default=grib_raw_data_dir())
    parser.add_argument("--zarr-dir", type=Path, default=data_dir() / "zarr")
    parser.add_argument("--metadata-dir", type=Path, default=data_dir() / "metadata")
    parser.add_argument("--region", default=DEFAULT_INPUT_REGION)
    parser.add_argument("--history-hours", type=int, default=DEFAULT_HISTORY_HOURS)
    parser.add_argument("--lead-hours", default=",".join(str(lead) for lead in DEFAULT_LEAD_HOURS))
    parser.add_argument("--time-chunk", type=int, default=168)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_single_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process exactly one ERA5 GRIB file into wind/wave Zarr caches.")
    parser.add_argument("--grib-path", type=Path, required=True)
    parser.add_argument("--zarr-dir", type=Path, default=data_dir() / "zarr")
    parser.add_argument("--region", default=DEFAULT_INPUT_REGION)
    parser.add_argument("--time-chunk", type=int, default=168)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    return parser


def _parse_lead_hours(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = build_zarr_cache(
        raw_dir=args.raw_dir,
        zarr_dir=args.zarr_dir,
        metadata_dir=args.metadata_dir,
        region_text=args.region,
        history_hours=args.history_hours,
        lead_hours=_parse_lead_hours(args.lead_hours),
        time_chunk=args.time_chunk,
        overwrite=args.overwrite,
    )
    print(f"wind: {paths.wind_store}")
    print(f"wave: {paths.wave_store}")
    print(f"metadata: {paths.metadata_dir}")
    return 0


def single_main(argv: Sequence[str] | None = None) -> int:
    args = build_single_arg_parser().parse_args(argv)
    paths = process_single_grib_to_zarr(
        grib_path=args.grib_path,
        zarr_dir=args.zarr_dir,
        region_text=args.region,
        time_chunk=args.time_chunk,
        overwrite=args.overwrite,
        append=args.append,
    )
    print(f"wind: {paths.wind_store}")
    print(f"wave: {paths.wave_store}")
    return 0
