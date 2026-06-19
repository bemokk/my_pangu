from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import xarray as xr

from .era5 import drop_extra_dims, normalize_spatial_coords, normalize_time_coord, parse_region, select_region


WIND_SHORT_NAMES = ("10u", "10v")
WAVE_SHORT_NAMES = ("swh", "mwp", "mwd")
OUTPUT_VAR_NAMES = {"10u": "u10", "10v": "v10"}


def _add_conda_dll_directories() -> None:
    if os.name != "nt":
        return
    env_root = Path(sys.executable).resolve().parent
    for directory in (env_root / "Library" / "bin", env_root):
        if not directory.exists():
            continue
        os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(directory))


_add_conda_dll_directories()


def _configure_ssl_certs() -> None:
    try:
        import certifi
    except Exception:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


_configure_ssl_certs()


def _open_grib_variable(path: Path, short_name: str) -> xr.Dataset:
    index_path = path.with_name(f"{path.name}.{short_name}.idx")
    return xr.open_dataset(
        path,
        engine="cfgrib",
        chunks={"time": 168},
        backend_kwargs={
            "filter_by_keys": {"shortName": short_name},
            "indexpath": str(index_path),
        },
    )


def _preprocess_grib_dataset(ds: xr.Dataset, region_text: str | None) -> xr.Dataset:
    ds = normalize_time_coord(ds)
    ds = normalize_spatial_coords(ds)
    ds = drop_extra_dims(ds)
    if region_text:
        ds = select_region(ds, parse_region(region_text))
    return ds


def _normalize_output_variable_name(ds: xr.Dataset, short_name: str) -> xr.Dataset:
    output_name = OUTPUT_VAR_NAMES.get(short_name)
    if output_name is None or output_name in ds.data_vars:
        return ds
    if short_name in ds.data_vars:
        return ds.rename({short_name: output_name})
    if len(ds.data_vars) == 1:
        return ds.rename({next(iter(ds.data_vars)): output_name})
    return ds


def _merge_group_variables(
    grib_path: Path,
    short_names: Sequence[str],
    region_text: str | None,
    opener: Callable[[Path, str], xr.Dataset],
) -> xr.Dataset:
    datasets = []
    for short_name in short_names:
        print(f"Reading {short_name} from {grib_path.name}", flush=True)
        ds = opener(grib_path, short_name)
        if not ds.data_vars:
            raise ValueError(f"No data variables found for shortName={short_name}")
        ds = _normalize_output_variable_name(ds, short_name)
        datasets.append(_preprocess_grib_dataset(ds, region_text))
    return xr.merge(datasets, compat="override", join="exact").sortby("time")


def _chunk_dataset(ds: xr.Dataset, time_chunk: int) -> xr.Dataset:
    chunk_map = {}
    for dim, size in ds.sizes.items():
        if dim == "time":
            chunk_map[dim] = min(int(time_chunk), int(size))
        else:
            chunk_map[dim] = int(size)
    return ds.chunk(chunk_map)


def _zarr_encoding(ds: xr.Dataset, time_chunk: int) -> dict[str, dict[str, tuple[int, ...]]]:
    chunks_by_dim = {}
    for dim, size in ds.sizes.items():
        if dim == "time":
            chunks_by_dim[dim] = min(int(time_chunk), int(size))
        else:
            chunks_by_dim[dim] = int(size)
    return {
        var_name: {"chunks": tuple(chunks_by_dim[dim] for dim in data_array.dims)}
        for var_name, data_array in ds.data_vars.items()
    }


def _write_zarr_group(ds: xr.Dataset, output_store: Path, group: str, mode: str, time_chunk: int) -> None:
    ds = _chunk_dataset(ds, time_chunk)
    try:
        import dask
        import dask.base
    except Exception:
        ds.to_zarr(
            output_store,
            group=group,
            mode=mode,
            encoding=_zarr_encoding(ds, time_chunk),
            consolidated=True,
            zarr_format=2,
            safe_chunks=False,
        )
        return
    dask.base._DISTRIBUTED_AVAILABLE = False
    with dask.config.set(scheduler="synchronous"):
        ds.to_zarr(
            output_store,
            group=group,
            mode=mode,
            encoding=_zarr_encoding(ds, time_chunk),
            consolidated=True,
            zarr_format=2,
            safe_chunks=False,
        )


def _prepare_output_store(output_store: Path, overwrite: bool) -> None:
    if not output_store.exists():
        output_store.parent.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(f"Zarr store already exists: {output_store}")
    if output_store.is_dir():
        shutil.rmtree(output_store)
    else:
        output_store.unlink()
    output_store.parent.mkdir(parents=True, exist_ok=True)


def convert_grib_to_zarr(
    grib_path: Path,
    output_store: Path,
    region_text: str | None = None,
    time_chunk: int = 168,
    overwrite: bool = False,
    opener: Callable[[Path, str], xr.Dataset] = _open_grib_variable,
) -> Path:
    grib_path = Path(grib_path)
    output_store = Path(output_store)
    if not grib_path.exists():
        raise FileNotFoundError(f"GRIB file does not exist: {grib_path}")
    if time_chunk <= 0:
        raise ValueError("time_chunk must be positive")

    _prepare_output_store(output_store, overwrite)

    print(f"Writing wind group: {output_store}::wind", flush=True)
    wind = _merge_group_variables(grib_path, WIND_SHORT_NAMES, region_text, opener)
    try:
        _write_zarr_group(wind, output_store, "wind", "w", time_chunk)
    finally:
        wind.close()

    print(f"Writing wave group: {output_store}::wave", flush=True)
    wave = _merge_group_variables(grib_path, WAVE_SHORT_NAMES, region_text, opener)
    try:
        _write_zarr_group(wave, output_store, "wave", "a", time_chunk)
    finally:
        wave.close()

    return output_store


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one ERA5 GRIB file to a grouped Zarr store.")
    parser.add_argument("--grib-path", type=Path, required=True)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--region", default="5,50,95,150")
    parser.add_argument("--time-chunk", type=int, default=168)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_store = convert_grib_to_zarr(
        grib_path=args.grib_path,
        output_store=args.output_store,
        region_text=args.region,
        time_chunk=args.time_chunk,
        overwrite=args.overwrite,
    )
    print(output_store)
    return 0
