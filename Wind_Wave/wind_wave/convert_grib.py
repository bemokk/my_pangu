from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

from .config import converted_data_dir, grib_raw_data_dir
from .era5 import drop_extra_dims, normalize_spatial_coords, normalize_time_coord, parse_region, select_region


WIND_SHORT_NAMES = ("u10", "v10")
WAVE_SHORT_NAMES = ("swh", "mwp", "mwd")
WIND_OUTPUT_NAME = "wind.nc"
WAVE_OUTPUT_NAME = "wave.nc"


def _add_conda_dll_directories() -> None:
    if os.name != "nt":
        return
    env_root = Path(sys.executable).resolve().parent
    candidates = (env_root / "Library" / "bin", env_root)
    for directory in candidates:
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


@dataclass(frozen=True)
class ConvertedYearPair:
    year: int
    wind_nc: Path
    wave_nc: Path


def parse_years(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        raise ValueError("years cannot be empty")
    if ":" in value:
        start_text, end_text = [part.strip() for part in value.split(":", maxsplit=1)]
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError("year range must be ascending")
        return tuple(range(start, end + 1))
    years = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not years:
        raise ValueError("years cannot be empty")
    return years


def discover_grib_files(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"GRIB raw data directory does not exist: {raw_dir}")
    files = sorted(
        path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in {".grib", ".grib2"}
    )
    if not files:
        raise FileNotFoundError(f"No .grib or .grib2 files found in: {raw_dir}")
    return files


def build_year_output_pairs(output_root: Path, years: Sequence[int]) -> list[ConvertedYearPair]:
    output_root = Path(output_root)
    return [
        ConvertedYearPair(
            year=int(year),
            wind_nc=output_root / str(year) / WIND_OUTPUT_NAME,
            wave_nc=output_root / str(year) / WAVE_OUTPUT_NAME,
        )
        for year in years
    ]


def _open_grib_variable(path: Path, short_name: str) -> xr.Dataset:
    try:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"shortName": short_name},
                "indexpath": "",
            },
        )
    except (KeyError, EOFError):
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "no messages" in message or "no valid message" in message or "no match" in message:
            raise KeyError(short_name) from exc
        raise RuntimeError(
            "Failed to open GRIB with cfgrib. Install ecCodes/cfgrib in the pangu "
            "environment before converting GRIB2 data."
        ) from exc


def _preprocess_grib_dataset(ds: xr.Dataset, region_text: str | None) -> xr.Dataset:
    ds = normalize_time_coord(ds)
    ds = normalize_spatial_coords(ds)
    ds = drop_extra_dims(ds)
    if region_text:
        ds = select_region(ds, parse_region(region_text))
    return ds


def _combine_variable(
    grib_files: Sequence[Path],
    short_name: str,
    region_text: str | None,
    opener: Callable[[Path, str], xr.Dataset] = _open_grib_variable,
) -> xr.Dataset:
    datasets = []
    for path in grib_files:
        try:
            ds = opener(path, short_name)
        except (KeyError, EOFError):
            continue
        if not ds.data_vars:
            continue
        datasets.append(_preprocess_grib_dataset(ds, region_text))
    if not datasets:
        raise ValueError(f"No GRIB messages found for shortName={short_name}")
    combined = xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal", compat="override")
    times = pd.Index(pd.to_datetime(combined["time"].values))
    if times.has_duplicates:
        combined = combined.isel(time=~times.duplicated(keep="first"))
    return combined.sortby("time")


def _merge_variables(
    grib_files: Sequence[Path],
    short_names: Sequence[str],
    region_text: str | None,
    opener: Callable[[Path, str], xr.Dataset] = _open_grib_variable,
) -> xr.Dataset:
    datasets = [_combine_variable(grib_files, short_name, region_text, opener) for short_name in short_names]
    return xr.merge(datasets, compat="override", join="outer").sortby("time")


def _dataset_for_year(ds: xr.Dataset, year: int) -> xr.Dataset:
    time_values = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    mask = time_values.year == year
    if not mask.any():
        raise ValueError(f"No data found for year {year}")
    return ds.isel(time=mask)


def _write_netcdf(ds: xr.Dataset, path: Path) -> None:
    try:
        import dask
        import dask.base
        import xarray.backends.api as xarray_api
        import xarray.backends.locks as xarray_locks
    except Exception:
        ds.to_netcdf(path, engine="netcdf4")
        return
    dask.base._DISTRIBUTED_AVAILABLE = False
    xarray_api._get_scheduler = lambda *args, **kwargs: "threaded"
    xarray_locks._get_scheduler = lambda *args, **kwargs: "threaded"
    with dask.config.set(scheduler="synchronous"):
        ds.to_netcdf(path, engine="netcdf4")


def convert_grib_to_yearly_netcdf(
    raw_dir: Path,
    output_root: Path,
    years: Sequence[int],
    region_text: str | None = None,
    overwrite: bool = False,
    opener: Callable[[Path, str], xr.Dataset] = _open_grib_variable,
) -> list[ConvertedYearPair]:
    grib_files = discover_grib_files(raw_dir)
    output_pairs = build_year_output_pairs(output_root, years)

    missing_pairs = [pair for pair in output_pairs if overwrite or not (pair.wind_nc.exists() and pair.wave_nc.exists())]
    if not missing_pairs:
        return output_pairs

    wind = _merge_variables(grib_files, WIND_SHORT_NAMES, region_text, opener)
    wave = _merge_variables(grib_files, WAVE_SHORT_NAMES, region_text, opener)

    for pair in missing_pairs:
        pair.wind_nc.parent.mkdir(parents=True, exist_ok=True)
        _write_netcdf(_dataset_for_year(wind, pair.year), pair.wind_nc)
        _write_netcdf(_dataset_for_year(wave, pair.year), pair.wave_nc)
    return output_pairs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ERA5 GRIB2 wind/wave data to yearly NetCDF files.")
    parser.add_argument("--raw-dir", type=Path, default=grib_raw_data_dir())
    parser.add_argument("--output-dir", type=Path, default=converted_data_dir())
    parser.add_argument("--years", default="2016:2024")
    parser.add_argument("--region", default="5,50,95,150")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    years = parse_years(args.years)
    try:
        pairs = convert_grib_to_yearly_netcdf(
            raw_dir=args.raw_dir,
            output_root=args.output_dir,
            years=years,
            region_text=args.region,
            overwrite=args.overwrite,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for pair in pairs:
        print(f"{pair.year}: {pair.wind_nc} | {pair.wave_nc}")
    return 0
