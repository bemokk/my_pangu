"""Read one Pangu initial state without materializing a full annual archive."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np

from .config import (
    ERA5_ROOT,
    GDAS_ROOT,
    GRID_SHAPE,
    PRESSURE_LEVELS,
    SURFACE_ORDER,
    UPPER_ORDER,
)
from .experiments import Experiment, time_label


def _as_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _time_labels(ds: nc.Dataset) -> list[datetime]:
    name = next((candidate for candidate in ("valid_time", "time") if candidate in ds.variables), None)
    if name is None:
        raise ValueError(f"No valid_time/time coordinate in {ds.filepath()}")
    variable = ds[name]
    values = nc.num2date(
        variable[:],
        variable.units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return [_as_datetime(value) for value in np.atleast_1d(values)]


def _find_variable(ds: nc.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {candidates} found in {ds.filepath()}; variables={list(ds.variables)}")


def _find_coordinate(ds: nc.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {candidates} found in {ds.filepath()}")


def _read_at_time(variable, time_index: int, expected_ndim: int) -> np.ndarray:
    dimensions = tuple(variable.dimensions)
    time_axis = next((index for index, name in enumerate(dimensions) if name in ("valid_time", "time")), None)
    if time_axis is None:
        raise ValueError(f"Variable {variable.name} has no time dimension: {dimensions}")
    key = [slice(None)] * variable.ndim
    key[time_axis] = time_index
    values = variable[tuple(key)]
    if np.ma.getmaskarray(values).any():
        raise ValueError(f"Missing values in variable {variable.name}")
    array = np.asarray(values, dtype=np.float32)
    remaining_dims = [name for index, name in enumerate(dimensions) if index != time_axis]
    aliases = {"lat": "latitude", "lon": "longitude", "level": "pressure_level", "isobaricInhPa": "pressure_level"}
    remaining_dims = [aliases.get(name, name) for name in remaining_dims]
    expected_dims = ["latitude", "longitude"] if expected_ndim == 2 else ["pressure_level", "latitude", "longitude"]
    for index in reversed(range(len(remaining_dims))):
        if remaining_dims[index] not in expected_dims and array.shape[index] == 1:
            array = np.take(array, 0, axis=index)
            remaining_dims.pop(index)
    if set(remaining_dims) != set(expected_dims) or len(remaining_dims) != expected_ndim:
        raise ValueError(f"Unexpected dimensions for {variable.name}: {remaining_dims}")
    array = np.transpose(array, [remaining_dims.index(name) for name in expected_dims])
    if array.ndim != expected_ndim:
        raise ValueError(f"Variable {variable.name} after time selection has shape {array.shape}, expected {expected_ndim}D")
    return array


def _validate_grid(ds: nc.Dataset, latitude: np.ndarray, longitude: np.ndarray, source: Path) -> None:
    lat_name = _find_coordinate(ds, ("latitude", "lat"))
    lon_name = _find_coordinate(ds, ("longitude", "lon"))
    expected_lat = np.linspace(90.0, -90.0, GRID_SHAPE[0], dtype=np.float64)
    expected_lon = np.arange(GRID_SHAPE[1], dtype=np.float64) * 0.25
    file_lat = np.asarray(ds[lat_name][:], dtype=np.float64)
    file_lon = np.asarray(ds[lon_name][:], dtype=np.float64)
    if file_lat.shape != expected_lat.shape or file_lon.shape != expected_lon.shape:
        raise ValueError(f"Grid coordinate dimensions do not match {GRID_SHAPE}: {source}")
    if not np.allclose(file_lat, expected_lat, rtol=0, atol=1e-6) or not np.allclose(file_lon, expected_lon, rtol=0, atol=1e-6):
        raise ValueError(f"Grid mismatch in {source}; expected 90..-90 and 0..359.75 at 0.25 degrees")
    latitude[:] = file_lat.astype(np.float32)
    longitude[:] = file_lon.astype(np.float32)


@dataclass
class InitialState:
    upper: np.ndarray
    surface: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    source_files: tuple[Path, ...]
    negative_specific_humidity_count: int = 0


class Era5Archive:
    """ERA5 annual surface + quarterly upper archive reader."""

    def __init__(self, root: Path = ERA5_ROOT):
        self.root = Path(root)
        self._surface_index: dict[datetime, int] | None = None
        self._surface_path: Path | None = None
        self._surface_files: dict[datetime, Path] = {}
        self._surface_year: int | None = None
        self._upper_index: dict[datetime, tuple[Path, int]] | None = None
        self._upper_year: int | None = None

    def _surface_file(self, year: int) -> Path:
        candidates = (self.root / str(year) / "surface" / "surface.nc", self.root / str(year) / "surface.nc")
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"ERA5 surface file not found under {self.root / str(year)}")

    def _build_surface_index(self, year: int) -> None:
        daily = sorted((self.root / str(year)).glob("*/surface.nc"))
        daily = [path for path in daily if path.parent.name not in ("surface", "upper")]
        files = daily or [self._surface_file(year)]
        index = {}
        paths = {}
        for path in files:
            with nc.Dataset(path) as ds:
                labels = _time_labels(ds)
            for offset, stamp in enumerate(labels):
                if stamp in index:
                    raise ValueError(f"Duplicate ERA5 surface times in {path}")
                index[stamp] = offset
                paths[stamp] = path
        self._surface_files = paths
        self._surface_index = index
        self._surface_year = year

    def _build_upper_index(self, year: int) -> None:
        upper_dir = self.root / str(year) / "upper"
        files = sorted(upper_dir.glob("*.nc"))
        daily = sorted((self.root / str(year)).glob("*/upper.nc"))
        daily = [path for path in daily if path.parent.name not in ("surface", "upper")]
        files = daily or files
        if not files:
            raise FileNotFoundError(f"ERA5 upper files not found under {upper_dir}")
        index: dict[datetime, tuple[Path, int]] = {}
        for path in files:
            with nc.Dataset(path) as ds:
                for time_index, value in enumerate(_time_labels(ds)):
                    if value in index:
                        raise ValueError(f"Duplicate ERA5 upper time {value} in {path} and {index[value][0]}")
                    index[value] = (path, time_index)
        self._upper_index = index
        self._upper_year = year

    def available_times(self, year: int) -> set[datetime]:
        self._build_surface_index(year)
        self._build_upper_index(year)
        return set(self._surface_index or {}) & set(self._upper_index or {})

    def load(self, valid_time: datetime) -> InitialState:
        valid_time = _as_datetime(valid_time)
        year = valid_time.year
        if self._surface_index is None or self._surface_year != year:
            self._build_surface_index(year)
        if self._upper_index is None or self._upper_year != year:
            self._build_upper_index(year)
        if valid_time not in self._surface_index:
            raise FileNotFoundError(f"ERA5 surface has no exact time {valid_time}")
        if valid_time not in self._upper_index:
            raise FileNotFoundError(f"ERA5 upper archive has no exact time {valid_time}")

        surface_path = self._surface_files[valid_time]
        upper_path, upper_time_index = self._upper_index[valid_time]
        surface = np.empty((4, *GRID_SHAPE), dtype=np.float32)
        upper = np.empty((5, len(PRESSURE_LEVELS), *GRID_SHAPE), dtype=np.float32)
        latitude = np.empty(GRID_SHAPE[0], dtype=np.float32)
        longitude = np.empty(GRID_SHAPE[1], dtype=np.float32)
        with nc.Dataset(surface_path) as ds:
            _validate_grid(ds, latitude, longitude, surface_path)
            candidates = {
                "msl": ("msl", "mean_sea_level_pressure"),
                "u10": ("u10", "10m_u_component_of_wind"),
                "v10": ("v10", "10m_v_component_of_wind"),
                "t2m": ("t2m", "2m_temperature"),
            }
            for channel, name in enumerate(SURFACE_ORDER):
                surface[channel] = _read_at_time(ds[_find_variable(ds, candidates[name])], self._surface_index[valid_time], 2)
        with nc.Dataset(upper_path) as ds:
            _validate_grid(ds, latitude, longitude, upper_path)
            levels_name = _find_coordinate(ds, ("pressure_level", "level", "isobaricInhPa"))
            levels = np.asarray(ds[levels_name][:], dtype=np.float64)
            if not np.array_equal(levels, np.asarray(PRESSURE_LEVELS, dtype=np.float64)):
                raise ValueError(f"Unexpected pressure levels in {upper_path}: {levels.tolist()}")
            candidates = {
                "z": ("z", "geopotential"),
                "q": ("q", "specific_humidity"),
                "t": ("t", "temperature"),
                "u": ("u", "u_component_of_wind"),
                "v": ("v", "v_component_of_wind"),
            }
            for channel, name in enumerate(UPPER_ORDER):
                upper[channel] = _read_at_time(ds[_find_variable(ds, candidates[name])], upper_time_index, 3)
        if not np.isfinite(surface).all() or not np.isfinite(upper).all():
            raise ValueError(f"Non-finite values in ERA5 input at {valid_time}")
        negative_q = int(np.count_nonzero(upper[1] < 0))
        if negative_q:
            warnings.warn(f"ERA5 q has {negative_q} negative grid values at {valid_time}; values are preserved", RuntimeWarning)
        return InitialState(upper, surface, latitude, longitude, (surface_path, upper_path), negative_q)


class NpyDirectorySource:
    """Read legacy GDAS per-time NPY directories."""

    def __init__(self, root: Path = GDAS_ROOT):
        self.root = Path(root)

    def load(self, valid_time: datetime) -> InitialState:
        from ..gdas.paths import input_directory

        directory = input_directory(self.root, valid_time)
        upper_path = directory / "input_upper.npy"
        surface_path = directory / "input_surface.npy"
        if not upper_path.exists() or not surface_path.exists():
            raise FileNotFoundError(f"GDAS input requires {upper_path} and {surface_path}")
        upper = np.asarray(np.load(upper_path, allow_pickle=False), dtype=np.float32)
        surface = np.asarray(np.load(surface_path, allow_pickle=False), dtype=np.float32)
        if upper.shape != (5, len(PRESSURE_LEVELS), *GRID_SHAPE) or surface.shape != (4, *GRID_SHAPE):
            raise ValueError(f"Invalid GDAS input shapes at {directory}: upper={upper.shape}, surface={surface.shape}")
        if not np.isfinite(upper).all() or not np.isfinite(surface).all():
            raise ValueError(f"Non-finite GDAS input at {directory}")
        latitude = np.linspace(90, -90, GRID_SHAPE[0], dtype=np.float32)
        longitude = np.arange(GRID_SHAPE[1], dtype=np.float32) * 0.25
        files = (upper_path, surface_path)
        if (directory / "surface.nc").exists():
            with nc.Dataset(directory / "surface.nc") as ds:
                _validate_grid(ds, latitude, longitude, directory / "surface.nc")
            files += (directory / "surface.nc",)
        return InitialState(upper, surface, latitude, longitude, files)


def source_for_experiment(experiment: Experiment, *, era5_root: Path = ERA5_ROOT, gdas_root: Path = GDAS_ROOT):
    if experiment.source == "era5":
        return Era5Archive(era5_root or ERA5_ROOT)
    return NpyDirectorySource(gdas_root or GDAS_ROOT)

