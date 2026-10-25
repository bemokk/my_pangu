"""Convert one GDAS/FNL GRIB2 file to Pangu NetCDF and NPY inputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import warnings
from typing import Callable

import numpy as np

from .constants import (
    GRAVITY,
    GRID,
    PRESSURE_LEVELS,
    SURFACE_ORDER,
    SURFACE_TARGETS,
    SURFACE_UNITS,
    UPPER_ORDER,
    UPPER_SHORT_NAMES,
    UPPER_UNITS,
)
from .download import parse_init_time


def _pygrib():
    try:
        import pygrib
    except ImportError as exc:
        raise RuntimeError("pygrib is required to convert GDAS GRIB2 files") from exc
    return pygrib


def _values(message) -> np.ndarray:
    values = message.values
    if np.ma.getmaskarray(values).any():
        # pygrib 2.1.8 can mask valid gh values close to its 9999 sentinel,
        # even when the GRIB explicitly contains no missing grid points.
        # Unknown metadata must retain the mask, just like genuine missing data.
        try:
            no_missing = (
                message["numberOfMissing"] == 0
                and message["bitmapPresent"] == 0
                and message["missingValueManagementUsed"] == 0
                and message["numberOfDataPoints"] == message["numberOfValues"] == values.size
            )
        except Exception:  # ecCodes raises different exceptions for absent keys.
            no_missing = False
        if no_missing:
            count = int(np.ma.getmaskarray(values).sum())
            warnings.warn(
                f"GDAS {message.shortName}@{message.level}: preserving {count} original values "
                "masked by pygrib; GRIB metadata confirms no missing grid points",
                RuntimeWarning,
            )
            return np.ma.getdata(values).astype(np.float32, copy=False)
    return np.ma.filled(values, np.nan).astype(np.float32, copy=False)


def normalize_grid(message) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _values(message)
    latitudes, longitudes = message.latlons()
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    if values.ndim != 2 or latitudes.shape != values.shape or longitudes.shape != values.shape:
        raise ValueError(f"unexpected GDAS field shape: values={values.shape}, lat={latitudes.shape}, lon={longitudes.shape}")
    if not np.allclose(latitudes, latitudes[:, :1], rtol=0, atol=1e-5) or not np.allclose(longitudes, longitudes[:1, :], rtol=0, atol=1e-5):
        raise ValueError("GDAS requires a regular latitude/longitude grid")
    latitude = latitudes[:, 0].astype(np.float32)
    longitude = np.mod(longitudes[0, :], 360.0).astype(np.float32)
    longitude_order = np.argsort(longitude)
    longitude = longitude[longitude_order]
    values = values[:, longitude_order]
    if latitude[0] < latitude[-1]:
        latitude = latitude[::-1]
        values = values[::-1, :]
    return values, latitude, longitude


def _matches(message, target: dict) -> bool:
    return all(getattr(message, key, None) == value for key, value in target.items())


def _check_grid(latitude: np.ndarray, longitude: np.ndarray, expected_shape: tuple[int, int] | None) -> None:
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("GDAS grid coordinates contain non-finite values")
    if expected_shape is not None and (latitude.size, longitude.size) != tuple(expected_shape):
        raise ValueError(f"unexpected GDAS grid: latitude={latitude.shape}, longitude={longitude.shape}; expected {expected_shape}")
    if latitude.size > 1 and not np.all(np.diff(latitude) < 0):
        raise ValueError("GDAS latitude must be strictly descending")
    if longitude.size > 1 and not np.all(np.diff(longitude) > 0):
        raise ValueError("GDAS longitude must be strictly ascending")
    if expected_shape == GRID:
        expected_latitude = np.linspace(90.0, -90.0, GRID[0], dtype=np.float32)
        expected_longitude = np.arange(GRID[1], dtype=np.float32) * 0.25
        if not np.allclose(latitude, expected_latitude, rtol=0, atol=1e-4) or not np.allclose(longitude, expected_longitude, rtol=0, atol=1e-4):
            raise ValueError("GDAS grid does not match the required 0.25-degree global grid")


def _same_grid(lat_a, lon_a, lat_b, lon_b) -> bool:
    return (lat_a.shape == lat_b.shape and lon_a.shape == lon_b.shape
            and np.allclose(lat_a, lat_b, rtol=0, atol=1e-5)
            and np.allclose(lon_a, lon_b, rtol=0, atol=1e-5))


def _check_time(message, expected_time: datetime) -> None:
    if message.validDate != expected_time or message.analDate != expected_time:
        raise ValueError(f"GDAS field time does not match the analysis file: {message.validDate}, expected {expected_time}")


def extract_surface_fields(grib_path: Path, *, expected_shape: tuple[int, int] | None = GRID):
    pygrib = _pygrib()
    init_time = parse_init_time(grib_path)
    fields: dict[str, np.ndarray] = {}
    latitude = longitude = None
    grib = pygrib.open(str(grib_path))
    try:
        for message in grib:
            for output_name, target in SURFACE_TARGETS.items():
                if output_name in fields or not _matches(message, target):
                    continue
                _check_time(message, init_time)
                values, current_latitude, current_longitude = normalize_grid(message)
                if latitude is None:
                    latitude, longitude = current_latitude, current_longitude
                elif not _same_grid(latitude, longitude, current_latitude, current_longitude):
                    raise ValueError(f"surface field {output_name} uses a different grid")
                fields[output_name] = values
            if len(fields) == len(SURFACE_TARGETS):
                break
    finally:
        grib.close()
    missing = [name for name in SURFACE_ORDER if name not in fields]
    if missing:
        raise ValueError(f"missing GDAS surface fields: {missing}")
    if latitude is None or longitude is None:
        raise ValueError("no valid GDAS surface grid found")
    _check_grid(latitude, longitude, expected_shape)
    if not all(np.isfinite(fields[name]).all() for name in SURFACE_ORDER):
        raise ValueError("GDAS surface fields contain missing or non-finite values")
    return fields, latitude, longitude


def extract_upper_fields(grib_path: Path, *, expected_shape: tuple[int, int] | None = GRID):
    pygrib = _pygrib()
    init_time = parse_init_time(grib_path)
    level_index = {level: index for index, level in enumerate(PRESSURE_LEVELS)}
    variable_index = {name: index for index, name in enumerate(UPPER_ORDER)}
    latitude = longitude = None
    data = None
    found: set[tuple[str, int]] = set()
    grib = pygrib.open(str(grib_path))
    try:
        for message in grib:
            short_name = getattr(message, "shortName", None)
            level_type = getattr(message, "typeOfLevel", None)
            level = getattr(message, "level", None)
            if short_name not in UPPER_SHORT_NAMES or level_type != "isobaricInhPa" or level not in level_index:
                continue
            output_name = UPPER_SHORT_NAMES[short_name]
            key = (output_name, int(level))
            if key in found:
                continue
            _check_time(message, init_time)
            values, current_latitude, current_longitude = normalize_grid(message)
            if latitude is None:
                latitude, longitude = current_latitude, current_longitude
                data = np.full((len(UPPER_ORDER), len(PRESSURE_LEVELS), *values.shape), np.nan, dtype=np.float32)
            elif not _same_grid(latitude, longitude, current_latitude, current_longitude):
                raise ValueError(f"upper field {output_name}@{level} uses a different grid")
            if short_name == "gh":
                values = values * np.float32(GRAVITY)
            data[variable_index[output_name], level_index[int(level)]] = values
            found.add(key)
            if len(found) == len(UPPER_ORDER) * len(PRESSURE_LEVELS):
                break
    finally:
        grib.close()
    missing = [f"{name}@{level}hPa" for name in UPPER_ORDER for level in PRESSURE_LEVELS if (name, level) not in found]
    if missing:
        raise ValueError("missing GDAS upper fields: " + ", ".join(missing))
    if data is None or latitude is None or longitude is None:
        raise ValueError("no valid GDAS upper-air grid found")
    _check_grid(latitude, longitude, expected_shape)
    if not np.isfinite(data).all():
        raise ValueError("GDAS upper fields contain missing or non-finite values")
    return data, latitude, longitude


def _atomic_netcdf(path: Path, writer: Callable) -> None:
    import netCDF4 as nc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with nc.Dataset(str(temporary), "w", format="NETCDF4") as dataset:
            writer(dataset)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_npy(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as stream:
        np.save(stream, data, allow_pickle=False)
    temporary.replace(path)


def write_surface(path: Path, init_time: datetime, fields, latitude, longitude) -> None:
    import netCDF4 as nc

    def writer(dataset):
        dataset.createDimension("valid_time", 1)
        dataset.createDimension("latitude", latitude.size)
        dataset.createDimension("longitude", longitude.size)
        variable = dataset.createVariable("valid_time", "f8", ("valid_time",))
        variable.units = "hours since 1970-01-01 00:00:00"
        variable.calendar = "standard"
        variable[:] = nc.date2num([init_time], variable.units, variable.calendar)
        dataset.createVariable("latitude", "f4", ("latitude",))[:] = latitude
        dataset.createVariable("longitude", "f4", ("longitude",))[:] = longitude
        dataset["latitude"].units = "degrees_north"
        dataset["longitude"].units = "degrees_east"
        for name in SURFACE_ORDER:
            output = dataset.createVariable(name, "f4", ("valid_time", "latitude", "longitude"), zlib=True, complevel=1)
            output.units = SURFACE_UNITS[name]
            output[0] = fields[name]

    _atomic_netcdf(path, writer)


def write_upper(path: Path, init_time: datetime, data, latitude, longitude) -> None:
    import netCDF4 as nc

    def writer(dataset):
        dataset.createDimension("valid_time", 1)
        dataset.createDimension("pressure_level", len(PRESSURE_LEVELS))
        dataset.createDimension("latitude", latitude.size)
        dataset.createDimension("longitude", longitude.size)
        variable = dataset.createVariable("valid_time", "f8", ("valid_time",))
        variable.units = "hours since 1970-01-01 00:00:00"
        variable.calendar = "standard"
        variable[:] = nc.date2num([init_time], variable.units, variable.calendar)
        dataset.createVariable("pressure_level", "i4", ("pressure_level",))[:] = PRESSURE_LEVELS
        dataset.createVariable("latitude", "f4", ("latitude",))[:] = latitude
        dataset.createVariable("longitude", "f4", ("longitude",))[:] = longitude
        dataset["latitude"].units = "degrees_north"
        dataset["longitude"].units = "degrees_east"
        dataset["pressure_level"].units = "hPa"
        for index, name in enumerate(UPPER_ORDER):
            output = dataset.createVariable(name, "f4", ("valid_time", "pressure_level", "latitude", "longitude"), zlib=True, complevel=1)
            output.units = UPPER_UNITS[name]
            output[0] = data[index]

    _atomic_netcdf(path, writer)


OUTPUT_FILES = ("surface.nc", "upper.nc", "input_surface.npy", "input_upper.npy")
CONVERSION_VERSION = 1


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _is_complete(directory: Path, identity: dict) -> bool:
    try:
        manifest = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
        return (manifest["version"] == CONVERSION_VERSION and manifest["source"] == identity
                and all(manifest["sha256"][name] == _file_hash(directory / name) for name in OUTPUT_FILES))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _verify_outputs(directory: Path, surface, upper) -> None:
    import netCDF4 as nc

    for filename, names, values in (("surface.nc", SURFACE_ORDER, surface), ("upper.nc", UPPER_ORDER, upper)):
        with nc.Dataset(str(directory / filename)) as dataset:
            for index, name in enumerate(names):
                expected = values[name] if filename == "surface.nc" else values[index]
                actual = dataset[name][0]
                if np.ma.getmaskarray(actual).any() or not np.array_equal(actual, expected):
                    raise IOError(f"GDAS NetCDF readback differs from GRIB values: {filename} {name}")
    for filename, expected in (("input_surface.npy", np.stack([surface[name] for name in SURFACE_ORDER])),
                               ("input_upper.npy", upper)):
        actual = np.load(directory / filename, allow_pickle=False, mmap_mode="r")
        try:
            if actual.dtype != np.float32 or not np.array_equal(actual, expected):
                raise IOError(f"GDAS NPY readback differs from GRIB values: {filename}")
        finally:
            actual._mmap.close()


def convert_grib(grib_path: Path, output_root: Path | None = None, *, overwrite: bool = False,
                 expected_shape: tuple[int, int] | None = GRID) -> Path:
    grib_path = Path(grib_path).resolve()
    if not grib_path.is_file():
        raise FileNotFoundError(grib_path)
    init_time = parse_init_time(grib_path)
    identity = _source_identity(grib_path)
    if output_root is None:
        from ..core.config import GDAS_ROOT
        output_root = GDAS_ROOT
    from .paths import prepared_directory

    output_dir = prepared_directory(output_root, init_time)
    if _is_complete(output_dir, identity) and not overwrite:
        return output_dir
    surface, surface_lat, surface_lon = extract_surface_fields(grib_path, expected_shape=expected_shape)
    upper, upper_lat, upper_lon = extract_upper_fields(grib_path, expected_shape=expected_shape)
    if not _same_grid(surface_lat, surface_lon, upper_lat, upper_lon):
        raise ValueError("surface and upper fields use different grids")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "prepared.json"
    if manifest_path.exists():
        manifest_path.unlink()
    write_surface(output_dir / "surface.nc", init_time, surface, surface_lat, surface_lon)
    write_upper(output_dir / "upper.nc", init_time, upper, upper_lat, upper_lon)
    _write_npy(output_dir / "input_surface.npy", np.stack([surface[name] for name in SURFACE_ORDER]).astype(np.float32))
    _write_npy(output_dir / "input_upper.npy", upper)
    _verify_outputs(output_dir, surface, upper)
    if _source_identity(grib_path) != identity:
        raise RuntimeError(f"GDAS source changed during conversion: {grib_path}")
    manifest = {"version": CONVERSION_VERSION, "source": identity, "valid_time": init_time.isoformat(),
                "sha256": {name: _file_hash(output_dir / name) for name in OUTPUT_FILES}}
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return output_dir

