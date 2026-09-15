"""Convert GDAS/FNL pressure-level fields to Pangu NetCDF and NPY inputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import netCDF4 as nc
import numpy as np
import pygrib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
VARIABLES = ["z", "q", "t", "u", "v"]
SHORT_NAME_MAP = {"gh": "z", "q": "q", "t": "t", "u": "u", "v": "v"}
UNITS = {
    "z": "m2 s-2",
    "q": "kg kg-1",
    "t": "K",
    "u": "m s-1",
    "v": "m s-1",
}


def parse_init_time(path: Path) -> datetime:
    match = re.search(r"\.(\d{10})\.f\d+\.grib2$", path.name)
    if not match:
        raise ValueError(f"Cannot parse initialization time from {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H")


def extract_fields(grib_path: Path):
    level_index = {level: index for index, level in enumerate(PRESSURE_LEVELS)}
    variable_index = {name: index for index, name in enumerate(VARIABLES)}
    upper_data = np.full((5, 13, 721, 1440), np.nan, dtype=np.float32)
    found: set[tuple[str, int]] = set()
    latitude = longitude = longitude_order = None
    reverse_latitude = False

    grib = pygrib.open(str(grib_path))
    try:
        for message in grib:
            if (
                message.shortName not in SHORT_NAME_MAP
                or message.typeOfLevel != "isobaricInhPa"
                or message.level not in level_index
            ):
                continue

            output_name = SHORT_NAME_MAP[message.shortName]
            key = (output_name, int(message.level))
            if key in found:
                continue

            values = np.ma.filled(message.values, np.nan).astype(np.float32)
            if message.shortName == "gh":
                values *= np.float32(9.80665)

            if latitude is None:
                latitudes, longitudes = message.latlons()
                latitude = latitudes[:, 0].astype(np.float32)
                longitude = np.mod(longitudes[0, :], 360.0).astype(np.float32)
                longitude_order = np.argsort(longitude)
                longitude = longitude[longitude_order]
                reverse_latitude = bool(latitude[0] < latitude[-1])
                if reverse_latitude:
                    latitude = latitude[::-1]

            values = values[:, longitude_order]
            if reverse_latitude:
                values = values[::-1, :]
            upper_data[variable_index[output_name], level_index[int(message.level)]] = values
            found.add(key)
            if len(found) == len(VARIABLES) * len(PRESSURE_LEVELS):
                break
    finally:
        grib.close()

    missing = [
        f"{name}@{level}hPa"
        for name in VARIABLES
        for level in PRESSURE_LEVELS
        if (name, level) not in found
    ]
    if missing:
        raise RuntimeError("Missing GDAS upper fields: " + ", ".join(missing))
    if latitude is None or longitude is None:
        raise RuntimeError("No valid GDAS upper-air grid was found")
    if latitude.shape != (721,) or longitude.shape != (1440,):
        raise RuntimeError(
            f"Unexpected GDAS grid: latitude={latitude.shape}, longitude={longitude.shape}"
        )
    return upper_data, latitude, longitude


def write_netcdf(
    path: Path,
    init_time: datetime,
    upper_data: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as dataset:
            dataset.createDimension("valid_time", 1)
            dataset.createDimension("pressure_level", len(PRESSURE_LEVELS))
            dataset.createDimension("latitude", latitude.size)
            dataset.createDimension("longitude", longitude.size)

            time_variable = dataset.createVariable("valid_time", "f8", ("valid_time",))
            time_variable.units = "hours since 1970-01-01 00:00:00"
            time_variable.calendar = "standard"
            time_variable[:] = nc.date2num(
                [init_time], units=time_variable.units, calendar=time_variable.calendar
            )
            dataset.createVariable("pressure_level", "i4", ("pressure_level",))[:] = PRESSURE_LEVELS
            dataset.createVariable("latitude", "f4", ("latitude",))[:] = latitude
            dataset.createVariable("longitude", "f4", ("longitude",))[:] = longitude

            for index, name in enumerate(VARIABLES):
                variable = dataset.createVariable(
                    name,
                    "f4",
                    ("valid_time", "pressure_level", "latitude", "longitude"),
                    zlib=True,
                    complevel=1,
                )
                variable.units = UNITS[name]
                variable[0, :, :, :] = upper_data[index]
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def convert(grib_path: Path) -> Path:
    init_time = parse_init_time(grib_path)
    output_dir = (
        PROJECT_ROOT
        / "model_input"
        / "single_time_point"
        / "gdas"
        / init_time.strftime("%Y-%m-%d-%H-%M")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    upper_data, latitude, longitude = extract_fields(grib_path)
    if upper_data.shape != (5, 13, 721, 1440):
        raise RuntimeError(f"Unexpected upper array shape: {upper_data.shape}")

    write_netcdf(output_dir / "upper.nc", init_time, upper_data, latitude, longitude)
    np.save(output_dir / "input_upper.npy", upper_data)
    print(f"[GDAS] upper input ready: {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grib_file", required=True, type=Path)
    args = parser.parse_args()
    convert(args.grib_file.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
