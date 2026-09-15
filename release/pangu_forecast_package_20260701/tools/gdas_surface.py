"""Convert GDAS/FNL surface fields to Pangu NetCDF and NPY inputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import netCDF4 as nc
import numpy as np
import pygrib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "msl": {"shortName": "prmsl", "typeOfLevel": "meanSea", "level": 0},
    "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
    "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
    "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
}
UNITS = {
    "msl": "Pa",
    "u10": "m s-1",
    "v10": "m s-1",
    "t2m": "K",
}


def parse_init_time(path: Path) -> datetime:
    match = re.search(r"\.(\d{10})\.f\d+\.grib2$", path.name)
    if not match:
        raise ValueError(f"Cannot parse initialization time from {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H")


def normalize_grid(message) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.ma.filled(message.values, np.nan).astype(np.float32)
    latitudes, longitudes = message.latlons()
    latitude = latitudes[:, 0].astype(np.float32)
    longitude = np.mod(longitudes[0, :], 360.0).astype(np.float32)

    longitude_order = np.argsort(longitude)
    longitude = longitude[longitude_order]
    values = values[:, longitude_order]
    if latitude[0] < latitude[-1]:
        latitude = latitude[::-1]
        values = values[::-1, :]
    return values, latitude, longitude


def extract_fields(grib_path: Path):
    fields: dict[str, np.ndarray] = {}
    latitude = longitude = None
    grib = pygrib.open(str(grib_path))
    try:
        for message in grib:
            for output_name, target in TARGETS.items():
                if output_name in fields:
                    continue
                if (
                    message.shortName == target["shortName"]
                    and message.typeOfLevel == target["typeOfLevel"]
                    and message.level == target["level"]
                ):
                    values, current_latitude, current_longitude = normalize_grid(message)
                    fields[output_name] = values
                    if latitude is None:
                        latitude = current_latitude
                        longitude = current_longitude
            if len(fields) == len(TARGETS):
                break
    finally:
        grib.close()

    missing = [name for name in TARGETS if name not in fields]
    if missing:
        raise RuntimeError(f"Missing GDAS surface variables: {missing}")
    if latitude is None or longitude is None:
        raise RuntimeError("No valid GDAS surface grid was found")
    if latitude.shape != (721,) or longitude.shape != (1440,):
        raise RuntimeError(
            f"Unexpected GDAS grid: latitude={latitude.shape}, longitude={longitude.shape}"
        )
    return fields, latitude, longitude


def write_netcdf(
    path: Path,
    init_time: datetime,
    fields: dict[str, np.ndarray],
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as dataset:
            dataset.createDimension("valid_time", 1)
            dataset.createDimension("latitude", latitude.size)
            dataset.createDimension("longitude", longitude.size)

            time_variable = dataset.createVariable("valid_time", "f8", ("valid_time",))
            time_variable.units = "hours since 1970-01-01 00:00:00"
            time_variable.calendar = "standard"
            time_variable[:] = nc.date2num(
                [init_time], units=time_variable.units, calendar=time_variable.calendar
            )
            dataset.createVariable("latitude", "f4", ("latitude",))[:] = latitude
            dataset.createVariable("longitude", "f4", ("longitude",))[:] = longitude

            for name in ["msl", "u10", "v10", "t2m"]:
                variable = dataset.createVariable(
                    name,
                    "f4",
                    ("valid_time", "latitude", "longitude"),
                    zlib=True,
                    complevel=1,
                )
                variable.units = UNITS[name]
                variable[0, :, :] = fields[name]
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
    fields, latitude, longitude = extract_fields(grib_path)

    surface_data = np.stack(
        [fields[name] for name in ["msl", "u10", "v10", "t2m"]], axis=0
    ).astype(np.float32)
    if surface_data.shape != (4, 721, 1440):
        raise RuntimeError(f"Unexpected surface array shape: {surface_data.shape}")

    write_netcdf(output_dir / "surface.nc", init_time, fields, latitude, longitude)
    np.save(output_dir / "input_surface.npy", surface_data)
    print(f"[GDAS] surface input ready: {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grib_file", required=True, type=Path)
    args = parser.parse_args()
    convert(args.grib_file.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
