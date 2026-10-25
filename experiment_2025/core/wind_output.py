"""Write and verify compressed wind-only files on the supplied input grid."""

from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np


def validate_wind_file(path: Path, latitude: np.ndarray, longitude: np.ndarray, metadata: dict) -> bool:
    if not path.exists():
        return False
    try:
        with nc.Dataset(path) as ds:
            if set(ds.variables) != {"u10", "v10", "latitude", "longitude", "valid_time"}:
                return False
            if any(getattr(ds, key, None) != value for key, value in metadata.items()):
                return False
            if not np.array_equal(ds["latitude"][:], latitude) or not np.array_equal(ds["longitude"][:], longitude):
                return False
            expected_time = (datetime.fromisoformat(metadata["valid_time_utc"]) - datetime(1970, 1, 1)).total_seconds()
            if float(ds["valid_time"][...]) != expected_time:
                return False
            for name in ("u10", "v10"):
                var = ds[name]
                if var.shape != (len(latitude), len(longitude)) or var.dtype != np.dtype("float32"):
                    return False
                values = var[:]
                if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
                    return False
    except (OSError, RuntimeError, ValueError, KeyError):
        return False
    return True


def write_wind_file(path: Path, surface: np.ndarray, latitude: np.ndarray, longitude: np.ndarray, metadata: dict) -> None:
    """Copy channels 1/2 verbatim; publish only after the temporary file validates."""
    if surface.shape != (4, len(latitude), len(longitude)):
        raise ValueError(f"Unexpected surface shape: {surface.shape}")
    wind = np.asarray(surface[1:3], dtype=np.float32)
    if not np.isfinite(wind).all():
        raise ValueError("Wind output contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as ds:
            ds.setncatts({"Conventions": "CF-1.8", **metadata})
            ds.createDimension("latitude", len(latitude))
            ds.createDimension("longitude", len(longitude))
            lat = ds.createVariable("latitude", "f4", ("latitude",))
            lon = ds.createVariable("longitude", "f4", ("longitude",))
            lat.setncatts({"units": "degrees_north", "standard_name": "latitude"})
            lon.setncatts({"units": "degrees_east", "standard_name": "longitude"})
            lat[:] = latitude
            lon[:] = longitude
            time = ds.createVariable("valid_time", "f8", ())
            time.setncatts({"standard_name": "time", "units": "seconds since 1970-01-01 00:00:00", "calendar": "standard"})
            time[...] = (datetime.fromisoformat(metadata["valid_time_utc"]) - datetime(1970, 1, 1)).total_seconds()
            for index, (name, standard_name) in enumerate((("u10", "eastward_wind"), ("v10", "northward_wind"))):
                var = ds.createVariable(name, "f4", ("latitude", "longitude"), zlib=True, complevel=4, shuffle=True)
                var.setncatts({"units": "m s-1", "standard_name": standard_name, "long_name": f"10 metre {standard_name}", "coordinates": "valid_time"})
                var[:] = wind[index]
        if not validate_wind_file(temporary, latitude, longitude, metadata):
            raise RuntimeError(f"Wind output verification failed: {temporary}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
