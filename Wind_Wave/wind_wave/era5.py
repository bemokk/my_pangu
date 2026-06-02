from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr


TIME_CANDIDATES = ("time", "valid_time")
LAT_CANDIDATES = ("latitude", "lat")
LON_CANDIDATES = ("longitude", "lon")


def find_data_var(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    candidate_set = {candidate.lower() for candidate in candidates}

    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate

    for var_name, data_array in ds.data_vars.items():
        attrs = data_array.attrs
        values = (
            str(attrs.get("shortName", "")).lower(),
            str(attrs.get("GRIB_shortName", "")).lower(),
            str(attrs.get("standard_name", "")).lower(),
            str(attrs.get("long_name", "")).lower(),
        )
        if any(value in candidate_set for value in values):
            return var_name

    raise KeyError(f"Could not find any variable matching {list(candidates)}")


def normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for candidate in TIME_CANDIDATES:
        if candidate in ds.coords or candidate in ds.dims or candidate in ds.variables:
            if candidate != "time":
                rename[candidate] = "time"
            break
    else:
        raise KeyError("Dataset does not contain a time or valid_time coordinate")

    if rename:
        ds = ds.rename(rename)

    if "time" not in ds.dims:
        if "time" not in ds.coords:
            raise KeyError("Dataset does not contain a usable time coordinate")
        ds = ds.expand_dims(time=np.atleast_1d(ds["time"].values))

    return ds.sortby("time")


def normalize_spatial_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for candidate in LAT_CANDIDATES:
        if candidate in ds.coords or candidate in ds.dims or candidate in ds.variables:
            if candidate != "latitude":
                rename[candidate] = "latitude"
            break
    else:
        raise KeyError("Dataset does not contain a latitude or lat coordinate")

    for candidate in LON_CANDIDATES:
        if candidate in ds.coords or candidate in ds.dims or candidate in ds.variables:
            if candidate != "longitude":
                rename[candidate] = "longitude"
            break
    else:
        raise KeyError("Dataset does not contain a longitude or lon coordinate")

    if rename:
        ds = ds.rename(rename)
    return ds


def drop_extra_dims(ds: xr.Dataset, allowed_dims: set[str] | None = None) -> xr.Dataset:
    allowed = allowed_dims or {"time", "latitude", "longitude"}
    for dim in list(ds.dims):
        if dim in allowed:
            continue
        if dim == "expver":
            ds = ds.max(dim=dim, skipna=True)
        elif ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
        else:
            raise ValueError(f"Unsupported extra dimension {dim} with size {ds.sizes[dim]}")
    return ds


def align_wind_to_wave_grid(wind_ds: xr.Dataset, wave_ds: xr.Dataset) -> xr.Dataset:
    try:
        return wind_ds.sel(
            latitude=wave_ds["latitude"],
            longitude=wave_ds["longitude"],
        )
    except Exception as exc:
        raise ValueError(
            "Wind grid cannot be aligned to wave latitude/longitude coordinates"
        ) from exc


def direction_degrees_to_unit(degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(np.asarray(degrees, dtype=np.float32))
    sin_v = np.sin(radians).astype(np.float32)
    cos_v = np.cos(radians).astype(np.float32)
    return sin_v, cos_v
