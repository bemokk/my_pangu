"""Export cropped Wipha-case ERA5, G0, and E120 10 m wind fields to one NetCDF.

The spatial selection follows ``old/Buoy/plots/plot_wind_compare.py``:
10--30 degrees north and 105--130 degrees east.  No figure is produced.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import netCDF4 as nc
import xarray as xr

warnings.filterwarnings(
    "ignore",
    message=r"Engine 'cfgrib' loading failed:",
    category=RuntimeWarning,
    module=r"xarray\.backends\.plugins",
)


STATS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATS_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_2025.core.config import ERA5_ROOT, OUTPUT_ROOT  # noqa: E402
from experiment_2025.core.experiments import EXPERIMENTS, time_label, wind_path  # noqa: E402


CASE_NAME = "Wipha"
CASE_START = datetime(2025, 7, 19, 0, 0)
LEAD_HOURS = (0, 24, 48, 72, 96, 120)
AREA = (30.0, 105.0, 10.0, 130.0)  # lat_max, lon_min, lat_min, lon_max
OUTPUT_NAME = "wipha_2025071900_era5_realtime_g0_e120_surface_wind_10N30N_105E130E.nc"
SOURCE_CONFIG = {
    "era5_realtime": {"label": "ERA5 realtime analysis", "experiment": None},
    "g0": {"label": "G0 (gdas_direct)", "experiment": "gdas_direct"},
    "e120": {"label": "E120 (era5_lagged_120h)", "experiment": "era5_lagged_120h"},
}


def find_coord_or_dim(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in ds.coords or candidate in ds.dims or candidate in ds.variables:
            return candidate
    return None


def rename_common_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    time_name = find_coord_or_dim(ds, ("valid_time", "time"))
    lat_name = find_coord_or_dim(ds, ("latitude", "lat"))
    lon_name = find_coord_or_dim(ds, ("longitude", "lon"))
    if time_name and time_name != "valid_time":
        rename[time_name] = "valid_time"
    if lat_name and lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name and lon_name != "longitude":
        rename[lon_name] = "longitude"
    return ds.rename(rename) if rename else ds


def select_first_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        if ds.sizes["valid_time"] != 1:
            raise ValueError(f"Expected one valid time, found {ds.sizes['valid_time']}")
        return ds.isel(valid_time=0, drop=True)
    if "time" in ds.dims:
        if ds.sizes["time"] != 1:
            raise ValueError(f"Expected one time, found {ds.sizes['time']}")
        return ds.isel(time=0, drop=True)
    return ds


def normalize_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        raise KeyError("Dataset must contain latitude and longitude coordinates")
    longitude = ((ds["longitude"] + 360.0) % 360.0).astype(np.float32)
    latitude = ds["latitude"].astype(np.float32)
    ds = ds.assign_coords(longitude=longitude, latitude=latitude)
    return ds.sortby("longitude").sortby("latitude", ascending=False)


def crop_area(ds: xr.Dataset, area: tuple[float, float, float, float]) -> xr.Dataset:
    lat_max, lon_min, lat_min, lon_max = area
    ds = normalize_lon_lat(ds)
    ds = ds.sel(latitude=slice(lat_max, lat_min))
    if lon_min <= lon_max:
        return ds.sel(longitude=slice(lon_min, lon_max))
    left = ds.sel(longitude=slice(lon_min, 359.999))
    right = ds.sel(longitude=slice(0.0, lon_max))
    return xr.concat((left, right), dim="longitude")


def find_var_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    wanted = {candidate.lower() for candidate in candidates}
    for name, variable in ds.data_vars.items():
        attributes = {
            str(variable.attrs.get(key, "")).lower()
            for key in ("shortName", "GRIB_shortName", "standard_name", "long_name")
        }
        if attributes & wanted:
            return name
    return None


def drop_extra_dims(da: xr.DataArray) -> xr.DataArray:
    for dimension in list(da.dims):
        if dimension in ("latitude", "longitude"):
            continue
        if da.sizes[dimension] != 1:
            raise ValueError(f"Variable {da.name!r} has unsupported dimension {dimension!r}")
        da = da.isel({dimension: 0}, drop=True)
    return da


def load_surface_wind(path: Path, area: tuple[float, float, float, float]) -> xr.Dataset:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as source:
        ds = source.load()
    ds = crop_area(select_first_time(rename_common_coords(ds)), area)
    u_name = find_var_name(ds, ("u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u10m"))
    v_name = find_var_name(ds, ("v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "v10m"))
    if u_name is None or v_name is None:
        raise KeyError(f"Unable to identify u10/v10 in {path}; variables={list(ds.data_vars)}")
    u10 = drop_extra_dims(ds[u_name]).transpose("latitude", "longitude").astype(np.float32)
    v10 = drop_extra_dims(ds[v_name]).transpose("latitude", "longitude").astype(np.float32)
    wind_speed = np.hypot(u10, v10).astype(np.float32)
    return xr.Dataset({"u10": u10, "v10": v10, "wind_speed": wind_speed})


def era5_realtime_path(era5_root: Path, valid_time: datetime) -> Path:
    return era5_root / str(valid_time.year) / time_label(valid_time) / "surface.nc"


def source_paths(era5_root: Path, output_root: Path) -> dict[str, list[Path]]:
    paths = {source: [] for source in SOURCE_CONFIG}
    for lead_hour in LEAD_HOURS:
        valid_time = CASE_START + timedelta(hours=lead_hour)
        paths["era5_realtime"].append(era5_realtime_path(era5_root, valid_time))
        paths["g0"].append(
            wind_path(output_root, EXPERIMENTS["gdas_direct"], CASE_START, lead_hour)
        )
        paths["e120"].append(
            wind_path(output_root, EXPERIMENTS["era5_lagged_120h"], CASE_START, lead_hour)
        )
    return paths


def align_to_reference(field: xr.Dataset, reference: xr.Dataset) -> xr.Dataset:
    same_lat = np.array_equal(field.latitude.values, reference.latitude.values)
    same_lon = np.array_equal(field.longitude.values, reference.longitude.values)
    if same_lat and same_lon:
        return field
    aligned = field.interp(
        latitude=reference.latitude,
        longitude=reference.longitude,
        method="nearest",
    )
    return aligned.assign_coords(
        latitude=reference.latitude.values,
        longitude=reference.longitude.values,
    )


def build_dataset(era5_root: Path, output_root: Path) -> xr.Dataset:
    paths = source_paths(era5_root, output_root)
    missing = [str(path) for source in paths.values() for path in source if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))

    values: dict[str, dict[str, list[np.ndarray]]] = {
        source: {variable: [] for variable in ("u10", "v10", "wind_speed")}
        for source in SOURCE_CONFIG
    }
    reference_coordinates = None
    for index, lead_hour in enumerate(LEAD_HOURS):
        reference = load_surface_wind(paths["era5_realtime"][index], AREA)
        if reference_coordinates is None:
            reference_coordinates = reference[["u10"]].drop_vars("u10")
        elif not (
            np.array_equal(reference.latitude.values, reference_coordinates.latitude.values)
            and np.array_equal(reference.longitude.values, reference_coordinates.longitude.values)
        ):
            raise ValueError(f"ERA5 cropped grid changed at lead {lead_hour} h")

        fields = {
            "era5_realtime": reference,
            "g0": align_to_reference(load_surface_wind(paths["g0"][index], AREA), reference),
            "e120": align_to_reference(load_surface_wind(paths["e120"][index], AREA), reference),
        }
        for source, field in fields.items():
            for variable in values[source]:
                array = np.asarray(field[variable].values, dtype=np.float32)
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite {source}/{variable} values at lead {lead_hour} h")
                values[source][variable].append(array)

    if reference_coordinates is None:
        raise RuntimeError("No wind fields were loaded")
    valid_times = np.asarray(
        [CASE_START + timedelta(hours=lead) for lead in LEAD_HOURS], dtype="datetime64[ns]"
    )
    result = xr.Dataset(
        coords={
            "lead_hour": np.asarray(LEAD_HOURS, dtype=np.int32),
            "valid_time": ("lead_hour", valid_times),
            "latitude": reference_coordinates.latitude.values.astype(np.float32),
            "longitude": reference_coordinates.longitude.values.astype(np.float32),
        }
    )
    for source, source_values in values.items():
        for variable, arrays in source_values.items():
            name = f"{source}_{variable}"
            result[name] = (("lead_hour", "latitude", "longitude"), np.stack(arrays))
            result[name].attrs.update(
                {
                    "units": "m s-1",
                    "long_name": f"{SOURCE_CONFIG[source]['label']} 10 m {variable.replace('_', ' ')}",
                    "source_key": source,
                }
            )
    result.lead_hour.attrs.update({"units": "hours", "long_name": "forecast lead from case target time"})
    result.latitude.attrs.update({"units": "degrees_north", "standard_name": "latitude"})
    result.longitude.attrs.update({"units": "degrees_east", "standard_name": "longitude"})
    result.valid_time.attrs.update({"standard_name": "time", "long_name": "UTC valid time"})
    result.attrs.update(
        {
            "Conventions": "CF-1.8",
            "title": "Typhoon Wipha 10 m surface wind fields: ERA5 realtime, G0, and E120",
            "case_name": CASE_NAME,
            "case_target_time_utc": CASE_START.isoformat(),
            "spatial_extent": "10-30N, 105-130E (inclusive)",
            "area_order_latmax_lonmin_latmin_lonmax": json.dumps(AREA),
            "g0_definition": "gdas_direct initialized at 2025-07-19T00:00:00 UTC",
            "e120_definition": "era5_lagged_120h initialized at 2025-07-14T00:00:00 UTC; model hour=120+lead_hour",
            "era5_realtime_definition": "ERA5 analysis at each valid_time",
            "source_files_json": json.dumps(
                {source: [str(path) for path in source_list] for source, source_list in paths.items()},
                ensure_ascii=False,
            ),
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "history": "Cropped and combined without plotting by export_wipha_surface_wind_nc.py",
        }
    )
    return result


def validate_output(path: Path) -> dict:
    with nc.Dataset(path) as output:
        expected_sizes = {"lead_hour": 6, "latitude": 81, "longitude": 101}
        actual_sizes = {name: len(output.dimensions[name]) for name in expected_sizes}
        if actual_sizes != expected_sizes:
            raise ValueError(f"Unexpected output sizes: {actual_sizes}")
        expected_variables = {
            f"{source}_{variable}"
            for source in SOURCE_CONFIG
            for variable in ("u10", "v10", "wind_speed")
        }
        data_variables = set(output.variables) - {
            "lead_hour", "valid_time", "latitude", "longitude"
        }
        if data_variables != expected_variables:
            raise ValueError(f"Unexpected output variables: {sorted(data_variables)}")
        output_leads = tuple(np.asarray(output.variables["lead_hour"][:]).tolist())
        if output_leads != LEAD_HOURS:
            raise ValueError(f"Unexpected lead hours: {output_leads}")
        latitude = np.asarray(output.variables["latitude"][:])
        longitude = np.asarray(output.variables["longitude"][:])
        bounds = {
            "lat_max": float(latitude.max()),
            "lat_min": float(latitude.min()),
            "lon_min": float(longitude.min()),
            "lon_max": float(longitude.max()),
        }
        if bounds != {"lat_max": 30.0, "lat_min": 10.0, "lon_min": 105.0, "lon_max": 130.0}:
            raise ValueError(f"Unexpected bounds: {bounds}")
        for variable in expected_variables:
            if not np.isfinite(np.asarray(output.variables[variable][:])).all():
                raise ValueError(f"Nonfinite values in {variable}")
        return {
            "sizes": actual_sizes,
            "bounds": bounds,
            "variables": sorted(expected_variables),
            "file_size_bytes": path.stat().st_size,
        }


def write_dataset(dataset: xr.Dataset, path: Path, overwrite: bool) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with nc.Dataset(temporary, "w", format="NETCDF4") as output:
            output.createDimension("lead_hour", dataset.sizes["lead_hour"])
            output.createDimension("latitude", dataset.sizes["latitude"])
            output.createDimension("longitude", dataset.sizes["longitude"])

            lead = output.createVariable("lead_hour", "i4", ("lead_hour",))
            lead[:] = dataset.lead_hour.values
            lead.setncatts(dict(dataset.lead_hour.attrs))

            valid_time = output.createVariable("valid_time", "f8", ("lead_hour",))
            valid_time.units = "hours since 2025-07-19 00:00:00 UTC"
            valid_time.calendar = "standard"
            valid_time.standard_name = "time"
            valid_time.long_name = "UTC valid time"
            valid_time[:] = dataset.lead_hour.values.astype(float)

            latitude = output.createVariable("latitude", "f4", ("latitude",))
            latitude[:] = dataset.latitude.values
            latitude.setncatts(dict(dataset.latitude.attrs))
            longitude = output.createVariable("longitude", "f4", ("longitude",))
            longitude[:] = dataset.longitude.values
            longitude.setncatts(dict(dataset.longitude.attrs))

            for name, array in dataset.data_vars.items():
                variable = output.createVariable(
                    name,
                    "f4",
                    ("lead_hour", "latitude", "longitude"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=(1, dataset.sizes["latitude"], dataset.sizes["longitude"]),
                )
                variable[:] = np.asarray(array.values, dtype=np.float32)
                variable.setncatts(dict(array.attrs))
            output.setncatts(dict(dataset.attrs))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return validate_output(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-root", type=Path, default=ERA5_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=STATS_ROOT / "data" / "wipha_surface_wind",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    destination = args.data_root / OUTPUT_NAME
    dataset = build_dataset(args.era5_root, args.output_root)
    report = write_dataset(dataset, destination, args.overwrite)
    print(f"Saved: {destination}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
