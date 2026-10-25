# -*- coding: utf-8 -*-
"""Plot 2 x 6 wind-speed bias maps for GDAS and ERA5_Lagged.

Rows:
1. GDAS_Realtime minus ERA5 realtime
2. ERA5_Lagged minus ERA5 realtime

Columns are forecast lead times 12, 24, 36, 48, 60, and 72 h. All panels
share one symmetric wind-speed-bias colorbar at the right-hand side.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator

os.environ["SHAPE_RESTORE_SHX"] = "YES"

try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception:
    gpd = None
    box = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# User-adjustable plotting settings.
AREA = [30, 105, 10, 130]  # [lat_max, lon_min, lat_min, lon_max]
LEAD_HOURS = [12, 24, 36, 48, 60, 72]
CASE_START = np.datetime64("2025-07-17T00:00")
GDAS_INIT_LABEL = "2025-07-17-00-00"
ERA5_LAGGED_INIT_LABEL = "2025-07-12-00-00"
ERA5_LAGGED_OFFSET_HOURS = 120

FIGSIZE = (22, 7.8)
PANEL_WSPACE = 0.12
PANEL_HSPACE = 0.18
BIAS_LIMIT = None  # Set a positive value, e.g. 15, to fix the common color scale.
COLORBAR_RECT = [0.935, 0.28, 0.012, 0.44]  # [left, bottom, width, height]
FONT_SIZE = 11
TITLE_SIZE = 11
LON_TICK_INTERVAL = 10
LAT_TICK_INTERVAL = 5
FONT_FAMILY = ["Times New Roman", "SimSun", "SimHei", "Microsoft YaHei", "DejaVu Serif"]
LAND_BOUNDARY_FILE = None
SHOW_FIG = False

OUT_DIR = PROJECT_ROOT / "Buoy" / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = "Figure_wind_speed_bias_GDAS_ERA5Lagged_12_72h_2x6"


def find_coord_or_dim(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in ds.coords or candidate in ds.dims or candidate in ds.variables:
            return candidate
    return None


def rename_common_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}
    time_name = find_coord_or_dim(ds, ["valid_time", "time"])
    lat_name = find_coord_or_dim(ds, ["latitude", "lat"])
    lon_name = find_coord_or_dim(ds, ["longitude", "lon"])
    if time_name and time_name != "valid_time":
        rename_dict[time_name] = "valid_time"
    if lat_name and lat_name != "latitude":
        rename_dict[lat_name] = "latitude"
    if lon_name and lon_name != "longitude":
        rename_dict[lon_name] = "longitude"
    return ds.rename(rename_dict) if rename_dict else ds


def normalize_lon_lat(ds: xr.Dataset | xr.DataArray):
    if "longitude" not in ds.coords or "latitude" not in ds.coords:
        raise RuntimeError("Dataset must contain longitude and latitude coordinates")
    longitude = ((ds["longitude"] + 360) % 360).astype(np.float32)
    return ds.assign_coords(longitude=longitude).sortby("longitude").sortby(
        "latitude", ascending=False
    )


def crop_area(ds: xr.Dataset | xr.DataArray, area):
    lat_max, lon_min, lat_min, lon_max = area
    ds = normalize_lon_lat(ds).sel(latitude=slice(lat_max, lat_min))
    lon_min = (lon_min + 360) % 360
    lon_max = (lon_max + 360) % 360
    if lon_min <= lon_max:
        return ds.sel(longitude=slice(lon_min, lon_max))
    return xr.concat(
        [
            ds.sel(longitude=slice(lon_min, 359.999)),
            ds.sel(longitude=slice(0, lon_max)),
        ],
        dim="longitude",
    )


def drop_extra_dims(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in {"latitude", "longitude"}:
            continue
        if da.sizes[dim] != 1:
            raise RuntimeError(
                f"Variable {da.name} has unsupported dimension {dim}={da.sizes[dim]}"
            )
        da = da.isel({dim: 0}, drop=True)
    return da


def find_var_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    candidates_lower = {candidate.lower() for candidate in candidates}
    for variable in ds.data_vars:
        attrs = ds[variable].attrs
        values = {
            str(attrs.get("shortName", "")).lower(),
            str(attrs.get("GRIB_shortName", "")).lower(),
            str(attrs.get("standard_name", "")).lower(),
            str(attrs.get("long_name", "")).lower(),
        }
        if values & candidates_lower:
            return variable
    return None


def valid_time_from_lead(lead_hour: int) -> np.datetime64:
    return CASE_START + np.timedelta64(int(lead_hour), "h")


def format_time_label(valid_time: np.datetime64) -> str:
    return np.datetime_as_string(valid_time, unit="m").replace("T", "-").replace(":", "-")


def era5_realtime_file(lead_hour: int) -> Path:
    valid_label = format_time_label(valid_time_from_lead(lead_hour))
    return PROJECT_ROOT / "model_input" / "single_time_point" / "era5" / valid_label / "surface.nc"


def gdas_forecast_file(lead_hour: int) -> Path:
    valid_label = format_time_label(valid_time_from_lead(lead_hour))
    return (
        PROJECT_ROOT
        / "model_output"
        / "gdas"
        / GDAS_INIT_LABEL
        / str(lead_hour)
        / f"output_surface_{valid_label}.nc"
    )


def era5_lagged_forecast_file(lead_hour: int) -> Path:
    valid_label = format_time_label(valid_time_from_lead(lead_hour))
    model_step_hour = ERA5_LAGGED_OFFSET_HOURS + lead_hour
    return (
        PROJECT_ROOT
        / "model_output"
        / "era5"
        / ERA5_LAGGED_INIT_LABEL
        / str(model_step_hour)
        / f"output_surface_{valid_label}.nc"
    )


def load_wind_speed(nc_path: Path) -> xr.DataArray:
    if not nc_path.exists():
        raise FileNotFoundError(f"File not found: {nc_path}")
    with xr.open_dataset(nc_path, engine="netcdf4", decode_times=True) as raw:
        ds = raw.load()
    ds = rename_common_coords(ds)
    if "valid_time" in ds.dims:
        ds = ds.isel(valid_time=0, drop=True)
    ds = normalize_lon_lat(ds)
    u_name = find_var_name(
        ds, ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u10m"]
    )
    v_name = find_var_name(
        ds, ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "v10m"]
    )
    if u_name is None or v_name is None:
        raise KeyError(f"Unable to identify U10/V10 in {nc_path}: {list(ds.data_vars)}")
    u10 = crop_area(drop_extra_dims(ds[u_name]), AREA).transpose("latitude", "longitude")
    v10 = crop_area(drop_extra_dims(ds[v_name]), AREA).transpose("latitude", "longitude")
    speed = np.hypot(u10, v10)
    speed.name = "wind_speed"
    return speed


@lru_cache(maxsize=1)
def load_land_geodataframe():
    if gpd is None:
        return None
    if LAND_BOUNDARY_FILE is not None and Path(LAND_BOUNDARY_FILE).exists():
        try:
            return gpd.read_file(LAND_BOUNDARY_FILE)
        except Exception as exc:
            print(f"Warning: failed to read land boundary {LAND_BOUNDARY_FILE}: {exc}")
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception as exc:
        print(f"Warning: land boundary is unavailable: {exc}")
        return None


def add_land_background(ax) -> None:
    world = load_land_geodataframe()
    if world is None or box is None:
        return
    lat_max, lon_min, lat_min, lon_max = AREA
    try:
        world = world.set_crs(epsg=4326) if world.crs is None else world.to_crs(epsg=4326)
        bounds = box(lon_min, lat_min, lon_max, lat_max)
        clipped = world[world.intersects(bounds)].copy()
        if clipped.empty:
            return
        clipped.boundary.plot(ax=ax, color="#666666", linewidth=0.7, zorder=4)
    except Exception as exc:
        print(f"Warning: failed to draw land boundary: {exc}")


def style_axis(ax) -> None:
    lat_max, lon_min, lat_min, lon_max = AREA
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.grid(True, color="#BFBFBF", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
    ax.xaxis.set_major_locator(MultipleLocator(LON_TICK_INTERVAL))
    ax.yaxis.set_major_locator(MultipleLocator(LAT_TICK_INTERVAL))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"{value:g}E$^\circ$"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"{value:g}N$^\circ$"))
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)


def load_bias_fields(lead_hour: int) -> tuple[xr.DataArray, xr.DataArray]:
    reference = load_wind_speed(era5_realtime_file(lead_hour))
    gdas = load_wind_speed(gdas_forecast_file(lead_hour)).interp_like(reference, method="nearest")
    era5_lagged = load_wind_speed(era5_lagged_forecast_file(lead_hour)).interp_like(
        reference, method="nearest"
    )
    return gdas - reference, era5_lagged - reference


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": FONT_FAMILY,
            "font.sans-serif": ["SimHei", "SimSun", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": FONT_SIZE,
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def make_figure() -> None:
    configure_matplotlib()
    fields = {lead: load_bias_fields(lead) for lead in LEAD_HOURS}
    all_values = np.concatenate(
        [field.values.ravel() for pair in fields.values() for field in pair]
    )
    abs_limit = float(BIAS_LIMIT) if BIAS_LIMIT is not None else float(np.nanmax(np.abs(all_values)))
    if not np.isfinite(abs_limit) or abs_limit <= 0:
        raise RuntimeError("Unable to determine a valid symmetric wind-speed-bias color scale")
    norm = TwoSlopeNorm(vmin=-abs_limit, vcenter=0.0, vmax=abs_limit)

    fig, axes = plt.subplots(2, 6, figsize=FIGSIZE, constrained_layout=False)
    fig.subplots_adjust(
        left=0.065,
        right=0.92,
        bottom=0.10,
        top=0.91,
        wspace=PANEL_WSPACE,
        hspace=PANEL_HSPACE,
    )

    panel_letters = "abcdefghijkl"
    image = None
    for column, lead_hour in enumerate(LEAD_HOURS):
        for row, field in enumerate(fields[lead_hour]):
            ax = axes[row, column]
            image = ax.pcolormesh(
                field["longitude"].values,
                field["latitude"].values,
                field.values,
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
                zorder=2.5,
            )
            add_land_background(ax)
            style_axis(ax)
            ax.set_title(
                f"({panel_letters[row * len(LEAD_HOURS) + column]}) T+{lead_hour} h",
                fontsize=TITLE_SIZE,
                fontweight="bold",
                loc="left",
            )
            if column > 0:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.tick_params(labelbottom=False)

    fig.text(
        0.018,
        0.705,
        "GDAS_Realtime - ERA5 realtime",
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )
    fig.text(
        0.018,
        0.305,
        "ERA5_Lagged - ERA5 realtime",
        rotation=90,
        va="center",
        ha="center",
        fontweight="bold",
    )

    if image is None:
        raise RuntimeError("No bias field was plotted")
    colorbar_axis = fig.add_axes(COLORBAR_RECT)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Wind speed bias (m s$^{-1}$)")

    out_png = OUT_DIR / f"{OUT_STEM}.png"
    out_svg = OUT_DIR / f"{OUT_STEM}.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    print(f"Saved figure: {out_svg}")

    if SHOW_FIG:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
