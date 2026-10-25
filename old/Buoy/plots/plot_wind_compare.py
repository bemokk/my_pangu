# -*- coding: utf-8 -*-
"""Plot 24, 48, and 72 h spatial wind structure comparisons.

Rows are forecast lead times. Columns are:
1. ERA5 reference wind speed with 10 m wind vectors
2. GDAS_Realtime minus ERA5 wind speed bias
3. ERA5_Lagged minus ERA5 wind speed bias
"""

from __future__ import annotations

import os
import sys
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
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

os.environ["SHAPE_RESTORE_SHX"] = "YES"

try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception:
    gpd = None
    box = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AREA = [30, 105, 10, 130]  # [lat_max, lon_min, lat_min, lon_max]
LEAD_HOURS = [24, 48, 72]
CASE_START = np.datetime64("2025-07-17T00:00")
GDAS_INIT_LABEL = "2025-07-17-00-00"
ERA5_LAGGED_INIT_LABEL = "2025-07-12-00-00"
ERA5_LAGGED_OFFSET_HOURS = 120

VECTOR_STRIDE = 6
REFERENCE_VECTOR_COLOR = "white"
REFERENCE_VECTOR_SCALE = 9
REFERENCE_VECTOR_WIDTH = 0.0045
SHOW_FIG = False
FONT_SCALE = 1
FONT_FAMILY = ["Times New Roman", "SimSun", "SimHei", "Microsoft YaHei", "DejaVu Serif"]
BASE_FONT_SIZES = {
    "default": 14,
    "title": 15,
    "axis_label": 13,
    "legend": 12,
    "tick": 12,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}


def select_chinese_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ["SimSun", "Microsoft YaHei", "SimHei", "Microsoft JhengHei"]:
        if family in available:
            return family
    return "DejaVu Sans"


CHINESE_FONT = font_manager.FontProperties(family=select_chinese_font())
CHINESE_TITLE_FONT = font_manager.FontProperties(
    family=FONT_FAMILY, size=FONT_SIZES["title"], weight="bold"
)
CHINESE_LEGEND_FONT = font_manager.FontProperties(
    family=CHINESE_FONT.get_family(), size=FONT_SIZES["legend"]
)

# Shared horizontal colorbars: [left, bottom, width, height] in figure coordinates.
# Their centers sit between columns 1/2 and columns 2/3, respectively.
REFERENCE_COLORBAR_RECT = [0.225, 0.045, 0.22, 0.018]
BIAS_COLORBAR_RECT = [0.555, 0.045, 0.22, 0.018]

OUT_DIR = PROJECT_ROOT / "Buoy" / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAND_BOUNDARY_FILE = None


def find_coord_or_dim(ds: xr.Dataset, candidates):
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

    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds


def select_first_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        return ds.isel(valid_time=0, drop=True)
    if "time" in ds.dims:
        return ds.isel(time=0, drop=True)
    return ds


def normalize_lon_lat(ds):
    if "longitude" not in ds.coords:
        raise RuntimeError("Dataset does not contain longitude coordinate")
    if "latitude" not in ds.coords:
        raise RuntimeError("Dataset does not contain latitude coordinate")

    lon = ((ds["longitude"] + 360) % 360).astype(np.float32)
    ds = ds.assign_coords(longitude=lon)
    ds = ds.sortby("longitude")
    ds = ds.sortby("latitude", ascending=False)
    return ds


def crop_area(ds, area):
    lat_max, lon_min, lat_min, lon_max = area
    lon_min = (lon_min + 360) % 360
    lon_max = (lon_max + 360) % 360

    ds = normalize_lon_lat(ds)
    ds = ds.sel(latitude=slice(lat_max, lat_min))

    if lon_min <= lon_max:
        return ds.sel(longitude=slice(lon_min, lon_max))

    left = ds.sel(longitude=slice(lon_min, 359.999))
    right = ds.sel(longitude=slice(0, lon_max))
    return xr.concat([left, right], dim="longitude")


def drop_extra_dims_da(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in ["latitude", "longitude"]:
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        else:
            raise RuntimeError(f"Variable {da.name} has unsupported extra dimension {dim}")
    return da


def find_var_name(ds: xr.Dataset, candidates):
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate

    candidates_lower = [str(candidate).lower() for candidate in candidates]
    for var in ds.data_vars:
        attrs = ds[var].attrs
        attr_values = [
            str(attrs.get("shortName", "")).lower(),
            str(attrs.get("GRIB_shortName", "")).lower(),
            str(attrs.get("standard_name", "")).lower(),
            str(attrs.get("long_name", "")).lower(),
        ]
        if any(value in candidates_lower for value in attr_values):
            return var
    return None


def valid_time_from_lead(lead_hour: int) -> np.datetime64:
    return CASE_START + np.timedelta64(int(lead_hour), "h")


def format_time_label(valid_time: np.datetime64) -> str:
    timestamp = np.datetime_as_string(valid_time, unit="m")
    return timestamp.replace("T", "-").replace(":", "-")


def era5_reference_file(lead_hour: int) -> Path:
    valid_label = format_time_label(valid_time_from_lead(lead_hour))
    return PROJECT_ROOT / "model_input" / "single_time_point" / "era5" / valid_label / "surface.nc"


def gdas_forecast_file(lead_hour: int) -> Path:
    valid_label = format_time_label(valid_time_from_lead(lead_hour))
    return (
        PROJECT_ROOT
        / "model_output"
        / "gdas"
        / GDAS_INIT_LABEL
        / str(int(lead_hour))
        / f"output_surface_{valid_label}.nc"
    )


def era5_lagged_forecast_file(lead_hour: int) -> Path:
    valid_label = format_time_label(valid_time_from_lead(lead_hour))
    model_step_hour = ERA5_LAGGED_OFFSET_HOURS + int(lead_hour)
    return (
        PROJECT_ROOT
        / "model_output"
        / "era5"
        / ERA5_LAGGED_INIT_LABEL
        / str(model_step_hour)
        / f"output_surface_{valid_label}.nc"
    )


def load_surface_wind10_components(nc_path: Path, area):
    if not nc_path.exists():
        raise FileNotFoundError(f"File not found: {nc_path}")

    with xr.open_dataset(nc_path, engine="netcdf4", decode_times=True) as raw:
        ds = raw.load()

    ds = rename_common_coords(ds)
    ds = select_first_time(ds)
    ds = normalize_lon_lat(ds)

    u10_name = find_var_name(
        ds,
        ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u10m"],
    )
    v10_name = find_var_name(
        ds,
        ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "v10m"],
    )
    if u10_name is None or v10_name is None:
        raise KeyError(f"Unable to identify U10/V10 variables in {nc_path}: {list(ds.data_vars)}")

    u10 = drop_extra_dims_da(ds[u10_name])
    v10 = drop_extra_dims_da(ds[v10_name])
    u10 = crop_area(u10, area).transpose("latitude", "longitude")
    v10 = crop_area(v10, area).transpose("latitude", "longitude")

    wind10 = np.sqrt(u10 ** 2 + v10 ** 2)
    wind10.name = "wind10"
    wind10.attrs["units"] = "m s^-1"
    return wind10, u10, v10


def load_land_geodataframe():
    if gpd is None:
        return None

    if LAND_BOUNDARY_FILE is not None:
        path = Path(LAND_BOUNDARY_FILE)
        if path.exists():
            try:
                return gpd.read_file(path)
            except Exception as exc:
                print(f"Warning: failed to read LAND_BOUNDARY_FILE {path}: {exc}")

    try:
        world_path = gpd.datasets.get_path("naturalearth_lowres")
        return gpd.read_file(world_path)
    except Exception as exc:
        print(f"Warning: failed to load geopandas built-in land boundary: {exc}")
        return None


def add_land_background(ax, extent):
    world = load_land_geodataframe()
    if world is None or box is None:
        return

    lon_min, lon_max, lat_min, lat_max = extent
    try:
        if world.crs is None:
            world = world.set_crs(epsg=4326)
        else:
            world = world.to_crs(epsg=4326)

        bbox = box(lon_min, lat_min, lon_max, lat_max)
        try:
            world_clip = gpd.clip(world, bbox)
        except Exception:
            world_clip = world[world.intersects(bbox)].copy()
        if world_clip.empty:
            return

        world_clip.plot(ax=ax, facecolor="#F2E8D5", edgecolor="gray", linewidth=0.7, zorder=0)
        world_clip.boundary.plot(ax=ax, color="gray", linewidth=0.8, zorder=1)
    except Exception as exc:
        print(f"Warning: failed to draw land boundary: {exc}")


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#BFBFBF", linestyle="--", linewidth=0.8, alpha=0.7, zorder=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}E°"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}N°"))


def plot_field(
    ax,
    da,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    norm=None,
    vector_u=None,
    vector_v=None,
    vector_color=REFERENCE_VECTOR_COLOR,
    vector_scale=32,
    vector_width=0.002,
    vector_alpha=0.75,
):
    lon = da["longitude"].values
    lat = da["latitude"].values
    im = ax.pcolormesh(
        lon,
        lat,
        da.values,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        zorder=2.5,
    )

    lat_max, lon_min, lat_min, lon_max = AREA
    add_land_background(ax, [lon_min, lon_max, lat_min, lat_max])
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(
        title,
        loc="left",
        fontproperties=CHINESE_TITLE_FONT,
    )
    style_axis(ax)
    ax.set_aspect("equal", adjustable="box")

    if vector_u is not None and vector_v is not None:
        stride = max(1, int(VECTOR_STRIDE))
        quiver = ax.quiver(
            vector_u["longitude"].values[::stride],
            vector_u["latitude"].values[::stride],
            vector_u.values[::stride, ::stride],
            vector_v.values[::stride, ::stride],
            color=vector_color,
            angles="xy",
            scale_units="xy",
            scale=vector_scale,
            width=vector_width,
            alpha=vector_alpha,
            zorder=4,
        )
        quiver.set_path_effects([pe.withStroke(linewidth=0.9, foreground="#222222")])

    return im


def load_lead_fields(lead_hour: int) -> dict[str, xr.DataArray]:
    wind_ref, ref_u, ref_v = load_surface_wind10_components(era5_reference_file(lead_hour), AREA)
    wind_gdas, _, _ = load_surface_wind10_components(gdas_forecast_file(lead_hour), AREA)
    wind_era5lag, _, _ = load_surface_wind10_components(
        era5_lagged_forecast_file(lead_hour),
        AREA,
    )

    wind_gdas = wind_gdas.interp_like(wind_ref, method="nearest")
    wind_era5lag = wind_era5lag.interp_like(wind_ref, method="nearest")
    ref_u = ref_u.interp_like(wind_ref, method="nearest")
    ref_v = ref_v.interp_like(wind_ref, method="nearest")

    return {
        "wind_ref": wind_ref,
        "ref_u": ref_u,
        "ref_v": ref_v,
        "err_gdas": wind_gdas - wind_ref,
        "err_era5lag": wind_era5lag - wind_ref,
    }


def make_figure():
    fields_by_lead = {lead_hour: load_lead_fields(lead_hour) for lead_hour in LEAD_HOURS}

    wind_values = np.concatenate(
        [fields["wind_ref"].values.ravel() for fields in fields_by_lead.values()]
    )
    err_values = np.concatenate(
        [fields["err_gdas"].values.ravel() for fields in fields_by_lead.values()]
        + [fields["err_era5lag"].values.ravel() for fields in fields_by_lead.values()]
    )
    wind_vmin = 0.0
    wind_vmax = float(np.ceil(np.nanmax(wind_values) / 5.0) * 5.0)
    err_absmax = float(np.nanmax(np.abs(err_values)))
    err_norm = TwoSlopeNorm(vmin=-err_absmax, vcenter=0.0, vmax=err_absmax)

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": FONT_FAMILY,
            "font.sans-serif": ["SimHei", "SimSun", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": FONT_SIZES["default"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 13.5), constrained_layout=False)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.97, wspace=0.08, hspace=0.25)
    vector_legend_handle = Line2D(
        [0],
        [0],
        color=REFERENCE_VECTOR_COLOR,
        marker=r"$\rightarrow$",
        linestyle="None",
        markersize=13,
        markeredgecolor="#222222",
        markeredgewidth=0.8,
        label="ERA5 10米风矢量",
    )
    panel_letters = iter("abcdefghi")
    column_titles = [
        "ERA5参考场10米风速与风矢量",
        "GDAS实时预报 - ERA5参考场",
        "ERA5延迟预报 - ERA5参考场",
    ]

    reference_image = None
    bias_image = None
    for row_index, lead_hour in enumerate(LEAD_HOURS):
        fields = fields_by_lead[lead_hour]
        titles = [
            f"({next(panel_letters)}) {lead_hour}h预报  {column_titles[0]}",
            f"({next(panel_letters)}) {lead_hour}h预报  {column_titles[1]}",
            f"({next(panel_letters)}) {lead_hour}h预报  {column_titles[2]}",
        ]

        reference_image = plot_field(
            axes[row_index, 0],
            fields["wind_ref"],
            title=titles[0],
            cmap="viridis",
            vmin=wind_vmin,
            vmax=wind_vmax,
            vector_u=fields["ref_u"],
            vector_v=fields["ref_v"],
            vector_color=REFERENCE_VECTOR_COLOR,
            vector_scale=REFERENCE_VECTOR_SCALE,
            vector_width=REFERENCE_VECTOR_WIDTH,
        )
        if row_index == 0:
            axes[row_index, 0].legend(
                handles=[vector_legend_handle],
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#CFCFCF",
                framealpha=0.82,
                borderaxespad=0.2,
                prop=CHINESE_LEGEND_FONT,
            )
        bias_image = plot_field(
            axes[row_index, 1],
            fields["err_gdas"],
            title=titles[1],
            cmap="RdBu_r",
            norm=err_norm,
        )
        bias_image = plot_field(
            axes[row_index, 2],
            fields["err_era5lag"],
            title=titles[2],
            cmap="RdBu_r",
            norm=err_norm,
        )

    if reference_image is None or bias_image is None:
        raise RuntimeError("No wind field was plotted")

    reference_cax = fig.add_axes(REFERENCE_COLORBAR_RECT)
    reference_cbar = fig.colorbar(reference_image, cax=reference_cax, orientation="horizontal")
    reference_cbar.set_ticks(np.arange(0.0, wind_vmax + 0.1, 5.0))
    reference_cbar.set_label(
        "10米风速 (m s$^{-1}$)",
        fontproperties=CHINESE_FONT,
        fontsize=FONT_SIZES["axis_label"],
    )

    bias_cax = fig.add_axes(BIAS_COLORBAR_RECT)
    bias_cbar = fig.colorbar(bias_image, cax=bias_cax, orientation="horizontal")
    bias_cbar.set_label(
        "10米风速偏差 (m s$^{-1}$)",
        fontproperties=CHINESE_FONT,
        fontsize=FONT_SIZES["axis_label"],
    )

    out_png = OUT_DIR / "Figure_T24_T48_T72_Wind10_structure_compare_3x3.png"
    out_svg = OUT_DIR / "Figure_T24_T48_T72_Wind10_structure_compare_3x3.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")

    print(f"Saved figure: {out_png}")
    print(f"Saved figure: {out_svg}")

    if SHOW_FIG:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
