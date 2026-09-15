# -*- coding: utf-8 -*-
"""Analyze Typhoon Wipha radial 10-m wind structure against ERA5 reference.

ERA5 is used only as a common reference field. It is not treated as
independent observational truth.
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
import pandas as pd
import xarray as xr
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "wipha_track_relative_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CASE_START = pd.Timestamp("2025-07-17 00:00:00")
KEY_LEADS = [24, 48, 72]
ALL_LEADS = list(range(0, 73, 3))
GDAS_INIT_LABEL = "2025-07-17-00-00"
ERA5_LAGGED_INIT_LABEL = "2025-07-12-00-00"
ERA5_LAGGED_OFFSET_HOURS = 120
WIPHA_SID = "2025197N14131"

IBTRACS_PATH = PROJECT_ROOT / "ibtracs.WP.list.v04r01.csv"
ERA5_MULTI_PATH = PROJECT_ROOT / "model_input" / "multi_time_point" / "wind10_july_august.nc"
LAND_SEA_MASK_PATH = PROJECT_ROOT / "src" / "lsm.nc"

RADIAL_BINS = [
    ("0-150 km", 0.0, 150.0),
    ("150-300 km", 150.0, 300.0),
    ("300-600 km", 300.0, 600.0),
    ("0-600 km", 0.0, 600.0),
]
PLOT_RADIUS_BINS = [item[0] for item in RADIAL_BINS[:3]]
MODEL_NAMES = ["GDAS_Realtime", "ERA5_Lagged"]
EARTH_RADIUS_KM = 6371.0088
OCEAN_LSM_THRESHOLD = 0.5
SUBSET_LAT_PAD_DEG = 6.0
SUBSET_LON_PAD_DEG = 7.0

COLORS = {
    "ERA5 reference": "#43A3EF",
    "GDAS_Realtime": "#EF767B",
    "ERA5_Lagged": "#FEA040",
}
DISPLAY_LABELS = {
    "ERA5 reference": "ERA5参考场",
    "GDAS_Realtime": "GDAS实时预报",
    "ERA5_Lagged": "ERA5延迟预报",
}
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


def configure_matplotlib() -> None:
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
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#BFBFBF", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)


def find_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in ds.coords or candidate in ds.dims or candidate in ds.data_vars:
            return candidate
    return None


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    lat_name = find_name(ds, ["latitude", "lat"])
    lon_name = find_name(ds, ["longitude", "lon"])
    time_name = find_name(ds, ["valid_time", "time"])
    if lat_name and lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name and lon_name != "longitude":
        rename[lon_name] = "longitude"
    if time_name and time_name != "valid_time":
        rename[time_name] = "valid_time"
    if rename:
        ds = ds.rename(rename)
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise RuntimeError("Dataset does not contain latitude/longitude coordinates")
    lon = ((ds["longitude"] + 360.0) % 360.0).astype(np.float32)
    return ds.assign_coords(longitude=lon).sortby("longitude").sortby(
        "latitude", ascending=False
    )


def find_wind_component(ds: xr.Dataset, component: str) -> str:
    candidates = {
        "u": ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u10m"],
        "v": ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "v10m"],
    }[component]
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    lowered = {candidate.lower() for candidate in candidates}
    for variable in ds.data_vars:
        attrs = ds[variable].attrs
        values = {
            str(attrs.get("shortName", "")).lower(),
            str(attrs.get("GRIB_shortName", "")).lower(),
            str(attrs.get("standard_name", "")).lower(),
            str(attrs.get("long_name", "")).lower(),
        }
        if values & lowered:
            return variable
    raise KeyError(f"Unable to identify {component}10 in variables: {list(ds.data_vars)}")


def drop_singleton_dims(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in {"latitude", "longitude"}:
            continue
        if da.sizes[dim] != 1:
            raise RuntimeError(f"Unsupported dimension {dim}={da.sizes[dim]} in {da.name}")
        da = da.isel({dim: 0}, drop=True)
    return da


def subset_bounds(center_lat: float, center_lon: float) -> tuple[float, float, float, float]:
    return (
        center_lat + SUBSET_LAT_PAD_DEG,
        center_lat - SUBSET_LAT_PAD_DEG,
        center_lon - SUBSET_LON_PAD_DEG,
        center_lon + SUBSET_LON_PAD_DEG,
    )


def subset_dataset(ds: xr.Dataset, bounds) -> xr.Dataset:
    lat_max, lat_min, lon_min, lon_max = bounds
    return ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))


def components_from_dataset(ds: xr.Dataset, bounds) -> tuple[xr.DataArray, xr.DataArray]:
    ds = subset_dataset(normalize_dataset(ds), bounds)
    u_name = find_wind_component(ds, "u")
    v_name = find_wind_component(ds, "v")
    u10 = drop_singleton_dims(ds[u_name]).transpose("latitude", "longitude").load()
    v10 = drop_singleton_dims(ds[v_name]).transpose("latitude", "longitude").load()
    return u10, v10


def load_components(path: Path, bounds) -> tuple[xr.DataArray, xr.DataArray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing wind field: {path}")
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as raw:
        ds = normalize_dataset(raw)
        if "valid_time" in ds.dims:
            ds = ds.isel(valid_time=0, drop=True)
        return components_from_dataset(ds, bounds)


def reference_components(
    reference_ds: xr.Dataset, valid_time: pd.Timestamp, bounds
) -> tuple[xr.DataArray, xr.DataArray]:
    selected = reference_ds.sel(valid_time=np.datetime64(valid_time), method="nearest")
    selected_time = pd.Timestamp(selected["valid_time"].values)
    if abs((selected_time - valid_time).total_seconds()) > 60:
        raise RuntimeError(f"ERA5 reference time mismatch: {valid_time} vs {selected_time}")
    return components_from_dataset(selected, bounds)


def gdas_path(lead_hour: int, valid_time: pd.Timestamp) -> Path:
    if lead_hour == 0:
        return (
            PROJECT_ROOT
            / "model_input"
            / "single_time_point"
            / "gdas"
            / GDAS_INIT_LABEL
            / "surface.nc"
        )
    valid_label = valid_time.strftime("%Y-%m-%d-%H-%M")
    return (
        PROJECT_ROOT
        / "model_output"
        / "gdas"
        / GDAS_INIT_LABEL
        / str(lead_hour)
        / f"output_surface_{valid_label}.nc"
    )


def era5_lagged_path(lead_hour: int, valid_time: pd.Timestamp) -> Path:
    valid_label = valid_time.strftime("%Y-%m-%d-%H-%M")
    model_step = ERA5_LAGGED_OFFSET_HOURS + lead_hour
    return (
        PROJECT_ROOT
        / "model_output"
        / "era5"
        / ERA5_LAGGED_INIT_LABEL
        / str(model_step)
        / f"output_surface_{valid_label}.nc"
    )


def load_track() -> pd.DataFrame:
    if not IBTRACS_PATH.exists():
        write_missing_center_report(f"IBTrACS file is missing: {IBTRACS_PATH}")
        raise FileNotFoundError(IBTRACS_PATH)
    columns = ["SID", "SEASON", "NAME", "ISO_TIME", "LAT", "LON"]
    track = pd.read_csv(IBTRACS_PATH, usecols=columns, low_memory=False)
    track = track[
        (track["SID"].astype(str) == WIPHA_SID)
        & (track["NAME"].astype(str).str.strip() == "WIPHA")
    ].copy()
    track["ISO_TIME"] = pd.to_datetime(track["ISO_TIME"], errors="coerce")
    track["LAT"] = pd.to_numeric(track["LAT"], errors="coerce")
    track["LON"] = pd.to_numeric(track["LON"], errors="coerce")
    track = track.dropna(subset=["ISO_TIME", "LAT", "LON"]).sort_values("ISO_TIME")
    if len(track) < 2:
        write_missing_center_report(
            f"No usable Wipha center track found for SID={WIPHA_SID} in {IBTRACS_PATH}"
        )
        raise RuntimeError("Missing Wipha center track")
    return track


def write_missing_center_report(reason: str) -> None:
    path = OUTPUT_DIR / "missing_tc_center_report.md"
    path.write_text(
        "# Missing typhoon center report\n\n"
        f"Radial analysis was stopped because {reason}.\n\n"
        "Required fields: valid time, typhoon center latitude, and typhoon center longitude.\n",
        encoding="utf-8",
    )


def center_at(track: pd.DataFrame, valid_time: pd.Timestamp) -> tuple[float, float]:
    times = track["ISO_TIME"].astype("int64").to_numpy(dtype=np.int64)
    target = np.int64(valid_time.value)
    if target < times.min() or target > times.max():
        write_missing_center_report(f"{valid_time} is outside the available track time range")
        raise RuntimeError(f"No typhoon center available at {valid_time}")
    lat = float(np.interp(target, times, track["LAT"].to_numpy(dtype=float)))
    lon = float(np.interp(target, times, track["LON"].to_numpy(dtype=float)))
    return lat, lon


def load_land_sea_mask() -> xr.DataArray | None:
    if not LAND_SEA_MASK_PATH.exists():
        return None
    with xr.open_dataset(LAND_SEA_MASK_PATH, engine="netcdf4", decode_times=True) as raw:
        ds = normalize_dataset(raw)
        variable = "lsm" if "lsm" in ds.data_vars else next(iter(ds.data_vars), None)
        if variable is None:
            return None
        da = ds[variable]
        if "valid_time" in da.dims:
            da = da.isel(valid_time=0, drop=True)
        return drop_singleton_dims(da).load()


def haversine_distance_grid(
    latitude: np.ndarray, longitude: np.ndarray, center_lat: float, center_lon: float
) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(latitude, longitude, indexing="ij")
    lat1 = np.deg2rad(center_lat)
    lat2 = np.deg2rad(lat2d)
    dlat = lat2 - lat1
    dlon = np.deg2rad(lon2d - center_lon)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    return np.hypot(u10, v10)


def meteorological_wind_direction(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    return (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0


def signed_circular_difference(model_dir: np.ndarray, ref_dir: np.ndarray) -> np.ndarray:
    return ((model_dir - ref_dir + 180.0) % 360.0) - 180.0


def radius_mask(distance: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if upper == 600.0:
        return (distance >= lower) & (distance <= upper)
    return (distance >= lower) & (distance < upper)


def summarize_model(
    lead_hour: int,
    valid_time: pd.Timestamp,
    radius_label: str,
    model_name: str,
    mask: np.ndarray,
    ref_speed: np.ndarray,
    ref_dir: np.ndarray,
    model_speed: np.ndarray,
    model_dir: np.ndarray,
) -> dict:
    valid = mask & np.isfinite(ref_speed) & np.isfinite(ref_dir)
    valid &= np.isfinite(model_speed) & np.isfinite(model_dir)
    n_grid = int(valid.sum())
    if n_grid == 0:
        values = {key: np.nan for key in [
            "era5_mean_wind_speed", "model_mean_wind_speed", "speed_bias",
            "speed_mae", "speed_rmse", "speed_median_error", "speed_max_abs_error",
            "dir_bias_deg", "dir_mae_deg", "dir_rmse_deg",
            "dir_median_abs_error_deg", "dir_max_abs_error_deg",
        ]}
    else:
        speed_error = model_speed[valid] - ref_speed[valid]
        direction_error = signed_circular_difference(model_dir[valid], ref_dir[valid])
        values = {
            "era5_mean_wind_speed": float(np.mean(ref_speed[valid])),
            "model_mean_wind_speed": float(np.mean(model_speed[valid])),
            "speed_bias": float(np.mean(speed_error)),
            "speed_mae": float(np.mean(np.abs(speed_error))),
            "speed_rmse": float(np.sqrt(np.mean(speed_error**2))),
            "speed_median_error": float(np.median(speed_error)),
            "speed_max_abs_error": float(np.max(np.abs(speed_error))),
            "dir_bias_deg": float(np.mean(direction_error)),
            "dir_mae_deg": float(np.mean(np.abs(direction_error))),
            "dir_rmse_deg": float(np.sqrt(np.mean(direction_error**2))),
            "dir_median_abs_error_deg": float(np.median(np.abs(direction_error))),
            "dir_max_abs_error_deg": float(np.max(np.abs(direction_error))),
        }
    return {
        "lead_hour": lead_hour,
        "valid_time": valid_time.strftime("%Y-%m-%d %H:%M UTC"),
        "radius_bin": radius_label,
        "model": model_name,
        "n_grid": n_grid,
        **values,
        "better_speed_rmse": "",
        "better_dir_mae": "",
    }


def winner(first_value: float, second_value: float, tie_threshold: float) -> str:
    if not np.isfinite(first_value) or not np.isfinite(second_value):
        return "Tie"
    if abs(first_value - second_value) < tie_threshold:
        return "Tie"
    return MODEL_NAMES[0] if first_value < second_value else MODEL_NAMES[1]


def assign_winners(statistics: pd.DataFrame) -> pd.DataFrame:
    for (_, _), group in statistics.groupby(["lead_hour", "radius_bin"], sort=False):
        by_model = group.set_index("model")
        speed_winner = winner(
            by_model.loc[MODEL_NAMES[0], "speed_rmse"],
            by_model.loc[MODEL_NAMES[1], "speed_rmse"],
            0.1,
        )
        direction_winner = winner(
            by_model.loc[MODEL_NAMES[0], "dir_mae_deg"],
            by_model.loc[MODEL_NAMES[1], "dir_mae_deg"],
            2.0,
        )
        statistics.loc[group.index, "better_speed_rmse"] = speed_winner
        statistics.loc[group.index, "better_dir_mae"] = direction_winner
    return statistics


def process_lead(
    lead_hour: int,
    track: pd.DataFrame,
    reference_ds: xr.Dataset,
    lsm: xr.DataArray | None,
    include_radial_bins: bool,
) -> list[dict]:
    valid_time = CASE_START + pd.Timedelta(hours=lead_hour)
    center_lat, center_lon = center_at(track, valid_time)
    bounds = subset_bounds(center_lat, center_lon)

    ref_u, ref_v = reference_components(reference_ds, valid_time, bounds)
    gdas_u, gdas_v = load_components(gdas_path(lead_hour, valid_time), bounds)
    lagged_u, lagged_v = load_components(era5_lagged_path(lead_hour, valid_time), bounds)
    gdas_u = gdas_u.interp_like(ref_u, method="linear")
    gdas_v = gdas_v.interp_like(ref_v, method="linear")
    lagged_u = lagged_u.interp_like(ref_u, method="linear")
    lagged_v = lagged_v.interp_like(ref_v, method="linear")

    ref_u_values = ref_u.values
    ref_v_values = ref_v.values
    gdas_u_values = gdas_u.values
    gdas_v_values = gdas_v.values
    lagged_u_values = lagged_u.values
    lagged_v_values = lagged_v.values

    distance = haversine_distance_grid(
        ref_u["latitude"].values,
        ref_u["longitude"].values,
        center_lat,
        center_lon,
    )
    common_valid = np.isfinite(ref_u_values) & np.isfinite(ref_v_values)
    common_valid &= np.isfinite(gdas_u_values) & np.isfinite(gdas_v_values)
    common_valid &= np.isfinite(lagged_u_values) & np.isfinite(lagged_v_values)
    if lsm is not None:
        local_lsm = lsm.interp_like(ref_u, method="nearest").values
        common_valid &= np.isfinite(local_lsm) & (local_lsm < OCEAN_LSM_THRESHOLD)

    ref_speed = wind_speed(ref_u_values, ref_v_values)
    ref_dir = meteorological_wind_direction(ref_u_values, ref_v_values)
    model_arrays = {
        "GDAS_Realtime": (
            wind_speed(gdas_u_values, gdas_v_values),
            meteorological_wind_direction(gdas_u_values, gdas_v_values),
        ),
        "ERA5_Lagged": (
            wind_speed(lagged_u_values, lagged_v_values),
            meteorological_wind_direction(lagged_u_values, lagged_v_values),
        ),
    }

    bins = RADIAL_BINS if include_radial_bins else [RADIAL_BINS[-1]]
    rows = []
    for radius_label, lower, upper in bins:
        mask = common_valid & radius_mask(distance, lower, upper)
        for model_name, (model_speed, model_dir) in model_arrays.items():
            rows.append(
                summarize_model(
                    lead_hour,
                    valid_time,
                    radius_label,
                    model_name,
                    mask,
                    ref_speed,
                    ref_dir,
                    model_speed,
                    model_dir,
                )
            )
    print(
        f"Processed T+{lead_hour:02d} h: center=({center_lat:.2f}, {center_lon:.2f}), "
        f"ocean grid n={int((common_valid & (distance <= 600.0)).sum())}"
    )
    return rows


def make_radial_figure(statistics: pd.DataFrame) -> tuple[Path, Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.86, hspace=0.34)
    x = np.arange(len(PLOT_RADIUS_BINS))
    panel_letters = "abcdef"

    speed_max = float(
        statistics[statistics["radius_bin"].isin(PLOT_RADIUS_BINS)][
            ["era5_mean_wind_speed", "model_mean_wind_speed"]
        ].max().max()
    )
    direction_max = float(
        statistics[statistics["radius_bin"].isin(PLOT_RADIUS_BINS)]["dir_mae_deg"].max()
    )

    for column, lead_hour in enumerate(KEY_LEADS):
        lead_data = statistics[
            (statistics["lead_hour"] == lead_hour)
            & statistics["radius_bin"].isin(PLOT_RADIUS_BINS)
        ]
        counts = []
        era5_means = []
        for radius_label in PLOT_RADIUS_BINS:
            group = lead_data[lead_data["radius_bin"] == radius_label]
            counts.append(int(group["n_grid"].iloc[0]) if not group.empty else 0)
            era5_means.append(float(group["era5_mean_wind_speed"].iloc[0]) if not group.empty else np.nan)
        tick_labels = [f"{label}\nn={count}" for label, count in zip(PLOT_RADIUS_BINS, counts)]

        top_ax = axes[0, column]
        speed_bar_width = 0.24
        speed_series = [("ERA5 reference", np.asarray(era5_means, dtype=float))]
        for model_name in ["GDAS_Realtime", "ERA5_Lagged"]:
            model_data = lead_data[lead_data["model"] == model_name].set_index("radius_bin")
            speed_series.append(
                (
                    model_name,
                    np.asarray(
                        [model_data.loc[label, "model_mean_wind_speed"] for label in PLOT_RADIUS_BINS],
                        dtype=float,
                    ),
                )
            )
        for offset, (series_name, values) in zip(
            [-speed_bar_width, 0.0, speed_bar_width], speed_series
        ):
            top_ax.bar(
                x + offset,
                values,
                width=speed_bar_width,
                color=COLORS[series_name],
                alpha=0.9,
                edgecolor="#555555",
                linewidth=0.6,
                label=DISPLAY_LABELS[series_name],
            )
        top_ax.set_title(
            f"({panel_letters[column]}) {lead_hour}h预报",
            loc="left",
            fontproperties=CHINESE_TITLE_FONT,
        )
        top_ax.set_xticks(x, tick_labels)
        if column == 0:
            top_ax.set_ylabel(
                "10米平均风速 (m s$^{-1}$)",
                fontproperties=CHINESE_FONT,
                fontsize=FONT_SIZES["axis_label"],
            )
        top_ax.set_ylim(0.0, speed_max * 1.24)
        style_axis(top_ax)
        if column == 0:
            top_ax.legend(
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#CFCFCF",
                framealpha=0.82,
                prop=CHINESE_LEGEND_FONT,
            )

        bottom_ax = axes[1, column]
        width = 0.34
        for offset, model_name in [(-width / 2, "GDAS_Realtime"), (width / 2, "ERA5_Lagged")]:
            model_data = lead_data[lead_data["model"] == model_name].set_index("radius_bin")
            values = np.asarray(
                [model_data.loc[label, "dir_mae_deg"] for label in PLOT_RADIUS_BINS], dtype=float
            )
            bottom_ax.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[model_name],
                alpha=0.88,
                edgecolor="#555555",
                linewidth=0.6,
                label=DISPLAY_LABELS[model_name],
            )
        bottom_ax.set_title(
            f"({panel_letters[column + 3]}) {lead_hour}h预报",
            loc="left",
            fontproperties=CHINESE_TITLE_FONT,
        )
        bottom_ax.set_xticks(x, tick_labels)
        if column == 0:
            bottom_ax.set_ylabel(
                #"相对于ERA5参考场的风向MAE（圆周角差，°）",
                "风向MAE（°）",
                fontproperties=CHINESE_FONT,
                fontsize=FONT_SIZES["axis_label"],
            )
        bottom_ax.set_ylim(0.0, direction_max * 1.24)
        style_axis(bottom_ax)
        if column == 0:
            bottom_ax.legend(
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#CFCFCF",
                framealpha=0.82,
                prop=CHINESE_LEGEND_FONT,
            )

    fig.tight_layout(rect=[0.02, 0.02, 0.99, 0.99])
    fig.subplots_adjust(wspace=0.20)
    png_path = OUTPUT_DIR / "figure_wipha_radial_wind_speed_direction_structure.png"
    svg_path = OUTPUT_DIR / "figure_wipha_radial_wind_speed_direction_structure.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def make_timeseries_figure(timeseries: pd.DataFrame) -> tuple[Path, Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.11, top=0.95, hspace=0.18)
    for model_name, marker in [("GDAS_Realtime", "^"), ("ERA5_Lagged", "s")]:
        data = timeseries[timeseries["model"] == model_name].sort_values("lead_hour")
        axes[0].plot(
            data["lead_hour"],
            data["speed_rmse"],
            color=COLORS[model_name],
            marker=marker,
            markersize=4,
            linewidth=1.4,
            label=DISPLAY_LABELS[model_name],
        )
        axes[1].plot(
            data["lead_hour"],
            data["dir_mae_deg"],
            color=COLORS[model_name],
            marker=marker,
            markersize=4,
            linewidth=1.4,
            label=DISPLAY_LABELS[model_name],
        )
    axes[0].set_title(
        "(a) 600 km范围内的风速误差",
        loc="left",
        fontproperties=CHINESE_TITLE_FONT,
    )
    axes[0].set_ylabel(
        "风速RMSE (m s$^{-1}$)",
        fontproperties=CHINESE_FONT,
        fontsize=FONT_SIZES["axis_label"],
    )
    axes[0].legend(
        loc="upper left", frameon=True, facecolor="white", edgecolor="#CFCFCF",
        framealpha=0.82, prop=CHINESE_LEGEND_FONT
    )
    axes[1].set_title(
        "(b) 600 km范围内的风向误差",
        loc="left",
        fontproperties=CHINESE_TITLE_FONT,
    )
    axes[1].set_ylabel(
        "风向MAE (°)",
        fontproperties=CHINESE_FONT,
        fontsize=FONT_SIZES["axis_label"],
    )
    axes[1].set_xlabel(
        "自2025-07-17 00 UTC起的预报时效 (h)",
        fontproperties=CHINESE_FONT,
        fontsize=FONT_SIZES["axis_label"],
    )
    axes[1].set_xticks(np.arange(0, 73, 6))
    for ax in axes:
        style_axis(ax)
    png_path = OUTPUT_DIR / "figure_wipha_wind_speed_direction_error_timeseries.png"
    svg_path = OUTPUT_DIR / "figure_wipha_wind_speed_direction_error_timeseries.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def make_radial_speed_mae_timeseries(statistics: pd.DataFrame) -> tuple[Path, Path]:
    configure_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 11.2), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.09, top=0.97, hspace=0.20)
    panel_letters = "abc"
    marker_styles = {"GDAS_Realtime": "^", "ERA5_Lagged": "s"}

    for index, (ax, radius_label) in enumerate(zip(axes, PLOT_RADIUS_BINS)):
        radius_data = statistics[statistics["radius_bin"] == radius_label]
        series_data = {}
        for model_name in ["GDAS_Realtime", "ERA5_Lagged"]:
            model_data = radius_data[radius_data["model"] == model_name].sort_values("lead_hour")
            series_data[model_name] = (model_data["lead_hour"], model_data["speed_mae"])

        for series_name, (lead_hours, values) in series_data.items():
            ax.plot(
                lead_hours,
                values,
                color=COLORS[series_name],
                marker=marker_styles[series_name],
                markersize=4.5,
                markeredgecolor="#555555",
                markeredgewidth=0.45,
                linewidth=1.5,
                label=DISPLAY_LABELS[series_name],
            )
        ax.set_title(
            f"({panel_letters[index]}) {radius_label}范围内10米风速MAE",
            loc="left",
            fontproperties=CHINESE_TITLE_FONT,
        )
        ax.set_ylabel(
            "风速MAE (m s$^{-1}$)",
            fontproperties=CHINESE_FONT,
            fontsize=FONT_SIZES["axis_label"],
        )
        ax.set_xticks(np.arange(0, 73, 6))
        style_axis(ax)
        if index == 0:
            ax.legend(
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#CFCFCF",
                framealpha=0.82,
                prop=CHINESE_LEGEND_FONT,
            )

    axes[-1].set_xlabel(
        "自2025-07-17 00 UTC起的预报时效 (h)",
        fontproperties=CHINESE_FONT,
        fontsize=FONT_SIZES["axis_label"],
    )
    png_path = OUTPUT_DIR / "figure_wipha_radial_wind_speed_mae_timeseries.png"
    svg_path = OUTPUT_DIR / "figure_wipha_radial_wind_speed_mae_timeseries.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def make_paper_summary(statistics: pd.DataFrame) -> pd.DataFrame:
    overall = statistics[statistics["radius_bin"] == "0-600 km"]
    rows = []
    for lead_hour in KEY_LEADS:
        group = overall[overall["lead_hour"] == lead_hour].set_index("model")
        rows.append(
            {
                "lead_hour": lead_hour,
                "GDAS_speed_bias": group.loc["GDAS_Realtime", "speed_bias"],
                "GDAS_speed_rmse": group.loc["GDAS_Realtime", "speed_rmse"],
                "GDAS_dir_mae_deg": group.loc["GDAS_Realtime", "dir_mae_deg"],
                "ERA5Lagged_speed_bias": group.loc["ERA5_Lagged", "speed_bias"],
                "ERA5Lagged_speed_rmse": group.loc["ERA5_Lagged", "speed_rmse"],
                "ERA5Lagged_dir_mae_deg": group.loc["ERA5_Lagged", "dir_mae_deg"],
                "better_speed": group.loc["GDAS_Realtime", "better_speed_rmse"],
                "better_direction": group.loc["GDAS_Realtime", "better_dir_mae"],
            }
        )
    return pd.DataFrame(rows)


def write_notes(
    statistics: pd.DataFrame,
    generated_files: list[Path],
    used_mask: bool,
    timeseries_generated: bool,
) -> Path:
    lines = [
        "# Wipha radial wind speed and direction analysis notes",
        "",
        "## Data and experiment definitions",
        "",
        f"- ERA5 reference: `{ERA5_MULTI_PATH}` at each valid time.",
        f"- GDAS_Realtime: Pangu forecast initialized from GDAS at {CASE_START:%Y-%m-%d %H:%M} UTC.",
        f"- ERA5_Lagged: Pangu forecast initialized from ERA5 at 2025-07-12 00:00 UTC, with a 120 h availability lag.",
        f"- Typhoon center: `{IBTRACS_PATH}`, WIPHA SID `{WIPHA_SID}`, using IBTrACS `LAT` and `LON` with linear time interpolation when needed.",
        f"- Land-sea mask: `{LAND_SEA_MASK_PATH}`; used = `{used_mask}`. Ocean is defined as lsm < {OCEAN_LSM_THRESHOLD}.",
        "",
        "ERA5 reference is used as a common reference field, not as independent observational truth.",
        "",
        "## Key valid times and model steps",
        "",
        "| lead_hour | valid_time (UTC) | GDAS model step | ERA5_Lagged model step |",
        "|---:|---|---:|---:|",
    ]
    for lead in KEY_LEADS:
        valid_time = CASE_START + pd.Timedelta(hours=lead)
        lines.append(
            f"| {lead} | {valid_time:%Y-%m-%d %H:%M} | {lead} | {ERA5_LAGGED_OFFSET_HOURS + lead} |"
        )
    lines.extend(
        [
            "",
            "## Methods",
            "",
            "- All forecast u10/v10 components are linearly interpolated to the ERA5 reference grid before wind diagnostics are calculated.",
            "- Analysis uses common valid ocean grid points within 600 km of the IBTrACS center.",
            "- Radius bins: 0-150 km, 150-300 km, 300-600 km, and overall 0-600 km.",
            "- Wind speed: `sqrt(u10^2 + v10^2)`.",
            "- Meteorological wind direction: `(270 - atan2(v10, u10) * 180 / pi) % 360`.",
            "- Signed direction error: `((model_dir - ERA5_dir + 180) % 360) - 180`.",
            "- Direction MAE and RMSE are calculated from circular angular differences, not ordinary angle subtraction.",
            "- Direction bias is the arithmetic mean of signed circular errors in [-180, 180] degrees.",
            "- Winner ties: speed RMSE difference < 0.1 m/s; direction MAE difference < 2 degrees.",
            "",
            "## Sample sizes",
            "",
        ]
    )
    for lead in KEY_LEADS:
        group = statistics[
            (statistics["lead_hour"] == lead) & (statistics["model"] == "GDAS_Realtime")
        ]
        summary = ", ".join(
            f"{row.radius_bin}: n={int(row.n_grid)}" for row in group.itertuples()
        )
        lines.append(f"- T+{lead} h: {summary}.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    lines.extend(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in generated_files)
    lines.extend(
        [
            "",
            f"The optional 0-72 h time-series figure was generated: `{timeseries_generated}`.",
            "",
            "## Interpretation limits",
            "",
            "- Results quantify agreement with the ERA5 reference field and are not an independent observational validation.",
            "- Grid-point statistics are spatially correlated and should not be interpreted as independent samples.",
            "- Near-coastal results depend on the 0.25-degree land-sea mask and interpolation behavior.",
        ]
    )
    notes_path = OUTPUT_DIR / "wipha_radial_wind_speed_direction_notes.md"
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return notes_path


def main() -> None:
    track = load_track()
    lsm = load_land_sea_mask()
    if not ERA5_MULTI_PATH.exists():
        raise FileNotFoundError(f"Missing ERA5 multi-time reference file: {ERA5_MULTI_PATH}")

    all_rows = []
    with xr.open_dataset(ERA5_MULTI_PATH, engine="netcdf4", decode_times=True) as raw_reference:
        reference_ds = normalize_dataset(raw_reference)
        for lead_hour in ALL_LEADS:
            all_rows.extend(
                process_lead(
                    lead_hour,
                    track,
                    reference_ds,
                    lsm,
                    include_radial_bins=True,
                )
            )

    all_statistics = pd.DataFrame(all_rows)
    key_statistics = all_statistics[all_statistics["lead_hour"].isin(KEY_LEADS)].copy()
    key_statistics = assign_winners(key_statistics)
    column_order = [
        "lead_hour", "valid_time", "radius_bin", "model", "n_grid",
        "era5_mean_wind_speed", "model_mean_wind_speed", "speed_bias", "speed_mae",
        "speed_rmse", "speed_median_error", "speed_max_abs_error", "dir_bias_deg",
        "dir_mae_deg", "dir_rmse_deg", "dir_median_abs_error_deg",
        "dir_max_abs_error_deg", "better_speed_rmse", "better_dir_mae",
    ]
    key_statistics = key_statistics[column_order].sort_values(
        ["lead_hour", "radius_bin", "model"],
        key=lambda series: series.map({label: index for index, label in enumerate(
            [item[0] for item in RADIAL_BINS] + MODEL_NAMES
        )}) if series.name in {"radius_bin", "model"} else series,
    )
    statistics_csv = OUTPUT_DIR / "table_wipha_radial_wind_speed_direction_statistics.csv"
    key_statistics.to_csv(statistics_csv, index=False, float_format="%.6f")

    paper_summary = make_paper_summary(key_statistics)
    paper_csv = OUTPUT_DIR / "table_wipha_key_lead_error_summary_for_paper.csv"
    paper_summary.to_csv(paper_csv, index=False, float_format="%.6f")

    timeseries = all_statistics[all_statistics["radius_bin"] == "0-600 km"].copy()
    timeseries_csv = OUTPUT_DIR / "supplementary_wipha_0_72h_error_statistics.csv"
    timeseries.to_csv(timeseries_csv, index=False, float_format="%.6f")

    radial_png, radial_svg = make_radial_figure(key_statistics)
    timeseries_png, timeseries_svg = make_timeseries_figure(timeseries)
    radial_timeseries_png, radial_timeseries_svg = make_radial_speed_mae_timeseries(
        all_statistics
    )
    generated = [
        statistics_csv,
        paper_csv,
        timeseries_csv,
        radial_png,
        radial_svg,
        timeseries_png,
        timeseries_svg,
        radial_timeseries_png,
        radial_timeseries_svg,
    ]
    notes_path = write_notes(key_statistics, generated, lsm is not None, True)
    generated.append(notes_path)

    print("Generated files:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
