# -*- coding: utf-8 -*-
"""
visualize_era5_surface_china_sea_msl_u10_t2m.py

功能：
读取 ERA5 surface.nc 文件，并可视化中国海区域的：
1. MSL
2. U10
3. T2M
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# 1. 用户配置区
# ============================================================

ERA5_SURFACE_FILE = Path(
    PROJECT_ROOT
    / "model_input"
    / "single_time_point"
    / "era5"
    / "2025-07-17-00-00"
    / "surface.nc"
)

# 中国海区域：lat_max, lon_min, lat_min, lon_max
AREA = [42, 103, 13, 130]

OUT_DIR = PROJECT_ROOT / "src" / "comparison_results" / "era5_surface_visual"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 基础工具函数
# ============================================================

def find_coord_or_dim(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
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
        ds = ds.isel(valid_time=0, drop=True)
    elif "time" in ds.dims:
        ds = ds.isel(time=0, drop=True)
    return ds


def normalize_lon_lat(ds: xr.Dataset | xr.DataArray):
    if "longitude" not in ds.coords:
        raise RuntimeError("数据中没有 longitude 坐标")
    if "latitude" not in ds.coords:
        raise RuntimeError("数据中没有 latitude 坐标")

    lon = ((ds["longitude"] + 360) % 360).astype(np.float32)
    ds = ds.assign_coords(longitude=lon)
    ds = ds.sortby("longitude")
    ds = ds.sortby("latitude", ascending=False)

    return ds


def crop_area(ds: xr.Dataset | xr.DataArray, area):
    """
    area = [lat_max, lon_min, lat_min, lon_max]
    """
    lat_max, lon_min, lat_min, lon_max = area

    lon_min = (lon_min + 360) % 360
    lon_max = (lon_max + 360) % 360

    ds = normalize_lon_lat(ds)
    ds = ds.sel(latitude=slice(lat_max, lat_min))

    if lon_min <= lon_max:
        ds = ds.sel(longitude=slice(lon_min, lon_max))
    else:
        ds1 = ds.sel(longitude=slice(lon_min, 359.999))
        ds2 = ds.sel(longitude=slice(0, lon_max))
        ds = xr.concat([ds1, ds2], dim="longitude")

    return ds


def find_var_name(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.data_vars:
            return c

    candidates_lower = [str(c).lower() for c in candidates]

    for var in ds.data_vars:
        attrs = ds[var].attrs
        attr_values = [
            str(attrs.get("shortName", "")).lower(),
            str(attrs.get("GRIB_shortName", "")).lower(),
            str(attrs.get("standard_name", "")).lower(),
            str(attrs.get("long_name", "")).lower(),
        ]

        for v in attr_values:
            if v in candidates_lower:
                return var

    return None


def drop_extra_dims_da(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim in ["latitude", "longitude"]:
            continue

        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
        else:
            raise RuntimeError(
                f"变量 {da.name} 存在无法自动处理的额外维度: {dim}, size={da.sizes[dim]}"
            )

    return da


# ============================================================
# 3. 读取 ERA5 surface 数据
# ============================================================

def load_era5_surface(path: Path, area):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    with xr.open_dataset(path, decode_times=True) as raw:
        ds = raw.load()

    ds = rename_common_coords(ds)
    ds = select_first_time(ds)
    ds = normalize_lon_lat(ds)

    print("=" * 100)
    print("ERA5 surface 文件读取成功")
    print(f"文件路径: {path}")
    print(f"变量列表: {list(ds.data_vars)}")
    print(f"原始维度: {dict(ds.sizes)}")
    print("=" * 100)

    msl_name = find_var_name(
        ds,
        ["msl", "mean_sea_level_pressure", "MSL", "prmsl"]
    )

    u10_name = find_var_name(
        ds,
        ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u10m"]
    )

    t2m_name = find_var_name(
        ds,
        ["t2m", "2m_temperature", "temperature_2m"]
    )

    if msl_name is None:
        raise KeyError(f"无法识别 MSL 变量。当前变量: {list(ds.data_vars)}")

    if u10_name is None:
        raise KeyError(f"无法识别 U10 变量。当前变量: {list(ds.data_vars)}")

    if t2m_name is None:
        raise KeyError(f"无法识别 T2M 变量。当前变量: {list(ds.data_vars)}")

    msl = drop_extra_dims_da(ds[msl_name])
    u10 = drop_extra_dims_da(ds[u10_name])
    t2m = drop_extra_dims_da(ds[t2m_name])

    msl_crop = crop_area(msl, area).transpose("latitude", "longitude")
    u10_crop = crop_area(u10, area).transpose("latitude", "longitude")
    t2m_crop = crop_area(t2m, area).transpose("latitude", "longitude")

    # MSL: Pa -> hPa
    if float(msl_crop.mean()) > 2000:
        msl_plot = msl_crop / 100.0
    else:
        msl_plot = msl_crop
    msl_plot.attrs["units"] = "hPa"

    # T2M: K -> °C
    if float(t2m_crop.mean()) > 100:
        t2m_plot = t2m_crop - 273.15
    else:
        t2m_plot = t2m_crop
    t2m_plot.attrs["units"] = "°C"

    print("\n裁剪后区域：")
    print(f"AREA = {area}")
    print(f"MSL min/max: {float(msl_plot.min()):.2f} / {float(msl_plot.max()):.2f} hPa")
    print(f"U10 min/max: {float(u10_crop.min()):.2f} / {float(u10_crop.max()):.2f} m/s")
    print(f"T2M min/max: {float(t2m_plot.min()):.2f} / {float(t2m_plot.max()):.2f} °C")

    return msl_plot, u10_crop, t2m_plot


# ============================================================
# 4. 绘图函数
# ============================================================

def style_axis(ax):
    ax.set_facecolor("#EAEAF2")
    ax.grid(True, color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")


def plot_da(ax, da: xr.DataArray, title: str, cbar_label: str, cmap: str):
    lon = da["longitude"].values
    lat = da["latitude"].values

    im = ax.pcolormesh(
        lon,
        lat,
        da.values,
        shading="auto",
        cmap=cmap
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    style_axis(ax)
    ax.set_aspect("equal", adjustable="box")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)

    return im


def make_surface_figure(msl_hpa: xr.DataArray, u10: xr.DataArray, t2m_c: xr.DataArray):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5.5),
        constrained_layout=True
    )

    plot_da(
        axes[0],
        msl_hpa,
        title="ERA5 MSL over China Seas",
        cbar_label="MSL (hPa)",
        cmap="viridis"
    )

    plot_da(
        axes[1],
        u10,
        title="ERA5 U10 over China Seas",
        cbar_label="U10 (m s$^{-1}$)",
        cmap="RdBu_r"
    )

    plot_da(
        axes[2],
        t2m_c,
        title="ERA5 T2M over China Seas",
        cbar_label="T2M (°C)",
        cmap="coolwarm"
    )

    fig.suptitle(
        "ERA5 surface fields at 2025-07-17 00:00 UTC",
        fontsize=15,
        fontweight="bold"
    )

    png_path = OUT_DIR / "era5_surface_msl_u10_t2m_2025-07-17-00-00.png"
    svg_path = OUT_DIR / "era5_surface_msl_u10_t2m_2025-07-17-00-00.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print("\n图像已保存：")
    print(png_path)
    print(svg_path)


# ============================================================
# 5. 主程序
# ============================================================

def main():
    msl_hpa, u10, t2m_c = load_era5_surface(
        path=ERA5_SURFACE_FILE,
        area=AREA
    )

    make_surface_figure(msl_hpa, u10, t2m_c)


if __name__ == "__main__":
    main()
