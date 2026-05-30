# -*- coding: utf-8 -*-
"""
plot_figure3_4_wind10_error.py

功能：
绘制图3-4 T+24 h Wind10 误差场，包括：
(a) ERA5_reference Wind10
(b) GDAS_Realtime - ERA5_reference
(c) ERA5_Lagged - ERA5_reference

支持两种布局：
1. LAYOUT = "1x3"
2. LAYOUT = "3x1"
"""

from pathlib import Path
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 若需要从不完整shp恢复 .shx，可保留这句
os.environ["SHAPE_RESTORE_SHX"] = "YES"

try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception:
    gpd = None
    box = None


# ============================================================
# 1. 用户配置区
# ============================================================

# -----------------------------
# 图形布局，可选 "1x3" 或 "3x1"
# -----------------------------
LAYOUT = "3x1"

# -----------------------------
# 分析区域（中国海区域）
# 格式：[lat_max, lon_min, lat_min, lon_max]
# -----------------------------
AREA = [30, 105, 10, 130]

# -----------------------------
# T+24 h 对应的三个文件
# -----------------------------
ERA5_FILE = Path(
    r"/model_input/single_time_point/era5/2025-07-18-00-00/surface.nc"
)

GDAS_FILE = Path(
    r"/model_output/gdas/2025-07-17-00-00/24/output_surface_2025-07-18-00-00.nc"
)

ERA5_LAGGED_FILE = Path(
    r"/model_output/era5/2025-07-12-00-00/144/output_surface_2025-07-18-00-00.nc"
)

# -----------------------------
# 输出目录
# -----------------------------
OUT_DIR = Path(r"/src/comparison_results/chapter3_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 陆地边界 shp 文件（可选）
# 如果没有，可设为 None
# -----------------------------
LAND_BOUNDARY_FILE = None
# 例如：
# LAND_BOUNDARY_FILE = r"E:\PyCharm_WorkSpace\pangu\src\naturalearth\ne_10m_land.shp"

# 是否显示图
SHOW_FIG = True


# ============================================================
# 2. 通用工具函数
# ============================================================

def find_coord_or_dim(ds: xr.Dataset, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims or c in ds.variables:
            return c
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


def normalize_lon_lat(ds):
    if "longitude" not in ds.coords:
        raise RuntimeError("数据中没有 longitude 坐标")
    if "latitude" not in ds.coords:
        raise RuntimeError("数据中没有 latitude 坐标")

    lon = ((ds["longitude"] + 360) % 360).astype(np.float32)
    ds = ds.assign_coords(longitude=lon)
    ds = ds.sortby("longitude")
    ds = ds.sortby("latitude", ascending=False)

    return ds


def crop_area(ds, area):
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


# ============================================================
# 3. 读取并计算 Wind10
# ============================================================

def load_surface_wind10(nc_path: Path, area):
    if not nc_path.exists():
        raise FileNotFoundError(f"文件不存在：{nc_path}")

    with xr.open_dataset(nc_path, decode_times=True) as raw:
        ds = raw.load()

    ds = rename_common_coords(ds)
    ds = select_first_time(ds)
    ds = normalize_lon_lat(ds)

    u10_name = find_var_name(
        ds,
        ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u10m"]
    )
    v10_name = find_var_name(
        ds,
        ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "v10m"]
    )

    if u10_name is None or v10_name is None:
        raise KeyError(
            f"无法识别 U10/V10 变量。当前变量列表：{list(ds.data_vars)}"
        )

    u10 = drop_extra_dims_da(ds[u10_name])
    v10 = drop_extra_dims_da(ds[v10_name])

    u10 = crop_area(u10, area).transpose("latitude", "longitude")
    v10 = crop_area(v10, area).transpose("latitude", "longitude")

    wind10 = np.sqrt(u10 ** 2 + v10 ** 2)
    wind10.name = "wind10"
    wind10.attrs["units"] = "m s^-1"

    return wind10


# ============================================================
# 4. 陆地背景
# ============================================================

def load_land_geodataframe():
    if gpd is None:
        return None

    if LAND_BOUNDARY_FILE is not None:
        p = Path(LAND_BOUNDARY_FILE)
        if p.exists():
            try:
                world = gpd.read_file(p)
                return world
            except Exception as e:
                print(f"警告：读取 LAND_BOUNDARY_FILE 失败：{p}，原因：{e}")

    # 尝试 geopandas 自带
    try:
        world_path = gpd.datasets.get_path("naturalearth_lowres")
        world = gpd.read_file(world_path)
        return world
    except Exception as e:
        print(f"提示：无法加载 geopandas 内置陆地边界，原因：{e}")

    return None


def add_land_background(ax, extent):
    """
    extent = [lon_min, lon_max, lat_min, lat_max]
    """
    world = load_land_geodataframe()
    if world is None:
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

        world_clip.plot(
            ax=ax,
            facecolor="#F2E8D5",
            edgecolor="gray",
            linewidth=0.7,
            zorder=0
        )
        world_clip.boundary.plot(
            ax=ax,
            color="gray",
            linewidth=0.8,
            zorder=1
        )

    except Exception as e:
        print(f"警告：绘制陆地背景失败，原因：{e}")


# ============================================================
# 5. 绘图函数
# ============================================================

def style_axis(ax):
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, linestyle="--", alpha=0.35, zorder=2)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")


def plot_field(ax, da, title, cmap="viridis", vmin=None, vmax=None, norm=None, cbar_label=""):
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
        zorder=2.5
    )

    lat_max, lon_min, lat_min, lon_max = AREA
    add_land_background(ax, [lon_min, lon_max, lat_min, lat_max])

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_title(title, fontsize=12, fontweight="bold")
    style_axis(ax)
    ax.set_aspect("equal", adjustable="box")

    cbar = plt.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label(cbar_label)

    return im


def make_figure():
    # 读取数据
    wind_ref = load_surface_wind10(ERA5_FILE, AREA)
    wind_gdas = load_surface_wind10(GDAS_FILE, AREA)
    wind_era5lag = load_surface_wind10(ERA5_LAGGED_FILE, AREA)

    # 对齐网格（保险起见）
    wind_gdas = wind_gdas.interp_like(wind_ref, method="nearest")
    wind_era5lag = wind_era5lag.interp_like(wind_ref, method="nearest")

    # 误差场
    err_gdas = wind_gdas - wind_ref
    err_era5lag = wind_era5lag - wind_ref

    # 配色范围
    wind_vmin = float(np.nanmin(wind_ref.values))
    wind_vmax = float(np.nanmax(wind_ref.values))

    err_absmax = float(
        np.nanmax(
            np.abs(
                np.concatenate([
                    err_gdas.values.ravel(),
                    err_era5lag.values.ravel()
                ])
            )
        )
    )
    err_norm = TwoSlopeNorm(vmin=-err_absmax, vcenter=0.0, vmax=err_absmax)

    # 版式
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    if LAYOUT == "1x3":
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    elif LAYOUT == "3x1":
        fig, axes = plt.subplots(3, 1, figsize=(6.2, 14.5), constrained_layout=True)
    else:
        raise ValueError('LAYOUT 只能取 "1x3" 或 "3x1"')

    # 子图
    plot_field(
        axes[0],
        wind_ref,
        title="(a) ERA5_reference Wind10",
        cmap="viridis",
        vmin=wind_vmin,
        vmax=wind_vmax,
        cbar_label="Wind10 (m s$^{-1}$)"
    )

    plot_field(
        axes[1],
        err_gdas,
        title="(b) GDAS_Realtime - ERA5_reference",
        cmap="RdBu_r",
        norm=err_norm,
        cbar_label="Wind10 error (m s$^{-1}$)"
    )

    plot_field(
        axes[2],
        err_era5lag,
        title="(c) ERA5_Lagged - ERA5_reference",
        cmap="RdBu_r",
        norm=err_norm,
        cbar_label="Wind10 error (m s$^{-1}$)"
    )

    fig.suptitle(
        "Figure 3-4  T+24 h Wind10 distribution and error fields",
        fontsize=15,
        fontweight="bold"
    )

    out_png = OUT_DIR / f"Figure3_4_T24_Wind10_error_{LAYOUT}.png"
    out_svg = OUT_DIR / f"Figure3_4_T24_Wind10_error_{LAYOUT}.svg"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")

    print(f"图片已保存：{out_png}")
    print(f"图片已保存：{out_svg}")

    if SHOW_FIG:
        plt.show()

    plt.close(fig)


# ============================================================
# 6. 主程序
# ============================================================

if __name__ == "__main__":
    make_figure()