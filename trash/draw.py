
# data_dir = r"E:\pyCharmProject\pangu\model_output\gdas\2025-08-01-00-00to2025-08-08-00-00gdas_fnl"


import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

# ================== 配置区 ==================
data_dir = r"../model_input/single_time_point/era5/2025-08-01-00-00"
output_dir = None

LAT_MIN, LAT_MAX = 0.0, 60.0
LON_MIN, LON_MAX = 120.0, 180.0

GRID_DX = 2   # 经度网格间隔（度） ← 可调
GRID_DY = 2   # 纬度网格间隔（度） ← 可调

TARGET_QV_DENSITY = 25

MSL_CANDS = ["msl", "mean_sea_level_pressure", "prmsl", "PRMSL_meansealevel"]
U10_CANDS = ["u_component_of_wind_10m", "u10", "UGRD_10maboveground"]
V10_CANDS = ["v_component_of_wind_10m", "v10", "VGRD_10maboveground"]
# ============================================


def find_var(ds, cands):
    for c in cands:
        if c in ds.variables:
            return c
    for c in cands:
        for v in ds.variables:
            if c.lower() in v.lower():
                return v
    return None


def to_2d(arr):
    a = np.array(arr)
    if a.ndim == 3:
        return a[0]
    return np.squeeze(a)


def plot_with_map(msl, u10, v10, lat, lon, outpath, title):
    # Pa → hPa
    if np.nanmax(msl) > 2000:
        msl = msl / 100.0

    Lon, Lat = np.meshgrid(lon, lat)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=proj)

    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)

    # ===== 地图要素 =====
    ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=3)

    # ===== 经纬网格线（核心）=====
    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.6,
        linestyle="--"
    )

    # 网格间隔（可调）
    gl.xlocator = mticker.FixedLocator(
        np.arange(LON_MIN, LON_MAX + GRID_DX, GRID_DX)
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(LAT_MIN, LAT_MAX + GRID_DY, GRID_DY)
    )

    # 标签格式
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # 只显示左 / 下（ncvue 风格）
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    # ===== MSL =====
    pcm = ax.pcolormesh(
        Lon, Lat, msl,
        transform=proj,
        shading="auto",
        zorder=1
    )
    cb = plt.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("MSL (hPa)")

    cs = ax.contour(
        Lon, Lat, msl,
        colors="k",
        linewidths=0.5,
        transform=proj,
        zorder=2
    )
    ax.clabel(cs, fmt="%.0f", fontsize=8)

    # ===== 风矢量 =====
    ny, nx = msl.shape
    sx = max(1, nx // TARGET_QV_DENSITY)
    sy = max(1, ny // TARGET_QV_DENSITY)

    q = ax.quiver(
        Lon[::sy, ::sx], Lat[::sy, ::sx],
        u10[::sy, ::sx], v10[::sy, ::sx],
        transform=proj,
        scale=700,
        zorder=4
    )
    ax.quiverkey(q, 0.9, 1.03, 10, "10 m/s", labelpos="E")

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    for fp in files:

        if not "surface" in fp:
            continue

        print("处理:", fp)
        ds = xr.open_dataset(fp)

        msl_name = find_var(ds, MSL_CANDS)
        u10_name = find_var(ds, U10_CANDS)
        v10_name = find_var(ds, V10_CANDS)

        if not all([msl_name, u10_name, v10_name]):
            print("  缺少变量，跳过")
            ds.close()
            continue

        lat = ds["latitude"].values
        lon = ds["longitude"].values

        # 经度统一到 0–360
        if lon.min() < 0:
            lon = (lon + 360) % 360

        lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
        lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)

        msl = to_2d(ds[msl_name].values)[np.ix_(lat_mask, lon_mask)]
        u10 = to_2d(ds[u10_name].values)[np.ix_(lat_mask, lon_mask)]
        v10 = to_2d(ds[v10_name].values)[np.ix_(lat_mask, lon_mask)]

        lat_sel = lat[lat_mask]
        lon_sel = lon[lon_mask]

        outdir = output_dir or os.path.dirname(fp)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, os.path.basename(fp).replace(".nc", ".png"))

        plot_with_map(msl, u10, v10, lat_sel, lon_sel, outpath, os.path.basename(fp))
        ds.close()

        print("  保存:", outpath)


if __name__ == "__main__":
    main()
