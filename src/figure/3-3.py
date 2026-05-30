# -*- coding: utf-8 -*-
"""
track_typhoon_center_wipha_20250717.py

功能：
1. 读取盘古 surface 输出文件；
2. 在指定经纬度范围内，以 MSL 最小值定位台风中心；
3. 输出各时效台风中心经纬度、最低气压、最大10m风速；
4. 绘制台风路径图；
5. 可选读取 ERA5 参考场，用于后续计算路径误差；
6. 路径图加载陆地边界背景；
7. 路径图标签隐藏 T+3、T+6、T+12，避免标签拥挤。

适用案例：
2025年第6号台风“韦帕”；
目标起报时间：2025-07-17 00:00；
预报时效：1, 3, 6, 12, 24, 48, 72 h。

重要设置：
1. GDAS_Realtime 和 ERA5_Lagged 预测场使用 SEARCH_BOX；
2. ERA5_reference 真值场使用 ERA5_REFERENCE_SEARCH_BOX；
3. ERA5_reference 只在右下区域搜索，避免左上方其他低压中心干扰。
"""

from pathlib import Path
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# 陆地边界背景相关库，可选
try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception:
    gpd = None
    box = None


# ============================================================
# 1. 用户配置区
# ============================================================

PROJECT_ROOT = Path(r"/")

# 目标起报时间：用于分析 2025-07-17 之后连续72h路径
TARGET_START_TIME = "2025-07-17-00-00"

# 预报时效
LEAD_HOURS = [1, 3, 6, 12, 24, 48, 72]

# ------------------------------------------------------------
# 预测场搜索范围：用于 GDAS_Realtime 和 ERA5_Lagged
# 格式：[lat_max, lon_min, lat_min, lon_max]
# ------------------------------------------------------------
SEARCH_BOX = [30, 105, 10, 130]

# ------------------------------------------------------------
# ERA5 真值专用搜索范围：只搜索右下区域
# 用于避免左上方其他低压中心干扰 ERA5_reference 的台风中心定位
# 格式：[lat_max, lon_min, lat_min, lon_max]
# ------------------------------------------------------------
ERA5_REFERENCE_SEARCH_BOX = [23, 112, 10, 130]

# 如果 ERA5 真值仍跳到错误低压中心，可以继续缩小，例如：
# ERA5_REFERENCE_SEARCH_BOX = [22, 113, 10, 128]
# 如果后期台风中心超出范围，可以适当放宽，例如：
# ERA5_REFERENCE_SEARCH_BOX = [25, 110, 10, 130]

# 是否读取 ERA5 参考场
ADD_ERA5_REFERENCE = True

# 预测方案配置
SCHEMES = [
    {
        "name": "GDAS_Realtime",
        "data_type": "gdas",
        "delay_hours": 0,
    },
    {
        "name": "ERA5_Lagged",
        "data_type": "era5",
        "delay_hours": 120,
    },
]

# 输出目录
OUT_DIR = PROJECT_ROOT / "src" / "comparison_results" / "wipha_20250717_track"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 是否显示图
SHOW_FIG = True

# ------------------------------------------------------------
# 陆地边界文件
# ------------------------------------------------------------
# 如果你有自己的 shp 文件，可以填写路径，例如：
LAND_BOUNDARY_FILE = r"/src/ne_10m_land/ne_10m_land.shp"
# 如果为 None，程序会优先尝试 cartopy 的 Natural Earth 数据；
# 如果失败，再尝试 geopandas 自带数据；
# 如果仍失败，则跳过陆地背景。
# LAND_BOUNDARY_FILE = None

# 路径图中隐藏的时效标签
HIDE_LABEL_LEADS = {3, 6, 12}


# ============================================================
# 2. 基础工具函数
# ============================================================

def parse_datetime(s: str) -> datetime:
    s = str(s).strip()
    fmts = [
        "%Y-%m-%d-%H-%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y%m%d%H",
    ]

    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    raise ValueError(f"无法识别日期格式: {s}")


def time_to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H-%M")


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

    # 统一经度到 0–360
    lon = ((ds["longitude"] + 360) % 360).astype(np.float32)
    ds = ds.assign_coords(longitude=lon)
    ds = ds.sortby("longitude")

    # 纬度统一为从北到南
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
    """
    将多余的一维维度去掉，例如 valid_time、time、step 等。
    """
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
    """
    根据变量名和属性查找变量。
    """
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


def haversine_km(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的大圆距离，单位 km。
    """
    r = 6371.0

    lon1 = math.radians(float(lon1))
    lat1 = math.radians(float(lat1))
    lon2 = math.radians(float(lon2))
    lat2 = math.radians(float(lat2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))
    return r * c


# ============================================================
# 3. 文件路径工具
# ============================================================

def get_prediction_file(
    project_root: Path,
    data_type: str,
    model_start_time: datetime,
    model_forecast_hour: int,
    valid_time: datetime,
    kind: str = "surface",
) -> Path:
    """
    预测文件路径：
    model_output/{data_type}/{model_start_time}/{model_forecast_hour}/output_surface_{valid_time}.nc
    """
    model_start_str = time_to_str(model_start_time)
    valid_time_str = time_to_str(valid_time)

    pred_base = project_root / "model_output" / data_type / model_start_str

    filename = f"output_{kind}_{valid_time_str}.nc"

    candidate = pred_base / str(model_forecast_hour) / filename

    if candidate.exists():
        return candidate

    matches = list(pred_base.rglob(filename)) if pred_base.exists() else []

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"找不到预测文件：{filename}\n"
        f"优先查找路径：{candidate}\n"
        f"递归查找目录：{pred_base}"
    )


def get_era5_truth_file(project_root: Path, valid_time: datetime, kind: str = "surface") -> Path:
    """
    ERA5参考场路径：
    model_input/single_time_point/era5/{valid_time}/surface.nc
    """
    valid_time_str = time_to_str(valid_time)

    path = (
        project_root
        / "model_input"
        / "single_time_point"
        / "era5"
        / valid_time_str
        / f"{kind}.nc"
    )

    if not path.exists():
        raise FileNotFoundError(f"找不到ERA5参考场文件：{path}")

    return path


# ============================================================
# 4. 台风中心定位
# ============================================================

def open_surface_dataset(path: Path) -> xr.Dataset:
    with xr.open_dataset(path, decode_times=True) as raw:
        ds = raw.load()

    ds = rename_common_coords(ds)
    ds = select_first_time(ds)
    ds = normalize_lon_lat(ds)

    return ds


def extract_surface_vars(ds: xr.Dataset):
    """
    提取 MSL、U10、V10。
    """
    msl_candidates = [
        "msl",
        "mean_sea_level_pressure",
        "MSL",
        "prmsl",
    ]

    u10_candidates = [
        "u10",
        "10m_u_component_of_wind",
        "u_component_of_wind_10m",
        "u10m",
    ]

    v10_candidates = [
        "v10",
        "10m_v_component_of_wind",
        "v_component_of_wind_10m",
        "v10m",
    ]

    msl_name = find_var_name(ds, msl_candidates)
    u10_name = find_var_name(ds, u10_candidates)
    v10_name = find_var_name(ds, v10_candidates)

    if msl_name is None:
        raise KeyError(f"无法识别 MSL 变量。当前变量：{list(ds.data_vars)}")

    if u10_name is None:
        print("警告：未找到 U10 变量，最大风速将输出 NaN。")

    if v10_name is None:
        print("警告：未找到 V10 变量，最大风速将输出 NaN。")

    msl = drop_extra_dims_da(ds[msl_name])
    u10 = drop_extra_dims_da(ds[u10_name]) if u10_name else None
    v10 = drop_extra_dims_da(ds[v10_name]) if v10_name else None

    return msl, u10, v10


def locate_typhoon_center(path: Path, search_box) -> dict:
    """
    在 search_box 范围内寻找 MSL 最小值作为台风中心。

    返回：
    center_lon, center_lat, min_msl_pa, min_msl_hpa, max_wind10_ms, wind10_at_center_ms
    """
    ds = open_surface_dataset(path)

    msl, u10, v10 = extract_surface_vars(ds)

    msl_crop = crop_area(msl, search_box)
    msl_crop = msl_crop.transpose("latitude", "longitude")

    arr = msl_crop.values.astype(float)

    if np.all(~np.isfinite(arr)):
        raise RuntimeError(f"搜索范围内 MSL 全为 NaN，请检查 search_box：{search_box}")

    flat_idx = np.nanargmin(arr)
    i_lat, i_lon = np.unravel_index(flat_idx, arr.shape)

    center_lat = float(msl_crop["latitude"].values[i_lat])
    center_lon = float(msl_crop["longitude"].values[i_lon])
    min_msl_raw = float(arr[i_lat, i_lon])

    # MSL 通常为 Pa；如果已经是 hPa，则不转换
    if min_msl_raw > 2000:
        min_msl_pa = min_msl_raw
        min_msl_hpa = min_msl_raw / 100.0
    else:
        min_msl_pa = min_msl_raw * 100.0
        min_msl_hpa = min_msl_raw

    max_wind10 = np.nan
    wind10_at_center = np.nan

    if u10 is not None and v10 is not None:
        u10_crop = crop_area(u10, search_box).interp_like(msl_crop, method="nearest")
        v10_crop = crop_area(v10, search_box).interp_like(msl_crop, method="nearest")

        wind10 = np.sqrt(u10_crop ** 2 + v10_crop ** 2)
        wind10_arr = wind10.values.astype(float)

        max_wind10 = float(np.nanmax(wind10_arr))
        wind10_at_center = float(wind10_arr[i_lat, i_lon])

    return {
        "center_lon": center_lon,
        "center_lat": center_lat,
        "min_msl_pa": min_msl_pa,
        "min_msl_hpa": min_msl_hpa,
        "max_wind10_ms": max_wind10,
        "wind10_at_center_ms": wind10_at_center,
    }


# ============================================================
# 5. 路径提取
# ============================================================

def build_prediction_track_for_scheme(scheme: dict) -> pd.DataFrame:
    target_start = parse_datetime(TARGET_START_TIME)

    name = scheme["name"]
    data_type = scheme["data_type"]
    delay_hours = int(scheme.get("delay_hours", 0))

    # 对 GDAS：model_start = 2025-07-17 00，model_hour = lead_hour
    # 对 ERA5_Lagged：model_start = 2025-07-12 00，model_hour = 120 + lead_hour
    model_start_time = target_start - timedelta(hours=delay_hours)

    rows = []

    print("=" * 100)
    print(f"开始提取方案：{name}")
    print(f"data_type       : {data_type}")
    print(f"target_start    : {time_to_str(target_start)}")
    print(f"model_start     : {time_to_str(model_start_time)}")
    print(f"delay_hours     : {delay_hours}")
    print(f"search_box      : {SEARCH_BOX}")
    print("=" * 100)

    for lead_hour in LEAD_HOURS:
        valid_time = target_start + timedelta(hours=int(lead_hour))
        model_forecast_hour = delay_hours + int(lead_hour)

        try:
            file_path = get_prediction_file(
                project_root=PROJECT_ROOT,
                data_type=data_type,
                model_start_time=model_start_time,
                model_forecast_hour=model_forecast_hour,
                valid_time=valid_time,
                kind="surface",
            )

            center_info = locate_typhoon_center(file_path, SEARCH_BOX)

            row = {
                "scheme": name,
                "data_type": data_type,
                "target_start_time": time_to_str(target_start),
                "model_start_time": time_to_str(model_start_time),
                "lead_hour": int(lead_hour),
                "model_forecast_hour": int(model_forecast_hour),
                "valid_time": time_to_str(valid_time),
                "file": str(file_path),
                "search_box": str(SEARCH_BOX),
                **center_info,
            }

            rows.append(row)

            print(
                f"T+{lead_hour:>2} h | "
                f"center=({center_info['center_lon']:.2f}, {center_info['center_lat']:.2f}) | "
                f"minMSL={center_info['min_msl_hpa']:.1f} hPa | "
                f"maxWind10={center_info['max_wind10_ms']:.2f} m/s"
            )

        except Exception as e:
            print(f"T+{lead_hour} h 处理失败：{e}")

            rows.append({
                "scheme": name,
                "data_type": data_type,
                "target_start_time": time_to_str(target_start),
                "model_start_time": time_to_str(model_start_time),
                "lead_hour": int(lead_hour),
                "model_forecast_hour": int(model_forecast_hour),
                "valid_time": time_to_str(valid_time),
                "file": "",
                "search_box": str(SEARCH_BOX),
                "center_lon": np.nan,
                "center_lat": np.nan,
                "min_msl_pa": np.nan,
                "min_msl_hpa": np.nan,
                "max_wind10_ms": np.nan,
                "wind10_at_center_ms": np.nan,
                "error": str(e),
            })

    return pd.DataFrame(rows)


def build_era5_reference_track() -> pd.DataFrame:
    """
    读取 ERA5 参考场，提取参考台风中心。

    注意：
    ERA5_reference 单独使用 ERA5_REFERENCE_SEARCH_BOX，
    只搜索右下区域，避免左上方其他低压中心干扰。
    """
    target_start = parse_datetime(TARGET_START_TIME)
    rows = []

    print("=" * 100)
    print("开始提取 ERA5_reference 路径")
    print(f"target_start: {time_to_str(target_start)}")
    print(f"search_box  : {ERA5_REFERENCE_SEARCH_BOX}")
    print("=" * 100)

    for lead_hour in LEAD_HOURS:
        valid_time = target_start + timedelta(hours=int(lead_hour))

        try:
            file_path = get_era5_truth_file(PROJECT_ROOT, valid_time, kind="surface")

            center_info = locate_typhoon_center(
                file_path,
                ERA5_REFERENCE_SEARCH_BOX
            )

            row = {
                "scheme": "ERA5_reference",
                "data_type": "era5_reference",
                "target_start_time": time_to_str(target_start),
                "model_start_time": "",
                "lead_hour": int(lead_hour),
                "model_forecast_hour": np.nan,
                "valid_time": time_to_str(valid_time),
                "file": str(file_path),
                "search_box": str(ERA5_REFERENCE_SEARCH_BOX),
                **center_info,
            }

            rows.append(row)

            print(
                f"T+{lead_hour:>2} h | "
                f"center=({center_info['center_lon']:.2f}, {center_info['center_lat']:.2f}) | "
                f"minMSL={center_info['min_msl_hpa']:.1f} hPa | "
                f"maxWind10={center_info['max_wind10_ms']:.2f} m/s"
            )

        except Exception as e:
            print(f"ERA5_reference T+{lead_hour} h 处理失败：{e}")

            rows.append({
                "scheme": "ERA5_reference",
                "data_type": "era5_reference",
                "target_start_time": time_to_str(target_start),
                "model_start_time": "",
                "lead_hour": int(lead_hour),
                "model_forecast_hour": np.nan,
                "valid_time": time_to_str(valid_time),
                "file": "",
                "search_box": str(ERA5_REFERENCE_SEARCH_BOX),
                "center_lon": np.nan,
                "center_lat": np.nan,
                "min_msl_pa": np.nan,
                "min_msl_hpa": np.nan,
                "max_wind10_ms": np.nan,
                "wind10_at_center_ms": np.nan,
                "error": str(e),
            })

    return pd.DataFrame(rows)


def add_center_error_against_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    若存在 ERA5_reference，则计算各方案中心误差。
    """
    df = df.copy()

    ref = df[df["scheme"] == "ERA5_reference"][
        ["lead_hour", "center_lon", "center_lat"]
    ].rename(
        columns={
            "center_lon": "ref_center_lon",
            "center_lat": "ref_center_lat",
        }
    )

    if ref.empty:
        df["center_error_km_vs_era5"] = np.nan
        return df

    df = df.merge(ref, on="lead_hour", how="left")

    errors = []

    for _, row in df.iterrows():
        if row["scheme"] == "ERA5_reference":
            errors.append(0.0)
            continue

        if (
            pd.isna(row["center_lon"])
            or pd.isna(row["center_lat"])
            or pd.isna(row["ref_center_lon"])
            or pd.isna(row["ref_center_lat"])
        ):
            errors.append(np.nan)
        else:
            errors.append(
                haversine_km(
                    row["center_lon"],
                    row["center_lat"],
                    row["ref_center_lon"],
                    row["ref_center_lat"],
                )
            )

    df["center_error_km_vs_era5"] = errors

    return df


# ============================================================
# 6. 陆地背景与路径图绘制
# ============================================================

def load_land_geodataframe():
    """
    加载陆地边界数据。

    优先级：
    1. 用户指定 LAND_BOUNDARY_FILE；
    2. cartopy Natural Earth；
    3. geopandas 自带 naturalearth_lowres；
    4. 全部失败则返回 None。
    """
    if gpd is None:
        print("警告：未安装 geopandas，无法加载陆地边界背景。")
        return None

    # 1. 用户指定 shp 文件
    if LAND_BOUNDARY_FILE is not None:
        p = Path(LAND_BOUNDARY_FILE)
        if p.exists():
            try:
                world = gpd.read_file(p)
                return world
            except Exception as e:
                print(f"警告：读取 LAND_BOUNDARY_FILE 失败：{p}，原因：{e}")
        else:
            print(f"警告：LAND_BOUNDARY_FILE 不存在：{p}")

    # 2. cartopy Natural Earth
    try:
        from cartopy.io import shapereader
        shp = shapereader.natural_earth(
            resolution="10m",
            category="physical",
            name="land"
        )
        world = gpd.read_file(shp)
        return world
    except Exception as e:
        print(f"提示：无法通过 cartopy 加载 Natural Earth land，原因：{e}")

    # 3. geopandas 自带数据
    try:
        world_path = gpd.datasets.get_path("naturalearth_lowres")
        world = gpd.read_file(world_path)
        return world
    except Exception as e:
        print(f"提示：无法通过 geopandas.datasets 加载 naturalearth_lowres，原因：{e}")

    return None


def add_land_background(ax, extent):
    """
    在路径图背景中加载陆地边界。

    extent = [lon_min, lon_max, lat_min, lat_max]
    """
    lon_min, lon_max, lat_min, lat_max = extent

    world = load_land_geodataframe()

    if world is None:
        print("警告：未能加载陆地边界，路径图将不显示陆地背景。")
        return

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
            print("警告：裁剪后的陆地边界为空，请检查绘图范围。")
            return

        # 陆地填充
        world_clip.plot(
            ax=ax,
            facecolor="#F2E8D5",
            edgecolor="gray",
            linewidth=0.7,
            zorder=0
        )

        # 边界线
        world_clip.boundary.plot(
            ax=ax,
            color="gray",
            linewidth=0.8,
            zorder=1
        )

    except Exception as e:
        print(f"警告：绘制陆地背景失败，原因：{e}")


def plot_track(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 7.2))

    ax.set_facecolor("#F7F7F7")

    style_map = {
        "ERA5_reference": {
            "color": "black",
            "linestyle": "-",
            "marker": "o",
            "label": "ERA5_reference",
        },
        "GDAS_Realtime": {
            "color": "#4C72B0",
            "linestyle": "-",
            "marker": "o",
            "label": "GDAS_Realtime",
        },
        "ERA5_Lagged": {
            "color": "#DD8452",
            "linestyle": "--",
            "marker": "s",
            "label": "ERA5_Lagged",
        },
    }

    # 不同方案的标签偏移，减少重叠
    label_offsets = {
        "ERA5_reference": (0.18, 0.14),
        "GDAS_Realtime": (0.12, -0.25),
        "ERA5_Lagged": (0.12, 0.22),
    }

    # 画图仍使用总范围 SEARCH_BOX，方便显示全部方案路径
    lat_max, lon_min, lat_min, lon_max = SEARCH_BOX

    # 先画陆地背景
    add_land_background(
        ax,
        extent=[lon_min, lon_max, lat_min, lat_max]
    )

    # 再画路径
    for scheme, group in df.groupby("scheme"):
        group = group.sort_values("lead_hour")

        group = group[
            np.isfinite(group["center_lon"]) &
            np.isfinite(group["center_lat"])
        ]

        if group.empty:
            continue

        st = style_map.get(
            scheme,
            {
                "color": None,
                "linestyle": "-",
                "marker": "o",
                "label": scheme,
            }
        )

        ax.plot(
            group["center_lon"],
            group["center_lat"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.8,
            markersize=7,
            color=st["color"],
            label=st["label"],
            zorder=3,
        )

        dx, dy = label_offsets.get(scheme, (0.15, 0.15))

        for _, row in group.iterrows():
            lead = int(row["lead_hour"])

            # 关键修改：隐藏 T+3、T+6、T+12 标签
            if lead in HIDE_LABEL_LEADS:
                continue

            ax.text(
                row["center_lon"] + dx,
                row["center_lat"] + dy,
                f"T+{lead}",
                fontsize=10,
                color=st["color"],
                zorder=4,
            )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    ax.set_title(
        "Typhoon center track from MSL minimum\n"
        f"Target start: {TARGET_START_TIME}",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )

    ax.grid(True, linestyle="--", alpha=0.35, zorder=2)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    ax.set_aspect("equal", adjustable="box")

    png_path = OUT_DIR / "wipha_20250717_typhoon_center_track.png"
    svg_path = OUT_DIR / "wipha_20250717_typhoon_center_track.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"\n路径图已保存：{png_path}")
    print(f"路径图已保存：{svg_path}")

    if SHOW_FIG:
        plt.show()

    plt.close(fig)


# ============================================================
# 7. 主程序
# ============================================================

def main():
    all_tracks = []

    # 预测方案
    for scheme in SCHEMES:
        df_scheme = build_prediction_track_for_scheme(scheme)
        all_tracks.append(df_scheme)

    # ERA5参考路径
    if ADD_ERA5_REFERENCE:
        df_ref = build_era5_reference_track()
        all_tracks.append(df_ref)

    df_all = pd.concat(all_tracks, ignore_index=True)

    # 计算相对于ERA5参考场的中心定位误差
    df_all = add_center_error_against_reference(df_all)

    # 保存结果
    csv_path = OUT_DIR / "wipha_20250717_typhoon_center_track.csv"
    xlsx_path = OUT_DIR / "wipha_20250717_typhoon_center_track.xlsx"

    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_all.to_excel(xlsx_path, index=False)

    print("\n" + "=" * 100)
    print("台风中心路径结果已保存")
    print(csv_path)
    print(xlsx_path)
    print("=" * 100)

    preview_cols = [
        "scheme",
        "lead_hour",
        "valid_time",
        "center_lon",
        "center_lat",
        "search_box",
        "min_msl_hpa",
        "max_wind10_ms",
        "center_error_km_vs_era5",
    ]

    print("\n路径结果预览：")
    print(df_all[preview_cols].to_string(index=False))

    # 绘制路径图
    plot_track(df_all)


if __name__ == "__main__":
    main()