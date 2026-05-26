
"""
用途：
1. 同时支持 GDAS 预测结果和 5 天延迟 ERA5 预测结果的批量对比。
2. 以“目标起报时间 target_time”为批量循环对象：
   - GDAS:  预测起报时间 = target_time，模型时效 = lead_hour
   - ERA5:  预测起报时间 = target_time - 120h，模型时效 = 120h + lead_hour
3. 自动裁切到中国海区域，并使用 lsm.nc 进行海洋掩膜统计。
4. 可选：去掉陆地内部湖泊/水库等孤立水体，只保留与区域边界连通的海洋水体。

使用方式：
直接修改“用户配置区”，然后运行本文件即可。
"""

from __future__ import annotations

import traceback
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. 用户配置区：只需要改这里
# ============================================================

PROJECT_ROOT = Path(r"E:\PyCharm_WorkSpace\pangu")

# 可选："gdas" 或 "era5"
DATA_TYPE = "era5"

# 这里填写“希望评价的目标起报时间”，不是 ERA5 延迟后的真实模型起报时间。
# 例如：希望评价 2025-07-01 00 时对应的 1/3/6/12/24/48/72h 预报，
# 则 TARGET_START_TIME = "2025-07-01-00-00"。
# 如果 DATA_TYPE="era5"，程序会自动去找：model_output/era5/2025-06-26-00-00/121/...
TARGET_START_TIME = "2025-07-01-00-00"
TARGET_END_TIME   = "2025-07-31-00-00"

# 批量间隔。日批量一般为 24。
RUN_INTERVAL_HOURS = 24

# 希望评价的预报时效。这里是“相对于目标起报时间”的时效。
# 对 GDAS：实际查找 1,3,6,12,24,48,72 文件夹。
# 对 ERA5 延迟 5 天：实际查找 121,123,126,132,144,168,192 文件夹。
LEAD_HOURS = [1, 3, 6, 12, 24, 48, 72]

# ERA5 延迟小时数。5 天 = 120 小时。
# GDAS 自动使用 0 小时，不受此参数影响。
ERA5_DELAY_HOURS = 120

# 中国海及邻近区域：lat_max, lon_min, lat_min, lon_max
AREA = [42, 103, 13, 130]

# lsm.nc 路径。None 表示自动查找。
# 推荐你固定写成实际路径，避免误读其他 lsm.nc。
LSM_PATH = PROJECT_ROOT / "src" / "lsm.nc"
# LSM_PATH = None

# 输出根目录。None 表示自动生成。
OUT_ROOT = None

# ERA5 lsm 通常：1=陆地，0=海洋；<=0.5 作为水体/海洋。
OCEAN_THRESHOLD = 0.5

# 是否去掉陆地内部湖泊/水库等孤立水体。
# True：只保留与裁剪区域边界连通的水体，适合中国海区域。
# False：只使用 lsm <= 0.5。
REMOVE_INLAND_WATER = True

# 输出每个 target_time 的单独 CSV，另外也会输出总表。
SAVE_PER_TARGET_CSV = True


# ============================================================
# 2. 常量配置
# ============================================================

PRESSURE_LEVELS = np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    dtype=np.float32
)

PRED_SURFACE_VARS = {
    "msl": ["mean_sea_level_pressure", "msl"],
    "u10": ["u_component_of_wind_10m", "10m_u_component_of_wind", "u10"],
    "v10": ["v_component_of_wind_10m", "10m_v_component_of_wind", "v10"],
    "t2m": ["temperature_2m", "2m_temperature", "t2m"],
}

TRUE_SURFACE_VARS = {
    "msl": ["msl", "mean_sea_level_pressure"],
    "u10": ["u10", "10m_u_component_of_wind"],
    "v10": ["v10", "10m_v_component_of_wind"],
    "t2m": ["t2m", "2m_temperature"],
}

PRED_UPPER_VARS = {
    "z": ["geopotential", "z"],
    "q": ["specific_humidity", "q"],
    "t": ["temperature", "t"],
    "u": ["u_component_of_wind", "u"],
    "v": ["v_component_of_wind", "v"],
}

TRUE_UPPER_VARS = {
    "z": ["z", "geopotential"],
    "q": ["q", "specific_humidity"],
    "t": ["t", "temperature"],
    "u": ["u", "u_component_of_wind"],
    "v": ["v", "v_component_of_wind"],
}


# ============================================================
# 3. 时间与路径工具
# ============================================================

def parse_datetime(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return s

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


def get_delay_hours(data_type: str) -> int:
    if data_type.lower() == "era5":
        return int(ERA5_DELAY_HOURS)
    if data_type.lower() == "gdas":
        return 0
    raise ValueError("DATA_TYPE 只能是 'gdas' 或 'era5'")


def get_pred_start_time(target_time: datetime, data_type: str) -> datetime:
    """
    根据目标起报时间计算真实模型输出目录对应的起报时间。

    GDAS：
        target_time = 2025-07-01 00
        pred_start_time = 2025-07-01 00

    ERA5 5天延迟：
        target_time = 2025-07-01 00
        pred_start_time = 2025-06-26 00
    """
    delay_hours = get_delay_hours(data_type)
    return target_time - timedelta(hours=delay_hours)


def get_model_forecast_hour(lead_hour: int, data_type: str) -> int:
    """
    根据希望评价的 lead_hour，得到模型输出文件夹里的真实时效。

    GDAS:
        lead_hour=1   -> model_forecast_hour=1
    ERA5 5天延迟:
        lead_hour=1   -> model_forecast_hour=121
        lead_hour=72  -> model_forecast_hour=192
    """
    return int(get_delay_hours(data_type) + int(lead_hour))


def find_existing_lsm(project_root: Path, user_lsm: str | Path | None = None) -> Path:
    if user_lsm is not None:
        p = Path(user_lsm)
        if not p.exists():
            raise FileNotFoundError(f"指定的 lsm.nc 不存在: {p}")
        return p

    candidates = [
        project_root / "lsm.nc",
        project_root / "src" / "lsm.nc",
        project_root / "model_input" / "lsm.nc",
        project_root / "model_input" / "single_time_point" / "lsm.nc",
        project_root / "model_input" / "single_time_point" / "era5" / "lsm.nc",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("没有找到 lsm.nc。请在用户配置区设置 LSM_PATH。")


def get_truth_file(project_root: Path, valid_time_str: str, kind: str) -> Path:
    path = (
        project_root
        / "model_input"
        / "single_time_point"
        / "era5"
        / valid_time_str
        / f"{kind}.nc"
    )

    if not path.exists():
        raise FileNotFoundError(f"找不到 ERA5 真值文件: {path}")

    return path


def find_pred_file(pred_base: Path, model_forecast_hour: int, kind: str, valid_time_str: str) -> Path:
    """
    查找预测文件。

    例如 ERA5 延迟场：
    pred_base = E:/.../model_output/era5/2025-06-26-00-00
    model_forecast_hour = 121
    kind = surface
    valid_time_str = 2025-07-01-01-00

    优先查找：
    E:/.../model_output/era5/2025-06-26-00-00/121/output_surface_2025-07-01-01-00.nc
    """
    filename = f"output_{kind}_{valid_time_str}.nc"

    candidate = pred_base / str(model_forecast_hour) / filename
    if candidate.exists():
        return candidate

    matches = list(pred_base.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"找不到预测文件: {filename}\n"
        f"优先查找路径: {candidate}\n"
        f"递归搜索目录: {pred_base}"
    )


# ============================================================
# 4. Dataset 标准化工具
# ============================================================

def find_coord_or_dim(ds: xr.Dataset, candidates) -> str | None:
    for c in candidates:
        if c in ds.coords or c in ds.dims or c in ds.variables:
            return c
    return None


def rename_common_coords(ds: xr.Dataset, need_pressure: bool = False) -> xr.Dataset:
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

    if need_pressure:
        plev_name = find_coord_or_dim(ds, ["pressure_level", "level", "isobaricInhPa"])
        if plev_name and plev_name != "pressure_level":
            rename_dict[plev_name] = "pressure_level"

    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds


def select_first_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        ds = ds.isel(valid_time=0, drop=True)
    return ds


def normalize_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" not in ds.coords:
        raise RuntimeError("数据中没有 longitude 坐标")
    if "latitude" not in ds.coords:
        raise RuntimeError("数据中没有 latitude 坐标")

    lon = ((ds["longitude"] + 360) % 360).astype(np.float32)
    ds = ds.assign_coords(longitude=lon)
    ds = ds.sortby("longitude")
    ds = ds.sortby("latitude", ascending=False)
    return ds


def ensure_pressure_coordinate(ds: xr.Dataset) -> xr.Dataset:
    if "pressure_level" not in ds.dims and "level" in ds.dims:
        ds = ds.rename({"level": "pressure_level"})

    if "pressure_level" not in ds.dims:
        for dim, size in ds.sizes.items():
            if size == len(PRESSURE_LEVELS) and dim not in [
                "latitude", "longitude", "valid_time", "time", "lat", "lon"
            ]:
                ds = ds.rename({dim: "pressure_level"})
                break

    if "pressure_level" not in ds.dims:
        raise RuntimeError(
            f"无法识别高空垂直维度。当前维度: {dict(ds.sizes)}，当前坐标: {list(ds.coords)}"
        )

    if "pressure_level" not in ds.coords:
        if ds.sizes["pressure_level"] == len(PRESSURE_LEVELS):
            ds = ds.assign_coords(pressure_level=PRESSURE_LEVELS)
            return ds
        raise RuntimeError(
            f"pressure_level 没有坐标值，且层数不是 13。当前层数: {ds.sizes['pressure_level']}"
        )

    vals = np.asarray(ds["pressure_level"].values, dtype=np.float32)

    if len(vals) == len(PRESSURE_LEVELS):
        vals_sorted = np.sort(vals)
        is_zero_based = np.allclose(vals_sorted, np.arange(13), atol=0.01)
        is_one_based = np.allclose(vals_sorted, np.arange(1, 14), atol=0.01)
        if is_zero_based or is_one_based:
            ds = ds.assign_coords(pressure_level=PRESSURE_LEVELS)
            return ds

    return ds


def normalize_pressure(ds: xr.Dataset) -> xr.Dataset:
    ds = ensure_pressure_coordinate(ds)

    vals = np.asarray(ds["pressure_level"].values, dtype=np.float32)

    if np.nanmax(vals) > 2000:
        vals = vals / 100.0
        ds = ds.assign_coords(pressure_level=vals)

    vals = np.asarray(ds["pressure_level"].values, dtype=np.float32)
    available_before = vals.copy()

    standard_like = []
    for p in PRESSURE_LEVELS:
        standard_like.append(np.any(np.isclose(vals, p, atol=0.1)))

    if not any(standard_like) and len(vals) == len(PRESSURE_LEVELS):
        print(
            "警告：检测到高空层坐标不是标准气压层，"
            "但层数为 13，已按盘古标准层顺序赋值为 "
            "[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]。"
            f"原始坐标值为: {available_before.tolist()}"
        )
        ds = ds.assign_coords(pressure_level=PRESSURE_LEVELS)
        return ds

    ds = ds.sortby("pressure_level", ascending=False)
    available = np.asarray(ds["pressure_level"].values, dtype=np.float32)

    missing = []
    for p in PRESSURE_LEVELS:
        if not np.any(np.isclose(available, p, atol=0.1)):
            missing.append(float(p))

    if missing:
        raise RuntimeError(f"缺少气压层: {missing}。当前 pressure_level 为: {available.tolist()}")

    ds = ds.sel(pressure_level=PRESSURE_LEVELS, method="nearest")
    ds = ds.assign_coords(pressure_level=PRESSURE_LEVELS)
    return ds


def drop_extra_dims(ds: xr.Dataset, allowed_dims: set) -> xr.Dataset:
    for dim in list(ds.dims):
        if dim in allowed_dims:
            continue
        if dim == "expver":
            ds = ds.max(dim="expver", skipna=True)
        elif ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
        else:
            raise RuntimeError(f"发现无法自动处理的额外维度: {dim}, size={ds.sizes[dim]}")
    return ds


def find_var_name(ds: xr.Dataset, candidates) -> str | None:
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


def subset_and_rename_vars(ds: xr.Dataset, var_map: dict) -> xr.Dataset:
    out = {}

    for std_name, candidates in var_map.items():
        src_name = find_var_name(ds, candidates)
        if src_name is None:
            raise KeyError(f"找不到变量 {std_name}，候选名: {candidates}，文件变量: {list(ds.data_vars)}")
        out[std_name] = ds[src_name]

    return xr.Dataset(out, coords=ds.coords)


def crop_area(ds: xr.Dataset, area) -> xr.Dataset:
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


def load_surface_dataset(path: Path, var_map: dict, area) -> xr.Dataset:
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as raw:
        ds = raw.load()

    ds = rename_common_coords(ds, need_pressure=False)
    ds = select_first_time(ds)
    ds = subset_and_rename_vars(ds, var_map)
    ds = drop_extra_dims(ds, {"latitude", "longitude"})
    ds = crop_area(ds, area)
    ds = ds.transpose("latitude", "longitude")
    return ds.astype(np.float32)


def load_upper_dataset(path: Path, var_map: dict, area) -> xr.Dataset:
    with xr.open_dataset(path, engine="netcdf4", decode_times=True) as raw:
        ds = raw.load()

    ds = rename_common_coords(ds, need_pressure=True)
    ds = select_first_time(ds)
    ds = subset_and_rename_vars(ds, var_map)
    ds = drop_extra_dims(ds, {"pressure_level", "latitude", "longitude"})
    ds = normalize_pressure(ds)
    ds = crop_area(ds, area)
    ds = ds.transpose("pressure_level", "latitude", "longitude")
    return ds.astype(np.float32)


# ============================================================
# 5. LSM 海陆掩膜
# ============================================================

def keep_boundary_connected_water(water_mask: xr.DataArray, connectivity: int = 2) -> xr.DataArray:
    """
    只保留与裁剪区域边界连通的水体。
    用于去掉陆地内部湖泊、水库等孤立水体。
    不依赖 scipy。
    """
    values = np.asarray(water_mask.values).astype(bool)
    if values.ndim != 2:
        raise RuntimeError(f"water_mask 必须是二维数组，当前维度为 {values.shape}")

    ny, nx = values.shape
    kept = np.zeros_like(values, dtype=bool)
    q = deque()

    # 将四周边界上为 True 的水体加入队列。
    for j in range(nx):
        if values[0, j]:
            kept[0, j] = True
            q.append((0, j))
        if values[ny - 1, j] and not kept[ny - 1, j]:
            kept[ny - 1, j] = True
            q.append((ny - 1, j))

    for i in range(ny):
        if values[i, 0] and not kept[i, 0]:
            kept[i, 0] = True
            q.append((i, 0))
        if values[i, nx - 1] and not kept[i, nx - 1]:
            kept[i, nx - 1] = True
            q.append((i, nx - 1))

    if connectivity == 2:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        i, j = q.popleft()
        for di, dj in neighbors:
            ii, jj = i + di, j + dj
            if 0 <= ii < ny and 0 <= jj < nx:
                if values[ii, jj] and not kept[ii, jj]:
                    kept[ii, jj] = True
                    q.append((ii, jj))

    out = xr.DataArray(
        kept,
        coords=water_mask.coords,
        dims=water_mask.dims,
        name="ocean_mask",
        attrs={"description": "Boundary-connected ocean mask; inland isolated water removed."},
    )
    return out


def load_lsm_mask(
    lsm_path: Path,
    target_ds: xr.Dataset,
    area,
    ocean_threshold: float = 0.5,
    remove_inland_water: bool = False,
) -> xr.DataArray:
    """
    ERA5 lsm 通常：
    lsm = 1 表示陆地
    lsm = 0 表示海洋/水体

    返回 ocean_mask：
    True  表示海洋，需要保留；
    False 表示陆地，需要剔除。
    """
    with xr.open_dataset(lsm_path, engine="netcdf4", decode_times=True) as raw:
        ds = raw.load()

    ds = rename_common_coords(ds, need_pressure=False)
    ds = select_first_time(ds)

    lsm_candidates = ["lsm", "land_sea_mask", "land_binary_mask", "LANDSEA"]
    lsm_name = find_var_name(ds, lsm_candidates)

    if lsm_name is None:
        if len(ds.data_vars) == 1:
            lsm_name = list(ds.data_vars)[0]
        else:
            raise KeyError(f"无法识别 lsm 变量，文件变量: {list(ds.data_vars)}")

    lsm = xr.Dataset({"lsm": ds[lsm_name]}, coords=ds.coords)
    lsm = drop_extra_dims(lsm, {"latitude", "longitude"})
    lsm = crop_area(lsm, area)

    lsm_on_target = lsm["lsm"].interp(
        latitude=target_ds["latitude"],
        longitude=target_ds["longitude"],
        method="nearest",
    )

    water_mask = lsm_on_target <= ocean_threshold

    if remove_inland_water:
        ocean_mask = keep_boundary_connected_water(water_mask, connectivity=2)
    else:
        ocean_mask = water_mask.rename("ocean_mask")

    return ocean_mask


# ============================================================
# 6. 指标计算
# ============================================================

def calc_metrics(pred: xr.DataArray, truth: xr.DataArray, mask: xr.DataArray | None = None) -> dict:
    pred, truth = xr.align(pred, truth, join="inner")

    if mask is not None:
        mask = mask.reindex_like(truth, method=None)
        pred = pred.where(mask)
        truth = truth.where(mask)

    p = pred.values.astype(np.float64).ravel()
    o = truth.values.astype(np.float64).ravel()

    valid = np.isfinite(p) & np.isfinite(o)
    p = p[valid]
    o = o[valid]

    n = len(p)

    if n == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "bias": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "pred_mean": np.nan,
            "truth_mean": np.nan,
            "diff_std": np.nan,
        }

    diff = p - o
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))

    if n < 2 or np.nanstd(p) == 0 or np.nanstd(o) == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(p, o)[0, 1])

    return {
        "n": int(n),
        "rmse": rmse,
        "bias": bias,
        "mae": mae,
        "corr": corr,
        "pred_mean": float(np.mean(p)),
        "truth_mean": float(np.mean(o)),
        "diff_std": float(np.std(diff)),
    }


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return np.sqrt(u ** 2 + v ** 2)


def add_common_meta(
    row: dict,
    data_type: str,
    target_time: datetime,
    pred_start_time: datetime,
    lead_hour: int,
    model_forecast_hour: int,
    valid_time: datetime,
) -> dict:
    row.update({
        "data_type": data_type,
        "target_time": time_to_str(target_time),
        "pred_start_time": time_to_str(pred_start_time),
        "lead_hour": int(lead_hour),
        "model_forecast_hour": int(model_forecast_hour),
        "valid_time": time_to_str(valid_time),
    })
    return row


# ============================================================
# 7. 单时效比较
# ============================================================

def compare_one_surface(
    pred_path: Path,
    truth_path: Path,
    lsm_path: Path,
    area,
    data_type: str,
    target_time: datetime,
    pred_start_time: datetime,
    lead_hour: int,
    model_forecast_hour: int,
    valid_time: datetime,
):
    pred = load_surface_dataset(pred_path, PRED_SURFACE_VARS, area)
    truth = load_surface_dataset(truth_path, TRUE_SURFACE_VARS, area)

    pred = pred.interp(
        latitude=truth["latitude"],
        longitude=truth["longitude"],
        method="linear",
    )

    ocean_mask = load_lsm_mask(
        lsm_path=lsm_path,
        target_ds=truth,
        area=area,
        ocean_threshold=OCEAN_THRESHOLD,
        remove_inland_water=REMOVE_INLAND_WATER,
    )

    rows = []

    for var in ["msl", "u10", "v10", "t2m"]:
        m = calc_metrics(pred[var], truth[var], ocean_mask)
        row = {
            "data_group": "surface",
            "pressure_level": np.nan,
            "variable": var,
            **m,
        }
        rows.append(add_common_meta(row, data_type, target_time, pred_start_time, lead_hour, model_forecast_hour, valid_time))

    pred_ws = wind_speed(pred["u10"], pred["v10"])
    truth_ws = wind_speed(truth["u10"], truth["v10"])
    m = calc_metrics(pred_ws, truth_ws, ocean_mask)
    row = {
        "data_group": "surface",
        "pressure_level": np.nan,
        "variable": "wind10",
        **m,
    }
    rows.append(add_common_meta(row, data_type, target_time, pred_start_time, lead_hour, model_forecast_hour, valid_time))

    return rows


def compare_one_upper(
    pred_path: Path,
    truth_path: Path,
    lsm_path: Path,
    area,
    data_type: str,
    target_time: datetime,
    pred_start_time: datetime,
    lead_hour: int,
    model_forecast_hour: int,
    valid_time: datetime,
):
    pred = load_upper_dataset(pred_path, PRED_UPPER_VARS, area)
    truth = load_upper_dataset(truth_path, TRUE_UPPER_VARS, area)

    pred = pred.interp(
        pressure_level=truth["pressure_level"],
        latitude=truth["latitude"],
        longitude=truth["longitude"],
        method="linear",
    )

    ocean_mask = load_lsm_mask(
        lsm_path=lsm_path,
        target_ds=truth,
        area=area,
        ocean_threshold=OCEAN_THRESHOLD,
        remove_inland_water=REMOVE_INLAND_WATER,
    )

    rows = []

    for plev in PRESSURE_LEVELS:
        pred_level = pred.sel(pressure_level=plev)
        truth_level = truth.sel(pressure_level=plev)

        for var in ["z", "q", "t", "u", "v"]:
            m = calc_metrics(pred_level[var], truth_level[var], ocean_mask)
            row = {
                "data_group": "upper",
                "pressure_level": float(plev),
                "variable": var,
                **m,
            }
            rows.append(add_common_meta(row, data_type, target_time, pred_start_time, lead_hour, model_forecast_hour, valid_time))

        pred_ws = wind_speed(pred_level["u"], pred_level["v"])
        truth_ws = wind_speed(truth_level["u"], truth_level["v"])
        m = calc_metrics(pred_ws, truth_ws, ocean_mask)
        row = {
            "data_group": "upper",
            "pressure_level": float(plev),
            "variable": "wind",
            **m,
        }
        rows.append(add_common_meta(row, data_type, target_time, pred_start_time, lead_hour, model_forecast_hour, valid_time))

    return rows


# ============================================================
# 8. 单个 target_time 比较
# ============================================================

def run_compare_for_target(
    project_root: Path,
    data_type: str,
    target_time: datetime,
    lsm_path: Path,
    area,
    lead_hours,
    out_dir: Path | None = None,
):
    pred_start_time = get_pred_start_time(target_time, data_type)
    pred_start_str = time_to_str(pred_start_time)
    target_str = time_to_str(target_time)

    pred_base = project_root / "model_output" / data_type / pred_start_str

    if out_dir is None:
        out_dir = project_root / "comparison_results" / f"{target_str}_{data_type}_vs_ERA5_china_sea"

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"开始处理 target_time: {target_str}")
    print(f"data_type          : {data_type}")
    print(f"pred_start_time    : {pred_start_str}")
    print(f"prediction dir     : {pred_base}")
    print(f"lead_hours         : {lead_hours}")
    print(f"delay_hours        : {get_delay_hours(data_type)}")
    print(f"area               : {area}")
    print(f"lsm                : {lsm_path}")
    print(f"out_dir            : {out_dir}")
    print("=" * 100)

    all_surface_rows = []
    all_upper_rows = []
    error_rows = []

    if not pred_base.exists():
        msg = f"预测输出目录不存在: {pred_base}"
        print(msg)
        error_rows.append({
            "data_type": data_type,
            "target_time": target_str,
            "pred_start_time": pred_start_str,
            "lead_hour": np.nan,
            "model_forecast_hour": np.nan,
            "valid_time": "",
            "data_group": "all",
            "error": msg,
        })
        return all_surface_rows, all_upper_rows, error_rows

    for lead_hour in lead_hours:
        lead_hour = int(lead_hour)
        model_forecast_hour = get_model_forecast_hour(lead_hour, data_type)
        valid_time = target_time + timedelta(hours=lead_hour)
        valid_time_str = time_to_str(valid_time)

        print("\n" + "-" * 100)
        print(f"评价目标时效 lead_hour       : +{lead_hour} h")
        print(f"模型实际输出时效 model_hour  : +{model_forecast_hour} h")
        print(f"有效时刻 valid_time          : {valid_time_str}")

        try:
            pred_surface = find_pred_file(pred_base, model_forecast_hour, "surface", valid_time_str)
            true_surface = get_truth_file(project_root, valid_time_str, "surface")

            print(f"surface 预测: {pred_surface}")
            print(f"surface 真值: {true_surface}")

            rows = compare_one_surface(
                pred_path=pred_surface,
                truth_path=true_surface,
                lsm_path=lsm_path,
                area=area,
                data_type=data_type,
                target_time=target_time,
                pred_start_time=pred_start_time,
                lead_hour=lead_hour,
                model_forecast_hour=model_forecast_hour,
                valid_time=valid_time,
            )
            all_surface_rows.extend(rows)
            print(f"surface 完成: target={target_str}, lead=+{lead_hour} h, model=+{model_forecast_hour} h")

        except Exception as e:
            print(f"surface 失败: target={target_str}, lead=+{lead_hour} h, model=+{model_forecast_hour} h")
            print("错误信息:", e)
            traceback.print_exc()
            error_rows.append({
                "data_type": data_type,
                "target_time": target_str,
                "pred_start_time": pred_start_str,
                "lead_hour": lead_hour,
                "model_forecast_hour": model_forecast_hour,
                "valid_time": valid_time_str,
                "data_group": "surface",
                "error": str(e),
            })

        try:
            pred_upper = find_pred_file(pred_base, model_forecast_hour, "upper", valid_time_str)
            true_upper = get_truth_file(project_root, valid_time_str, "upper")

            print(f"upper 预测: {pred_upper}")
            print(f"upper 真值: {true_upper}")

            rows = compare_one_upper(
                pred_path=pred_upper,
                truth_path=true_upper,
                lsm_path=lsm_path,
                area=area,
                data_type=data_type,
                target_time=target_time,
                pred_start_time=pred_start_time,
                lead_hour=lead_hour,
                model_forecast_hour=model_forecast_hour,
                valid_time=valid_time,
            )
            all_upper_rows.extend(rows)
            print(f"upper 完成: target={target_str}, lead=+{lead_hour} h, model=+{model_forecast_hour} h")

        except Exception as e:
            print(f"upper 失败: target={target_str}, lead=+{lead_hour} h, model=+{model_forecast_hour} h")
            print("错误信息:", e)
            traceback.print_exc()
            error_rows.append({
                "data_type": data_type,
                "target_time": target_str,
                "pred_start_time": pred_start_str,
                "lead_hour": lead_hour,
                "model_forecast_hour": model_forecast_hour,
                "valid_time": valid_time_str,
                "data_group": "upper",
                "error": str(e),
            })

    if SAVE_PER_TARGET_CSV:
        if all_surface_rows:
            pd.DataFrame(all_surface_rows).to_csv(
                out_dir / "surface_metrics_china_sea.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if all_upper_rows:
            pd.DataFrame(all_upper_rows).to_csv(
                out_dir / "upper_metrics_china_sea.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if all_surface_rows or all_upper_rows:
            pd.concat(
                [pd.DataFrame(all_surface_rows), pd.DataFrame(all_upper_rows)],
                ignore_index=True,
            ).to_csv(out_dir / "all_metrics_china_sea.csv", index=False, encoding="utf-8-sig")
        if error_rows:
            pd.DataFrame(error_rows).to_csv(out_dir / "compare_errors.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f"target_time 处理结束: {target_str}")
    print(f"surface 结果行数: {len(all_surface_rows)}")
    print(f"upper 结果行数  : {len(all_upper_rows)}")
    print(f"错误数量        : {len(error_rows)}")
    print("=" * 100)

    return all_surface_rows, all_upper_rows, error_rows


# ============================================================
# 9. 批量比较主流程
# ============================================================

def build_default_out_root(project_root: Path, data_type: str, target_start: datetime, target_end: datetime) -> Path:
    return (
        project_root
        / "src"
        / "comparison_results"
        / f"batch_{data_type}_{target_start.strftime('%Y%m%d%H')}_{target_end.strftime('%Y%m%d%H')}_china_sea"
    )


def run_batch_compare():
    project_root = Path(PROJECT_ROOT)
    data_type = DATA_TYPE.lower().strip()

    if data_type not in ["gdas", "era5"]:
        raise ValueError("DATA_TYPE 只能是 'gdas' 或 'era5'")

    target_start = parse_datetime(TARGET_START_TIME)
    target_end = parse_datetime(TARGET_END_TIME)
    lsm_path = find_existing_lsm(project_root, LSM_PATH)

    out_root = Path(OUT_ROOT) if OUT_ROOT is not None else build_default_out_root(project_root, data_type, target_start, target_end)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("批量比较开始")
    print(f"project_root       : {project_root}")
    print(f"data_type          : {data_type}")
    print(f"target_start       : {time_to_str(target_start)}")
    print(f"target_end         : {time_to_str(target_end)}")
    print(f"run_interval_hours : {RUN_INTERVAL_HOURS}")
    print(f"lead_hours         : {LEAD_HOURS}")
    print(f"delay_hours        : {get_delay_hours(data_type)}")
    print(f"area               : {AREA}")
    print(f"lsm_path           : {lsm_path}")
    print(f"remove_inland_water: {REMOVE_INLAND_WATER}")
    print(f"out_root           : {out_root}")
    print("=" * 100)

    all_surface_rows = []
    all_upper_rows = []
    all_error_rows = []
    success_targets = []
    failed_targets = []

    current = target_start
    while current <= target_end:
        target_str = time_to_str(current)
        pred_start_time = get_pred_start_time(current, data_type)
        pred_start_str = time_to_str(pred_start_time)

        target_out_dir = out_root / f"target_{target_str}__pred_{pred_start_str}"

        surface_rows, upper_rows, error_rows = run_compare_for_target(
            project_root=project_root,
            data_type=data_type,
            target_time=current,
            lsm_path=lsm_path,
            area=AREA,
            lead_hours=LEAD_HOURS,
            out_dir=target_out_dir,
        )

        all_surface_rows.extend(surface_rows)
        all_upper_rows.extend(upper_rows)
        all_error_rows.extend(error_rows)

        # 只要当前日期有任意结果行，就认为该 target 完成；否则记为失败/缺失。
        if surface_rows or upper_rows:
            success_targets.append(target_str)
        else:
            failed_targets.append(target_str)

        current += timedelta(hours=RUN_INTERVAL_HOURS)

    # 批量总表输出
    surface_all_csv = out_root / "batch_surface_metrics_china_sea.csv"
    upper_all_csv = out_root / "batch_upper_metrics_china_sea.csv"
    all_csv = out_root / "batch_all_metrics_china_sea.csv"
    error_csv = out_root / "batch_compare_errors.csv"
    summary_csv = out_root / "batch_summary.csv"

    df_surface = pd.DataFrame(all_surface_rows)
    df_upper = pd.DataFrame(all_upper_rows)
    df_all = pd.concat([df_surface, df_upper], ignore_index=True)
    df_error = pd.DataFrame(all_error_rows)

    if not df_surface.empty:
        df_surface.to_csv(surface_all_csv, index=False, encoding="utf-8-sig")
    if not df_upper.empty:
        df_upper.to_csv(upper_all_csv, index=False, encoding="utf-8-sig")
    if not df_all.empty:
        df_all.to_csv(all_csv, index=False, encoding="utf-8-sig")
    if not df_error.empty:
        df_error.to_csv(error_csv, index=False, encoding="utf-8-sig")

    summary_rows = []
    for x in success_targets:
        summary_rows.append({"target_time": x, "status": "success"})
    for x in failed_targets:
        summary_rows.append({"target_time": x, "status": "failed_or_no_result"})
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("批量比较结束")
    print(f"成功 target 数量 : {len(success_targets)}")
    print(f"失败 target 数量 : {len(failed_targets)}")
    print(f"surface 总行数   : {len(all_surface_rows)}")
    print(f"upper 总行数     : {len(all_upper_rows)}")
    print(f"错误记录数       : {len(all_error_rows)}")
    print("\n输出文件：")
    print(f"surface 总表: {surface_all_csv}")
    print(f"upper 总表  : {upper_all_csv}")
    print(f"全部总表   : {all_csv}")
    print(f"错误日志   : {error_csv}")
    print(f"批量汇总   : {summary_csv}")
    print("=" * 100)


if __name__ == "__main__":
    run_batch_compare()