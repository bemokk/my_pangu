import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. 用户需要修改的路径
# ============================================================

# 你打包下载的 ERA5 nc 文件所在文件夹
# 这个文件夹里可以有多个 nc 文件，有些是 surface，有些是 upper
INPUT_NC_DIR = "multi_time_point"

# 拆分后的输出路径
# 会自动生成：
# E:\PyCharm_WorkSpace\pangu\model_input\single_time_point\era5\2025-07-01-00-00
OUTPUT_BASE_DIR = "single_time_point\era5"

# 是否覆盖已有的 surface.nc / upper.nc / npy 文件
OVERWRITE = True

# 如果你还想额外保存 surface.npy / upper.npy，可以改成 True
# 默认仍然保存为你原工作流使用的 input_surface.npy / input_upper.npy
WRITE_SHORT_NPY_ALIAS = False


# ============================================================
# 2. 变量顺序，必须与盘古输入保持一致
# ============================================================

SURFACE_ORDER = ["msl", "u10", "v10", "t2m"]
UPPER_ORDER = ["z", "q", "t", "u", "v"]

PRESSURE_LEVELS = np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    dtype=np.int32
)

# 兼容短变量名和长变量名
SURFACE_VAR_CANDIDATES = {
    "msl": [
        "msl",
        "mean_sea_level_pressure",
    ],
    "u10": [
        "u10",
        "10m_u_component_of_wind",
    ],
    "v10": [
        "v10",
        "10m_v_component_of_wind",
    ],
    "t2m": [
        "t2m",
        "2m_temperature",
    ],
}

UPPER_VAR_CANDIDATES = {
    "z": [
        "z",
        "geopotential",
    ],
    "q": [
        "q",
        "specific_humidity",
    ],
    "t": [
        "t",
        "temperature",
    ],
    "u": [
        "u",
        "u_component_of_wind",
    ],
    "v": [
        "v",
        "v_component_of_wind",
    ],
}


# ============================================================
# 3. 工具函数
# ============================================================

def find_var_name(ds, candidates):
    """
    在 Dataset 中查找变量名。
    优先按变量名匹配，其次按 attrs 中的 shortName / GRIB_shortName 匹配。
    """
    for name in candidates:
        if name in ds.data_vars:
            return name

    candidates_lower = [c.lower() for c in candidates]

    for var in ds.data_vars:
        attrs = ds[var].attrs
        attr_values = [
            str(attrs.get("shortName", "")).lower(),
            str(attrs.get("GRIB_shortName", "")).lower(),
            str(attrs.get("standard_name", "")).lower(),
            str(attrs.get("long_name", "")).lower(),
        ]

        for value in attr_values:
            if value in candidates_lower:
                return var

    return None


def find_coord_name(ds, candidates, required=True):
    """
    查找坐标名或维度名。
    常见 ERA5 文件中可能是：
    time / valid_time
    latitude / lat
    longitude / lon
    pressure_level / level
    """
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name

    if required:
        raise RuntimeError(f"找不到坐标: {candidates}")

    return None


def normalize_common_coords(ds, need_pressure=False):
    """
    统一坐标名称和方向：
    time/valid_time -> valid_time
    lat -> latitude
    lon -> longitude
    level -> pressure_level

    同时保证：
    latitude: 90 -> -90
    longitude: 0 -> 359.75
    pressure_level: 1000 -> 50
    """

    rename_dict = {}

    time_name = find_coord_name(ds, ["valid_time", "time"], required=True)
    lat_name = find_coord_name(ds, ["latitude", "lat"], required=True)
    lon_name = find_coord_name(ds, ["longitude", "lon"], required=True)

    if time_name != "valid_time":
        rename_dict[time_name] = "valid_time"

    if lat_name != "latitude":
        rename_dict[lat_name] = "latitude"

    if lon_name != "longitude":
        rename_dict[lon_name] = "longitude"

    if need_pressure:
        plev_name = find_coord_name(
            ds,
            ["pressure_level", "level", "isobaricInhPa"],
            required=True
        )

        if plev_name != "pressure_level":
            rename_dict[plev_name] = "pressure_level"

    if rename_dict:
        ds = ds.rename(rename_dict)

    # 如果 valid_time 不是维度，而只是标量坐标，则扩展成维度
    if "valid_time" not in ds.dims:
        if "valid_time" in ds.coords:
            time_value = ds["valid_time"].values
        else:
            raise RuntimeError("valid_time 不是维度，且找不到有效时间坐标")

        ds = ds.expand_dims(valid_time=[time_value])

    # 经度转为 0 ~ 360 并升序排列
    lon_values = ((ds["longitude"] + 360) % 360).astype(np.float32)
    ds = ds.assign_coords(longitude=lon_values)
    ds = ds.sortby("longitude")

    # 纬度保证为 90 -> -90
    ds = ds.sortby("latitude", ascending=False)

    if need_pressure:
        ds = ds.assign_coords(
            pressure_level=ds["pressure_level"].astype(np.int32)
        )
        ds = ds.sortby("pressure_level", ascending=False)

    return ds


def drop_extra_dims(ds, allowed_dims):
    """
    处理 ERA5 文件中可能存在的额外维度，例如 expver。
    对 expver 使用 max(skipna=True) 合并。
    其他长度为 1 的维度直接 squeeze。
    """
    for dim in list(ds.dims):
        if dim in allowed_dims:
            continue

        if dim == "expver":
            ds = ds.max(dim="expver", skipna=True)
        elif ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
        else:
            raise RuntimeError(
                f"发现无法自动处理的额外维度: {dim}, size={ds.sizes[dim]}"
            )

    return ds


def build_surface_dataset(raw_ds):
    """
    从原始 nc 文件中提取 surface 变量，并标准化为：
    valid_time, latitude, longitude
    """
    var_map = {}

    for out_name, candidates in SURFACE_VAR_CANDIDATES.items():
        src_name = find_var_name(raw_ds, candidates)
        if src_name is None:
            return None
        var_map[out_name] = src_name

    ds = xr.Dataset(
        {
            out_name: raw_ds[src_name]
            for out_name, src_name in var_map.items()
        }
    )

    ds = normalize_common_coords(ds, need_pressure=False)
    ds = drop_extra_dims(
        ds,
        allowed_dims={"valid_time", "latitude", "longitude"}
    )

    ds = ds[SURFACE_ORDER]
    ds = ds.transpose("valid_time", "latitude", "longitude")
    ds = ds.astype(np.float32)

    return ds


def build_upper_dataset(raw_ds):
    """
    从原始 nc 文件中提取 upper 变量，并标准化为：
    valid_time, pressure_level, latitude, longitude
    """
    var_map = {}

    for out_name, candidates in UPPER_VAR_CANDIDATES.items():
        src_name = find_var_name(raw_ds, candidates)
        if src_name is None:
            return None
        var_map[out_name] = src_name

    ds = xr.Dataset(
        {
            out_name: raw_ds[src_name]
            for out_name, src_name in var_map.items()
        }
    )

    ds = normalize_common_coords(ds, need_pressure=True)
    ds = drop_extra_dims(
        ds,
        allowed_dims={"valid_time", "pressure_level", "latitude", "longitude"}
    )

    available_plevs = set(int(x) for x in ds["pressure_level"].values)
    missing_plevs = [int(p) for p in PRESSURE_LEVELS if int(p) not in available_plevs]

    if missing_plevs:
        raise RuntimeError(f"upper 文件缺少气压层: {missing_plevs}")

    ds = ds.sel(pressure_level=PRESSURE_LEVELS)
    ds = ds[UPPER_ORDER]
    ds = ds.transpose("valid_time", "pressure_level", "latitude", "longitude")
    ds = ds.astype(np.float32)

    return ds


def time_to_folder_name(time_value):
    """
    np.datetime64 / pandas Timestamp -> YYYY-MM-DD-HH-MM
    """
    dt = pd.to_datetime(time_value)
    return dt.strftime("%Y-%m-%d-%H-%M")


def write_single_surface(ds_one, out_dir):
    """
    写出单时次 surface.nc 和 input_surface.npy
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_path = out_dir / "surface.nc"
    npy_path = out_dir / "input_surface.npy"

    if OVERWRITE or not nc_path.exists():
        encoding = {var: {"dtype": "float32"} for var in ds_one.data_vars}
        ds_one.to_netcdf(
            nc_path,
            engine="netcdf4",
            format="NETCDF4",
            encoding=encoding
        )

    surface_data = np.zeros((4, 721, 1440), dtype=np.float32)

    for i, var in enumerate(SURFACE_ORDER):
        arr = (
            ds_one[var]
            .isel(valid_time=0)
            .transpose("latitude", "longitude")
            .values
            .astype(np.float32)
        )

        if arr.shape != (721, 1440):
            raise RuntimeError(
                f"{var} shape 异常: {arr.shape}, 期望为 (721, 1440)"
            )

        surface_data[i] = arr

    if OVERWRITE or not npy_path.exists():
        np.save(npy_path, surface_data)

    if WRITE_SHORT_NPY_ALIAS:
        np.save(out_dir / "surface.npy", surface_data)


def write_single_upper(ds_one, out_dir):
    """
    写出单时次 upper.nc 和 input_upper.npy
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_path = out_dir / "upper.nc"
    npy_path = out_dir / "input_upper.npy"

    if OVERWRITE or not nc_path.exists():
        encoding = {var: {"dtype": "float32"} for var in ds_one.data_vars}
        ds_one.to_netcdf(
            nc_path,
            engine="netcdf4",
            format="NETCDF4",
            encoding=encoding
        )

    upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)

    for i, var in enumerate(UPPER_ORDER):
        arr = (
            ds_one[var]
            .isel(valid_time=0)
            .transpose("pressure_level", "latitude", "longitude")
            .values
            .astype(np.float32)
        )

        if arr.shape != (13, 721, 1440):
            raise RuntimeError(
                f"{var} shape 异常: {arr.shape}, 期望为 (13, 721, 1440)"
            )

        upper_data[i] = arr

    if OVERWRITE or not npy_path.exists():
        np.save(npy_path, upper_data)

    if WRITE_SHORT_NPY_ALIAS:
        np.save(out_dir / "upper.npy", upper_data)


def process_dataset_by_time(ds, kind, source_file):
    """
    将一个 Dataset 按 valid_time 拆分。
    """
    times = np.atleast_1d(ds["valid_time"].values)

    count = 0

    for time_value in times:
        folder_name = time_to_folder_name(time_value)
        out_dir = Path(OUTPUT_BASE_DIR) / folder_name

        ds_one = ds.sel(valid_time=[time_value])

        print(f"  正在写出 {kind}: {folder_name}")

        if kind == "surface":
            write_single_surface(ds_one, out_dir)
        elif kind == "upper":
            write_single_upper(ds_one, out_dir)
        else:
            raise ValueError(f"未知类型: {kind}")

        count += 1

    print(f"  {source_file} 中 {kind} 拆分完成，共 {count} 个时次")
    return count


def process_one_nc_file(nc_path):
    """
    处理单个 nc 文件：
    可能是 surface 文件，也可能是 upper 文件。
    如果一个文件里同时包含 surface 和 upper，也可以同时处理。
    """
    print("=" * 100)
    print(f"正在读取: {nc_path}")

    surface_count = 0
    upper_count = 0

    with xr.open_dataset(nc_path, engine="netcdf4", decode_times=True) as raw_ds:

        surface_ds = build_surface_dataset(raw_ds)
        if surface_ds is not None:
            print("  检测到 surface 变量")
            surface_count = process_dataset_by_time(
                surface_ds,
                kind="surface",
                source_file=nc_path
            )

        upper_ds = build_upper_dataset(raw_ds)
        if upper_ds is not None:
            print("  检测到 upper 变量")
            upper_count = process_dataset_by_time(
                upper_ds,
                kind="upper",
                source_file=nc_path
            )

    if surface_count == 0 and upper_count == 0:
        print("  未检测到完整的 surface 或 upper 变量，跳过该文件")

    return surface_count, upper_count


def main():
    input_dir = Path(INPUT_NC_DIR)
    output_dir = Path(OUTPUT_BASE_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(input_dir.rglob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"在目录中没有找到 nc 文件: {INPUT_NC_DIR}")

    print(f"共找到 {len(nc_files)} 个 nc 文件")
    print(f"输入目录: {INPUT_NC_DIR}")
    print(f"输出目录: {OUTPUT_BASE_DIR}")

    total_surface = 0
    total_upper = 0
    failed_files = []

    for nc_path in nc_files:
        try:
            surface_count, upper_count = process_one_nc_file(nc_path)
            total_surface += surface_count
            total_upper += upper_count

        except Exception as e:
            print(f"处理失败: {nc_path}")
            print("错误信息:", e)
            traceback.print_exc()
            failed_files.append(str(nc_path))

    print("\n" + "=" * 100)
    print("全部处理结束")
    print(f"成功拆分 surface 时次数量: {total_surface}")
    print(f"成功拆分 upper 时次数量: {total_upper}")
    print(f"失败文件数量: {len(failed_files)}")

    if failed_files:
        print("\n失败文件列表:")
        for f in failed_files:
            print(f)


if __name__ == "__main__":
    main()