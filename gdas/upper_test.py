import pygrib
import numpy as np
import xarray as xr
import netCDF4 as nc
import os

grib_file = r"E:\pyCharmProject\pangu\gdas\grib2\fnl\gdas1.fnl0p25.2018073000.f00.grib2"
ENABLE_COPY_Q = True

# --- 新增：存储多个日期的 q 变量的 NetCDF 文件路径 ---
MULTI_Q_NC_FILE = r"E:\pyCharmProject\pangu\model_input\multiple_time_point\era5\q_2018-07-01-05-10-15-20-25-30.nc"  # 请替换为实际的路径

# 从 GRIB 文件名解析时间
time_str_raw = grib_file.split(".")[-3]  # "2018030700"

# 构建用于保存目录的字符串格式: YYYY-MM-DD-HH-00
dir_time_str = f"{time_str_raw[0:4]}-{time_str_raw[4:6]}-{time_str_raw[6:8]}-{time_str_raw[8:10]}-00"
dir_path = os.path.join(r"/model_input/single_time_point/gdas", dir_time_str)

# 构建标准的 ISO 时间格式用于 xarray 的 time 选择: YYYY-MM-DDTHH:00:00
target_time_iso = f"{time_str_raw[0:4]}-{time_str_raw[4:6]}-{time_str_raw[6:8]}T{time_str_raw[8:10]}:00:00"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

nc_file = os.path.join(dir_path, "upper.nc")
npy_file = os.path.join(dir_path, "input_upper.npy")


def copy_q(target_time):
    """
    从包含多个时间的 NC 文件读取特定时间的 q
    返回字典结构: dict[plev] = 2D array (lat, lon)
    """
    if not os.path.exists(MULTI_Q_NC_FILE):
        raise FileNotFoundError(f"包含多个日期的 q 文件不存在: {MULTI_Q_NC_FILE}")

    # 使用 xarray 方便地进行基于时间的切片
    with xr.open_dataset(MULTI_Q_NC_FILE) as ds:
        try:
            # 将 time 改为 valid_time
            ds_sel = ds.sel(valid_time=target_time)
        except KeyError:
            raise ValueError(
                f"在 {MULTI_Q_NC_FILE} 中找不到时间 {target_time} 的数据！请检查该时间点是否存在，或时间格式是否匹配。")

        # 提取 numpy 数组
        # 假设维度名为 pressure_level, latitude, longitude
        q = ds_sel["q"].values.astype(np.float32)  # (plev, lat, lon)
        lat_q = ds_sel["latitude"].values
        lon_q = ds_sel["longitude"].values
        plev_q = ds_sel["pressure_level"].values

    # -------- 经度处理：0~360 --------
    lon_q = (lon_q + 360) % 360
    lon_idx = np.argsort(lon_q)
    lon_q = lon_q[lon_idx]
    q = q[:, :, lon_idx]

    # -------- 纬度处理：90 → -90 降序 --------
    if lat_q[0] < lat_q[-1]:
        lat_q = lat_q[::-1]
        q = q[:, ::-1, :]

    result_q = {}

    # 改进的字典组装：通过实际的气压层数值去匹配，防止 NC 文件的层级顺序/数量与你的不完全一致导致错位
    for i, p in enumerate(plev_q):
        if p in pressure_levels:
            result_q[p] = q[i]

    return result_q


# 目标气压层（hPa），降序
pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# 变量映射（shortName -> 输出变量名）
VAR_MAP = {
    "gh": "z",
    "q": "q",
    "t": "t",
    "u": "u",
    "v": "v",
}

# 存储结构：data[var][plev] = 2D(lat, lon)
data = {v: {} for v in VAR_MAP.values()}

lat = lon = valid_time = None

grbs = pygrib.open(grib_file)

for m in grbs:
    if (
            m.shortName in VAR_MAP
            and m.typeOfLevel == "isobaricInhPa"
            and m.level in pressure_levels
    ):

        out_var = VAR_MAP[m.shortName]

        vals = m.values.astype(np.float32)
        if m.shortName == "gh":
            vals *= 9.80665

        lats, lons = m.latlons()

        # 只在第一次时建立坐标
        if lat is None:
            lat = lats[:, 0]
            lon = (lons[0, :] + 360) % 360
            valid_time = np.datetime64(m.validDate)

            # 经度升序
            lon_idx = np.argsort(lon)
            lon = lon[lon_idx]
            vals = vals[:, lon_idx]

            # 纬度降序
            if lat[0] < lat[-1]:
                lat = lat[::-1]
                vals = vals[::-1, :]
        else:
            # 对后续层保持同样的顺序
            vals = vals[:, lon_idx]
            if lat[0] < lat[-1]:
                vals = vals[::-1, :]

        data[out_var][m.level] = vals

grbs.close()

# --- 调用新的 copy_q 函数，传入从 grib 提取的 ISO 时间 ---
if ENABLE_COPY_Q:
    data["q"] = copy_q(target_time_iso)

# 检查是否缺层
for vari in data:
    missing = set(pressure_levels) - set(data[vari].keys())
    if missing:
        raise RuntimeError(f"{vari} 缺少气压层: {sorted(missing)}")

# 组装为 4D 数组
ds_vars = {}
plevs = np.array(sorted(pressure_levels, reverse=True), dtype=np.int32)

for vari in data:
    arr = np.stack([data[vari][p] for p in plevs], axis=0)
    ds_vars[vari] = (("valid_time", "pressure_level", "latitude", "longitude"),
                     arr[None, ...])

# 构建 Dataset
ds = xr.Dataset(
    ds_vars,
    coords={
        "valid_time": [valid_time],
        "pressure_level": plevs,
        "latitude": lat,
        "longitude": lon,
    },
)

ds = ds.transpose("valid_time", "pressure_level", "latitude", "longitude")

# 写出 NetCDF
encoding = {v: {"dtype": "float32"} for v in ds.data_vars}
ds.to_netcdf(nc_file, format="NETCDF4", encoding=encoding)
print(f"upper.nc已保存至: {nc_file}")

# 写出 npy
upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
with nc.Dataset(nc_file) as ncf:
    upper_data[0] = ncf.variables['z'][:].astype(np.float32)
    upper_data[1] = ncf.variables['q'][:].astype(np.float32)
    upper_data[2] = ncf.variables['t'][:].astype(np.float32)
    upper_data[3] = ncf.variables['u'][:].astype(np.float32)
    upper_data[4] = ncf.variables['v'][:].astype(np.float32)

np.save(npy_file, upper_data)
print(f"upper.npy已保存至: {npy_file}")