import os
import pygrib
import numpy as np
import xarray as xr
import netCDF4 as nc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--grib_file", required=True, help="输入 GRIB2 文件路径")
args = parser.parse_args()

grib_file = args.grib_file

time = os.path.basename(grib_file).split(".")[-3]
time_str = time[0:4] + "-" + time[4:6] + "-" + time[6:8] + "-" + time[8:10] + "-00"

base_dir = r"E:\PyCharm_WorkSpace\pangu\model_input\single_time_point\gdas"
dir_path = os.path.join(base_dir, time_str)

os.makedirs(dir_path, exist_ok=True)

nc_path = os.path.join(dir_path, "upper.nc")
npy_path = os.path.join(dir_path, "input_upper.npy")

# 目标气压层，单位 hPa，降序
pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# 变量映射
VAR_MAP = {
    "gh": "z",
    "q":  "q",
    "t":  "t",
    "u":  "u",
    "v":  "v",
}

# 存储结构：data[var][plev] = 2D(lat, lon)
data = {v: {} for v in VAR_MAP.values()}

lat = None
lon = None
valid_time = None
lon_idx = None
reverse_lat = False

grbs = pygrib.open(grib_file)

for m in grbs:
    if (
        m.shortName in VAR_MAP
        and m.typeOfLevel == "isobaricInhPa"
        and m.level in pressure_levels
    ):

        out_var = VAR_MAP[m.shortName]

        vals = m.values.astype(np.float32)

        # gh 是位势高度，Pangu 需要位势 z，所以乘以重力加速度
        if m.shortName == "gh":
            vals *= 9.80665

        lats, lons = m.latlons()

        # 第一次建立统一坐标和排序索引
        if lat is None:
            lat = lats[:, 0].astype(np.float32)
            lon = ((lons[0, :] + 360) % 360).astype(np.float32)
            valid_time = np.datetime64(m.validDate)

            lon_idx = np.argsort(lon)
            lon = lon[lon_idx]

            if lat[0] < lat[-1]:
                reverse_lat = True
                lat = lat[::-1]

        # 所有变量、所有气压层使用同样的经纬度处理
        vals = vals[:, lon_idx]

        if reverse_lat:
            vals = vals[::-1, :]

        data[out_var][m.level] = vals

grbs.close()

# 检查是否缺变量或缺气压层
for vari in data:
    missing = set(pressure_levels) - set(data[vari].keys())
    if missing:
        raise RuntimeError(f"{vari} 缺少气压层: {sorted(missing)}")

if lat is None or lon is None or valid_time is None:
    raise RuntimeError("没有从 GRIB 文件中读取到任何有效的高空变量")

# 组装 Dataset
plevs = np.array(pressure_levels, dtype=np.int32)

ds_vars = {}

for vari in ["z", "q", "t", "u", "v"]:
    arr = np.stack([data[vari][p] for p in plevs], axis=0).astype(np.float32)
    ds_vars[vari] = (
        ("valid_time", "pressure_level", "latitude", "longitude"),
        arr[None, :, :, :]
    )

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

# 明确写为 NETCDF4 格式
encoding = {var: {"dtype": "float32"} for var in ds.data_vars}

ds.to_netcdf(
    nc_path,
    engine="netcdf4",
    format="NETCDF4",
    encoding=encoding
)

ds.close()

print("upper.nc 已保存:", nc_path)
print("变量:", list(ds_vars.keys()))

# 检查文件是否正常生成
if not os.path.exists(nc_path):
    raise RuntimeError("upper.nc 没有成功生成")

if os.path.getsize(nc_path) == 0:
    raise RuntimeError("upper.nc 文件大小为 0，写出失败")

# 读取 upper.nc 并保存 input_upper.npy
upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)

with nc.Dataset(nc_path, "r") as ds_nc:
    upper_data[0] = ds_nc.variables["z"][0, :, :, :].astype(np.float32)
    upper_data[1] = ds_nc.variables["q"][0, :, :, :].astype(np.float32)
    upper_data[2] = ds_nc.variables["t"][0, :, :, :].astype(np.float32)
    upper_data[3] = ds_nc.variables["u"][0, :, :, :].astype(np.float32)
    upper_data[4] = ds_nc.variables["v"][0, :, :, :].astype(np.float32)

np.save(npy_path, upper_data)

print("input_upper.npy 已保存:", npy_path)
print("upper_data shape:", upper_data.shape)