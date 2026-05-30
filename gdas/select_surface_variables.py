import os
import pygrib
import numpy as np
import xarray as xr
import netCDF4 as nc
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--grib_file", required=True, help="输入 GRIB2 文件路径")
args = parser.parse_args()

grib_file = args.grib_file

time = os.path.basename(grib_file).split(".")[-3]
time_str = time[0:4] + "-" + time[4:6] + "-" + time[6:8] + "-" + time[8:10] + "-00"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
base_dir = PROJECT_ROOT / "model_input" / "single_time_point" / "gdas"
dir_path = base_dir / time_str

os.makedirs(dir_path, exist_ok=True)

nc_path = os.path.join(dir_path, "surface.nc")
npy_path = os.path.join(dir_path, "input_surface.npy")

TARGETS = [
    dict(shortName="prmsl", typeOfLevel="meanSea", level=0, out="msl"),
    dict(shortName="10u", typeOfLevel="heightAboveGround", level=10, out="u10"),
    dict(shortName="10v", typeOfLevel="heightAboveGround", level=10, out="v10"),
    dict(shortName="2t", typeOfLevel="heightAboveGround", level=2, out="t2m"),
]

grbs = pygrib.open(grib_file)
found = {}

for m in grbs:
    for tgt in TARGETS:
        if (
            m.shortName == tgt["shortName"]
            and m.typeOfLevel == tgt["typeOfLevel"]
            and m.level == tgt["level"]
            and tgt["out"] not in found
        ):
            data = m.values.astype(np.float32)
            lats, lons = m.latlons()

            lat = lats[:, 0].astype(np.float32)
            lon = ((lons[0, :] + 360) % 360).astype(np.float32)

            # 经度排序为 0 ~ 360
            idx = np.argsort(lon)
            lon = lon[idx]
            data = data[:, idx]

            # 纬度保证为 90 -> -90
            if lat[0] < lat[-1]:
                lat = lat[::-1]
                data = data[::-1, :]

            found[tgt["out"]] = dict(
                data=data[None, :, :],
                lat=lat,
                lon=lon,
                time=np.datetime64(m.validDate),
            )

grbs.close()

# 检查是否四个变量都找到了
required_vars = ["msl", "u10", "v10", "t2m"]
missing = [v for v in required_vars if v not in found]

if missing:
    raise RuntimeError(f"以下变量没有找到: {missing}")

# 构建 Dataset
ref = found[required_vars[0]]

ds = xr.Dataset(
    {
        var: (("valid_time", "latitude", "longitude"), found[var]["data"])
        for var in required_vars
    },
    coords={
        "valid_time": [ref["time"]],
        "latitude": ref["lat"],
        "longitude": ref["lon"],
    },
)

ds = ds.transpose("valid_time", "latitude", "longitude")

# 明确写成 NETCDF4 格式
ds.to_netcdf(
    nc_path,
    engine="netcdf4",
    format="NETCDF4",
    encoding={var: {"dtype": "float32"} for var in ds.data_vars}
)

ds.close()

print("成功写出 NetCDF 文件:", nc_path)
print("成功写出变量:", list(required_vars))

# 检查文件是否存在且不是空文件
if not os.path.exists(nc_path):
    raise RuntimeError("surface.nc 没有成功生成")

if os.path.getsize(nc_path) == 0:
    raise RuntimeError("surface.nc 文件大小为 0，写出失败")

# 读取 nc 并保存 npy
surface_data = np.zeros((4, 721, 1440), dtype=np.float32)

with nc.Dataset(nc_path, "r") as ds_nc:
    surface_data[0] = ds_nc.variables["msl"][0, :, :].astype(np.float32)
    surface_data[1] = ds_nc.variables["u10"][0, :, :].astype(np.float32)
    surface_data[2] = ds_nc.variables["v10"][0, :, :].astype(np.float32)
    surface_data[3] = ds_nc.variables["t2m"][0, :, :].astype(np.float32)

np.save(npy_path, surface_data)

print("input_surface.npy 已保存:", npy_path)
print("surface_data shape:", surface_data.shape)
