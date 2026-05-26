import os

import pygrib
import numpy as np
import xarray as xr
import netCDF4 as nc
import os

grib_file = r"E:\pyCharmProject\pangu\gdas\grib2\fnl\gdas1.fnl0p25.2025070100.f00.grib2"

time = grib_file.split(".")[-3]
time_str = time[0:4]+"-"+time[4:6]+"-"+time[6:8]+"-"+time[8:10]+"-00"
dir_path = os.path.join(r"/model_input/single_time_point/gdas", time_str)

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

nc_file = dir_path+"/surface.nc"
npy_file = dir_path+"/input_surface.npy"

TARGETS = [
    dict(shortName="prmsl", typeOfLevel="meanSea", level=0,
         out="msl"),
    dict(shortName="10u", typeOfLevel="heightAboveGround", level=10,
         out="u10"),
    dict(shortName="10v", typeOfLevel="heightAboveGround", level=10,
         out="v10"),
    dict(shortName="2t", typeOfLevel="heightAboveGround", level=2,
         out="t2m"),
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

            lat = lats[:, 0]
            lon = (lons[0, :] + 360) % 360

            # 经度排序
            idx = np.argsort(lon)
            lon = lon[idx]
            data = data[:, idx]

            # 纬度 90 -> -90
            if lat[0] < lat[-1]:
                lat = lat[::-1]
                data = data[::-1, :]

            found[tgt["out"]] = dict(
                data=data[None, ...],  # 加 time 维
                lat=lat,
                lon=lon,
                time=np.datetime64(m.validDate),
            )

grbs.close()

if not found:
    raise RuntimeError("没有找到任何变量")

# 构建 Dataset
ds = xr.Dataset(
    {
        k: (("valid_time", "latitude", "longitude"), v["data"])
        for k, v in found.items()
    },
    coords={
        "valid_time": [next(iter(found.values()))["time"]],
        "latitude": next(iter(found.values()))["lat"],
        "longitude": next(iter(found.values()))["lon"],
    },
)

ds = ds.transpose("valid_time", "latitude", "longitude")
ds.to_netcdf(nc_file, encoding={v: {"dtype": "float32"} for v in ds.data_vars})

print("成功写出变量:", list(ds.data_vars))


surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
with nc.Dataset(nc_file) as nc_file:
    surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
    surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
    surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
    surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
np.save(npy_file, surface_data)
print("surface.npy已保存")