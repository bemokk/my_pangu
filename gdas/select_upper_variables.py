import pygrib
import numpy as np
import xarray as xr
import netCDF4 as nc
import os

grib_file = r"E:\pyCharmProject\pangu\gdas\grib2\fnl\gdas1.fnl0p25.2018030700.f00.grib2"
ENABLE_COPY_Q = True

time = grib_file.split(".")[-3]
time_str = time[0:4]+"-"+time[4:6]+"-"+time[6:8]+"-"+time[8:10]+"-00"
dir_path = os.path.join(r"/model_input/single_time_point/gdas", time_str)

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

nc_file = dir_path+"/upper.nc"
npy_file = dir_path+"/input_upper.npy"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def copy_q():
    """
    从 ERA5 upper.nc 读取 q
    返回 shape = (nplev, nlat, nlon)
    气压: 1000~50降序
    经度: 0~360 升序
    纬度: 90~-90 降序
    """

    q_path = (
        "../model_input/single_time_point/era5/"
        + f"{time[0:4]}-{time[4:6]}-{time[6:8]}-{time[8:10]}-00"
        + "/upper.nc"
    )

    if not os.path.exists(q_path):
        raise FileNotFoundError(f"ERA5 q 文件不存在: {q_path}")

    with nc.Dataset(q_path) as f:
        q = f.variables["q"][:]  # (time, plev, lat, lon)
        lat_q = f.variables["latitude"][:]
        lon_q = f.variables["longitude"][:]
        plev_q = f.variables["pressure_level"][:]

    # 去掉 time 维度
    q = q[0].astype(np.float32)  # (plev, lat, lon)

    # -------- 经度处理：0~360 --------
    lon_q = (lon_q + 360) % 360
    lon_idx = np.argsort(lon_q)
    lon_q = lon_q[lon_idx]
    q = q[:, :, lon_idx]

    # -------- 纬度处理：90 → -90 --------
    if lat_q[0] < lat_q[-1]:
        lat_q = lat_q[::-1]
        q = q[:, ::-1, :]

    result_q = {}
    for i in range(len(pressure_levels)):
        # print(q[i].shape)
        result_q[pressure_levels[i]] = q[i]

    return result_q

# 目标气压层（hPa），降序
pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# 变量映射（shortName -> 输出变量名）
VAR_MAP = {
    "gh": "z",
    "q":  "q",
    "t":  "t",
    "u":  "u",
    "v":  "v",
}

# 存储结构：data[var][plev] = 2D(lat, lon)
data = {v: {} for v in VAR_MAP.values()}

lat = lon = valid_time = None

grbs = pygrib.open(grib_file)
q_cache = None

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

if ENABLE_COPY_Q == True:
    data["q"] = copy_q()

grbs.close()


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
print("upper.nc已保存")


upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
with nc.Dataset(nc_file) as nc_file:
    upper_data[0] = nc_file.variables['z'][:].astype(np.float32)
    upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
    upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
    upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
    upper_data[4] = nc_file.variables['v'][:].astype(np.float32)

np.save(dir_path+'/input_upper.npy', upper_data)
print("upper.npy已保存")