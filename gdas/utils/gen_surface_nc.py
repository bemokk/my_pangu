import xarray as xr
import numpy as np
import os
import netCDF4 as nc

# ---------- 配置 ----------
src = "nc/gdas/2018091100.nc"  # 原始文件路径，按需修改
dst = "nc/处理后/surface.nc"  # 输出文件

surface_names_requested = {
    "msl": "PRMSL_meansealevel",
    "u10": "UGRD_10maboveground",
    "v10": "VGRD_10maboveground",
    "t2m": "TMP_2maboveground",
    "sp": "PRES_surface"
}

open_kwargs = {}
# ---------- end config ----------

# 打开数据集
ds = xr.open_dataset(src, **open_kwargs)
lat_name = 'latitude'
lon_name = 'longitude'




# 确认要取出的原始变量在文件中是否存在
orig_vars = list(surface_names_requested.values())
vars_present = [v for v in orig_vars if v in ds.data_vars]
vars_missing = [v for v in orig_vars if v not in ds.data_vars]
if vars_missing:
    print("警告：以下原始变量在 nc 中未找到，将被跳过：", vars_missing)
if not vars_present:
    raise RuntimeError("没有找到任何待提取的变量，终止。")

# 保留要写入的新数据中的坐标（纬度、经度、time（如果存在））
keep_coords = [lat_name, lon_name]
if 'time' in ds.coords:
    keep_coords.append('time')

# 取子集（只保留需要的变量和坐标）
ds_sub = ds[vars_present + keep_coords]
print("子集变量：", list(ds_sub.data_vars))


# 重命名：把原始名 -> 目标名（xarray.rename 接受 dict old_name:new_name）
rename_map = {orig: new for new, orig in surface_names_requested.items() if orig in vars_present}
ds_sub = ds_sub.rename(rename_map)
print("重命名映射（old->new）：", rename_map)

# 翻转纬度：如果纬度是升序（-90..90），则翻转成降序（90..-90）
lat_vals = ds_sub[lat_name].values
# 处理可能的 mask / object 类型，确保能比较
if np.asarray(lat_vals).size == 0:
    raise RuntimeError("纬度坐标为空")
first = float(np.asarray(lat_vals).flat[0])
last  = float(np.asarray(lat_vals).flat[-1])
if first < last:
    print("检测到纬度为升序（-90 -> 90），正在翻转为降序（90 -> -90）...")
    ds_sub = ds_sub.isel({lat_name: slice(None, None, -1)})  # 按索引翻转
else:
    print("纬度已是降序（90 -> -90），无需翻转。")

# 更新 latitude 属性（可选）
if 'attrs' in ds_sub[lat_name].__dir__():
    ds_sub[lat_name].attrs['axis'] = 'Y'

# 为输出变量设置压缩编码（可根据需要调整）
encoding = {}
for var in ds_sub.data_vars:
    # 不对坐标变量设置压缩（xarray 会自动处理 coords）
    encoding[var] = {"zlib": True, "complevel": 4}

# 写入 netCDF
if os.path.exists(dst):
    print("注意：目标文件已存在，将被覆盖：", dst)
ds_sub.to_netcdf(dst, format='NETCDF4', encoding=encoding)
print("写入完成：", dst)



# Convert the surface data to npy
surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
with nc.Dataset('nc/处理后/surface.nc') as nc_file:
    surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
    surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
    surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
    surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
np.save('../nc/处理后/2018091100/input_surface.npy', surface_data)