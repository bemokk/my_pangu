
#检查 NetCDF 文件中的变量种类，并诊断每个变量的空间分辨率 (0.25 还是 0.5)。

import xarray as xr
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def inspect_nc_resolution(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"错误：找不到文件 {file_path}")
        return
    try:
        ds = xr.open_dataset(path)
        print(f"\n{'=' * 50}")
        print(f"正在诊断文件: {path.name}")
        print(f"{'=' * 50}")
        variables = list(ds.data_vars.keys())
        print(f"文件中共包含 {len(variables)} 个数据变量:")
        print(f"   {variables}\n")

        for var_name in variables:
            var_data = ds[var_name]
            dims = var_data.dims

            lat_name = next((d for d in dims if d.lower() in ['lat', 'latitude']), None)
            lon_name = next((d for d in dims if d.lower() in ['lon', 'longitude']), None)

            if lat_name and lon_name:
                lats = ds[lat_name].values
                lons = ds[lon_name].values

                if len(lats) > 1 and len(lons) > 1:
                    lat_res = round(abs(lats[1] - lats[0]), 4)
                    lon_res = round(abs(lons[1] - lons[0]), 4)
                    if lat_res == 0.25 and lon_res == 0.25:
                        status = "0.25 度"
                    elif lat_res == 0.5 and lon_res == 0.5:
                        status = "0.5 度"
                    else:
                        status = f"异常分辨率: {lat_res} x {lon_res}]"
                    print(f"变量: {var_name:<10} | 维度: {dims}")
                    print(f"空间分辨率: {lat_res}°(纬度) x {lon_res}°(经度)  {status}\n")
                else:
                    print(f"变量: {var_name:<10} | 维度: {dims}")
                    print(f"空间点数不足，无法计算步长。\n")
            else:
                print(f"变量: {var_name:<10} | 维度: {dims}")
                print(f"该变量不包含标准的经纬度坐标。\n")

        ds.close()

    except Exception as e:
        print(f"读取或解析文件出错: {e}")


if __name__ == "__main__":
    test_file = BASE_DIR / "unified_wave_data" / "unified_wind_wave_2026032812_12h.nc"

    inspect_nc_resolution(test_file)
