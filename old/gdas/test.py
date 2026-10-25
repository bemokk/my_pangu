import netCDF4 as nc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 打开nc文件
file = nc.Dataset(str(PROJECT_ROOT / "gdas" / "nc" / "处理后" / "2025072900" / "upper.nc"), 'r')

# 获取pressure_level变量
pressure_levels = file.variables['pressure_level']

# 打印pressure_level的所有值
print(f"pressure_level的形状: {pressure_levels.shape}")
print(f"pressure_level的数据类型: {pressure_levels.dtype}")
print(f"pressure_level的单位: {pressure_levels.units if hasattr(pressure_levels, 'units') else '无单位信息'}")

print("\n所有pressure_level的值:")
for i, level in enumerate(pressure_levels[:]):
    print(f"索引 {i}: {level}")

# 也可以直接打印为数组
print("\npressure_level数组:")
print(pressure_levels[:])

file.close()
