import pygrib
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# 打开GRIB2文件
grb_file = pygrib.open(r'E:\pyCharmProject\pangu\gdas\grib2\fnl\gdas1.fnl0p25.2025072900.f00.grib2')  # 替换为你的文件名

# 方法1：根据索引选择第11条记录
u_wind = grb_file.message(580)  # 索引从1开始

# 方法2：根据条件选择（更推荐，避免索引变化）
# u_wind = grb_file.select(name='U component of wind', typeOfLevel='planetaryBoundaryLayer', level=0)[0]

# 获取数据
data = u_wind.values  # 风场数据
lat, lon = u_wind.latlons()  # 经纬度网格

# 获取数据的元信息
print(f"变量名: {u_wind.name}")
print(f"参数ID: {u_wind.parameterNumber}")
print(f"数据类型: {u_wind.typeOfLevel}")
print(f"层级: {u_wind.level}")
print(f"时间: {u_wind.validDate}")
print(f"数据范围: {data.min():.2f} 到 {data.max():.2f}")
print(f"数据形状: {data.shape}")

# 创建图形
fig = plt.figure(figsize=(14, 10))

# 设置投影（根据数据范围选择合适的投影）
# 如果是全球数据，使用PlateCarree；如果是区域数据，可以使用其他投影
ax = plt.axes(projection=ccrs.PlateCarree())

# 绘制风场U分量
# 注意：U分量是东西方向的风，通常单位是m/s
im = ax.contourf(lon, lat, data,
                 levels=50,  # 等值线数量
                 cmap='RdBu_r',  # 红蓝色系，红色为正（西风），蓝色为负（东风）
                 transform=ccrs.PlateCarree())

# 添加海岸线、国界等地理要素
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS, linewidth=0.5)

# 添加网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False  # 不显示顶部标签
gl.right_labels = False  # 不显示右侧标签
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.8)
cbar.set_label('U Component of Wind (m/s)', fontsize=12)

# 设置标题
plt.title(f'U Component of Wind (Planetary Boundary Layer)\n'
          f'Level: {u_wind.level} {u_wind.typeOfLevel}\n'
          f'Valid: {u_wind.validDate}',
          fontsize=14, pad=20)

# 设置经纬度范围（自动根据数据范围）
ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('U_wind_component.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 关闭文件
grb_file.close()

