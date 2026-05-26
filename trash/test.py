import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 1. 数据准备
# ==========================================
gdas_data = [
    ('2025-08-01-00-00', 33, 141.5),
    ('2025-08-02-00-00', 37, 142.3743),
    ('2025-08-03-00-00', 40, 151.5),
    ('2025-08-04-02-00', 44, 163),
]

offcier_data = [
    ('2025-08-01-00-00', 32.5000, 141.8000),
    ('2025-08-01-03-00', 33.2000, 141.6000),
    ('2025-08-01-06-00', 33.6000, 141.5000),
    ('2025-08-01-09-00', 34.3000, 141.6000),
    ('2025-08-01-12-00', 34.5000, 141.5000),
    ('2025-08-01-15-00', 34.9000, 141.5000),
    ('2025-08-01-18-00', 35.1000, 141.6000),
    ('2025-08-01-21-00', 36.0000, 142.4000),
    ('2025-08-02-00-00', 36.4000, 142.7000),
    ('2025-08-02-03-00', 36.7000, 143.1000),
    ('2025-08-02-06-00', 37.4000, 143.9000),
    ('2025-08-02-09-00', 37.9000, 144.9000),
    ('2025-08-02-12-00', 38.3000, 145.9000),
    ('2025-08-02-15-00', 38.7000, 147.1000),
    ('2025-08-02-18-00', 38.9000, 148.1000),
    ('2025-08-02-21-00', 39.3000, 149.6000),
    ('2025-08-03-00-00', 39.7000, 151.2000),
    ('2025-08-03-03-00', 40.2000, 152.0000),
    ('2025-08-03-06-00', 40.6000, 154.3000),
    ('2025-08-03-09-00', 41.3000, 156.6000),
    ('2025-08-03-12-00', 41.3000, 157.5000),
    ('2025-08-03-15-00', 41.5000, 159.0000),
    ('2025-08-04-03-00', 42.9000, 164.1000),
]


def process_data(data):
    df = pd.DataFrame(data, columns=['timestamp', 'lat', 'lon'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H-%M')
    return df


df_gdas = process_data(gdas_data)
df_off = process_data(offcier_data)


# ==========================================
# 2. 数据平滑插值 (Interpolation)
# ==========================================
def smooth_path(df, interval='1H', kind='quadratic'):
    # 将时间转换为数值方便插值
    t_start = df['timestamp'].min()
    t_end = df['timestamp'].max()
    new_timeline = pd.date_range(start=t_start, end=t_end, freq=interval)

    x_old = (df['timestamp'] - t_start).dt.total_seconds().values
    x_new = (new_timeline - t_start).total_seconds().values

    # 对经纬度分别进行样条插值
    f_lat = interp1d(x_old, df['lat'].values, kind=kind)
    f_lon = interp1d(x_old, df['lon'].values, kind=kind)

    return pd.DataFrame({
        'timestamp': new_timeline,
        'lat': f_lat(x_new),
        'lon': f_lon(x_new)
    })


# 平滑 GDAS 数据
df_gdas_smooth = smooth_path(df_gdas, interval='1H', kind='quadratic')

# 统一时间轴
start_time = min(df_gdas_smooth['timestamp'].min(), df_off['timestamp'].min())
end_time = max(df_gdas_smooth['timestamp'].max(), df_off['timestamp'].max())
master_timeline = pd.date_range(start=start_time, end=end_time, freq='1H')

# 对齐数据（GDAS已平滑，Official进行线性填充以保证动画连贯）
df_gdas_final = df_gdas_smooth.set_index('timestamp').reindex(master_timeline).interpolate(
    method='linear').reset_index()
df_off_final = df_off.set_index('timestamp').reindex(master_timeline).interpolate(method='time').reset_index()

# ==========================================
# 3. 创建地图 (使用 Cartopy)
# ==========================================
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# 添加地图特征
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

# 自动计算显示范围 (加一些缓冲空间)
all_lons = pd.concat([df_gdas_final['lon'], df_off_final['lon']])
all_lats = pd.concat([df_gdas_final['lat'], df_off_final['lat']])
pad = 4  # 缓冲度数
ax.set_extent([all_lons.min() - pad, all_lons.max() + pad, all_lats.min() - pad, all_lats.max() + pad],
              crs=ccrs.PlateCarree())

# 添加网格线
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

# 设置标题
ax.set_title("Typhoon Path Comparison: GDAS (Smoothed) vs Official", fontsize=16, pad=20)

# ==========================================
# 4. 初始化绘图元素
# ==========================================
# transform=ccrs.PlateCarree() 确保数据按经纬度正确映射

# 1. 静态完整路径（背景轨迹）
ax.plot(df_gdas_final['lon'], df_gdas_final['lat'], 'b--', alpha=0.3, transform=ccrs.PlateCarree(),
        label='GDAS Prediction (Smoothed)')
ax.plot(df_off_final['lon'], df_off_final['lat'], 'r-', alpha=0.3, transform=ccrs.PlateCarree(), label='Official Path')

# 2. 动态移动元素
line_gdas, = ax.plot([], [], 'b--', linewidth=2.5, transform=ccrs.PlateCarree())
point_gdas, = ax.plot([], [], 'bo', markersize=9, markeredgecolor='white', transform=ccrs.PlateCarree(),
                      label='GDAS Position')

line_off, = ax.plot([], [], 'r-', linewidth=2.5, transform=ccrs.PlateCarree())
point_off, = ax.plot([], [], 'ro', markersize=9, markeredgecolor='white', transform=ccrs.PlateCarree(),
                     label='Official Position')

# 3. 时间标签
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

ax.legend(loc='lower right', fontsize=12, framealpha=0.9)


# ==========================================
# 5. 动画更新逻辑
# ==========================================
def update(frame_idx):
    current_time = master_timeline[frame_idx]

    # 获取截至当前时间的数据
    gdas_slice = df_gdas_final.iloc[:frame_idx + 1].dropna()
    off_slice = df_off_final.iloc[:frame_idx + 1].dropna()

    # 更新 GDAS
    if not gdas_slice.empty:
        line_gdas.set_data(gdas_slice['lon'], gdas_slice['lat'])
        point_gdas.set_data([gdas_slice.iloc[-1]['lon']], [gdas_slice.iloc[-1]['lat']])

    # 更新 Official
    if not off_slice.empty:
        line_off.set_data(off_slice['lon'], off_slice['lat'])
        point_off.set_data([off_slice.iloc[-1]['lon']], [off_slice.iloc[-1]['lat']])

    time_text.set_text(f"Time: {current_time.strftime('%Y-%m-%d %H:%M')}")

    return line_gdas, point_gdas, line_off, point_off, time_text


# 生成动画
print("正在生成高清地图GIF，请稍候...")
ani = animation.FuncAnimation(fig, update, frames=len(master_timeline), interval=50, blit=True)

# 保存
ani.save('typhoon_cartopy_smooth.gif', writer='pillow', fps=20)
print("完成！文件已保存为: typhoon_cartopy_smooth.gif")
plt.close()