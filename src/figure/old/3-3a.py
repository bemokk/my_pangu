import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ================= 配置区域 =================
GDAS_PATH = PROJECT_ROOT / "src" / "monthly_result" / "2018-07" / "Monthly_Avg_Surface_GDAS.csv"
ERA5_PATH = PROJECT_ROOT / "src" / "monthly_result" / "2018-07" / "Monthly_Avg_Surface_ERA5.csv"

# 提取的最大预报时效
MAX_HOUR = 24

# 对应的列名
COL_HOUR = 'forecast_hour'
COL_RMSE = 't2m_rmse'
COL_BIAS = 't2m_bias'

# ================= 数据读取与过滤 =================
df_gdas = pd.read_csv(GDAS_PATH)
df_gdas = df_gdas[df_gdas[COL_HOUR] <= MAX_HOUR].sort_values(by=COL_HOUR)

# 读取 ERA5 数据
df_era5 = pd.read_csv(ERA5_PATH)
df_era5 = df_era5[df_era5[COL_HOUR] <= MAX_HOUR].sort_values(by=COL_HOUR)

# ================= 全局绘图样式设置 =================
plt.rcParams['font.family'] = 'Arial'  # 学术常用无衬线字体
plt.rcParams['font.size'] = 8         # 全局默认字体大小
plt.rcParams['axes.labelsize'] = 10    # X轴和Y轴标签的字体大小
plt.rcParams['xtick.labelsize'] = 7   # X轴刻度数字的大小
plt.rcParams['ytick.labelsize'] = 7   # Y轴刻度数字的大小
plt.rcParams['legend.fontsize'] = 6   # 图例字体的基础大小
plt.rcParams['axes.linewidth'] = 1.2   # 坐标轴边框加粗
plt.rcParams['xtick.direction'] = 'in' # 刻度线向内
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# ================= 创建画板 =================
# 创建上下排列的两个子图，共享 X 轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)

# --------- 子图 (a): RMSE ---------
# 绘制 GDAS (红色，实线，带标记点以突出第1小时)
ax1.plot(df_gdas[COL_HOUR], df_gdas[COL_RMSE],
         color='#D62728', linewidth=1.5, linestyle='-', marker='o', markersize=5, label='GDAS (Real-time)')

# 绘制 ERA5 (蓝色，虚线)
ax1.plot(df_era5[COL_HOUR], df_era5[COL_RMSE],
         color='#1F77B4', linewidth=1, linestyle='--', marker='s', markersize=4, label='ERA5 (Lagged)')

ax1.set_ylabel('RMSE (K)', fontsize=8, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', frameon=True, edgecolor='black')

# --------- 子图 (b): Bias ---------
ax2.plot(df_gdas[COL_HOUR], df_gdas[COL_BIAS],
         color='#D62728', linewidth=1.5, linestyle='-', marker='o', markersize=5)

ax2.plot(df_era5[COL_HOUR], df_era5[COL_BIAS],
         color='#1F77B4', linewidth=1, linestyle='--', marker='s', markersize=4)

# # 绘制一条 Y=0 的基准线
# ax2.axhline(0, color='black', linewidth=1.2, linestyle='-')

ax2.set_ylabel('Bias (K)', fontsize=8, fontweight='bold')
ax2.set_xlabel('Forecast Hour (h)', fontsize=8, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

# ================= 坐标轴细节优化 =================
# 设置 X 轴的刻度：1, 3, 6, 9, 12, 15, 18, 21, 24
xticks = [1, 3, 6, 9, 12, 15, 18, 21, 24]
ax2.set_xticks(xticks)
ax2.set_xlim(0, 25)

# 调整布局，去掉上下子图之间多余的空白
plt.subplots_adjust(hspace=0.1)

# ================= 保存与显示 =================
output_path = PROJECT_ROOT / "src" / "monthly_result" / "2018-07" / "Fig3_3_Background.png"
# bbox_inches='tight' 可确保保存时边缘不被裁剪
plt.savefig(output_path, format='png', bbox_inches='tight', dpi=600)
print(f"背景图谱已成功生成并保存至: {output_path}")

plt.show()
