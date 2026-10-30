# 四组实验风场统计

`statistics.py` 对四组 2025 年全年实验结果进行逐观测点验证：

- 观测只读取 `icoads/manifest.json` 中 `kind=updated` 且 `format=nc` 的文件；`daily_preliminary`、`daily_total_updated` 和 DAT.GZ 不参与统计。
- 区域为 `AREA = [42, 103, 13, 130]`，即纬度 13–42°N、经度 103–130°E。
- 使用 `old/src/ne_10m_land.zip` 和 `ne_10m_lakes.zip` 的 Natural Earth 10m 陆地/湖泊多边形；陆地边界点剔除，湖泊多边形内按水面保留。
- 模型 u10/v10 在 ICOADS 观测位置做双线性插值。
- 风速由 `sqrt(u10**2 + v10**2)` 计算；风向采用气象风向（风从哪里吹来）。
- 风向误差使用 ±180° 环形差，风向 CC 使用圆形相关系数。
- 月份按观测有效时间 `valid_time` 统计，预报时效为 0–120 h、每 3 h。
- 风速和风向的 RMSE、MAE、CC 都先按每个目标日、每个预报时效计算，再对有效日期等权取算术平均。RMSE 在每天内部先开平方，再对每日 RMSE 求平均。
- `n` 仍为观测配对总数；`rmse_daily_count`、`mae_daily_count`、`cc_daily_count` 分别记录各指标参与平均的有效日数。RMSE/MAE 至少需要一条有效配对，CC 至少需要两条且相关系数有定义；各指标独立排除无效日。
- `by_lead_month` 按有效时间月份、固定时效平均逐日指标；`by_month` 对该月份全部有效的“目标日×时效”指标等权平均，其计数为日时效案例数。

运行单个实验：

```powershell
$env:PYTHONPATH = 'E:\PyCharm_WorkSpace\pangu'
python E:\PyCharm_WorkSpace\pangu\experiment_2025\Statistics\statistics.py `
  --experiments gdas_direct --workers 1
```

四组独立运行后合并：

```powershell
python E:\PyCharm_WorkSpace\pangu\experiment_2025\Statistics\merge_statistics.py
```

最终 CSV 位于 `Statistics/data`：

- `wind_speed_metrics_by_lead.csv`
- `wind_speed_metrics_by_month.csv`
- `wind_speed_metrics_by_lead_month.csv`
- `wind_direction_metrics_by_lead.csv`
- `wind_direction_metrics_by_month.csv`
- `wind_direction_metrics_by_lead_month.csv`
- `daily_metrics_by_target.csv`：每个目标日、预报时效和实验的风速/风向 RMSE、MAE、CC；
- `daily_cc_by_target.csv`：保留兼容文件名，内容与 `daily_metrics_by_target.csv` 相同；
- `missing_outputs.csv`

每个指标文件包含 `experiment`、分组字段、`n`、`rmse`、`mae` 和 `cc`。
四个实验的独立中间结果保存在 `data/_per_experiment`，便于断点检查和重新合并。

## 全年 ICOADS 记录分布

`annual_icoads_distribution.py` 按同一 `daily_updated` 口径扫描 2025 年 ICOADS NetCDF，使用 `[30, 105, 10, 130]` 区域和陆海掩膜，仅将有有效风向（1–360°）、风速（0–75 m/s）的海上记录计入空间分布。结果包括：

- `icoads_2025_grid_counts_1deg.csv`：1° 网格记录数；
- `icoads_2025_platform_summary.csv`：平台年记录数、平均位置、位置范围和平台类型；
- `icoads_2025_monthly_counts.csv`：逐月原始区域记录、海上记录和有效风场记录；
- `icoads_distribution_run_summary.json`：文件数、记录数、掩膜剔除数和运行口径。

运行：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' `
  E:\PyCharm_WorkSpace\pangu\experiment_2025\Statistics\annual_icoads_distribution.py
```


`merge_statistics.py` 合并前会检查各 `_per_experiment` 的 `AREA` 是否一致；本次全年统计直接由 `statistics.py` 写入 `Statistics/data`，避免旧区域中间结果覆盖新结果。

## 1 月和 8 月风向频率

`direction_frequency.py` 使用 16 方位扇区，按 00 UTC 起报的 24/48/72/96/120 h 预报，统计 1 月和 8 月的 ICOADS 观测及四组实验风向频率。输出：

- `data/wind_direction_frequency_jan_aug.csv`；
- `data/wind_direction_frequency_jan_aug_summary.json`。

仅使用 `daily_updated`、区域 `[42, 103, 13, 130]` 和陆海掩膜。

