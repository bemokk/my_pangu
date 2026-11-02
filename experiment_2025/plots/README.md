# 按预报时效误差图

`plot_metrics.py` 读取：

- `Statistics/data/wind_speed_metrics_by_lead.csv`
- `Statistics/data/wind_direction_metrics_by_lead.csv`

每个变量生成 RMSE、MAE、CC 三联图。颜色固定为：

```text
E0   #43A3EF
E120 #009E73
G0   #FEA040
G120 #EF767B
```

运行：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\yolo_compare\python.exe' `
  E:\PyCharm_WorkSpace\pangu\experiment_2025\plots\plot_metrics.py
```

图片、SVG 和 PDF 输出到 `plots/figures`。

同时生成合并的六子图：`figures/wind_speed_wind_direction_metrics_by_lead.*`，上排为风速 RMSE、MAE、CC，下排为风向 RMSE、MAE、CC。

## ICOADS 年度分布图

`plot_icoads_distribution.py` 读取 `Statistics/data` 中的年度 ICOADS 网格、平台汇总和运行摘要，生成：

- `figures/icoads_2025_distribution.*`：1° 网格密度、平台年平均位置和陆地边界。

推荐使用可保存图片的环境运行：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\yolo_compare\python.exe' `
  E:\PyCharm_WorkSpace\pangu\experiment_2025\plots\plot_icoads_distribution.py
```

若环境安装了 Cartopy，会叠加海岸线和国界；未安装 Cartopy 时脚本使用经纬度坐标回退图，统计内容不变。



## 月度风向频率雷达图

`plot_direction_frequency_radar.py` 根据 `Statistics/data/wind_direction_frequency_jan_aug.csv`，生成两张 2×3 雷达图，每张包含 24/48/72/96/120 h 五个子图，第六个位置放图例：

- `figures/wind_direction_frequency_radar_january.*`
- `figures/wind_direction_frequency_radar_august.*`

运行统计和绘图：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' `
  E:\PyCharm_WorkSpace\pangu\experiment_2025\Statistics\direction_frequency.py
& 'C:\Users\SLDUO\anaconda3\envs\yolo_compare\python.exe' `
  E:\PyCharm_WorkSpace\pangu\experiment_2025\plots\plot_direction_frequency_radar.py
```

## 按观测风级的风速误差图

`plot_wind_speed_beaufort_metrics.py` 读取 `Statistics/data/wind_speed_metrics_by_beaufort_2025.csv`，分别生成 RMSE 和 MAE 两张 2×3 子图，每张图包含 24/48/72/96/120 h 五个预报时效，第六个子图放置图例：

- `figures/wind_speed_beaufort_rmse_2025.*`
- `figures/wind_speed_beaufort_mae_2025.*`

运行：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\yolo_compare\python.exe' `
  E:\PyCharm_WorkSpace\pangu\experiment_2025\plots\plot_wind_speed_beaufort_metrics.py
```

## 台风海洋站风场对比

`plot_typhoon_station_wind_timeseries.py` 只读取已经匹配好的 CSV，输出 PNG、SVG 和 PDF。
原始观测读取、时区转换、NetCDF 插值及运行摘要已移至
`Statistics/typhoon_station_wind_timeseries.py`，绘图不需要模式原始数据、NetCDF4 或 Cartopy。

已有统计 CSV 时可直接绘图：

```powershell
python E:\pyCharmProject\pangu\experiment_2025\plots\plot_typhoon_station_wind_timeseries.py `
  --matched-csv 'C:\path\typhoon_station_wind_timeseries_2025_beijing.csv' `
  --figures-root 'C:\path\figures'
```

指定 `--matched-csv` 时，从CSV记录的原始时间与UTC时间差自动识别时区解释；若显式指定了不一致的
`--timezone-mode`，会报错而不生成错误标注的图。旧命令中的 `--plot-only` 仍可使用，但已无需填写。

默认读取 `Statistics/data/typhoon_station_wind_timeseries_2025_beijing.csv`，也可以用
`--data-root` 指定统计目录，以 `--timezone-mode utc` 或 `both` 选择其他已匹配的CSV。
默认输出为 `plots/figures/typhoon_station_wind_timeseries_beijing.*`。

案例和站点从CSV自动识别，同一站点的不同台风分别绘制，并按UTC观测窗口开始时间排序。
横轴为 `plot_lead_hour` 记录的窗口经过时间，零点日期与UTC时刻取自CSV，不再假定为固定起报预报时效。
缺少CSV时，应先在有原始资料的机器上运行统计脚本；绘图入口不再接受
`--station-file`、`--output-root` 或 `--match-only`。

三例合并图可使用筛选后的CSV：

```powershell
python E:\pyCharmProject\pangu\experiment_2025\plots\plot_typhoon_station_wind_timeseries.py `
  --matched-csv E:\pyCharmProject\pangu\experiment_2025\Statistics\data\typhoon_three_cases_chronological_beijing.csv `
  --output-stem typhoon_three_cases_chronological
```

`--output-stem` 指定文件名前缀。默认各风速子图根据观测及四组预报的最小、最大值独立缩放并保留边距；
可显式使用 `--speed-max` 统一风速上限。风向按360°周期连续展开、将各曲线按整周对齐后，纵轴始终保持360°跨度，只根据各案例的数值范围上下平移，
保留所有有效点及原始缺测间断。风向纵轴为展开值，例如−90°与270°等价；原始CSV数值不变。
若风向多次转过整周，则选择跨界次数最少的共同切分角，跨界处断开连线并用小点保留样本，避免画出虚假的整周跳变。
SVG文字可编辑，PDF嵌入TrueType字体。横轴为窗口经过时间，固定起报时刻在子图标题中单独标注。
如果CSV含有 `expected_target_time_utc` 而实际起报字段不一致，对应案例会标记为预览、提示数据待更新，
不能直接作为固定起报的正式预报图。


