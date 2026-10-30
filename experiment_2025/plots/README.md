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


