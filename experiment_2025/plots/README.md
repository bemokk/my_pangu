# 按预报时效误差图

`plot_metrics.py` 读取：

- `Statistics/data/wind_speed_metrics_by_lead.csv`
- `Statistics/data/wind_direction_metrics_by_lead.csv`

每个变量生成 RMSE、MAE、CC 三联图。颜色固定为：

```text
E0   #43A3EF
E120 #FEA040
G0   #009E73
G120 #EF767B
```

运行：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\yolo_compare\python.exe' `
  E:\PyCharm_WorkSpace\pangu\experiment_2025\plots\plot_metrics.py
```

图片、SVG 和 PDF 输出到 `plots/figures`。
