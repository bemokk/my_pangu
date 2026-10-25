# 四组实验风场统计

`statistics.py` 对四组 2025 年全年实验结果进行逐观测点验证：

- 观测只读取 `icoads/manifest.json` 中 `kind=updated` 且 `format=nc` 的文件；`daily_preliminary`、`daily_total_updated` 和 DAT.GZ 不参与统计。
- 区域为 `AREA = [30, 105, 10, 130]`，即纬度 10–30°N、经度 105–130°E。
- 使用 `old/src/ne_10m_land.zip` 和 `ne_10m_lakes.zip` 的 Natural Earth 10m 陆地/湖泊多边形；陆地边界点剔除，湖泊多边形内按水面保留。
- 模型 u10/v10 在 ICOADS 观测位置做双线性插值。
- 风速由 `sqrt(u10**2 + v10**2)` 计算；风向采用气象风向（风从哪里吹来）。
- 风向误差使用 ±180° 环形差，风向 CC 使用圆形相关系数。
- 月份按观测有效时间 `valid_time` 统计，预报时效为 0–120 h、每 3 h。

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
- `missing_outputs.csv`

每个指标文件包含 `experiment`、分组字段、`n`、`rmse`、`mae` 和 `cc`。
四个实验的独立中间结果保存在 `data/_per_experiment`，便于断点检查和重新合并。
