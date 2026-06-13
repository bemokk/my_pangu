# Buoy 绘图脚本中文化与字号配置设计

日期：2026-06-13

## 目标

统一修改以下六个绘图脚本，使图中文字默认放大 25%，支持人工调整，并将适合中文表达的图中文字与实验名称中文化：

- `Buoy/plots/plot_china_sea_hex_counts.py`
- `Buoy/plots/plot_wind_speed_metrics_figure2.py`
- `Buoy/plots/plot_wind_speed_beaufort_metrics.py`
- `Buoy/plots/plot_wind_speed_beaufort_sample_counts.py`
- `Buoy/plots/plot_spatial_hex_best_rmse.py`
- `Buoy/plots/plot_wipha_track_forecast_error.py`

同时删除各图的大标题和底部说明小字，保留必要的子图标题、坐标轴标签、图例、色标和数据标注。

## 配置方式

每个脚本独立提供以下顶部配置，便于直接人工修改，避免引入共享模块后影响其他绘图程序：

```python
FONT_SCALE = 1.25
FONT_FAMILY = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
TEXT_LABELS = {
    # 脚本使用的中文显示文字
}
```

每个脚本还应保留一组基础字号，最终字号由基础字号乘以 `FONT_SCALE` 得到。人工调整方式为：

- 整体放大或缩小：修改 `FONT_SCALE`
- 单独调整某类文字：修改对应基础字号
- 修改实验名称或其他图中文字：修改 `TEXT_LABELS`
- 修改中文字体候选顺序：修改 `FONT_FAMILY`

Matplotlib 字体配置使用无衬线字体候选列表，并设置 `axes.unicode_minus = False`，减少中文字体和负号显示问题。

## 中文化规则

适合中文替代的文字使用中文；专业缩写、单位和通用符号保持原样。

推荐实验名称：

- `ERA5 realtime` → `ERA5实时场`
- `ERA5 lagged 5d forecast` → `ERA5延迟5天预报`
- `GDAS forecast` → `GDAS实时预报`
- `Observed track` → `观测路径`

推荐常用标签：

- `Forecast lead time (h)` → `预报时效（h）`
- `Track error (km)` → `路径误差（km）`
- `Valid sample count` → `有效样本数`
- `Observed Beaufort wind force` → `观测蒲福风力等级`
- `Correlation coefficient` → `相关系数`
- `Record count per hexagon` → `每个六边形网格的记录数`
- `Matched but below sample threshold` → `匹配样本不足阈值`
- `No matched sample` → `无匹配样本`

保留以下形式：

- `RMSE`、`MAE`、`CC`
- `m s$^{-1}$`、`km`、`h`
- 子图编号 `(a)`、`(b)`、`(c)`

## 各脚本修改范围

### plot_china_sea_hex_counts.py

- 增加独立字号、字体和中文标签配置。
- 放大坐标、色标和六边形数字标注。
- 色标标签改为中文。
- 删除整图大标题和底部说明小字。
- 不修改六边形统计、地图范围和输出路径。

### plot_wind_speed_metrics_figure2.py

- 增加独立字号、字体和中文标签配置。
- 图例中的三组实验名称中文化。
- 横轴标签和相关系数纵轴标签中文化。
- 删除整图大标题和底部说明小字。
- 保留三个子图及其现有排列、数据统计和输出路径。

### plot_wind_speed_beaufort_metrics.py

- 增加独立字号、字体和中文标签配置。
- 图例实验名称、横轴标签和预报时效子图标题中文化。
- 删除整图大标题和底部说明小字。
- 不修改蒲福风级统计、子图排列和输出路径。

### plot_wind_speed_beaufort_sample_counts.py

- 增加独立字号、字体和中文标签配置。
- 横轴、纵轴和预报时效子图标题中文化。
- 放大柱顶样本数标注。
- 删除整图大标题和底部说明小字。
- 不修改样本统计和输出路径。

### plot_spatial_hex_best_rmse.py

- 保持当前三张子图布局和第一张图左上角图例位置。
- 增加独立字号、字体和中文标签配置。
- 图例实验名称、样本不足状态、无样本状态和预报时效子图标题中文化。
- 放大图例、坐标和网格内统计标注。
- 当前脚本已无大标题和底部说明小字，不新增此类文字。
- 不修改最佳模式判定、样本阈值、六边形统计和输出路径。

### plot_wipha_track_forecast_error.py

- 在调用共享绘图风格后，用本脚本配置覆盖字号和字体，避免修改共享模块并影响其他图。
- 本脚本内重新定义可人工修改的中文实验名称。
- 路径图标题、误差图标题、坐标轴和图例中文化。
- 删除整图大标题。
- 保留路径、误差计算、陆地边界和输出路径。

## 不修改内容

- 不修改任何数据读取、筛选、统计和误差计算逻辑。
- 不修改图片文件名和输出目录。
- 不修改子图数量、排列和现有地图显示范围。
- 不修改六边形尺寸、样本阈值、颜色和线型，除非中文文字变长后必须做最小布局适配。
- 不修改六个目标脚本之外的共享模块。

## 验证方式

实施后执行以下检查：

1. 使用 `python -m py_compile` 检查六个脚本语法。
2. 搜索并确认六个脚本中已移除 `fig.suptitle` 和底部说明用的 `fig.text`。
3. 搜索并确认实验名称均通过本脚本的 `TEXT_LABELS` 配置显示。
4. 检查每个脚本均存在 `FONT_SCALE = 1.25`、`FONT_FAMILY` 和可编辑的 `TEXT_LABELS`。
5. 检查 Git diff，确认没有统计逻辑、输入路径和输出路径的非预期变化。

如运行环境允许，再使用 `conda pangu` 生成图片进行中文字体与布局目视检查；若运行依赖或数据不足，则明确记录未完成的渲染验证。

## 风险与处理

- 中文字体在不同机器上可能不可用：使用多个候选字体，并允许人工调整 `FONT_FAMILY`。
- 中文文字长度可能造成图例拥挤：只做必要的图例位置或边距微调，不改变图的统计结构。
- 25% 放大后标注可能重叠：保留单项基础字号配置，允许对密集数字标注单独调小。
