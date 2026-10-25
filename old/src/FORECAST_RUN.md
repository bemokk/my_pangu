# 四组全年风场试验

从项目根目录运行，使用项目的 `pangu` Python 环境。所有起报时间和有效时间均为 UTC。

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' src\main.py --dry-run
```

`--dry-run` 只检查输入 NPY 和模型文件路径，不执行推理、不创建输出。缺文件时返回退出码 1。
全年输入需覆盖 2024-12-27 至 2025-12-31 的每日 00 UTC，分别放在
`model_input/single_time_point/era5/<初始资料时刻>/` 和
`model_input/single_time_point/gdas/<初始资料时刻>/`，每个时次需要
`input_surface.npy`（float32，4×721×1440）和
`input_upper.npy`（float32，5×13×721×1440）。同源试验共享这些输入。
如果保留 `surface.nc`，会读取并核对网格；仅有 NPY 时使用标准网格：
纬度 90 至 -90，经度 0 至 359.75，间隔 0.25°。

## 运行

```powershell
# 默认：2025 年每天 00 UTC，四组试验，0、3、6……72 小时
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' src\main.py

# 小范围运行
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' src\main.py --start-date 2025-07-17 --end-date 2025-07-17 --experiments gdas_direct --lead-hours 0,3,6
```

支持 `--hour`、`--experiments`、`--lead-hours`、`--project-root`、`--output-root`、
`--cpu` 和 `--overwrite`。日期首尾均包含。0 小时始终保存，即使参数只指定正时效。
默认使用 GPU；GPU 不可用时报告错误，CPU 运行需显式指定 `--cpu`。

| 试验名称 | 初始资料时刻 | 实际模型时效 | 0 小时输出 |
|---|---|---|---|
| gdas_direct | GDAS：T0 | L | 输入 GDAS 的 u10/v10 副本 |
| era5_lagged_120h | ERA5：T0−120h | 120+L | 积分 120 小时后的 u10/v10 |
| era5_direct | ERA5：T0 | L | 输入 ERA5 的 u10/v10 副本 |
| gdas_lagged_120h | GDAS：T0−120h | 120+L | 积分 120 小时后的 u10/v10 |

L 是相对 T0 的评价时效，有效时刻为 T0+L。各终点沿用原有 24/6/3/1 小时
贪心组合：例如 +30h 为 24+6h，+27h 为 24+3h，不改成连续 3h 积分。

## 输出与续跑

```text
model_output/experiments/<试验名称>/<T0年份>/<T0>/<L>/wind10_<有效时刻>.nc
```

例：`gdas_lagged_120h/2025/2025-07-17-00-00/24/wind10_2025-07-18-00-00.nc`，
属性记录初始资料时刻 2025-07-12 00 UTC、评价时效 24h、模型实际时效 144h。

最终 NetCDF 只含 u10、v10 两个风分量，以及经纬度、有效时间和试验元数据。
风分量为 float32，使用无损 zlib 压缩，不重复保存最终 NPY，也不保存高空风、
气压、温度、湿度、风速或风向。直接起报的 0 小时分量与输入 NPY 完全一致。

推理期间完整状态保存在该任务的 `timeline_cache` 中。全部请求时效验证通过后，
只删除该任务的缓存。失败/中断保留检查点，重新执行相同命令可跳过已完成输出
并复用有效缓存。临时文件以 `.part` 写入，写出和验证成功后替换正式文件。
缓存记录输入/模型文件大小及修改时间；资料或模型变化时拒绝复用旧缓存，可用
`--overwrite` 重建所选任务。不要同时启动两个进程处理同一试验、同一个 T0。

运行记录位于 `model_output/experiments/_runs/run_<时间戳>.json`，逐任务保存完成或失败
状态。个别任务失败后继续处理其他任务，最终有失败时返回退出码 1。

旧版 `infer(start_time, forecast_hour, dataType)` 保留为单时效便捷接口，但 `gdas`/
`era5` 现在明确表示直接起报，输出到新目录，且会额外保存 0 小时。
批量时效应使用 `infer_experiment` 或新入口，避免每个时效单独调用造成缓存反复清理。

本次没有迁移/删除已有 7 月结果，没有修改比对与绘图。当前旧版比对、绘图脚本
还不能直接读取新试验目录。气压中心和高空场分析仍需原有完整数据。
