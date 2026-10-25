# experiment_2025

这是 2025 年全年四组 Pangu 推理的独立代码。`old/` 只作为上一轮实验的归档，不会被本目录导入或修改。

目录按职责整理：core/ 为推理和数据模型核心模块，cli/ 为批量推理入口，	ools/ 为 ERA5 拆分/转换工具，scripts/ 为 PowerShell 启动脚本；GDAS、ICOADS、统计和绘图分别位于对应目录。


四组试验：

| 名称 | 初始资料 | 模型实际积分 |
|---|---|---|
| `gdas_direct` | GDAS：T0 | L |
| `era5_lagged_120h` | ERA5：T0−120h | 120+L |
| `era5_direct` | ERA5：T0 | L |
| `gdas_lagged_120h` | GDAS：T0−120h | 120+L |

代码职责：`config.py` 管理路径和输入约定，`experiments.py` 定义四组试验，
`input_data.py` 读取初始资料，`inference.py` 管理模型积分和检查点，
`wind_output.py` 写出压缩风场，`main.py` 管理全年批量任务。

从项目根目录运行（也可直接运行 `experiment_2025/main.py`）：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025 --dry-run
```

默认 ERA5 根目录是 `H:\pangu_2025_data\model_input\era5`，目录格式为：

```text
era5/raw/<year>/surface/*.nc       # 原始年度/多日文件
era5/raw/<year>/upper/*.nc         # 原始季度/多日文件
era5/<year>/<YYYY-MM-DD-HH-MM>/surface.nc
era5/<year>/<YYYY-MM-DD-HH-MM>/upper.nc
```

使用 `python -m experiment_2025.tools.split_era5` 从 raw 拆分 2024-12-27 至 2025-12-31
共370天。输出为无损压缩的每日 NetCDF，逐变量回读并校验源值一致。
原始文件保留不变，成功的每日文件可在重新执行时跳过；拆分记录位于 ERA5 根目录
`split_report.json`。拆分全部完成后再进行批量推理。

每日 NC 转为盘古输入 NPY：

```powershell
python -m experiment_2025.tools.convert_era5_npy
```

默认转换 2024-12-27 至 2025-12-31，保存在同一日期目录：
`input_surface.npy` 为 float32 `(4,721,1440)`，顺序 `msl/u10/v10/t2m`；
`input_upper.npy` 为 float32 `(5,13,721,1440)`，顺序 `z/q/t/u/v`。
13层顺序为 1000、925、850、700、600、500、400、300、250、200、150、100、50 hPa。
原 NC 和负比湿原值保留；每个 NPY 写出后完整回读，与 NC 选取的数组逐值比较。
再次运行会验证并跳过一致的 NPY。支持 `--root`、`--start-date`、`--end-date`，
逐日统计报告保存在 ERA5 根目录的 `npy_conversion_<开始日期>_<结束日期>.json`。
当前 ERA5 推理读取器仍读每日 NC，转为 NPY 不会自动改变推理的数据来源。

地面文件必须包含 `msl/u10/v10/t2m` 或对应的 ERA5 长变量名；高层文件必须包含 `z/q/t/u/v`、13 个指定气压层和 `valid_time`。
读取器优先使用每日目录，也兼容原有 `era5/<year>/surface/surface.nc` 和
`era5/<year>/upper/*.nc` 归档结构。程序按单个时次读取，全年文件不会一次性加载进内存。

GDAS 默认读取 `H:\pangu_2025_data\model_input\gdas\<年份>\<初始时间>\`
下的 `input_surface.npy` 和 `input_upper.npy`，可通过 `--gdas-root` 指定其他目录。
输入形状分别为 float32 `(4,721,1440)` 和 `(5,13,721,1440)`。
NPY 按 `msl/u10/v10/t2m`、`z/q/t/u/v` 排序，纬度 90→−90，经度 0→359.75。

可使用新的 GDAS 流程下载并转换输入：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.gdas run `
  --start 2025-07-01T00:00 --end 2025-07-31T00:00 --interval-hours 24
```

原始 GRIB2 默认保存到 `H:\pangu_2025_data\model_input\gdas\raw\<年份>`，转换结果保存到上述 GDAS
输入目录；也可用 `--raw-root` 和 `--output-root` 覆盖路径。流程支持 00/06/12/18
UTC 时次（默认间隔24h）、断点续传、完整文件跳过和 `--overwrite` 重建。
`download` 只下载，`process` 只处理已有 GRIB2；详细说明见 `gdas/README.md`。

ERA5 比湿中的负值保留原值并报告数量，本次不加入自动裁剪。原始文件不被修改。
时间必须精确匹配，缺日期、重复日期、缺测值、气压层或网格不匹配时会报告错误。

输出默认写入 H 盘：

```text
H:/pangu_2025_data/model_output/experiment_2025/
  <试验>/<T0年份>/<T0>/<评价时效>/wind10_<有效时刻>.nc
```

最终输出仅保存压缩的 float32 `u10/v10` NetCDF 和坐标、有效时间、试验元数据。
不另存最终 NPY、气压、温度或高层风。直接起报的 0h 为初始风分量的精确副本，
滞后起报的 0h 为初始资料积分 120h 后在 T0 的风分量。
实际模型积分保持旧版 24/6/3/1h 贪心组合：例如 30h=24+6h，27h=24+3h。

完整气象状态只在推理期间保存在当前任务的 `timeline_cache`，
所有请求时效写出并验证成功后自动删除。任务失败或中断会保留缓存，重新运行可续跑。
输入/模型文件变化后拒绝复用旧缓存，可指定 `--overwrite` 重建所选任务。
同一个试验、同一个 T0 不要由两个进程同时写入。逐任务记录存入输出目录 `_runs/`。

```powershell
# 单日 ERA5 试运行；默认保存0、3、6……72h
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025 --start-date 2025-07-01 --end-date 2025-07-01 --experiments era5_direct

# 四组全年推理（请先用 --dry-run 检查资料）
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025
```

支持 `--era5-root`、`--gdas-root`、`--model-root`、`--output-root`、`--hour`、
`--lead-hours 0,3,6`、`--cpu`、`--overwrite`。默认 CUDA，CPU 需显式指定。
ONNX 模型默认读取项目根目录 `models/`，不从 `old/` 读取代码或模型。
`--dry-run` 只检查输入时间覆盖、GDAS NPY 路径和模型文件，不创建结果，不执行模型。

滞后试验在 2025-01-01 起报需要 ERA5 2024-12-27 00 UTC 的输入。
本批 raw 已补充 2024-12-27 至 12-31，每日拆分范围包含这些资料。
当前 ERA5 年度归档只有每日 00 UTC，适用于每日 00 UTC 起报。

比对和绘图暂不迁移，仍保留在 `old/` 中。

一周测试（2025-07-01 至 07-07，每天00 UTC，0–120h每3h）可在 PowerShell 中运行
`experiment_2025/scripts/run_era5_week.ps1`。控制台每30秒显示当前任务与累计用时，
逐任务 JSON 记录输入、模型加载、ONNX 积分、检查点写入、风场写出及总耗时；
各阶段之和不包含路径检查、缓存读取和最终验证等开销，应以总耗时为准。
控制台日志位于输出根目录 `_runs/era5_week_console_*.log`，默认 GPU。

2025 全年 ERA5 直接起报（每天00 UTC，0–120h每3h）的精简窗口入口为
`experiment_2025/scripts/run_era5_year.ps1`。每个起报日只有一条 `[运行中]` 原行更新，
完成0–120h预报后换行保留 `[已完成] 日期 用时`；每30秒更新当前行用时。
完整输出保存至 `_runs/era5_year_detail_*.log`，逐日计时保存在 `_runs/run_*.json`。
已完成且校验通过的日期自动跳过，失败时保留缓存，重新执行可续跑。

本模块独立测试：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m pytest experiment_2025/tests experiment_2025/gdas/tests -q
```


