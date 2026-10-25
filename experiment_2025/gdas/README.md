# GDAS 下载与处理

参考 `old/gdas` 重构，不导入旧代码。下载 GDAS/FNL 0.25° 分析场（f00），
生成与 `experiment_2025.core.input_data.NpyDirectorySource` 兼容的每日输入。

在项目根目录 `E:\PyCharm_WorkSpace\pangu` 运行，所有时间均为 UTC：

```powershell
# 单日：下载并转换
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.gdas run `
  --start 2025-07-01T00:00 --end 2025-07-01T00:00 --interval-hours 24

# 全年试验所需的 370 天，包含滞后起报所需的 2024-12-27 至 12-31
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.gdas run `
  --start 2024-12-27 --end 2025-12-31 --interval-hours 24 --continue-on-error

# 只下载 GRIB2，不转换
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.gdas download `
  --start 2025-07-01 --end 2025-07-01 --interval-hours 24

# 只处理已有文件；可以直接读取旧目录中的原始 GRIB2
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.gdas process `
  --start 2025-07-01 --end 2025-07-31 --interval-hours 24 `
  --raw-root 'E:\PyCharm_WorkSpace\pangu\old\gdas\grib2\fnl'
```

支持 00、06、12、18 UTC，时间间隔默认 **24 小时**，适配全年每天一次起报。
如需每天四次资料，显式设置 `--interval-hours 6`；间隔必须是 6 的正整数倍。
日期范围包含首尾；仅写日期表示当天 00 UTC。带时区的 ISO 时间会转换为 UTC。

默认路径：

| 内容 | 路径 | 覆盖方式 |
|---|---|---|
| 原始 GRIB2 根目录 | `H:\pangu_2025_data\model_input\gdas\raw` | `--raw-root` 或 `PANGU_GDAS_RAW_ROOT` |
| 转换结果 | `H:\pangu_2025_data\model_input\gdas` | `--output-root` 或 `PANGU_GDAS_ROOT` |

文件按资料的实际起报年份存放，跨年日期自动分流。
`--raw-root` 和 `--output-root` 均传入根目录，不应包含额外年份。
目录结构如下，文件列表属于各日期目录：

```text
gdas/raw/<year>/gdas1.fnl0p25.<YYYYMMDDHH>.f00.grib2
gdas/<year>/<YYYY-MM-DD-HH-MM>/
  surface.nc            # msl/u10/v10/t2m，带 valid_time 和坐标
  upper.nc              # z/q/t/u/v，带 valid_time、pressure_level 和坐标
  input_surface.npy     # float32 (4,721,1440)
  input_upper.npy       # float32 (5,13,721,1440)
  prepared.json         # 源文件身份、转换版本、输出 SHA256，最后写入
```

高空层序为 `1000,925,850,700,600,500,400,300,250,200,150,100,50` hPa。
统一纬度 90→−90、经度 0→359.75，严格校验全球 0.25° 网格和每个字段的分析/有效时次。
`gh` 乘以 9.80665 转为位势 `z`；比湿保留原值。缺变量、缺层、缺测值或非有限值时报错。
pygrib 2.1.8 可能将有效的约9999米位势高度误加掩码。只有在 GRIB 同时确认
`numberOfMissing=0`、`bitmapPresent=0`、`missingValueManagementUsed=0`，且
`numberOfDataPoints=numberOfValues=实际网格点数` 时，读取器才保留掩码下的原始数值，
并报告恢复点数。不进行插值或补值；真实缺测、未知缺测元数据仍保留掩码并报错，
恢复后也继续执行 NaN/Inf 检查。原始 GRIB2 不被修改。
NetCDF 使用无损压缩；写出后逐变量回读，核对 NPY 与提取值一致。

下载使用重试、HTTP Range 分段续传和 `.part` 文件，初始分段为 8 MiB，
连接中断后自动缩小分段（最小 1 MiB），每段最多尝试 5 次，
检查字节数及所有 GRIB2 消息的边界，
避免 HTML 错误页面或截断文件被当成成功下载。远端 401/403/404/410 会报告具体错误。
转换后的四个文件各自通过临时文件替换；全部校验通过才写入 `prepared.json`。
下载器自动将 raw 根目录已有的扁平 GRIB2 或 `.part` 移入对应年份，继续复用/续传；
目标文件已存在时保留两份，不直接覆盖。只处理模式也支持旧版扁平原始目录。
推理和资料检查优先读取 `<year>/<起报时间>`，仍支持显式指定的旧版扁平输入目录。
重跑时校验源文件大小/修改时间及输出哈希，完整结果跳过，损坏或未完成结果自动重建；
`--overwrite` 强制重建转换结果。同一时次请只运行一个写入进程，转换期间不要启动该时次的推理。

默认遇到错误立即停止。`--continue-on-error` 继续剩余时次，最后汇总失败并返回非零退出码；
重跑同一命令即可补齐失败时次。保留 `download_gdas.py`（下载并处理）和 `gen_nc.py`
（只处理）两个入口，支持直接执行并接收相同日期参数。

每个批任务将时间、输出路径、耗时、完成或失败状态写入 `_runs/*.json`。
只下载的报告位于 `gdas/raw/_runs/`，下载并转换和只处理的报告位于 `gdas/_runs/`。
首次失败停止时也保留报告；强制终止时记录保留到最近完成的时次。
运行前使用 `--dry-run` 查看日期数量、现有文件数量和实际存放路径。
预览仅检查路径存在，不代替下载和转换后的完整性校验；不联网、不创建文件：

```powershell
python -m experiment_2025.gdas run --start 2024-12-27 --end 2025-12-31 --dry-run
```

依赖 `numpy`、`requests`、`netCDF4`、`pygrib`，`tqdm` 为可选进度条。
Windows 下建议使用现有 `pangu` Conda 环境；安装 pygrib 可用 `conda install -c conda-forge pygrib`。

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m pytest experiment_2025/gdas/tests experiment_2025/tests -q -p no:cacheprovider
```

