# ICOADS 验证数据

本目录存放全年 2025 年预报验证所需的 ICOADS v3.0.2 NRT 日文件。

全年起报日为 2025-01-01 至 2025-12-31，输出时效为 0、3、…、120 h，
因此观测有效日期覆盖 2025-01-01 至 2026-01-05，共 370 天。

```text
icoads/
  nc/YYYY/MM/icoads3.0.2_daily_*.nc
  manifest.json
```

下载器按同一观测日期优先选择 `daily_updated`，没有更新版时选择
`daily_preliminary`；若两者都没有，则保存唯一可用的 `daily_total_updated`/`.dat.gz` 文件，并在清单中标记版本。所有文件的循环日期、URL、大小和 SHA-256 都写入 `manifest.json`。
文件采用 `.part` 临时文件和签名校验，重复运行会跳过完整文件并从断点继续。

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.icoads.download --list-only
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.icoads.download
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -m experiment_2025.icoads.download --verify-only
```

