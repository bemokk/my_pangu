# GRIB2 转 NetCDF 与多年训练说明

## 1. 目标

新增 2016-2024 年 ERA5 风场和波浪场数据后，推荐先将 GRIB2 批量转换为按年份组织的 NetCDF，再使用现有训练流程读取。

原始数据保留在：

```text
Wind_Wave\data\raw
```

转换后的数据建议放在：

```text
Wind_Wave\data\converted\<year>\wind.nc
Wind_Wave\data\converted\<year>\wave.nc
```

例如：

```text
Wind_Wave\data\converted\2016\wind.nc
Wind_Wave\data\converted\2016\wave.nc
...
Wind_Wave\data\converted\2024\wind.nc
Wind_Wave\data\converted\2024\wave.nc
```

## 2. 转换命令

在仓库根目录 `E:\PyCharm_WorkSpace\pangu` 下运行：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' Wind_Wave/convert_grib.py `
  --raw-dir Wind_Wave/data/raw `
  --output-dir Wind_Wave/data/converted `
  --years 2016:2024 `
  --region 5,50,95,150
```

脚本会读取 `.grib` 和 `.grib2` 文件，提取：

```text
wind: u10, v10
wave: swh, mwp, mwd
```

并按年份写出 `wind.nc` 和 `wave.nc`。

## 3. 依赖要求

GRIB2 读取依赖 `cfgrib` 和 ecCodes。当前代码会在缺少 ecCodes 时给出明确错误。

如果转换时报错 `Cannot find the ecCodes library`，需要先在 `pangu` 环境安装 ecCodes，例如：

```powershell
conda install -n pangu -c conda-forge eccodes cfgrib -y
```

安装后可检查：

```powershell
@'
for name in ("cfgrib", "eccodes"):
    module = __import__(name)
    print(name, getattr(module, "__version__", "unknown"))
'@ | & 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' -
```

## 4. 多年训练命令

转换完成后，可以用 converted 数据源训练：

```powershell
& 'C:\Users\SLDUO\anaconda3\envs\pangu\python.exe' Wind_Wave/train.py `
  --data-source converted `
  --converted-dir Wind_Wave/data/converted `
  --years 2016:2024 `
  --epochs 5 `
  --batch-size 8 `
  --spatial-stride 16 `
  --hidden-channels 16 `
  --num-workers 0 `
  --preload-spatial `
  --model-variant m2-wave0-direct `
  --run-name multi_year_m2_wave0_direct
```

如果想使用新数据完整的 5-50N 输入范围，可加：

```powershell
--input-region 5,50,95,150
```

输出区域默认仍为中国近海：

```text
15-40N, 105-135E
```

## 5. 注意事项

1. `data/raw` 中的原始 GRIB2 不会被删除。
2. 转换脚本默认不覆盖已有 `wind.nc` 和 `wave.nc`；如需重转，加 `--overwrite`。
3. 如果多个 GRIB 文件按变量或时间拆分，转换脚本会按变量合并并按年份切分。
4. 真实多年训练数据量明显大于 2025 单年实验，建议先用 `--years 2016 --max-samples 16` 做烟测，再跑 2016-2024 全量训练。
