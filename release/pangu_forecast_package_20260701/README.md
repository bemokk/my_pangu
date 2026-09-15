# 盘古天气预报项目

本项目用于下载 ERA5 或 GDAS 数据，将其转换为盘古天气模型输入，并使用 GPU 执行 ONNX 推理。

## 项目结构

```text
pangu_forecast_package_20260701/
|-- src/
|   |-- download.py       下载并转换 ERA5/GDAS
|   `-- main.py           执行盘古模型 GPU 推理
|-- tools/                下载、转换、推理和解码的内部代码
|-- models/               盘古 ONNX 模型
|-- model_input/          转换后的模型输入
|-- model_output/         预测结果
`-- requirements.txt      GPU 环境依赖
```

一般只需修改和运行 `src/download.py`、`src/main.py`。

## 前置准备

### 1. 安装依赖

建议使用 Python 3.10。在 Anaconda Prompt 或 PowerShell 中执行：

```powershell
conda create -n pangu python=3.10 -y
conda activate pangu
cd E:\PyCharm_WorkSpace\pangu\release\pangu_forecast_package_20260701
python -m pip install -r requirements.txt
```

如果 Windows 下 `pygrib` 安装失败，可使用 Conda：

```powershell
conda install -c conda-forge pygrib
```

GDAS 转换直接使用 `pygrib` 读取 GRIB2；Conda 会自动安装其底层 ecCodes 运行时，无需单独指定 `eccodes`。本项目不使用 `xarray` 或 `cfgrib`。

### 2. 放置盘古模型

将以下文件放入 `models/`：

```text
pangu_weather_1.onnx
pangu_weather_3.onnx
pangu_weather_6.onnx
pangu_weather_24.onnx
```

检查依赖、模型和 CUDA：

```powershell
python tools\verify_setup.py --device gpu
```

输出包含以下内容才表示 GPU 可用：

```text
Test session providers: CUDAExecutionProvider, CPUExecutionProvider
```

### 3. 配置 CDS API

只有下载 ERA5 时需要配置 CDS API：

1. 注册并登录 Copernicus Climate Data Store。
2. 打开官方 CDS API 设置页面：`https://cds.climate.copernicus.eu/how-to-api`。
3. 将登录后页面生成的配置原样复制到用户主目录下的 `.cdsapirc`。

Windows 文件位置通常为：

```text
C:\Users\你的用户名\.cdsapirc
```

还需要在 CDS 网站上接受 ERA5 single levels 和 pressure levels 数据集的许可条款。不要把个人 API Key 写进项目或发送给他人。

## 使用方法

### 1. 下载并转换数据

打开 `src/download.py`，修改顶部“用户配置区”：

```python
DATA_TYPE = "gdas"  # 可选 "gdas" 或 "era5"
START_TIME = datetime(2026, 6, 29, 0, 0)
END_TIME = datetime(2026, 6, 29, 18, 0)
TIME_INTERVAL_HOURS = 6
```

- GDAS 支持每天 `00、06、12、18 UTC`，时间间隔必须是 6 的倍数。
- ERA5 可根据需要设置时间间隔，例如 1、6、12 或 24 小时。
- 只下载一个时次时，让 `START_TIME` 与 `END_TIME` 相同。
- GDAS 时次尚未发布时，程序会退出并打印原因。

运行：

```powershell
python src\download.py
```

输入文件生成在：

```text
model_input/single_time_point/{gdas或era5}/YYYY-MM-DD-HH-MM/
```

### 2. 执行推理

打开 `src/main.py`，修改顶部“用户配置区”：

```python
DATA_TYPE = "gdas"  # 必须与下载的数据一致
START_TIME = datetime(2026, 6, 29, 0, 0)
END_TIME = datetime(2026, 6, 29, 18, 0)
FORECAST_HOURS = [3, 12, 72]
INITIAL_TIME_INTERVAL_HOURS = 6
STOP_ON_ERROR = True
```

`START_TIME` 和 `END_TIME` 必须与 `model_input` 中已有的初始时刻对应。程序固定使用 GPU。

运行：

```powershell
python src\main.py
```

预测结果生成在：

```text
model_output/{gdas或era5}/初始时刻/预报小时/
```
