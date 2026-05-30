import re
import time
import requests
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from urllib.parse import urljoin

# ============================================================
# 1. 参数设置
# ============================================================
BASE_URL = "https://www.ncei.noaa.gov/data/international-comprehensive-ocean-atmosphere/v3/archive/nrt/daily/2025/07/"

# 中国近海范围，可根据论文或你的Pangu研究区调整
LON_MIN, LON_MAX = 103, 130
LAT_MIN, LAT_MAX = 13, 42

# 需要的UTC时次
TARGET_HOURS = np.array([0, 1, 3, 6, 12], dtype=float)

# True：严格匹配整点；False：允许±30分钟
exact_hour = True
time_tolerance = 0.01 if exact_hour else 0.5

ROOT_DIR = Path(__file__).resolve().parent / "icoads_202507"
NC_DIR = ROOT_DIR / "nc"
OUT_DIR = ROOT_DIR / "output"
NC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ============================================================
# 2. 下载函数：先下载到本地，再读取
# ============================================================
def download_nc(url, local_path, max_retry=3):
    """
    下载NetCDF文件到本地。
    如果本地已有且文件头正常，则跳过下载。
    """
    if local_path.exists() and local_path.stat().st_size > 1000:
        with open(local_path, "rb") as f:
            sig = f.read(8)
        if sig.startswith(b"CDF") or sig.startswith(b"\x89HDF"):
            print(f"本地已存在，跳过下载：{local_path.name}")
            return local_path
        else:
            print(f"本地文件异常，重新下载：{local_path.name}")
            local_path.unlink()

    tmp_path = local_path.with_suffix(local_path.suffix + ".part")

    for i in range(1, max_retry + 1):
        try:
            print(f"下载第 {i} 次：{url}")

            with requests.get(
                url,
                headers=HEADERS,
                stream=True,
                timeout=(20, 300)
            ) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP状态码异常：{r.status_code}")

                content_type = r.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError(f"返回的是HTML页面，不是NetCDF文件：{content_type}")

                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total > 0:
                                pct = downloaded / total * 100
                                print(f"\r  进度：{downloaded/1024/1024:.2f} MB / {total/1024/1024:.2f} MB ({pct:.1f}%)", end="")
                            else:
                                print(f"\r  已下载：{downloaded/1024/1024:.2f} MB", end="")

                print()

            # 检查文件头
            with open(tmp_path, "rb") as f:
                sig = f.read(8)

            if not (sig.startswith(b"CDF") or sig.startswith(b"\x89HDF")):
                raise RuntimeError("下载文件不是有效NetCDF文件，可能下载到了错误页面。")

            tmp_path.replace(local_path)
            print(f"下载完成：{local_path.name}")
            return local_path

        except Exception as e:
            print(f"下载失败：{local_path.name}，原因：{e}")
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(3)

    raise RuntimeError(f"多次下载失败：{url}")


# ============================================================
# 3. 获取2025年7月每日 daily_updated NetCDF 文件列表
# ============================================================
html = requests.get(BASE_URL, headers=HEADERS, timeout=60).text

# 只匹配 daily_updated，不匹配 daily_total_updated
nc_files = sorted(set(re.findall(
    r'icoads3\.0\.2_daily_updated_d202507\d{2}_c\d{8}\.nc',
    html
)))

print(f"找到 {len(nc_files)} 个 NetCDF 文件")
if len(nc_files) == 0:
    raise RuntimeError("没有找到NetCDF文件，请检查BASE_URL或网页结构是否变化。")

# ============================================================
# 4. 逐日下载、读取、筛选
# ============================================================
all_records = []

for fn in nc_files:
    url = urljoin(BASE_URL, fn)
    local_nc = NC_DIR / fn

    print("\n" + "=" * 80)
    print(f"处理：{fn}")

    # 先下载到本地
    local_nc = download_nc(url, local_nc)

    # 再读取本地nc文件，不要直接读取url
    with xr.open_dataset(local_nc, engine="netcdf4") as ds:
        print(ds)

        keep_vars = [
            "time", "date", "HR", "lat", "lon",
            "ID", "II", "PT", "D", "W", "UID",
            "SLP", "AT", "SST"
        ]
        keep_vars = [v for v in keep_vars if v in ds.variables]

        df = ds[keep_vars].to_dataframe().reset_index(drop=True)

    # 字符串字段清理
    for c in ["ID", "UID", "date"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 字段重命名
    df = df.rename(columns={
        "lat": "latitude",
        "lon": "longitude",
        "HR": "hour_utc",
        "ID": "platform_id",
        "II": "id_indicator",
        "PT": "platform_type",
        "D": "wind_dir_deg",
        "W": "wind_speed_ms"
    })

    # 构造时间
    if "time" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["datetime_utc"] = pd.NaT

    # 如果time解析失败，则用date + hour_utc构造
    if df["datetime_utc"].isna().all() and "date" in df.columns and "hour_utc" in df.columns:
        df["datetime_utc"] = (
            pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
            + pd.to_timedelta(df["hour_utc"].astype(float), unit="h")
        )

    # 必要字段检查
    need_cols = ["longitude", "latitude", "hour_utc", "wind_speed_ms", "wind_dir_deg"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        print(f"跳过该文件，缺少字段：{missing}")
        continue

    # 转为数值
    for c in ["longitude", "latitude", "hour_utc", "wind_speed_ms", "wind_dir_deg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ============================================================
    # 找到距离目标时次最近的小时：修复 HR 存在 NaN 的问题
    # ============================================================

    # 先确保 datetime_utc 是时间格式
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")

    # 如果 hour_utc 存在缺失值，则用 datetime_utc 补充
    # 例如 2025-07-21 06:00:00 -> 6.0
    if "hour_utc" not in df.columns:
        df["hour_utc"] = np.nan

    df["hour_utc"] = pd.to_numeric(df["hour_utc"], errors="coerce")

    hour_from_time = (
            df["datetime_utc"].dt.hour
            + df["datetime_utc"].dt.minute / 60.0
            + df["datetime_utc"].dt.second / 3600.0
    )

    df["hour_utc"] = df["hour_utc"].fillna(hour_from_time)

    # 初始化结果列
    df["target_hour_utc"] = np.nan
    df["hour_diff"] = np.nan

    # 只对 hour_utc 有效的记录计算最近目标时次
    valid_hr = df["hour_utc"].notna().to_numpy()

    if valid_hr.any():
        hr_valid = df.loc[valid_hr, "hour_utc"].to_numpy(dtype=float)

        diff = np.abs(hr_valid[:, None] - TARGET_HOURS[None, :])
        nearest_idx = np.argmin(diff, axis=1)

        df.loc[valid_hr, "target_hour_utc"] = TARGET_HOURS[nearest_idx]
        df.loc[valid_hr, "hour_diff"] = np.abs(
            df.loc[valid_hr, "hour_utc"].to_numpy(dtype=float)
            - df.loc[valid_hr, "target_hour_utc"].to_numpy(dtype=float)
        )
    else:
        print("警告：该文件中 hour_utc 和 datetime_utc 都无法提供有效小时信息，跳过该文件。")
        continue

    # target_hour_utc 转为整数型，保留 NaN
    df["target_hour_utc"] = df["target_hour_utc"].astype("Int64")

    # 筛选中国近海、目标时次、有效风速风向
    mask = (
        df["longitude"].between(LON_MIN, LON_MAX)
        & df["latitude"].between(LAT_MIN, LAT_MAX)
        & (df["hour_diff"] <= time_tolerance)
        & df["wind_speed_ms"].between(0, 75)
        & df["wind_dir_deg"].between(1, 360)
    )

    sub = df.loc[mask].copy()

    if len(sub) == 0:
        print("该日筛选后无记录。")
        continue

    # 只保留常用字段
    final_cols = [
        "datetime_utc", "target_hour_utc",
        "platform_id", "id_indicator", "platform_type",
        "latitude", "longitude",
        "wind_speed_ms", "wind_dir_deg",
        "hour_utc", "hour_diff"
    ]
    final_cols = [c for c in final_cols if c in sub.columns]
    sub = sub[final_cols]

    print(f"该日筛选后记录数：{len(sub)}")
    all_records.append(sub)

# ============================================================
# 5. 合并、去重、保存
# ============================================================
if len(all_records) == 0:
    print("最终没有筛选到任何记录。")
else:
    out = pd.concat(all_records, ignore_index=True)

    dedup_cols = [
        "datetime_utc", "platform_id", "latitude", "longitude",
        "wind_speed_ms", "wind_dir_deg"
    ]
    dedup_cols = [c for c in dedup_cols if c in out.columns]

    out = out.drop_duplicates(subset=dedup_cols)
    out = out.sort_values(["datetime_utc", "platform_id", "latitude", "longitude"])

    out_file = OUT_DIR / "icoads_202507_china_nearshore_00_01_03_06_12UTC.csv"
    out.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("全部完成")
    print(f"输出文件：{out_file}")
    print(f"总记录数：{len(out)}")
    print(out.head(20))
