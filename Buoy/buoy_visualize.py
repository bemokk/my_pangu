import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import folium
from folium.plugins import MarkerCluster

# ============================================================
# 1. 路径与参数设置
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent / "icoads_202507"
NC_DIR = ROOT_DIR / "nc"
OUT_DIR = ROOT_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 中国近海范围，可根据你的研究区修改
LON_MIN, LON_MAX = 103, 130
LAT_MIN, LAT_MAX = 13, 42

# 是否只筛选 00、01、03、06、12 UTC
FILTER_TARGET_HOURS = True
TARGET_HOURS = np.array([0, 1, 3, 6, 12], dtype=float)

# True：严格整点；False：允许前后30分钟
EXACT_HOUR = True
TIME_TOLERANCE = 0.01 if EXACT_HOUR else 0.5

# 变量中文说明，可以后续继续补充
VAR_NAME_CN = {
    "D": "风向",
    "W": "风速",
    "SLP": "海平面气压",
    "AT": "气温",
    "SST": "海表温度",
    "N": "总云量",
    "WW": "现在天气",
    "VV": "能见度",
    "RH": "相对湿度",
    "ID": "平台识别号",
    "UID": "唯一记录号",
    "PT": "平台类型",
    "lat": "纬度",
    "lon": "经度",
    "HR": "观测小时",
    "date": "观测日期",
}

# 常用字段重命名
RENAME_DICT = {
    "lat": "latitude",
    "lon": "longitude",
    "HR": "hour_utc",
    "ID": "platform_id",
    "UID": "uid",
    "II": "id_indicator",
    "PT": "platform_type",
    "D": "wind_dir_deg",
    "W": "wind_speed_ms",
    "SLP": "sea_level_pressure",
    "AT": "air_temperature",
    "SST": "sea_surface_temperature",
}

# ============================================================
# 2. 工具函数
# ============================================================
def clean_string_series(s):
    """清理字符串或bytes字段。"""
    return (
        s.astype(str)
        .str.replace("b'", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.strip()
    )


def list_nc_variables(nc_files):
    """
    打印并保存每个nc文件的变量清单。
    """
    rows = []

    print("\n" + "=" * 80)
    print("开始读取 nc 文件变量清单")
    print("=" * 80)

    for nc_file in nc_files:
        print(f"\n文件：{nc_file.name}")

        with xr.open_dataset(nc_file, engine="netcdf4") as ds:
            for var in ds.variables:
                da = ds[var]
                long_name = da.attrs.get("long_name", "")
                standard_name = da.attrs.get("standard_name", "")
                units = da.attrs.get("units", "")
                dims = ",".join(da.dims)
                dtype = str(da.dtype)

                cn_name = VAR_NAME_CN.get(var, "")

                print(
                    f"  {var:15s} | {cn_name:10s} | "
                    f"dims={dims:20s} | dtype={dtype:10s} | "
                    f"units={units}"
                )

                rows.append({
                    "file": nc_file.name,
                    "variable": var,
                    "chinese_name": cn_name,
                    "dims": dims,
                    "dtype": dtype,
                    "units": units,
                    "long_name": long_name,
                    "standard_name": standard_name,
                })

    var_df = pd.DataFrame(rows)
    out_file = OUT_DIR / "nc_variables_summary_202507.csv"
    var_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n变量清单已保存：{out_file}")

    return var_df


def build_datetime(df):
    """
    根据 time 或 date + hour_utc 构造 datetime_utc。
    """
    if "time" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["datetime_utc"] = pd.NaT

    if df["datetime_utc"].isna().all():
        if "date" in df.columns and "hour_utc" in df.columns:
            df["datetime_utc"] = (
                pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
                + pd.to_timedelta(pd.to_numeric(df["hour_utc"], errors="coerce"), unit="h")
            )

    return df


def add_target_hour(df):
    """
    添加最近目标时次。
    """
    hr = pd.to_numeric(df["hour_utc"], errors="coerce").to_numpy(dtype=float)

    target_hour = np.full(len(df), np.nan)
    hour_diff = np.full(len(df), np.nan)

    valid = np.isfinite(hr)

    if valid.any():
        abs_diff = np.abs(hr[valid, None] - TARGET_HOURS[None, :])
        idx = abs_diff.argmin(axis=1)
        target_hour[valid] = TARGET_HOURS[idx]
        hour_diff[valid] = np.abs(hr[valid] - target_hour[valid])

    df["target_hour_utc"] = target_hour
    df["hour_diff"] = hour_diff

    return df


def get_available_vars_for_group(g, candidate_cols):
    """
    判断某个平台实际收集了哪些变量。
    只要该变量至少有一个非空值，就认为该平台收集了该变量。
    """
    available = []

    for col in candidate_cols:
        if col not in g.columns:
            continue

        s = g[col]

        if s.notna().any():
            available.append(col)

    return "，".join(available)


# ============================================================
# 3. 主程序：读取nc，筛选中国近海观测
# ============================================================
nc_files = sorted(NC_DIR.glob("*.nc"))

if len(nc_files) == 0:
    raise FileNotFoundError(f"没有在该目录找到nc文件：{NC_DIR}")

print(f"找到 {len(nc_files)} 个 nc 文件")

# 3.1 打印并保存变量清单
var_df = list_nc_variables(nc_files)

# 3.2 逐个读取文件，筛选观测点
all_records = []

for nc_file in nc_files:
    print("\n" + "=" * 80)
    print(f"读取文件：{nc_file.name}")

    with xr.open_dataset(nc_file, engine="netcdf4") as ds:
        # 记录原始变量名
        original_vars = list(ds.variables)

        # 转为DataFrame
        df = ds.to_dataframe().reset_index(drop=True)

    # 清理字符串字段
    for c in ["ID", "UID", "date"]:
        if c in df.columns:
            df[c] = clean_string_series(df[c])

    # 重命名字段
    df = df.rename(columns=RENAME_DICT)

    # 构造时间
    df = build_datetime(df)

    # 必要字段检查
    required_cols = ["latitude", "longitude", "hour_utc"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print(f"跳过该文件，缺少必要字段：{missing}")
        continue

    # 转为数值
    numeric_cols = [
        "latitude", "longitude", "hour_utc",
        "wind_speed_ms", "wind_dir_deg",
        "sea_level_pressure", "air_temperature",
        "sea_surface_temperature"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 添加目标时次
    df = add_target_hour(df)

    # 空间范围筛选
    mask = (
        df["longitude"].between(LON_MIN, LON_MAX)
        & df["latitude"].between(LAT_MIN, LAT_MAX)
    )

    # 时次筛选
    if FILTER_TARGET_HOURS:
        mask = mask & (df["hour_diff"] <= TIME_TOLERANCE)

    # 风速风向质量控制
    if "wind_speed_ms" in df.columns:
        mask = mask & (df["wind_speed_ms"].isna() | df["wind_speed_ms"].between(0, 75))

    if "wind_dir_deg" in df.columns:
        mask = mask & (df["wind_dir_deg"].isna() | df["wind_dir_deg"].between(0, 360))

    sub = df.loc[mask].copy()

    if len(sub) == 0:
        print("筛选后无记录")
        continue

    sub["source_nc_file"] = nc_file.name

    print(f"筛选后记录数：{len(sub)}")
    all_records.append(sub)

if len(all_records) == 0:
    raise RuntimeError("所有文件筛选后均无记录，请检查经纬度范围或时次设置。")

records = pd.concat(all_records, ignore_index=True)

# 如果没有platform_id，则用uid补充
if "platform_id" not in records.columns:
    records["platform_id"] = ""

records["platform_id"] = records["platform_id"].fillna("").astype(str).str.strip()

if "uid" in records.columns:
    records.loc[records["platform_id"].eq("") | records["platform_id"].eq("nan"), "platform_id"] = (
        "UID_" + records["uid"].astype(str)
    )

records.loc[records["platform_id"].eq("") | records["platform_id"].eq("nan"), "platform_id"] = "UNKNOWN"

# 保存逐条观测记录
records_file = OUT_DIR / "platform_records_202507.csv"
records.to_csv(records_file, index=False, encoding="utf-8-sig")
print(f"\n逐条观测记录已保存：{records_file}")
print(f"总记录数：{len(records)}")

# ============================================================
# 4. 按平台汇总：位置、观测次数、收集变量
# ============================================================
# 候选观测变量：排除经纬度、时间、ID类字段
exclude_cols = {
    "datetime_utc", "time", "date", "hour_utc", "target_hour_utc", "hour_diff",
    "latitude", "longitude", "platform_id", "uid", "id_indicator",
    "source_nc_file"
}

candidate_cols = []

for col in records.columns:
    if col not in exclude_cols:
        # 至少有一个非空值，才纳入候选变量
        if records[col].notna().any():
            candidate_cols.append(col)

summary_rows = []

for platform_id, g in records.groupby("platform_id"):
    lat_min = g["latitude"].min()
    lat_max = g["latitude"].max()
    lon_min = g["longitude"].min()
    lon_max = g["longitude"].max()

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # 粗略判断平台类型：
    # 位置变化很小：近固定平台，可能是锚定浮标/固定站
    # 位置变化较大：移动船舶或漂流浮标
    if lat_range <= 0.10 and lon_range <= 0.10:
        position_type = "近固定平台/疑似浮标"
    else:
        position_type = "移动平台/船舶或漂流浮标"

    vars_collected = get_available_vars_for_group(g, candidate_cols)

    summary_rows.append({
        "platform_id": platform_id,
        "position_type": position_type,
        "obs_count": len(g),
        "first_time": g["datetime_utc"].min(),
        "last_time": g["datetime_utc"].max(),
        "mean_latitude": g["latitude"].mean(),
        "mean_longitude": g["longitude"].mean(),
        "min_latitude": lat_min,
        "max_latitude": lat_max,
        "min_longitude": lon_min,
        "max_longitude": lon_max,
        "lat_range": lat_range,
        "lon_range": lon_range,
        "platform_type_values": "，".join(sorted(g["platform_type"].dropna().astype(str).unique())) if "platform_type" in g.columns else "",
        "vars_collected": vars_collected,
        "wind_speed_count": g["wind_speed_ms"].notna().sum() if "wind_speed_ms" in g.columns else 0,
        "wind_dir_count": g["wind_dir_deg"].notna().sum() if "wind_dir_deg" in g.columns else 0,
        "pressure_count": g["sea_level_pressure"].notna().sum() if "sea_level_pressure" in g.columns else 0,
        "air_temp_count": g["air_temperature"].notna().sum() if "air_temperature" in g.columns else 0,
        "sst_count": g["sea_surface_temperature"].notna().sum() if "sea_surface_temperature" in g.columns else 0,
    })

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values(["position_type", "obs_count"], ascending=[True, False])

summary_file = OUT_DIR / "platform_summary_202507.csv"
summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("平台/浮标/船舶汇总信息")
print("=" * 80)
print(f"平台数量：{len(summary)}")
print(f"汇总文件已保存：{summary_file}")

print("\n前20个平台：")
print(summary.head(20)[[
    "platform_id",
    "position_type",
    "obs_count",
    "first_time",
    "last_time",
    "mean_latitude",
    "mean_longitude",
    "vars_collected"
]])

# 单独输出疑似固定浮标
fixed = summary[summary["position_type"].eq("近固定平台/疑似浮标")].copy()
fixed_file = OUT_DIR / "fixed_or_buoy_like_platforms_202507.csv"
fixed.to_csv(fixed_file, index=False, encoding="utf-8-sig")

print("\n疑似固定平台/浮标数量：", len(fixed))
print(f"疑似浮标汇总已保存：{fixed_file}")

# ============================================================
# 5. 生成交互式地图
# ============================================================
center_lat = (LAT_MIN + LAT_MAX) / 2
center_lon = (LON_MIN + LON_MAX) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="OpenStreetMap"
)

# 研究区范围框
folium.Rectangle(
    bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
    tooltip="中国近海筛选范围",
    fill=False
).add_to(m)

marker_cluster = MarkerCluster(name="观测平台").add_to(m)

for _, row in summary.iterrows():
    lat = row["mean_latitude"]
    lon = row["mean_longitude"]

    if pd.isna(lat) or pd.isna(lon):
        continue

    radius = 4 + min(np.log10(row["obs_count"] + 1) * 3, 10)

    popup_html = f"""
    <b>平台ID：</b>{row['platform_id']}<br>
    <b>类型判断：</b>{row['position_type']}<br>
    <b>观测次数：</b>{row['obs_count']}<br>
    <b>时间范围：</b>{row['first_time']} 至 {row['last_time']}<br>
    <b>平均位置：</b>{row['mean_latitude']:.3f}, {row['mean_longitude']:.3f}<br>
    <b>纬度范围：</b>{row['min_latitude']:.3f} ~ {row['max_latitude']:.3f}<br>
    <b>经度范围：</b>{row['min_longitude']:.3f} ~ {row['max_longitude']:.3f}<br>
    <b>收集变量：</b>{row['vars_collected']}<br>
    <b>风速记录数：</b>{row['wind_speed_count']}<br>
    <b>风向记录数：</b>{row['wind_dir_count']}<br>
    <b>气压记录数：</b>{row['pressure_count']}<br>
    <b>气温记录数：</b>{row['air_temp_count']}<br>
    <b>海温记录数：</b>{row['sst_count']}<br>
    """

    tooltip_text = f"{row['platform_id']} | {row['position_type']} | {row['obs_count']}条"

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=500),
        tooltip=tooltip_text,
        fill=True,
        fill_opacity=0.7
    ).add_to(marker_cluster)

# 对移动平台绘制轨迹，避免太乱，只绘制观测次数较多的前20个移动平台
moving = summary[summary["position_type"].str.contains("移动平台", na=False)]
moving_top_ids = moving.sort_values("obs_count", ascending=False).head(20)["platform_id"].tolist()

for pid in moving_top_ids:
    g = records[records["platform_id"].eq(pid)].copy()
    g = g.sort_values("datetime_utc")

    coords = g[["latitude", "longitude"]].dropna().values.tolist()

    if len(coords) >= 2:
        folium.PolyLine(
            locations=coords,
            tooltip=f"{pid} 轨迹",
            weight=2,
            opacity=0.6
        ).add_to(m)

folium.LayerControl().add_to(m)

map_file = OUT_DIR / "platform_map_202507.html"
m.save(map_file)

print("\n" + "=" * 80)
print("地图生成完成")
print("=" * 80)
print(f"地图文件：{map_file}")
print("用浏览器打开这个HTML文件即可查看平台位置。")
