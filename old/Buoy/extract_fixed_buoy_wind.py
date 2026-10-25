import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from paths import FIXED_BUOY_WIND_DIR

ROOT_DIR = Path(__file__).resolve().parent / "icoads_202507"
NC_DIR = ROOT_DIR / "nc"
OUT_DIR = FIXED_BUOY_WIND_DIR

START_DATE = pd.Timestamp("2025-07-01")
END_DATE = pd.Timestamp("2025-08-04 23:59:59")
TARGET_HOURS = np.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=float)
TIME_TOLERANCE_HOURS = 0.01

MAX_POSITION_RANGE_DEG = 0.1
MIN_OBS_PER_PLATFORM = 2

# Set this to True if you only want platforms in the China nearshore domain.
APPLY_CHINA_NEARSHORE_FILTER = False
LON_MIN, LON_MAX = 103, 130
LAT_MIN, LAT_MAX = 13, 42

DETAIL_OUT = OUT_DIR / "fixed_buoy_wind_records_20250701_20250804_3hourly.csv"
SUMMARY_OUT = OUT_DIR / "fixed_buoy_platform_summary_20250701_20250804_3hourly.csv"

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
}


def clean_string_series(series: pd.Series) -> pd.Series:
    """Normalize ICOADS byte-string identifiers into plain strings."""
    cleaned = (
        series.astype(str)
        .str.replace("b'", "", regex=False)
        .str.replace('b"', "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    return cleaned.mask(cleaned.str.lower().isin(["", "nan", "none", "nat"]))


def add_datetime_utc(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["datetime_utc"] = pd.NaT

    if "date" in df.columns and "hour_utc" in df.columns:
        fallback_time = (
            pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
            + pd.to_timedelta(pd.to_numeric(df["hour_utc"], errors="coerce"), unit="h")
        )
        df["datetime_utc"] = df["datetime_utc"].fillna(fallback_time)

    df["datetime_utc"] = df["datetime_utc"].dt.round("s")
    return df


def minimal_longitude_range(longitudes: pd.Series) -> float:
    lon = pd.to_numeric(longitudes, errors="coerce").dropna().to_numpy(dtype=float)
    if lon.size <= 1:
        return 0.0

    lon = np.mod(lon, 360.0)
    lon.sort()
    gaps = np.diff(np.r_[lon, lon[0] + 360.0])
    return float(360.0 - gaps.max())


def nc_file_date(nc_file: Path) -> pd.Timestamp | None:
    match = re.search(r"_d(\d{8})_", nc_file.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def read_nc_records(nc_file: Path) -> pd.DataFrame:
    keep_vars = [
        "time",
        "date",
        "HR",
        "lat",
        "lon",
        "ID",
        "UID",
        "II",
        "PT",
        "D",
        "W",
    ]

    with xr.open_dataset(nc_file, engine="netcdf4", decode_timedelta=True) as ds:
        available_vars = [var for var in keep_vars if var in ds.variables]
        df = ds[available_vars].to_dataframe().reset_index(drop=True)

    for column in ["ID", "UID", "date"]:
        if column in df.columns:
            df[column] = clean_string_series(df[column])

    df = df.rename(columns=RENAME_DICT)
    df = add_datetime_utc(df)

    for column in ["latitude", "longitude", "hour_utc", "wind_speed_ms", "wind_dir_deg"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "platform_id" not in df.columns:
        df["platform_id"] = pd.NA
    df["platform_id"] = clean_string_series(df["platform_id"])

    if "uid" in df.columns:
        df["uid"] = clean_string_series(df["uid"])

    df["source_nc_file"] = nc_file.name
    return df


def load_all_records(nc_files: list[Path]) -> pd.DataFrame:
    frames = []

    for index, nc_file in enumerate(nc_files, start=1):
        print(f"[{index:02d}/{len(nc_files):02d}] 读取 {nc_file.name}")
        frame = read_nc_records(nc_file)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def filter_valid_wind_records(records: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["datetime_utc", "platform_id", "latitude", "longitude", "wind_speed_ms", "wind_dir_deg"]
    missing = [col for col in required_cols if col not in records.columns]
    if missing:
        raise RuntimeError(f"缺少必要字段：{missing}")

    hour_from_time = (
        records["datetime_utc"].dt.hour
        + records["datetime_utc"].dt.minute / 60.0
        + records["datetime_utc"].dt.second / 3600.0
    )

    if "hour_utc" not in records.columns:
        records["hour_utc"] = np.nan
    records["hour_utc"] = records["hour_utc"].fillna(hour_from_time)

    hour_diff = np.abs(records["hour_utc"].to_numpy(dtype=float)[:, None] - TARGET_HOURS[None, :])
    nearest_hour_idx = np.nanargmin(np.where(np.isnan(hour_diff), np.inf, hour_diff), axis=1)
    records["target_hour_utc"] = TARGET_HOURS[nearest_hour_idx].astype(int)
    records["hour_diff"] = np.abs(records["hour_utc"] - records["target_hour_utc"])

    mask = (
        records["platform_id"].notna()
        & records["datetime_utc"].notna()
        & records["datetime_utc"].between(START_DATE, END_DATE)
        & (records["hour_diff"] <= TIME_TOLERANCE_HOURS)
        & records["latitude"].between(-90, 90)
        & records["longitude"].notna()
        & records["wind_speed_ms"].between(0, 75)
        & records["wind_dir_deg"].between(1, 360)
    )

    if APPLY_CHINA_NEARSHORE_FILTER:
        mask = (
            mask
            & records["longitude"].between(LON_MIN, LON_MAX)
            & records["latitude"].between(LAT_MIN, LAT_MAX)
        )

    return records.loc[mask].copy()


def build_platform_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for platform_id, group in records.groupby("platform_id", dropna=True):
        lat_range = float(group["latitude"].max() - group["latitude"].min())
        lon_range = minimal_longitude_range(group["longitude"])

        platform_type_values = ""
        if "platform_type" in group.columns:
            values = group["platform_type"].dropna().astype(int).astype(str).unique()
            platform_type_values = ",".join(sorted(values))

        rows.append(
            {
                "platform_id": platform_id,
                "obs_count": int(len(group)),
                "first_time": group["datetime_utc"].min(),
                "last_time": group["datetime_utc"].max(),
                "mean_latitude": group["latitude"].mean(),
                "mean_longitude": group["longitude"].mean(),
                "min_latitude": group["latitude"].min(),
                "max_latitude": group["latitude"].max(),
                "min_longitude": group["longitude"].min(),
                "max_longitude": group["longitude"].max(),
                "lat_range_deg": lat_range,
                "lon_range_deg": lon_range,
                "platform_type_values": platform_type_values,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    fixed_mask = (
        (summary["obs_count"] >= MIN_OBS_PER_PLATFORM)
        & (summary["lat_range_deg"] < MAX_POSITION_RANGE_DEG)
        & (summary["lon_range_deg"] < MAX_POSITION_RANGE_DEG)
    )

    return summary.loc[fixed_mask].sort_values(
        ["obs_count", "platform_id"],
        ascending=[False, True],
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(NC_DIR.glob("*.nc"))
    nc_files = [
        nc_file for nc_file in nc_files
        if (file_date := nc_file_date(nc_file)) is not None
        and START_DATE.normalize() <= file_date <= END_DATE.normalize()
    ]
    if not nc_files:
        raise FileNotFoundError(f"没有在目录中找到 NetCDF 文件：{NC_DIR}")

    print(f"NetCDF 目录：{NC_DIR}")
    print(f"找到 {len(nc_files)} 个 NetCDF 文件")
    print(f"时间范围：{START_DATE:%Y-%m-%d} 至 {END_DATE:%Y-%m-%d}")
    print(f"目标UTC时次：{', '.join(str(int(h)) for h in TARGET_HOURS)}")

    records = load_all_records(nc_files)
    records = filter_valid_wind_records(records)
    print(f"有效风观测记录数：{len(records)}")

    fixed_summary = build_platform_summary(records)
    fixed_summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    fixed_ids = set(fixed_summary["platform_id"]) if not fixed_summary.empty else set()
    fixed_records = records[records["platform_id"].isin(fixed_ids)].copy()
    fixed_records = fixed_records.sort_values(["platform_id", "datetime_utc"])

    output_cols = [
        "platform_id",
        "datetime_utc",
        "latitude",
        "longitude",
        "wind_dir_deg",
        "wind_speed_ms",
        "hour_utc",
        "target_hour_utc",
        "platform_type",
        "source_nc_file",
    ]
    output_cols = [col for col in output_cols if col in fixed_records.columns]
    fixed_records = fixed_records[output_cols]

    fixed_records.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")

    print("抽取完成")
    print(f"固定/近固定平台数量：{len(fixed_summary)}")
    print(f"固定/近固定平台风记录数：{len(fixed_records)}")
    print(f"明细输出：{DETAIL_OUT}")
    print(f"平台汇总：{SUMMARY_OUT}")


if __name__ == "__main__":
    main()
