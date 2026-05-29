from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT_DIR = Path(__file__).resolve().parent / "icoads_202507"
NC_DIR = ROOT_DIR / "nc"
OUT_DIR = ROOT_DIR / "output"

# AREA follows the common meteorological order: [lat_max, lon_min, lat_min, lon_max].
AREA = [42, 103, 12, 131]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA

AREA_LABEL = f"area_{LAT_MAX:g}_{LON_MIN:g}_{LAT_MIN:g}_{LON_MAX:g}".replace(".", "p")
DETAIL_OUT = OUT_DIR / f"china_sea_all_platform_records_{AREA_LABEL}.csv"
SUMMARY_OUT = OUT_DIR / f"china_sea_all_platform_summary_{AREA_LABEL}.csv"

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


def read_area_records(nc_file: Path) -> pd.DataFrame:
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

    for column in ["latitude", "longitude", "hour_utc", "id_indicator", "platform_type", "wind_speed_ms", "wind_dir_deg"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "platform_id" not in df.columns:
        df["platform_id"] = pd.NA
    df["platform_id"] = clean_string_series(df["platform_id"])

    if "uid" in df.columns:
        df["uid"] = clean_string_series(df["uid"])
        missing_id = df["platform_id"].isna()
        df.loc[missing_id, "platform_id"] = "UID_" + df.loc[missing_id, "uid"].astype(str)

    df["platform_id"] = df["platform_id"].fillna("UNKNOWN")

    mask = (
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
    )

    area_df = df.loc[mask].copy()
    area_df["source_nc_file"] = nc_file.name

    output_cols = [
        "datetime_utc",
        "date",
        "hour_utc",
        "platform_id",
        "uid",
        "id_indicator",
        "platform_type",
        "latitude",
        "longitude",
        "wind_dir_deg",
        "wind_speed_ms",
        "source_nc_file",
    ]
    output_cols = [col for col in output_cols if col in area_df.columns]
    return area_df[output_cols]


def join_unique_values(series: pd.Series) -> str:
    values = series.dropna().astype(int).astype(str).unique()
    return ",".join(sorted(values))


def minimal_longitude_range(longitudes: pd.Series) -> float:
    lon = pd.to_numeric(longitudes, errors="coerce").dropna().to_numpy(dtype=float)
    if lon.size <= 1:
        return 0.0

    lon = np.mod(lon, 360.0)
    lon.sort()
    gaps = np.diff(np.r_[lon, lon[0] + 360.0])
    return float(360.0 - gaps.max())


def build_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for platform_id, group in records.groupby("platform_id", dropna=False):
        first_time = group["datetime_utc"].min() if "datetime_utc" in group.columns else pd.NaT
        last_time = group["datetime_utc"].max() if "datetime_utc" in group.columns else pd.NaT

        rows.append(
            {
                "platform_id": platform_id,
                "obs_count": int(len(group)),
                "first_time": first_time,
                "last_time": last_time,
                "mean_latitude": group["latitude"].mean(),
                "mean_longitude": group["longitude"].mean(),
                "min_latitude": group["latitude"].min(),
                "max_latitude": group["latitude"].max(),
                "min_longitude": group["longitude"].min(),
                "max_longitude": group["longitude"].max(),
                "lat_range_deg": group["latitude"].max() - group["latitude"].min(),
                "lon_range_deg": minimal_longitude_range(group["longitude"]),
                "id_indicator_values": join_unique_values(group["id_indicator"]) if "id_indicator" in group.columns else "",
                "platform_type_values": join_unique_values(group["platform_type"]) if "platform_type" in group.columns else "",
                "wind_dir_count": int(group["wind_dir_deg"].notna().sum()) if "wind_dir_deg" in group.columns else 0,
                "wind_speed_count": int(group["wind_speed_ms"].notna().sum()) if "wind_speed_ms" in group.columns else 0,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(["obs_count", "platform_id"], ascending=[False, True])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(NC_DIR.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {NC_DIR}")

    all_records = []
    for index, nc_file in enumerate(nc_files, start=1):
        print(f"[{index:02d}/{len(nc_files):02d}] 读取 {nc_file.name}")
        area_df = read_area_records(nc_file)
        print(f"  区域内记录数：{len(area_df)}")
        if not area_df.empty:
            all_records.append(area_df)

    if not all_records:
        raise RuntimeError(f"没有找到落在 AREA={AREA} 内的记录。")

    records = pd.concat(all_records, ignore_index=True)
    records = records.sort_values(["datetime_utc", "platform_id", "latitude", "longitude"], na_position="last")
    records.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")

    summary = build_summary(records)
    summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    print("\n倒查完成")
    print(f"AREA: {AREA}")
    print(f"记录数：{len(records)}")
    print(f"平台数：{len(summary)}")
    print(f"时间范围：{records['datetime_utc'].min()} 至 {records['datetime_utc'].max()}")
    print(f"明细输出：{DETAIL_OUT}")
    print(f"汇总输出：{SUMMARY_OUT}")


if __name__ == "__main__":
    main()
