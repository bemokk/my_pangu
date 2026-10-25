from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from land_mask import filter_ocean_records
from paths import (
    CHINA_SEA_RECORDS_DIR,
    DEFAULT_CHINA_SEA_DETAIL_CSV,
    DEFAULT_CHINA_SEA_SUMMARY_CSV,
    default_icoads_nc_dirs,
)


# AREA follows the common meteorological order: [lat_max, lon_min, lat_min, lon_max].
AREA = [42, 103, 13, 130]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA

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


def read_area_records(nc_file: Path):
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

    for column in [
        "latitude",
        "longitude",
        "hour_utc",
        "id_indicator",
        "platform_type",
        "wind_speed_ms",
        "wind_dir_deg",
    ]:
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

    mask = df["latitude"].between(LAT_MIN, LAT_MAX) & df["longitude"].between(LON_MIN, LON_MAX)

    area_df = df.loc[mask].copy()
    area_count = len(area_df)
    area_df, dropped_land_count = filter_ocean_records(
        area_df,
        lon_min=LON_MIN,
        lat_min=LAT_MIN,
        lon_max=LON_MAX,
        lat_max=LAT_MAX,
    )
    area_df["source_nc_file"] = nc_file.name
    area_df["source_nc_dir"] = nc_file.parent.parent.name

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
        "source_nc_dir",
    ]
    output_cols = [col for col in output_cols if col in area_df.columns]
    return area_df[output_cols], area_count, dropped_land_count


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
                "id_indicator_values": join_unique_values(group["id_indicator"])
                if "id_indicator" in group.columns
                else "",
                "platform_type_values": join_unique_values(group["platform_type"])
                if "platform_type" in group.columns
                else "",
                "wind_dir_count": int(group["wind_dir_deg"].notna().sum())
                if "wind_dir_deg" in group.columns
                else 0,
                "wind_speed_count": int(group["wind_speed_ms"].notna().sum())
                if "wind_speed_ms" in group.columns
                else 0,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(["obs_count", "platform_id"], ascending=[False, True])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract China Sea platform records from one or more ICOADS nc directories.",
    )
    parser.add_argument(
        "--nc-dir",
        action="append",
        type=Path,
        default=None,
        help="Input directory containing ICOADS .nc files. May be repeated.",
    )
    parser.add_argument("--out-dir", type=Path, default=CHINA_SEA_RECORDS_DIR)
    parser.add_argument("--detail-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser


def resolve_input_dirs(raw_dirs: list[Path] | None) -> list[Path]:
    dirs = [path.resolve() for path in raw_dirs] if raw_dirs else [path.resolve() for path in default_icoads_nc_dirs()]
    missing = [path for path in dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input nc directories do not exist: {missing}")
    if not dirs:
        raise FileNotFoundError("No ICOADS nc directories found under Buoy/icoads_*/nc")
    return dirs


def main() -> None:
    args = build_parser().parse_args()
    nc_dirs = resolve_input_dirs(args.nc_dir)
    out_dir = args.out_dir.resolve()
    detail_out = args.detail_out.resolve() if args.detail_out else DEFAULT_CHINA_SEA_DETAIL_CSV
    summary_out = args.summary_out.resolve() if args.summary_out else DEFAULT_CHINA_SEA_SUMMARY_CSV

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(nc_file for nc_dir in nc_dirs for nc_file in nc_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {nc_dirs}")

    print("Input nc directories:")
    for nc_dir in nc_dirs:
        print(f"  {nc_dir}")
    print(f"Output directory: {out_dir}")

    all_records = []
    total_area_count = 0
    total_dropped_land_count = 0
    for index, nc_file in enumerate(nc_files, start=1):
        print(f"[{index:02d}/{len(nc_files):02d}] Reading {nc_file.name}")
        area_df, area_count, dropped_land_count = read_area_records(nc_file)
        total_area_count += area_count
        total_dropped_land_count += dropped_land_count
        print(
            f"  in area: {area_count}, "
            f"dropped land: {dropped_land_count}, "
            f"kept ocean: {len(area_df)}"
        )
        if not area_df.empty:
            all_records.append(area_df)

    if not all_records:
        raise RuntimeError(f"No records found inside AREA={AREA}.")

    records = pd.concat(all_records, ignore_index=True)
    records = records.sort_values(
        ["datetime_utc", "platform_id", "latitude", "longitude"],
        na_position="last",
    )
    records.to_csv(detail_out, index=False, encoding="utf-8-sig")

    summary = build_summary(records)
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print("\nExtraction complete")
    print(f"AREA: {AREA}")
    print(f"Raw records in area: {total_area_count}")
    print(f"Dropped land records: {total_dropped_land_count}")
    print(f"Ocean records: {len(records)}")
    print(f"Platforms: {len(summary)}")
    print(f"Time range: {records['datetime_utc'].min()} to {records['datetime_utc'].max()}")
    print(f"Detail output: {detail_out}")
    print(f"Summary output: {summary_out}")


if __name__ == "__main__":
    main()
