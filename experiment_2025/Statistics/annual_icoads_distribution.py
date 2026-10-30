"""Summarize the annual ICOADS observation distribution.

Only NetCDF files marked updated in manifest.json are used. A record is
counted when it is inside AREA, has valid ICOADS wind direction and speed,
and is outside Natural Earth land polygons (lakes are treated as water).
The script writes a compact spatial grid, a platform summary and monthly
totals; it does not materialize the full year's observations.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from land_mask import load_land_union, load_lake_union, records_on_land_mask


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ICOADS_ROOT = DEFAULT_ROOT / "icoads"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
AREA = [42.0, 103.0, 13.0, 130.0]  # [lat_max, lon_min, lat_min, lon_max]
YEAR = 2025
GRID_SIZE_DEG = 1.0


def chartostring(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind == "S" and arr.ndim == 2:
        return np.asarray(nc.chartostring(arr))
    return arr.astype(str)


def numeric_values(variable) -> np.ndarray:
    return np.ma.asarray(variable[:]).astype(float).filled(np.nan)


def platform_values(variable) -> np.ndarray:
    values = chartostring(variable[:])
    values = np.char.strip(values.astype(str))
    values = np.where(np.isin(np.char.lower(values), ["", "nan", "none", "--"]), "UNKNOWN", values)
    return values


def int_values(variable) -> np.ndarray:
    return numeric_values(variable)


def iter_manifest_files(root: Path, year: int):
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("records", {})
    selected = []
    for stamp, row in sorted(rows.items()):
        if row.get("kind") != "updated" or row.get("format", "nc") != "nc":
            continue
        try:
            day = datetime.strptime(stamp, "%Y-%m-%d")
        except ValueError:
            continue
        if day.year != year:
            continue
        path = root / row["path"]
        selected.append((day, path, row))
    return manifest, selected


def make_grid(grid_size: float) -> dict[tuple[int, int], dict[str, float]]:
    lat_max, lon_min, lat_min, lon_max = AREA
    n_lon = int(math.ceil((lon_max - lon_min) / grid_size))
    n_lat = int(math.ceil((lat_max - lat_min) / grid_size))
    grid = {}
    for lat_index in range(n_lat):
        for lon_index in range(n_lon):
            grid[(lat_index, lon_index)] = {
                "grid_lat_index": lat_index,
                "grid_lon_index": lon_index,
                "grid_lat_min": lat_min + lat_index * grid_size,
                "grid_lat_max": min(lat_max, lat_min + (lat_index + 1) * grid_size),
                "grid_lon_min": lon_min + lon_index * grid_size,
                "grid_lon_max": min(lon_max, lon_min + (lon_index + 1) * grid_size),
                "record_count": 0,
            }
    return grid


def update_platforms(platforms: dict[str, dict], ids, types, indicators, lat, lon, day, hour, speed, direction):
    for platform_id in np.unique(ids):
        take = ids == platform_id
        if not np.any(take):
            continue
        row = platforms.setdefault(
            str(platform_id),
            {
                "platform_id": str(platform_id),
                "obs_count": 0,
                "sum_latitude": 0.0,
                "sum_longitude": 0.0,
                "min_latitude": float("inf"),
                "max_latitude": float("-inf"),
                "min_longitude": float("inf"),
                "max_longitude": float("-inf"),
                "first_time": None,
                "last_time": None,
                "wind_speed_count": 0,
                "wind_dir_count": 0,
                "platform_types": set(),
                "id_indicators": set(),
            },
        )
        count = int(take.sum())
        row["obs_count"] += count
        row["sum_latitude"] += float(np.sum(lat[take]))
        row["sum_longitude"] += float(np.sum(lon[take]))
        row["min_latitude"] = min(row["min_latitude"], float(np.min(lat[take])))
        row["max_latitude"] = max(row["max_latitude"], float(np.max(lat[take])))
        row["min_longitude"] = min(row["min_longitude"], float(np.min(lon[take])))
        row["max_longitude"] = max(row["max_longitude"], float(np.max(lon[take])))
        row["wind_speed_count"] += int(np.isfinite(speed[take]).sum())
        row["wind_dir_count"] += int(np.isfinite(direction[take]).sum())
        row["platform_types"].update(str(int(value)) for value in types[take] if np.isfinite(value))
        row["id_indicators"].update(str(int(value)) for value in indicators[take] if np.isfinite(value))
        hours = hour[take]
        first = day + timedelta(hours=float(np.min(hours)))
        last = day + timedelta(hours=float(np.max(hours)))
        if row["first_time"] is None or first < row["first_time"]:
            row["first_time"] = first
        if row["last_time"] is None or last > row["last_time"]:
            row["last_time"] = last


def read_one_file(path: Path, day: datetime, grid: dict, platforms: dict, monthly: dict, area, land_union, lake_union):
    with nc.Dataset(path) as ds:
        required = ["lat", "lon", "HR", "D", "W", "ID", "PT", "II"]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(f"{path.name} missing variables: {missing}")
        lat = numeric_values(ds.variables["lat"])
        lon = numeric_values(ds.variables["lon"])
        hour = numeric_values(ds.variables["HR"])
        direction = numeric_values(ds.variables["D"])
        speed = numeric_values(ds.variables["W"])
        ids = platform_values(ds.variables["ID"])
        types = int_values(ds.variables["PT"])
        indicators = int_values(ds.variables["II"])

    lat_max, lon_min, lat_min, lon_max = area
    finite_location = np.isfinite(lat) & np.isfinite(lon)
    in_area = finite_location & (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    area_count = int(in_area.sum())
    if not area_count:
        return {"raw_area": 0, "land_removed": 0, "valid_wind": 0, "missing_wind": 0}

    indices = np.flatnonzero(in_area)
    on_land = records_on_land_mask(lon[indices], lat[indices], land_union, lake_union)
    ocean_indices = indices[~on_land]
    land_removed = int(on_land.sum())
    valid = (
        np.isfinite(speed[ocean_indices])
        & np.isfinite(direction[ocean_indices])
        & np.isfinite(hour[ocean_indices])
        & (speed[ocean_indices] >= 0.0)
        & (speed[ocean_indices] <= 75.0)
        & (direction[ocean_indices] >= 1.0)
        & (direction[ocean_indices] <= 360.0)
    )
    selected = ocean_indices[valid]
    month = day.month
    monthly[month]["raw_area_records"] += area_count
    monthly[month]["land_records_removed"] += land_removed
    monthly[month]["ocean_records"] += int(ocean_indices.size)
    monthly[month]["valid_wind_records"] += int(selected.size)
    monthly[month]["updated_file_count"] += 1
    if selected.size:
        lat_sel, lon_sel = lat[selected], lon[selected]
        n_lat = int(math.ceil((lat_max - lat_min) / GRID_SIZE_DEG))
        n_lon = int(math.ceil((lon_max - lon_min) / GRID_SIZE_DEG))
        lat_idx = np.clip(np.floor((lat_sel - lat_min) / GRID_SIZE_DEG).astype(int), 0, n_lat - 1)
        lon_idx = np.clip(np.floor((lon_sel - lon_min) / GRID_SIZE_DEG).astype(int), 0, n_lon - 1)
        unique_cells, cell_counts = np.unique(np.column_stack([lat_idx, lon_idx]), axis=0, return_counts=True)
        for key, count in zip(unique_cells, cell_counts):
            grid[(int(key[0]), int(key[1]))]["record_count"] += int(count)
        update_platforms(
            platforms, ids[selected], types[selected], indicators[selected],
            lat_sel, lon_sel, day, hour[selected], speed[selected], direction[selected],
        )
    monthly[month]["missing_wind_records"] += int(ocean_indices.size - selected.size)
    return {
        "raw_area": area_count,
        "land_removed": land_removed,
        "valid_wind": int(selected.size),
        "missing_wind": int(ocean_indices.size - selected.size),
    }


def write_outputs(output_root: Path, grid, platforms, monthly, summary):
    output_root.mkdir(parents=True, exist_ok=True)
    grid_frame = pd.DataFrame(grid.values()).sort_values(["grid_lat_index", "grid_lon_index"])
    grid_frame.to_csv(output_root / "icoads_2025_grid_counts_1deg.csv", index=False, encoding="utf-8-sig")

    platform_rows = []
    for row in platforms.values():
        count = row["obs_count"]
        platform_rows.append(
            {
                "platform_id": row["platform_id"],
                "obs_count": count,
                "first_time": row["first_time"].isoformat(sep=" ") if row["first_time"] else "",
                "last_time": row["last_time"].isoformat(sep=" ") if row["last_time"] else "",
                "mean_latitude": row["sum_latitude"] / count,
                "mean_longitude": row["sum_longitude"] / count,
                "min_latitude": row["min_latitude"],
                "max_latitude": row["max_latitude"],
                "min_longitude": row["min_longitude"],
                "max_longitude": row["max_longitude"],
                "lat_range_deg": row["max_latitude"] - row["min_latitude"],
                "lon_range_deg": row["max_longitude"] - row["min_longitude"],
                "id_indicator_values": ",".join(sorted(row["id_indicators"])),
                "platform_type_values": ",".join(sorted(row["platform_types"])),
                "wind_dir_count": row["wind_dir_count"],
                "wind_speed_count": row["wind_speed_count"],
            }
        )
    platform_frame = pd.DataFrame(platform_rows)
    if not platform_frame.empty:
        platform_frame = platform_frame.sort_values(["obs_count", "platform_id"], ascending=[False, True])
    platform_frame.to_csv(output_root / "icoads_2025_platform_summary.csv", index=False, encoding="utf-8-sig")

    monthly_frame = pd.DataFrame([monthly[index] for index in range(1, 13)])
    monthly_frame.to_csv(output_root / "icoads_2025_monthly_counts.csv", index=False, encoding="utf-8-sig")
    (output_root / "icoads_distribution_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icoads-root", type=Path, default=DEFAULT_ICOADS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--year", type=int, default=YEAR)
    args = parser.parse_args(argv)

    manifest, files = iter_manifest_files(args.icoads_root, args.year)
    if not files:
        raise FileNotFoundError(f"No updated NetCDF files for {args.year} in {args.icoads_root}")
    area = AREA
    lat_max, lon_min, lat_min, lon_max = area
    land_union = load_land_union(lon_min, lat_min, lon_max, lat_max)
    lake_union = load_lake_union(lon_min, lat_min, lon_max, lat_max)
    grid = make_grid(GRID_SIZE_DEG)
    platforms = {}
    monthly = {
        month: {
            "year": args.year, "month": month, "updated_file_count": 0,
            "raw_area_records": 0, "land_records_removed": 0, "ocean_records": 0,
            "valid_wind_records": 0, "missing_wind_records": 0,
        }
        for month in range(1, 13)
    }
    skipped = []
    totals = defaultdict(int)
    for index, (day, path, row) in enumerate(files, start=1):
        if not path.exists():
            skipped.append({"date": day.strftime("%Y-%m-%d"), "path": str(path), "error": "missing file"})
            continue
        try:
            result = read_one_file(path, day, grid, platforms, monthly, area, land_union, lake_union)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"date": day.strftime("%Y-%m-%d"), "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        for key, value in result.items():
            totals[key] += value
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {day:%Y-%m-%d}: valid wind {result['valid_wind']:,}")

    summary = {
        "year": args.year,
        "area": area,
        "grid_size_deg": GRID_SIZE_DEG,
        "source_policy": "manifest kind=updated and format=nc; preliminary and DAT.GZ excluded",
        "updated_files_in_manifest": len(files),
        "files_read": len(files) - len(skipped),
        "skipped_files": skipped,
        "raw_area_records": totals["raw_area"],
        "land_and_lake_records_removed": totals["land_removed"],
        "ocean_records": totals["raw_area"] - totals["land_removed"],
        "valid_wind_records": totals["valid_wind"],
        "missing_or_invalid_wind_records": totals["missing_wind"],
        "platform_count": len(platforms),
        "nonempty_grid_count": sum(row["record_count"] > 0 for row in grid.values()),
        "mask": "Natural Earth 10m land and lakes; land boundaries are land and lake polygons are water",
        "manifest_start_date": manifest.get("start_date"),
        "manifest_end_date": manifest.get("end_date"),
    }
    write_outputs(args.output_root, grid, platforms, monthly, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV/JSON outputs: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
