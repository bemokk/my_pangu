"""Seasonal East China Sea wind-speed verification using the supplied GB/T mask."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from shapely.geometry import shape
import shapely

STATS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATS_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(STATS_ROOT))

from experiment_2025.Statistics.statistics import (  # noqa: E402
    DEFAULT_ICOADS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GridSampler,
    iter_targets,
    load_daily_updated,
    make_partial,
    metrics,
    model_path,
)


MASK_ZIP = STATS_ROOT / "east_china_sea_gbt36109_2018.zip"
MASK_MEMBER = "east_china_sea_gbt36109_2018/east_china_sea_gbt36109_approx.geojson"
# Broad rectangle used only to load candidate ICOADS observations before polygon filtering.
LOAD_AREA = [36.0, 117.0, 23.0, 133.0]  # lat_max, lon_min, lat_min, lon_max
SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
SEASON_ORDER = tuple(SEASON_MONTHS)
LEADS = tuple(range(0, 121, 3))
EXPERIMENTS = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")


def load_east_china_sea_polygon(zip_path: Path):
    with ZipFile(zip_path) as archive:
        geojson = json.loads(archive.read(MASK_MEMBER).decode("utf-8"))
    polygon = shape(geojson["features"][0]["geometry"])
    if not polygon.is_valid:
        raise ValueError("East China Sea polygon is invalid")
    if hasattr(shapely, "prepare"):
        shapely.prepare(polygon)
    return polygon, geojson["features"][0].get("properties", {})


def filter_region(observations: dict, polygon) -> dict:
    inside = shapely.intersects_xy(polygon, observations["lon"], observations["lat"])
    return {key: values[inside] for key, values in observations.items()}


def season_for_month(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season
    raise ValueError(month)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> tuple[list[dict], list[dict], dict]:
    polygon, polygon_properties = load_east_china_sea_polygon(args.mask_zip)
    obs_by_time, obs_info = load_daily_updated(args.icoads_root, LOAD_AREA)
    region_obs = {}
    for valid_time, observations in obs_by_time.items():
        selected = filter_region(observations, polygon)
        if selected["lat"].size:
            region_obs[valid_time] = selected

    first_model = model_path(args.output_root, "era5_direct", datetime(2025, 1, 1), 0)
    if not first_model.exists():
        raise FileNotFoundError(first_model)
    sampler = GridSampler(first_model, LOAD_AREA)

    daily_rows: list[dict] = []
    missing: list[dict] = []
    metric_values = defaultdict(lambda: {"rmse": [], "mae": [], "n": []})
    task_specs = []
    for target in iter_targets(args.start_date, args.end_date):
        for lead in LEADS:
            valid_time = target + timedelta(hours=lead)
            observations = region_obs.get(valid_time)
            if observations is None:
                continue
            season = season_for_month(valid_time.month)
            for experiment in args.experiments:
                task_specs.append((experiment, target, lead, valid_time, season, observations))

    total_tasks = len(task_specs)
    completed = 0
    lock = threading.Lock()

    def task(spec):
        experiment, target, lead, valid_time, season, observations = spec
        path = model_path(args.output_root, experiment, target, lead)
        if not path.is_file():
            return experiment, target, lead, valid_time, season, None, {
                "experiment": experiment,
                "target_time": target.strftime("%Y-%m-%d-%H-%M"),
                "valid_time": valid_time.strftime("%Y-%m-%d-%H-%M"),
                "lead_hour": lead,
                "path": str(path),
                "reason": "missing_output",
            }
        try:
            pred_u, pred_v = sampler.sample(path, observations["lat"], observations["lon"])
            part = make_partial(pred_u, pred_v, observations["speed"], observations["direction"])
            daily = metrics(part)
            return experiment, target, lead, valid_time, season, daily, None
        except Exception as exc:  # noqa: BLE001
            return experiment, target, lead, valid_time, season, None, {
                "experiment": experiment,
                "target_time": target.strftime("%Y-%m-%d-%H-%M"),
                "valid_time": valid_time.strftime("%Y-%m-%d-%H-%M"),
                "lead_hour": lead,
                "path": str(path),
                "reason": f"sample_failed: {type(exc).__name__}: {exc}",
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, spec) for spec in task_specs]
        for future in as_completed(futures):
            experiment, target, lead, valid_time, season, daily, error = future.result()
            if daily is not None and np.isfinite(daily["rmse"]) and np.isfinite(daily["mae"]):
                metric_values[(experiment, season, lead)]["rmse"].append(float(daily["rmse"]))
                metric_values[(experiment, season, lead)]["mae"].append(float(daily["mae"]))
                metric_values[(experiment, season, lead)]["n"].append(int(daily["n"]))
                daily_rows.append(
                    {
                        "experiment": experiment,
                        "target_time": target.strftime("%Y-%m-%d-%H-%M"),
                        "valid_time": valid_time.strftime("%Y-%m-%d-%H-%M"),
                        "season": season,
                        "lead_hour": lead,
                        "n": int(daily["n"]),
                        "rmse": float(daily["rmse"]),
                        "mae": float(daily["mae"]),
                    }
                )
            if error:
                missing.append(error)
            with lock:
                completed += 1
                if completed % args.progress_every == 0 or completed == total_tasks:
                    print(f"[东海统计进度] {completed}/{total_tasks} 任务", flush=True)

    rows = []
    for season in SEASON_ORDER:
        for lead in LEADS:
            for experiment in args.experiments:
                values = metric_values[(experiment, season, lead)]
                rows.append(
                    {
                        "experiment": experiment,
                        "season": season,
                        "season_months": ",".join(str(m) for m in SEASON_MONTHS[season]),
                        "lead_hour": lead,
                        "n_total": int(sum(values["n"])),
                        "rmse": float(np.mean(values["rmse"])) if values["rmse"] else np.nan,
                        "mae": float(np.mean(values["mae"])) if values["mae"] else np.nan,
                        "daily_count": len(values["rmse"]),
                    }
                )
    daily_rows.sort(key=lambda row: (row["experiment"], row["season"], row["lead_hour"], row["valid_time"]))
    rows.sort(key=lambda row: (SEASON_ORDER.index(row["season"]), row["lead_hour"], row["experiment"]))
    summary = {
        "mask_zip": str(args.mask_zip),
        "mask_member": MASK_MEMBER,
        "mask_properties": polygon_properties,
        "mask_status": "analyst-generated approximate reconstruction; NOT official digital boundary",
        "mask_bounds_lonlat": list(polygon.bounds),
        "load_area": LOAD_AREA,
        "seasons": {season: list(months) for season, months in SEASON_MONTHS.items()},
        "leads": list(LEADS),
        "experiments": list(args.experiments),
        "aggregation_method": "daily_arithmetic_mean",
        "metric_definition": "Daily wind-speed RMSE and MAE are computed over East China Sea ICOADS/model pairs, then averaged equally over valid target days.",
        "missing_outputs": len(missing),
        "missing": missing,
        "observation_info": obs_info,
        "region_observation_times": len(region_obs),
        "tasks": total_tasks,
    }
    return rows, daily_rows, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-zip", type=Path, default=MASK_ZIP)
    parser.add_argument("--icoads-root", type=Path, default=DEFAULT_ICOADS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=STATS_ROOT / "data")
    parser.add_argument("--start-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2025, 1, 1))
    parser.add_argument("--end-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2025, 12, 31))
    parser.add_argument("--workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    rows, daily_rows, summary = run(args)
    args.data_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.data_root / "east_china_sea_wind_speed_metrics_by_season_lead_2025.csv", rows)
    write_csv(args.data_root / "east_china_sea_wind_speed_daily_by_target_2025.csv", daily_rows)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    (args.data_root / "east_china_sea_wind_speed_metrics_summary_2025.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"CSV: {args.data_root / 'east_china_sea_wind_speed_metrics_by_season_lead_2025.csv'}")
    print(f"Daily CSV: {args.data_root / 'east_china_sea_wind_speed_daily_by_target_2025.csv'}")
    print(f"Missing outputs: {summary['missing_outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
