"""Sample-pooled wind MAE and ICOADS counts by sea and Beaufort group.

The four sea polygons come from ``china_four_seas_mask_merged_south.zip``.
South China Sea observations are additionally restricted to latitude > 13 N.
Only two result files are written: one long CSV table and one JSON summary.
"""
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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import shapely
from shapely.geometry import shape


STATS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATS_ROOT.parents[1]
if str(STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(STATS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_2025.Statistics.statistics import (  # noqa: E402
    AREA,
    DEFAULT_ICOADS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GridSampler,
    angular_error,
    iter_targets,
    load_daily_updated,
    model_path,
    wind_direction,
)


MASK_ZIP = STATS_ROOT / "china_four_seas_mask_merged_south.zip"
MASK_MEMBER = "china_four_seas_mask/four_seas_approx.geojson"
EXPERIMENTS = (
    "gdas_direct",
    "era5_lagged_120h",
    "era5_direct",
    "gdas_lagged_120h",
)
LEADS = tuple(range(0, 121, 3))
SEA_NAMES = {1: "Bohai Sea", 2: "Yellow Sea", 3: "East China Sea", 4: "South China Sea"}
SEA_NAMES_ZH = {1: "渤海", 2: "黄海", 3: "东海", 4: "南海北部"}
BEAUFORT_LABELS = ("<=2", "3", "4", "5", "6", "7", "8", "9", "10", "11", ">=12")
# np.digitize with right=False gives [edge_i, edge_i+1), matching Beaufort thresholds.
BEAUFORT_EDGES = np.asarray((3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7))
OVERALL_LABEL = "overall"


@dataclass
class MaeSums:
    n: int = 0
    speed_abs_sum: float = 0.0
    direction_abs_sum: float = 0.0

    def add(self, n: int, speed_abs_sum: float, direction_abs_sum: float) -> None:
        self.n += int(n)
        self.speed_abs_sum += float(speed_abs_sum)
        self.direction_abs_sum += float(direction_abs_sum)


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def beaufort_codes(speed: np.ndarray) -> np.ndarray:
    """Return integer codes for <=2, 3, ..., 11, >=12."""
    return np.digitize(speed, BEAUFORT_EDGES, right=False).astype(np.int8)


def load_sea_polygons(zip_path: Path) -> tuple[list[tuple[int, object]], dict]:
    with ZipFile(zip_path) as archive:
        geojson = json.loads(archive.read(MASK_MEMBER).decode("utf-8"))
    polygons = []
    feature_properties = []
    for feature in geojson["features"]:
        sea_id = int(feature["properties"]["sea_id"])
        polygon = shape(feature["geometry"])
        if not polygon.is_valid:
            raise ValueError(f"Invalid polygon for sea_id={sea_id}")
        if hasattr(shapely, "prepare"):
            shapely.prepare(polygon)
        polygons.append((sea_id, polygon))
        feature_properties.append(feature.get("properties", {}))
    polygons.sort(key=lambda item: item[0])
    return polygons, {
        "member": MASK_MEMBER,
        "features": feature_properties,
        "boundary_rule": "intersects/covers; lower sea_id wins shared boundaries",
    }


def add_region_and_beaufort(observations: dict, polygons: list[tuple[int, object]]) -> dict:
    """Classify observations; zero means outside all four sea polygons."""
    lat = observations["lat"]
    lon = observations["lon"]
    sea_id = np.zeros(lat.shape, dtype=np.int8)
    for candidate_id, polygon in polygons:
        take = (sea_id == 0) & shapely.intersects_xy(polygon, lon, lat)
        if candidate_id == 4:
            take &= lat > 13.0
        sea_id[take] = candidate_id
    selected = sea_id > 0
    if not np.any(selected):
        return {}
    result = {key: np.asarray(values)[selected] for key, values in observations.items()}
    result["sea_id"] = sea_id[selected]
    result["beaufort_code"] = beaufort_codes(result["speed"])
    return result


def observation_counts(
    observations_by_time: dict[datetime, dict], eligible_times: set[datetime]
) -> dict[tuple[int, str], int]:
    """Count each ICOADS record once, independent of experiment and lead."""
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for valid_time, observations in observations_by_time.items():
        if valid_time not in eligible_times:
            continue
        for sea_id in SEA_NAMES:
            in_sea = observations["sea_id"] == sea_id
            counts[(sea_id, OVERALL_LABEL)] += int(np.count_nonzero(in_sea))
            for code, label in enumerate(BEAUFORT_LABELS):
                counts[(sea_id, label)] += int(
                    np.count_nonzero(in_sea & (observations["beaufort_code"] == code))
                )
    return counts


def task_metrics(pred_u: np.ndarray, pred_v: np.ndarray, observations: dict) -> list[tuple[int, str, int, float, float]]:
    pred_speed = np.sqrt(pred_u * pred_u + pred_v * pred_v)
    pred_direction = wind_direction(pred_u, pred_v)
    valid = (
        np.isfinite(pred_speed)
        & np.isfinite(pred_direction)
        & np.isfinite(observations["speed"])
        & np.isfinite(observations["direction"])
    )
    speed_abs = np.abs(pred_speed - observations["speed"])
    direction_abs = np.abs(angular_error(pred_direction, observations["direction"]))
    results = []
    for sea_id in SEA_NAMES:
        in_sea = valid & (observations["sea_id"] == sea_id)
        if not np.any(in_sea):
            continue
        results.append(
            (
                sea_id,
                OVERALL_LABEL,
                int(np.count_nonzero(in_sea)),
                float(np.sum(speed_abs[in_sea])),
                float(np.sum(direction_abs[in_sea])),
            )
        )
        for code, label in enumerate(BEAUFORT_LABELS):
            take = in_sea & (observations["beaufort_code"] == code)
            if np.any(take):
                results.append(
                    (
                        sea_id,
                        label,
                        int(np.count_nonzero(take)),
                        float(np.sum(speed_abs[take])),
                        float(np.sum(direction_abs[take])),
                    )
                )
    return results


def find_grid_sample(args) -> Path:
    for experiment in args.experiments:
        for target in iter_targets(args.start_date, args.end_date):
            for lead in LEADS:
                path = model_path(args.output_root, experiment, target, lead)
                if path.is_file():
                    return path
    raise FileNotFoundError("No model wind output was found in the requested date range")


def run(args) -> tuple[list[dict], dict]:
    polygons, mask_info = load_sea_polygons(args.mask_zip)
    raw_observations, observation_info = load_daily_updated(args.icoads_root, AREA)
    region_observations = {}
    for valid_time, observations in raw_observations.items():
        classified = add_region_and_beaufort(observations, polygons)
        if classified:
            region_observations[valid_time] = classified

    target_lead_pairs = [
        (target, lead, target + timedelta(hours=lead))
        for target in iter_targets(args.start_date, args.end_date)
        for lead in LEADS
    ]
    eligible_times = {valid_time for _, _, valid_time in target_lead_pairs}
    ico_counts = observation_counts(region_observations, eligible_times)

    grid_sample = find_grid_sample(args)
    sampler = GridSampler(grid_sample, AREA)
    task_specs = []
    for target, lead, valid_time in target_lead_pairs:
        observations = region_observations.get(valid_time)
        if observations is None:
            continue
        for experiment in args.experiments:
            task_specs.append((experiment, target, lead, valid_time, observations))

    totals: dict[tuple[str, int, str], MaeSums] = defaultdict(MaeSums)
    missing = []
    completed = 0
    progress_lock = threading.Lock()

    def task(spec):
        experiment, target, lead, valid_time, observations = spec
        path = model_path(args.output_root, experiment, target, lead)
        if not path.is_file():
            return experiment, None, {
                "experiment": experiment,
                "target_time": target.strftime("%Y-%m-%d-%H-%M"),
                "valid_time": valid_time.strftime("%Y-%m-%d-%H-%M"),
                "lead_hour": lead,
                "path": str(path),
                "reason": "missing_output",
            }
        try:
            pred_u, pred_v = sampler.sample(path, observations["lat"], observations["lon"])
            return experiment, task_metrics(pred_u, pred_v, observations), None
        except Exception as exc:  # noqa: BLE001
            return experiment, None, {
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
            experiment, partials, error = future.result()
            if partials:
                for sea_id, group, n, speed_abs_sum, direction_abs_sum in partials:
                    totals[(experiment, sea_id, group)].add(n, speed_abs_sum, direction_abs_sum)
            if error:
                missing.append(error)
            with progress_lock:
                completed += 1
                if completed % args.progress_every == 0 or completed == len(task_specs):
                    print(f"[四海域风级统计] {completed}/{len(task_specs)} 任务", flush=True)

    rows = []
    group_order = (*BEAUFORT_LABELS, OVERALL_LABEL)
    for sea_id in SEA_NAMES:
        for group in group_order:
            for experiment in args.experiments:
                values = totals[(experiment, sea_id, group)]
                rows.append(
                    {
                        "sea_id": sea_id,
                        "sea_name_zh": SEA_NAMES_ZH[sea_id],
                        "sea_name_en": SEA_NAMES[sea_id],
                        "beaufort_group": group,
                        "experiment": experiment,
                        "wind_speed_mae_ms": values.speed_abs_sum / values.n if values.n else float("nan"),
                        "wind_direction_mae_deg": values.direction_abs_sum / values.n if values.n else float("nan"),
                        "icoads_record_count": ico_counts[(sea_id, group)],
                        "paired_n": values.n,
                    }
                )

    summary = {
        "start_date": args.start_date.strftime("%Y-%m-%d"),
        "end_date": args.end_date.strftime("%Y-%m-%d"),
        "lead_hours": list(LEADS),
        "experiments": list(args.experiments),
        "area": AREA,
        "mask_zip": str(args.mask_zip),
        "mask": mask_info,
        "sea_names": {str(key): value for key, value in SEA_NAMES_ZH.items()},
        "south_china_sea_rule": "sea_id=4 and latitude > 13.0 degrees north",
        "beaufort_groups": list(BEAUFORT_LABELS),
        "beaufort_edges_ms": BEAUFORT_EDGES.tolist(),
        "beaufort_basis": "ICOADS observed wind speed",
        "aggregation_method": "sample_pooled over all valid observation-forecast pairs and lead hours 0..120",
        "wind_speed_mae": "mean(abs(sqrt(pred_u^2 + pred_v^2) - observed_speed))",
        "wind_direction_mae": "mean(abs(((pred_direction - observed_direction + 180) % 360) - 180))",
        "icoads_record_count": "Each eligible quality-controlled ICOADS record counted once, independent of experiment and lead.",
        "paired_n": "Valid observation-forecast pairs used as the MAE denominator; observations can pair with multiple targets/leads.",
        "observation_policy": "Same load_daily_updated quality control and Natural Earth land/lake exclusion as Statistics/statistics.py",
        "observation_info": observation_info,
        "region_observation_times": len(region_observations),
        "grid_sample": str(grid_sample),
        "tasks": len(task_specs),
        "missing_outputs": len(missing),
        "missing": sorted(missing, key=lambda item: (item["experiment"], item["target_time"], item["lead_hour"])),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-zip", type=Path, default=MASK_ZIP)
    parser.add_argument("--icoads-root", type=Path, default=DEFAULT_ICOADS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=STATS_ROOT / "data")
    parser.add_argument("--start-date", type=parse_date, default=datetime(2025, 1, 1))
    parser.add_argument("--end-date", type=parse_date, default=datetime(2025, 12, 31))
    parser.add_argument("--workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args(argv)
    if args.end_date < args.start_date or args.workers < 1:
        parser.error("Invalid date range or worker count")

    started = time.perf_counter()
    rows, summary = run(args)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    csv_path = args.data_root / "four_seas_beaufort_wind_metrics_2025.csv"
    summary_path = args.data_root / "four_seas_beaufort_wind_metrics_2025_summary.json"
    write_csv(csv_path, rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Rows: {len(rows)}; missing outputs: {summary['missing_outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
