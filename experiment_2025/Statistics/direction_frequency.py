"""Compute monthly ICOADS/model wind-direction frequencies for radar plots."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

STATS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATS_ROOT.parents[1]
if str(STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(STATS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_2025.core.experiments import EXPERIMENTS
from statistics import (
    AREA,
    DEFAULT_ICOADS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GridSampler,
    iter_targets,
    load_daily_updated,
    model_path,
    wind_direction,
)


MASK_ZIP = STATS_ROOT / "east_china_sea_gbt36109_2018.zip"
MASK_MEMBER = "east_china_sea_gbt36109_2018/east_china_sea_gbt36109_approx.geojson"
# Broad rectangle used only to load candidate ICOADS records before polygon filtering.
LOAD_AREA = [36.0, 117.0, 23.0, 133.0]  # lat_max, lon_min, lat_min, lon_max
MONTHS = (1, 8)
LEADS = (24, 72, 120)
DIRECTION_SECTORS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)
EXPERIMENT_ORDER = ("era5_direct", "era5_lagged_120h", "gdas_direct", "gdas_lagged_120h")
SERIES_LABELS = {
    "observed": "Observed",
    "era5_direct": "E0",
    "era5_lagged_120h": "E120",
    "gdas_direct": "G0",
    "gdas_lagged_120h": "G120",
}


def direction_sector(values: np.ndarray) -> np.ndarray:
    values = np.mod(np.asarray(values, dtype=float), 360.0)
    return np.floor(values / 22.5).astype(int) % 16


def add_counts(target: dict[tuple[int, int, str, int], int], month: int, lead: int, series: str, values: np.ndarray) -> None:
    sectors = direction_sector(values)
    unique, counts = np.unique(sectors, return_counts=True)
    for sector, count in zip(unique, counts):
        target[(month, lead, series, int(sector))] += int(count)


def load_east_china_sea_polygon(zip_path: Path):
    """Load the approximate GB/T 36109-2018 East China Sea polygon."""
    with ZipFile(zip_path) as archive:
        geojson = json.loads(archive.read(MASK_MEMBER).decode("utf-8"))
    polygon = shape(geojson["features"][0]["geometry"])
    if not polygon.is_valid:
        raise ValueError(f"Invalid East China Sea polygon in {zip_path}")
    if hasattr(shapely, "prepare"):
        shapely.prepare(polygon)
    return polygon, geojson["features"][0].get("properties", {})


def filter_region(observations: dict, polygon) -> dict:
    """Keep ICOADS records inside or on the East China Sea polygon."""
    inside = shapely.intersects_xy(polygon, observations["lon"], observations["lat"])
    return {key: values[inside] for key, values in observations.items()}


def run(args) -> tuple[pd.DataFrame, dict]:
    polygon, polygon_properties = load_east_china_sea_polygon(args.mask_zip)
    obs_by_time, obs_info = load_daily_updated(args.icoads_root, LOAD_AREA)
    region_obs = {}
    for valid_time, observations in obs_by_time.items():
        selected = filter_region(observations, polygon)
        if selected["lat"].size:
            region_obs[valid_time] = selected

    first_model = model_path(args.output_root, "era5_direct", datetime(2025, 1, 1), 0)
    if not first_model.exists():
        raise FileNotFoundError(f"Model output not found: {first_model}")
    sampler = GridSampler(first_model, LOAD_AREA)
    counts: dict[tuple[int, int, str, int], int] = defaultdict(int)
    totals: dict[tuple[int, int, str], int] = defaultdict(int)
    missing_outputs = []

    for month in MONTHS:
        valid_times = sorted(time for time in region_obs if time.month == month)
        for lead in LEADS:
            for valid_time in valid_times:
                target = valid_time - timedelta(hours=lead)
                if target < args.start_date or target > args.end_date:
                    continue
                if target.hour != 0:
                    continue
                observations = region_obs[valid_time]
                obs_direction = observations["direction"]
                valid_obs = np.isfinite(obs_direction)
                if np.any(valid_obs):
                    obs_values = obs_direction[valid_obs]
                    add_counts(counts, month, lead, "observed", obs_values)
                    totals[(month, lead, "observed")] += int(obs_values.size)

                for experiment in EXPERIMENT_ORDER:
                    path = model_path(args.output_root, experiment, target, lead)
                    if not path.is_file():
                        missing_outputs.append(
                            {
                                "experiment": experiment,
                                "target_time": target.strftime("%Y-%m-%d-%H-%M"),
                                "valid_time": valid_time.strftime("%Y-%m-%d-%H-%M"),
                                "lead_hour": lead,
                                "path": str(path),
                            }
                        )
                        continue
                    pred_u, pred_v = sampler.sample(path, observations["lat"], observations["lon"])
                    valid = np.isfinite(pred_u) & np.isfinite(pred_v) & valid_obs
                    if not np.any(valid):
                        continue
                    predicted = wind_direction(pred_u[valid], pred_v[valid])
                    add_counts(counts, month, lead, experiment, predicted)
                    totals[(month, lead, experiment)] += int(predicted.size)

    rows = []
    series_order = ("observed",) + EXPERIMENT_ORDER
    for month in MONTHS:
        for lead in LEADS:
            for series in series_order:
                total = totals[(month, lead, series)]
                for sector_code, sector in enumerate(DIRECTION_SECTORS):
                    count = counts[(month, lead, series, sector_code)]
                    rows.append(
                        {
                            "month": month,
                            "month_label": datetime(2025, month, 1).strftime("%B"),
                            "lead_hour": lead,
                            "series": series,
                            "series_label": SERIES_LABELS[series],
                            "direction_sector": sector,
                            "sector_code": sector_code,
                            "count": count,
                            "frequency": count / total if total else np.nan,
                            "sample_count": total,
                        }
                    )
    frame = pd.DataFrame(rows)
    summary = {
        "mask_zip": str(args.mask_zip),
        "mask_member": MASK_MEMBER,
        "mask_properties": polygon_properties,
        "mask_status": "analyst-generated approximate reconstruction; NOT official digital boundary",
        "mask_bounds_lonlat": list(polygon.bounds),
        "load_area": LOAD_AREA,
        "months": list(MONTHS),
        "leads": list(LEADS),
        "direction_sectors": list(DIRECTION_SECTORS),
        "source_policy": "ICOADS manifest kind=updated and format=nc only; Natural Earth 10m land/lake mask applied",
        "experiments": list(EXPERIMENT_ORDER),
        "missing_outputs": missing_outputs,
        "observation_info": obs_info,
        "region_observation_times": len(region_obs),
        "sample_counts": {
            f"{month}-{lead}-{series}": totals[(month, lead, series)]
            for month in MONTHS for lead in LEADS for series in series_order
        },
    }
    return frame, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icoads-root", type=Path, default=DEFAULT_ICOADS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mask-zip", type=Path, default=MASK_ZIP)
    parser.add_argument("--data-root", type=Path, default=STATS_ROOT / "data")
    parser.add_argument("--start-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2025, 1, 1))
    parser.add_argument("--end-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2025, 12, 31, 23, 59, 59))
    args = parser.parse_args(argv)
    frame, summary = run(args)
    args.data_root.mkdir(parents=True, exist_ok=True)
    output_csv = args.data_root / "east_china_sea_wind_direction_frequency_jan_aug.csv"
    output_summary = args.data_root / "east_china_sea_wind_direction_frequency_jan_aug_summary.json"
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"CSV: {output_csv}")
    print(f"Summary: {output_summary}")
    print(f"Missing outputs: {len(summary['missing_outputs'])}")
    print(frame.groupby(["month", "lead_hour", "series"], as_index=False)["sample_count"].first().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
