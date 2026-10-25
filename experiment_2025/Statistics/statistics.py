"""Statistics for the four experiment wind forecasts against ICOADS daily_updated data.

The program uses only ICOADS records whose manifest ``kind`` is ``updated`` and
whose file is NetCDF.  It samples the model u10/v10 fields at each observation
location with bilinear interpolation, then writes speed and direction RMSE,
MAE, and correlation by lead hour and valid-time month.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np

from experiment_2025.core.experiments import EXPERIMENTS, time_label, wind_path
from land_mask import load_land_union, load_lake_union, records_on_land_mask


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(r"H:\pangu_2025_data\model_output\experiment_2025")
DEFAULT_ICOADS_ROOT = DEFAULT_ROOT / "icoads"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
AREA = [30.0, 105.0, 10.0, 130.0]  # lat_max, lon_min, lat_min, lon_max
EXPERIMENT_NAMES = ("gdas_direct", "era5_lagged_120h", "era5_direct", "gdas_lagged_120h")
LEADS = tuple(range(0, 121, 3))


@dataclass
class Partial:
    n: int = 0
    speed_sse: float = 0.0
    speed_abs: float = 0.0
    speed_sum_pred: float = 0.0
    speed_sum_obs: float = 0.0
    speed_sum_pred2: float = 0.0
    speed_sum_obs2: float = 0.0
    speed_sum_cross: float = 0.0
    dir_sse: float = 0.0
    dir_abs: float = 0.0
    dir_sum_err: float = 0.0
    dir_sin_pred: float = 0.0
    dir_cos_pred: float = 0.0
    dir_sin_obs: float = 0.0
    dir_cos_obs: float = 0.0
    dir_sin_pred2: float = 0.0
    dir_cos_pred2: float = 0.0
    dir_sin_obs2: float = 0.0
    dir_cos_obs2: float = 0.0
    dir_sin_cos_pred: float = 0.0
    dir_sin_cos_obs: float = 0.0
    dir_sinpred_sinobs: float = 0.0
    dir_sinpred_cosobs: float = 0.0
    dir_cospred_sinobs: float = 0.0
    dir_cospred_cosobs: float = 0.0

    def add(self, other: "Partial") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def iter_targets(start: datetime, end: datetime):
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end:
        yield current
        current += timedelta(days=1)


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def chartostring(value) -> np.ndarray:
    """Convert a NetCDF char array into one string per observation."""
    arr = np.asarray(value)
    if arr.dtype.kind == "S" and arr.ndim == 2:
        return nc.chartostring(arr)
    return arr.astype(str)


def numeric_values(variable) -> np.ndarray:
    """Return a masked NetCDF variable as float, including integer variables."""
    return np.ma.asarray(variable[:]).astype(float).filled(np.nan)


def load_daily_updated(ico_root: Path, area: list[float]) -> tuple[dict[datetime, dict[str, np.ndarray]], dict]:
    manifest = json_load(ico_root / "manifest.json")
    rows = manifest.get("records", {})
    obs_by_time: dict[datetime, dict[str, np.ndarray]] = {}
    selected_days = []
    skipped = []
    land_removed = 0
    lat_max, lon_min, lat_min, lon_max = area
    for stamp, row in sorted(rows.items()):
        if row.get("kind") != "updated" or row.get("format", "nc") != "nc":
            continue
        path = ico_root / row["path"]
        try:
            with nc.Dataset(path) as ds:
                ds.set_auto_mask(True)
                dates = chartostring(ds.variables["date"][:])
                hour = numeric_values(ds.variables["HR"])
                lat = numeric_values(ds.variables["lat"])
                lon = numeric_values(ds.variables["lon"])
                speed = numeric_values(ds.variables["W"])
                direction = numeric_values(ds.variables["D"])
        except Exception as exc:  # noqa: BLE001
            skipped.append({"date": stamp, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        date_text = np.char.strip(dates.astype(str))
        wanted = (
            (np.char.str_len(date_text) == 8)
            & np.isfinite(hour)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & np.isfinite(speed)
            & np.isfinite(direction)
            & (lat >= lat_min)
            & (lat <= lat_max)
            & (lon >= lon_min)
            & (lon <= lon_max)
            & (speed >= 0.0)
            & (speed <= 75.0)
            & (direction >= 1.0)
            & (direction <= 360.0)
        )
        candidate_indices = np.flatnonzero(wanted)
        if candidate_indices.size:
            land_union = load_land_union(lon_min, lat_min, lon_max, lat_max)
            lake_union = load_lake_union(lon_min, lat_min, lon_max, lat_max)
            land_mask = records_on_land_mask(
                lon[candidate_indices], lat[candidate_indices], land_union, lake_union
            )
            land_removed += int(land_mask.sum())
            wanted[candidate_indices[land_mask]] = False
        if not np.any(wanted):
            selected_days.append(stamp)
            continue
        date_text = date_text[wanted]
        hour = hour[wanted]
        lat = lat[wanted]
        lon = lon[wanted]
        speed = speed[wanted]
        direction = direction[wanted]
        unique_hours = np.unique(hour.astype(int))
        for hour_value in unique_hours:
            take = np.isclose(hour, hour_value, atol=1e-6)
            if not np.any(take):
                continue
            day = datetime.strptime(str(date_text[np.flatnonzero(take)[0]]), "%Y%m%d").date()
            valid_time = datetime(day.year, day.month, day.day, int(hour_value))
            obs_by_time[valid_time] = {
                "lat": lat[take],
                "lon": lon[take],
                "speed": speed[take],
                "direction": direction[take],
            }
        selected_days.append(stamp)
    return obs_by_time, {
        "daily_updated_days": selected_days,
        "skipped_files": skipped,
        "land_and_lake_records_removed": land_removed,
        "land_records_removed": land_removed,
        "land_mask": "Natural Earth 10m land and lakes; land boundary treated as land, lake polygons treated as water",
    }


class GridSampler:
    """Read only the model grid rectangle surrounding the observation area."""

    def __init__(self, path: Path, area: list[float]):
        with nc.Dataset(path) as ds:
            self.latitude = np.asarray(ds.variables["latitude"][:], dtype=float)
            self.longitude = np.asarray(ds.variables["longitude"][:], dtype=float)
        if self.latitude[0] > self.latitude[-1]:
            self.lat_ascending = self.latitude[::-1]
            self.lat_reversed = True
        else:
            self.lat_ascending = self.latitude
            self.lat_reversed = False
        self.lon_ascending = self.longitude if self.longitude[0] < self.longitude[-1] else self.longitude[::-1]
        self.lon_reversed = self.longitude[0] > self.longitude[-1]
        self.area = area

    def _indices(self, lat: np.ndarray, lon: np.ndarray):
        # Longitude files use 0..359.75. Keep observations in the requested 105..130 box.
        lon = np.mod(lon, 360.0)
        i1 = np.searchsorted(self.lat_ascending, lat, side="right")
        i1 = np.clip(i1, 1, len(self.lat_ascending) - 1)
        i0 = i1 - 1
        j1 = np.searchsorted(self.lon_ascending, lon, side="right")
        j1 = np.clip(j1, 1, len(self.lon_ascending) - 1)
        j0 = j1 - 1
        wl = (lat - self.lat_ascending[i0]) / (self.lat_ascending[i1] - self.lat_ascending[i0])
        wo = (lon - self.lon_ascending[j0]) / (self.lon_ascending[j1] - self.lon_ascending[j0])
        return i0, i1, j0, j1, wl, wo

    def sample(self, path: Path, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i0, i1, j0, j1, wl, wo = self._indices(lat, lon)
        sorted_row_min, sorted_row_max = int(i0.min()), int(i1.max())
        sorted_col_min, sorted_col_max = int(j0.min()), int(j1.max())
        if self.lat_reversed:
            row_min = len(self.latitude) - 1 - sorted_row_max
            row_max = len(self.latitude) - 1 - sorted_row_min
        else:
            row_min, row_max = sorted_row_min, sorted_row_max
        if self.lon_reversed:
            col_min = len(self.longitude) - 1 - sorted_col_max
            col_max = len(self.longitude) - 1 - sorted_col_min
        else:
            col_min, col_max = sorted_col_min, sorted_col_max
        with nc.Dataset(path) as ds:
            ds.set_auto_mask(False)
            u = np.asarray(ds.variables["u10"][row_min : row_max + 1, col_min : col_max + 1], dtype=float)
            v = np.asarray(ds.variables["v10"][row_min : row_max + 1, col_min : col_max + 1], dtype=float)
        if self.lat_reversed:
            u = u[::-1, :]
            v = v[::-1, :]
        if self.lon_reversed:
            u = u[:, ::-1]
            v = v[:, ::-1]
        a0, a1 = i0 - sorted_row_min, i1 - sorted_row_min
        b0, b1 = j0 - sorted_col_min, j1 - sorted_col_min

        def interp(values):
            q00 = values[a0, b0]
            q01 = values[a0, b1]
            q10 = values[a1, b0]
            q11 = values[a1, b1]
            return ((1 - wl) * ((1 - wo) * q00 + wo * q01) + wl * ((1 - wo) * q10 + wo * q11))

        return interp(u), interp(v)


def wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def angular_error(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return (pred - obs + 180.0) % 360.0 - 180.0


def make_partial(pred_u, pred_v, obs_speed, obs_direction) -> Partial:
    pred_speed = np.sqrt(pred_u * pred_u + pred_v * pred_v)
    pred_direction = wind_direction(pred_u, pred_v)
    valid = (
        np.isfinite(pred_speed)
        & np.isfinite(pred_direction)
        & np.isfinite(obs_speed)
        & np.isfinite(obs_direction)
    )
    pred_speed = pred_speed[valid]
    obs_speed = obs_speed[valid]
    pred_direction = np.radians(pred_direction[valid])
    obs_direction = np.radians(obs_direction[valid])
    result = Partial(n=int(valid.sum()))
    if result.n == 0:
        return result
    speed_diff = pred_speed - obs_speed
    result.speed_sse = float(np.sum(speed_diff * speed_diff))
    result.speed_abs = float(np.sum(np.abs(speed_diff)))
    result.speed_sum_pred = float(np.sum(pred_speed))
    result.speed_sum_obs = float(np.sum(obs_speed))
    result.speed_sum_pred2 = float(np.sum(pred_speed * pred_speed))
    result.speed_sum_obs2 = float(np.sum(obs_speed * obs_speed))
    result.speed_sum_cross = float(np.sum(pred_speed * obs_speed))
    err = angular_error(np.degrees(pred_direction), np.degrees(obs_direction))
    result.dir_sse = float(np.sum(err * err))
    result.dir_abs = float(np.sum(np.abs(err)))
    result.dir_sum_err = float(np.sum(err))
    sp, cp = np.sin(pred_direction), np.cos(pred_direction)
    so, co = np.sin(obs_direction), np.cos(obs_direction)
    result.dir_sin_pred, result.dir_cos_pred = float(sp.sum()), float(cp.sum())
    result.dir_sin_obs, result.dir_cos_obs = float(so.sum()), float(co.sum())
    result.dir_sin_pred2, result.dir_cos_pred2 = float(np.sum(sp * sp)), float(np.sum(cp * cp))
    result.dir_sin_obs2, result.dir_cos_obs2 = float(np.sum(so * so)), float(np.sum(co * co))
    result.dir_sin_cos_pred, result.dir_sin_cos_obs = float(np.sum(sp * cp)), float(np.sum(so * co))
    result.dir_sinpred_sinobs, result.dir_sinpred_cosobs = float(np.sum(sp * so)), float(np.sum(sp * co))
    result.dir_cospred_sinobs, result.dir_cospred_cosobs = float(np.sum(cp * so)), float(np.sum(cp * co))
    return result


def pearson(part: Partial) -> float:
    if part.n < 2:
        return float("nan")
    numerator = part.n * part.speed_sum_cross - part.speed_sum_pred * part.speed_sum_obs
    denominator = math.sqrt(
        (part.n * part.speed_sum_pred2 - part.speed_sum_pred**2)
        * (part.n * part.speed_sum_obs2 - part.speed_sum_obs**2)
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def circular_corr(part: Partial) -> float:
    if part.n < 2:
        return float("nan")
    mean_p = math.atan2(part.dir_sin_pred, part.dir_cos_pred)
    mean_o = math.atan2(part.dir_sin_obs, part.dir_cos_obs)
    ap, bp = math.cos(mean_p), math.sin(mean_p)
    ao, bo = math.cos(mean_o), math.sin(mean_o)
    # sin(theta - mean) = sin(theta)*cos(mean) - cos(theta)*sin(mean)
    pp = ap * ap * part.dir_sin_pred2 + bp * bp * part.dir_cos_pred2 - 2 * ap * bp * part.dir_sin_cos_pred
    oo = ao * ao * part.dir_sin_obs2 + bo * bo * part.dir_cos_obs2 - 2 * ao * bo * part.dir_sin_cos_obs
    po = (
        ap * ao * part.dir_sinpred_sinobs
        - ap * bo * part.dir_sinpred_cosobs
        - bp * ao * part.dir_cospred_sinobs
        + bp * bo * part.dir_cospred_cosobs
    )
    denominator = math.sqrt(max(pp, 0.0) * max(oo, 0.0))
    return float(po / denominator) if denominator > 0 else float("nan")


def metrics(part: Partial) -> dict:
    n = part.n
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "cc": float("nan")}
    return {
        "n": n,
        "rmse": math.sqrt(part.speed_sse / n),
        "mae": part.speed_abs / n,
        "cc": pearson(part),
    }


def direction_metrics(part: Partial) -> dict:
    n = part.n
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "cc": float("nan")}
    return {
        "n": n,
        "rmse": math.sqrt(part.dir_sse / n),
        "mae": part.dir_abs / n,
        "cc": circular_corr(part),
    }


def model_path(output_root: Path, experiment: str, target: datetime, lead: int) -> Path:
    return wind_path(output_root, EXPERIMENTS[experiment], target, lead)


def run_statistics(args) -> tuple[dict, list[dict], dict]:
    obs_by_time, obs_info = load_daily_updated(args.icoads_root, AREA)
    grid_path = None
    for experiment in args.experiments:
        for target in iter_targets(args.start_date, args.end_date):
            for lead in LEADS:
                candidate = model_path(args.output_root, experiment, target, lead)
                if candidate.is_file():
                    grid_path = candidate
                    break
            if grid_path:
                break
        if grid_path:
            break
    if grid_path is None:
        raise FileNotFoundError("No wind output files were found")
    sampler = GridSampler(grid_path, AREA)
    keys = [(exp, lead, month) for exp in args.experiments for lead in LEADS for month in range(1, 13)]
    totals = {key: Partial() for key in keys}
    missing = []
    task_specs = []
    for target in iter_targets(args.start_date, args.end_date):
        for lead in LEADS:
            valid_time = target + timedelta(hours=lead)
            observations = obs_by_time.get(valid_time)
            if observations is None:
                continue
            for experiment in args.experiments:
                task_specs.append((experiment, target, lead, valid_time, observations))
    total_tasks = len(task_specs)
    completed = 0
    lock = threading.Lock()

    def task(spec):
        experiment, target, lead, valid_time, observations = spec
        path = model_path(args.output_root, experiment, target, lead)
        if not path.is_file():
            return experiment, lead, valid_time.month, None, {"experiment": experiment, "target_time": time_label(target), "lead_hour": lead, "valid_time": time_label(valid_time), "path": str(path), "reason": "missing_output"}
        try:
            pu, pv = sampler.sample(path, observations["lat"], observations["lon"])
            part = make_partial(pu, pv, observations["speed"], observations["direction"])
            return experiment, lead, valid_time.month, part, None
        except Exception as exc:  # noqa: BLE001
            return experiment, lead, valid_time.month, None, {"experiment": experiment, "target_time": time_label(target), "lead_hour": lead, "valid_time": time_label(valid_time), "path": str(path), "reason": f"sample_failed: {type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, spec) for spec in task_specs]
        for future in as_completed(futures):
            experiment, lead, month, part, error = future.result()
            if part is not None:
                totals[(experiment, lead, month)].add(part)
            if error:
                missing.append(error)
            with lock:
                completed += 1
                if completed % args.progress_every == 0 or completed == total_tasks:
                    print(f"[统计进度] {completed}/{total_tasks} 任务", flush=True)
    return totals, missing, {"observations": obs_info, "grid_sample": str(grid_path), "tasks": total_tasks}


def rows_for_totals(totals: dict, group: str, variable: str) -> list[dict]:
    rows = []
    for (experiment, lead, month), part in sorted(totals.items()):
        if group == "lead":
            # caller combines months separately; this branch is not used for aggregation.
            continue
        if variable == "wind_speed":
            values = metrics(part)
        else:
            values = direction_metrics(part)
        rows.append({"experiment": experiment, "lead_hour": lead, "month": month, "variable": variable, **values})
    return rows


def aggregate(totals: dict, by: str, variable: str) -> list[dict]:
    groups: dict[tuple, Partial] = {}
    for (experiment, lead, month), part in totals.items():
        key = (experiment, lead) if by == "lead" else (experiment, month)
        groups.setdefault(key, Partial()).add(part)
    output = []
    for key, part in sorted(groups.items()):
        row = {"experiment": key[0]}
        if by == "lead":
            row["lead_hour"] = key[1]
        else:
            row["month"] = key[1]
        row["variable"] = variable
        row.update(metrics(part) if variable == "wind_speed" else direction_metrics(part))
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_results(args, totals, missing, info):
    args.data_root.mkdir(parents=True, exist_ok=True)
    speed_lm = [row for row in rows_for_totals(totals, "lead_month", "wind_speed")]
    direction_lm = [row for row in rows_for_totals(totals, "lead_month", "wind_direction")]
    write_csv(args.data_root / "wind_speed_metrics_by_lead_month.csv", speed_lm)
    write_csv(args.data_root / "wind_direction_metrics_by_lead_month.csv", direction_lm)
    write_csv(args.data_root / "wind_speed_metrics_by_lead.csv", aggregate(totals, "lead", "wind_speed"))
    write_csv(args.data_root / "wind_direction_metrics_by_lead.csv", aggregate(totals, "lead", "wind_direction"))
    write_csv(args.data_root / "wind_speed_metrics_by_month.csv", aggregate(totals, "month", "wind_speed"))
    write_csv(args.data_root / "wind_direction_metrics_by_month.csv", aggregate(totals, "month", "wind_direction"))
    write_csv(args.data_root / "missing_outputs.csv", missing)
    summary = {
        "area": AREA,
        "experiments": list(args.experiments),
        "lead_hours": list(LEADS),
        "start_date": args.start_date.strftime("%Y-%m-%d"),
        "end_date": args.end_date.strftime("%Y-%m-%d"),
        "observation_policy": "ICOADS manifest kind=updated and format=nc only; preliminary and DAT.GZ excluded; Natural Earth 10m land/lake mask applied",
        "area_policy": "latitude 10..30 N, longitude 105..130 E",
        "info": info,
        "missing_outputs": len(missing),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (args.data_root / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, default=datetime(2025, 1, 1))
    parser.add_argument("--end-date", type=parse_date, default=datetime(2025, 12, 31))
    parser.add_argument("--icoads-root", type=Path, default=DEFAULT_ICOADS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENT_NAMES, default=list(EXPERIMENT_NAMES))
    args = parser.parse_args(argv)
    if args.end_date < args.start_date or args.workers < 1:
        parser.error("invalid date range or worker count")
    started = time.perf_counter()
    totals, missing, info = run_statistics(args)
    info["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    save_results(args, totals, missing, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

