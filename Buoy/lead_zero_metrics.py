from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import compare_buoy_wind_statistics as compare
from paths import DEFAULT_CHINA_SEA_DETAIL_CSV, WIND_MODEL_STATISTICS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h"
MATCHED_SAMPLES_CSV = STATS_DIR / "matched_buoy_model_wind_samples.csv"
ERA5_DELAY_HOURS = 120


@dataclass(frozen=True)
class ZeroHourSource:
    dataset: str
    dataset_label: str
    path: Path
    valid_time: datetime


def load_target_times(csv_path: Path = MATCHED_SAMPLES_CSV) -> list[datetime]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Matched samples CSV not found: {csv_path}")

    targets = pd.read_csv(csv_path, usecols=["target_time"])["target_time"].dropna().unique()
    return sorted(compare.parse_datetime(value) for value in targets)


def zero_hour_source(dataset: str, target_time: datetime) -> ZeroHourSource:
    label_by_dataset = {config.dataset: config.label for config in compare.DATASETS}
    valid_str = compare.time_to_str(target_time)

    if dataset == "era5_realtime":
        if compare.ERA5_REALTIME_WIND10_NC.exists():
            path = compare.ERA5_REALTIME_WIND10_NC
        else:
            path = PROJECT_ROOT / "model_input" / "single_time_point" / "era5" / valid_str / "surface.nc"
        return ZeroHourSource(dataset, label_by_dataset[dataset], path, target_time)

    if dataset == "era5_lagged_5d":
        pred_start = target_time - timedelta(hours=ERA5_DELAY_HOURS)
        path = (
            PROJECT_ROOT
            / "model_output"
            / "era5"
            / compare.time_to_str(pred_start)
            / str(ERA5_DELAY_HOURS)
            / f"output_surface_{valid_str}.nc"
        )
        return ZeroHourSource(dataset, label_by_dataset[dataset], path, target_time)

    if dataset == "gdas_forecast":
        path = PROJECT_ROOT / "model_input" / "single_time_point" / "gdas" / valid_str / "surface.nc"
        return ZeroHourSource(dataset, label_by_dataset[dataset], path, target_time)

    raise ValueError(f"Unsupported dataset for 0h metrics: {dataset}")


def load_zero_hour_observations(
    target_times: list[datetime],
    buoy_csv: Path = DEFAULT_CHINA_SEA_DETAIL_CSV,
) -> pd.DataFrame:
    records = compare.load_buoy_records(buoy_csv, compare.AREA)
    target_index = pd.to_datetime(target_times)
    return records[records["datetime_utc"].isin(target_index)].copy()


def sample_zero_hour_matches(
    target_times: list[datetime] | None = None,
    records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if target_times is None:
        target_times = load_target_times()
    if records is None:
        records = load_zero_hour_observations(target_times)

    records_by_time = {
        key.to_pydatetime(): value.copy()
        for key, value in records.groupby("datetime_utc")
    }

    rows = []
    samplers: dict[Path, compare.SurfaceWindSampler] = {}
    try:
        for target_time in target_times:
            obs = records_by_time.get(target_time)
            if obs is None or obs.empty:
                continue

            for config in compare.DATASETS:
                source = zero_hour_source(config.dataset, target_time)
                if not source.path.exists():
                    print(f"Warning: missing 0h source for {config.dataset}: {source.path}")
                    continue

                sampler = samplers.get(source.path)
                if sampler is None:
                    sampler = compare.SurfaceWindSampler(source.path).open()
                    samplers[source.path] = sampler

                sampled = sampler.sample(obs, valid_time=source.valid_time).reset_index(drop=True)
                out = obs.reset_index(drop=True).copy()
                out = pd.concat([out, sampled], axis=1)
                out.insert(1, "dataset", source.dataset)
                out.insert(2, "dataset_label", source.dataset_label)
                out.insert(3, "lead_hour", 0)
                rows.append(out)
    finally:
        for sampler in samplers.values():
            sampler.close()

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_lead_zero_metric_rows(variable: str) -> pd.DataFrame:
    matches = sample_zero_hour_matches()
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for (dataset, dataset_label), group in matches.groupby(["dataset", "dataset_label"], sort=False):
        if variable == "wind_speed":
            metrics = compare.scalar_metrics(group["pred_speed_ms"], group["obs_speed_ms"])
        elif variable == "wind_direction":
            metrics = compare.direction_metrics(group["pred_dir_deg"], group["obs_dir_deg"])
        else:
            raise ValueError(f"Unsupported variable: {variable}")

        rows.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label,
                "lead_hour": 0,
                "variable": variable,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def append_lead_zero_rows(
    df: pd.DataFrame,
    zero_rows: pd.DataFrame,
    lead_col: str = "lead_hour",
) -> pd.DataFrame:
    if zero_rows.empty:
        return df.copy()

    out = df.copy()
    out = out[~((out[lead_col] == 0) & out["dataset"].isin(zero_rows["dataset"]))]
    return pd.concat([zero_rows, out], ignore_index=True, sort=False)
