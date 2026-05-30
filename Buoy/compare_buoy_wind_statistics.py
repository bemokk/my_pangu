from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from paths import DEFAULT_CHINA_SEA_DETAIL_CSV, WIND_MODEL_STATISTICS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUOY_CSV = DEFAULT_CHINA_SEA_DETAIL_CSV
ERA5_REALTIME_WIND10_NC = (
    PROJECT_ROOT
    / "model_input"
    / "multi_time_point"
    / "wind10_july_august.nc"
)
OUT_DIR = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h"

AREA = [42.0, 103.0, 13.0, 130.0]
LEAD_HOURS = list(range(3, 73, 3))
ERA5_DELAY_HOURS = 120

SURFACE_WIND_VARS = {
    "u10": ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "u10m"],
    "v10": ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "v10m"],
}

DIRECTION_SECTORS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    label: str


DATASETS = [
    DatasetConfig("era5_realtime", "ERA5 realtime"),
    DatasetConfig("era5_lagged_5d", "ERA5 lagged 5d forecast"),
    DatasetConfig("gdas_forecast", "GDAS forecast"),
]


def parse_datetime(value: str | datetime | pd.Timestamp) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    text = str(value).strip()
    for fmt in (
        "%Y-%m-%d-%H-%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y%m%d%H",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported datetime format: {value}")


def time_to_str(value: datetime | pd.Timestamp) -> str:
    return parse_datetime(value).strftime("%Y-%m-%d-%H-%M")


def parse_lead_hours(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one lead hour is required.")
    return sorted(set(values))


def find_var_name(ds: xr.Dataset, candidates: list[str]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Could not find any of {candidates}; available vars: {list(ds.data_vars)}")


def open_dataset(path: Path) -> xr.Dataset:
    engines = xr.backends.plugins.list_engines()
    if "netcdf4" in engines:
        return xr.open_dataset(path, engine="netcdf4", decode_times=True)
    return xr.open_dataset(path, decode_times=True)


def normalize_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for src in ("lat", "Latitude", "LAT"):
        if src in ds.coords or src in ds.dims:
            rename[src] = "latitude"
            break
    for src in ("lon", "Longitude", "LON"):
        if src in ds.coords or src in ds.dims:
            rename[src] = "longitude"
            break
    if rename:
        ds = ds.rename(rename)

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise KeyError(f"Missing latitude/longitude coordinates in {list(ds.coords)}")

    lon = ds["longitude"]
    if float(lon.min()) < 0:
        ds = ds.assign_coords(longitude=(lon % 360))

    ds = ds.sortby("latitude")
    ds = ds.sortby("longitude")
    return ds


def find_time_coord(ds: xr.Dataset) -> str | None:
    for name in ("valid_time", "time"):
        if name in ds.coords or name in ds.dims:
            return name
    return None


def drop_extra_dims(da: xr.DataArray) -> xr.DataArray:
    for dim in list(da.dims):
        if dim not in {"latitude", "longitude"}:
            da = da.isel({dim: 0}, drop=True)
    return da


class SurfaceWindSampler:
    def __init__(self, nc_path: Path):
        self.nc_path = Path(nc_path)
        self._raw: xr.Dataset | None = None
        self._ds: xr.Dataset | None = None

    def __enter__(self) -> "SurfaceWindSampler":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def open(self) -> "SurfaceWindSampler":
        if self._ds is not None:
            return self

        self._raw = open_dataset(self.nc_path)
        ds = normalize_lon_lat(self._raw)
        u_name = find_var_name(ds, SURFACE_WIND_VARS["u10"])
        v_name = find_var_name(ds, SURFACE_WIND_VARS["v10"])
        self._ds = normalize_lon_lat(
            xr.Dataset(
                {
                    "u10": ds[u_name],
                    "v10": ds[v_name],
                }
            )
        ).astype(np.float32)
        return self

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None
        if self._raw is not None:
            self._raw.close()
            self._raw = None

    def surface_wind_at(self, valid_time: datetime | None = None) -> xr.Dataset:
        self.open()
        if self._ds is None:
            raise RuntimeError("Surface wind dataset is not open.")

        ds = self._ds
        time_coord = find_time_coord(ds)
        if valid_time is not None and time_coord is not None:
            ds = ds.sel({time_coord: np.datetime64(valid_time)}, drop=True)

        return xr.Dataset(
            {
                "u10": drop_extra_dims(ds["u10"]),
                "v10": drop_extra_dims(ds["v10"]),
            }
        )

    def sample(self, records: pd.DataFrame, valid_time: datetime | None = None) -> pd.DataFrame:
        ds = self.surface_wind_at(valid_time=valid_time)

        lat = xr.DataArray(records["latitude"].to_numpy(dtype=float), dims="points")
        lon = xr.DataArray(records["longitude"].to_numpy(dtype=float), dims="points")

        sampled = ds.interp(latitude=lat, longitude=lon, method="linear")
        out = pd.DataFrame(
            {
                "pred_u10_ms": np.asarray(sampled["u10"].values, dtype=float),
                "pred_v10_ms": np.asarray(sampled["v10"].values, dtype=float),
            },
            index=records.index,
        )
        out["pred_speed_ms"] = wind_speed(out["pred_u10_ms"], out["pred_v10_ms"])
        out["pred_dir_deg"] = wind_direction_from_uv(out["pred_u10_ms"], out["pred_v10_ms"])
        return out


def load_surface_wind(nc_path: Path, valid_time: datetime | None = None) -> xr.Dataset:
    with SurfaceWindSampler(nc_path) as sampler:
        return sampler.surface_wind_at(valid_time=valid_time).load()


def sample_surface_wind(
    nc_path: Path,
    records: pd.DataFrame,
    valid_time: datetime | None = None,
) -> pd.DataFrame:
    with SurfaceWindSampler(nc_path) as sampler:
        return sampler.sample(records, valid_time=valid_time)


def wind_speed(u: pd.Series | np.ndarray, v: pd.Series | np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(u, dtype=float) ** 2 + np.asarray(v, dtype=float) ** 2)


def wind_direction_from_uv(u: pd.Series | np.ndarray, v: pd.Series | np.ndarray) -> np.ndarray:
    direction = (270.0 - np.degrees(np.arctan2(np.asarray(v, dtype=float), np.asarray(u, dtype=float)))) % 360.0
    return direction


def angular_error_deg(pred_dir: pd.Series | np.ndarray, obs_dir: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(pred_dir, dtype=float) - np.asarray(obs_dir, dtype=float) + 180.0) % 360.0 - 180.0


def circular_mean_deg(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    radians = np.deg2rad(x % 360.0)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    if mean_sin == 0 and mean_cos == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)


def circular_corrcoef_deg(pred_dir: pd.Series | np.ndarray, obs_dir: pd.Series | np.ndarray) -> float:
    pred = np.asarray(pred_dir, dtype=float)
    obs = np.asarray(obs_dir, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]
    if len(pred) < 2:
        return float("nan")

    pred_rad = np.deg2rad(pred % 360.0)
    obs_rad = np.deg2rad(obs % 360.0)
    pred_mean = np.deg2rad(circular_mean_deg(pred))
    obs_mean = np.deg2rad(circular_mean_deg(obs))

    pred_centered = np.sin(pred_rad - pred_mean)
    obs_centered = np.sin(obs_rad - obs_mean)
    denom = math.sqrt(float(np.sum(pred_centered**2) * np.sum(obs_centered**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(pred_centered * obs_centered) / denom)


def scalar_metrics(pred: pd.Series | np.ndarray, obs: pd.Series | np.ndarray) -> dict:
    pred_arr = np.asarray(pred, dtype=float)
    obs_arr = np.asarray(obs, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(obs_arr)
    pred_arr = pred_arr[mask]
    obs_arr = obs_arr[mask]
    n = len(pred_arr)
    if n == 0:
        return empty_metrics()

    diff = pred_arr - obs_arr
    if n < 2 or np.nanstd(pred_arr) == 0 or np.nanstd(obs_arr) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(pred_arr, obs_arr)[0, 1])

    return {
        "n": int(n),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "corr": corr,
        "pred_mean": float(np.mean(pred_arr)),
        "obs_mean": float(np.mean(obs_arr)),
        "diff_std": float(np.std(diff)),
    }


def direction_metrics(pred_dir: pd.Series | np.ndarray, obs_dir: pd.Series | np.ndarray) -> dict:
    pred = np.asarray(pred_dir, dtype=float)
    obs = np.asarray(obs_dir, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[mask]
    obs = obs[mask]
    n = len(pred)
    if n == 0:
        return empty_metrics()

    err = angular_error_deg(pred, obs)
    return {
        "n": int(n),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "corr": circular_corrcoef_deg(pred, obs),
        "pred_mean": circular_mean_deg(pred),
        "obs_mean": circular_mean_deg(obs),
        "diff_std": float(np.std(err)),
    }


def empty_metrics() -> dict:
    return {
        "n": 0,
        "rmse": float("nan"),
        "mae": float("nan"),
        "bias": float("nan"),
        "corr": float("nan"),
        "pred_mean": float("nan"),
        "obs_mean": float("nan"),
        "diff_std": float("nan"),
    }


def beaufort_group(speed_ms: pd.Series | np.ndarray) -> pd.Categorical:
    bins = [-np.inf, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, np.inf]
    labels = ["<=2", "3", "4", "5", "6", "7", ">=8"]
    return pd.cut(speed_ms, bins=bins, labels=labels, right=False, ordered=True)


def direction_sector(direction_deg: pd.Series | np.ndarray) -> pd.Categorical:
    values = np.asarray(direction_deg, dtype=float)
    idx = np.floor(((values % 360.0) + 11.25) / 22.5).astype(int) % 16
    labels = np.asarray(DIRECTION_SECTORS, dtype=object)[idx]
    labels[~np.isfinite(values)] = None
    return pd.Categorical(labels, categories=DIRECTION_SECTORS, ordered=True)


def load_buoy_records(csv_path: Path, area: list[float]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Buoy CSV not found: {csv_path}")

    records = pd.read_csv(csv_path)
    records["datetime_utc"] = pd.to_datetime(records["datetime_utc"], errors="coerce")

    lat_max, lon_min, lat_min, lon_max = area
    records = records.dropna(
        subset=["datetime_utc", "latitude", "longitude", "wind_dir_deg", "wind_speed_ms"]
    ).copy()
    records = records[
        records["latitude"].between(lat_min, lat_max)
        & records["longitude"].between(lon_min, lon_max)
        & records["wind_dir_deg"].between(1, 360)
        & records["wind_speed_ms"].between(0, 75)
    ].copy()

    records = records.rename(
        columns={
            "wind_speed_ms": "obs_speed_ms",
            "wind_dir_deg": "obs_dir_deg",
        }
    )
    records = records.sort_values("datetime_utc").reset_index(drop=True)
    records.insert(0, "record_id", np.arange(len(records), dtype=int))
    records["obs_beaufort_group"] = beaufort_group(records["obs_speed_ms"]).astype(str)
    records["obs_dir_sector"] = direction_sector(records["obs_dir_deg"]).astype(str)
    return records


def default_target_range(records: pd.DataFrame, lead_hours: list[int]) -> tuple[datetime, datetime]:
    min_time = records["datetime_utc"].min().to_pydatetime()
    max_time = records["datetime_utc"].max().to_pydatetime() - timedelta(hours=max(lead_hours))

    start = min_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end = max_time.replace(hour=0, minute=0, second=0, microsecond=0)
    if end < start:
        raise ValueError("Buoy time range is too short for the requested lead hours.")
    return start, end


def prediction_path(
    dataset: str,
    target_time: datetime,
    lead_hour: int,
    valid_time: datetime,
) -> tuple[Path, datetime, int]:
    valid_str = time_to_str(valid_time)

    if dataset == "era5_realtime":
        if ERA5_REALTIME_WIND10_NC.exists():
            path = ERA5_REALTIME_WIND10_NC
        else:
            path = PROJECT_ROOT / "model_input" / "single_time_point" / "era5" / valid_str / "surface.nc"
        return path, valid_time, 0

    if dataset == "era5_lagged_5d":
        pred_start = target_time - timedelta(hours=ERA5_DELAY_HOURS)
        model_hour = ERA5_DELAY_HOURS + lead_hour
        path = (
            PROJECT_ROOT
            / "model_output"
            / "era5"
            / time_to_str(pred_start)
            / str(model_hour)
            / f"output_surface_{valid_str}.nc"
        )
        return path, pred_start, model_hour

    if dataset == "gdas_forecast":
        pred_start = target_time
        model_hour = lead_hour
        path = (
            PROJECT_ROOT
            / "model_output"
            / "gdas"
            / time_to_str(pred_start)
            / str(model_hour)
            / f"output_surface_{valid_str}.nc"
        )
        return path, pred_start, model_hour

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_matches(
    records: pd.DataFrame,
    lead_hours: list[int],
    target_start: datetime,
    target_end: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched_rows = []
    missing_rows = []
    missing_observation_rows = []
    records_by_time = {key.to_pydatetime(): value for key, value in records.groupby("datetime_utc")}
    realtime_sampler = SurfaceWindSampler(ERA5_REALTIME_WIND10_NC) if ERA5_REALTIME_WIND10_NC.exists() else None

    target_times = pd.date_range(target_start, target_end, freq="24h").to_pydatetime().tolist()
    total_cases = len(target_times) * len(lead_hours) * len(DATASETS)
    case_index = 0

    for target_time in target_times:
        for lead_hour in lead_hours:
            valid_time = target_time + timedelta(hours=lead_hour)
            obs = records_by_time.get(valid_time)
            if obs is None or obs.empty:
                missing_observation_rows.append(
                    {
                        "target_time": time_to_str(target_time),
                        "lead_hour": lead_hour,
                        "valid_time": time_to_str(valid_time),
                        "reason": "no_buoy_observation",
                    }
                )
                continue

            for config in DATASETS:
                case_index += 1
                path, pred_start, model_hour = prediction_path(config.dataset, target_time, lead_hour, valid_time)
                if not path.exists():
                    missing_rows.append(
                        {
                            "dataset": config.dataset,
                            "dataset_label": config.label,
                            "target_time": time_to_str(target_time),
                            "pred_start_time": time_to_str(pred_start),
                            "lead_hour": lead_hour,
                            "model_forecast_hour": model_hour,
                            "valid_time": time_to_str(valid_time),
                            "path": str(path.relative_to(PROJECT_ROOT)),
                            "reason": "missing_file",
                        }
                    )
                    continue

                try:
                    if config.dataset == "era5_realtime" and realtime_sampler is not None and path == ERA5_REALTIME_WIND10_NC:
                        sampled = realtime_sampler.sample(obs, valid_time=valid_time)
                    else:
                        sample_valid_time = valid_time if config.dataset == "era5_realtime" else None
                        sampled = sample_surface_wind(path, obs, valid_time=sample_valid_time)
                except Exception as exc:  # noqa: BLE001 - keep batch jobs moving.
                    missing_rows.append(
                        {
                            "dataset": config.dataset,
                            "dataset_label": config.label,
                            "target_time": time_to_str(target_time),
                            "pred_start_time": time_to_str(pred_start),
                            "lead_hour": lead_hour,
                            "model_forecast_hour": model_hour,
                            "valid_time": time_to_str(valid_time),
                            "path": str(path.relative_to(PROJECT_ROOT)),
                            "reason": f"read_or_sample_failed: {exc}",
                        }
                    )
                    continue

                out = obs.reset_index(drop=True).copy()
                sampled = sampled.reset_index(drop=True)
                out = pd.concat([out, sampled], axis=1)
                out.insert(1, "dataset", config.dataset)
                out.insert(2, "dataset_label", config.label)
                out.insert(3, "target_time", time_to_str(target_time))
                out.insert(4, "pred_start_time", time_to_str(pred_start))
                out.insert(5, "lead_hour", lead_hour)
                out.insert(6, "model_forecast_hour", model_hour)
                out.insert(7, "valid_time", time_to_str(valid_time))
                out["source_model_path"] = str(path.relative_to(PROJECT_ROOT))
                out["speed_error_ms"] = out["pred_speed_ms"] - out["obs_speed_ms"]
                out["direction_error_deg"] = angular_error_deg(out["pred_dir_deg"], out["obs_dir_deg"])
                out["direction_abs_error_deg"] = np.abs(out["direction_error_deg"])
                out["pred_beaufort_group"] = beaufort_group(out["pred_speed_ms"]).astype(str)
                out["pred_dir_sector"] = direction_sector(out["pred_dir_deg"]).astype(str)
                matched_rows.append(out)

                if case_index % 50 == 0:
                    print(f"Processed {case_index}/{total_cases} dataset cases...")

    if matched_rows:
        matched = pd.concat(matched_rows, ignore_index=True)
    else:
        matched = pd.DataFrame()

    missing = pd.DataFrame(missing_rows)
    missing_observations = pd.DataFrame(missing_observation_rows)
    if realtime_sampler is not None:
        realtime_sampler.close()
    return matched, missing, missing_observations


def metrics_by_group(df: pd.DataFrame, group_cols: list[str], variable: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        if variable == "wind_speed":
            metrics = scalar_metrics(group["pred_speed_ms"], group["obs_speed_ms"])
        elif variable == "wind_direction":
            metrics = direction_metrics(group["pred_dir_deg"], group["obs_dir_deg"])
        else:
            raise ValueError(f"Unsupported variable: {variable}")
        row.update({"variable": variable, **metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def direction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "dataset_label", "lead_hour"]
    for keys, group in df.groupby(group_cols, dropna=False, observed=False):
        base = dict(zip(group_cols, keys))
        obs_counts = group["obs_dir_sector"].value_counts().reindex(DIRECTION_SECTORS, fill_value=0)
        pred_counts = group["pred_dir_sector"].value_counts().reindex(DIRECTION_SECTORS, fill_value=0)
        obs_total = int(obs_counts.sum())
        pred_total = int(pred_counts.sum())
        for sector in DIRECTION_SECTORS:
            rows.append(
                {
                    **base,
                    "direction_sector": sector,
                    "obs_count": int(obs_counts[sector]),
                    "obs_frequency": float(obs_counts[sector] / obs_total) if obs_total else np.nan,
                    "pred_count": int(pred_counts[sector]),
                    "pred_frequency": float(pred_counts[sector] / pred_total) if pred_total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_and_save(
    matched: pd.DataFrame,
    missing: pd.DataFrame,
    missing_observations: pd.DataFrame,
    out_dir: Path,
    lead_hours: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    matched_csv = out_dir / "matched_buoy_model_wind_samples.csv"
    speed_by_lead_csv = out_dir / "wind_speed_metrics_by_lead.csv"
    direction_by_lead_csv = out_dir / "wind_direction_metrics_by_lead.csv"
    speed_by_beaufort_csv = out_dir / "wind_speed_metrics_by_beaufort.csv"
    direction_frequency_csv = out_dir / "wind_direction_frequency_by_lead.csv"
    missing_csv = out_dir / "missing_files.csv"
    missing_observations_csv = out_dir / "missing_observations.csv"
    summary_txt = out_dir / "run_summary.txt"

    matched.to_csv(matched_csv, index=False, encoding="utf-8-sig")
    missing.to_csv(missing_csv, index=False, encoding="utf-8-sig")
    missing_observations.to_csv(missing_observations_csv, index=False, encoding="utf-8-sig")

    speed_by_lead = metrics_by_group(matched, ["dataset", "dataset_label", "lead_hour"], "wind_speed")
    direction_by_lead = metrics_by_group(matched, ["dataset", "dataset_label", "lead_hour"], "wind_direction")
    speed_by_beaufort = metrics_by_group(
        matched,
        ["dataset", "dataset_label", "lead_hour", "obs_beaufort_group"],
        "wind_speed",
    )
    dir_freq = direction_frequency(matched)

    speed_by_lead.to_csv(speed_by_lead_csv, index=False, encoding="utf-8-sig")
    direction_by_lead.to_csv(direction_by_lead_csv, index=False, encoding="utf-8-sig")
    speed_by_beaufort.to_csv(speed_by_beaufort_csv, index=False, encoding="utf-8-sig")
    dir_freq.to_csv(direction_frequency_csv, index=False, encoding="utf-8-sig")

    lines = [
        "Buoy wind comparison summary",
        f"lead_hours={','.join(str(x) for x in lead_hours)}",
        f"matched_rows={len(matched)}",
        f"missing_cases={len(missing)}",
        f"missing_observation_times={len(missing_observations)}",
        "",
        "matched_rows_by_dataset:",
        matched.groupby("dataset").size().to_string() if not matched.empty else "(none)",
        "",
        "matched_rows_by_dataset_and_lead:",
        matched.groupby(["dataset", "lead_hour"]).size().to_string() if not matched.empty else "(none)",
        "",
        "outputs:",
        str(matched_csv.relative_to(PROJECT_ROOT)),
        str(speed_by_lead_csv.relative_to(PROJECT_ROOT)),
        str(direction_by_lead_csv.relative_to(PROJECT_ROOT)),
        str(speed_by_beaufort_csv.relative_to(PROJECT_ROOT)),
        str(direction_frequency_csv.relative_to(PROJECT_ROOT)),
        str(missing_csv.relative_to(PROJECT_ROOT)),
        str(missing_observations_csv.relative_to(PROJECT_ROOT)),
    ]
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare buoy wind observations with ERA5 realtime, ERA5 lagged, and GDAS forecast data.",
    )
    parser.add_argument("--buoy-csv", type=Path, default=BUOY_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--lead-hours", default=",".join(str(x) for x in LEAD_HOURS))
    parser.add_argument("--target-start", default=None, help="Default: first buoy day at 00 UTC.")
    parser.add_argument(
        "--target-end",
        default=None,
        help="Default: last complete target day for the maximum requested lead.",
    )
    return parser


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = build_parser().parse_args()
    lead_hours = parse_lead_hours(args.lead_hours)
    buoy_csv = resolve_project_path(args.buoy_csv)
    out_dir = resolve_project_path(args.out_dir)

    records = load_buoy_records(buoy_csv, AREA)
    default_start, default_end = default_target_range(records, lead_hours)
    target_start = parse_datetime(args.target_start) if args.target_start else default_start
    target_end = parse_datetime(args.target_end) if args.target_end else default_end

    print(f"Buoy records: {len(records)}")
    print(f"Buoy time range: {records['datetime_utc'].min()} to {records['datetime_utc'].max()}")
    print(f"Target range: {time_to_str(target_start)} to {time_to_str(target_end)}")
    print(f"Lead hours: {lead_hours}")

    matched, missing, missing_observations = build_matches(records, lead_hours, target_start, target_end)
    if matched.empty:
        raise RuntimeError("No matched samples were produced. Check input paths and lead hours.")

    summarize_and_save(matched, missing, missing_observations, out_dir, lead_hours)


if __name__ == "__main__":
    main()
