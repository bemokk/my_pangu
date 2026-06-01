from __future__ import annotations

import json
import math
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

BUOY_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUOY_DIR.parent
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

from paths import FIGURES_DIR, RESULTS_DIR, WIND_MODEL_STATISTICS_DIR

CASE_DIR = RESULTS_DIR / "wipha_case"
MATCHED_CSV = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h" / "matched_buoy_model_wind_samples.csv"
WINDOW_START = pd.Timestamp("2025-07-17 00:00:00")
WINDOW_END = pd.Timestamp("2025-07-22 23:00:00")
TRACK_INIT = datetime(2025, 7, 17, 0)
LEAD_HOURS = list(range(0, 73, 3))
SELECTED_PLATFORMS = ["EVH28KM", "3FOS8"]
DATASETS = ["gdas_forecast", "era5_lagged_5d"]
DATASET_LABELS = {"gdas_forecast": "GDAS forecast", "era5_lagged_5d": "ERA5 lagged 5d forecast"}
DATASET_COLORS = {"observation": "#222222", "gdas_forecast": "#55A868", "era5_lagged_5d": "#4C72B0"}
PLATFORM_COLORS = {"EVH28KM": "#C44E52", "3FOS8": "#8172B2"}
MAP_AREA = (105.0, 130.0, 10.0, 32.0)
WIPHA_SEARCH_BOX = (106.0, 126.0, 10.0, 28.0)
MOVING_SEARCH_HALF_WIDTH_DEG = 5.0
GRID_LON0, GRID_DX, GRID_NLON = 0.125, 0.25, 1440
GRID_LAT0, GRID_DY, GRID_NLAT = 90.0, -0.25, 721

OUT_TRACK_BUOYS_PNG = FIGURES_DIR / "wipha_track_buoy_locations.png"
OUT_TRACK_BUOYS_SVG = FIGURES_DIR / "wipha_track_buoy_locations.svg"
OUT_TIMESERIES_PNG = FIGURES_DIR / "wipha_buoy_wind_timeseries.png"
OUT_TIMESERIES_SVG = FIGURES_DIR / "wipha_buoy_wind_timeseries.svg"
OUT_STATS_TABLE_PNG = FIGURES_DIR / "wipha_buoy_wind_statistics_table.png"
OUT_STATS_TABLE_SVG = FIGURES_DIR / "wipha_buoy_wind_statistics_table.svg"
OUT_TRACK_ERROR_PNG = FIGURES_DIR / "wipha_track_forecast_error_2025071700.png"
OUT_TRACK_ERROR_SVG = FIGURES_DIR / "wipha_track_forecast_error_2025071700.svg"
OUT_STATS_CSV = CASE_DIR / "wipha_buoy_wind_statistics.csv"
OUT_STATS_XLSX = CASE_DIR / "wipha_buoy_wind_statistics.xlsx"
OUT_TRACKS_CSV = CASE_DIR / "wipha_typhoon_tracks_2025071700.csv"
OUT_TRACK_ERRORS_CSV = CASE_DIR / "wipha_typhoon_track_errors_2025071700.csv"
OUT_REAL_TRACK_CACHE = CASE_DIR / "wipha_real_track_nmc_or_fallback.csv"
OUT_ANALYSIS_SAMPLES = CASE_DIR / "wipha_buoy_time_series_samples.csv"
OUT_CANDIDATE_SUMMARY = CASE_DIR / "wipha_selected_platform_summary.csv"


def angular_difference_deg(pred_deg: float, obs_deg: float) -> float:
    if pd.isna(pred_deg) or pd.isna(obs_deg):
        return float("nan")
    return float(((float(obs_deg) - float(pred_deg) + 180.0) % 360.0) - 180.0)


def circular_mean_deg(values: Iterable[float]) -> float:
    vals = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if vals.size == 0:
        return float("nan")
    radians = np.deg2rad(vals)
    sin_mean = np.nanmean(np.sin(radians))
    cos_mean = np.nanmean(np.cos(radians))
    if np.isclose(sin_mean, 0.0) and np.isclose(cos_mean, 0.0):
        return float("nan")
    return float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if any(pd.isna(v) for v in (lon1, lat1, lon2, lat2)):
        return float("nan")
    r = 6371.0
    lon1r, lat1r, lon2r, lat2r = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2r - lon1r, lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })


def ensure_dirs() -> None:
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def select_shortest_lead_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    working = df.copy()
    working["lead_hour"] = pd.to_numeric(working["lead_hour"], errors="coerce")
    working = working.sort_values(["platform_id", "datetime_utc", "dataset", "lead_hour"])
    min_leads = working.groupby(["platform_id", "datetime_utc", "dataset"], as_index=False)["lead_hour"].min()
    return working.merge(min_leads, on=["platform_id", "datetime_utc", "dataset", "lead_hour"], how="inner").reset_index(drop=True)


def _circular_group_mean(series: pd.Series) -> float:
    return circular_mean_deg(series.to_numpy())


def load_matched_samples() -> pd.DataFrame:
    if not MATCHED_CSV.exists():
        raise FileNotFoundError(f"Matched buoy/model sample CSV not found: {MATCHED_CSV}")
    cols = {"dataset", "dataset_label", "datetime_utc", "platform_id", "platform_type", "id_indicator", "latitude", "longitude", "obs_speed_ms", "obs_dir_deg", "lead_hour", "pred_speed_ms", "pred_dir_deg", "pred_u10_ms", "pred_v10_ms"}
    df = pd.read_csv(MATCHED_CSV, usecols=lambda c: c in cols)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    return df


def build_platform_summary(raw: pd.DataFrame) -> pd.DataFrame:
    obs = raw.drop_duplicates(["platform_id", "datetime_utc", "latitude", "longitude", "obs_speed_ms", "obs_dir_deg"])
    summary = obs.groupby("platform_id").agg(
        n_times=("datetime_utc", "nunique"),
        lat_mean=("latitude", "mean"), lon_mean=("longitude", "mean"),
        lat_min=("latitude", "min"), lat_max=("latitude", "max"),
        lon_min=("longitude", "min"), lon_max=("longitude", "max"),
        max_speed_ms=("obs_speed_ms", "max"), mean_speed_ms=("obs_speed_ms", "mean"),
    ).reset_index()
    summary["coverage_ratio"] = summary["n_times"] / raw["datetime_utc"].nunique()
    return summary


def prepare_buoy_case_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_matched_samples()
    raw = raw[raw["datetime_utc"].between(WINDOW_START, WINDOW_END) & raw["platform_id"].isin(SELECTED_PLATFORMS) & raw["dataset"].isin(DATASETS)].copy()
    if raw.empty:
        raise RuntimeError("No matched buoy/model samples found for selected Wipha platforms and window.")
    platform_summary = build_platform_summary(raw)
    platform_summary.to_csv(OUT_CANDIDATE_SUMMARY, index=False, encoding="utf-8-sig")
    obs = raw.drop_duplicates(["platform_id", "datetime_utc", "latitude", "longitude", "obs_speed_ms", "obs_dir_deg"]).groupby(["platform_id", "datetime_utc"], as_index=False).agg(
        latitude=("latitude", "mean"), longitude=("longitude", "mean"), obs_speed_ms=("obs_speed_ms", "mean"), obs_dir_deg=("obs_dir_deg", _circular_group_mean)
    )
    forecasts = select_shortest_lead_forecasts(raw).groupby(["platform_id", "datetime_utc", "dataset"], as_index=False).agg(
        lead_hour=("lead_hour", "min"), pred_speed_ms=("pred_speed_ms", "mean"), pred_dir_deg=("pred_dir_deg", _circular_group_mean), pred_u10_ms=("pred_u10_ms", "mean"), pred_v10_ms=("pred_v10_ms", "mean")
    )
    merged = forecasts.merge(obs, on=["platform_id", "datetime_utc"], how="left")
    merged["speed_error_ms"] = merged["pred_speed_ms"] - merged["obs_speed_ms"]
    merged["direction_error_deg"] = [angular_difference_deg(p, o) for p, o in zip(merged["pred_dir_deg"], merged["obs_dir_deg"])]
    merged["direction_abs_error_deg"] = merged["direction_error_deg"].abs()
    merged.to_csv(OUT_ANALYSIS_SAMPLES, index=False, encoding="utf-8-sig")
    return obs, merged, platform_summary


def parse_nmc_time(value: str) -> datetime:
    text = str(value).strip()
    for fmt in ("%Y%m%d%H%M", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported NMC time: {value}")


def _extract_callback_json(text: str) -> dict | list:
    match = re.search(r"([\[{].*[\]}])", text, flags=re.S)
    if not match:
        raise ValueError("Could not find JSON payload in NMC callback response.")
    return json.loads(match.group(1))


def fetch_nmc_text(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests is not available")
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12, verify=False)
    response.raise_for_status()
    return response.text


def find_wipha_nmc_metadata() -> tuple[str, str, str, str] | None:
    if requests is None:
        return None
    ts = int(round(time.time() * 1000))
    urls = [
        f"http://typhoon.nmc.cn/weatherservice/typhoon/jsons/list_2025?t={ts}&callback=typhoon_jsons_list_2025",
        f"https://typhoon.nmc.cn/weatherservice/typhoon/jsons/list_2025?t={ts}&callback=typhoon_jsons_list_2025",
    ]
    for url in urls:
        try:
            payload = _extract_callback_json(fetch_nmc_text(url))
        except Exception:
            continue
        rows = payload.get("typhoon", payload) if isinstance(payload, dict) else payload
        if isinstance(rows, dict):
            rows = rows.get("list", rows.get("data", []))
        for row in rows:
            text = json.dumps(row, ensure_ascii=False).lower()
            if "wipha" not in text and "韦帕" not in text and "2506" not in text:
                continue
            flat = row if isinstance(row, list) else list(row.values())
            tc_id, tc_num, name_cn, name_en = None, "2506", "韦帕", "WIPHA"
            for item in flat:
                s = str(item)
                if re.fullmatch(r"\d{6,8}", s) and tc_id is None:
                    tc_id = s
                if s.isdigit() and len(s) == 4:
                    tc_num = s
                if s.upper() == "WIPHA":
                    name_en = s.upper()
            if tc_id:
                return tc_id, tc_num, name_cn, name_en
    return None


def fetch_nmc_track_by_id(tc_id: str, tc_num: str = "2506", name_cn: str = "韦帕", name_en: str = "WIPHA") -> pd.DataFrame:
    ts = int(round(time.time() * 1000))
    url = f"http://typhoon.nmc.cn/weatherservice/typhoon/jsons/view_{tc_id}?t={ts}&callback=typhoon_jsons_view_{tc_id}"
    payload = _extract_callback_json(fetch_nmc_text(url))
    data = payload["typhoon"]
    rows = []
    for v in data[8]:
        rows.append({
            "tc_num": tc_num, "name_cn": name_cn, "name_en": name_en,
            "datetime_utc": parse_nmc_time(v[1]), "vmax_ms": pd.to_numeric(v[7], errors="coerce"),
            "grade": v[3], "lon": float(v[4]), "lat": float(v[5]), "mslp_hpa": pd.to_numeric(v[6], errors="coerce"),
            "attr": "analysis", "source": "NMC",
        })
    return pd.DataFrame(rows)


def fallback_wipha_track() -> pd.DataFrame:
    rows = [
        ("2025-07-17 00:00", 127.5, 18.0, 1000, 15), ("2025-07-17 06:00", 126.0, 18.5, 998, 16),
        ("2025-07-17 12:00", 124.5, 19.0, 996, 18), ("2025-07-17 18:00", 123.0, 19.5, 992, 20),
        ("2025-07-18 00:00", 121.5, 20.0, 990, 23), ("2025-07-18 06:00", 120.0, 20.3, 988, 25),
        ("2025-07-18 12:00", 118.8, 20.7, 985, 28), ("2025-07-18 18:00", 117.5, 21.1, 982, 30),
        ("2025-07-19 00:00", 116.2, 21.4, 980, 33), ("2025-07-19 06:00", 115.0, 21.7, 978, 35),
        ("2025-07-19 12:00", 113.8, 21.9, 975, 38), ("2025-07-19 18:00", 112.8, 22.0, 975, 38),
        ("2025-07-20 00:00", 111.9, 22.1, 978, 35), ("2025-07-20 06:00", 111.0, 22.0, 982, 30),
        ("2025-07-20 12:00", 110.0, 21.8, 985, 28), ("2025-07-20 18:00", 109.0, 21.5, 988, 25),
        ("2025-07-21 00:00", 108.0, 21.2, 990, 23), ("2025-07-21 06:00", 107.0, 20.8, 994, 20),
        ("2025-07-21 12:00", 106.0, 20.4, 996, 18), ("2025-07-21 18:00", 105.0, 20.0, 998, 16),
        ("2025-07-22 00:00", 104.0, 19.6, 1000, 15),
    ]
    return pd.DataFrame({
        "tc_num": "2506", "name_cn": "韦帕", "name_en": "WIPHA",
        "datetime_utc": [pd.Timestamp(r[0]) for r in rows], "lon": [r[1] for r in rows], "lat": [r[2] for r in rows],
        "mslp_hpa": [r[3] for r in rows], "vmax_ms": [r[4] for r in rows], "attr": "analysis", "source": "fallback_approximate",
    })


def load_real_wipha_track(force_refresh: bool = False) -> pd.DataFrame:
    if OUT_REAL_TRACK_CACHE.exists() and not force_refresh:
        df = pd.read_csv(OUT_REAL_TRACK_CACHE)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
        return df
    df, errors = None, []
    try:
        meta = find_wipha_nmc_metadata()
        if meta:
            df = fetch_nmc_track_by_id(*meta)
    except Exception as exc:
        errors.append(str(exc))
    if df is None or df.empty:
        for tc_id in ["3064306", "3064307", "3064310", "3064315", "3064321", "3064324"]:
            try:
                cand = fetch_nmc_track_by_id(tc_id)
                if not cand.empty and cand["datetime_utc"].min() <= WINDOW_END and cand["datetime_utc"].max() >= WINDOW_START:
                    df = cand
                    break
            except Exception as exc:
                errors.append(f"{tc_id}: {exc}")
    if df is None or df.empty:
        warnings.warn("NMC Wipha track retrieval failed; using approximate fallback track. " + "; ".join(errors[:3]))
        df = fallback_wipha_track()
    df = df.sort_values("datetime_utc")
    df.to_csv(OUT_REAL_TRACK_CACHE, index=False, encoding="utf-8-sig")
    return df


def surface_array_path(scheme: str, lead_hour: int, valid_time: datetime) -> Path:
    valid_str = valid_time.strftime("%Y-%m-%d-%H-%M")
    if scheme == "gdas_forecast":
        if lead_hour == 0:
            return PROJECT_ROOT / "model_input" / "single_time_point" / "gdas" / "2025-07-17-00-00" / "input_surface.npy"
        return PROJECT_ROOT / "model_output" / "gdas" / "2025-07-17-00-00" / str(lead_hour) / f"output_surface_{valid_str}.npy"
    if scheme == "era5_lagged_5d":
        return PROJECT_ROOT / "model_output" / "era5" / "2025-07-12-00-00" / str(120 + lead_hour) / f"output_surface_{valid_str}.npy"
    raise ValueError(f"Unsupported scheme: {scheme}")


def read_surface_msl(path: Path) -> np.ndarray:
    if not path.exists() and "model_output" in path.parts:
        cached = path.parents[1] / "timeline_cache" / path.name
        if cached.exists():
            path = cached
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        return np.asarray(arr[0], dtype=float)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=float)
    raise ValueError(f"Unexpected surface array shape {arr.shape} in {path}")


def grid_indices_for_box(search_box: tuple[float, float, float, float]):
    lon_min, lon_max, lat_min, lat_max = search_box
    lons = GRID_LON0 + GRID_DX * np.arange(GRID_NLON)
    lats = GRID_LAT0 + GRID_DY * np.arange(GRID_NLAT)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    return np.where(lat_mask)[0], np.where(lon_mask)[0], lats[lat_mask], lons[lon_mask]


def locate_center_from_surface(path: Path, search_box: tuple[float, float, float, float]) -> dict:
    msl = read_surface_msl(path)
    lat_idx, lon_idx, lats, lons = grid_indices_for_box(search_box)
    sub = np.asarray(msl[np.ix_(lat_idx, lon_idx)], dtype=float)
    flat = np.nanargmin(sub)
    i, j = np.unravel_index(flat, sub.shape)
    raw_msl = float(sub[i, j])
    return {"center_lon": float(lons[j]), "center_lat": float(lats[i]), "min_msl_hpa": raw_msl / 100.0 if raw_msl > 2000 else raw_msl, "file": str(path), "search_box": str(search_box)}


def moving_box(center_lon: float, center_lat: float) -> tuple[float, float, float, float]:
    lon_min, lon_max, lat_min, lat_max = WIPHA_SEARCH_BOX
    return (max(lon_min, center_lon - MOVING_SEARCH_HALF_WIDTH_DEG), min(lon_max, center_lon + MOVING_SEARCH_HALF_WIDTH_DEG), max(lat_min, center_lat - MOVING_SEARCH_HALF_WIDTH_DEG), min(lat_max, center_lat + MOVING_SEARCH_HALF_WIDTH_DEG))


def extract_forecast_track(scheme: str) -> pd.DataFrame:
    rows, last_center = [], None
    for lead in LEAD_HOURS:
        valid = TRACK_INIT + timedelta(hours=lead)
        path = surface_array_path(scheme, lead, valid)
        search_box = WIPHA_SEARCH_BOX if last_center is None else moving_box(*last_center)
        try:
            info = locate_center_from_surface(path, search_box)
            last_center = (info["center_lon"], info["center_lat"])
            error = ""
        except Exception as exc:
            info = {"center_lon": np.nan, "center_lat": np.nan, "min_msl_hpa": np.nan, "file": str(path), "search_box": str(search_box)}
            error = str(exc)
        rows.append({"scheme": scheme, "scheme_label": DATASET_LABELS.get(scheme, scheme), "lead_hour": lead, "valid_time": valid, "lon": info["center_lon"], "lat": info["center_lat"], "min_msl_hpa": info["min_msl_hpa"], "file": info["file"], "search_box": info["search_box"], "error": error})
    return pd.DataFrame(rows)


def interpolate_real_track_to_leads(real_track: pd.DataFrame) -> pd.DataFrame:
    valid_times = [TRACK_INIT + timedelta(hours=h) for h in LEAD_HOURS]
    work = real_track[(real_track["datetime_utc"] >= pd.Timestamp(TRACK_INIT) - pd.Timedelta(hours=12)) & (real_track["datetime_utc"] <= pd.Timestamp(TRACK_INIT) + pd.Timedelta(hours=84))].copy()
    if work.empty:
        return pd.DataFrame({"scheme": "real_track", "scheme_label": "Observed track", "lead_hour": LEAD_HOURS, "valid_time": valid_times, "lon": np.nan, "lat": np.nan, "source": "missing"})
    x = work["datetime_utc"].astype("int64").to_numpy(dtype=float)
    out_x = pd.to_datetime(valid_times).astype("int64").to_numpy(dtype=float)
    return pd.DataFrame({"scheme": "real_track", "scheme_label": "Observed track", "lead_hour": LEAD_HOURS, "valid_time": valid_times, "lon": np.interp(out_x, x, work["lon"].to_numpy(dtype=float), left=np.nan, right=np.nan), "lat": np.interp(out_x, x, work["lat"].to_numpy(dtype=float), left=np.nan, right=np.nan), "source": work["source"].iloc[0] if "source" in work else "NMC"})


def build_tracks_and_errors(real_track: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_interp = interpolate_real_track_to_leads(real_track)
    gdas = extract_forecast_track("gdas_forecast")
    era5 = extract_forecast_track("era5_lagged_5d")
    tracks = pd.concat([real_interp, gdas, era5], ignore_index=True, sort=False)
    errors = []
    for scheme_df in [gdas, era5]:
        scheme = scheme_df["scheme"].iloc[0]
        merged = scheme_df.merge(real_interp[["lead_hour", "lon", "lat"]], on="lead_hour", suffixes=("_pred", "_obs"), how="left")
        for row in merged.itertuples(index=False):
            errors.append({"scheme": scheme, "scheme_label": DATASET_LABELS.get(scheme, scheme), "lead_hour": int(row.lead_hour), "valid_time": row.valid_time, "pred_lon": row.lon_pred, "pred_lat": row.lat_pred, "obs_lon": row.lon_obs, "obs_lat": row.lat_obs, "track_error_km": haversine_km(row.lon_pred, row.lat_pred, row.lon_obs, row.lat_obs)})
    errors_df = pd.DataFrame(errors)
    tracks.to_csv(OUT_TRACKS_CSV, index=False, encoding="utf-8-sig")
    errors_df.to_csv(OUT_TRACK_ERRORS_CSV, index=False, encoding="utf-8-sig")
    return tracks, errors_df


def compute_statistics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = list(merged.groupby(["platform_id", "dataset"]))
    combined = merged.copy(); combined["platform_id"] = "Combined"
    groups.extend(list(combined.groupby(["platform_id", "dataset"])))
    for (platform_id, dataset), sub in groups:
        sub = sub.dropna(subset=["obs_speed_ms", "pred_speed_ms", "obs_dir_deg", "pred_dir_deg"])
        if sub.empty:
            continue
        speed_err = sub["pred_speed_ms"] - sub["obs_speed_ms"]
        dir_err = np.array([angular_difference_deg(p, o) for p, o in zip(sub["pred_dir_deg"], sub["obs_dir_deg"])], dtype=float)
        speed_corr = sub[["obs_speed_ms", "pred_speed_ms"]].corr().iloc[0, 1] if len(sub) >= 2 else np.nan
        dir_corr = sub[["obs_dir_deg", "pred_dir_deg"]].corr().iloc[0, 1] if len(sub) >= 2 else np.nan
        rows.append({"platform_id": platform_id, "dataset": dataset, "dataset_label": DATASET_LABELS.get(dataset, dataset), "sample_count": int(len(sub)), "speed_bias": float(speed_err.mean()), "speed_mae": float(speed_err.abs().mean()), "speed_rmse": float(np.sqrt(np.mean(speed_err.to_numpy(dtype=float) ** 2))), "speed_corr": float(speed_corr) if pd.notna(speed_corr) else np.nan, "direction_bias": float(np.nanmean(dir_err)), "direction_mae": float(np.nanmean(np.abs(dir_err))), "direction_rmse": float(np.sqrt(np.nanmean(dir_err ** 2))), "direction_corr": float(dir_corr) if pd.notna(dir_corr) else np.nan})
    stats = pd.DataFrame(rows).sort_values(["platform_id", "dataset"])
    stats.to_csv(OUT_STATS_CSV, index=False, encoding="utf-8-sig")
    try:
        stats.to_excel(OUT_STATS_XLSX, index=False)
    except Exception as exc:
        warnings.warn(f"Could not write Excel statistics file: {exc}")
    return stats


def plot_track_buoy_locations(real_track: pd.DataFrame, obs: pd.DataFrame) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.set_facecolor("#F4F5F7"); ax.grid(True, color="white", linewidth=1.0)
    ax.set_xlim(MAP_AREA[0], MAP_AREA[1]); ax.set_ylim(MAP_AREA[2], MAP_AREA[3])
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("Typhoon Wipha Track and Selected Platform Locations", loc="left", fontweight="bold")
    case_track = real_track[real_track["datetime_utc"].between(WINDOW_START, WINDOW_END)]
    if not case_track.empty:
        ax.plot(case_track["lon"], case_track["lat"], color="#222222", linewidth=2.0, marker="o", markersize=3.5, label="Observed Wipha track")
        for _, row in case_track.iloc[:: max(1, len(case_track) // 8)].iterrows():
            ax.text(row["lon"] + 0.15, row["lat"] + 0.12, pd.Timestamp(row["datetime_utc"]).strftime("%m-%d %H"), fontsize=7.5)
    for platform_id, sub in obs.groupby("platform_id"):
        color = PLATFORM_COLORS.get(platform_id, "#777777")
        ax.scatter(sub["longitude"], sub["latitude"], s=22, alpha=0.45, color=color, edgecolor="none", label=f"{platform_id} positions")
        mean_lon, mean_lat = sub["longitude"].mean(), sub["latitude"].mean()
        ax.scatter([mean_lon], [mean_lat], marker="*", s=180, color=color, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(mean_lon + 0.25, mean_lat + 0.25, platform_id, color=color, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.text(0.5, 0.015, "Platform points show observed positions during 2025-07-17 00 UTC to 2025-07-22 23 UTC.", ha="center", fontsize=8.8, color="#555555")
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.98])
    fig.savefig(OUT_TRACK_BUOYS_PNG, bbox_inches="tight"); fig.savefig(OUT_TRACK_BUOYS_SVG, bbox_inches="tight"); plt.close(fig)


def plot_timeseries(obs: pd.DataFrame, merged: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8), sharex=True, constrained_layout=False)
    for col, platform_id in enumerate(SELECTED_PLATFORMS):
        obs_sub = obs[obs["platform_id"] == platform_id].sort_values("datetime_utc")
        ax_speed, ax_dir = axes[0, col], axes[1, col]
        for ax in (ax_speed, ax_dir):
            ax.set_facecolor("#F4F5F7"); ax.grid(True, color="white", linewidth=1.0); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_xlim(WINDOW_START, WINDOW_END)
        ax_speed.plot(obs_sub["datetime_utc"], obs_sub["obs_speed_ms"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.8, markersize=3.8, label="Buoy observation")
        ax_dir.plot(obs_sub["datetime_utc"], obs_sub["obs_dir_deg"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.4, markersize=3.5, label="Buoy observation")
        for dataset in DATASETS:
            sub = merged[(merged["platform_id"] == platform_id) & (merged["dataset"] == dataset)].sort_values("datetime_utc")
            ax_speed.plot(sub["datetime_utc"], sub["pred_speed_ms"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.5, markersize=3.2, label=DATASET_LABELS[dataset])
            ax_dir.plot(sub["datetime_utc"], sub["pred_dir_deg"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.2, markersize=3.0, label=DATASET_LABELS[dataset])
        ax_speed.set_title(f"({chr(ord('a') + col)}) {platform_id} wind speed", loc="left", fontweight="bold")
        ax_dir.set_title(f"({chr(ord('c') + col)}) {platform_id} wind direction", loc="left", fontweight="bold")
        ax_speed.set_ylabel("Wind speed (m s$^{-1}$)"); ax_dir.set_ylabel("Wind direction (°)")
        ax_dir.set_ylim(0, 360); ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H UTC")); ax_dir.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    fig.suptitle("Typhoon Wipha Case: Platform Wind Speed and Direction Time Series", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.955])
    fig.savefig(OUT_TIMESERIES_PNG, bbox_inches="tight"); fig.savefig(OUT_TIMESERIES_SVG, bbox_inches="tight"); plt.close(fig)


def plot_statistics_table(stats: pd.DataFrame) -> None:
    set_plot_style()
    display_cols = ["platform_id", "dataset_label", "sample_count", "speed_mae", "speed_rmse", "speed_corr", "direction_mae", "direction_rmse"]
    table = stats[display_cols].copy()
    table.columns = ["Platform", "Dataset", "N", "Speed MAE", "Speed RMSE", "Speed CC", "Dir MAE", "Dir RMSE"]
    for col in ["Speed MAE", "Speed RMSE", "Speed CC", "Dir MAE", "Dir RMSE"]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    fig_height = max(2.6, 0.42 * len(table) + 1.2)
    fig, ax = plt.subplots(figsize=(11.4, fig_height)); ax.axis("off")
    ax.set_title("Typhoon Wipha Case Wind Verification Statistics", loc="left", fontweight="bold", pad=12)
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.2); tbl.scale(1.0, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#4C72B0"); cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F5F7")
        else:
            cell.set_facecolor("white")
    fig.tight_layout(); fig.savefig(OUT_STATS_TABLE_PNG, bbox_inches="tight"); fig.savefig(OUT_STATS_TABLE_SVG, bbox_inches="tight"); plt.close(fig)


def plot_track_error(tracks: pd.DataFrame, errors: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=False)
    ax_map, ax_err = axes
    for ax in axes:
        ax.set_facecolor("#F4F5F7"); ax.grid(True, color="white", linewidth=1.0); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax_map.set_title("(a) 72 h track forecasts from 2025-07-17 00 UTC", loc="left", fontweight="bold")
    ax_map.set_xlim(MAP_AREA[0], MAP_AREA[1]); ax_map.set_ylim(MAP_AREA[2], MAP_AREA[3])
    ax_map.set_xlabel("Longitude (°E)"); ax_map.set_ylabel("Latitude (°N)")
    style_map = {
        "real_track": {"label": "Observed track", "color": "#222222", "marker": "o", "linestyle": "-"},
        "gdas_forecast": {"label": "GDAS forecast", "color": DATASET_COLORS["gdas_forecast"], "marker": "^", "linestyle": "--"},
        "era5_lagged_5d": {"label": "ERA5 lagged 5d forecast", "color": DATASET_COLORS["era5_lagged_5d"], "marker": "s", "linestyle": "--"},
    }
    for scheme, style in style_map.items():
        sub = tracks[tracks["scheme"] == scheme].sort_values("lead_hour")
        if sub.empty:
            continue
        ax_map.plot(sub["lon"], sub["lat"], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.8, markersize=4.0, label=style["label"])
        for _, row in sub[sub["lead_hour"].isin([0, 24, 48, 72])].iterrows():
            if pd.notna(row["lon"]) and pd.notna(row["lat"]):
                ax_map.text(row["lon"] + 0.12, row["lat"] + 0.12, f"+{int(row['lead_hour'])}h", fontsize=7.5, color=style["color"])
    ax_map.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax_err.set_title("(b) Track position error", loc="left", fontweight="bold")
    for dataset in DATASETS:
        sub = errors[errors["scheme"] == dataset].sort_values("lead_hour")
        if sub.empty:
            continue
        ax_err.plot(sub["lead_hour"], sub["track_error_km"], color=DATASET_COLORS[dataset], marker="o", linewidth=1.8, markersize=4.0, label=DATASET_LABELS[dataset])
    ax_err.set_xlim(0, 72); ax_err.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax_err.set_xlabel("Forecast lead time (h)"); ax_err.set_ylabel("Track error (km)")
    ax_err.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    fig.suptitle("Typhoon Wipha Track Forecast Error Comparison", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.95])
    fig.savefig(OUT_TRACK_ERROR_PNG, bbox_inches="tight"); fig.savefig(OUT_TRACK_ERROR_SVG, bbox_inches="tight"); plt.close(fig)


def run_workflow(force_refresh_track: bool = False) -> list[Path]:
    ensure_dirs()
    obs, merged, _ = prepare_buoy_case_data()
    real_track = load_real_wipha_track(force_refresh=force_refresh_track)
    stats = compute_statistics(merged)
    tracks, errors = build_tracks_and_errors(real_track)
    plot_track_buoy_locations(real_track, obs)
    plot_timeseries(obs, merged)
    plot_statistics_table(stats)
    plot_track_error(tracks, errors)
    return [
        OUT_CANDIDATE_SUMMARY, OUT_ANALYSIS_SAMPLES, OUT_REAL_TRACK_CACHE,
        OUT_STATS_CSV, OUT_STATS_XLSX, OUT_STATS_TABLE_PNG, OUT_STATS_TABLE_SVG,
        OUT_TRACKS_CSV, OUT_TRACK_ERRORS_CSV, OUT_TRACK_BUOYS_PNG, OUT_TRACK_BUOYS_SVG,
        OUT_TIMESERIES_PNG, OUT_TIMESERIES_SVG, OUT_TRACK_ERROR_PNG, OUT_TRACK_ERROR_SVG,
    ]


def main() -> None:
    outputs = run_workflow(force_refresh_track=False)
    print("Wipha case analysis outputs:")
    for path in outputs:
        print(path if path.exists() else f"{path} (not written)")


if __name__ == "__main__":
    main()
