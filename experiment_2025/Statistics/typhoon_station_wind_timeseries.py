"""Match typhoon station observations with four Pangu experiments.

This stage reads raw station CSV/Excel and forecast NetCDF files and writes
matched CSV files plus a JSON run summary. It does not import Matplotlib.
The selected cases use one fixed UTC target initialization per typhoon.
Case start/end fields are raw Beijing-time observation windows.
Plot the resulting CSV with plots/plot_typhoon_station_wind_timeseries.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

STATS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATS_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(STATS_ROOT))

DEFAULT_STATION_FILE = PROJECT_ROOT / "experiment_2025" / "icoads" / "2025_全部逐小时_风速风向_清洗.csv"
DEFAULT_OUTPUT_ROOT = Path(r"H:\pangu_2025_data\model_output\experiment_2025")
DEFAULT_DATA_ROOT = STATS_ROOT / "data"

EXPERIMENTS = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")

CASES = (
    {
        "typhoon": "韦帕 (Wipha, 2025)",
        "station": "遮浪",
        "lon": 115.6,
        "lat": 22.7,
        "start": datetime(2025, 7, 19),
        "end": datetime(2025, 7, 22),
        "target_time_utc": datetime(2025, 7, 18),
    },
    {
        "typhoon": "范斯高 (Francisco, 2025)",
        "station": "大陈",
        "lon": 121.9,
        "lat": 28.5,
        "start": datetime(2025, 7, 24, 8),
        "end": datetime(2025, 7, 27, 8),
        "target_time_utc": datetime(2025, 7, 24),
    },
    {
        "typhoon": "杨柳 (Podul, 2025)",
        "station": "北礵",
        "lon": 120.3,
        "lat": 26.7,
        "start": datetime(2025, 8, 12, 8),
        "end": datetime(2025, 8, 15, 8),
        "target_time_utc": datetime(2025, 8, 12),
    },
)


def _find_column(columns: list[str], *tokens: str) -> str:
    for column in columns:
        if all(token in column for token in tokens):
            return column
    raise KeyError(f"Cannot find column containing {tokens!r}; available={columns!r}")


def load_station_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="gb18030")
    else:
        frame = pd.read_excel(path)
    columns = [str(column).strip() for column in frame.columns]
    frame.columns = columns
    time_col = _find_column(columns, "时")
    station_col = _find_column(columns, "站")
    lon_col = _find_column(columns, "经")
    lat_col = _find_column(columns, "纬")
    speed_col = _find_column(columns, "风速")
    direction_col = _find_column(columns, "风向")
    frame = frame.rename(
        columns={
            time_col: "station_time_raw",
            station_col: "station",
            lon_col: "longitude",
            lat_col: "latitude",
            speed_col: "obs_speed",
            direction_col: "obs_direction",
        }
    )
    frame["station_time_raw"] = pd.to_datetime(frame["station_time_raw"], errors="coerce")
    for column in ("longitude", "latitude", "obs_speed", "obs_direction"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["station"] = frame["station"].astype(str).str.strip()
    frame = frame.dropna(
        subset=["station_time_raw", "station", "longitude", "latitude", "obs_speed", "obs_direction"]
    ).copy()
    return frame.sort_values(["station", "station_time_raw"]).reset_index(drop=True)


def _target_and_lead(valid_time: datetime, target_time: datetime | None = None) -> tuple[datetime, int]:
    if target_time is None:
        target_time = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_hours = (valid_time - target_time).total_seconds() / 3600
    lead_hour = int(elapsed_hours)
    if not elapsed_hours.is_integer() or lead_hour < 0 or lead_hour > 120 or lead_hour % 3:
        raise ValueError(f"Station timestamp does not map to a 3-hour model lead: {valid_time}")
    return target_time, lead_hour


def _nearest_model_time(valid_time_utc: datetime) -> tuple[datetime, float]:
    """Return the nearest 3-hour model valid time and offset in hours."""
    day_start = valid_time_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_hours = (valid_time_utc - day_start).total_seconds() / 3600.0
    lower_hour = int(np.floor(elapsed_hours / 3.0) * 3)
    lower = day_start + timedelta(hours=lower_hour)
    upper = lower + timedelta(hours=3)
    candidates = (lower, upper)
    model_time = min(candidates, key=lambda item: abs((item - valid_time_utc).total_seconds()))
    offset_hours = (model_time - valid_time_utc).total_seconds() / 3600.0
    return model_time, offset_hours


def match_model_data(
    station_data: pd.DataFrame,
    output_root: Path,
    timezone_mode: str,
) -> tuple[pd.DataFrame, list[dict]]:
    # Load the NetCDF stack only when matching is requested.
    if timezone_mode not in {"utc", "beijing"}:
        raise ValueError(f"Unsupported timezone mode: {timezone_mode}")
    from experiment_2025.Statistics.statistics import GridSampler, model_path, wind_direction

    sampler_target = CASES[0].get("target_time_utc") or CASES[0]["start"]
    first_path = model_path(output_root, "era5_direct", sampler_target, 0)
    if not first_path.exists():
        raise FileNotFoundError(f"Cannot initialize model grid sampler; missing {first_path}")
    sampler = GridSampler(first_path, [42.0, 103.0, 13.0, 130.0])
    rows: list[dict] = []
    missing: list[dict] = []

    for case in CASES:
        selected = station_data[
            (station_data["station"] == case["station"])
            & (station_data["station_time_raw"] >= pd.Timestamp(case["start"]))
            & (station_data["station_time_raw"] <= pd.Timestamp(case["end"]))
        ].copy()
        if selected.empty:
            raise ValueError(f"No station records found for {case['station']} in requested window")
        for record in selected.itertuples(index=False):
            station_time_raw = record.station_time_raw.to_pydatetime()
            if timezone_mode == "beijing":
                station_time_utc = station_time_raw - timedelta(hours=8)
            else:
                station_time_utc = station_time_raw
            model_valid_time_utc, match_offset_hours = _nearest_model_time(station_time_utc)
            target_time, lead_hour = _target_and_lead(model_valid_time_utc, case.get("target_time_utc"))
            case_start_utc = case["start"] - timedelta(hours=8) if timezone_mode == "beijing" else case["start"]
            plot_lead_hour = (station_time_utc - case_start_utc).total_seconds() / 3600.0
            row = {
                "typhoon": case["typhoon"],
                "station": case["station"],
                "longitude": case["lon"],
                "latitude": case["lat"],
                "station_time_raw": station_time_raw.strftime("%Y-%m-%d %H:%M"),
                "station_time_utc": station_time_utc.strftime("%Y-%m-%d %H:%M"),
                "model_valid_time_utc": model_valid_time_utc.strftime("%Y-%m-%d %H:%M"),
                "match_offset_hours": float(match_offset_hours),
                "plot_lead_hour": float(plot_lead_hour),
                "target_time_utc": target_time.strftime("%Y-%m-%d %H:%M"),
                "lead_hour": lead_hour,
                "obs_speed": float(record.obs_speed),
                "obs_direction": float(record.obs_direction),
            }
            for experiment in EXPERIMENTS:
                path = model_path(output_root, experiment, target_time, lead_hour)
                if not path.is_file():
                    missing.append(
                        {
                            "experiment": experiment,
                            "station": case["station"],
                            "timezone_mode": timezone_mode,
                            "station_time_raw": station_time_raw.strftime("%Y-%m-%d %H:%M"),
                            "station_time_utc": station_time_utc.strftime("%Y-%m-%d %H:%M"),
                            "model_valid_time_utc": model_valid_time_utc.strftime("%Y-%m-%d %H:%M"),
                            "match_offset_hours": float(match_offset_hours),
                            "target_time_utc": target_time.strftime("%Y-%m-%d %H:%M"),
                            "lead_hour": lead_hour,
                            "path": str(path),
                        }
                    )
                    row[f"{experiment}_speed"] = np.nan
                    row[f"{experiment}_direction"] = np.nan
                    continue
                pred_u, pred_v = sampler.sample(
                    path,
                    np.asarray([case["lat"]], dtype=float),
                    np.asarray([case["lon"]], dtype=float),
                )
                u = float(pred_u[0])
                v = float(pred_v[0])
                row[f"{experiment}_speed"] = float(np.hypot(u, v))
                row[f"{experiment}_direction"] = float(wind_direction(np.asarray([u]), np.asarray([v]))[0])
            rows.append(row)
    return pd.DataFrame(rows), missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-file", type=Path, default=DEFAULT_STATION_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--timezone-mode", choices=("utc", "beijing", "both"), default="beijing",
        help="raw station timestamps are confirmed Beijing time; utc/both are optional sensitivity runs",
    )
    args = parser.parse_args(argv)

    station_data = load_station_data(args.station_file)
    args.data_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "station_file": str(args.station_file),
        "model_output_root": str(args.output_root),
        "initialization_policy": (
            "fixed" if all(case.get("target_time_utc") is not None for case in CASES)
            else "daily_00_utc" if all(case.get("target_time_utc") is None for case in CASES)
            else "mixed"
        ),
        "plot_time_policy": "elapsed hours within each observation window, not forecast lead",
        "time_interpretation": {
            "utc": "Workbook timestamps are used directly as UTC model-valid times.",
            "beijing": "Workbook timestamps are treated as Beijing time; 8 hours are subtracted before lookup. The nearest 3-hour model valid time is used and match_offset_hours is recorded.",
        },
        "experiments": list(EXPERIMENTS),
        "modes": {},
    }
    modes = ("utc", "beijing") if args.timezone_mode == "both" else (args.timezone_mode,)
    for mode in modes:
        matched, missing = match_model_data(station_data, args.output_root, mode)
        csv_path = args.data_root / f"typhoon_station_wind_timeseries_2025_{mode}.csv"
        matched.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary["modes"][mode] = {
            "csv": str(csv_path),
            "cases": [
                {
                    "typhoon": case["typhoon"],
                    "station": case["station"],
                    "longitude": case["lon"],
                    "latitude": case["lat"],
                    "start": case["start"].strftime("%Y-%m-%d %H:%M"),
                    "end": case["end"].strftime("%Y-%m-%d %H:%M"),
                    "target_time_utc": (case["target_time_utc"].strftime("%Y-%m-%d %H:%M")
                                        if case.get("target_time_utc") is not None else None),
                    "record_count": int((
                        (matched["typhoon"] == case["typhoon"])
                        & (matched["station"] == case["station"])
                    ).sum()),
                }
                for case in CASES
            ],
            "missing_outputs": missing,
            "matched_rows": int(len(matched)),
            "match_offset_hours": sorted(matched["match_offset_hours"].unique().tolist()),
        }
        if mode == "utc":
            # Preserve the historical UTC-compatible alias only for UTC runs.
            matched.to_csv(args.data_root / "typhoon_station_wind_timeseries_2025.csv",
                           index=False, encoding="utf-8-sig")

    summary_path = args.data_root / "typhoon_station_wind_timeseries_2025_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    for mode, info in summary["modes"].items():
        print(f"{mode}: {info['csv']}; missing outputs={len(info['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
