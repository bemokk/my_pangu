"""Plot typhoon station wind observations and four Pangu experiments.

The station workbook has no timezone metadata, so two independent matching
interpretations are produced:

* ``utc``: workbook timestamps are treated as UTC;
* ``beijing``: workbook timestamps are treated as Beijing time and converted
  to UTC by subtracting eight hours before looking up model output.

The model output is available every three hours.  Station records are matched
to the nearest available three-hour model valid time and that offset is stored
in the matched CSV.  The plot x-axis is the elapsed hour from the beginning of
each requested typhoon window.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLOTS_ROOT.parents[1]
STATS_ROOT = PROJECT_ROOT / "experiment_2025" / "Statistics"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(STATS_ROOT))

DEFAULT_STATION_FILE = PROJECT_ROOT / "experiment_2025" / "icoads" / "2025_全部逐小时_风速风向_清洗.csv"
DEFAULT_OUTPUT_ROOT = Path(r"H:\pangu_2025_data\model_output\experiment_2025")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "experiment_2025" / "Statistics" / "data"
DEFAULT_FIGURES_ROOT = PLOTS_ROOT / "figures"

EXPERIMENTS = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")
EXPERIMENT_LABELS = {
    "era5_direct": "E0",
    "gdas_direct": "G0",
    "era5_lagged_120h": "E120",
    "gdas_lagged_120h": "G120",
}
STYLES = {
    "observed": {"label": "观测", "color": "#333333", "linestyle": "-", "linewidth": 1.8},
    "era5_direct": {"label": "E0", "color": "#43A3EF", "linestyle": "-", "linewidth": 1.45},
    "gdas_direct": {"label": "G0", "color": "#FEA040", "linestyle": "-", "linewidth": 1.45},
    "era5_lagged_120h": {"label": "E120", "color": "#009E73", "linestyle": "--", "linewidth": 1.45},
    "gdas_lagged_120h": {"label": "G120", "color": "#EF767B", "linestyle": "--", "linewidth": 1.45},
}

CASES = (
    {
        "typhoon": "丹娜丝 (Danas, 2025)",
        "station": "大陈",
        "lon": 121.9,
        "lat": 28.5,
        "start": datetime(2025, 7, 7),
        "end": datetime(2025, 7, 10),
    },
    {
        "typhoon": "韦帕 (Wipha, 2025)",
        "station": "遮浪",
        "lon": 115.6,
        "lat": 22.7,
        "start": datetime(2025, 7, 19),
        "end": datetime(2025, 7, 22),
    },
    {
        "typhoon": "竹节草 (Co-may, 2025)",
        "station": "嵊山",
        "lon": 122.8,
        "lat": 30.8,
        "start": datetime(2025, 7, 29),
        "end": datetime(2025, 8, 1),
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


def _target_and_lead(valid_time: datetime) -> tuple[datetime, int]:
    target_time = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
    lead_hour = int((valid_time - target_time).total_seconds() // 3600)
    if lead_hour < 0 or lead_hour > 120 or lead_hour % 3:
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
    # Import the NetCDF/statistics stack only for matching.  This keeps
    # ``--plot-only`` usable in a lightweight plotting environment.
    from experiment_2025.Statistics.statistics import GridSampler, model_path, wind_direction

    first_path = model_path(output_root, "era5_direct", CASES[0]["start"], 0)
    if not first_path.exists():
        raise FileNotFoundError(f"Cannot initialize model grid sampler; missing {first_path}")
    sampler = GridSampler(first_path, [42.0, 103.0, 13.0, 130.0])
    if timezone_mode not in {"utc", "beijing"}:
        raise ValueError(f"Unsupported timezone mode: {timezone_mode}")
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
            target_time, lead_hour = _target_and_lead(model_valid_time_utc)
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


def _break_direction_wrap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    jumps = np.abs(np.diff(values)) > 180.0
    values[1:][jumps] = np.nan
    return values


def _configure_axes(ax, direction: bool, lead_hours: np.ndarray, start_date_utc: datetime) -> None:
    ax.grid(True, color="#BFBFBF", linewidth=0.65, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    # Leave a small margin before the 0 h tick so the two-line first label is
    # fully visible instead of being clipped at the left axes border.
    ax.set_xlim(-2.5, float(np.nanmax(lead_hours)))
    max_lead = int(np.ceil(np.nanmax(lead_hours) / 12.0) * 12)
    ticks = np.arange(0, max_lead + 1, 12, dtype=float)
    labels = [f"0\n{start_date_utc:%Y-%m-%d}"] + [f"{int(tick)}h" for tick in ticks[1:]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8, pad=2)
    if direction:
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_ylabel("风向 (°)")
    else:
        ax.set_ylabel("风速 (m s$^{-1}$)")
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.8)


def plot_figure(frame: pd.DataFrame, timezone_mode: str, figures_root: Path) -> Path:
    figures_root.mkdir(parents=True, exist_ok=True)
    timezone_label = "UTC" if timezone_mode == "utc" else "北京时间 (UTC+8)"
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.5), sharex=False, constrained_layout=False)
    panel_letters = iter("abcdef")

    for row_index, case in enumerate(CASES):
        sub = frame[frame["station"] == case["station"]].copy()
        lead_hours = sub["plot_lead_hour"].to_numpy(float)
        start_date_utc = case["start"] - timedelta(hours=8) if timezone_mode == "beijing" else case["start"]
        speed_ax, direction_ax = axes[row_index]
        for ax, direction in ((speed_ax, False), (direction_ax, True)):
            _configure_axes(ax, direction, lead_hours, start_date_utc)

        speed_ax.set_title(
            f"({next(panel_letters)}) {case['typhoon']} · {case['station']}站 "
            f"({case['lon']:.1f}°E, {case['lat']:.1f}°N) 风速",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )
        direction_ax.set_title(
            f"({next(panel_letters)}) {case['typhoon']} · {case['station']}站 风向",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )

        series = {
            "observed": (sub["obs_speed"].to_numpy(float), sub["obs_direction"].to_numpy(float)),
            **{
                experiment: (
                    sub[f"{experiment}_speed"].to_numpy(float),
                    sub[f"{experiment}_direction"].to_numpy(float),
                )
                for experiment in EXPERIMENTS
            },
        }
        for name, (speed, direction) in series.items():
            style = STYLES[name]
            speed_ax.plot(lead_hours, speed, label=style["label"], **{k: style[k] for k in ("color", "linestyle", "linewidth")})
            direction_ax.plot(
                lead_hours,
                _break_direction_wrap(direction),
                label=style["label"],
                **{k: style[k] for k in ("color", "linestyle", "linewidth")},
            )
        if row_index == len(CASES) - 1:
            speed_ax.set_xlabel("预报时效")
            direction_ax.set_xlabel("预报时效")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=5, frameon=True)
    fig.suptitle(
        f"2025年台风海洋站10 m风场对比（{timezone_label}）\n"
        "E0、G0为实线；E120、G120为虚线",
        fontsize=14,
        y=0.995,
    )
    # Avoid bbox/tight-layout font measurement crashes on some Windows
    # Matplotlib builds with mixed Chinese and fallback fonts.
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.88, hspace=0.42, wspace=0.18)
    stem = figures_root / f"typhoon_station_wind_timeseries_{timezone_mode}"
    output = stem.with_suffix(".png")
    fig.savefig(output, dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-file", type=Path, default=DEFAULT_STATION_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--match-only", action="store_true", help="match the station/model data without drawing figures")
    parser.add_argument("--plot-only", action="store_true", help="draw figures from --matched-csv")
    parser.add_argument("--matched-csv", type=Path, default=None)
    parser.add_argument("--timezone-mode", choices=("utc", "beijing"), default=None)
    args = parser.parse_args(argv)

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"],
            "font.size": 10,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    # Plot-only mode is useful in environments where NetCDF dependencies are
    # unavailable.  The CSV must have been created by a matching run.
    if args.plot_only:
        if args.matched_csv is None:
            raise ValueError("--plot-only requires --matched-csv")
        mode = args.timezone_mode or "utc"
        matched = pd.read_csv(args.matched_csv)
        output = plot_figure(matched, mode, args.figures_root)
        print(f"Figure: {output}")
        return 0

    station_data = load_station_data(args.station_file)
    args.data_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "station_file": str(args.station_file),
        "model_output_root": str(args.output_root),
        "time_interpretation": {
            "utc": "Workbook timestamps are used directly as UTC model-valid times.",
            "beijing": "Workbook timestamps are treated as Beijing time; 8 hours are subtracted before model lookup. Because model output is every 3 hours, the nearest valid model time is used and match_offset_hours is recorded.",
        },
        "experiments": list(EXPERIMENTS),
        "modes": {},
    }
    utc_frame = None
    for mode in ("utc", "beijing"):
        matched, missing = match_model_data(station_data, args.output_root, mode)
        if mode == "utc":
            utc_frame = matched.copy()
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
                    "record_count": int((matched["station"] == case["station"]).sum()),
                }
                for case in CASES
            ],
            "missing_outputs": missing,
            "matched_rows": int(len(matched)),
            "match_offset_hours": sorted(matched["match_offset_hours"].unique().tolist()),
        }
        if not args.match_only:
            plot_figure(matched, mode, args.figures_root)
    # Keep the historical filename as a UTC-compatible alias, while the two
    # explicit files above remain the authoritative timezone-specific outputs.
    if utc_frame is not None:
        utc_frame.to_csv(
            args.data_root / "typhoon_station_wind_timeseries_2025.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary_path = args.data_root / "typhoon_station_wind_timeseries_2025_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    for mode in ("utc", "beijing"):
        mode_info = summary["modes"][mode]
        print(f"{mode}: {mode_info['csv']}; missing outputs={len(mode_info['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
