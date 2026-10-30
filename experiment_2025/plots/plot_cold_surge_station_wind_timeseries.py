"""Plot the 5 February 2025 cold-surge station wind comparison.

The requested forecast valid window is 2025-02-05 00 UTC through
2025-02-10 00 UTC.  Direct experiments use the 2025-02-05 00 UTC input;
lagged experiments use the 2025-01-31 00 UTC input and are evaluated from
their 2025-02-05 T0 output directory.  The station CSV is tested under two
timezone assumptions independently.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

TARGET_TIME = datetime(2025, 2, 5, 0)
END_TIME = TARGET_TIME + timedelta(hours=120)
EXPERIMENTS = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")
STYLES = {
    "observed": {"label": "观测", "color": "#333333", "linestyle": "-", "linewidth": 1.8},
    "era5_direct": {"label": "E0", "color": "#43A3EF", "linestyle": "-", "linewidth": 1.45},
    "gdas_direct": {"label": "G0", "color": "#FEA040", "linestyle": "-", "linewidth": 1.45},
    "era5_lagged_120h": {"label": "E120", "color": "#009E73", "linestyle": "--", "linewidth": 1.45},
    "gdas_lagged_120h": {"label": "G120", "color": "#EF767B", "linestyle": "--", "linewidth": 1.45},
}
CASES = (
    {"station": "连云港", "display": "连云港", "lon": 119.4, "lat": 34.8},
    {"station": "南麂", "display": "南麂岛", "lon": 121.1, "lat": 27.5},
)


def load_station_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="gb18030")
    else:
        frame = pd.read_excel(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    rename = {
        next(column for column in frame.columns if "时间" in column): "station_time_raw",
        next(column for column in frame.columns if "站点" in column): "station",
        next(column for column in frame.columns if "经度" in column): "longitude",
        next(column for column in frame.columns if "纬度" in column): "latitude",
        next(column for column in frame.columns if "风速" in column): "obs_speed",
        next(column for column in frame.columns if "风向" in column): "obs_direction",
    }
    frame = frame.rename(columns=rename)
    frame["station_time_raw"] = pd.to_datetime(frame["station_time_raw"], errors="coerce")
    for column in ("longitude", "latitude", "obs_speed", "obs_direction"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["station"] = frame["station"].astype(str).str.strip()
    return frame.dropna(subset=["station_time_raw", "station", "obs_speed", "obs_direction"]).copy()


def nearest_model_time(valid_time_utc: datetime) -> tuple[datetime, float]:
    day_start = valid_time_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed_hours = (valid_time_utc - day_start).total_seconds() / 3600.0
    lower = day_start + timedelta(hours=int(np.floor(elapsed_hours / 3.0) * 3))
    upper = lower + timedelta(hours=3)
    model_time = min((lower, upper), key=lambda item: abs((item - valid_time_utc).total_seconds()))
    return model_time, (model_time - valid_time_utc).total_seconds() / 3600.0


def match_data(station_data: pd.DataFrame, output_root: Path, timezone_mode: str) -> tuple[pd.DataFrame, list[dict]]:
    from experiment_2025.Statistics.statistics import GridSampler, model_path, wind_direction

    from_path = model_path(output_root, "era5_direct", TARGET_TIME, 0)
    if not from_path.is_file():
        raise FileNotFoundError(f"Cannot initialize model grid sampler; missing {from_path}")
    sampler = GridSampler(from_path, [42.0, 103.0, 13.0, 130.0])
    rows: list[dict] = []
    missing: list[dict] = []
    for case in CASES:
        selected = station_data[station_data["station"] == case["station"]].copy()
        selected["station_time_utc_tmp"] = selected["station_time_raw"] - (
            pd.Timedelta(hours=8) if timezone_mode == "beijing" else pd.Timedelta(0)
        )
        selected = selected[
            (selected["station_time_utc_tmp"] >= pd.Timestamp(TARGET_TIME))
            & (selected["station_time_utc_tmp"] <= pd.Timestamp(END_TIME))
        ]
        if selected.empty:
            raise ValueError(f"No records found for {case['station']} under {timezone_mode} interpretation")
        for record in selected.sort_values("station_time_raw").itertuples(index=False):
            station_time_raw = record.station_time_raw.to_pydatetime()
            station_time_utc = station_time_raw - timedelta(hours=8) if timezone_mode == "beijing" else station_time_raw
            model_valid_time, match_offset = nearest_model_time(station_time_utc)
            model_lead = int(round((model_valid_time - TARGET_TIME).total_seconds() / 3600.0))
            if not (0 <= model_lead <= 120 and model_lead % 3 == 0):
                raise ValueError(f"Invalid model lead {model_lead} for {station_time_utc}")
            row = {
                "station": case["station"],
                "station_display": case["display"],
                "longitude": case["lon"],
                "latitude": case["lat"],
                "station_time_raw": station_time_raw.strftime("%Y-%m-%d %H:%M"),
                "station_time_utc": station_time_utc.strftime("%Y-%m-%d %H:%M"),
                "model_valid_time_utc": model_valid_time.strftime("%Y-%m-%d %H:%M"),
                "plot_lead_hour": (station_time_utc - TARGET_TIME).total_seconds() / 3600.0,
                "model_lead_hour": model_lead,
                "match_offset_hours": float(match_offset),
                "obs_speed": float(record.obs_speed),
                "obs_direction": float(record.obs_direction),
            }
            for experiment in EXPERIMENTS:
                path = model_path(output_root, experiment, TARGET_TIME, model_lead)
                if not path.is_file():
                    missing.append({"experiment": experiment, "station": case["station"], "path": str(path)})
                    row[f"{experiment}_speed"] = np.nan
                    row[f"{experiment}_direction"] = np.nan
                    continue
                pred_u, pred_v = sampler.sample(
                    path, np.asarray([case["lat"]], dtype=float), np.asarray([case["lon"]], dtype=float)
                )
                u, v = float(pred_u[0]), float(pred_v[0])
                row[f"{experiment}_speed"] = float(np.hypot(u, v))
                row[f"{experiment}_direction"] = float(wind_direction(np.asarray([u]), np.asarray([v]))[0])
            rows.append(row)
    return pd.DataFrame(rows), missing


def break_direction_wrap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[1:][np.abs(np.diff(values)) > 180.0] = np.nan
    return values


def plot_figure(frame: pd.DataFrame, timezone_mode: str, figures_root: Path) -> Path:
    figures_root.mkdir(parents=True, exist_ok=True)
    timezone_label = "UTC假设" if timezone_mode == "utc" else "北京时间假设（转换为UTC匹配）"
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), sharex=True, constrained_layout=False)
    for row_index, case in enumerate(CASES):
        sub = frame[frame["station"] == case["station"]].sort_values("plot_lead_hour")
        x = sub["plot_lead_hour"].to_numpy(float)
        ticks = np.arange(0, 121, 12, dtype=float)
        labels = ["0\n2025-02-05 00 UTC"] + [f"{int(value)}h" for value in ticks[1:]]
        speed_ax, direction_ax = axes[row_index]
        for ax, direction in ((speed_ax, False), (direction_ax, True)):
            ax.set_xlim(-2.5, 120)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=8)
            ax.grid(True, color="#BFBFBF", linewidth=0.65, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            if direction:
                ax.set_ylim(0, 360)
                ax.set_yticks([0, 90, 180, 270, 360])
                ax.set_ylabel("风向 (°)")
            else:
                ax.set_ylabel("风速 (m s$^{-1}$)")
            for spine in ax.spines.values():
                spine.set_color("#444444")
                spine.set_linewidth(0.8)
        speed_ax.set_title(f"({chr(97 + row_index * 2)}) {case['display']} 风速", loc="left", fontweight="bold", fontsize=11)
        direction_ax.set_title(f"({chr(98 + row_index * 2)}) {case['display']} 风向", loc="left", fontweight="bold", fontsize=11)
        series = {"observed": (sub["obs_speed"].to_numpy(float), sub["obs_direction"].to_numpy(float))}
        series.update({experiment: (sub[f"{experiment}_speed"].to_numpy(float), sub[f"{experiment}_direction"].to_numpy(float)) for experiment in EXPERIMENTS})
        for name, (speed, direction) in series.items():
            style = STYLES[name]
            kwargs = {key: style[key] for key in ("color", "linestyle", "linewidth")}
            speed_ax.plot(x, speed, label=style["label"], **kwargs)
            direction_ax.plot(x, break_direction_wrap(direction), label=style["label"], **kwargs)
        if row_index == 1:
            speed_ax.set_xlabel("预报时效")
            direction_ax.set_xlabel("预报时效")
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=5, frameon=True)
    fig.suptitle(
        f"2025年2月5日寒潮海洋站10 m风场对比（{timezone_label}）\n"
        "E0、G0：2025-02-05 00 UTC；E120、G120：2025-01-31 00 UTC起报",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.09, top=0.80, hspace=0.38, wspace=0.18)
    stem = figures_root / f"cold_surge_2025_station_wind_timeseries_{timezone_mode}"
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
    parser.add_argument("--match-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--matched-csv", type=Path, default=None)
    parser.add_argument("--timezone-mode", choices=("utc", "beijing"), default="utc")
    args = parser.parse_args(argv)
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"], "font.size": 10, "axes.unicode_minus": False})
    if args.plot_only:
        if args.matched_csv is None:
            raise ValueError("--plot-only requires --matched-csv")
        output = plot_figure(pd.read_csv(args.matched_csv), args.timezone_mode, args.figures_root)
        print(f"Figure: {output}")
        return 0

    station_data = load_station_data(args.station_file)
    args.data_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "station_file": str(args.station_file),
        "target_time_utc": TARGET_TIME.strftime("%Y-%m-%d %H:%M"),
        "end_time_utc": END_TIME.strftime("%Y-%m-%d %H:%M"),
        "direct_initial_time_utc": TARGET_TIME.strftime("%Y-%m-%d %H:%M"),
        "lagged_initial_time_utc": (TARGET_TIME - timedelta(hours=120)).strftime("%Y-%m-%d %H:%M"),
        "modes": {},
    }
    for mode in ("utc", "beijing"):
        frame, missing = match_data(station_data, args.output_root, mode)
        csv_path = args.data_root / f"cold_surge_2025_station_wind_timeseries_{mode}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary["modes"][mode] = {"csv": str(csv_path), "rows": int(len(frame)), "missing_outputs": missing, "match_offsets": sorted(frame["match_offset_hours"].unique().tolist())}
        if not args.match_only:
            plot_figure(frame, mode, args.figures_root)
    summary_path = args.data_root / "cold_surge_2025_station_wind_timeseries_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    for mode, info in summary["modes"].items():
        print(f"{mode}: rows={info['rows']}; missing outputs={len(info['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
