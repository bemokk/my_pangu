"""Compare station winds for Typhoon Yangliu and Typhoon Francisco in 2025."""
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
EXPERIMENTS = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")
STYLES = {
    "observed": {"label": "观测", "color": "#333333", "linestyle": "-", "linewidth": 1.6},
    "era5_direct": {"label": "E0", "color": "#43A3EF", "linestyle": "-", "linewidth": 1.25},
    "gdas_direct": {"label": "G0", "color": "#FEA040", "linestyle": "-", "linewidth": 1.25},
    "era5_lagged_120h": {"label": "E120", "color": "#009E73", "linestyle": "--", "linewidth": 1.25},
    "gdas_lagged_120h": {"label": "G120", "color": "#EF767B", "linestyle": "--", "linewidth": 1.25},
}
CASES = (
    {"typhoon": "杨柳", "station": "东山", "lon": 117.5, "lat": 23.8, "target": datetime(2025, 8, 12)},
    {"typhoon": "杨柳", "station": "北礵", "lon": 120.3, "lat": 26.7, "target": datetime(2025, 8, 12)},
    {"typhoon": "范斯高", "station": "南麂", "lon": 121.1, "lat": 27.5, "target": datetime(2025, 7, 24)},
    {"typhoon": "范斯高", "station": "北礵", "lon": 120.3, "lat": 26.7, "target": datetime(2025, 7, 24)},
    {"typhoon": "范斯高", "station": "大陈", "lon": 121.9, "lat": 28.5, "target": datetime(2025, 7, 24)},
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
    def find(token: str) -> str:
        return next(column for column in frame.columns if token in column)
    frame = frame.rename(columns={
        find("时间"): "station_time_raw", find("站点"): "station",
        find("经度"): "longitude", find("纬度"): "latitude",
        find("风速"): "obs_speed", find("风向"): "obs_direction",
    })
    frame["station_time_raw"] = pd.to_datetime(frame["station_time_raw"], errors="coerce")
    for column in ("longitude", "latitude", "obs_speed", "obs_direction"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["station"] = frame["station"].astype(str).str.strip()
    return frame.dropna(subset=["station_time_raw", "station", "obs_speed", "obs_direction"]).copy()


def nearest_model_time(valid_time: datetime) -> tuple[datetime, float]:
    day = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (valid_time - day).total_seconds() / 3600.0
    lower = day + timedelta(hours=int(np.floor(elapsed / 3.0) * 3))
    upper = lower + timedelta(hours=3)
    selected = min((lower, upper), key=lambda item: abs((item - valid_time).total_seconds()))
    return selected, (selected - valid_time).total_seconds() / 3600.0


def match_data(station_data: pd.DataFrame, output_root: Path, timezone_mode: str) -> tuple[pd.DataFrame, list[dict]]:
    from experiment_2025.Statistics.statistics import GridSampler, model_path, wind_direction

    first_path = model_path(output_root, "era5_direct", CASES[0]["target"], 0)
    if not first_path.is_file():
        raise FileNotFoundError(first_path)
    sampler = GridSampler(first_path, [42.0, 103.0, 13.0, 130.0])
    rows, missing = [], []
    prediction_cache: dict[tuple[str, datetime, int, float, float], tuple[float, float]] = {}
    for case in CASES:
        target = case["target"]
        end = target + timedelta(hours=120)
        selected = station_data[station_data["station"] == case["station"]].copy()
        selected["station_time_utc_tmp"] = selected["station_time_raw"] - (pd.Timedelta(hours=8) if timezone_mode == "beijing" else pd.Timedelta(0))
        selected = selected[(selected["station_time_utc_tmp"] >= pd.Timestamp(target)) & (selected["station_time_utc_tmp"] <= pd.Timestamp(end))]
        if selected.empty:
            raise ValueError(f"No records for {case['station']} / {case['typhoon']} / {timezone_mode}")
        for record in selected.sort_values("station_time_raw").itertuples(index=False):
            raw = record.station_time_raw.to_pydatetime()
            valid_utc = raw - timedelta(hours=8) if timezone_mode == "beijing" else raw
            model_valid, offset = nearest_model_time(valid_utc)
            lead = int(round((model_valid - target).total_seconds() / 3600.0))
            if lead < 0 or lead > 120 or lead % 3:
                raise ValueError(f"Invalid model lead {lead} for {valid_utc}")
            row = {
                "typhoon": case["typhoon"], "station": case["station"], "longitude": case["lon"], "latitude": case["lat"],
                "target_time_utc": target.strftime("%Y-%m-%d %H:%M"), "station_time_raw": raw.strftime("%Y-%m-%d %H:%M"),
                "station_time_utc": valid_utc.strftime("%Y-%m-%d %H:%M"), "model_valid_time_utc": model_valid.strftime("%Y-%m-%d %H:%M"),
                "plot_lead_hour": (valid_utc - target).total_seconds() / 3600.0, "model_lead_hour": lead, "match_offset_hours": float(offset),
                "obs_speed": float(record.obs_speed), "obs_direction": float(record.obs_direction),
            }
            for experiment in EXPERIMENTS:
                path = model_path(output_root, experiment, target, lead)
                if not path.is_file():
                    missing.append({"experiment": experiment, "typhoon": case["typhoon"], "station": case["station"], "path": str(path)})
                    row[f"{experiment}_speed"] = np.nan
                    row[f"{experiment}_direction"] = np.nan
                    continue
                cache_key = (experiment, target, lead, case["lat"], case["lon"])
                if cache_key not in prediction_cache:
                    u, v = sampler.sample(path, np.asarray([case["lat"]]), np.asarray([case["lon"]]))
                    prediction_cache[cache_key] = (float(u[0]), float(v[0]))
                u, v = prediction_cache[cache_key]
                row[f"{experiment}_speed"] = float(np.hypot(u, v))
                row[f"{experiment}_direction"] = float(wind_direction(np.asarray([u]), np.asarray([v]))[0])
            rows.append(row)
    return pd.DataFrame(rows), missing


def break_wrap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[1:][np.abs(np.diff(values)) > 180] = np.nan
    return values


def plot_figure(frame: pd.DataFrame, timezone_mode: str, figures_root: Path) -> Path:
    figures_root.mkdir(parents=True, exist_ok=True)
    label = "UTC假设" if timezone_mode == "utc" else "北京时间假设（转换为UTC匹配）"
    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True, constrained_layout=False)
    ticks = np.arange(0, 121, 12, dtype=float)
    for i, case in enumerate(CASES):
        sub = frame[(frame["typhoon"] == case["typhoon"]) & (frame["station"] == case["station"])].sort_values("plot_lead_hour")
        x = sub["plot_lead_hour"].to_numpy(float)
        start_label = case["target"].strftime("%Y-%m-%d 00 UTC")
        tick_labels = [f"0\n{start_label}"] + [f"{int(value)}h" for value in ticks[1:]]
        speed_ax, direction_ax = axes[i]
        for ax, is_direction in ((speed_ax, False), (direction_ax, True)):
            ax.set_xlim(-2.5, 120)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.grid(True, color="#BFBFBF", linewidth=0.6, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            if is_direction:
                ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360]); ax.set_ylabel("风向 (°)")
            else:
                ax.set_ylabel("风速 (m s$^{-1}$)")
            for spine in ax.spines.values():
                spine.set_color("#444444"); spine.set_linewidth(0.8)
        row_letter = chr(97 + 2 * i)
        speed_ax.set_title(f"({row_letter}) {case['typhoon']} · {case['station']} 风速", loc="left", fontweight="bold", fontsize=10)
        direction_ax.set_title(f"({chr(ord(row_letter) + 1)}) {case['typhoon']} · {case['station']} 风向", loc="left", fontweight="bold", fontsize=10)
        series = {"observed": (sub["obs_speed"].to_numpy(float), sub["obs_direction"].to_numpy(float))}
        series.update({e: (sub[f"{e}_speed"].to_numpy(float), sub[f"{e}_direction"].to_numpy(float)) for e in EXPERIMENTS})
        for name, (speed, direction) in series.items():
            style = STYLES[name]; kwargs = {k: style[k] for k in ("color", "linestyle", "linewidth")}
            speed_ax.plot(x, speed, label=style["label"], **kwargs)
            direction_ax.plot(x, break_wrap(direction), label=style["label"], **kwargs)
        if i == len(CASES) - 1:
            speed_ax.set_xlabel("预报时效"); direction_ax.set_xlabel("预报时效")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, frameon=True)
    fig.suptitle(f"2025年杨柳、范斯高海洋站10 m风场对比（{label}）\nE0、G0为实线；E120、G120为虚线", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.055, top=0.89, hspace=0.48, wspace=0.17)
    stem = figures_root / f"two_typhoons_2025_station_wind_timeseries_{timezone_mode}"
    output = stem.with_suffix(".png")
    fig.savefig(output, dpi=300); fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".pdf"), dpi=300)
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
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"], "font.size": 9, "axes.unicode_minus": False})
    if args.plot_only:
        if args.matched_csv is None:
            raise ValueError("--plot-only requires --matched-csv")
        print(f"Figure: {plot_figure(pd.read_csv(args.matched_csv), args.timezone_mode, args.figures_root)}")
        return 0
    station_data = load_station_data(args.station_file)
    args.data_root.mkdir(parents=True, exist_ok=True)
    summary = {"station_file": str(args.station_file), "modes": {}, "cases": [{"typhoon": c["typhoon"], "station": c["station"], "target_time_utc": c["target"].strftime("%Y-%m-%d %H:%M"), "lagged_initial_time_utc": (c["target"] - timedelta(hours=120)).strftime("%Y-%m-%d %H:%M")} for c in CASES]}
    for mode in ("utc", "beijing"):
        frame, missing = match_data(station_data, args.output_root, mode)
        csv_path = args.data_root / f"two_typhoons_2025_station_wind_timeseries_{mode}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        rows_by_case = {f"{typhoon}_{station}": int(count) for (typhoon, station), count in frame.groupby(["typhoon", "station"]).size().items()}
        summary["modes"][mode] = {"csv": str(csv_path), "rows": int(len(frame)), "rows_by_case": rows_by_case, "missing_outputs": missing, "match_offsets": sorted(frame["match_offset_hours"].unique().tolist())}
        if not args.match_only:
            plot_figure(frame, mode, args.figures_root)
    summary_path = args.data_root / "two_typhoons_2025_station_wind_timeseries_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Summary: {summary_path}")
    for mode, info in summary["modes"].items():
        print(f"{mode}: rows={info['rows']}; missing outputs={len(info['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
