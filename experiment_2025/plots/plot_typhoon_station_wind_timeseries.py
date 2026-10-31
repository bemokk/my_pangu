"""Plot typhoon station winds from a previously matched CSV.

Raw station parsing, timezone conversion, NetCDF sampling and result summaries
are handled by Statistics/typhoon_station_wind_timeseries.py. This program only
reads matched CSV files and writes PNG/SVG/PDF figures. Cases and window start
times are taken from the CSV, including multiple typhoons at the same station.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLOTS_ROOT.parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "experiment_2025" / "Statistics" / "data"
DEFAULT_FIGURES_ROOT = PLOTS_ROOT / "figures"

EXPERIMENTS = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")

STYLES = {
    "observed": {"label": "观测", "color": "#333333", "linestyle": "-", "linewidth": 1.8},
    "era5_direct": {"label": "E0", "color": "#43A3EF", "linestyle": "-", "linewidth": 1.45},
    "gdas_direct": {"label": "G0", "color": "#FEA040", "linestyle": "-", "linewidth": 1.45},
    "era5_lagged_120h": {"label": "E120", "color": "#009E73", "linestyle": "--", "linewidth": 1.45},
    "gdas_lagged_120h": {"label": "G120", "color": "#EF767B", "linestyle": "--", "linewidth": 1.45},
}


def validate_matched_data(frame: pd.DataFrame) -> None:
    required = {
        "typhoon", "station", "longitude", "latitude", "station_time_raw",
        "station_time_utc", "plot_lead_hour", "obs_speed", "obs_direction",
        *(f"{experiment}_{variable}" for experiment in EXPERIMENTS for variable in ("speed", "direction")),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Matched CSV is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Matched CSV contains no station records")
    for column in ("typhoon", "station", "longitude", "latitude", "plot_lead_hour"):
        if frame[column].isna().any():
            raise ValueError(f"Matched CSV contains missing values in {column}")
    if not np.isfinite(frame["plot_lead_hour"].to_numpy(float)).all():
        raise ValueError("Matched CSV contains nonfinite elapsed hours")


def infer_timezone_mode(frame: pd.DataFrame) -> str:
    """Read the recorded time offset without re-matching any observations."""
    raw = pd.to_datetime(frame["station_time_raw"], errors="raise")
    utc = pd.to_datetime(frame["station_time_utc"], errors="raise")
    offsets = (raw - utc).dt.total_seconds().to_numpy() / 3600.0
    if np.all(np.isclose(offsets, 8.0)):
        return "beijing"
    if np.all(np.isclose(offsets, 0.0)):
        return "utc"
    raise ValueError("Matched CSV must use one consistent UTC or Beijing time interpretation")


def _break_direction_wrap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    jumps = np.abs(np.diff(values)) > 180.0
    values[1:][jumps] = np.nan
    return values


def _initialization_label(frame: pd.DataFrame) -> str:
    if "target_time_utc" not in frame:
        return ""
    targets = pd.to_datetime(frame["target_time_utc"], errors="raise")
    if "expected_target_time_utc" in frame:
        expected = pd.to_datetime(frame["expected_target_time_utc"], errors="raise")
        if expected.nunique() != 1:
            raise ValueError("One fixed target time is required for each selected case")
        if not targets.eq(expected).all():
            return f"【预览】{expected.iloc[0]:%m-%d %H:%M} UTC固定起报数据待更新"
    if targets.nunique() == 1:
        return f"固定起报：{targets.iloc[0]:%Y-%m-%d %H:%M} UTC"
    return "现有CSV包含多次起报"


def _configure_axes(ax, direction: bool, lead_hours: np.ndarray, start_date_utc: datetime) -> None:
    ax.grid(True, color="#BFBFBF", linewidth=0.65, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    # Leave a small margin before the 0 h tick so the two-line first label is
    # fully visible instead of being clipped at the left axes border.
    ax.set_xlim(-2.5, float(np.nanmax(lead_hours)))
    max_lead = int(np.ceil(np.nanmax(lead_hours) / 12.0) * 12)
    ticks = np.arange(0, max_lead + 1, 12, dtype=float)
    labels = [f"0\n{start_date_utc:%Y-%m-%d %H:%M} UTC"] + [f"{int(tick)}h" for tick in ticks[1:]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8, pad=2)
    if direction:
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_ylabel("风向 (°)")
    else:
        # Keep Chinese outside Matplotlib's math-text font selection.
        ax.set_ylabel("风速 (m·s⁻¹)")
    for label in ax.get_xticklabels():
        label.set_rotation(0)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.8)


def plot_figure(frame: pd.DataFrame, timezone_mode: str, figures_root: Path,
                output_stem="typhoon_station_wind_timeseries", speed_max=None,
                title=None) -> Path:
    validate_matched_data(frame)
    actual_mode = infer_timezone_mode(frame)
    if timezone_mode != actual_mode:
        raise ValueError(f"Requested timezone mode {timezone_mode!r} differs from CSV {actual_mode!r}")
    groups = list(frame.groupby(["typhoon", "station"], sort=False))
    prepared = []
    for (typhoon, station), sub in groups:
        sub = sub.sort_values("plot_lead_hour", kind="stable").copy()
        starts = pd.to_datetime(sub["station_time_utc"], errors="raise") - pd.to_timedelta(sub["plot_lead_hour"], unit="h")
        if starts.isna().any() or starts.nunique() != 1:
            raise ValueError(f"Inconsistent window start in matched CSV for {typhoon}, {station}")
        prepared.append((typhoon, station, sub, starts.iloc[0].to_pydatetime()))
    prepared.sort(key=lambda item: (item[3], item[0], item[1]))
    figures_root.mkdir(parents=True, exist_ok=True)
    timezone_label = "观测原始时间为UTC" if timezone_mode == "utc" else "观测原始时间为北京时间，已转换为UTC"
    figure_height = 3.5 * len(groups)
    fig, axes = plt.subplots(len(groups), 2, figsize=(13.0, figure_height),
                             sharex=False, squeeze=False, constrained_layout=False)

    for row_index, (typhoon, station, sub, start_date_utc) in enumerate(prepared):
        lead_hours = sub["plot_lead_hour"].to_numpy(float)
        case = {"typhoon": typhoon, "station": station, "lon": sub["longitude"].iloc[0], "lat": sub["latitude"].iloc[0]}
        speed_ax, direction_ax = axes[row_index]
        for ax, direction in ((speed_ax, False), (direction_ax, True)):
            _configure_axes(ax, direction, lead_hours, start_date_utc)
        if speed_max is not None:
            if not np.isfinite(speed_max) or speed_max <= 0:
                raise ValueError("speed_max must be a positive finite number")
            speed_ax.set_ylim(0, speed_max)

        initialization = _initialization_label(sub)
        speed_title = (
            f"({chr(97 + row_index * 2)}) {case['typhoon']} · {case['station']}站 "
            f"({case['lon']:.1f}°E, {case['lat']:.1f}°N) 风速"
        )
        if initialization:
            speed_title += "\n" + initialization
        speed_ax.set_title(
            speed_title,
            loc="left",
            fontweight="bold",
            fontsize=10 if initialization else 11,
        )
        direction_ax.set_title(
            f"({chr(98 + row_index * 2)}) {case['typhoon']} · {case['station']}站 风向",
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
        if row_index == len(groups) - 1:
            speed_ax.set_xlabel("窗口经过时间（h）")
            direction_ax.set_xlabel("窗口经过时间（h）")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.5775 / figure_height), ncol=5, frameon=True)
    fig.suptitle(
        (title or f"2025年台风海洋站10 m风场对比（{timezone_label}）") + "\n"
        "E0、G0为实线；E120、G120为虚线",
        fontsize=14,
        y=1 - 0.0525 / figure_height,
    )
    # Avoid bbox/tight-layout font measurement crashes on some Windows
    # Matplotlib builds with mixed Chinese and fallback fonts.
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.7875 / figure_height,
                        top=1 - 1.26 / figure_height, hspace=0.42, wspace=0.18)
    stem = figures_root / f"{output_stem}_{timezone_mode}"
    output = stem.with_suffix(".png")
    fig.savefig(output, dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--matched-csv", type=Path, default=None)
    parser.add_argument("--output-stem", default="typhoon_station_wind_timeseries")
    parser.add_argument("--speed-max", type=float, default=None,
                        help="optional common upper wind-speed limit in m/s")
    parser.add_argument("--title", default=None)
    parser.add_argument("--timezone-mode", choices=("utc", "beijing", "both"), default=None,
                        help="select matched CSVs; defaults to Beijing, or is inferred for --matched-csv")
    parser.add_argument("--plot-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.matched_csv is not None and args.timezone_mode == "both":
        parser.error("--matched-csv selects one CSV; choose utc or beijing, or omit --timezone-mode")

    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun", "Microsoft YaHei", "DejaVu Serif"],
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    if args.matched_csv is not None:
        inputs = [(args.matched_csv, args.timezone_mode)]
    else:
        mode = args.timezone_mode or "beijing"
        modes = ("utc", "beijing") if mode == "both" else (mode,)
        inputs = [(args.data_root / f"typhoon_station_wind_timeseries_2025_{mode}.csv", mode) for mode in modes]
    for csv_path, mode in inputs:
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Matched CSV not found: {csv_path}. "
                "Run experiment_2025/Statistics/typhoon_station_wind_timeseries.py first."
            )
        frame = pd.read_csv(csv_path)
        validate_matched_data(frame)
        output = plot_figure(frame, mode or infer_timezone_mode(frame), args.figures_root,
                             args.output_stem, args.speed_max, args.title)
        print(f"Figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
