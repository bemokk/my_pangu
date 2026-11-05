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
try:
    from .publication_style import FONT_FAMILY, format_figure, set_style as set_publication_style
except ImportError:
    from publication_style import FONT_FAMILY, format_figure, set_style as set_publication_style
set_publication_style()

from matplotlib.ticker import MaxNLocator
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


def _continuous_directions(series: list[np.ndarray]) -> list[np.ndarray]:
    """Unwrap finite runs, then align their 360-degree branches across curves.

    This changes only the displayed equivalent angle. Missing observations stay
    missing, and no real point is removed to break a 0/360-degree crossing.
    """
    unwrapped = []
    for values in series:
        result = np.asarray(values, dtype=float).copy()
        valid = np.flatnonzero(np.isfinite(result))
        for run in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
            if len(run):
                result[run] = np.rad2deg(np.unwrap(np.deg2rad(result[run])))
        unwrapped.append(result)
    reference = next((a[np.isfinite(a)] for a in unwrapped if np.isfinite(a).any()), None)
    if reference is None:
        return unwrapped
    center = (reference.min() + reference.max()) / 2
    for values in unwrapped:
        valid = np.flatnonzero(np.isfinite(values))
        for run in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
            if len(run):
                midpoint = (values[run].min() + values[run].max()) / 2
                values[run] += 360 * np.floor((center - midpoint) / 360 + 0.5)
    finite = np.concatenate([a[np.isfinite(a)] for a in unwrapped])
    if finite.max() - finite.min() > 360:
        # Multiple rotations cannot fit continuously into one displayed cycle.
        # Choose a common circular cut with the fewest discontinuities.
        candidates = []
        for lower in range(-180, 180, 30):
            mapped = [lower + np.mod(np.asarray(a, dtype=float) - lower, 360) for a in series]
            crossings = sum(np.count_nonzero(np.abs(np.diff(a)) > 180) for a in mapped)
            candidates.append(((crossings, abs(lower), lower), mapped))
        unwrapped = min(candidates, key=lambda item: item[0])[1]
    return unwrapped


def _direction_line(hours: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert a separator at a circular cut without removing either sample."""
    crossings = np.flatnonzero(np.abs(np.diff(angles)) > 180) + 1
    return np.insert(hours, crossings, np.nan), np.insert(angles, crossings, np.nan)


def _data_limits(series: list[np.ndarray], nonnegative=False) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=float) for a in series])
    values = values[np.isfinite(values)]
    if not len(values):
        return (0.0, 1.0) if nonnegative else (0.0, 360.0)
    low, high = float(values.min()), float(values.max())
    padding = 0.05 * max(high - low, 1.0)
    return max(0.0, low - padding) if nonnegative else low - padding, high + padding


def _direction_limits(series: list[np.ndarray]) -> tuple[float, float]:
    """Translate a fixed 360-degree window to contain all displayed angles."""
    values = np.concatenate(series)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 360.0
    low, high = float(values.min()), float(values.max())
    if high - low > 360 + 1e-9:
        raise ValueError("Continuous wind directions exceed a 360-degree display window")
    centered_lower = (low + high) / 2 - 180
    lower = float(np.clip(30 * np.round(centered_lower / 30), high - 360, low))
    return lower, lower + 360


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
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel("风向 (°，展开值)")
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
                title=None, *, fig=None, axes=None, panel_offset=0,
                show_initialization=True) -> Path:
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
    embedded = fig is not None
    if embedded:
        if axes is None or np.shape(axes) != (len(groups), 2):
            raise ValueError("Embedded time series require one pair of axes per case")
    else:
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
            f"({chr(97 + panel_offset + row_index * 2)}) {case['typhoon']} · {case['station']}站 "
            f"({case['lon']:.1f}°E, {case['lat']:.1f}°N) 风速"
        )
        if initialization and show_initialization:
            speed_title += "\n" + initialization
        speed_ax.set_title(
            speed_title,
            loc="left",
            fontweight="bold",
            fontsize=10 if initialization else 11,
        )
        direction_ax.set_title(
            f"({chr(98 + panel_offset + row_index * 2)}) {case['typhoon']} · {case['station']}站 风向",
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
        direction_values = _continuous_directions([values[1] for values in series.values()])
        if speed_max is None:
            speed_ax.set_ylim(*_data_limits([values[0] for values in series.values()], nonnegative=True))
        speed_ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        direction_lower, direction_upper = _direction_limits(direction_values)
        direction_ax.set_ylim(direction_lower, direction_upper)
        direction_ax.set_yticks(np.linspace(direction_lower, direction_upper, 7))
        for (name, (speed, _)), direction in zip(series.items(), direction_values):
            style = STYLES[name]
            speed_ax.plot(lead_hours, speed, label=style["label"], **{k: style[k] for k in ("color", "linestyle", "linewidth")})
            direction_hours, direction_line = _direction_line(lead_hours, direction)
            direction_ax.plot(
                direction_hours,
                direction_line,
                marker="." if len(direction_line) > len(direction) else None,
                markersize=2,
                label=style["label"],
                **{k: style[k] for k in ("color", "linestyle", "linewidth")},
            )
        if row_index == len(groups) - 1:
            speed_ax.set_xlabel("窗口经过时间（h）")
            direction_ax.set_xlabel("窗口经过时间（h）")

    if embedded:
        axes[0, 0].legend(loc="upper left", ncol=2, fontsize=8, frameon=True,
                          facecolor="white", edgecolor="black", framealpha=1, fancybox=False)
        return fig

    # Avoid bbox/tight-layout font measurement crashes on some Windows
    # Matplotlib builds with mixed Chinese and fallback fonts.
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.7875 / figure_height,
                        top=1 - 0.60 / figure_height, hspace=0.42, wspace=0.18)
    stem = figures_root / f"{output_stem}_{timezone_mode}"
    output = stem.with_suffix(".png")
    format_figure(fig)
    fig.savefig(output, dpi=600)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"), dpi=600)
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
        "font.family": FONT_FAMILY,
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
