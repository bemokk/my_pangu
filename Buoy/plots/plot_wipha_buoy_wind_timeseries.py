from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from plots.wipha_case_common import (
    MATCHED_CSV,
    OUT_ANALYSIS_SAMPLES,
    DATASET_COLORS,
    DATASET_LABELS,
    DATASETS,
    OUT_TIMESERIES_PNG,
    OUT_TIMESERIES_SVG,
    SELECTED_PLATFORMS,
    WINDOW_END,
    WINDOW_START,
    angular_difference_deg,
    circular_mean_deg,
    ensure_dirs,
    set_plot_style,
)

TIMESERIES_FORECAST_INIT_TIMES = {
    "gdas_forecast": pd.Timestamp("2025-07-18 00:00:00"),
    "era5_lagged_5d": pd.Timestamp("2025-07-13 00:00:00"),
}


def _parse_forecast_start_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y-%m-%d-%H-%M", errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(values[parsed.isna()], errors="coerce")
        parsed.loc[parsed.isna()] = fallback
    return parsed


def select_fixed_init_timeseries_samples(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()

    selected = raw.copy()
    selected["datetime_utc"] = pd.to_datetime(selected["datetime_utc"], errors="coerce")
    selected["pred_start_time"] = _parse_forecast_start_time(selected["pred_start_time"])
    expected_start = selected["dataset"].map(TIMESERIES_FORECAST_INIT_TIMES)
    three_hour_time = (
        selected["datetime_utc"].dt.minute.eq(0)
        & selected["datetime_utc"].dt.second.eq(0)
        & selected["datetime_utc"].dt.hour.mod(3).eq(0)
    )
    selected = selected[
        selected["pred_start_time"].eq(expected_start)
        & three_hour_time
    ].copy()
    selected["_dataset_order"] = selected["dataset"].map({dataset: index for index, dataset in enumerate(DATASETS)})
    selected = selected.sort_values(["_dataset_order", "platform_id", "datetime_utc"]).drop(columns="_dataset_order")
    return selected.reset_index(drop=True)


def _circular_group_mean(series: pd.Series) -> float:
    return circular_mean_deg(series.to_numpy())


def prepare_fixed_init_timeseries_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = {
        "dataset",
        "datetime_utc",
        "pred_start_time",
        "platform_id",
        "latitude",
        "longitude",
        "obs_speed_ms",
        "obs_dir_deg",
        "lead_hour",
        "pred_speed_ms",
        "pred_dir_deg",
        "pred_u10_ms",
        "pred_v10_ms",
    }
    raw = pd.read_csv(MATCHED_CSV, usecols=lambda c: c in cols)
    raw = raw[
        raw["platform_id"].isin(SELECTED_PLATFORMS)
        & raw["dataset"].isin(DATASETS)
    ].copy()
    raw = select_fixed_init_timeseries_samples(raw)
    raw = raw[raw["datetime_utc"].between(WINDOW_START, WINDOW_END)].copy()
    if raw.empty:
        raise RuntimeError("No fixed-initialization Wipha wind samples found for the selected platforms and window.")

    obs = raw.drop_duplicates(["platform_id", "datetime_utc", "latitude", "longitude", "obs_speed_ms", "obs_dir_deg"]).groupby(["platform_id", "datetime_utc"], as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        obs_speed_ms=("obs_speed_ms", "mean"),
        obs_dir_deg=("obs_dir_deg", _circular_group_mean),
    )
    merged = raw.groupby(["platform_id", "datetime_utc", "dataset", "pred_start_time"], as_index=False).agg(
        lead_hour=("lead_hour", "min"),
        pred_speed_ms=("pred_speed_ms", "mean"),
        pred_dir_deg=("pred_dir_deg", _circular_group_mean),
        pred_u10_ms=("pred_u10_ms", "mean"),
        pred_v10_ms=("pred_v10_ms", "mean"),
    ).merge(obs, on=["platform_id", "datetime_utc"], how="left")
    merged["speed_error_ms"] = merged["pred_speed_ms"] - merged["obs_speed_ms"]
    merged["direction_error_deg"] = [angular_difference_deg(p, o) for p, o in zip(merged["pred_dir_deg"], merged["obs_dir_deg"])]
    merged["direction_abs_error_deg"] = merged["direction_error_deg"].abs()
    merged.to_csv(OUT_ANALYSIS_SAMPLES, index=False, encoding="utf-8-sig")
    return obs, merged


def timeseries_time_axis_bounds(obs: pd.DataFrame, merged: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    times = pd.concat(
        [
            pd.to_datetime(obs.get("datetime_utc", pd.Series(dtype="datetime64[ns]")), errors="coerce"),
            pd.to_datetime(merged.get("datetime_utc", pd.Series(dtype="datetime64[ns]")), errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if times.empty:
        raise ValueError("Cannot determine time-series x-axis limits from empty data.")
    return pd.Timestamp(times.min()), pd.Timestamp(times.max())


def plot_timeseries(obs: pd.DataFrame, merged: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8), sharex=True, constrained_layout=False)
    x_min, x_max = timeseries_time_axis_bounds(obs, merged)
    for col, platform_id in enumerate(SELECTED_PLATFORMS):
        obs_sub = obs[obs["platform_id"] == platform_id].sort_values("datetime_utc")
        ax_speed, ax_dir = axes[0, col], axes[1, col]
        for ax in (ax_speed, ax_dir):
            ax.set_facecolor("#F4F5F7")
            ax.grid(True, color="white", linewidth=1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))

        ax_speed.plot(obs_sub["datetime_utc"], obs_sub["obs_speed_ms"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.8, markersize=3.8, label="Buoy observation")
        ax_dir.plot(obs_sub["datetime_utc"], obs_sub["obs_dir_deg"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.4, markersize=3.5, label="Buoy observation")

        for dataset in DATASETS:
            sub = merged[(merged["platform_id"] == platform_id) & (merged["dataset"] == dataset)].sort_values("datetime_utc")
            init_label = TIMESERIES_FORECAST_INIT_TIMES[dataset].strftime("%m-%d %H UTC")
            dataset_label = f"{DATASET_LABELS[dataset]} ({init_label} init)"
            ax_speed.plot(sub["datetime_utc"], sub["pred_speed_ms"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.5, markersize=3.2, label=dataset_label)
            ax_dir.plot(sub["datetime_utc"], sub["pred_dir_deg"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.2, markersize=3.0, label=dataset_label)

        ax_speed.set_title(f"({chr(ord('a') + col)}) {platform_id} wind speed", loc="left", fontweight="bold")
        ax_dir.set_title(f"({chr(ord('c') + col)}) {platform_id} wind direction", loc="left", fontweight="bold")
        ax_speed.set_ylabel("Wind speed (m s$^{-1}$)")
        ax_dir.set_ylabel("Wind direction (°)")
        ax_dir.set_ylim(0, 360)
        ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H UTC"))
        ax_dir.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    fig.suptitle("Typhoon Wipha Case: Platform Wind Speed and Direction Time Series", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.955])
    fig.savefig(OUT_TIMESERIES_PNG, bbox_inches="tight")
    fig.savefig(OUT_TIMESERIES_SVG, bbox_inches="tight")
    plt.close(fig)


def generate() -> list[Path]:
    ensure_dirs()
    obs, merged = prepare_fixed_init_timeseries_data()
    plot_timeseries(obs, merged)
    return [OUT_TIMESERIES_PNG, OUT_TIMESERIES_SVG]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
