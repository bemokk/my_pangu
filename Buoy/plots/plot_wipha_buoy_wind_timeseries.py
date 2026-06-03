from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

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
    WINDOW_END,
    WINDOW_START,
    angular_difference_deg,
    ensure_dirs,
    haversine_km,
    set_plot_style,
)
from plots.plot_wipha_track_buoy_locations import VIRTUAL_POINT_STATIONS

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


def _station_definition_lookup() -> dict[str, dict]:
    return {station["station_id"]: station for station in VIRTUAL_POINT_STATIONS}


def _attach_virtual_station_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    candidate_frames = []
    for station in VIRTUAL_POINT_STATIONS:
        sub = raw.copy()
        sub["station_id"] = station["station_id"]
        sub["station_label"] = station["label"]
        sub["station_lon"] = station["lon"]
        sub["station_lat"] = station["lat"]
        sub["station_radius_km"] = station["radius_km"]
        sub["distance_to_station_km"] = [
            haversine_km(station["lon"], station["lat"], lon, lat)
            for lon, lat in zip(sub["longitude"], sub["latitude"])
        ]
        sub = sub[sub["distance_to_station_km"].le(station["radius_km"])].copy()
        if not sub.empty:
            candidate_frames.append(sub)
    if not candidate_frames:
        return pd.DataFrame(columns=list(raw.columns) + [
            "station_id",
            "station_label",
            "station_lon",
            "station_lat",
            "station_radius_km",
            "distance_to_station_km",
        ])
    return pd.concat(candidate_frames, ignore_index=True)


def select_one_record_per_station_lead(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    keys = [
        "station_id",
        "lead_hour",
        "datetime_utc",
        "record_id",
        "platform_id",
        "latitude",
        "longitude",
        "obs_speed_ms",
        "obs_dir_deg",
        "distance_to_station_km",
    ]
    available = candidates.groupby(keys + ["dataset"]).size().unstack(fill_value=0).reset_index()
    common_records = available[
        (available.get("gdas_forecast", 0) > 0)
        & (available.get("era5_lagged_5d", 0) > 0)
    ].copy()
    if common_records.empty:
        return candidates.iloc[0:0].copy()

    chosen = (
        common_records
        .sort_values(["station_id", "lead_hour", "distance_to_station_km", "record_id", "platform_id"])
        .drop_duplicates(["station_id", "lead_hour"], keep="first")
    )
    chosen_keys = ["station_id", "lead_hour", "datetime_utc", "record_id", "platform_id"]
    selected = candidates.merge(chosen[chosen_keys], on=chosen_keys, how="inner")
    selected = selected.sort_values(["station_id", "lead_hour", "dataset", "distance_to_station_km"])
    selected = selected.drop_duplicates(["station_id", "lead_hour", "dataset"], keep="first")
    return selected.reset_index(drop=True)


def prepare_fixed_init_timeseries_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = {
        "record_id",
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
        raw["dataset"].isin(DATASETS)
    ].copy()
    raw = select_fixed_init_timeseries_samples(raw)
    raw = raw[raw["datetime_utc"].between(WINDOW_START, WINDOW_END)].copy()
    raw = raw.dropna(subset=["obs_speed_ms", "obs_dir_deg", "pred_speed_ms", "pred_dir_deg"])
    raw = _attach_virtual_station_candidates(raw)
    raw = select_one_record_per_station_lead(raw)
    if raw.empty:
        raise RuntimeError("No fixed-initialization Wipha wind samples found for the selected virtual stations and window.")

    raw["source_platform_id"] = raw["platform_id"]
    raw["platform_id"] = raw["station_id"]
    raw["platform_label"] = raw["station_label"]
    obs = raw.drop_duplicates(["platform_id", "lead_hour", "datetime_utc", "record_id"]).loc[
        :,
        [
            "platform_id",
            "platform_label",
            "lead_hour",
            "datetime_utc",
            "record_id",
            "source_platform_id",
            "latitude",
            "longitude",
            "obs_speed_ms",
            "obs_dir_deg",
            "distance_to_station_km",
        ],
    ].sort_values(["platform_id", "lead_hour"]).reset_index(drop=True)
    merged = raw.loc[
        :,
        [
            "platform_id",
            "platform_label",
            "lead_hour",
            "datetime_utc",
            "dataset",
            "pred_start_time",
            "record_id",
            "source_platform_id",
            "pred_speed_ms",
            "pred_dir_deg",
            "pred_u10_ms",
            "pred_v10_ms",
        ],
    ].merge(
        obs[
            [
                "platform_id",
                "lead_hour",
                "datetime_utc",
                "record_id",
                "latitude",
                "longitude",
                "obs_speed_ms",
                "obs_dir_deg",
                "distance_to_station_km",
            ]
        ],
        on=["platform_id", "lead_hour", "datetime_utc", "record_id"],
        how="left",
    )
    merged["speed_error_ms"] = merged["pred_speed_ms"] - merged["obs_speed_ms"]
    merged["direction_error_deg"] = [angular_difference_deg(p, o) for p, o in zip(merged["pred_dir_deg"], merged["obs_dir_deg"])]
    merged["direction_abs_error_deg"] = merged["direction_error_deg"].abs()
    merged = merged.sort_values(["platform_id", "lead_hour", "dataset"]).reset_index(drop=True)
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


def lead_axis_ticks(obs: pd.DataFrame, merged: pd.DataFrame) -> tuple[list[int], list[str], list[str]]:
    combined = pd.concat(
        [
            obs[["lead_hour", "datetime_utc"]],
            merged[["lead_hour", "datetime_utc"]],
        ],
        ignore_index=True,
    ).dropna(subset=["lead_hour", "datetime_utc"])
    if combined.empty:
        return [], []
    combined["lead_hour"] = combined["lead_hour"].astype(int)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"])
    lead_times = combined.groupby("lead_hour")["datetime_utc"].min().sort_index()
    ticks = [int(lead) for lead in lead_times.index]
    lead_labels = [f"+{int(lead)}h" for lead in lead_times.index]
    utc_labels = [f"{time:%m-%d %H}" for time in lead_times]
    return ticks, lead_labels, utc_labels


def plot_timeseries(obs: pd.DataFrame, merged: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 7.8), sharex="col", constrained_layout=False)
    station_order = [station["station_id"] for station in VIRTUAL_POINT_STATIONS]
    for col, platform_id in enumerate(station_order):
        station = _station_definition_lookup()[platform_id]
        obs_sub = obs[obs["platform_id"] == platform_id].sort_values("lead_hour")
        merged_sub = merged[merged["platform_id"] == platform_id]
        ticks, lead_labels, utc_labels = lead_axis_ticks(obs_sub, merged_sub)
        x_min, x_max = min(ticks), max(ticks)
        ax_speed, ax_dir = axes[0, col], axes[1, col]
        for ax in (ax_speed, ax_dir):
            ax.set_facecolor("#F4F5F7")
            ax.grid(True, color="white", linewidth=1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(x_min - 1.5, x_max + 1.5)
            ax.set_xticks(ticks)

        ax_speed.plot(obs_sub["lead_hour"], obs_sub["obs_speed_ms"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.8, markersize=3.8, label="Observation")
        ax_dir.plot(obs_sub["lead_hour"], obs_sub["obs_dir_deg"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.4, markersize=3.5, label="Observation")

        for dataset in DATASETS:
            sub = merged[(merged["platform_id"] == platform_id) & (merged["dataset"] == dataset)].sort_values("lead_hour")
            init_label = TIMESERIES_FORECAST_INIT_TIMES[dataset].strftime("%m-%d %H UTC")
            dataset_label = f"{DATASET_LABELS[dataset]} ({init_label} init)"
            ax_speed.plot(sub["lead_hour"], sub["pred_speed_ms"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.5, markersize=3.2, label=dataset_label)
            ax_dir.plot(sub["lead_hour"], sub["pred_dir_deg"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.2, markersize=3.0, label=dataset_label)

        station_title = f"{station['label']} ({station['lon']:.2f}E,{station['lat']:.2f}N; R={station['radius_km']:.0f} km)"
        ax_speed.set_title(f"({chr(ord('a') + col)}) {station_title} wind speed", loc="left", fontweight="bold")
        ax_dir.set_title(f"({chr(ord('c') + col)}) {station_title} wind direction", loc="left", fontweight="bold")
        ax_speed.set_ylabel("Wind speed (m s$^{-1}$)")
        ax_dir.set_ylabel("Wind direction (°)")
        ax_dir.set_ylim(0, 360)
        ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.set_xticklabels(lead_labels, fontsize=7.6, rotation=0)
        for tick, utc_label in zip(ticks, utc_labels):
            ax_dir.text(
                tick,
                -0.12,
                utc_label,
                transform=ax_dir.get_xaxis_transform(),
                fontsize=7.4,
                rotation=-45,
                ha="left",
                va="top",
                rotation_mode="anchor",
                clip_on=False,
            )
        ax_dir.set_xlabel("Forecast lead time and valid UTC time", labelpad=36)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    fig.suptitle("Typhoon Wipha Case: Virtual Station Wind Speed and Direction by Forecast Lead", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.08, 0.98, 0.955])
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
