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
    DATASET_COLORS,
    DATASET_LABELS,
    DATASETS,
    OUT_TIMESERIES_PNG,
    OUT_TIMESERIES_SVG,
    SELECTED_PLATFORMS,
    WINDOW_END,
    WINDOW_START,
    ensure_dirs,
    prepare_buoy_case_data,
    set_plot_style,
)


def plot_timeseries(obs: pd.DataFrame, merged: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.8), sharex=True, constrained_layout=False)
    for col, platform_id in enumerate(SELECTED_PLATFORMS):
        obs_sub = obs[obs["platform_id"] == platform_id].sort_values("datetime_utc")
        ax_speed, ax_dir = axes[0, col], axes[1, col]
        for ax in (ax_speed, ax_dir):
            ax.set_facecolor("#F4F5F7")
            ax.grid(True, color="white", linewidth=1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(WINDOW_START, WINDOW_END)

        ax_speed.plot(obs_sub["datetime_utc"], obs_sub["obs_speed_ms"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.8, markersize=3.8, label="Buoy observation")
        ax_dir.plot(obs_sub["datetime_utc"], obs_sub["obs_dir_deg"], color=DATASET_COLORS["observation"], marker="o", linewidth=1.4, markersize=3.5, label="Buoy observation")

        for dataset in DATASETS:
            sub = merged[(merged["platform_id"] == platform_id) & (merged["dataset"] == dataset)].sort_values("datetime_utc")
            ax_speed.plot(sub["datetime_utc"], sub["pred_speed_ms"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.5, markersize=3.2, label=DATASET_LABELS[dataset])
            ax_dir.plot(sub["datetime_utc"], sub["pred_dir_deg"], color=DATASET_COLORS[dataset], marker="s", linewidth=1.2, markersize=3.0, label=DATASET_LABELS[dataset])

        ax_speed.set_title(f"({chr(ord('a') + col)}) {platform_id} wind speed", loc="left", fontweight="bold")
        ax_dir.set_title(f"({chr(ord('c') + col)}) {platform_id} wind direction", loc="left", fontweight="bold")
        ax_speed.set_ylabel("Wind speed (m s$^{-1}$)")
        ax_dir.set_ylabel("Wind direction (°)")
        ax_dir.set_ylim(0, 360)
        ax_dir.set_yticks([0, 90, 180, 270, 360])
        ax_dir.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H UTC"))
        ax_dir.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    fig.suptitle("Typhoon Wipha Case: Platform Wind Speed and Direction Time Series", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.955])
    fig.savefig(OUT_TIMESERIES_PNG, bbox_inches="tight")
    fig.savefig(OUT_TIMESERIES_SVG, bbox_inches="tight")
    plt.close(fig)


def generate() -> list[Path]:
    ensure_dirs()
    obs, merged, _ = prepare_buoy_case_data()
    plot_timeseries(obs, merged)
    return [OUT_TIMESERIES_PNG, OUT_TIMESERIES_SVG]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
