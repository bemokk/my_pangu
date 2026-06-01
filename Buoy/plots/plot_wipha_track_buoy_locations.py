from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from plots.wipha_case_common import (
    DATASET_COLORS,
    FIGURES_DIR,
    MAP_AREA,
    OUT_TRACK_BUOYS_PNG,
    OUT_TRACK_BUOYS_SVG,
    PLATFORM_COLORS,
    WINDOW_END,
    WINDOW_START,
    ensure_dirs,
    load_real_wipha_track,
    prepare_buoy_case_data,
    set_plot_style,
)


def plot_track_buoy_locations(real_track: pd.DataFrame, obs: pd.DataFrame) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.set_facecolor("#F4F5F7")
    ax.grid(True, color="white", linewidth=1.0)
    ax.set_xlim(MAP_AREA[0], MAP_AREA[1])
    ax.set_ylim(MAP_AREA[2], MAP_AREA[3])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Typhoon Wipha Track and Selected Platform Locations", loc="left", fontweight="bold")

    case_track = real_track[real_track["datetime_utc"].between(WINDOW_START, WINDOW_END)]
    if not case_track.empty:
        ax.plot(
            case_track["lon"],
            case_track["lat"],
            color="#222222",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Observed Wipha track",
        )
        for _, row in case_track.iloc[:: max(1, len(case_track) // 8)].iterrows():
            ax.text(
                row["lon"] + 0.15,
                row["lat"] + 0.12,
                pd.Timestamp(row["datetime_utc"]).strftime("%m-%d %H"),
                fontsize=7.5,
            )

    for platform_id, sub in obs.groupby("platform_id"):
        color = PLATFORM_COLORS.get(platform_id, "#777777")
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=22,
            alpha=0.45,
            color=color,
            edgecolor="none",
            label=f"{platform_id} positions",
        )
        mean_lon, mean_lat = sub["longitude"].mean(), sub["latitude"].mean()
        ax.scatter([mean_lon], [mean_lat], marker="*", s=180, color=color, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(mean_lon + 0.25, mean_lat + 0.25, platform_id, color=color, fontweight="bold")

    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.text(
        0.5,
        0.015,
        "Platform points show observed positions during 2025-07-17 00 UTC to 2025-07-22 23 UTC.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.98])
    fig.savefig(OUT_TRACK_BUOYS_PNG, bbox_inches="tight")
    fig.savefig(OUT_TRACK_BUOYS_SVG, bbox_inches="tight")
    plt.close(fig)


def generate() -> list[Path]:
    ensure_dirs()
    obs, _, _ = prepare_buoy_case_data()
    real_track = load_real_wipha_track(force_refresh=False)
    plot_track_buoy_locations(real_track, obs)
    return [OUT_TRACK_BUOYS_PNG, OUT_TRACK_BUOYS_SVG]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
