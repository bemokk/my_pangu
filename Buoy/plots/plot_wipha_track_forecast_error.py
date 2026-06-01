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
    DATASET_LABELS,
    DATASETS,
    MAP_AREA,
    OUT_TRACK_ERROR_PNG,
    OUT_TRACK_ERROR_SVG,
    OUT_TRACK_ERRORS_CSV,
    OUT_TRACKS_CSV,
    build_tracks_and_errors,
    ensure_dirs,
    load_real_wipha_track,
    set_plot_style,
)


def plot_track_error(tracks: pd.DataFrame, errors: pd.DataFrame) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=False)
    ax_map, ax_err = axes
    for ax in axes:
        ax.set_facecolor("#F4F5F7")
        ax.grid(True, color="white", linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_map.set_title("(a) 72 h track forecasts from 2025-07-17 00 UTC", loc="left", fontweight="bold")
    ax_map.set_xlim(MAP_AREA[0], MAP_AREA[1])
    ax_map.set_ylim(MAP_AREA[2], MAP_AREA[3])
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")

    style_map = {
        "real_track": {"label": "Observed track", "color": "#222222", "marker": "o", "linestyle": "-"},
        "gdas_forecast": {"label": "GDAS forecast", "color": DATASET_COLORS["gdas_forecast"], "marker": "^", "linestyle": "--"},
        "era5_lagged_5d": {"label": "ERA5 lagged 5d forecast", "color": DATASET_COLORS["era5_lagged_5d"], "marker": "s", "linestyle": "--"},
    }
    for scheme, style in style_map.items():
        sub = tracks[tracks["scheme"] == scheme].sort_values("lead_hour")
        if sub.empty:
            continue
        ax_map.plot(sub["lon"], sub["lat"], color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.8, markersize=4.0, label=style["label"])
        for _, row in sub[sub["lead_hour"].isin([0, 24, 48, 72])].iterrows():
            if pd.notna(row["lon"]) and pd.notna(row["lat"]):
                ax_map.text(row["lon"] + 0.12, row["lat"] + 0.12, f"+{int(row['lead_hour'])}h", fontsize=7.5, color=style["color"])
    ax_map.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    ax_err.set_title("(b) Track position error", loc="left", fontweight="bold")
    for dataset in DATASETS:
        sub = errors[errors["scheme"] == dataset].sort_values("lead_hour")
        if sub.empty:
            continue
        ax_err.plot(sub["lead_hour"], sub["track_error_km"], color=DATASET_COLORS[dataset], marker="o", linewidth=1.8, markersize=4.0, label=DATASET_LABELS[dataset])
    ax_err.set_xlim(0, 72)
    ax_err.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax_err.set_xlabel("Forecast lead time (h)")
    ax_err.set_ylabel("Track error (km)")
    ax_err.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle("Typhoon Wipha Track Forecast Error Comparison", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.95])
    fig.savefig(OUT_TRACK_ERROR_PNG, bbox_inches="tight")
    fig.savefig(OUT_TRACK_ERROR_SVG, bbox_inches="tight")
    plt.close(fig)


def generate() -> list[Path]:
    ensure_dirs()
    real_track = load_real_wipha_track(force_refresh=False)
    tracks, errors = build_tracks_and_errors(real_track)
    plot_track_error(tracks, errors)
    return [OUT_TRACKS_CSV, OUT_TRACK_ERRORS_CSV, OUT_TRACK_ERROR_PNG, OUT_TRACK_ERROR_SVG]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
