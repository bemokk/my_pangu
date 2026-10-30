"""Plot January and August wind-direction frequency radar charts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "Statistics" / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "figures"
DEFAULT_DATA_FILE = "east_china_sea_wind_direction_frequency_jan_aug.csv"
SECTORS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)
SERIES_ORDER = ("observed", "era5_direct", "era5_lagged_120h", "gdas_direct", "gdas_lagged_120h")
SERIES_LABELS = {
    "observed": "Observed",
    "era5_direct": "E0",
    "era5_lagged_120h": "E120",
    "gdas_direct": "G0",
    "gdas_lagged_120h": "G120",
}
COLORS = {
    "observed": "#4D4D4D",
    "era5_direct": "#43A3EF",
    "era5_lagged_120h": "#009E73",
    "gdas_direct": "#FEA040",
    "gdas_lagged_120h": "#EF767B",
}
MARKERS = {"observed": "D", "era5_direct": "o", "era5_lagged_120h": "s", "gdas_direct": "^", "gdas_lagged_120h": "v"}
MONTHS = (1, 8)
LEADS = (24, 72, 120)


def closed(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.concatenate([values, values[:1]])


def load_frequency(data_root: Path) -> pd.DataFrame:
    path = data_root / DEFAULT_DATA_FILE
    frame = pd.read_csv(path)
    required = {"month", "lead_hour", "series", "sector_code", "frequency"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["month"] = pd.to_numeric(frame["month"], errors="raise").astype(int)
    frame["lead_hour"] = pd.to_numeric(frame["lead_hour"], errors="raise").astype(int)
    frame["sector_code"] = pd.to_numeric(frame["sector_code"], errors="raise").astype(int)
    frame["frequency"] = pd.to_numeric(frame["frequency"], errors="coerce")
    return frame


def plot(frame: pd.DataFrame, output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    months = MONTHS
    leads = LEADS
    frame = frame[frame["lead_hour"].isin(leads)].copy()
    angles = np.linspace(0, 2 * np.pi, len(SECTORS), endpoint=False)
    closed_angles = closed(angles)
    max_frequency = float(frame["frequency"].max())
    radial_max = max(0.2, np.ceil(max_frequency * 20.0) / 20.0)

    fig, axes = plt.subplots(
        2, 3, figsize=(13.2, 8.6), subplot_kw={"projection": "polar"}, constrained_layout=False
    )
    axes = np.asarray(axes)
    month_names = {1: "January", 8: "August"}
    for row_index, month in enumerate(months):
        month_name = month_names[month]
        for col_index, lead in enumerate(leads):
            ax = axes[row_index, col_index]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels(SECTORS, fontsize=8)
            ax.set_ylim(0, radial_max)
            ax.set_yticks(np.linspace(0, radial_max, 5)[1:])
            ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, radial_max, 5)[1:]], fontsize=7)
            ax.grid(True, color="#BFBFBF", linewidth=0.7, linestyle="--", alpha=0.75)
            ax.spines["polar"].set_color("#333333")
            ax.spines["polar"].set_linewidth(0.9)
            for series in SERIES_ORDER:
                sub = frame[(frame["month"] == month) & (frame["lead_hour"] == lead) & (frame["series"] == series)].sort_values("sector_code")
                if sub.empty:
                    continue
                values = sub.set_index("sector_code").reindex(range(16))["frequency"].to_numpy()
                ax.plot(
                    closed_angles,
                    closed(values),
                    color=COLORS[series],
                    marker=MARKERS[series],
                    markersize=3.0,
                    linewidth=1.2,
                    label=SERIES_LABELS[series],
                )
            panel = chr(ord("a") + row_index * len(leads) + col_index)
            ax.set_title(f"({panel}) {month_name}, {lead} h", y=1.12, fontsize=10, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(SERIES_ORDER),
        frameon=True,
        fontsize=10,
        title="Series",
        title_fontsize=10,
    )
    fig.suptitle("East China Sea 10 m wind-direction frequency", fontsize=13, y=0.985)
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.95], w_pad=1.5, h_pad=1.8)
    stem = output_root / "east_china_sea_wind_direction_frequency_radar_january_august"
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".svg"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=300, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=300, facecolor="white")
    plt.close(fig)
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    outputs = plot(load_frequency(args.data_root), args.output_root)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
