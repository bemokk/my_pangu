from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lead_zero_metrics import append_lead_zero_rows, build_lead_zero_metric_rows
from paths import FIGURES_DIR, WIND_MODEL_STATISTICS_DIR


STATS_DIR = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h"
DIRECTION_METRICS_CSV = STATS_DIR / "wind_direction_metrics_by_lead.csv"
DIRECTION_FREQUENCY_CSV = STATS_DIR / "wind_direction_frequency_by_lead.csv"

OUT_METRICS_PNG = FIGURES_DIR / "wind_direction_metrics_figure8_style.png"
OUT_METRICS_SVG = FIGURES_DIR / "wind_direction_metrics_figure8_style.svg"
OUT_FREQUENCY_PNG = FIGURES_DIR / "wind_direction_frequency_radar_24_48_72h.png"
OUT_FREQUENCY_SVG = FIGURES_DIR / "wind_direction_frequency_radar_24_48_72h.svg"

METRICS_LEAD_HOURS = list(range(0, 73, 3))
FREQUENCY_LEAD_HOURS = [24, 48, 72]
X_TICKS = [0, 12, 24, 36, 48, 60, 72]

DIRECTION_SECTORS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]
SECTOR_TO_CODE = {sector: index for index, sector in enumerate(DIRECTION_SECTORS)}

DATASET_STYLES = {
    "era5_realtime": {
        "label": "ERA5 realtime",
        "color": "#C44E52",
        "marker": "o",
        "linestyle": "-",
    },
    "era5_lagged_5d": {
        "label": "ERA5 lagged 5d forecast",
        "color": "#4C72B0",
        "marker": "s",
        "linestyle": "-",
    },
    "gdas_forecast": {
        "label": "GDAS forecast",
        "color": "#55A868",
        "marker": "^",
        "linestyle": "-",
    },
}
DATASET_ORDER = tuple(DATASET_STYLES)

OBS_STYLE = {
    "label": "Observed",
    "color": "#CC79A7",
    "marker": "D",
    "linestyle": "-",
}


def load_direction_metrics(
    csv_path: Path = DIRECTION_METRICS_CSV,
    include_lead_zero: bool = False,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Direction metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"dataset", "dataset_label", "lead_hour", "rmse", "mae"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    if include_lead_zero:
        zero_rows = build_lead_zero_metric_rows("wind_direction")
        df = append_lead_zero_rows(df, zero_rows)

    df = df[df["lead_hour"].isin(METRICS_LEAD_HOURS)].dropna(subset=["lead_hour", "rmse", "mae"])
    df["lead_hour"] = df["lead_hour"].astype(int)

    dataset_rank = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    df["dataset_rank"] = df["dataset"].map(dataset_rank)
    return df.sort_values(["lead_hour", "dataset_rank"]).reset_index(drop=True)


def load_direction_frequency(csv_path: Path = DIRECTION_FREQUENCY_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Direction frequency CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {
        "dataset",
        "dataset_label",
        "lead_hour",
        "direction_sector",
        "obs_frequency",
        "pred_frequency",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
    df["obs_frequency"] = pd.to_numeric(df["obs_frequency"], errors="coerce")
    df["pred_frequency"] = pd.to_numeric(df["pred_frequency"], errors="coerce")
    df["direction_sector"] = df["direction_sector"].astype(str)
    df["sector_code"] = df["direction_sector"].map(SECTOR_TO_CODE)

    df = df[
        df["lead_hour"].isin(FREQUENCY_LEAD_HOURS)
        & df["sector_code"].notna()
    ].dropna(subset=["lead_hour", "obs_frequency", "pred_frequency"])
    df["lead_hour"] = df["lead_hour"].astype(int)
    df["sector_code"] = df["sector_code"].astype(int)

    dataset_rank = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    df["dataset_rank"] = df["dataset"].map(dataset_rank)
    return df.sort_values(["lead_hour", "dataset_rank", "sector_code"]).reset_index(drop=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def style_metric_axis(ax, ylabel: str) -> None:
    ax.set_facecolor("#F4F5F7")
    ax.grid(True, color="white", linewidth=1.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-1, 73)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel(ylabel)


def plot_direction_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    for dataset, style in DATASET_STYLES.items():
        sub = df[df["dataset"] == dataset].sort_values("lead_hour")
        if sub.empty:
            continue
        ax.plot(
            sub["lead_hour"],
            sub[metric],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.0,
            linewidth=1.65,
            linestyle=style["linestyle"],
        )

    ax.set_title(title, loc="left", fontweight="bold")
    style_metric_axis(ax, ylabel)


def make_direction_metrics_figure(df: pd.DataFrame) -> None:
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6), constrained_layout=False)
    plot_direction_metric(axes[0], df, "rmse", "(a) RMSE", "RMSE (degree)")
    plot_direction_metric(axes[1], df, "mae", "(b) MAE", "MAE (degree)")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.9,
        borderaxespad=0.2,
    )
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.22)

    fig.suptitle("Wind Direction Forecast Error Against China Sea Buoy Observations", y=0.985, fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Lead times are 0 h and every 3 hours from 3 h to 72 h. Direction errors use circular angular differences.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.945])
    fig.savefig(OUT_METRICS_PNG, bbox_inches="tight")
    fig.savefig(OUT_METRICS_SVG, bbox_inches="tight")
    plt.close(fig)


def closed_values(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.concatenate([arr, arr[:1]])


def sector_angles() -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, len(DIRECTION_SECTORS), endpoint=False)


def frequency_series_for_dataset(df: pd.DataFrame, lead_hour: int, dataset: str, column: str) -> pd.Series:
    sub = df[(df["lead_hour"] == lead_hour) & (df["dataset"] == dataset)]
    return sub.set_index("direction_sector").reindex(DIRECTION_SECTORS)[column]


def observation_frequency(df: pd.DataFrame, lead_hour: int) -> pd.Series:
    lead_df = df[df["lead_hour"] == lead_hour]
    obs_by_sector = lead_df.pivot_table(
        index="direction_sector",
        values="obs_frequency",
        aggfunc="mean",
        observed=False,
    )
    return obs_by_sector.reindex(DIRECTION_SECTORS)["obs_frequency"]


def make_direction_frequency_figure(df: pd.DataFrame) -> None:
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base_angles = sector_angles()
    angles = np.concatenate([base_angles, base_angles[:1]])
    max_frequency = float(
        np.nanmax(
            [
                df["obs_frequency"].max(),
                df["pred_frequency"].max(),
            ]
        )
    )
    radial_max = max(0.2, np.ceil(max_frequency * 20.0) / 20.0)

    fig, axes = plt.subplots(
        1,
        len(FREQUENCY_LEAD_HOURS),
        figsize=(14.4, 5.3),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    if len(FREQUENCY_LEAD_HOURS) == 1:
        axes = [axes]

    for panel_index, (ax, lead_hour) in enumerate(zip(axes, FREQUENCY_LEAD_HOURS)):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(base_angles)
        ax.set_xticklabels(DIRECTION_SECTORS)
        ax.set_ylim(0, radial_max)
        ax.set_yticks(np.linspace(0, radial_max, 5)[1:])
        ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, radial_max, 5)[1:]])
        ax.grid(True, color="#D9D9D9", linewidth=0.75)

        obs = observation_frequency(df, lead_hour)
        ax.plot(
            angles,
            closed_values(obs),
            label=OBS_STYLE["label"],
            color=OBS_STYLE["color"],
            marker=OBS_STYLE["marker"],
            markersize=3.4,
            linewidth=1.8,
            linestyle=OBS_STYLE["linestyle"],
        )

        for dataset, style in DATASET_STYLES.items():
            pred = frequency_series_for_dataset(df, lead_hour, dataset, "pred_frequency")
            if pred.isna().all():
                continue
            ax.plot(
                angles,
                closed_values(pred),
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                markersize=3.2,
                linewidth=1.45,
                linestyle=style["linestyle"],
            )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(f"({panel_letter}) Lead {lead_hour} h", y=1.1, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Wind Direction Frequency by Forecast Lead Time", y=1.09, fontsize=14)
    fig.text(
        0.5,
        0.02,
        "Frequencies are grouped into 16 compass sectors. Observed frequencies use buoy wind direction.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0.01, 0.08, 0.99, 0.94])
    fig.savefig(OUT_FREQUENCY_PNG, bbox_inches="tight")
    fig.savefig(OUT_FREQUENCY_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_direction_metrics(include_lead_zero=True)
    frequency = load_direction_frequency()
    make_direction_metrics_figure(metrics)
    make_direction_frequency_figure(frequency)

    print(f"Metrics input: {DIRECTION_METRICS_CSV}")
    print(f"Frequency input: {DIRECTION_FREQUENCY_CSV}")
    print(f"Metrics leads: {METRICS_LEAD_HOURS}")
    print(f"Frequency leads: {FREQUENCY_LEAD_HOURS}")
    print(f"Metrics PNG: {OUT_METRICS_PNG}")
    print(f"Metrics SVG: {OUT_METRICS_SVG}")
    print(f"Frequency PNG: {OUT_FREQUENCY_PNG}")
    print(f"Frequency SVG: {OUT_FREQUENCY_SVG}")


if __name__ == "__main__":
    main()
