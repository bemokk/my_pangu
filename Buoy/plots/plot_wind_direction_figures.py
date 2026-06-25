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


FONT_SCALE = 1.0
FONT_FAMILY = ["Times New Roman", "SimSun", "SimHei", "Microsoft YaHei", "DejaVu Serif"]
TEXT_LABELS = {
    "era5_realtime": "ERA5实时场",
    "era5_lagged_5d": "ERA5延迟5天预报",
    "gdas_forecast": "GDAS实时预报",
    "observed": "观测",
    "lead_time": "预报时效（h）",
    "rmse": "RMSE (degree)",
    "mae": "MAE (degree)",
    "lead_panel": "({panel}) {lead_hour}h预报",
}
BASE_FONT_SIZES = {
    "default": 14,
    "title": 15,
    "axis_label": 13,
    "legend": 12,
    "tick": 12,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}

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
FREQUENCY_LEGEND_BBOX = (-0.6, 1.22)
SHOW_FREQUENCY_RADIAL_TICK_LABELS = False

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
        "label": TEXT_LABELS["era5_realtime"],
        "color": "#C44E52",
        "marker": "o",
        "linestyle": "-",
    },
    "era5_lagged_5d": {
        "label": TEXT_LABELS["era5_lagged_5d"],
        "color": "#4C72B0",
        "marker": "s",
        "linestyle": "-",
    },
    "gdas_forecast": {
        "label": TEXT_LABELS["gdas_forecast"],
        "color": "#55A868",
        "marker": "^",
        "linestyle": "-",
    },
}
DATASET_ORDER = tuple(DATASET_STYLES)

OBS_STYLE = {
    "label": TEXT_LABELS["observed"],
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
            "font.family": FONT_FAMILY,
            "font.serif": FONT_FAMILY,
            "font.sans-serif": ["SimHei", "SimSun", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": FONT_SIZES["default"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )


def style_metric_axis(ax, ylabel: str) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#BFBFBF", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.set_xlim(-1, 73)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel(TEXT_LABELS["lead_time"])
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
            linewidth=1.35,
            linestyle=style["linestyle"],
        )

    ax.set_title(title, loc="left", fontweight="bold")
    style_metric_axis(ax, ylabel)


def make_direction_metrics_figure(df: pd.DataFrame) -> None:
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6), constrained_layout=False)
    plot_direction_metric(axes[0], df, "rmse", "(a) RMSE", TEXT_LABELS["rmse"])
    plot_direction_metric(axes[1], df, "mae", "(b) MAE", TEXT_LABELS["mae"])

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CFCFCF",
        framealpha=0.82,
        borderaxespad=0.2,
    )
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.22)

    fig.tight_layout(rect=[0.04, 0.03, 0.98, 0.98])
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
        ax.set_facecolor("white")
        ax.set_xticks(base_angles)
        ax.set_xticklabels(DIRECTION_SECTORS)
        ax.set_ylim(0, radial_max)
        ax.set_yticks(np.linspace(0, radial_max, 5)[1:])
        if SHOW_FREQUENCY_RADIAL_TICK_LABELS:
            ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, radial_max, 5)[1:]])
        else:
            ax.set_yticklabels([])
        ax.grid(True, color="#BFBFBF", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.spines["polar"].set_visible(True)
        ax.spines["polar"].set_color("#333333")
        ax.spines["polar"].set_linewidth(1.0)

        obs = observation_frequency(df, lead_hour)
        ax.plot(
            angles,
            closed_values(obs),
            label=OBS_STYLE["label"],
            color=OBS_STYLE["color"],
            marker=OBS_STYLE["marker"],
            markersize=3.4,
            linewidth=1.35,
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
                linewidth=1.35,
                linestyle=style["linestyle"],
        )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            TEXT_LABELS["lead_panel"].format(panel=panel_letter, lead_hour=lead_hour),
            y=1.1,
            fontweight="bold",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=FREQUENCY_LEGEND_BBOX,
        frameon=True,
        facecolor="white",
        edgecolor="#CFCFCF",
        framealpha=0.82,
        borderaxespad=0.2,
    )
    fig.tight_layout(rect=[0.01, 0.04, 0.99, 0.96])
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
