from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from paths import FIGURES_DIR, WIND_MODEL_STATISTICS_DIR


FONT_SCALE = 1.25
FONT_FAMILY = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
TEXT_LABELS = {
    "era5_realtime": "ERA5实时场",
    "era5_lagged_5d": "ERA5延迟5天预报",
    "gdas_forecast": "GDAS实时预报",
    "observed_beaufort": "观测蒲福风力等级",
    "lead_panel": "({panel}) {lead_hour} h预报",
}
BASE_FONT_SIZES = {
    "default": 10.5,
    "title": 12,
    "axis_label": 10.5,
    "legend": 9.5,
    "tick": 9.5,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}

METRICS_CSV = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h" / "wind_speed_metrics_by_beaufort.csv"

OUT_RMSE_PNG = FIGURES_DIR / "wind_speed_beaufort_rmse_three_experiments_24_48_72h.png"
OUT_RMSE_SVG = FIGURES_DIR / "wind_speed_beaufort_rmse_three_experiments_24_48_72h.svg"
OUT_MAE_PNG = FIGURES_DIR / "wind_speed_beaufort_mae_three_experiments_24_48_72h.png"
OUT_MAE_SVG = FIGURES_DIR / "wind_speed_beaufort_mae_three_experiments_24_48_72h.svg"

LEAD_HOURS = [24, 48, 72]
BEAUFORT_ORDER = ["<=2", "3", "4", "5", "6", "7", ">=8"]
BEAUFORT_TO_CODE = {label: index for index, label in enumerate(BEAUFORT_ORDER)}

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

PLOT_METRICS = {
    "rmse": {
        "title": "Wind Speed RMSE by Beaufort Class",
        "ylabel": "RMSE (m s$^{-1}$)",
        "png": OUT_RMSE_PNG,
        "svg": OUT_RMSE_SVG,
    },
    "mae": {
        "title": "Wind Speed MAE by Beaufort Class",
        "ylabel": "MAE (m s$^{-1}$)",
        "png": OUT_MAE_PNG,
        "svg": OUT_MAE_SVG,
    },
}


def load_beaufort_metrics(csv_path: Path = METRICS_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Beaufort metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"dataset", "dataset_label", "lead_hour", "obs_beaufort_group", "n", "rmse", "mae"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df["obs_beaufort_group"] = df["obs_beaufort_group"].astype(str)
    df["beaufort_code"] = df["obs_beaufort_group"].map(BEAUFORT_TO_CODE)

    df = df[
        df["lead_hour"].isin(LEAD_HOURS)
        & df["beaufort_code"].notna()
    ].dropna(subset=["rmse", "mae", "n"])
    df["lead_hour"] = df["lead_hour"].astype(int)
    df["beaufort_code"] = df["beaufort_code"].astype(int)

    dataset_rank = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    df["dataset_rank"] = df["dataset"].map(dataset_rank)
    return df.sort_values(["lead_hour", "dataset_rank", "beaufort_code"]).reset_index(drop=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT_SIZES["default"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax, ylabel: str) -> None:
    ax.set_facecolor("#F4F5F7")
    ax.grid(True, color="white", linewidth=1.1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.25, len(BEAUFORT_ORDER) - 0.75)
    ax.set_xticks(range(len(BEAUFORT_ORDER)))
    ax.set_xticklabels(BEAUFORT_ORDER)
    ax.set_xlabel(TEXT_LABELS["observed_beaufort"])
    ax.set_ylabel(ylabel)


def plot_metric_panel(ax, df: pd.DataFrame, lead_hour: int, metric: str, ylabel: str) -> None:
    lead_df = df[df["lead_hour"] == lead_hour]
    for dataset, style in DATASET_STYLES.items():
        sub = lead_df[lead_df["dataset"] == dataset].sort_values("beaufort_code")
        if sub.empty:
            continue

        ax.plot(
            sub["beaufort_code"],
            sub[metric],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.6,
            linewidth=1.65,
            linestyle=style["linestyle"],
        )

    panel_index = LEAD_HOURS.index(lead_hour)
    panel_letter = chr(ord("a") + panel_index)
    ax.set_title(
        TEXT_LABELS["lead_panel"].format(panel=panel_letter, lead_hour=lead_hour),
        loc="left",
        fontweight="bold",
    )
    style_axis(ax, ylabel)


def make_metric_figure(df: pd.DataFrame, metric: str) -> None:
    if metric not in PLOT_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(PLOT_METRICS)}")

    config = PLOT_METRICS[metric]
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(LEAD_HOURS), 1, figsize=(7.2, 10.2), constrained_layout=False)
    if len(LEAD_HOURS) == 1:
        axes = [axes]
    else:
        axes = axes.ravel().tolist()

    for ax, lead_hour in zip(axes, LEAD_HOURS):
        plot_metric_panel(ax, df, lead_hour, metric, config["ylabel"])

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
    first_ymin, first_ymax = axes[0].get_ylim()
    axes[0].set_ylim(first_ymin, first_ymax + (first_ymax - first_ymin) * 0.22)
    for ax in axes[1:]:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.20)

    fig.tight_layout(rect=[0.04, 0.02, 0.98, 0.99])

    fig.savefig(config["png"], bbox_inches="tight")
    fig.savefig(config["svg"], bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_beaufort_metrics()
    make_metric_figure(df, "rmse")
    make_metric_figure(df, "mae")

    print(f"Input: {METRICS_CSV}")
    print(f"Lead hours: {LEAD_HOURS}")
    print(f"Datasets: {list(DATASET_ORDER)}")
    print(f"RMSE PNG: {OUT_RMSE_PNG}")
    print(f"RMSE SVG: {OUT_RMSE_SVG}")
    print(f"MAE PNG: {OUT_MAE_PNG}")
    print(f"MAE SVG: {OUT_MAE_SVG}")


if __name__ == "__main__":
    main()
