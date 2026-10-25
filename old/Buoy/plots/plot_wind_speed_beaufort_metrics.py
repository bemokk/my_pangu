from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from paths import FIGURES_DIR, WIND_MODEL_STATISTICS_DIR


FONT_SCALE = 1
FONT_FAMILY = ["Times New Roman", "SimSun", "SimHei", "Microsoft YaHei", "DejaVu Serif"]
TEXT_LABELS = {
    # Experiment legend labels can be adjusted manually here.
    "era5_realtime": "ERA5",
    "era5_lagged_5d": "ERA5_Lagged",
    "gdas_forecast": "GDAS_RealTime",
    "observed_beaufort": "观测蒲福风力等级",
    "lead_panel": "({panel}) {metric} {lead_hour}h预报",
}
BASE_FONT_SIZES = {
    "default": 12,
    "title": 15,
    "axis_label": 13,
    "legend": 12,
    "tick": 12,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}

METRICS_CSV = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h" / "wind_speed_metrics_by_beaufort.csv"

OUT_RMSE_PNG = FIGURES_DIR / "wind_speed_beaufort_rmse_three_experiments_24_48_72h.png"
OUT_RMSE_SVG = FIGURES_DIR / "wind_speed_beaufort_rmse_three_experiments_24_48_72h.svg"
OUT_MAE_PNG = FIGURES_DIR / "wind_speed_beaufort_mae_three_experiments_24_48_72h.png"
OUT_MAE_SVG = FIGURES_DIR / "wind_speed_beaufort_mae_three_experiments_24_48_72h.svg"
OUT_COMBINED_PNG = FIGURES_DIR / "wind_speed_beaufort_rmse_mae_three_experiments_24_48_72h.png"
OUT_COMBINED_SVG = FIGURES_DIR / "wind_speed_beaufort_rmse_mae_three_experiments_24_48_72h.svg"
OUT_RMSE_DATA_CSV = FIGURES_DIR / "wind_speed_beaufort_rmse_three_experiments_24_48_72h_plot_data.csv"
OUT_MAE_DATA_CSV = FIGURES_DIR / "wind_speed_beaufort_mae_three_experiments_24_48_72h_plot_data.csv"
OUT_COMBINED_DATA_CSV = FIGURES_DIR / "wind_speed_beaufort_rmse_mae_three_experiments_24_48_72h_plot_data.csv"
COMBINED_FIGURE_SIZE = (13.2, 7.2)

LEAD_HOURS = [24, 48, 72]
BEAUFORT_ORDER = ["<=2", "3", "4", "5", "6", "7", ">=8"]
BEAUFORT_TO_CODE = {label: index for index, label in enumerate(BEAUFORT_ORDER)}

DATASET_STYLES = {
    "era5_realtime": {
        "label": TEXT_LABELS["era5_realtime"],
        "color": "#43A3EF",
        "marker": "o",
        "linestyle": "-",
    },
    "era5_lagged_5d": {
        "label": TEXT_LABELS["era5_lagged_5d"],
        "color": "#FEA040",
        "marker": "s",
        "linestyle": "-",
    },
    "gdas_forecast": {
        "label": TEXT_LABELS["gdas_forecast"],
        "color": "#EF767B",
        "marker": "^",
        "linestyle": "-",
    },
}
DATASET_ORDER = tuple(DATASET_STYLES)

PLOT_METRICS = {
    "rmse": {
        "label": "RMSE",
        "ylabel": "RMSE (m s$^{-1}$)",
        "png": OUT_RMSE_PNG,
        "svg": OUT_RMSE_SVG,
    },
    "mae": {
        "label": "MAE",
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


def style_axis(ax, ylabel: str, show_xlabel: bool = True, show_ylabel: bool = True) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#BFBFBF", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.set_xlim(-0.25, len(BEAUFORT_ORDER) - 0.75)
    ax.set_xticks(range(len(BEAUFORT_ORDER)))
    ax.set_xticklabels(BEAUFORT_ORDER)
    ax.set_xlabel(TEXT_LABELS["observed_beaufort"] if show_xlabel else "")
    ax.set_ylabel(ylabel if show_ylabel else "")


def plot_metric_panel(
    ax,
    df: pd.DataFrame,
    lead_hour: int,
    metric: str,
    panel_letter: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
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
            linewidth=1.35,
            linestyle=style["linestyle"],
        )

    ax.set_title(
        TEXT_LABELS["lead_panel"].format(
            panel=panel_letter,
            metric=PLOT_METRICS[metric]["label"],
            lead_hour=lead_hour,
        ),
        loc="left",
        fontweight="bold",
    )
    style_axis(
        ax,
        PLOT_METRICS[metric]["ylabel"],
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
    )


def build_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    dataset_labels = {dataset: style["label"] for dataset, style in DATASET_STYLES.items()}
    for metric, config in PLOT_METRICS.items():
        metric_df = df[
            [
                "dataset",
                "lead_hour",
                "obs_beaufort_group",
                "beaufort_code",
                "n",
                metric,
            ]
        ].copy()
        metric_df["dataset_label"] = metric_df["dataset"].map(dataset_labels)
        metric_df["metric"] = metric
        metric_df["metric_label"] = config["label"]
        metric_df["ylabel"] = config["ylabel"]
        metric_df = metric_df.rename(
            columns={
                "obs_beaufort_group": "beaufort_label",
                metric: "value",
            }
        )
        frames.append(
            metric_df[
                [
                    "metric",
                    "metric_label",
                    "ylabel",
                    "lead_hour",
                    "dataset",
                    "dataset_label",
                    "beaufort_code",
                    "beaufort_label",
                    "n",
                    "value",
                ]
            ]
        )

    return pd.concat(frames, ignore_index=True).sort_values(
        ["metric", "lead_hour", "dataset", "beaufort_code"]
    )


def save_plot_data(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = build_plot_data(df)
    plot_data.to_csv(OUT_COMBINED_DATA_CSV, index=False, encoding="utf-8-sig")
    plot_data[plot_data["metric"] == "rmse"].to_csv(
        OUT_RMSE_DATA_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    plot_data[plot_data["metric"] == "mae"].to_csv(
        OUT_MAE_DATA_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def make_combined_metric_figure(df: pd.DataFrame) -> None:
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=COMBINED_FIGURE_SIZE, constrained_layout=False)

    panel_index = 0
    for row_index, metric in enumerate(("rmse", "mae")):
        for col_index, lead_hour in enumerate(LEAD_HOURS):
            panel_letter = chr(ord("a") + panel_index)
            plot_metric_panel(
                axes[row_index, col_index],
                df,
                lead_hour,
                metric,
                panel_letter,
                show_xlabel=row_index == 1,
                show_ylabel=col_index == 0,
            )
            panel_index += 1

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CFCFCF",
        framealpha=0.82,
        borderaxespad=0.2,
    )
    for ax in axes.ravel():
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.20)

    fig.tight_layout(rect=[0.04, 0.04, 0.99, 0.99])

    for path in [OUT_COMBINED_PNG, OUT_RMSE_PNG, OUT_MAE_PNG]:
        fig.savefig(path, bbox_inches="tight")
    for path in [OUT_COMBINED_SVG, OUT_RMSE_SVG, OUT_MAE_SVG]:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_beaufort_metrics()
    save_plot_data(df)
    make_combined_metric_figure(df)

    print(f"Input: {METRICS_CSV}")
    print(f"Lead hours: {LEAD_HOURS}")
    print(f"Datasets: {list(DATASET_ORDER)}")
    print(f"Combined plot data CSV: {OUT_COMBINED_DATA_CSV}")
    print(f"RMSE plot data CSV: {OUT_RMSE_DATA_CSV}")
    print(f"MAE plot data CSV: {OUT_MAE_DATA_CSV}")
    print(f"Combined PNG: {OUT_COMBINED_PNG}")
    print(f"Combined SVG: {OUT_COMBINED_SVG}")
    print(f"RMSE PNG: {OUT_RMSE_PNG}")
    print(f"RMSE SVG: {OUT_RMSE_SVG}")
    print(f"MAE PNG: {OUT_MAE_PNG}")
    print(f"MAE SVG: {OUT_MAE_SVG}")


if __name__ == "__main__":
    main()
