from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from lead_zero_metrics import append_lead_zero_rows, build_lead_zero_metric_rows
from paths import FIGURES_DIR, WIND_MODEL_STATISTICS_DIR


METRICS_CSV = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h" / "wind_speed_metrics_by_lead.csv"
OUT_PNG = FIGURES_DIR / "wind_speed_metrics_figure2_style.png"
OUT_SVG = FIGURES_DIR / "wind_speed_metrics_figure2_style.svg"

LEAD_HOURS = list(range(0, 73, 3))
X_TICKS = [0, 12, 24, 36, 48, 60, 72]

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

PLOT_METRICS = [
    ("rmse", "(a) RMSE", "RMSE (m s$^{-1}$)"),
    ("mae", "(b) MAE", "MAE (m s$^{-1}$)"),
    ("corr", "(c) CC", "Correlation coefficient"),
]


def load_metrics(csv_path: Path, include_lead_zero: bool = False) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"dataset", "lead_hour", "rmse", "mae", "corr"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df[df["dataset"].isin(DATASET_STYLES)].copy()
    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
    for metric, _, _ in PLOT_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if include_lead_zero:
        zero_rows = build_lead_zero_metric_rows("wind_speed")
        df = append_lead_zero_rows(df, zero_rows)

    return df[df["lead_hour"].isin(LEAD_HOURS)].sort_values(["dataset", "lead_hour"])


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def style_axis(ax, ylabel: str) -> None:
    ax.set_facecolor("#F4F5F7")
    ax.grid(True, color="white", linewidth=1.15)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-1, 73)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel(ylabel)


def plot_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
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
            linewidth=1.7,
            linestyle=style["linestyle"],
        )

    ax.set_title(title, loc="left", fontweight="bold")
    style_axis(ax, ylabel)


def make_figure(df: pd.DataFrame) -> None:
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), constrained_layout=False)
    for ax, (metric, title, ylabel) in zip(axes, PLOT_METRICS):
        plot_metric(ax, df, metric, title, ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.88,
    )
    fig.suptitle("Wind Speed Forecast Skill Against China Sea Buoy Observations", y=1.04, fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Lead times are 0 h and every 3 hours from 3 h to 72 h. Metrics use buoy wind speed as truth.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.94])

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_metrics(METRICS_CSV, include_lead_zero=True)
    make_figure(df)
    print(f"Input: {METRICS_CSV}")
    print(f"PNG: {OUT_PNG}")
    print(f"SVG: {OUT_SVG}")


if __name__ == "__main__":
    main()
