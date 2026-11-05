"""Plot East China Sea seasonal wind-speed RMSE and MAE."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from .publication_style import FONT_FAMILY, format_figure, set_style as set_publication_style
except ImportError:
    from publication_style import FONT_FAMILY, format_figure, set_style as set_publication_style
set_publication_style()

import pandas as pd


PLOTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLOTS_ROOT.parents[1]
DATA_CSV = PROJECT_ROOT / "experiment_2025" / "Statistics" / "data" / "east_china_sea_wind_speed_metrics_by_season_lead_2025.csv"
FIGURES_DIR = PLOTS_ROOT / "figures"
SEASONS = ("winter", "spring", "summer", "autumn")
SEASON_LABELS = {"winter": "Winter (Dec–Feb)", "spring": "Spring (Mar–May)", "summer": "Summer (Jun–Aug)", "autumn": "Autumn (Sep–Nov)"}
EXPERIMENTS = (
    ("era5_direct", "E0", "#43A3EF"),
    ("era5_lagged_120h", "E120", "#009E73"),
    ("gdas_direct", "G0", "#FEA040"),
    ("gdas_lagged_120h", "G120", "#EF767B"),
)


def main() -> None:
    frame = pd.read_csv(DATA_CSV)
    frame["lead_hour"] = pd.to_numeric(frame["lead_hour"], errors="raise").astype(int)
    frame["rmse"] = pd.to_numeric(frame["rmse"], errors="coerce")
    frame["mae"] = pd.to_numeric(frame["mae"], errors="coerce")

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharex=True)
    panel = 0
    for row, metric in enumerate(("rmse", "mae")):
        global_max = float(frame[metric].max())
        for col, season in enumerate(SEASONS):
            ax = axes[row, col]
            subset = frame[frame["season"] == season]
            for experiment, label, color in EXPERIMENTS:
                line = subset[subset["experiment"] == experiment].sort_values("lead_hour")
                ax.plot(line["lead_hour"], line[metric], color=color, linewidth=1.15, marker="o", markersize=2.6, label=label)
            ax.set_title(f"({chr(ord('a') + panel)}) {SEASON_LABELS[season]} {metric.upper()}", loc="left", pad=3)
            ax.set_xlim(-2, 122)
            ax.set_ylim(0, global_max * 1.08)
            ax.set_xticks(range(0, 121, 12))
            ax.grid(True, linestyle=":", linewidth=0.7, color="#BFC4C8", alpha=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=8, length=3, width=0.7)
            ax.set_ylabel(f"{metric.upper()} (m s$^{{-1}}$)")
            if row == 1:
                ax.set_xlabel("Forecast lead time (h)")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("#606060")
                spine.set_linewidth(0.75)
            panel += 1

    axes[0, 0].legend(loc="upper left", ncol=2, frameon=False, handlelength=2.0, columnspacing=1.0)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.09, hspace=0.28, wspace=0.22)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "east_china_sea_wind_speed_metrics_by_season_2025"
    outputs = [FIGURES_DIR / f"{stem}.png", FIGURES_DIR / f"{stem}.svg", FIGURES_DIR / f"{stem}.pdf"]
    format_figure(fig)
    fig.savefig(outputs[0], dpi=600, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=600, facecolor="white")
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
