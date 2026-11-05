"""Plot wind-speed RMSE and MAE by observed Beaufort wind-force group."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
try:
    from .publication_style import FONT_FAMILY, format_figure, set_style as set_publication_style
except ImportError:
    from publication_style import FONT_FAMILY, format_figure, set_style as set_publication_style
set_publication_style()

import pandas as pd


PLOTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLOTS_ROOT.parents[1]
STATS_DATA = PROJECT_ROOT / "experiment_2025" / "Statistics" / "data"
FIGURES_DIR = PLOTS_ROOT / "figures"
METRICS_CSV = STATS_DATA / "wind_speed_metrics_by_beaufort_2025.csv"

LEADS = (24, 72, 120)
BEAUFORT_ORDER = ("<=2", "3", "4", "5", "6", "7", ">=8")
EXPERIMENTS = ("E0", "G0", "E120", "G120")
STYLES = {
    "E0": {"label": "E0", "color": "#43A3EF", "marker": "o"},
    "G0": {"label": "G0", "color": "#FEA040", "marker": "s"},
    "E120": {"label": "E120", "color": "#009E73", "marker": "^"},
    "G120": {"label": "G120", "color": "#EF767B", "marker": "D"},
}
METRICS = {
    "rmse": ("RMSE", "RMSE (m s$^{-1}$)"),
    "mae": ("MAE", "MAE (m s$^{-1}$)"),
}
OUTPUT_STEM = "wind_speed_beaufort_rmse_mae_2025"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def load_data() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(METRICS_CSV)
    frame = pd.read_csv(METRICS_CSV)
    frame["lead_hour"] = pd.to_numeric(frame["lead_hour"], errors="coerce")
    frame["beaufort_group"] = frame["beaufort_group"].astype(str)
    frame = frame[frame["lead_hour"].isin(LEADS) & frame["beaufort_group"].isin(BEAUFORT_ORDER)].copy()
    frame["beaufort_code"] = pd.Categorical(
        frame["beaufort_group"], categories=BEAUFORT_ORDER, ordered=True
    ).codes
    return frame.sort_values(["lead_hour", "beaufort_code"])


def plot_metrics(frame: pd.DataFrame) -> tuple[Path, Path, Path]:
    """Draw RMSE and MAE together: RMSE on the first row, MAE on the second."""
    metric_order = ("rmse", "mae")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.8, 7.2),
        sharex="col",
        sharey="row",
    )

    y_limits = {}
    for metric in metric_order:
        maximum = float(frame[[f"{name}_{metric}" for name in EXPERIMENTS]].max().max())
        y_limits[metric] = (0.0, maximum * 1.12 if maximum > 0 else 1.0)

    for row_index, metric in enumerate(metric_order):
        title, ylabel = METRICS[metric]
        for col_index, lead in enumerate(LEADS):
            ax = axes[row_index, col_index]
            sub = frame[frame["lead_hour"] == lead]
            for experiment in EXPERIMENTS:
                column = f"{experiment}_{metric}"
                ax.plot(
                    sub["beaufort_code"],
                    sub[column],
                    color=STYLES[experiment]["color"],
                    marker=STYLES[experiment]["marker"],
                    linewidth=1.35,
                    markersize=4.2,
                    label=STYLES[experiment]["label"],
                )
            panel = chr(ord("a") + row_index * len(LEADS) + col_index)
            ax.set_title(f"({panel}) {title} {lead}h预报", loc="left", fontweight="bold")
            ax.set_xlim(-0.25, len(BEAUFORT_ORDER) - 0.75)
            ax.set_ylim(*y_limits[metric])
            ax.set_xticks(range(len(BEAUFORT_ORDER)))
            ax.set_xticklabels(BEAUFORT_ORDER)
            ax.grid(True, color="#BFBFBF", linewidth=0.65, linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            if row_index == 1:
                ax.set_xlabel("观测蒲福风力等级")
            if col_index == 0:
                ax.set_ylabel(ylabel)
            for spine in ax.spines.values():
                spine.set_color("#444444")
                spine.set_linewidth(0.8)

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.99], pad=1.0, w_pad=1.2, h_pad=1.2)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{OUTPUT_STEM}.png"
    svg_path = FIGURES_DIR / f"{OUTPUT_STEM}.svg"
    pdf_path = FIGURES_DIR / f"{OUTPUT_STEM}.pdf"
    format_figure(fig)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path, pdf_path


def main() -> None:
    set_style()
    frame = load_data()
    png, svg, pdf = plot_metrics(frame)
    print(f"Combined RMSE/MAE: {png}")
    print(f"Combined RMSE/MAE SVG: {svg}")
    print(f"Combined RMSE/MAE PDF: {pdf}")


if __name__ == "__main__":
    main()
