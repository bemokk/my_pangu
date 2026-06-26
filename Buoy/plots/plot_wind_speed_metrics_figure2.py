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


FONT_SCALE = 1
FONT_FAMILY = ["Times New Roman", "SimSun", "SimHei", "Microsoft YaHei", "DejaVu Serif"]
TEXT_LABELS = {
    "era5_realtime": "ERA5实时场",
    "era5_lagged_5d": "ERA5延迟5天预报",
    "gdas_forecast": "GDAS实时预报",
    "lead_time": "预报时效（h）",
    "correlation": "相关系数",
}
BASE_FONT_SIZES = {
    "default": 14,
    "title": 15,
    "axis_label": 13,
    "legend": 12,
    "tick": 12,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}

METRICS_CSV = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h" / "wind_speed_metrics_by_lead.csv"
OUT_PNG = FIGURES_DIR / "wind_speed_metrics_figure2_style.png"
OUT_SVG = FIGURES_DIR / "wind_speed_metrics_figure2_style.svg"

LEAD_HOURS = list(range(0, 73, 3))
X_TICKS = [0, 12, 24, 36, 48, 60, 72]

DATASET_STYLES = {
    "era5_realtime": {
        "label": TEXT_LABELS["era5_realtime"],
        "color": "#9E2F33",
        "marker": "o",
        "linestyle": "-",
    },
    "era5_lagged_5d": {
        "label": TEXT_LABELS["era5_lagged_5d"],
        "color": "#244C8F",
        "marker": "s",
        "linestyle": "-",
    },
    "gdas_forecast": {
        "label": TEXT_LABELS["gdas_forecast"],
        "color": "#2F7D45",
        "marker": "^",
        "linestyle": "-",
    },
}

PLOT_METRICS = [
    ("rmse", "(a) RMSE", "RMSE (m s$^{-1}$)"),
    ("mae", "(b) MAE", "MAE (m s$^{-1}$)"),
    ("corr", "(c) CC", TEXT_LABELS["correlation"]),
]

Y_LIMITS = {
    "rmse": (None, 4.6),
}


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


def style_axis(ax, ylabel: str) -> None:
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
            linewidth=1.35,
            linestyle=style["linestyle"],
        )

    ax.set_title(title, loc="left", fontweight="bold")
    style_axis(ax, ylabel)
    if metric in Y_LIMITS:
        ax.set_ylim(*Y_LIMITS[metric])


def make_figure(df: pd.DataFrame) -> None:
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.6), constrained_layout=False)
    for ax, (metric, title, ylabel) in zip(axes, PLOT_METRICS):
        plot_metric(ax, df, metric, title, ylabel)

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
    fig.tight_layout(rect=[0.04, 0.02, 0.98, 0.99])

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
