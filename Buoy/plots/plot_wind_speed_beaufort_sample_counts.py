from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from paths import FIGURES_DIR, WIND_MODEL_STATISTICS_DIR


FONT_SCALE = 1.0
FONT_FAMILY = ["Times New Roman", "SimSun", "SimHei", "Microsoft YaHei", "DejaVu Serif"]
TEXT_LABELS = {
    "observed_beaufort": "观测蒲福风力等级",
    "valid_sample_count": "有效样本数",
    "lead_panel": "({panel}) {lead_hour} h预报",
}
BASE_FONT_SIZES = {
    "default": 14,
    "title": 15,
    "axis_label": 13,
    "tick": 12,
    "bar_label": 11,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}

METRICS_CSV = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h" / "wind_speed_metrics_by_beaufort.csv"
OUT_PNG = FIGURES_DIR / "wind_speed_beaufort_sample_counts_24_48_72h.png"
OUT_SVG = FIGURES_DIR / "wind_speed_beaufort_sample_counts_24_48_72h.svg"

LEAD_HOURS = [24, 48, 72]
BEAUFORT_ORDER = ["<=2", "3", "4", "5", "6", "7", ">=8"]
BEAUFORT_TO_CODE = {label: index for index, label in enumerate(BEAUFORT_ORDER)}
BEAUFORT_COLORS = [
    "#9EC5E8",
    "#A8DDB5",
    "#F4C99B",
    "#C7B7E8",
    "#F2A7A0",
    "#B7D7D8",
    "#D8C5A5",
]
BEAUFORT_EDGE_COLORS = [
    "#6F9FC9",
    "#79B98A",
    "#D9A86F",
    "#9A87C5",
    "#D37D76",
    "#83B5B7",
    "#B49A75",
]

DATASETS = {"era5_realtime", "era5_lagged_5d", "gdas_forecast"}


def load_sample_counts(csv_path: Path = METRICS_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Beaufort metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"dataset", "lead_hour", "obs_beaufort_group", "n"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df[df["dataset"].isin(DATASETS)].copy()
    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["obs_beaufort_group"] = df["obs_beaufort_group"].astype(str)
    df["beaufort_code"] = df["obs_beaufort_group"].map(BEAUFORT_TO_CODE)

    df = df[
        df["lead_hour"].isin(LEAD_HOURS)
        & df["beaufort_code"].notna()
    ].dropna(subset=["n"])
    df["lead_hour"] = df["lead_hour"].astype(int)
    df["beaufort_code"] = df["beaufort_code"].astype(int)
    df["n"] = df["n"].astype(int)

    conflicts = (
        df.groupby(["lead_hour", "obs_beaufort_group"])["n"]
        .nunique()
        .reset_index(name="distinct_n")
    )
    conflicts = conflicts[conflicts["distinct_n"] > 1]
    if not conflicts.empty:
        raise ValueError(
            "Sample counts differ across datasets for the same lead/Beaufort group: "
            f"{conflicts.to_dict(orient='records')}"
        )

    counts = (
        df[["lead_hour", "obs_beaufort_group", "beaufort_code", "n"]]
        .drop_duplicates()
        .sort_values(["lead_hour", "beaufort_code"])
        .reset_index(drop=True)
    )
    return counts


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


def style_axis(ax, show_ylabel: bool = True) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color="#BFBFBF", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.set_xticks(range(len(BEAUFORT_ORDER)))
    ax.set_xticklabels(BEAUFORT_ORDER)
    ax.set_xlabel(TEXT_LABELS["observed_beaufort"])
    ax.set_ylabel(TEXT_LABELS["valid_sample_count"] if show_ylabel else "")


def add_bar_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZES["bar_label"],
            color="#333333",
        )


def plot_sample_counts(counts: pd.DataFrame) -> None:
    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(LEAD_HOURS), figsize=(13.8, 4.7), constrained_layout=False)
    if len(LEAD_HOURS) == 1:
        axes = [axes]

    max_n = int(counts["n"].max()) if not counts.empty else 1
    for panel_index, (ax, lead_hour) in enumerate(zip(axes, LEAD_HOURS)):
        sub = counts[counts["lead_hour"] == lead_hour].sort_values("beaufort_code")
        bar_colors = [BEAUFORT_COLORS[index] for index in sub["beaufort_code"]]
        edge_colors = [BEAUFORT_EDGE_COLORS[index] for index in sub["beaufort_code"]]
        bars = ax.bar(
            sub["beaufort_code"],
            sub["n"],
            width=0.68,
            color=bar_colors,
            edgecolor=edge_colors,
            linewidth=0.75,
        )
        add_bar_labels(ax, bars)
        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            TEXT_LABELS["lead_panel"].format(panel=panel_letter, lead_hour=lead_hour),
            loc="left",
            fontweight="bold",
        )
        ax.set_ylim(0, max_n * 1.16)
        style_axis(ax, show_ylabel=panel_index == 0)

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.98])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    counts = load_sample_counts()
    plot_sample_counts(counts)
    print(f"Input: {METRICS_CSV}")
    print(f"Lead hours: {LEAD_HOURS}")
    print(f"PNG: {OUT_PNG}")
    print(f"SVG: {OUT_SVG}")


if __name__ == "__main__":
    main()
