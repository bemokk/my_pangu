"""Section 3.1: lead-time RMSE/MAE and sample-pooled Beaufort MAE.

Panel (c) uses the supplied overall rows, which pool pairs across 0--120 h.
The >=8 category combines groups using pair counts, not equal group weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .publication_style import format_figure, set_style
    from .plot_metrics import load_metrics
    from .plot_wind_speed_beaufort_metrics import BEAUFORT_ORDER, EXPERIMENTS, STYLES
except ImportError:
    from publication_style import format_figure, set_style
    from plot_metrics import load_metrics
    from plot_wind_speed_beaufort_metrics import BEAUFORT_ORDER, EXPERIMENTS, STYLES


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_IDS = {
    "E0": "era5_direct",
    "G0": "gdas_direct",
    "E120": "era5_lagged_120h",
    "G120": "gdas_lagged_120h",
}
HIGH_GROUPS = {"8", "9", "10", "11", ">=12"}
OUTPUT_STEM = "overall_wind_speed_verification_2025"


def load_beaufort_mae(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    overall = frame.loc[frame["aggregation"] == "overall"].copy()
    expected = set(BEAUFORT_ORDER[:-1]) | HIGH_GROUPS
    if set(overall["beaufort_group"]) != expected or overall["beaufort_group"].duplicated().any():
        raise ValueError("Expected one overall row per original Beaufort group")
    overall["group"] = overall["beaufort_group"].replace(dict.fromkeys(HIGH_GROUPS, ">=8"))
    rows = []
    for group in BEAUFORT_ORDER:
        subset = overall.loc[overall["group"] == group]
        row = {"beaufort_group": group}
        for experiment in EXPERIMENTS:
            counts = subset[f"n_{experiment}"].to_numpy(dtype=float)
            values = subset[f"{experiment}_mae"].to_numpy(dtype=float)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError(f"Invalid pair counts for {experiment}, {group}")
            valid = counts > 0
            if not valid.any() or not np.isfinite(values[valid]).all() or (values[valid] < 0).any():
                raise ValueError(f"Missing or invalid MAE for {experiment}, {group}")
            row[f"{experiment}_mae"] = np.average(values[valid], weights=counts[valid])
        rows.append(row)
    return pd.DataFrame(rows)


def plot(data_root: Path, output_root: Path) -> list[Path]:
    speed = load_metrics(data_root / "wind_speed_metrics_by_lead.csv")
    for experiment_id in EXPERIMENT_IDS.values():
        subset = speed.loc[speed["experiment"] == experiment_id]
        if set(subset["lead_hour"]) != set(range(0, 121, 3)):
            raise ValueError(f"Incomplete 0--120 h series for {experiment_id}")
        if not np.isfinite(subset[["rmse", "mae"]].to_numpy()).all():
            raise ValueError(f"Nonfinite errors for {experiment_id}")
    beaufort = load_beaufort_mae(data_root / "wind_speed_beaufort_mae_0_120_2025.csv")
    set_style()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 8.4))
    for ax, metric in zip(axes[:2], ("rmse", "mae")):
        for experiment in EXPERIMENTS:
            subset = speed.loc[speed["experiment"] == EXPERIMENT_IDS[experiment]]
            style = STYLES[experiment]
            ax.plot(subset["lead_hour"], subset[metric], label=experiment,
                    color=style["color"], marker=style["marker"], markersize=2.5,
                    markevery=2, linewidth=1.3,
                    linestyle="--" if "120" in experiment else "-")
        ax.set(xlim=(0, 120), xlabel="预报时效（h）", ylabel=f"{metric.upper()} (m·s⁻¹)")
        ax.set_xticks(range(0, 121, 12))
        lo, hi = speed[metric].min(), speed[metric].max()
        ax.set_ylim(max(0, lo - (hi - lo) * 0.12), hi + (hi - lo) * 0.40)

    x = np.arange(len(BEAUFORT_ORDER))
    width = 0.19
    for index, experiment in enumerate(EXPERIMENTS):
        axes[2].bar(x + (index - 1.5) * width, beaufort[f"{experiment}_mae"],
                    width=width, color=STYLES[experiment]["color"],
                    edgecolor="white", linewidth=0.35, label=experiment,
                    hatch="//" if "120" in experiment else None)
    axes[2].set_xticks(x, ["≤2", "3", "4", "5", "6", "7", "≥8"])
    axes[2].set(xlabel="观测蒲福风力等级", ylabel="MAE (m·s⁻¹)",
                ylim=(0, beaufort[[f"{e}_mae" for e in EXPERIMENTS]].max().max() * 1.12))

    panel_names = (
        "(a) 风速 RMSE",
        "(b) 风速 MAE",
        "(c) 不同风力等级的风速 MAE（0–120 h平均）",
    )
    for ax, panel_name in zip(axes, panel_names):
        ax.set_title(panel_name, loc="left", fontsize=11, pad=6)
        ax.grid(axis="y", color="#BFBFBF", linewidth=0.6, linestyle="--", alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3, width=0.8)
        for spine in ax.spines.values():
            spine.set_color("black")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.07, top=0.96, hspace=0.46)
    format_figure(fig, legend_columns=4, legend_size=9)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [output_root / f"{OUTPUT_STEM}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(beaufort.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Statistics" / "data")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    for path in plot(args.data_root, args.output_root):
        print(path)


if __name__ == "__main__":
    main()
