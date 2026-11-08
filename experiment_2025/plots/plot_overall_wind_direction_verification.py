"""Plot lead-time direction errors and monthly sample-pooled 0--120 h MAE."""
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
    from .plot_overall_wind_speed_verification import EXPERIMENT_IDS, ROOT
    from .plot_wind_speed_beaufort_metrics import EXPERIMENTS, STYLES
except ImportError:
    from publication_style import format_figure, set_style
    from plot_metrics import load_metrics
    from plot_overall_wind_speed_verification import EXPERIMENT_IDS, ROOT
    from plot_wind_speed_beaufort_metrics import EXPERIMENTS, STYLES


OUTPUT_STEM = "overall_wind_direction_verification_2025"


def plot(data_root: Path, output_root: Path) -> list[Path]:
    direction = load_metrics(data_root / "wind_direction_metrics_by_lead.csv")
    monthly = pd.read_csv(data_root / "wind_direction_metrics_by_month.csv")
    monthly = monthly.loc[monthly["experiment"].isin(EXPERIMENT_IDS.values())].copy()
    monthly["month"] = pd.to_numeric(monthly["month"], errors="raise")
    if monthly.duplicated(["experiment", "month"]).any():
        raise ValueError("Duplicate experiment/month rows")
    for experiment_id in EXPERIMENT_IDS.values():
        leads = direction.loc[direction["experiment"] == experiment_id]
        months = monthly.loc[monthly["experiment"] == experiment_id]
        if set(leads["lead_hour"]) != set(range(0, 121, 3)) or set(months["month"]) != set(range(1, 13)):
            raise ValueError(f"Incomplete lead-time or monthly data for {experiment_id}")
        if not np.isfinite(leads[["rmse", "mae", "n"]].to_numpy()).all():
            raise ValueError(f"Nonfinite lead-time values for {experiment_id}")
        if not np.isfinite(months[["mae", "n"]].to_numpy()).all() or (months["n"] <= 0).any():
            raise ValueError(f"Invalid monthly values for {experiment_id}")
        # Both tables must describe the same set of pooled observation pairs.
        if leads["n"].sum() != months["n"].sum() or not np.isclose(
            np.average(leads["mae"], weights=leads["n"]),
            np.average(months["mae"], weights=months["n"]), rtol=0, atol=1e-8
        ):
            raise ValueError(f"Monthly and lead-time aggregation disagree for {experiment_id}")

    set_style()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 8.4))
    panels = (
        (direction, "lead_hour", "rmse", "RMSE (°)", "预报时效（h）", "(a) 风向 RMSE"),
        (direction, "lead_hour", "mae", "MAE (°)", "预报时效（h）", "(b) 风向 MAE"),
        (monthly, "month", "mae", "平均 MAE (°)", "月份", "(c) 各月风向 MAE（0–120 h平均）"),
    )
    for ax, (frame, xcolumn, metric, ylabel, xlabel, title) in zip(axes, panels):
        for experiment in EXPERIMENTS:
            subset = frame.loc[frame["experiment"] == EXPERIMENT_IDS[experiment]].sort_values(xcolumn)
            style = STYLES[experiment]
            ax.plot(subset[xcolumn], subset[metric], label=experiment,
                    color=style["color"], marker=style["marker"], markersize=3,
                    linewidth=1.3, linestyle="--" if "120" in experiment else "-")
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title, loc="left", fontsize=11, pad=6)
        lo, hi = frame[metric].min(), frame[metric].max()
        padding = max(float(hi - lo), 1.0)
        ax.set_ylim(max(0, lo - padding * 0.12), hi + padding * 0.40)
        ax.grid(color="#BFBFBF", linewidth=0.6, linestyle="--", alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3, width=0.8)
        for spine in ax.spines.values():
            spine.set_color("black")
    for ax in axes[:2]:
        ax.set_xlim(0, 120)
        ax.set_xticks(range(0, 121, 12))
    axes[2].set_xlim(0.7, 12.3)
    axes[2].set_xticks(range(1, 13))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.07, top=0.96, hspace=0.46)
    format_figure(fig, legend_columns=4, legend_size=9)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = [output_root / f"{OUTPUT_STEM}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
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
