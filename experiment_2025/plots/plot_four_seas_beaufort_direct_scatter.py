"""E0/G0 sample-pooled speed/direction MAE by sea and Beaufort group."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

try:
    from .publication_style import format_figure, set_style
except ImportError:
    from publication_style import format_figure, set_style


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("<=2", "3", "4", "5", "6", "7", ">=8", "overall")
SEAS = ((1, "渤海", "#008CFF"), (2, "黄海", "#00BB55"),
        (3, "东海", "#A83DFF"), (4, "南海北部", "#FF7900"))
EXPERIMENTS = (("era5_direct", "E0", "o"), ("gdas_direct", "G0", "^"))
OUTPUT_STEM = "four_seas_beaufort_wind_speed_direction_mae_E0_G0_2025"


def plot(data_root: Path, output_root: Path) -> list[Path]:
    frame = pd.read_csv(data_root / "four_seas_beaufort_wind_metrics_2025.csv")
    frame = frame.loc[frame["experiment"].isin([item[0] for item in EXPERIMENTS])].copy()
    if frame.duplicated(["sea_id", "beaufort_group", "experiment"]).any():
        raise ValueError("Duplicate sea/group/experiment rows")
    frame["beaufort_group"] = frame["beaufort_group"].replace(
        dict.fromkeys(("8", "9", "10", "11", ">=12"), ">=8")
    )
    positive = frame["paired_n"] > 0
    metrics = ("wind_speed_mae_ms", "wind_direction_mae_deg")
    if not np.isfinite(frame.loc[positive, list(metrics)].to_numpy()).all():
        raise ValueError("Nonfinite MAE for positive pair counts")
    for metric in metrics:
        frame[metric] = frame[metric].where(positive, 0) * frame["paired_n"]
    frame = frame.groupby(["sea_id", "beaufort_group", "experiment"], as_index=False).agg(
        paired_n=("paired_n", "sum"), wind_speed_mae_ms=("wind_speed_mae_ms", "sum"),
        wind_direction_mae_deg=("wind_direction_mae_deg", "sum")
    )
    for metric in metrics:
        frame[metric] = frame[metric] / frame["paired_n"].replace(0, np.nan)
    set_style()
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.8))
    x = np.arange(len(GROUPS))
    for experiment_index, (experiment, label, marker) in enumerate(EXPERIMENTS):
        for sea_index, (sea_id, sea_name, color) in enumerate(SEAS):
            subset = frame.loc[(frame["experiment"] == experiment) & (frame["sea_id"] == sea_id)]
            if set(subset["beaufort_group"]) != set(GROUPS):
                raise ValueError(f"Incomplete Beaufort rows for {sea_name}, {label}")
            subset = subset.set_index("beaufort_group").loc[list(GROUPS)]
            valid = subset["paired_n"].to_numpy() > 0
            # Small deterministic horizontal offsets separate nearby sea/experiment points.
            offset = (sea_index - 1.5) * 0.035 + (experiment_index - 0.5) * 0.16
            for ax, metric in zip(axes, metrics):
                mae = subset[metric].to_numpy()
                if not np.isfinite(mae[valid]).all() or (mae[valid] < 0).any():
                    raise ValueError(f"Invalid {metric} for {sea_name}, {label}")
                ax.scatter(x[valid] + offset, mae[valid], color=color, marker=marker,
                           s=38 if marker == "o" else 47, edgecolor="white", linewidth=0.35,
                           label=f"{sea_name}（{label}）", zorder=3)
    panels = (
        ("(a) 不同海域与风力等级的风速 MAE", "风速 MAE (m·s⁻¹)", 2),
        ("(b) 不同海域与风力等级的风向 MAE", "风向 MAE (°)", 10),
    )
    for ax, metric, (title, ylabel, interval) in zip(axes, metrics, panels):
        ax.set_xticks(x, ["≤2级", "3级", "4级", "5级", "6级", "7级", "≥8级", "平均"])
        maximum = frame.loc[frame["paired_n"] > 0, metric].max()
        ax.set(xlim=(-0.55, len(GROUPS) - 0.45), ylim=(0, maximum * 1.10),
               xlabel="观测蒲福风力等级", ylabel=ylabel)
        ax.set_title(title, loc="left", fontsize=11, pad=8)
        ax.yaxis.set_major_locator(MultipleLocator(interval))
        ax.yaxis.set_minor_locator(MultipleLocator(interval / 2))
        ax.grid(which="major", color="#BFBFBF", linewidth=0.65, alpha=0.8)
        ax.grid(which="minor", axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=0.95, hspace=0.40)
    format_figure(fig, legend_columns=2, legend_size=9)
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
