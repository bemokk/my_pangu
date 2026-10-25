"""Plot lead-time wind verification metrics in the supplied three-panel style."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "Statistics" / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "figures"

EXPERIMENTS = (
    ("era5_direct", "E0", "#43A3EF"),
    ("era5_lagged_120h", "E120", "#FEA040"),
    ("gdas_direct", "G0", "#009E73"),
    ("gdas_lagged_120h", "G120", "#EF767B"),
)
PANELS = (("rmse", "RMSE"), ("mae", "MAE"), ("cc", "CC"))


def load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"experiment", "lead_hour", "rmse", "mae", "cc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["lead_hour"] = pd.to_numeric(frame["lead_hour"], errors="raise").astype(int)
    frame = frame[frame["experiment"].isin([item[0] for item in EXPERIMENTS])].copy()
    expected_leads = set(range(0, 121, 3))
    actual_leads = set(frame["lead_hour"])
    if actual_leads != expected_leads:
        raise ValueError(f"{path} lead hours differ from 0..120 by 3 h: {sorted(actual_leads)}")
    if frame.duplicated(["experiment", "lead_hour"]).any():
        raise ValueError(f"{path} contains duplicate experiment/lead rows")
    return frame.sort_values(["lead_hour", "experiment"])


def y_label(metric: str, variable: str) -> str:
    if metric == "cc":
        return "CC"
    return f"{metric} ({'m s$^{-1}$' if variable == 'wind_speed' else '°'})"


def plot_metrics(frame: pd.DataFrame, variable: str, title: str, output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7.2), sharex=True)
    fig.patch.set_facecolor("white")
    leads = sorted(frame["lead_hour"].unique())

    for index, (metric, panel_title) in enumerate(PANELS):
        ax = axes[index]
        for experiment, label, color in EXPERIMENTS:
            subset = frame[frame["experiment"] == experiment].sort_values("lead_hour")
            ax.plot(
                subset["lead_hour"],
                subset[metric],
                color=color,
                linewidth=1.15,
                marker="o",
                markersize=2.7,
                markeredgewidth=0.25,
                markeredgecolor="white",
                label=label,
            )
        ax.set_title(panel_title, fontsize=10, pad=3)
        ax.set_ylabel(y_label(metric, variable), fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.7, color="#BFC4C8", alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8, length=3, width=0.7, colors="#303030")
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
            spine.set_color("#606060")
        ax.text(0.006, 0.98, f"({chr(ord('a') + index)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9)
        if metric == "cc":
            lo = float(frame[metric].min())
            hi = float(frame[metric].max())
            ax.set_ylim(max(0.0, min(0.45, lo - 0.03)), min(1.0, max(0.9, hi + 0.03)))

    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.015, 0.99),
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=2.1,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    axes[-1].set_xlabel("Forecast lead time (h)", fontsize=9)
    axes[-1].set_xticks(list(range(0, 121, 12)))
    axes[-1].set_xlim(-2, 122)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.085, hspace=0.18)

    stem = f"{variable}_metrics_by_lead"
    outputs = [output_root / f"{stem}.png", output_root / f"{stem}.svg", output_root / f"{stem}.pdf"]
    fig.savefig(outputs[0], dpi=300, facecolor="white")
    fig.savefig(outputs[1], facecolor="white")
    fig.savefig(outputs[2], dpi=300, facecolor="white")
    plt.close(fig)
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    outputs = []
    for variable, title in (("wind_speed", "10 m wind speed"), ("wind_direction", "10 m wind direction")):
        frame = load_metrics(args.data_root / f"{variable}_metrics_by_lead.csv")
        outputs.extend(plot_metrics(frame, variable, title, args.output_root))
    print("Generated:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
