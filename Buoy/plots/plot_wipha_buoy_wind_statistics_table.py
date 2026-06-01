from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from plots.wipha_case_common import (
    OUT_STATS_CSV,
    OUT_STATS_TABLE_PNG,
    OUT_STATS_TABLE_SVG,
    OUT_STATS_XLSX,
    compute_statistics,
    ensure_dirs,
    prepare_buoy_case_data,
    set_plot_style,
)


def plot_statistics_table(stats: pd.DataFrame) -> None:
    set_plot_style()
    display_cols = [
        "platform_id",
        "dataset_label",
        "sample_count",
        "speed_mae",
        "speed_rmse",
        "speed_corr",
        "direction_mae",
        "direction_rmse",
    ]
    table = stats[display_cols].copy()
    table.columns = ["Platform", "Dataset", "N", "Speed MAE", "Speed RMSE", "Speed CC", "Dir MAE", "Dir RMSE"]
    for col in ["Speed MAE", "Speed RMSE", "Speed CC", "Dir MAE", "Dir RMSE"]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    fig_height = max(2.6, 0.42 * len(table) + 1.2)
    fig, ax = plt.subplots(figsize=(11.4, fig_height))
    ax.axis("off")
    ax.set_title("Typhoon Wipha Case Wind Verification Statistics", loc="left", fontweight="bold", pad=12)
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.0, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F5F7")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT_STATS_TABLE_PNG, bbox_inches="tight")
    fig.savefig(OUT_STATS_TABLE_SVG, bbox_inches="tight")
    plt.close(fig)


def generate() -> list[Path]:
    ensure_dirs()
    _, merged, _ = prepare_buoy_case_data()
    stats = compute_statistics(merged)
    plot_statistics_table(stats)
    return [OUT_STATS_CSV, OUT_STATS_XLSX, OUT_STATS_TABLE_PNG, OUT_STATS_TABLE_SVG]


def main() -> None:
    for path in generate():
        print(path if path.exists() else f"{path} (not written)")


if __name__ == "__main__":
    main()
