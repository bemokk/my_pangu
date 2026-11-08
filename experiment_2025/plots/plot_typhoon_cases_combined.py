"""Combine a centered 60%-size track map with three rows of station wind panels."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

try:
    from .publication_style import set_style
    from .plot_typhoon_tracks_stations import plot as plot_tracks
    from .plot_typhoon_station_wind_timeseries import plot_figure as plot_winds
except ImportError:
    from publication_style import set_style
    from plot_typhoon_tracks_stations import plot as plot_tracks
    from plot_typhoon_station_wind_timeseries import plot_figure as plot_winds

ROOT = Path(__file__).resolve().parents[1]


def plot(data_root, figures_root):
    set_style()
    frame = pd.read_csv(data_root / "typhoon_three_cases_chronological_beijing.csv")
    # Reduce the map's physical width and height by 40%, removing freed row height.
    original_height = 18.8
    map_scale = .6
    figure_height = original_height * (1 - .36 * (1 - map_scale))
    fig = plt.figure(figsize=(13, figure_height))
    grid = fig.add_gridspec(3, 2,
                           left=.065, right=.985,
                           bottom=.04 * original_height / figure_height,
                           top=.55 * original_height / figure_height,
                           hspace=.46, wspace=.18)
    map_width = .92 * map_scale
    map_left = (.065 + .985 - map_width) / 2
    map_ax = fig.add_axes([map_left, .61 * original_height / figure_height,
                          map_width, .36 * map_scale * original_height / figure_height])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(2)]
                     for row in range(3)])
    plot_winds(frame, "beijing", figures_root, fig=fig, axes=axes, panel_offset=1,
               show_initialization=False, experiments=("gdas_direct", "era5_lagged_120h"))
    plot_tracks(data_root, figures_root, ax=map_ax)
    # Keep labels and the inset legend legible without crowding the smaller map.
    for text in map_ax.findobj(Text):
        text.set_fontsize(text.get_fontsize() * .75)
    map_ax.set_title("(a) 台风路径与海洋站分布", loc="left", fontweight="bold", fontsize=11, pad=7)
    figures_root.mkdir(parents=True, exist_ok=True)
    stem = figures_root / "typhoon_three_cases_combined_beijing"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=600,
                    bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Statistics" / "data")
    parser.add_argument("--figures-root", type=Path, default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    print(plot(args.data_root, args.figures_root))


if __name__ == "__main__":
    main()
