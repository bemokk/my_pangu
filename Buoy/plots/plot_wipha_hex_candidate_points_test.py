from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from land_mask import load_land_union
from paths import FIGURES_DIR
from plots.plot_spatial_hex_best_rmse import HEX_SIDE_DEG, regular_hexagon
from plots.wipha_case_common import set_plot_style


OUT_PNG = FIGURES_DIR / "wipha_hex_candidate_points_test.png"
OUT_SVG = FIGURES_DIR / "wipha_hex_candidate_points_test.svg"

CANDIDATE_HEXES = [
    {
        "hex_id": 204,
        "center_lon": 119.0,
        "center_lat": 23.392305,
        "target": "120E,20N",
        "kind": "primary",
        "records_per_dataset": 11,
    },
    {
        "hex_id": 203,
        "center_lon": 119.0,
        "center_lat": 21.660254,
        "target": "120E,20N",
        "kind": "backup",
        "records_per_dataset": 10,
    },
    {
        "hex_id": 186,
        "center_lon": 117.5,
        "center_lat": 22.526279,
        "target": "115E,21N",
        "kind": "primary",
        "records_per_dataset": 7,
    },
    {
        "hex_id": 167,
        "center_lon": 116.0,
        "center_lat": 21.660254,
        "target": "115E,21N",
        "kind": "backup",
        "records_per_dataset": 6,
    },
]

TARGET_POINTS = [
    {"label": "Target A 120E,20N", "lon": 120.0, "lat": 20.0},
    {"label": "Target B 115E,21N", "lon": 115.0, "lat": 21.0},
]


def plot_candidate_points() -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.8, 6.6))
    ax = plt.axes(projection=projection)
    ax.set_extent([109.5, 125.5, 14.5, 24.8], crs=projection)
    ax.set_facecolor("#EAF3F8")

    land_union = load_land_union(109.5, 14.5, 125.5, 24.8)
    ax.add_geometries(
        [land_union],
        crs=projection,
        facecolor="#D7D2C3",
        edgecolor="#777777",
        linewidth=0.35,
        zorder=1,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.48, edgecolor="#333333", zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.25, edgecolor="#777777", zorder=2)

    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.3,
        color="#777777",
        alpha=0.38,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    search_box = Rectangle(
        (110.0, 15.0),
        15.0,
        9.0,
        fill=False,
        edgecolor="#222222",
        linewidth=1.0,
        linestyle="--",
        transform=projection,
        zorder=4,
    )
    ax.add_patch(search_box)
    ax.text(
        110.25,
        23.65,
        "Expanded search range\n110-125E, 15-24N",
        fontsize=8.0,
        color="#222222",
        transform=projection,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        zorder=8,
    )

    style_by_kind = {
        "primary": {"color": "#C44E52", "marker": "o", "label": "Primary choice"},
        "backup": {"color": "#4C72B0", "marker": "^", "label": "Backup choice"},
    }

    used_labels = set()
    for candidate in CANDIDATE_HEXES:
        style = style_by_kind[candidate["kind"]]
        hexagon = regular_hexagon(candidate["center_lon"], candidate["center_lat"], HEX_SIDE_DEG)
        ax.add_geometries(
            [hexagon],
            crs=projection,
            facecolor=style["color"],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.18,
            zorder=3,
        )
        label = style["label"] if style["label"] not in used_labels else None
        used_labels.add(style["label"])
        ax.scatter(
            candidate["center_lon"],
            candidate["center_lat"],
            s=70,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.7,
            transform=projection,
            zorder=5,
            label=label,
        )
        text = (
            f"{candidate['target']} {candidate['kind']}\n"
            f"hex {candidate['hex_id']}\n"
            f"{candidate['center_lon']:.1f}E, {candidate['center_lat']:.2f}N\n"
            f"N={candidate['records_per_dataset']}/dataset"
        )
        ax.text(
            candidate["center_lon"] + 0.12,
            candidate["center_lat"] + 0.12,
            text,
            fontsize=7.8,
            color="#222222",
            transform=projection,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
            zorder=6,
        )

    for target in TARGET_POINTS:
        ax.scatter(
            target["lon"],
            target["lat"],
            s=95,
            marker="*",
            color="#DD8452",
            edgecolor="black",
            linewidth=0.7,
            transform=projection,
            zorder=7,
            label="Target locations" if target["label"].endswith("20N") else None,
        )
        ax.text(
            target["lon"] + 0.12,
            target["lat"] - 0.32,
            target["label"],
            fontsize=8.0,
            color="#6B3A16",
            fontweight="bold",
            transform=projection,
            zorder=8,
        )

    ax.set_title("Typhoon Wipha Hexagon Candidate Points", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.text(
        0.5,
        0.018,
        "Hex centers are candidates inside 110-125E, 15-24N for 2025-07-18 00 UTC forecast verification; shaded outlines show 1-degree hexagons.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.98])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_candidate_points()
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
