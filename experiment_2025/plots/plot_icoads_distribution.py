"""Plot the annual ICOADS spatial observation distribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:  # pragma: no cover - environment-dependent
    HAS_CARTOPY = False


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "Statistics" / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "figures"
LAND_BOUNDARY_FILE = Path(__file__).resolve().with_name("land_boundary_13_103_42_130.geojson")
AREA = [42.0, 103.0, 13.0, 130.0]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA


def load_data(data_root: Path):
    grid = pd.read_csv(data_root / "icoads_2025_grid_counts_1deg.csv")
    summary = json.loads((data_root / "icoads_distribution_run_summary.json").read_text(encoding="utf-8"))
    return grid, summary


def add_fallback_land_boundary(ax) -> None:
    """Draw and fill cached Natural Earth land polygons when Cartopy is unavailable."""
    if not LAND_BOUNDARY_FILE.exists():
        return
    geometry = json.loads(LAND_BOUNDARY_FILE.read_text(encoding="utf-8"))["geometry"]
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    for polygon in polygons:
        vertices = []
        codes = []
        for ring in polygon:
            points = np.asarray(ring, dtype=float)
            vertices.extend(points.tolist())
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 2) + [MplPath.CLOSEPOLY])
        ax.add_patch(
            PathPatch(
                MplPath(vertices, codes),
                facecolor="#E8E8E8",
                edgecolor="#4d4d4d",
                linewidth=0.48,
                zorder=5,
            )
        )


def plot_spatial(grid: pd.DataFrame, summary: dict, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    grid_size = float(summary["grid_size_deg"])
    lon_edges = np.arange(LON_MIN, LON_MAX + grid_size + 1e-9, grid_size)
    lat_edges = np.arange(LAT_MIN, LAT_MAX + grid_size + 1e-9, grid_size)
    counts = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1), dtype=float)
    for row in grid.itertuples():
        i, j = int(row.grid_lat_index), int(row.grid_lon_index)
        if 0 <= i < counts.shape[0] and 0 <= j < counts.shape[1]:
            counts[i, j] = float(row.record_count)
    nonzero = counts[counts > 0]
    norm = mcolors.LogNorm(vmin=1, vmax=max(1, float(nonzero.max()) if nonzero.size else 1))
    cmap = plt.get_cmap("YlOrRd")

    projection = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig = plt.figure(figsize=(9.6, 8.0))
    ax = plt.axes(projection=projection) if HAS_CARTOPY else plt.axes()
    if HAS_CARTOPY:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=projection)
        mesh = ax.pcolormesh(lon_edges, lat_edges, counts, cmap=cmap, norm=norm, shading="flat", transform=projection, alpha=0.88)
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#E8E8E8", edgecolor="#555555", linewidth=0.45, zorder=3)
        ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#dfeef7", edgecolor="#7a9aaa", linewidth=0.3, zorder=4)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.55, edgecolor="#333333", zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.25, edgecolor="#777777", zorder=5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#777777", alpha=0.38, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
    else:
        mesh = ax.pcolormesh(lon_edges, lat_edges, counts, cmap=cmap, norm=norm, shading="flat", alpha=0.88)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, linestyle="--", linewidth=0.35, color="#777777", alpha=0.38)
        add_fallback_land_boundary(ax)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.76, pad=0.025)
    cbar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.055, right=0.94, bottom=0.07, top=0.93)
    output = output_root / "icoads_2025_distribution"
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    return output.with_suffix(".png")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    grid, summary = load_data(args.data_root)
    outputs = [plot_spatial(grid, summary, args.output_root)]
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
