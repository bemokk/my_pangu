import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, box

from land_mask import filter_ocean_records, load_land_union
from paths import CHINA_SEA_RECORDS_DIR, DEFAULT_CHINA_SEA_DETAIL_CSV, FIGURES_DIR


FONT_SCALE = 1.25
FONT_FAMILY = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
TEXT_LABELS = {
    "colorbar": "每个六边形网格的记录数",
}
BASE_FONT_SIZES = {
    "default": 10,
    "axis_label": 10,
    "tick": 9,
    "annotation": 8,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}

DETAIL_CSV = DEFAULT_CHINA_SEA_DETAIL_CSV
# AREA follows the common meteorological order: [lat_max, lon_min, lat_min, lon_max].
AREA = [42, 103, 13, 130]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA

START_DATE = pd.Timestamp("2025-07-01 00:00:00")
END_DATE = pd.Timestamp("2025-08-04 23:59:59")

# Configurable regular hexagon side length, in degrees.
HEX_SIDE_DEG = 1.0
HEX_SIDE_LABEL = f"side{str(HEX_SIDE_DEG).replace('.', 'p')}deg"
HEX_COUNTS_CSV = CHINA_SEA_RECORDS_DIR / f"china_sea_hex_counts_area_42_103_13_130_{HEX_SIDE_LABEL}.csv"
HEX_COUNTS_PNG = FIGURES_DIR / f"china_sea_hex_counts_area_42_103_13_130_{HEX_SIDE_LABEL}.png"

# If True, cells with zero records are labeled as 0. This is usually too dense for 1-degree grids.
LABEL_ZERO_CELLS = False
LABEL_EDGE_MARGIN_DEG = 0.18 * HEX_SIDE_DEG
HEX_LABEL_FONT_SIZE = FONT_SIZES["annotation"] if HEX_SIDE_DEG <= 1.0 else 9 * FONT_SCALE
HEX_LABEL_FONT_COLOR = "black"


def regular_hexagon(center_lon: float, center_lat: float, side_deg: float) -> Polygon:
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    coords = [
        (center_lon + side_deg * np.cos(angle), center_lat + side_deg * np.sin(angle))
        for angle in angles
    ]
    return Polygon(coords)


def generate_hex_grid(area_polygon: Polygon, side_deg: float) -> pd.DataFrame:
    dx = 1.5 * side_deg
    dy = np.sqrt(3.0) * side_deg

    min_lon, min_lat, max_lon, max_lat = area_polygon.bounds
    centers = []
    col = 0
    lon = min_lon - 2 * side_deg

    while lon <= max_lon + 2 * side_deg:
        lat_offset = dy / 2.0 if col % 2 else 0.0
        lat = min_lat - 2 * dy + lat_offset

        while lat <= max_lat + 2 * dy:
            hexagon = regular_hexagon(lon, lat, side_deg)
            if hexagon.intersects(area_polygon):
                centers.append(
                    {
                        "hex_id": len(centers),
                        "center_lon": lon,
                        "center_lat": lat,
                        "geometry": hexagon,
                    }
                )
            lat += dy

        lon += dx
        col += 1

    return pd.DataFrame(centers)


def load_valid_records() -> pd.DataFrame:
    if not DETAIL_CSV.exists():
        raise FileNotFoundError(f"Detail CSV not found: {DETAIL_CSV}")

    records = pd.read_csv(DETAIL_CSV)
    records["datetime_utc"] = pd.to_datetime(records["datetime_utc"], errors="coerce")

    records = records.dropna(
        subset=["datetime_utc", "latitude", "longitude", "wind_dir_deg", "wind_speed_ms"]
    ).copy()

    records = records[
        records["datetime_utc"].between(START_DATE, END_DATE)
        & records["latitude"].between(LAT_MIN, LAT_MAX)
        & records["longitude"].between(LON_MIN, LON_MAX)
        & records["wind_dir_deg"].between(1, 360)
        & records["wind_speed_ms"].between(0, 75)
    ].copy()
    records, dropped_land_count = filter_ocean_records(
        records,
        lon_min=LON_MIN,
        lat_min=LAT_MIN,
        lon_max=LON_MAX,
        lat_max=LAT_MAX,
    )
    records.attrs["dropped_land_count"] = dropped_land_count

    return records


def count_records_by_hex(records: pd.DataFrame, hexes: pd.DataFrame) -> pd.DataFrame:
    points = records[["longitude", "latitude"]].to_numpy(dtype=float)
    counts = []

    for _, row in hexes.iterrows():
        coords = np.asarray(row.geometry.exterior.coords)
        path = MplPath(coords)
        counts.append(int(path.contains_points(points, radius=1e-9).sum()))

    hexes = hexes.copy()
    hexes["record_count"] = counts
    return hexes


def save_hex_counts(hexes: pd.DataFrame) -> None:
    CHINA_SEA_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    out = hexes.drop(columns="geometry").copy()
    out.to_csv(HEX_COUNTS_CSV, index=False, encoding="utf-8-sig")


def plot_hex_counts(hexes: pd.DataFrame, land_union, ocean_area, records: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT_SIZES["default"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "axes.unicode_minus": False,
        }
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13.5, 13.0))
    ax = plt.axes(projection=projection)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=projection)

    ax.set_facecolor("white")

    nonzero = hexes.loc[hexes["record_count"] > 0, "record_count"]
    if nonzero.empty:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        norm = mcolors.LogNorm(vmin=max(1, int(nonzero.min())), vmax=int(nonzero.max()))
    cmap = plt.get_cmap("turbo")

    for _, row in hexes.iterrows():
        count = int(row["record_count"])
        facecolor = "white" if count == 0 else cmap(norm(count))
        patch = MplPolygon(
            np.asarray(row.geometry.exterior.coords),
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.55,
            alpha=0.88 if count else 1.0,
            transform=projection,
            zorder=2,
        )
        ax.add_patch(patch)

    # Draw land on top so coastal hexagons visually follow the coastline.
    ax.add_geometries(
        [land_union],
        crs=projection,
        facecolor="#b8b8a6",
        edgecolor="#555555",
        linewidth=0.4,
        zorder=3,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.45, edgecolor="#333333", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.25, edgecolor="#777777", zorder=4)

    for _, row in hexes.iterrows():
        count = int(row["record_count"])
        if count == 0 and not LABEL_ZERO_CELLS:
            continue
        ocean_part = row.geometry.intersection(ocean_area)
        if ocean_part.is_empty:
            continue
        label_point = ocean_part.representative_point()
        if not (
            LON_MIN + LABEL_EDGE_MARGIN_DEG <= label_point.x <= LON_MAX - LABEL_EDGE_MARGIN_DEG
            and LAT_MIN + LABEL_EDGE_MARGIN_DEG <= label_point.y <= LAT_MAX - LABEL_EDGE_MARGIN_DEG
        ):
            continue

        ax.text(
            label_point.x,
            label_point.y,
            str(count),
            ha="center",
            va="center",
            fontsize=HEX_LABEL_FONT_SIZE,
            color=HEX_LABEL_FONT_COLOR,
            fontweight="bold" if count else "normal",
            transform=projection,
            zorder=5,
            clip_on=True,
        )

    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.25,
        color="#777777",
        alpha=0.35,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": FONT_SIZES["tick"]}
    gl.ylabel_style = {"size": FONT_SIZES["tick"]}

    if not nonzero.empty:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.62, pad=0.02)
        cbar.set_label(TEXT_LABELS["colorbar"], fontsize=FONT_SIZES["axis_label"])
        cbar.ax.tick_params(labelsize=FONT_SIZES["tick"])

    fig.savefig(HEX_COUNTS_PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    area_polygon = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    records = load_valid_records()
    land_union = load_land_union(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    ocean_area = area_polygon.difference(land_union)

    hexes = generate_hex_grid(area_polygon, HEX_SIDE_DEG)
    hexes = hexes.loc[hexes["geometry"].map(lambda geom: geom.intersects(ocean_area))].copy()
    hexes = count_records_by_hex(records, hexes)

    save_hex_counts(hexes)
    plot_hex_counts(hexes, land_union, ocean_area, records)

    print(f"Valid records: {len(records)}")
    print(f"Land records dropped from valid records: {records.attrs.get('dropped_land_count', 0)}")
    print(f"Hexagons over ocean: {len(hexes)}")
    print(f"Non-empty hexagons: {(hexes['record_count'] > 0).sum()}")
    print(f"Max count in one hexagon: {hexes['record_count'].max()}")
    print(f"CSV saved: {HEX_COUNTS_CSV}")
    print(f"Figure saved: {HEX_COUNTS_PNG}")


if __name__ == "__main__":
    main()
