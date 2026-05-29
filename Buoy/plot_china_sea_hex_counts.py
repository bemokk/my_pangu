from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


ROOT_DIR = Path(__file__).resolve().parent / "icoads_202507"
OUT_DIR = ROOT_DIR / "output"

# AREA follows the common meteorological order: [lat_max, lon_min, lat_min, lon_max].
AREA = [42, 103, 12, 131]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA
AREA_LABEL = f"area_{LAT_MAX:g}_{LON_MIN:g}_{LAT_MIN:g}_{LON_MAX:g}".replace(".", "p")
DETAIL_CSV = OUT_DIR / f"china_sea_all_platform_records_{AREA_LABEL}.csv"

START_DATE = pd.Timestamp("2025-07-01 00:00:00")
END_DATE = pd.Timestamp("2025-08-04 23:59:59")

# Configurable regular hexagon side length, in degrees.
HEX_SIDE_DEG = 1.0
HEX_SIDE_LABEL = f"side{str(HEX_SIDE_DEG).replace('.', 'p')}deg"
HEX_GRID_LAT_SHIFT_LAYERS = 1
HEX_SHIFT_LABEL = f"shift{HEX_GRID_LAT_SHIFT_LAYERS}layer"
HEX_COUNTS_CSV = OUT_DIR / f"china_sea_hex_counts_{AREA_LABEL}_{HEX_SIDE_LABEL}_{HEX_SHIFT_LABEL}.csv"
HEX_COUNTS_PNG = OUT_DIR / f"china_sea_hex_counts_{AREA_LABEL}_{HEX_SIDE_LABEL}_{HEX_SHIFT_LABEL}.png"

# If True, cells with zero records are labeled as 0. This is usually too dense for 1-degree grids.
LABEL_ZERO_CELLS = False


def regular_hexagon(center_lon: float, center_lat: float, side_deg: float) -> Polygon:
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    coords = [
        (center_lon + side_deg * np.cos(angle), center_lat + side_deg * np.sin(angle))
        for angle in angles
    ]
    return Polygon(coords)


def generate_hex_grid(area_polygon: Polygon, side_deg: float, lat_shift_layers: int = 0) -> pd.DataFrame:
    dx = 1.5 * side_deg
    dy = np.sqrt(3.0) * side_deg
    lat_shift = lat_shift_layers * dy

    min_lon, min_lat, max_lon, max_lat = area_polygon.bounds
    centers = []
    col = 0
    lon = min_lon - 2 * side_deg

    while lon <= max_lon + 2 * side_deg:
        lat_offset = dy / 2.0 if col % 2 else 0.0
        lat = min_lat - 2 * dy + lat_offset + lat_shift

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


def load_land_union(area_polygon: Polygon):
    land_path = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    reader = shpreader.Reader(land_path)
    land_geoms = [geom for geom in reader.geometries() if geom.intersects(area_polygon)]
    return unary_union(land_geoms)


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


def iter_polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def save_hex_counts(hexes: pd.DataFrame) -> None:
    out = hexes.drop(columns="geometry").copy()
    out.to_csv(HEX_COUNTS_CSV, index=False, encoding="utf-8-sig")


def plot_hex_counts(hexes: pd.DataFrame, land_union, ocean_area, records: pd.DataFrame) -> None:
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13.5, 13.0))
    ax = plt.axes(projection=projection)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=projection)

    ax.set_facecolor("white")

    ax.add_geometries(
        [land_union],
        crs=projection,
        facecolor="#b8b8a6",
        edgecolor="#555555",
        linewidth=0.35,
        zorder=1,
    )

    nonzero = hexes.loc[hexes["record_count"] > 0, "record_count"]
    if nonzero.empty:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        norm = mcolors.LogNorm(vmin=max(1, int(nonzero.min())), vmax=int(nonzero.max()))
    cmap = plt.get_cmap("turbo")

    label_points = []
    for _, row in hexes.iterrows():
        count = int(row["record_count"])
        facecolor = "white" if count == 0 else cmap(norm(count))
        plot_geometry = row.geometry.intersection(ocean_area)
        label_point = plot_geometry.representative_point() if not plot_geometry.is_empty else row.geometry.centroid
        label_points.append((label_point.x, label_point.y))

        for polygon in iter_polygons(plot_geometry):
            patch = MplPolygon(
                np.asarray(polygon.exterior.coords),
                closed=True,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.55,
                alpha=0.88 if count else 1.0,
                transform=projection,
                zorder=2,
            )
            ax.add_patch(patch)

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.45, edgecolor="#333333", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.25, edgecolor="#777777", zorder=4)

    for (_, row), (label_lon, label_lat) in zip(hexes.iterrows(), label_points):
        count = int(row["record_count"])
        if count == 0 and not LABEL_ZERO_CELLS:
            continue
        if count == 0:
            label_color = "#555555"
        else:
            label_color = "white" if norm(count) > 0.35 else "#222222"

        ax.text(
            label_lon,
            label_lat,
            str(count),
            ha="center",
            va="center",
            fontsize=6.4 if HEX_SIDE_DEG <= 1.0 else 8.5,
            color=label_color,
            fontweight="bold" if count else "normal",
            transform=projection,
            zorder=5,
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

    title = (
        f"Valid ICOADS Wind Record Counts by Hexagon "
        f"(side={HEX_SIDE_DEG:g} deg)"
    )
    subtitle = (
        f"AREA={AREA}, {START_DATE:%Y-%m-%d} to {END_DATE:%Y-%m-%d}, "
        f"hex grid shifted up {HEX_GRID_LAT_SHIFT_LAYERS} layer(s), "
        f"valid wind direction/speed records: {len(records):,}"
    )
    ax.set_title(title, fontsize=15, pad=14)
    fig.text(0.125, 0.91, subtitle, fontsize=9.5, color="#444444")

    if not nonzero.empty:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.62, pad=0.02)
        cbar.set_label("Record count per hexagon")

    fig.savefig(HEX_COUNTS_PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    area_polygon = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    records = load_valid_records()
    land_union = load_land_union(area_polygon)
    ocean_area = area_polygon.difference(land_union)

    hexes = generate_hex_grid(area_polygon, HEX_SIDE_DEG, HEX_GRID_LAT_SHIFT_LAYERS)
    hexes = hexes.loc[hexes["geometry"].map(lambda geom: geom.intersects(ocean_area))].copy()
    hexes = count_records_by_hex(records, hexes)

    save_hex_counts(hexes)
    plot_hex_counts(hexes, land_union, ocean_area, records)

    print(f"Valid records: {len(records)}")
    print(f"Hexagons over ocean: {len(hexes)}")
    print(f"Non-empty hexagons: {(hexes['record_count'] > 0).sum()}")
    print(f"Max count in one hexagon: {hexes['record_count'].max()}")
    print(f"CSV saved: {HEX_COUNTS_CSV}")
    print(f"Figure saved: {HEX_COUNTS_PNG}")


if __name__ == "__main__":
    main()
