from __future__ import annotations

import math
import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt

from paths import FIGURES_DIR
from plots.wipha_case_common import set_plot_style


OUT_PNG = FIGURES_DIR / "wipha_virtual_station_radius_test.png"
OUT_SVG = FIGURES_DIR / "wipha_virtual_station_radius_test.svg"

VIRTUAL_STATIONS = [
    {
        "station_id": "VS1",
        "label": "120E,20N",
        "lon": 120.0,
        "lat": 20.0,
        "radius_km": 300.0,
        "color": "#C44E52",
        "unique_times": 18,
    },
    {
        "station_id": "VS2",
        "label": "113E,21N",
        "lon": 113.0,
        "lat": 21.0,
        "radius_km": 400.0,
        "color": "#4C72B0",
        "unique_times": 15,
    },
]


def geodesic_circle_points(lon_deg: float, lat_deg: float, radius_km: float, n_points: int = 241) -> list[tuple[float, float]]:
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    if n_points < 4:
        raise ValueError("n_points must be at least 4")

    earth_radius_km = 6371.0
    angular_distance = radius_km / earth_radius_km
    lon1 = math.radians(lon_deg)
    lat1 = math.radians(lat_deg)
    points = []

    for index in range(n_points):
        bearing = 2.0 * math.pi * index / (n_points - 1)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance)
            + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
            math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
        )
        lon_out = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
        points.append((lon_out, math.degrees(lat2)))

    return points


def plot_virtual_station_radius_map() -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from land_mask import load_land_union

    set_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9.2, 6.8))
    ax = plt.axes(projection=projection)
    ax.set_extent([107.0, 124.5, 15.0, 25.6], crs=projection)
    ax.set_facecolor("#EAF3F8")

    land_union = load_land_union(107.0, 15.0, 124.5, 25.6)
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

    for station in VIRTUAL_STATIONS:
        circle = geodesic_circle_points(station["lon"], station["lat"], station["radius_km"])
        circle_lons = [point[0] for point in circle]
        circle_lats = [point[1] for point in circle]
        ax.plot(
            circle_lons,
            circle_lats,
            color=station["color"],
            linewidth=1.6,
            transform=projection,
            zorder=5,
            label=f"{station['label']} radius {station['radius_km']:.0f} km",
        )
        ax.fill(
            circle_lons,
            circle_lats,
            color=station["color"],
            alpha=0.12,
            transform=projection,
            zorder=4,
        )
        ax.scatter(
            station["lon"],
            station["lat"],
            s=110,
            marker="*",
            color=station["color"],
            edgecolor="black",
            linewidth=0.8,
            transform=projection,
            zorder=7,
        )
        ax.text(
            station["lon"] + 0.18,
            station["lat"] + 0.18,
            (
                f"{station['station_id']} {station['label']}\n"
                f"R={station['radius_km']:.0f} km\n"
                f"{station['unique_times']} valid times"
            ),
            fontsize=8.4,
            color="#222222",
            transform=projection,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0},
            zorder=8,
        )

    ax.set_title("Typhoon Wipha Virtual Fixed Stations and Search Radii", loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.text(
        0.5,
        0.018,
        "Radii are the rounded search distances needed for at least 15 distinct 3-hourly matched times during 2025-07-18 03 UTC to 2025-07-21 00 UTC.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.98])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_virtual_station_radius_map()
    print(OUT_PNG)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
