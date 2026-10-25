from __future__ import annotations

import math
import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from plots.wipha_case_common import (
    FIGURES_DIR,
    MAP_AREA,
    MATCHED_CSV,
    OUT_TRACK_BUOYS_PNG,
    OUT_TRACK_BUOYS_SVG,
    PROJECT_ROOT,
    WINDOW_END,
    WINDOW_START,
    ensure_dirs,
    haversine_km,
    set_plot_style,
)

LOCAL_WIPHA_TRACK_CSV = PROJECT_ROOT / "typhoon_2506_Wipha.csv"
TRACK_LABEL_HOURS = {0, 12}
FIXED_INIT_TIMES = {
    "gdas_forecast": "2025-07-18-00-00",
    "era5_lagged_5d": "2025-07-13-00-00",
}
VIRTUAL_POINT_STATIONS = [
    {
        "station_id": "Point 1",
        "label": "Point 1",
        "lon": 118.90,
        "lat": 21.32,
        "radius_km": 135.0,
        "color": "#C44E52",
        "text_offset": (0.65, 0.82),
    },
    {
        "station_id": "Point 2",
        "label": "Point 2",
        "lon": 115.64,
        "lat": 22.25,
        "radius_km": 110.0,
        "color": "#4C72B0",
        "text_offset": (-1.35, 1.08),
    },
]
LAND_TIME_LABEL_POSITIONS = {
    pd.Timestamp("2025-07-19 12:00"): (117.4, 20.55),
    pd.Timestamp("2025-07-20 00:00"): (115.55, 20.85),
    pd.Timestamp("2025-07-20 12:00"): (113.1, 22.8),
    pd.Timestamp("2025-07-21 00:00"): (110.8, 22.8),
    pd.Timestamp("2025-07-21 12:00"): (109.2, 22.7),
    pd.Timestamp("2025-07-22 00:00"): (107.0, 22.3),
}


def load_wipha_track_csv(csv_path: Path = LOCAL_WIPHA_TRACK_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Wipha typhoon track CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"tc_num", "name_cn", "name_en", "dateUTC", "vmax", "grade", "latTC", "lonTC", "mslp", "attr"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Wipha typhoon track CSV missing columns: {', '.join(missing)}")

    track = pd.DataFrame(
        {
            "tc_num": df["tc_num"],
            "name_cn": df["name_cn"],
            "name_en": df["name_en"],
            "datetime_utc": pd.to_datetime(df["dateUTC"].astype(str), format="%Y%m%d%H%M", errors="coerce"),
            "vmax_ms": pd.to_numeric(df["vmax"], errors="coerce"),
            "grade": df["grade"],
            "lon": pd.to_numeric(df["lonTC"], errors="coerce"),
            "lat": pd.to_numeric(df["latTC"], errors="coerce"),
            "mslp_hpa": pd.to_numeric(df["mslp"], errors="coerce"),
            "attr": df["attr"],
            "source": "typhoon_2506_Wipha.csv",
        }
    )
    track = track.dropna(subset=["datetime_utc", "lon", "lat"]).sort_values("datetime_utc").reset_index(drop=True)
    if track.empty:
        raise ValueError(f"No valid Wipha track rows found in {csv_path}")
    return track


def select_six_hour_track_points(track: pd.DataFrame) -> pd.DataFrame:
    if track.empty:
        return track.copy()
    working = track.copy()
    working["datetime_utc"] = pd.to_datetime(working["datetime_utc"], errors="coerce")
    working = working.dropna(subset=["datetime_utc"])
    return working[working["datetime_utc"].dt.hour.mod(6).eq(0)].reset_index(drop=True)


def select_twelve_hour_track_points(track: pd.DataFrame) -> pd.DataFrame:
    if track.empty:
        return track.copy()
    working = track.copy()
    working["datetime_utc"] = pd.to_datetime(working["datetime_utc"], errors="coerce")
    working = working.dropna(subset=["datetime_utc"])
    return working[working["datetime_utc"].dt.hour.isin(TRACK_LABEL_HOURS)].reset_index(drop=True)


def should_label_track_time(datetime_utc: pd.Timestamp) -> bool:
    return pd.Timestamp(datetime_utc).hour in TRACK_LABEL_HOURS


def track_time_label_annotation(datetime_utc: pd.Timestamp, lon: float, lat: float) -> dict:
    timestamp = pd.Timestamp(datetime_utc)
    annotation = {
        "text": timestamp.strftime("%m-%d %H"),
        "xy": (float(lon), float(lat)),
        "xytext": (float(lon) + 0.15, float(lat) + 0.12),
        "fontsize": 7.5,
    }
    if timestamp in LAND_TIME_LABEL_POSITIONS:
        annotation.update(
            {
                "xytext": LAND_TIME_LABEL_POSITIONS[timestamp],
                "textcoords": "data",
                "ha": "center",
                "va": "center",
                "bbox": {
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#555555",
                    "linewidth": 0.35,
                    "alpha": 0.82,
                },
                "arrowprops": {
                    "arrowstyle": "->",
                    "color": "#222222",
                    "linewidth": 0.8,
                    "shrinkA": 2,
                    "shrinkB": 2,
                    "connectionstyle": "arc3,rad=0.08",
                },
            }
        )
    return annotation


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


def load_virtual_station_common_samples(matched_csv: Path = MATCHED_CSV) -> pd.DataFrame:
    cols = ["record_id", "dataset", "pred_start_time", "lead_hour", "datetime_utc", "longitude", "latitude", "platform_id"]
    df = pd.read_csv(matched_csv, usecols=cols)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    dataset_filter = False
    for dataset, start_time in FIXED_INIT_TIMES.items():
        dataset_filter = dataset_filter | ((df["dataset"] == dataset) & (df["pred_start_time"] == start_time))
    df = df.loc[
        dataset_filter
        & df["lead_hour"].between(3, 72)
        & df["lead_hour"].mod(3).eq(0)
    ].copy()
    keys = ["datetime_utc", "record_id", "platform_id", "longitude", "latitude"]
    wide = df.groupby(keys + ["dataset"]).size().unstack(fill_value=0).reset_index()
    return wide[
        (wide.get("gdas_forecast", 0) > 0)
        & (wide.get("era5_lagged_5d", 0) > 0)
    ].reset_index(drop=True)


def summarize_virtual_station_coverage(
    common_samples: pd.DataFrame,
    stations: list[dict] = VIRTUAL_POINT_STATIONS,
) -> pd.DataFrame:
    rows = []
    for station in stations:
        sub = common_samples[
            common_samples.apply(
                lambda row: haversine_km(station["lon"], station["lat"], row["longitude"], row["latitude"])
                <= station["radius_km"],
                axis=1,
            )
        ].copy()
        rows.append(
            {
                "station_id": station["station_id"],
                "valid_time_count": int(sub["datetime_utc"].nunique()),
                "record_count": int(len(sub)),
                "platform_count": int(sub["platform_id"].nunique()),
                "valid_times": sorted(pd.to_datetime(sub["datetime_utc"]).unique()),
            }
        )
    return pd.DataFrame(rows)


def plot_track_buoy_locations(real_track: pd.DataFrame, station_summary: pd.DataFrame) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from land_mask import load_land_union

    set_plot_style()
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.4, 6.4))
    ax = plt.axes(projection=projection)
    ax.set_extent([MAP_AREA[0], MAP_AREA[1], MAP_AREA[2], MAP_AREA[3]], crs=projection)
    ax.set_facecolor("#EAF3F8")

    land_union = load_land_union(MAP_AREA[0], MAP_AREA[2], MAP_AREA[1], MAP_AREA[3])
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
    ax.set_title("Typhoon Wipha Track and Virtual Station Search Areas", loc="left", fontweight="bold")

    case_track = real_track[real_track["datetime_utc"].between(WINDOW_START, WINDOW_END)]
    case_track = select_six_hour_track_points(case_track)
    if not case_track.empty:
        ax.plot(
            case_track["lon"],
            case_track["lat"],
            color="#222222",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Observed Wipha track (6-hourly; labels 12-hourly)",
            transform=projection,
            zorder=5,
        )
        for _, row in case_track.iterrows():
            timestamp = pd.Timestamp(row["datetime_utc"])
            if not should_label_track_time(timestamp):
                continue

            annotation = track_time_label_annotation(timestamp, lon=row["lon"], lat=row["lat"])
            if "arrowprops" in annotation:
                map_transform = projection._as_mpl_transform(ax)
                ax.annotate(
                    annotation["text"],
                    xy=annotation["xy"],
                    xytext=annotation["xytext"],
                    xycoords=map_transform,
                    textcoords=map_transform,
                    fontsize=annotation["fontsize"],
                    ha=annotation["ha"],
                    va=annotation["va"],
                    bbox=annotation["bbox"],
                    arrowprops=annotation["arrowprops"],
                    zorder=6,
                )
            else:
                ax.text(
                    annotation["xytext"][0],
                    annotation["xytext"][1],
                    annotation["text"],
                    fontsize=annotation["fontsize"],
                    transform=projection,
                    zorder=6,
                )

    summary_by_id = station_summary.set_index("station_id").to_dict("index")
    for station in VIRTUAL_POINT_STATIONS:
        color = station["color"]
        circle = geodesic_circle_points(station["lon"], station["lat"], station["radius_km"])
        circle_lons = [point[0] for point in circle]
        circle_lats = [point[1] for point in circle]
        ax.plot(
            circle_lons,
            circle_lats,
            color=color,
            linewidth=1.6,
            label=f"{station['label']} radius {station['radius_km']:.0f} km",
            transform=projection,
            zorder=4,
        )
        ax.fill(
            circle_lons,
            circle_lats,
            color=color,
            alpha=0.12,
            transform=projection,
            zorder=3,
        )
        ax.scatter(
            [station["lon"]],
            [station["lat"]],
            marker="*",
            s=180,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            transform=projection,
            zorder=7,
        )
        summary = summary_by_id.get(station["station_id"], {})
        valid_time_count = int(summary.get("valid_time_count", 0))
        record_count = int(summary.get("record_count", 0))
        dx, dy = station["text_offset"]
        ax.text(
            station["lon"] + dx,
            station["lat"] + dy,
            (
                f"{station['label']} {station['lon']:.2f}E,{station['lat']:.2f}N\n"
                f"R={station['radius_km']:.0f} km\n"
                f"{valid_time_count} valid times, {record_count} records"
            ),
            color=color,
            fontweight="bold",
            fontsize=8.2,
            transform=projection,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
            zorder=8,
        )

    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.text(
        0.5,
        0.015,
        "Typhoon track is read from typhoon_2506_Wipha.csv; virtual station counts use matched GDAS and ERA5 lagged samples at fixed-init 3-hourly target times.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.98])
    fig.savefig(OUT_TRACK_BUOYS_PNG, bbox_inches="tight")
    fig.savefig(OUT_TRACK_BUOYS_SVG, bbox_inches="tight")
    plt.close(fig)


def generate() -> list[Path]:
    ensure_dirs()
    real_track = load_wipha_track_csv()
    common_samples = load_virtual_station_common_samples()
    station_summary = summarize_virtual_station_coverage(common_samples)
    plot_track_buoy_locations(real_track, station_summary)
    return [OUT_TRACK_BUOYS_PNG, OUT_TRACK_BUOYS_SVG]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
