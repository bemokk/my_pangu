from __future__ import annotations

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
    OUT_TRACK_BUOYS_PNG,
    OUT_TRACK_BUOYS_SVG,
    PLATFORM_COLORS,
    PROJECT_ROOT,
    WINDOW_END,
    WINDOW_START,
    ensure_dirs,
    prepare_buoy_case_data,
    set_plot_style,
)

LOCAL_WIPHA_TRACK_CSV = PROJECT_ROOT / "typhoon_2506_Wipha.csv"
LAND_TIME_LABEL_POSITIONS = {
    pd.Timestamp("2025-07-20 12:00"): (114.2, 22.8),
    pd.Timestamp("2025-07-20 18:00"): (113.0, 22.9),
    pd.Timestamp("2025-07-21 00:00"): (111.8, 22.8),
    pd.Timestamp("2025-07-21 06:00"): (110.5, 22.7),
    pd.Timestamp("2025-07-21 12:00"): (109.2, 22.7),
    pd.Timestamp("2025-07-21 18:00"): (108.0, 22.5),
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


def plot_track_buoy_locations(real_track: pd.DataFrame, obs: pd.DataFrame) -> None:
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
    ax.set_title("Typhoon Wipha Track and Selected Platform Locations", loc="left", fontweight="bold")

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
            label="Observed Wipha track (6-hourly)",
            transform=projection,
            zorder=5,
        )
        default_label_indices = set(range(0, len(case_track), max(1, len(case_track) // 8)))
        for label_index, (_, row) in enumerate(case_track.iterrows()):
            timestamp = pd.Timestamp(row["datetime_utc"])
            if label_index not in default_label_indices and timestamp not in LAND_TIME_LABEL_POSITIONS:
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

    for platform_id, sub in obs.groupby("platform_id"):
        color = PLATFORM_COLORS.get(platform_id, "#777777")
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=22,
            alpha=0.45,
            color=color,
            edgecolor="none",
            label=f"{platform_id} positions",
            transform=projection,
            zorder=4,
        )
        mean_lon, mean_lat = sub["longitude"].mean(), sub["latitude"].mean()
        ax.scatter(
            [mean_lon],
            [mean_lat],
            marker="*",
            s=180,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            transform=projection,
            zorder=7,
        )
        ax.text(
            mean_lon + 0.25,
            mean_lat + 0.25,
            platform_id,
            color=color,
            fontweight="bold",
            transform=projection,
            zorder=8,
        )

    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.text(
        0.5,
        0.015,
        "Typhoon track is read from typhoon_2506_Wipha.csv and plotted at 6-hour intervals; platform points show observations during 2025-07-17 00 UTC to 2025-07-22 23 UTC.",
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
    obs, _, _ = prepare_buoy_case_data()
    real_track = load_wipha_track_csv()
    plot_track_buoy_locations(real_track, obs)
    return [OUT_TRACK_BUOYS_PNG, OUT_TRACK_BUOYS_SVG]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
