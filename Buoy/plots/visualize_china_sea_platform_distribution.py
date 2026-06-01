import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import numpy as np
import folium
import pandas as pd
from folium.plugins import FastMarkerCluster, Fullscreen, HeatMap, MarkerCluster, MiniMap

from paths import DEFAULT_CHINA_SEA_DETAIL_CSV, FIGURES_DIR


AREA = [42, 103, 13, 130]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA
TARGET_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

# True includes records within +/-30 minutes of each target UTC hour.
# False keeps only near-exact target hours.
INCLUDE_HALF_HOUR_WINDOW = True
TIME_TOLERANCE_HOURS = 0.5 if INCLUDE_HALF_HOUR_WINDOW else 0.01

DETAIL_CSV = DEFAULT_CHINA_SEA_DETAIL_CSV
MAP_OUT = FIGURES_DIR / "china_sea_platform_distribution_area_42_103_13_130.html"


PLATFORM_TYPE_COLORS = {
    "5": "#d62728",
    "6": "#1f77b4",
    "7": "#2ca02c",
}


def normalize_longitude(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def platform_color(platform_type_values: str) -> str:
    first_value = str(platform_type_values).split(",")[0].strip()
    return PLATFORM_TYPE_COLORS.get(first_value, "#9467bd")


def build_platform_popup(row: pd.Series, map_lon: float) -> str:
    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.45;">
      <b>Platform ID:</b> {row['platform_id']}<br>
      <b>Platform type:</b> {row.get('platform_type_values', '')}<br>
      <b>ID indicator:</b> {row.get('id_indicator_values', '')}<br>
      <b>Records:</b> {int(row['obs_count'])}<br>
      <b>Time:</b> {row['first_time']} to {row['last_time']}<br>
      <b>Mean lat/lon:</b> {row['mean_latitude']:.4f}, {map_lon:.4f}<br>
      <b>Raw mean lon:</b> {row['mean_longitude']:.4f}<br>
      <b>Lat range:</b> {row['min_latitude']:.4f} to {row['max_latitude']:.4f}
      ({row['lat_range_deg']:.4f} deg)<br>
      <b>Lon range:</b> {row['min_longitude']:.4f} to {row['max_longitude']:.4f}
      ({row['lon_range_deg']:.4f} deg)<br>
      <b>Wind direction records:</b> {int(row.get('wind_dir_count', 0))}<br>
      <b>Wind speed records:</b> {int(row.get('wind_speed_count', 0))}<br>
    </div>
    """


def join_unique_values(series: pd.Series) -> str:
    values = series.dropna().astype(int).astype(str).unique()
    return ",".join(sorted(values))


def minimal_longitude_range(longitudes: pd.Series) -> float:
    lon = pd.to_numeric(longitudes, errors="coerce").dropna().to_numpy(dtype=float)
    if lon.size <= 1:
        return 0.0

    lon = np.mod(lon, 360.0)
    lon.sort()
    gaps = np.diff(np.r_[lon, lon[0] + 360.0])
    return float(360.0 - gaps.max())


def build_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for platform_id, group in records.groupby("platform_id", dropna=False):
        rows.append(
            {
                "platform_id": platform_id,
                "obs_count": int(len(group)),
                "first_time": group["datetime_utc"].min() if "datetime_utc" in group.columns else "",
                "last_time": group["datetime_utc"].max() if "datetime_utc" in group.columns else "",
                "mean_latitude": group["latitude"].mean(),
                "mean_longitude": group["longitude"].mean(),
                "min_latitude": group["latitude"].min(),
                "max_latitude": group["latitude"].max(),
                "min_longitude": group["longitude"].min(),
                "max_longitude": group["longitude"].max(),
                "lat_range_deg": group["latitude"].max() - group["latitude"].min(),
                "lon_range_deg": minimal_longitude_range(group["longitude"]),
                "id_indicator_values": join_unique_values(group["id_indicator"]) if "id_indicator" in group.columns else "",
                "platform_type_values": join_unique_values(group["platform_type"]) if "platform_type" in group.columns else "",
                "wind_dir_count": int(group["wind_dir_deg"].notna().sum()) if "wind_dir_deg" in group.columns else 0,
                "wind_speed_count": int(group["wind_speed_ms"].notna().sum()) if "wind_speed_ms" in group.columns else 0,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(["obs_count", "platform_id"], ascending=[False, True])


def add_target_hour(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()

    if "datetime_utc" in records.columns:
        records["datetime_utc"] = pd.to_datetime(records["datetime_utc"], errors="coerce")
        hour_from_time = (
            records["datetime_utc"].dt.hour
            + records["datetime_utc"].dt.minute / 60.0
            + records["datetime_utc"].dt.second / 3600.0
        )
    else:
        hour_from_time = pd.Series(pd.NA, index=records.index, dtype="float64")

    if "hour_utc" not in records.columns:
        records["hour_utc"] = pd.NA

    records["hour_utc"] = pd.to_numeric(records["hour_utc"], errors="coerce").fillna(hour_from_time)

    target_hours = np.array(TARGET_HOURS, dtype=float)
    hour_values = records["hour_utc"].to_numpy(dtype=float)
    raw_diff = np.abs(hour_values[:, None] - target_hours[None, :])
    diff = np.minimum(raw_diff, 24.0 - raw_diff)
    nearest_idx = np.nanargmin(np.where(np.isnan(diff), np.inf, diff), axis=1)

    records["target_hour_utc"] = target_hours[nearest_idx].astype(int)
    records["hour_diff"] = np.abs(records["hour_utc"] - records["target_hour_utc"])
    return records


def add_type_legend(map_obj: folium.Map) -> None:
    legend_html = """
    <div style="
      position: fixed;
      bottom: 30px;
      left: 50px;
      z-index: 9999;
      background: rgba(255, 255, 255, 0.93);
      padding: 10px 12px;
      border: 1px solid #bbb;
      border-radius: 4px;
      font-family: Arial, sans-serif;
      font-size: 13px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    ">
      <b>Platform type</b><br>
      <span style="color:#d62728;">●</span> 5<br>
      <span style="color:#1f77b4;">●</span> 6<br>
      <span style="color:#2ca02c;">●</span> 7<br>
      <span style="color:#9467bd;">●</span> Other
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def main() -> None:
    if not DETAIL_CSV.exists():
        raise FileNotFoundError(f"Detail CSV not found: {DETAIL_CSV}")

    records = pd.read_csv(DETAIL_CSV)
    total_records = len(records)
    records = add_target_hour(records)

    required_cols = ["latitude", "longitude", "wind_dir_deg", "wind_speed_ms"]
    records = records.dropna(subset=required_cols).copy()
    records = records[
        records["wind_dir_deg"].between(1, 360)
        & records["wind_speed_ms"].between(0, 75)
        & (records["hour_diff"] <= TIME_TOLERANCE_HOURS)
    ].copy()
    summary = build_summary(records)
    summary = summary.dropna(subset=["mean_latitude", "mean_longitude"]).copy()

    records["map_longitude"] = records["longitude"].map(normalize_longitude)
    summary["map_longitude"] = summary["mean_longitude"].map(normalize_longitude)

    center_lat = (LAT_MIN + LAT_MAX) / 2
    center_lon = (LON_MIN + LON_MAX) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer("CartoDB positron", name="Light", control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark", control=True).add_to(m)

    folium.Rectangle(
        bounds=[[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
        tooltip=f"AREA = {AREA}",
        color="#333333",
        weight=2,
        fill=False,
        name="AREA boundary",
    ).add_to(m)

    heat_points = records[["latitude", "map_longitude"]].values.tolist()
    HeatMap(
        heat_points,
        name="Observation density heatmap",
        radius=14,
        blur=18,
        min_opacity=0.25,
        max_zoom=8,
    ).add_to(m)

    FastMarkerCluster(
        records[["latitude", "map_longitude"]].values.tolist(),
        name="All observation records",
    ).add_to(m)

    platform_cluster = MarkerCluster(name="Platform mean positions").add_to(m)

    for _, row in summary.iterrows():
        lat = float(row["mean_latitude"])
        lon = float(row["map_longitude"])
        obs_count = int(row["obs_count"])
        radius = min(12, 4 + obs_count ** 0.35 / 2)

        tooltip = (
            f"{row['platform_id']} | type {row.get('platform_type_values', '')} | "
            f"{obs_count} records | {lat:.3f}, {lon:.3f}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=platform_color(row.get("platform_type_values", "")),
            weight=1,
            fill=True,
            fill_opacity=0.78,
            popup=folium.Popup(build_platform_popup(row, lon), max_width=430),
            tooltip=tooltip,
        ).add_to(platform_cluster)

    title_html = f"""
    <div style="
      position: fixed;
      top: 12px;
      left: 50px;
      z-index: 9999;
      background: rgba(255, 255, 255, 0.93);
      padding: 10px 12px;
      border: 1px solid #bbb;
      border-radius: 4px;
      font-family: Arial, sans-serif;
      font-size: 13px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    ">
      <b>China Sea and nearby ICOADS records</b><br>
      AREA = [42, 103, 13, 130], wind direction/speed required<br>
      UTC hours: 0, 3, 6, 9, 12, 15, 18, 21
      ({'±30 min' if INCLUDE_HALF_HOUR_WINDOW else 'exact only'})<br>
      Records shown: {len(records):,} / {total_records:,} | Platforms: {len(summary):,}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    add_type_legend(m)

    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]], padding=(20, 20))

    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(MAP_OUT)

    print(f"Map saved: {MAP_OUT}")
    print(f"Records plotted: {len(records)}")
    print(f"Platforms plotted: {len(summary)}")


if __name__ == "__main__":
    main()
