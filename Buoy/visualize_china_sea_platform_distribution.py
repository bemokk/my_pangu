from pathlib import Path

import folium
import pandas as pd
from folium.plugins import FastMarkerCluster, Fullscreen, HeatMap, MarkerCluster, MiniMap


ROOT_DIR = Path(__file__).resolve().parent / "icoads_202507"
OUT_DIR = ROOT_DIR / "output"

AREA = [42, 103, 13, 130]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA

DETAIL_CSV = OUT_DIR / "china_sea_all_platform_records_area_42_103_13_130.csv"
SUMMARY_CSV = OUT_DIR / "china_sea_all_platform_summary_area_42_103_13_130.csv"
MAP_OUT = OUT_DIR / "china_sea_platform_distribution_area_42_103_13_130.html"


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
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

    records = pd.read_csv(DETAIL_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    records = records.dropna(subset=["latitude", "longitude"]).copy()
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
      AREA = [42, 103, 13, 130], no time/type restriction<br>
      Records: {len(records):,} | Platforms: {len(summary):,}
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
