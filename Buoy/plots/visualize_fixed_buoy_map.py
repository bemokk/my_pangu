import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import folium
import pandas as pd
from folium.plugins import Fullscreen, MarkerCluster, MiniMap

from paths import FIGURES_DIR, FIXED_BUOY_WIND_DIR


SUMMARY_CSV = FIXED_BUOY_WIND_DIR / "fixed_buoy_platform_summary_20250701_20250804_3hourly.csv"
MAP_OUT = FIGURES_DIR / "fixed_buoy_map_20250701_20250804_3hourly.html"


def normalize_longitude(lon: float) -> float:
    """Convert 0-360 or arbitrary longitude to the -180..180 range for web maps."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def marker_color(platform_type_values: str) -> str:
    values = {value.strip() for value in str(platform_type_values).split(",") if value.strip()}

    if "6" in values:
        return "#1f77b4"
    if "14" in values or "16" in values:
        return "#2ca02c"
    return "#ff7f0e"


def build_popup(row: pd.Series, map_lon: float) -> str:
    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.45;">
      <b>Platform ID:</b> {row['platform_id']}<br>
      <b>Platform type:</b> {row.get('platform_type_values', '')}<br>
      <b>Records:</b> {int(row['obs_count'])}<br>
      <b>Time:</b> {row['first_time']} to {row['last_time']}<br>
      <b>Mean lat/lon:</b> {row['mean_latitude']:.4f}, {map_lon:.4f}<br>
      <b>Raw mean lon:</b> {row['mean_longitude']:.4f}<br>
      <b>Lat range:</b> {row['min_latitude']:.4f} to {row['max_latitude']:.4f}
      ({row['lat_range_deg']:.4f} deg)<br>
      <b>Lon range:</b> {row['min_longitude']:.4f} to {row['max_longitude']:.4f}
      ({row['lon_range_deg']:.4f} deg)<br>
    </div>
    """


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        raise RuntimeError(f"Summary CSV is empty: {SUMMARY_CSV}")

    df = df.dropna(subset=["mean_latitude", "mean_longitude"]).copy()
    df["map_longitude"] = df["mean_longitude"].map(normalize_longitude)

    center_lat = df["mean_latitude"].mean()
    center_lon = df["map_longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Light",
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark",
        control=True,
    ).add_to(m)

    cluster = MarkerCluster(name="Fixed or near-fixed buoys").add_to(m)

    for _, row in df.iterrows():
        lat = float(row["mean_latitude"])
        lon = float(row["map_longitude"])
        obs_count = int(row["obs_count"])
        radius = min(12, 4 + obs_count ** 0.35 / 2)

        tooltip = (
            f"{row['platform_id']} | "
            f"{obs_count} records | "
            f"{lat:.3f}, {lon:.3f}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=marker_color(row.get("platform_type_values", "")),
            weight=1,
            fill=True,
            fill_opacity=0.72,
            popup=folium.Popup(build_popup(row, lon), max_width=420),
            tooltip=tooltip,
        ).add_to(cluster)

    sw = [df["mean_latitude"].min(), df["map_longitude"].min()]
    ne = [df["mean_latitude"].max(), df["map_longitude"].max()]
    m.fit_bounds([sw, ne], padding=(20, 20))

    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    title_html = f"""
    <div style="
      position: fixed;
      top: 12px;
      left: 50px;
      z-index: 9999;
      background: rgba(255, 255, 255, 0.92);
      padding: 10px 12px;
      border: 1px solid #bbb;
      border-radius: 4px;
      font-family: Arial, sans-serif;
      font-size: 13px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    ">
      <b>Fixed / near-fixed buoy positions</b><br>
      2025-07-01 to 2025-08-04, 3-hourly UTC records<br>
      Platforms: {len(df)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(MAP_OUT)

    print(f"Map saved: {MAP_OUT}")
    print(f"Platforms plotted: {len(df)}")


if __name__ == "__main__":
    main()
