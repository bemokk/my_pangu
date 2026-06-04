from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from cartopy.io import shapereader
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "wipha_track_relative_analysis"
SAMPLES_CSV = OUT_DIR / "plotting_samples_0_600km.csv"
TRACK_CSV = PROJECT_ROOT / "Buoy" / "results" / "wipha_track_relative_analysis" / "data" / "typhoon_2506_Wipha.csv"
PNG_OUT = OUT_DIR / "figure_icoads_sample_coverage.png"
PDF_OUT = OUT_DIR / "figure_icoads_sample_coverage.pdf"

CASE_START = pd.Timestamp("2025-07-17 00:00:00")
CASE_END = CASE_START + pd.Timedelta(hours=72)

WIDTH = 2200
HEIGHT = 1800
PLOT_LEFT = 190
PLOT_RIGHT = 1660
PLOT_TOP = 190
PLOT_BOTTOM = 1510
COLORBAR_LEFT = 1780
COLORBAR_TOP = 330
COLORBAR_WIDTH = 55
COLORBAR_HEIGHT = 1000


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size=size)


FONT_TITLE = font(58, bold=True)
FONT_AXIS = font(39)
FONT_TICK = font(32)
FONT_LABEL = font(26)
FONT_SMALL = font(24)
FONT_LEGEND = font(32)
FONT_LEGEND_TITLE = font(34)


def viridis_like(value: float, vmin: float = 0.0, vmax: float = 72.0) -> tuple[int, int, int]:
    # A compact piecewise approximation to viridis, enough for stable PNG rendering without matplotlib.
    anchors = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=float,
    )
    t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    scaled = t * (len(anchors) - 1)
    i = int(np.floor(scaled))
    if i >= len(anchors) - 1:
        return tuple(anchors[-1].astype(int))
    frac = scaled - i
    rgb = anchors[i] * (1.0 - frac) + anchors[i + 1] * frac
    return tuple(rgb.astype(int))


def load_samples() -> pd.DataFrame:
    if not SAMPLES_CSV.exists():
        raise FileNotFoundError(f"Missing plotting samples: {SAMPLES_CSV}")
    samples = pd.read_csv(SAMPLES_CSV, parse_dates=["obs_time", "valid_time"])
    return samples.dropna(subset=["lon", "lat", "obs_wind_speed", "lead_hour_from_case_start"]).copy()


def load_track() -> pd.DataFrame:
    if not TRACK_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRACK_CSV)
    if not {"dateUTC", "lonTC", "latTC"}.issubset(df.columns):
        return pd.DataFrame()
    track = pd.DataFrame(
        {
            "time": pd.to_datetime(df["dateUTC"].astype(str), format="%Y%m%d%H%M", errors="coerce"),
            "lon": pd.to_numeric(df["lonTC"], errors="coerce"),
            "lat": pd.to_numeric(df["latTC"], errors="coerce"),
        }
    ).dropna()
    return track[track["time"] >= CASE_START].sort_values("time").reset_index(drop=True)


def project(lon: float, lat: float, extent: tuple[float, float, float, float]) -> tuple[float, float]:
    lon_min, lon_max, lat_min, lat_max = extent
    x = PLOT_LEFT + (lon - lon_min) / (lon_max - lon_min) * (PLOT_RIGHT - PLOT_LEFT)
    y = PLOT_BOTTOM - (lat - lat_min) / (lat_max - lat_min) * (PLOT_BOTTOM - PLOT_TOP)
    return x, y


def draw_polyline(draw: ImageDraw.ImageDraw, coords: list[tuple[float, float]], extent: tuple[float, float, float, float], **kwargs) -> None:
    pts = [project(lon, lat, extent) for lon, lat in coords]
    if len(pts) >= 2:
        draw.line(pts, **kwargs)


def draw_polygon(draw: ImageDraw.ImageDraw, coords: list[tuple[float, float]], extent: tuple[float, float, float, float], fill, outline) -> None:
    pts = [project(lon, lat, extent) for lon, lat in coords]
    if len(pts) >= 3:
        draw.polygon(pts, fill=fill)
        draw.line(pts + [pts[0]], fill=outline, width=2)


def geometry_bounds_intersect(bounds, extent: tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    lon_min, lon_max, lat_min, lat_max = extent
    return not (maxx < lon_min or minx > lon_max or maxy < lat_min or miny > lat_max)


def draw_land_boundaries(draw: ImageDraw.ImageDraw, extent: tuple[float, float, float, float]) -> None:
    land_shp = shapereader.natural_earth(resolution="50m", category="physical", name="land")
    for record in shapereader.Reader(land_shp).records():
        geom = record.geometry
        if not geometry_bounds_intersect(geom.bounds, extent):
            continue
        geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in geoms:
            if not geometry_bounds_intersect(poly.bounds, extent):
                continue
            draw_polygon(draw, list(poly.exterior.coords), extent, fill=(232, 226, 211), outline=(122, 122, 122))

    coastline_shp = shapereader.natural_earth(resolution="50m", category="physical", name="coastline")
    for record in shapereader.Reader(coastline_shp).records():
        geom = record.geometry
        if not geometry_bounds_intersect(geom.bounds, extent):
            continue
        geoms = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in geoms:
            if geometry_bounds_intersect(line.bounds, extent):
                draw_polyline(draw, list(line.coords), extent, fill=(80, 80, 80), width=3)


def draw_centered_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill=(0, 0, 0)) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=fnt, fill=fill)


def draw_rotated_text(base: Image.Image, xy: tuple[int, int], text: str, fnt, angle: int, fill=(0, 0, 0)) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=fnt)
    w, h = bbox[2] - bbox[0] + 16, bbox[3] - bbox[1] + 16
    layer = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    d = ImageDraw.Draw(layer)
    d.text((8, 8), text, font=fnt, fill=fill)
    rotated = layer.rotate(angle, expand=True)
    base.alpha_composite(rotated, (xy[0], xy[1]))


def draw_axes(img: Image.Image, draw: ImageDraw.ImageDraw, extent: tuple[float, float, float, float]) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    draw.rectangle([PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM], outline=(0, 0, 0), width=4)
    lon_ticks = np.arange(math.ceil(lon_min), math.floor(lon_max) + 0.1, 2.0)
    lat_ticks = np.arange(math.ceil(lat_min), math.floor(lat_max) + 0.1, 2.0)
    for lon in lon_ticks:
        x, _ = project(lon, lat_min, extent)
        draw.line([(x, PLOT_TOP), (x, PLOT_BOTTOM)], fill=(218, 218, 218), width=2)
        draw.line([(x, PLOT_BOTTOM), (x, PLOT_BOTTOM + 12)], fill=(0, 0, 0), width=3)
        draw_centered_text(draw, (x, PLOT_BOTTOM + 45), f"{lon:.0f}", FONT_TICK)
    for lat in lat_ticks:
        _, y = project(lon_min, lat, extent)
        draw.line([(PLOT_LEFT, y), (PLOT_RIGHT, y)], fill=(218, 218, 218), width=2)
        draw.line([(PLOT_LEFT - 12, y), (PLOT_LEFT, y)], fill=(0, 0, 0), width=3)
        draw.text((PLOT_LEFT - 78, y - 18), f"{lat:.0f}", font=FONT_TICK, fill=(0, 0, 0))
    draw_centered_text(draw, ((PLOT_LEFT + PLOT_RIGHT) / 2, HEIGHT - 110), "Longitude (degree east)", FONT_AXIS)
    draw_rotated_text(img, (32, 775), "Latitude (degree north)", FONT_AXIS, 90)


def draw_colorbar(img: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    for i in range(COLORBAR_HEIGHT):
        lead = 72.0 * (1.0 - i / max(COLORBAR_HEIGHT - 1, 1))
        color = viridis_like(lead)
        draw.line(
            [(COLORBAR_LEFT, COLORBAR_TOP + i), (COLORBAR_LEFT + COLORBAR_WIDTH, COLORBAR_TOP + i)],
            fill=color,
            width=1,
        )
    draw.rectangle(
        [COLORBAR_LEFT, COLORBAR_TOP, COLORBAR_LEFT + COLORBAR_WIDTH, COLORBAR_TOP + COLORBAR_HEIGHT],
        outline=(0, 0, 0),
        width=3,
    )
    for lead in range(0, 73, 10):
        y = COLORBAR_TOP + COLORBAR_HEIGHT - lead / 72.0 * COLORBAR_HEIGHT
        draw.line([(COLORBAR_LEFT + COLORBAR_WIDTH, y), (COLORBAR_LEFT + COLORBAR_WIDTH + 12, y)], fill=(0, 0, 0), width=3)
        draw.text((COLORBAR_LEFT + COLORBAR_WIDTH + 20, y - 18), str(lead), font=FONT_TICK, fill=(0, 0, 0))
    draw_rotated_text(img, (COLORBAR_LEFT + 155, COLORBAR_TOP + 150), "Lead time from 2025-07-17 00 UTC (h)", FONT_AXIS, 90)


def draw_legends(draw: ImageDraw.ImageDraw) -> None:
    box = [PLOT_LEFT + 20, PLOT_BOTTOM - 245, PLOT_LEFT + 430, PLOT_BOTTOM - 25]
    draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255, 238), outline=(200, 200, 200), width=3)
    draw.text((box[0] + 18, box[1] + 14), "Observed wind speed", font=FONT_LEGEND_TITLE, fill=(0, 0, 0))
    for idx, ws in enumerate([5, 10, 15, 20]):
        y = box[1] + 70 + idx * 38
        r = math.sqrt(18.0 + ws * 7.0)
        draw.ellipse([box[0] + 95 - r, y - r, box[0] + 95 + r, y + r], fill=(112, 112, 112), outline=(255, 255, 255), width=2)
        draw.text((box[0] + 155, y - 18), f"{ws} m/s", font=FONT_LEGEND, fill=(0, 0, 0))

    box2 = [PLOT_RIGHT - 500, PLOT_TOP + 20, PLOT_RIGHT - 20, PLOT_TOP + 120]
    draw.rounded_rectangle(box2, radius=8, fill=(255, 255, 255, 238), outline=(200, 200, 200), width=3)
    draw.ellipse([box2[0] + 32, box2[1] + 25, box2[0] + 72, box2[1] + 65], fill=viridis_like(6), outline=(255, 255, 255), width=2)
    draw.text((box2[0] + 110, box2[1] + 25), "ICOADS observation", font=FONT_LEGEND, fill=(0, 0, 0))
    draw.line([(box2[0] + 30, box2[1] + 82), (box2[0] + 90, box2[1] + 82)], fill=(215, 48, 39), width=7)
    draw.ellipse([box2[0] + 55, box2[1] + 72, box2[0] + 75, box2[1] + 92], fill=(215, 48, 39))
    draw.text((box2[0] + 110, box2[1] + 66), "Best track", font=FONT_LEGEND, fill=(0, 0, 0))


def draw_track(draw: ImageDraw.ImageDraw, track: pd.DataFrame, extent: tuple[float, float, float, float]) -> None:
    if track.empty:
        return
    track_6h = track[((track["time"] - CASE_START).dt.total_seconds() % (6 * 3600) == 0)].copy()
    if track_6h.empty:
        track_6h = track.copy()
    pts = [project(row.lon, row.lat, extent) for row in track_6h.itertuples(index=False)]
    if len(pts) >= 2:
        draw.line(pts, fill=(215, 48, 39), width=8)
    for row, (x, y) in zip(track_6h.itertuples(index=False), pts):
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(215, 48, 39), outline=(215, 48, 39))

    labeled = track_6h[((track_6h["time"] - CASE_START).dt.total_seconds() % (12 * 3600) == 0)].copy()
    label_offsets = {
        0: (25, -32),
        12: (25, -32),
        24: (25, -32),
        36: (25, -32),
        48: (25, 20),
        60: (25, 20),
        72: (25, 20),
    }
    for row in labeled.itertuples(index=False):
        lead = int((row.time - CASE_START).total_seconds() // 3600)
        x, y = project(row.lon, row.lat, extent)
        draw.ellipse([x - 13, y - 13, x + 13, y + 13], fill=(215, 48, 39), outline=(255, 255, 255), width=3)
        label = f"{row.time:%m-%d %H}"
        dx, dy = label_offsets.get(lead, (25, -32))
        tx, ty = x + dx, y + dy
        bbox = draw.textbbox((tx, ty), label, font=FONT_SMALL)
        pad = 6
        rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
        draw.rounded_rectangle(rect, radius=5, fill=(255, 255, 255, 232), outline=(215, 48, 39), width=2)
        draw.line([(x, y), (rect[0] if dx >= 0 else rect[2], (rect[1] + rect[3]) / 2)], fill=(215, 48, 39), width=2)
        draw.text((tx, ty), label, font=FONT_SMALL, fill=(0, 0, 0))


def draw_samples(draw: ImageDraw.ImageDraw, samples: pd.DataFrame, extent: tuple[float, float, float, float]) -> None:
    for row in samples.itertuples(index=False):
        x, y = project(row.lon, row.lat, extent)
        size = 18.0 + min(max(float(row.obs_wind_speed), 0.0), 25.0) * 7.0
        r = math.sqrt(size) * 1.8
        color = viridis_like(float(row.lead_hour_from_case_start))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=2)


def main() -> None:
    samples = load_samples()
    track = load_track()
    extent_track = track[track["time"].between(CASE_START, CASE_END)].copy()
    lon_min = float(samples["lon"].min() - 2.0)
    lon_max = float(samples["lon"].max() + 2.0)
    lat_min = float(samples["lat"].min() - 2.0)
    lat_max = float(samples["lat"].max() + 2.0)
    if not extent_track.empty:
        lon_min = min(lon_min, float(extent_track["lon"].min() - 1.0))
        lon_max = max(lon_max, float(extent_track["lon"].max() + 1.0))
        lat_min = min(lat_min, float(extent_track["lat"].min() - 1.0))
        lat_max = max(lat_max, float(extent_track["lat"].max() + 1.0))
    extent = (lon_min, lon_max, lat_min, lat_max)

    img = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    draw.rectangle([PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM], fill=(248, 251, 252))
    land_layer = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 0))
    draw_land_boundaries(ImageDraw.Draw(land_layer, "RGBA"), extent)
    img.alpha_composite(land_layer.crop((PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM)), (PLOT_LEFT, PLOT_TOP))
    draw_axes(img, draw, extent)
    plot_layer = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 0))
    plot_draw = ImageDraw.Draw(plot_layer, "RGBA")
    draw_track(plot_draw, track, extent)
    draw_samples(plot_draw, samples, extent)
    img.alpha_composite(plot_layer.crop((PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM)), (PLOT_LEFT, PLOT_TOP))
    draw.rectangle([PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM], outline=(0, 0, 0), width=4)
    draw_colorbar(img, draw)
    draw_legends(draw)
    title = f"ICOADS samples within 600 km of Typhoon Wipha, n = {len(samples)}"
    title_bbox = draw.textbbox((0, 0), title, font=FONT_TITLE)
    draw.text(((WIDTH - (title_bbox[2] - title_bbox[0])) / 2, 45), title, font=FONT_TITLE, fill=(0, 0, 0))

    rgb = img.convert("RGB")
    rgb.save(PNG_OUT, dpi=(320, 320))
    rgb.save(PDF_OUT, resolution=320.0)
    print(PNG_OUT)
    print(PDF_OUT)
    print(f"samples={len(samples)}")
    print("track_points=6h")
    print("track_labels=12h")


if __name__ == "__main__":
    main()
