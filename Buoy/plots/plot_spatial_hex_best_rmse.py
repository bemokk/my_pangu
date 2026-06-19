from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

from paths import FIGURES_DIR, WIND_MODEL_STATISTICS_DIR


FONT_SCALE = 1.0
FONT_FAMILY = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
TEXT_LABELS = {
    "era5_lagged_5d": "ERA5延迟5天预报",
    "era5_lagged_5d_short": "ERA5延迟5天",
    "gdas_forecast": "GDAS实时预报",
    "gdas_forecast_short": "GDAS",
    "insufficient_data": "匹配样本不足阈值",
    "no_matched_sample": "无匹配样本",
    "no_eligible_hexagons": "没有六边形满足样本阈值",
    "lead_panel": "({panel}) {lead_hour}h预报",
}
BASE_FONT_SIZES = {
    "default": 12,
    "title": 16,
    "axis_label": 15,
    "legend": 12,
    "tick": 18,
    "summary": 14,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}

AREA = [42, 103, 13, 130]
LAT_MAX, LON_MIN, LAT_MIN, LON_MAX = AREA
HEX_SIDE_DEG = 1.0

STATS_DIR = WIND_MODEL_STATISTICS_DIR / "wind_model_statistics_3_72h"
MATCHED_SAMPLES_CSV = STATS_DIR / "matched_buoy_model_wind_samples.csv"
OUT_CSV = STATS_DIR / "spatial_hex_best_rmse_era5_5d_vs_gdas_24_48_72h.csv"
OUT_PNG = FIGURES_DIR / "spatial_hex_best_rmse_era5_5d_vs_gdas_24_48_72h.png"
OUT_SVG = FIGURES_DIR / "spatial_hex_best_rmse_era5_5d_vs_gdas_24_48_72h.svg"

LEAD_HOURS = [24, 48, 72]
MIN_SAMPLES_PER_DATASET = 5

DATASET_STYLES = {
    "era5_lagged_5d": {
        "label": TEXT_LABELS["era5_lagged_5d"],
        "short_label": TEXT_LABELS["era5_lagged_5d_short"],
        "color": "#4C72B0",
    },
    "gdas_forecast": {
        "label": TEXT_LABELS["gdas_forecast"],
        "short_label": TEXT_LABELS["gdas_forecast_short"],
        "color": "#55A868",
    },
}
DATASET_ORDER = tuple(DATASET_STYLES)


def regular_hexagon(center_lon: float, center_lat: float, side_deg: float):
    from shapely.geometry import Polygon

    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    coords = [
        (center_lon + side_deg * np.cos(angle), center_lat + side_deg * np.sin(angle))
        for angle in angles
    ]
    return Polygon(coords)


def generate_hex_grid(area_polygon, side_deg: float) -> pd.DataFrame:
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


def rmse(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def ensure_speed_error(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    if "speed_error_ms" in records.columns:
        records["speed_error_ms"] = pd.to_numeric(records["speed_error_ms"], errors="coerce")
        return records

    required = {"pred_speed_ms", "obs_speed_ms"}
    missing = required - set(records.columns)
    if missing:
        raise KeyError(
            "Need speed_error_ms or pred_speed_ms/obs_speed_ms columns; "
            f"missing {sorted(missing)}"
        )

    records["pred_speed_ms"] = pd.to_numeric(records["pred_speed_ms"], errors="coerce")
    records["obs_speed_ms"] = pd.to_numeric(records["obs_speed_ms"], errors="coerce")
    records["speed_error_ms"] = records["pred_speed_ms"] - records["obs_speed_ms"]
    return records


def build_ocean_hex_grid() -> tuple[pd.DataFrame, object, object]:
    from land_mask import load_land_union
    from shapely.geometry import box

    area_polygon = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    land_union = load_land_union(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    ocean_area = area_polygon.difference(land_union)

    hexes = generate_hex_grid(area_polygon, HEX_SIDE_DEG)
    hexes = hexes.loc[hexes["geometry"].map(lambda geom: geom.intersects(ocean_area))].copy()
    return hexes, land_union, ocean_area


def load_matched_samples(
    csv_path: Path = MATCHED_SAMPLES_CSV,
    lead_hours: list[int] | tuple[int, ...] = LEAD_HOURS,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Matched sample CSV not found: {csv_path}")

    wanted_columns = {
        "record_id",
        "dataset",
        "dataset_label",
        "lead_hour",
        "latitude",
        "longitude",
        "obs_speed_ms",
        "pred_speed_ms",
        "speed_error_ms",
    }
    records = pd.read_csv(csv_path, usecols=lambda column: column in wanted_columns)

    required = {"dataset", "lead_hour", "latitude", "longitude"}
    missing = required - set(records.columns)
    if missing:
        raise KeyError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    records = ensure_speed_error(records)
    records["lead_hour"] = pd.to_numeric(records["lead_hour"], errors="coerce")
    records["latitude"] = pd.to_numeric(records["latitude"], errors="coerce")
    records["longitude"] = pd.to_numeric(records["longitude"], errors="coerce")

    records = records[
        records["dataset"].isin(DATASET_ORDER)
        & records["lead_hour"].isin(lead_hours)
        & records["latitude"].between(LAT_MIN, LAT_MAX)
        & records["longitude"].between(LON_MIN, LON_MAX)
    ].copy()
    records = records.dropna(subset=["lead_hour", "latitude", "longitude", "speed_error_ms"])
    records["lead_hour"] = records["lead_hour"].astype(int)
    return records


def assign_hex_ids(records: pd.DataFrame, hexes: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        out = records.copy()
        out["hex_id"] = pd.Series(dtype=int)
        return out

    points = records[["longitude", "latitude"]].to_numpy(dtype=float)
    assigned_hex = np.full(len(records), -1, dtype=int)
    unassigned = np.ones(len(records), dtype=bool)

    for hex_row in hexes.itertuples(index=False):
        coords = np.asarray(hex_row.geometry.exterior.coords)
        path = MplPath(coords)
        inside = path.contains_points(points, radius=1e-9) & unassigned
        if inside.any():
            assigned_hex[inside] = int(hex_row.hex_id)
            unassigned[inside] = False

    out = records.copy()
    out["hex_id"] = assigned_hex
    return out[out["hex_id"] >= 0].copy()


def compute_hex_rmse_winners(
    records: pd.DataFrame,
    hexes: pd.DataFrame,
    lead_hours: list[int] | tuple[int, ...] = LEAD_HOURS,
    min_samples_per_dataset: int = MIN_SAMPLES_PER_DATASET,
) -> pd.DataFrame:
    if min_samples_per_dataset < 1:
        raise ValueError("min_samples_per_dataset must be at least 1")

    required = {"lead_hour", "hex_id", "dataset"}
    missing = required - set(records.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    work = ensure_speed_error(records)
    work["lead_hour"] = pd.to_numeric(work["lead_hour"], errors="coerce")
    work["hex_id"] = pd.to_numeric(work["hex_id"], errors="coerce")
    work = work[
        work["dataset"].isin(DATASET_ORDER)
        & work["lead_hour"].isin(lead_hours)
    ].dropna(subset=["lead_hour", "hex_id", "speed_error_ms"])
    work["lead_hour"] = work["lead_hour"].astype(int)
    work["hex_id"] = work["hex_id"].astype(int)

    grouped = (
        work.groupby(["lead_hour", "hex_id", "dataset"], dropna=False)
        .agg(n=("speed_error_ms", "count"), rmse=("speed_error_ms", rmse))
        .reset_index()
    )
    metric_lookup = {
        (int(row.lead_hour), int(row.hex_id), row.dataset): (int(row.n), float(row.rmse))
        for row in grouped.itertuples(index=False)
    }

    if "record_id" in work.columns:
        observation_counts = (
            work.groupby(["lead_hour", "hex_id"])["record_id"].nunique().astype(int).to_dict()
        )
    else:
        observation_counts = (
            grouped.groupby(["lead_hour", "hex_id"])["n"].max().astype(int).to_dict()
        )

    rows = []
    hex_meta = hexes[["hex_id", "center_lon", "center_lat"]].copy()
    for lead_hour in lead_hours:
        for hex_row in hex_meta.itertuples(index=False):
            hex_id = int(hex_row.hex_id)
            row = {
                "lead_hour": int(lead_hour),
                "hex_id": hex_id,
                "center_lon": float(hex_row.center_lon),
                "center_lat": float(hex_row.center_lat),
                "observation_count": int(observation_counts.get((int(lead_hour), hex_id), 0)),
            }

            eligible = []
            for dataset in DATASET_ORDER:
                n, dataset_rmse = metric_lookup.get((int(lead_hour), hex_id, dataset), (0, np.nan))
                row[f"{dataset}_n"] = int(n)
                row[f"{dataset}_rmse"] = dataset_rmse
                if n >= min_samples_per_dataset and np.isfinite(dataset_rmse):
                    eligible.append((dataset, dataset_rmse))

            row["dataset_count_with_min_samples"] = len(eligible)
            if len(eligible) == len(DATASET_ORDER):
                eligible = sorted(eligible, key=lambda item: item[1])
                best_dataset, best_rmse = eligible[0]
                second_best_rmse = eligible[1][1]
                row["best_dataset"] = best_dataset
                row["best_dataset_label"] = DATASET_STYLES[best_dataset]["label"]
                row["best_rmse"] = best_rmse
                row["second_best_rmse"] = second_best_rmse
                row["best_rmse_margin"] = second_best_rmse - best_rmse
            else:
                row["best_dataset"] = "insufficient_data"
                row["best_dataset_label"] = TEXT_LABELS["insufficient_data"]
                row["best_rmse"] = np.nan
                row["second_best_rmse"] = np.nan
                row["best_rmse_margin"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def save_stats(stats: pd.DataFrame, csv_path: Path = OUT_CSV) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(csv_path, index=False, encoding="utf-8-sig")


def winner_summary_text(lead_stats: pd.DataFrame) -> str:
    valid = lead_stats[lead_stats["best_dataset"].isin(DATASET_ORDER)]
    if valid.empty:
        return TEXT_LABELS["no_eligible_hexagons"]

    total = len(valid)
    parts = []
    for dataset in DATASET_ORDER:
        count = int((valid["best_dataset"] == dataset).sum())
        percent = 100.0 * count / total
        parts.append(f"{DATASET_STYLES[dataset]['short_label']}: {count} ({percent:.1f}%)")
    return " | ".join(parts)


def plot_best_rmse_hexes(
    stats: pd.DataFrame,
    hexes: pd.DataFrame,
    land_union,
    png_path: Path = OUT_PNG,
    svg_path: Path = OUT_SVG,
) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon as MplPolygon

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT_SIZES["default"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(18.5, 6.7))
    map_bottom = 0.13
    map_height = 0.80
    map_gap = 0.024
    map_width = (
        map_height
        * (LON_MAX - LON_MIN)
        / (LAT_MAX - LAT_MIN)
        * (fig.get_figheight() / fig.get_figwidth())
    )
    map_left = 0.035
    map_axes = np.asarray(
        [
            fig.add_axes(
                [map_left + panel_index * (map_width + map_gap), map_bottom, map_width, map_height],
                projection=projection,
            )
            for panel_index in range(len(LEAD_HOURS))
        ]
    )

    for panel_index, (ax, lead_hour) in enumerate(zip(map_axes, LEAD_HOURS)):
        lead_stats = stats[stats["lead_hour"] == lead_hour].set_index("hex_id")
        ax.set_anchor("W")
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=projection)
        ax.set_facecolor("white")

        for hex_row in hexes.itertuples(index=False):
            hex_id = int(hex_row.hex_id)
            if hex_id in lead_stats.index:
                best_dataset = lead_stats.at[hex_id, "best_dataset"]
                observation_count = int(lead_stats.at[hex_id, "observation_count"])
            else:
                best_dataset = "insufficient_data"
                observation_count = 0

            if best_dataset in DATASET_STYLES:
                facecolor = DATASET_STYLES[best_dataset]["color"]
                alpha = 0.9
            elif observation_count > 0:
                facecolor = "#D7D7D7"
                alpha = 0.82
            else:
                facecolor = "white"
                alpha = 1.0

            patch = MplPolygon(
                np.asarray(hex_row.geometry.exterior.coords),
                closed=True,
                facecolor=facecolor,
                edgecolor="#333333",
                linewidth=0.35,
                alpha=alpha,
                transform=projection,
                zorder=2,
            )
            ax.add_patch(patch)

        ax.add_geometries(
            [land_union],
            crs=projection,
            facecolor="#B8B8A6",
            edgecolor="#555555",
            linewidth=0.35,
            zorder=3,
        )
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.38, edgecolor="#333333", zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.22, edgecolor="#777777", zorder=4)

        gl = ax.gridlines(
            crs=projection,
            draw_labels=True,
            linewidth=0.24,
            color="#777777",
            alpha=0.35,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        if panel_index > 0:
            gl.left_labels = False

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            TEXT_LABELS["lead_panel"].format(panel=panel_letter, lead_hour=lead_hour),
            loc="left",
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.02,
            winner_summary_text(lead_stats.reset_index()),
            ha="left",
            va="bottom",
            fontsize=FONT_SIZES["summary"],
            color="#333333",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.0},
            zorder=6,
        )

    handles = [
        Patch(
            facecolor=style["color"],
            edgecolor="#333333",
            label=style["label"],
        )
        for style in DATASET_STYLES.values()
    ]
    handles.extend(
        [
            Patch(facecolor="#D7D7D7", edgecolor="#333333", label=TEXT_LABELS["insufficient_data"]),
            Patch(facecolor="white", edgecolor="#333333", label=TEXT_LABELS["no_matched_sample"]),
        ]
    )
    map_axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#777777",
        borderaxespad=0.0,
    )
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    samples = load_matched_samples()
    hexes, land_union, _ = build_ocean_hex_grid()
    assigned_samples = assign_hex_ids(samples, hexes)
    stats = compute_hex_rmse_winners(
        assigned_samples,
        hexes,
        lead_hours=LEAD_HOURS,
        min_samples_per_dataset=MIN_SAMPLES_PER_DATASET,
    )
    save_stats(stats)
    plot_best_rmse_hexes(stats, hexes, land_union)

    print(f"Matched samples loaded: {len(samples):,}")
    print(f"Matched samples assigned to hexagons: {len(assigned_samples):,}")
    print(f"Ocean hexagons: {len(hexes):,}")
    print(f"Lead hours: {LEAD_HOURS}")
    print(f"Minimum samples per dataset: {MIN_SAMPLES_PER_DATASET}")
    print(f"CSV saved: {OUT_CSV}")
    print(f"PNG saved: {OUT_PNG}")
    print(f"SVG saved: {OUT_SVG}")


if __name__ == "__main__":
    main()
