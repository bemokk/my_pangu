from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import pandas as pd

from plots.plot_wipha_track_buoy_locations import LOCAL_WIPHA_TRACK_CSV, load_wipha_track_csv
from plots.plot_wipha_pressure_eye_check import OUT_ROOT as PRESSURE_EYE_OUT_ROOT
from plots.wipha_case_common import (
    DATASET_COLORS,
    DATASET_LABELS,
    DATASETS,
    MAP_AREA,
    OUT_TRACK_ERROR_PNG,
    OUT_TRACK_ERROR_SVG,
    OUT_TRACK_ERRORS_CSV,
    OUT_TRACKS_CSV,
    ensure_dirs,
    haversine_km,
    interpolate_real_track_to_leads,
    set_plot_style,
)


FONT_SCALE = 1.25
FONT_FAMILY = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
TEXT_LABELS = {
    "real_track": "观测路径",
    "gdas_forecast": "GDAS实时预报",
    "era5_lagged_5d": "ERA5延迟5天预报",
    "track_panel": "(a) 2025-07-17 00 UTC起报的72 h路径预报",
    "error_panel": "(b) 路径位置误差",
    "lead_time": "预报时效（h）",
    "track_error": "路径误差（km）",
}
BASE_FONT_SIZES = {
    "default": 10,
    "title": 12,
    "axis_label": 10,
    "legend": 9,
    "tick": 9,
    "annotation": 7.5,
}
FONT_SIZES = {name: size * FONT_SCALE for name, size in BASE_FONT_SIZES.items()}


def load_official_wipha_track(csv_path: Path = LOCAL_WIPHA_TRACK_CSV) -> pd.DataFrame:
    return load_wipha_track_csv(csv_path)


def pressure_eye_track_csv_path(scheme: str) -> Path:
    return PRESSURE_EYE_OUT_ROOT / scheme / f"{scheme}_pressure_eye_positions.csv"


def _parse_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_pressure_eye_track(scheme: str, csv_path: Path | None = None) -> pd.DataFrame:
    csv_path = csv_path or pressure_eye_track_csv_path(scheme)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Pressure-eye track CSV not found for {scheme}: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"lead_hour", "valid_time", "model_eye_lon", "model_eye_lat"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Pressure-eye track CSV missing columns: {', '.join(missing)}")

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.len().eq(0)].copy()
    track = pd.DataFrame(
        {
            "scheme": scheme,
            "scheme_label": df.get("scheme_label", DATASET_LABELS.get(scheme, scheme)),
            "lead_hour": pd.to_numeric(df["lead_hour"], errors="coerce"),
            "valid_time": pd.to_datetime(df["valid_time"], errors="coerce"),
            "lon": pd.to_numeric(df["model_eye_lon"], errors="coerce"),
            "lat": pd.to_numeric(df["model_eye_lat"], errors="coerce"),
            "min_msl_hpa": pd.to_numeric(df.get("model_eye_msl_hpa", pd.Series([pd.NA] * len(df))), errors="coerce"),
            "manual_override": df.get("manual_override", pd.Series([False] * len(df))).map(_parse_bool),
            "source": str(csv_path),
        }
    )
    track = track.dropna(subset=["lead_hour", "valid_time", "lon", "lat"])
    track["lead_hour"] = track["lead_hour"].astype(int)
    return track.sort_values("lead_hour").reset_index(drop=True)


def build_tracks_and_errors_from_pressure_eye(
    real_track: pd.DataFrame,
    track_paths: dict[str, Path] | None = None,
    write_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_interp = interpolate_real_track_to_leads(real_track)
    forecast_tracks = []
    for scheme in DATASETS:
        path = track_paths.get(scheme) if track_paths else None
        forecast_tracks.append(load_pressure_eye_track(scheme, path))

    tracks = pd.concat([real_interp] + forecast_tracks, ignore_index=True, sort=False)
    errors = []
    for scheme_df in forecast_tracks:
        scheme = scheme_df["scheme"].iloc[0]
        merged = scheme_df.merge(
            real_interp[["lead_hour", "lon", "lat"]],
            on="lead_hour",
            suffixes=("_pred", "_obs"),
            how="left",
        )
        for row in merged.itertuples(index=False):
            errors.append(
                {
                    "scheme": scheme,
                    "scheme_label": DATASET_LABELS.get(scheme, scheme),
                    "lead_hour": int(row.lead_hour),
                    "valid_time": row.valid_time,
                    "pred_lon": row.lon_pred,
                    "pred_lat": row.lat_pred,
                    "obs_lon": row.lon_obs,
                    "obs_lat": row.lat_obs,
                    "track_error_km": haversine_km(row.lon_pred, row.lat_pred, row.lon_obs, row.lat_obs),
                    "manual_override": bool(getattr(row, "manual_override", False)),
                }
            )
    errors_df = pd.DataFrame(errors)
    if write_outputs:
        tracks.to_csv(OUT_TRACKS_CSV, index=False, encoding="utf-8-sig")
        errors_df.to_csv(OUT_TRACK_ERRORS_CSV, index=False, encoding="utf-8-sig")
    return tracks, errors_df


def plot_track_error(tracks: pd.DataFrame, errors: pd.DataFrame) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from land_mask import load_land_union

    set_plot_style()
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
        }
    )
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13.0, 5.8), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax_map = fig.add_subplot(grid[0, 0], projection=projection)
    ax_err = fig.add_subplot(grid[0, 1])

    ax_map.set_title(TEXT_LABELS["track_panel"], loc="left", fontweight="bold")
    ax_map.set_extent([MAP_AREA[0], MAP_AREA[1], MAP_AREA[2], MAP_AREA[3]], crs=projection)
    ax_map.set_facecolor("#EAF3F8")
    land_union = load_land_union(MAP_AREA[0], MAP_AREA[2], MAP_AREA[1], MAP_AREA[3])
    ax_map.add_geometries(
        [land_union],
        crs=projection,
        facecolor="#D7D2C3",
        edgecolor="#777777",
        linewidth=0.35,
        zorder=1,
    )
    ax_map.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.45, edgecolor="#333333", zorder=2)
    ax_map.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.25, edgecolor="#777777", zorder=2)
    gl = ax_map.gridlines(crs=projection, draw_labels=True, linewidth=0.32, color="#777777", alpha=0.38, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    style_map = {
        "real_track": {"label": TEXT_LABELS["real_track"], "color": "#222222", "marker": "o", "linestyle": "-"},
        "gdas_forecast": {"label": TEXT_LABELS["gdas_forecast"], "color": DATASET_COLORS["gdas_forecast"], "marker": "^", "linestyle": "--"},
        "era5_lagged_5d": {"label": TEXT_LABELS["era5_lagged_5d"], "color": DATASET_COLORS["era5_lagged_5d"], "marker": "s", "linestyle": "--"},
    }
    for scheme, style in style_map.items():
        sub = tracks[tracks["scheme"] == scheme].sort_values("lead_hour")
        if sub.empty:
            continue
        ax_map.plot(
            sub["lon"],
            sub["lat"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.0,
            label=style["label"],
            transform=projection,
            zorder=5,
        )
        for _, row in sub[sub["lead_hour"].isin([0, 24, 48, 72])].iterrows():
            if pd.notna(row["lon"]) and pd.notna(row["lat"]):
                ax_map.text(
                    row["lon"] + 0.12,
                    row["lat"] + 0.12,
                    f"+{int(row['lead_hour'])}h",
                    fontsize=FONT_SIZES["annotation"],
                    color=style["color"],
                    transform=projection,
                    zorder=6,
                )
    ax_map.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    ax_err.set_facecolor("#F4F5F7")
    ax_err.grid(True, color="white", linewidth=1.0)
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)
    ax_err.set_title(TEXT_LABELS["error_panel"], loc="left", fontweight="bold")
    for dataset in DATASETS:
        sub = errors[errors["scheme"] == dataset].sort_values("lead_hour")
        if sub.empty:
            continue
        ax_err.plot(sub["lead_hour"], sub["track_error_km"], color=DATASET_COLORS[dataset], marker="o", linewidth=1.8, markersize=4.0, label=TEXT_LABELS[dataset])
    ax_err.set_xlim(0, 72)
    ax_err.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax_err.set_xlabel(TEXT_LABELS["lead_time"])
    ax_err.set_ylabel(TEXT_LABELS["track_error"])
    ax_err.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.98])
    fig.savefig(OUT_TRACK_ERROR_PNG, bbox_inches="tight")
    fig.savefig(OUT_TRACK_ERROR_SVG, bbox_inches="tight")
    plt.close(fig)


def generate() -> list[Path]:
    ensure_dirs()
    real_track = load_official_wipha_track()
    tracks, errors = build_tracks_and_errors_from_pressure_eye(real_track)
    plot_track_error(tracks, errors)
    return [OUT_TRACKS_CSV, OUT_TRACK_ERRORS_CSV, OUT_TRACK_ERROR_PNG, OUT_TRACK_ERROR_SVG]


def main() -> None:
    for path in generate():
        print(path)


if __name__ == "__main__":
    main()
