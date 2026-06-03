from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.plot_wipha_track_buoy_locations import load_wipha_track_csv
from plots.wipha_case_common import (
    DATASET_LABELS,
    FIGURES_DIR,
    GRID_DX,
    GRID_DY,
    GRID_LAT0,
    GRID_LON0,
    GRID_NLAT,
    GRID_NLON,
    TRACK_INIT,
    WIPHA_SEARCH_BOX,
    grid_indices_for_box,
    interpolate_real_track_to_leads,
    moving_box,
    read_surface_msl,
    set_plot_style,
    surface_array_path,
)

PLOT_BOX = (105.0, 130.0, 10.0, 27.5)
OUT_ROOT = FIGURES_DIR / "wipha_pressure_eye_check"
SCHEME_ALIASES = {
    "gdas": "gdas_forecast",
    "gdas_forecast": "gdas_forecast",
    "era5": "era5_lagged_5d",
    "era5_lagged": "era5_lagged_5d",
    "era5_lagged_5d": "era5_lagged_5d",
}
DEFAULT_SCHEMES = ["gdas_forecast", "era5_lagged_5d"]
MANUAL_EYE_COLUMNS = {"scheme", "lead_hour", "lon", "lat"}


def pressure_eye_check_leads() -> list[int]:
    return list(range(0, 73, 3))


def normalize_scheme(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key not in SCHEME_ALIASES:
        raise ValueError(f"Unsupported scheme: {value}")
    return SCHEME_ALIASES[key]


def schemes_to_run(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return DEFAULT_SCHEMES.copy()
    return [normalize_scheme(value)]


def scheme_output_dir(out_root: Path, scheme: str) -> Path:
    return Path(out_root) / scheme


def load_manual_eye_overrides(csv_path: Path | None) -> dict[tuple[str, int], dict[str, float]]:
    if csv_path is None:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manual eye override CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = sorted(MANUAL_EYE_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Manual eye override CSV missing columns: {', '.join(missing)}")

    valid_leads = set(pressure_eye_check_leads())
    overrides: dict[tuple[str, int], dict[str, float]] = {}
    for row in df.itertuples(index=False):
        scheme = normalize_scheme(getattr(row, "scheme"))
        lead_hour = int(getattr(row, "lead_hour"))
        if lead_hour not in valid_leads:
            raise ValueError(f"Manual eye override lead_hour must be one of {sorted(valid_leads)}: {lead_hour}")
        lon = float(getattr(row, "lon"))
        lat = float(getattr(row, "lat"))
        if not (PLOT_BOX[0] <= lon <= PLOT_BOX[1] and PLOT_BOX[2] <= lat <= PLOT_BOX[3]):
            raise ValueError(f"Manual eye override is outside plot domain for {scheme} +{lead_hour}h: lon={lon}, lat={lat}")
        overrides[(scheme, lead_hour)] = {"lon": lon, "lat": lat}
    return overrides


def upsert_manual_eye_override(csv_path: Path, scheme: str, lead_hour: int, lon: float, lat: float) -> Path:
    scheme = normalize_scheme(scheme)
    lead_hour = int(lead_hour)
    lon = float(lon)
    lat = float(lat)
    valid_leads = set(pressure_eye_check_leads())
    if lead_hour not in valid_leads:
        raise ValueError(f"Manual eye override lead_hour must be one of {sorted(valid_leads)}: {lead_hour}")
    if not (PLOT_BOX[0] <= lon <= PLOT_BOX[1] and PLOT_BOX[2] <= lat <= PLOT_BOX[3]):
        raise ValueError(f"Manual eye override is outside plot domain for {scheme} +{lead_hour}h: lon={lon}, lat={lat}")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        missing = sorted(MANUAL_EYE_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Manual eye override CSV missing columns: {', '.join(missing)}")
        df = df[list(MANUAL_EYE_COLUMNS)].copy()
        df["scheme"] = df["scheme"].map(normalize_scheme)
        df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="raise").astype(int)
        df["lon"] = pd.to_numeric(df["lon"], errors="raise").astype(float)
        df["lat"] = pd.to_numeric(df["lat"], errors="raise").astype(float)
        df = df[~((df["scheme"] == scheme) & (df["lead_hour"] == lead_hour))]
    else:
        df = pd.DataFrame(columns=["scheme", "lead_hour", "lon", "lat"])

    new_row = pd.DataFrame([{"scheme": scheme, "lead_hour": lead_hour, "lon": lon, "lat": lat}])
    out = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    out = out[["scheme", "lead_hour", "lon", "lat"]].sort_values(["scheme", "lead_hour"]).reset_index(drop=True)
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def _msl_to_hpa(msl: np.ndarray) -> np.ndarray:
    out = np.asarray(msl, dtype=float)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(out)
    return out / 100.0 if np.isfinite(median) and median > 2000 else out


def nearest_msl_hpa_at_point(msl: np.ndarray, lon: float, lat: float) -> float:
    lons = GRID_LON0 + GRID_DX * np.arange(GRID_NLON)
    lats = GRID_LAT0 + GRID_DY * np.arange(GRID_NLAT)
    lon_index = int(np.argmin(np.abs(lons - lon)))
    lat_index = int(np.argmin(np.abs(lats - lat)))
    value = float(np.asarray(msl)[lat_index, lon_index])
    return float(_msl_to_hpa(np.asarray([value]))[0])


def subset_msl_hpa(msl: np.ndarray, plot_box: tuple[float, float, float, float] = PLOT_BOX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_idx, lon_idx, lats, lons = grid_indices_for_box(plot_box)
    sub = np.asarray(msl[np.ix_(lat_idx, lon_idx)], dtype=float)
    return _msl_to_hpa(sub), lats, lons


def locate_center_from_msl(msl: np.ndarray, search_box: tuple[float, float, float, float]) -> dict:
    lat_idx, lon_idx, lats, lons = grid_indices_for_box(search_box)
    sub = _msl_to_hpa(np.asarray(msl[np.ix_(lat_idx, lon_idx)], dtype=float))
    if sub.size == 0 or np.isnan(sub).all():
        raise ValueError(f"No valid MSLP values in search box: {search_box}")
    flat = np.nanargmin(sub)
    i, j = np.unravel_index(flat, sub.shape)
    return {
        "center_lon": float(lons[j]),
        "center_lat": float(lats[i]),
        "min_msl_hpa": float(sub[i, j]),
        "search_box": str(search_box),
        "manual_override": False,
    }


def apply_manual_eye_override(
    model_eye: dict,
    manual_eye_overrides: dict[tuple[str, int], dict[str, float]],
    *,
    scheme: str,
    lead_hour: int,
    manual_msl_hpa: float | None = None,
) -> dict:
    key = (normalize_scheme(scheme), int(lead_hour))
    if key not in manual_eye_overrides:
        out = model_eye.copy()
        out["manual_override"] = False
        return out

    manual_eye = manual_eye_overrides[key]
    out = model_eye.copy()
    out["center_lon"] = float(manual_eye["lon"])
    out["center_lat"] = float(manual_eye["lat"])
    out["min_msl_hpa"] = float(manual_msl_hpa) if manual_msl_hpa is not None else float(model_eye.get("min_msl_hpa", np.nan))
    out["manual_override"] = True
    return out


def _pressure_levels(msl_hpa: np.ndarray) -> np.ndarray:
    low = float(np.nanpercentile(msl_hpa, 1.0))
    high = float(np.nanpercentile(msl_hpa, 99.0))
    low = np.floor(low / 2.0) * 2.0
    high = np.ceil(high / 2.0) * 2.0
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.linspace(float(np.nanmin(msl_hpa)), float(np.nanmax(msl_hpa)), 12)
    return np.arange(low, high + 2.0, 2.0)


def _official_track_by_lead() -> pd.DataFrame:
    real_track = load_wipha_track_csv()
    return interpolate_real_track_to_leads(real_track).set_index("lead_hour")


def _format_valid_time(valid_time: pd.Timestamp) -> str:
    return pd.Timestamp(valid_time).strftime("%Y-%m-%d %H:%M UTC")


def plot_pressure_eye_panel(
    *,
    scheme: str,
    lead_hour: int,
    valid_time: pd.Timestamp,
    msl_hpa: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    model_eye: dict,
    official_eye: pd.Series | None,
    output_path: Path,
) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    from land_mask import load_land_union

    set_plot_style()
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.8, 6.4))
    ax = plt.axes(projection=projection)
    ax.set_extent([PLOT_BOX[0], PLOT_BOX[1], PLOT_BOX[2], PLOT_BOX[3]], crs=projection)
    ax.set_facecolor("#EAF3F8")

    land_union = load_land_union(PLOT_BOX[0], PLOT_BOX[2], PLOT_BOX[1], PLOT_BOX[3])
    ax.add_geometries(
        [land_union],
        crs=projection,
        facecolor="#D7D2C3",
        edgecolor="#777777",
        linewidth=0.35,
        zorder=3,
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.45, edgecolor="#333333", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.25, edgecolor="#777777", zorder=4)

    levels = _pressure_levels(msl_hpa)
    mesh = ax.contourf(
        lons,
        lats,
        msl_hpa,
        levels=levels,
        cmap="Spectral_r",
        extend="both",
        transform=projection,
        zorder=1,
    )
    contour_step = 4.0 if len(levels) > 20 else 2.0
    contour_levels = np.arange(np.nanmin(levels), np.nanmax(levels) + contour_step, contour_step)
    contours = ax.contour(
        lons,
        lats,
        msl_hpa,
        levels=contour_levels,
        colors="#4A4A4A",
        linewidths=0.42,
        alpha=0.72,
        transform=projection,
        zorder=2,
    )
    ax.clabel(contours, inline=True, fmt="%.0f", fontsize=6.4)

    ax.scatter(
        model_eye["center_lon"],
        model_eye["center_lat"],
        marker="*",
        s=145,
        color="#D62728",
        edgecolor="white",
        linewidth=0.75,
        label="Model eye (manual)" if model_eye.get("manual_override") else "Model eye (MSLP minimum)",
        transform=projection,
        zorder=7,
    )
    eye_text = f"{model_eye['min_msl_hpa']:.1f} hPa"
    if model_eye.get("manual_override"):
        eye_text = f"{eye_text}\nmanual"
    ax.text(
        model_eye["center_lon"] + 0.22,
        model_eye["center_lat"] + 0.18,
        eye_text,
        fontsize=8.2,
        color="#B5161B",
        transform=projection,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.4},
        zorder=8,
    )

    if official_eye is not None and pd.notna(official_eye.get("lon")) and pd.notna(official_eye.get("lat")):
        ax.scatter(
            official_eye["lon"],
            official_eye["lat"],
            marker="X",
            s=76,
            color="#111111",
            edgecolor="white",
            linewidth=0.65,
            label="Official eye",
            transform=projection,
            zorder=8,
        )
        ax.text(
            official_eye["lon"] + 0.22,
            official_eye["lat"] - 0.32,
            "Official",
            fontsize=8.0,
            color="#111111",
            transform=projection,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=9,
        )

    gl = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.32,
        color="#777777",
        alpha=0.38,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.82, pad=0.035)
    cbar.set_label("Mean sea-level pressure (hPa)")

    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax.set_title(
        f"{DATASET_LABELS.get(scheme, scheme)} MSLP and Typhoon Wipha Eye Check",
        loc="left",
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.035,
        f"Valid: {_format_valid_time(valid_time)}    Lead: +{lead_hour:02d} h    Domain: 105-130E, 10-27.5N",
        fontsize=9.0,
        color="#333333",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_scheme_pressure_eye_check(
    scheme: str,
    out_root: Path = OUT_ROOT,
    manual_eye_overrides: dict[tuple[str, int], dict[str, float]] | None = None,
) -> list[Path]:
    scheme = normalize_scheme(scheme)
    manual_eye_overrides = manual_eye_overrides or {}
    out_dir = scheme_output_dir(out_root, scheme)
    out_dir.mkdir(parents=True, exist_ok=True)
    official_by_lead = _official_track_by_lead()

    generated: list[Path] = []
    rows = []
    last_center: tuple[float, float] | None = None
    for lead_hour in pressure_eye_check_leads():
        valid_time = pd.Timestamp(TRACK_INIT + timedelta(hours=lead_hour))
        path = surface_array_path(scheme, lead_hour, valid_time.to_pydatetime())
        search_box = WIPHA_SEARCH_BOX if last_center is None else moving_box(*last_center)
        row = {
            "scheme": scheme,
            "scheme_label": DATASET_LABELS.get(scheme, scheme),
            "lead_hour": lead_hour,
            "valid_time": valid_time,
            "surface_file": str(path),
            "search_box": str(search_box),
            "auto_model_eye_lon": np.nan,
            "auto_model_eye_lat": np.nan,
            "auto_model_eye_msl_hpa": np.nan,
            "model_eye_lon": np.nan,
            "model_eye_lat": np.nan,
            "model_eye_msl_hpa": np.nan,
            "manual_override": False,
            "official_eye_lon": np.nan,
            "official_eye_lat": np.nan,
            "figure": "",
            "error": "",
        }
        official_eye = official_by_lead.loc[lead_hour] if lead_hour in official_by_lead.index else None
        if official_eye is not None:
            row["official_eye_lon"] = official_eye.get("lon", np.nan)
            row["official_eye_lat"] = official_eye.get("lat", np.nan)
        try:
            msl = read_surface_msl(path)
            auto_model_eye = locate_center_from_msl(msl, search_box)
            manual_key = (scheme, lead_hour)
            manual_msl_hpa = None
            if manual_key in manual_eye_overrides:
                manual_msl_hpa = nearest_msl_hpa_at_point(
                    msl,
                    manual_eye_overrides[manual_key]["lon"],
                    manual_eye_overrides[manual_key]["lat"],
                )
            model_eye = apply_manual_eye_override(
                auto_model_eye,
                manual_eye_overrides,
                scheme=scheme,
                lead_hour=lead_hour,
                manual_msl_hpa=manual_msl_hpa,
            )
            last_center = (model_eye["center_lon"], model_eye["center_lat"])
            msl_hpa, lats, lons = subset_msl_hpa(msl)
            output_path = out_dir / f"{scheme}_{valid_time.strftime('%Y-%m-%d-%H-%M')}_lead{lead_hour:03d}.png"
            plot_pressure_eye_panel(
                scheme=scheme,
                lead_hour=lead_hour,
                valid_time=valid_time,
                msl_hpa=msl_hpa,
                lats=lats,
                lons=lons,
                model_eye=model_eye,
                official_eye=official_eye,
                output_path=output_path,
            )
            row["auto_model_eye_lon"] = auto_model_eye["center_lon"]
            row["auto_model_eye_lat"] = auto_model_eye["center_lat"]
            row["auto_model_eye_msl_hpa"] = auto_model_eye["min_msl_hpa"]
            row["model_eye_lon"] = model_eye["center_lon"]
            row["model_eye_lat"] = model_eye["center_lat"]
            row["model_eye_msl_hpa"] = model_eye["min_msl_hpa"]
            row["manual_override"] = bool(model_eye.get("manual_override"))
            row["figure"] = str(output_path)
            generated.append(output_path)
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    summary_path = out_dir / f"{scheme}_pressure_eye_positions.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    generated.append(summary_path)
    return generated


def generate(scheme: str = "all", out_root: Path = OUT_ROOT, manual_eye_csv: Path | None = None) -> list[Path]:
    manual_eye_overrides = load_manual_eye_overrides(manual_eye_csv)
    generated: list[Path] = []
    for item in schemes_to_run(scheme):
        generated.extend(generate_scheme_pressure_eye_check(item, out_root, manual_eye_overrides))
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Wipha MSLP fields and compare model-detected typhoon eye positions with official positions.",
    )
    parser.add_argument(
        "--scheme",
        default="all",
        help="Experiment to plot: all, gdas, gdas_forecast, era5, era5_lagged, or era5_lagged_5d.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_ROOT,
        help="Root directory for generated pressure-eye-check figures.",
    )
    parser.add_argument(
        "--manual-eye-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns scheme,lead_hour,lon,lat to override selected model eye positions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in generate(args.scheme, args.output_dir, args.manual_eye_csv):
        print(path)


if __name__ == "__main__":
    main()
