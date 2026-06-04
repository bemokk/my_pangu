from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "wipha_track_relative_analysis"
MATCHED_CSV = PROJECT_ROOT / "Buoy" / "results" / "wipha_track_relative_analysis" / "data" / "matched_icoads_model_samples.csv"
TRACK_CSV = PROJECT_ROOT / "Buoy" / "results" / "wipha_track_relative_analysis" / "data" / "typhoon_2506_Wipha.csv"

CASE_START_LABEL = "2025-07-17 00 UTC"
LEAD_HOURS = list(range(0, 73, 3))
MODEL_ORDER = ["GDAS_Realtime", "ERA5_Lagged"]
MODEL_COLORS = {"GDAS_Realtime": "#6FA8DC", "ERA5_Lagged": "#F4A582"}

PLOTTING_SAMPLES_CSV = OUT_DIR / "plotting_samples_0_600km.csv"
OVERALL_STATS_CSV = OUT_DIR / "overall_wind_speed_error_statistics.csv"
PERIOD_STATS_CSV = OUT_DIR / "period_wind_error_statistics.csv"
SUPPLEMENTARY_LEAD_STATS_CSV = OUT_DIR / "supplementary_lead_hour_statistics.csv"
NOTES_MD = OUT_DIR / "plotting_notes.md"


def load_and_filter_samples() -> pd.DataFrame:
    if not MATCHED_CSV.exists():
        raise FileNotFoundError(f"Missing matched sample CSV: {MATCHED_CSV}")
    df = pd.read_csv(MATCHED_CSV, parse_dates=["obs_time", "valid_time"])
    required = [
        "distance_to_typhoon_center_km",
        "lead_hour_from_case_start",
        "obs_wind_speed",
        "gdas_wind_speed",
        "era5lagged_wind_speed",
        "lon",
        "lat",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Matched sample CSV missing required columns: {missing}")

    speed_cols = ["obs_wind_speed", "gdas_wind_speed", "era5lagged_wind_speed"]
    mask = (
        (df["distance_to_typhoon_center_km"] <= 600.0)
        & df["lead_hour_from_case_start"].between(0, 72)
        & df[speed_cols].notna().all(axis=1)
        & df[speed_cols].apply(lambda s: s.between(0.0, 75.0)).all(axis=1)
    )
    out = df.loc[mask].copy()
    out["gdas_speed_error"] = out["gdas_wind_speed"] - out["obs_wind_speed"]
    out["era5lagged_speed_error"] = out["era5lagged_wind_speed"] - out["obs_wind_speed"]
    out["gdas_abs_error"] = out["gdas_speed_error"].abs()
    out["era5lagged_abs_error"] = out["era5lagged_speed_error"].abs()
    out["gdas_sq_error"] = out["gdas_speed_error"] ** 2
    out["era5lagged_sq_error"] = out["era5lagged_speed_error"] ** 2
    out["coarse_radius_bin"] = np.select(
        [
            (out["distance_to_typhoon_center_km"] >= 0.0) & (out["distance_to_typhoon_center_km"] < 400.0),
            (out["distance_to_typhoon_center_km"] >= 400.0) & (out["distance_to_typhoon_center_km"] <= 600.0),
        ],
        ["0-400 km", "400-600 km"],
        default="outside_0_600_km",
    )
    out["forecast_period"] = np.select(
        [
            (out["lead_hour_from_case_start"] >= 18) & (out["lead_hour_from_case_start"] < 48),
            (out["lead_hour_from_case_start"] >= 48) & (out["lead_hour_from_case_start"] <= 72),
        ],
        ["18-48 h", "48-72 h"],
        default="outside_18_72_h",
    )
    out = out.sort_values(["lead_hour_from_case_start", "obs_time", "platform_id", "lat", "lon"]).reset_index(drop=True)
    out.to_csv(PLOTTING_SAMPLES_CSV, index=False, encoding="utf-8-sig")
    return out


def load_track() -> pd.DataFrame | None:
    if not TRACK_CSV.exists():
        return None
    track = pd.read_csv(TRACK_CSV)
    required = {"dateUTC", "latTC", "lonTC"}
    if not required.issubset(track.columns):
        return None
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(track["dateUTC"].astype(str), format="%Y%m%d%H%M", errors="coerce"),
            "lon": pd.to_numeric(track["lonTC"], errors="coerce"),
            "lat": pd.to_numeric(track["latTC"], errors="coerce"),
        }
    ).dropna()
    start = pd.Timestamp("2025-07-17 00:00:00")
    end = start + pd.Timedelta(hours=72)
    out = out[out["time"].between(start, end)]
    return out.sort_values("time") if not out.empty else None


def stats_from_errors(model: str, error: pd.Series) -> dict[str, float | int | str]:
    err = pd.to_numeric(error, errors="coerce").dropna()
    abs_err = err.abs()
    return {
        "model": model,
        "n": int(len(err)),
        "bias": float(err.mean()) if len(err) else np.nan,
        "mae": float(abs_err.mean()) if len(err) else np.nan,
        "rmse": float(math.sqrt(np.mean(err**2))) if len(err) else np.nan,
        "median_abs_error": float(abs_err.median()) if len(err) else np.nan,
    }


def compute_overall_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        [
            stats_from_errors("GDAS_Realtime", samples["gdas_speed_error"]),
            stats_from_errors("ERA5_Lagged", samples["era5lagged_speed_error"]),
        ]
    )
    stats.to_csv(OVERALL_STATS_CSV, index=False, encoding="utf-8-sig")
    return stats


def compute_period_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("18-48 h", samples[(samples["lead_hour_from_case_start"] >= 18) & (samples["lead_hour_from_case_start"] < 48)]),
        ("48-72 h", samples[(samples["lead_hour_from_case_start"] >= 48) & (samples["lead_hour_from_case_start"] <= 72)]),
        ("18-72 h", samples[(samples["lead_hour_from_case_start"] >= 18) & (samples["lead_hour_from_case_start"] <= 72)]),
    ]
    rows: list[dict[str, float | int | str | bool]] = []
    for period, data in periods:
        gdas = stats_from_errors("GDAS_Realtime", data["gdas_speed_error"])
        era5 = stats_from_errors("ERA5_Lagged", data["era5lagged_speed_error"])
        if pd.notna(era5["rmse"]) and era5["rmse"] != 0:
            change = (float(era5["rmse"]) - float(gdas["rmse"])) / float(era5["rmse"]) * 100.0
        else:
            change = np.nan
        for row in [gdas, era5]:
            row["forecast_period"] = period
            row["low_sample"] = int(row["n"]) < 10
            row["rmse_change_percent_relative_to_era5lagged"] = change
            rows.append(row)
    cols = [
        "forecast_period",
        "model",
        "n",
        "bias",
        "mae",
        "rmse",
        "low_sample",
        "rmse_change_percent_relative_to_era5lagged",
        "median_abs_error",
    ]
    stats = pd.DataFrame(rows)[cols]
    stats.to_csv(PERIOD_STATS_CSV, index=False, encoding="utf-8-sig")
    return stats


def compute_supplementary_lead_statistics(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead in LEAD_HOURS:
        data = samples[samples["lead_hour_from_case_start"] == lead]
        gdas = stats_from_errors("GDAS_Realtime", data["gdas_speed_error"])
        era5 = stats_from_errors("ERA5_Lagged", data["era5lagged_speed_error"])
        for row in [gdas, era5]:
            row["lead_hour_from_case_start"] = lead
            row["low_sample"] = int(row["n"]) < 10
            rows.append(row)
    stats = pd.DataFrame(rows)
    cols = ["lead_hour_from_case_start", "model", "n", "bias", "mae", "rmse", "median_abs_error", "low_sample"]
    stats = stats[cols]
    stats.to_csv(SUPPLEMENTARY_LEAD_STATS_CSV, index=False, encoding="utf-8-sig")
    return stats


def annotate_bars(ax: plt.Axes, bars, labels: list[str], pad: float = 0.03, rotation: float = 0.0, fontsize: int = 8) -> None:
    ymax = ax.get_ylim()[1]
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * pad,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_sample_coverage(samples: pd.DataFrame, track: pd.DataFrame | None) -> list[Path]:
    lon_min, lon_max = samples["lon"].min() - 2.0, samples["lon"].max() + 2.0
    lat_min, lat_max = samples["lat"].min() - 2.0, samples["lat"].max() + 2.0
    sizes = 18.0 + samples["obs_wind_speed"].clip(0, 25) * 7.0
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(8.0, 6.2))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ECE6D8", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, alpha=0.5, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.55", alpha=0.55, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        scatter = ax.scatter(
            samples["lon"],
            samples["lat"],
            c=samples["lead_hour_from_case_start"],
            s=sizes,
            cmap="viridis",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            transform=ccrs.PlateCarree(),
            label="ICOADS observation",
            zorder=4,
        )
        if track is not None:
            ax.plot(track["lon"], track["lat"], color="#D73027", linewidth=1.7, marker="o", markersize=2.5, transform=ccrs.PlateCarree(), label="Best track", zorder=5)
    else:
        fig, ax = plt.subplots(figsize=(8.0, 6.2))
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
        scatter = ax.scatter(
            samples["lon"],
            samples["lat"],
            c=samples["lead_hour_from_case_start"],
            s=sizes,
            cmap="viridis",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            label="ICOADS observation",
            zorder=4,
        )
        if track is not None:
            ax.plot(track["lon"], track["lat"], color="#D73027", linewidth=1.7, marker="o", markersize=2.5, label="Best track", zorder=5)
        ax.set_xlabel("Longitude (degree east)")
        ax.set_ylabel("Latitude (degree north)")

    ax.set_title(f"ICOADS samples within 600 km of Typhoon Wipha, n = {len(samples)}", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label(f"Lead time from {CASE_START_LABEL} (h)")

    handles = [
        plt.Line2D([], [], linestyle="", marker="o", markersize=math.sqrt(18.0 + ws * 7.0), markerfacecolor="0.45", markeredgecolor="white", label=f"{ws} m/s")
        for ws in [5, 10, 15, 20]
    ]
    speed_legend = ax.legend(handles=handles, title="Observed wind speed", loc="lower left", frameon=True, fontsize=8, title_fontsize=8)
    ax.add_artist(speed_legend)
    if track is not None:
        ax.legend(loc="upper right", frameon=True, fontsize=8)
    return save_figure(fig, "figure_icoads_sample_coverage")


def plot_sample_count_distribution(samples: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    fig.suptitle("Sample distribution of ICOADS observations used for verification", fontsize=13, fontweight="bold")

    lead_counts = samples.groupby("lead_hour_from_case_start").size().reindex(LEAD_HOURS, fill_value=0)
    bars = axes[0].bar(lead_counts.index.astype(str), lead_counts.values, color="#8FBBD9", edgecolor="white")
    axes[0].set_title("(a) Samples by lead hour", loc="left", fontweight="bold")
    axes[0].set_xlabel("Lead time from case start (h)")
    axes[0].set_ylabel("Sample count")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set_ylim(0, max(lead_counts.max() * 1.35, 1))
    lead_labels = [f"n={int(v)}" + ("*" if 0 < v < 10 else "") if v > 0 else "0" for v in lead_counts.values]
    annotate_bars(axes[0], bars, lead_labels, pad=0.015, rotation=90, fontsize=7)

    radius_counts = samples["coarse_radius_bin"].value_counts().reindex(["0-400 km", "400-600 km"], fill_value=0)
    bars = axes[1].bar(radius_counts.index, radius_counts.values, color=["#9CCB86", "#F6C57A"], edgecolor="white")
    axes[1].set_title("(b) Samples by distance bin", loc="left", fontweight="bold")
    axes[1].set_xlabel("Distance to typhoon center")
    axes[1].set_ylabel("Sample count")
    axes[1].set_ylim(0, max(radius_counts.max() * 1.30, 1))
    radius_labels = [f"n={int(v)}" + ("\nlow sample" if v < 10 else "") for v in radius_counts.values]
    annotate_bars(axes[1], bars, radius_labels, pad=0.03)

    period_counts = pd.Series(
        {
            "18-48 h": int(((samples["lead_hour_from_case_start"] >= 18) & (samples["lead_hour_from_case_start"] < 48)).sum()),
            "48-72 h": int(((samples["lead_hour_from_case_start"] >= 48) & (samples["lead_hour_from_case_start"] <= 72)).sum()),
            "18-72 h": int(((samples["lead_hour_from_case_start"] >= 18) & (samples["lead_hour_from_case_start"] <= 72)).sum()),
        }
    )
    bars = axes[2].bar(period_counts.index, period_counts.values, color=["#C7A6D8", "#A9D6C8", "#B8B8B8"], edgecolor="white")
    axes[2].set_title("(c) Samples by forecast period", loc="left", fontweight="bold")
    axes[2].set_xlabel("Forecast period")
    axes[2].set_ylabel("Sample count")
    axes[2].set_ylim(0, max(period_counts.max() * 1.30, 1))
    period_labels = [f"n={int(v)}" + ("\nlow sample" if v < 10 else "") for v in period_counts.values]
    annotate_bars(axes[2], bars, period_labels, pad=0.03)
    fig.text(0.5, -0.02, "* indicates low sample, n < 10.", ha="center", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "figure_sample_count_distribution")


def jittered_scatter(ax: plt.Axes, x: float, values: pd.Series, color: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    jitter = rng.uniform(-0.055, 0.055, size=len(vals))
    ax.scatter(np.full(len(vals), x) + jitter, vals, s=22, alpha=0.55, color=color, edgecolor="white", linewidth=0.3, zorder=3)


def plot_overall_error(samples: pd.DataFrame, overall_stats: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))
    fig.suptitle("Overall wind speed error against ICOADS observations", fontsize=13, fontweight="bold")

    error_data = [samples["gdas_speed_error"].dropna(), samples["era5lagged_speed_error"].dropna()]
    box = axes[0].boxplot(error_data, tick_labels=MODEL_ORDER, patch_artist=True, widths=0.46, showfliers=False)
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.65)
    jittered_scatter(axes[0], 1, samples["gdas_speed_error"], MODEL_COLORS["GDAS_Realtime"], seed=11)
    jittered_scatter(axes[0], 2, samples["era5lagged_speed_error"], MODEL_COLORS["ERA5_Lagged"], seed=12)
    axes[0].axhline(0, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_title("(a) Wind speed error", loc="left", fontweight="bold")
    axes[0].set_ylabel("Model wind speed minus ICOADS wind speed (m/s)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    lines = []
    for model in MODEL_ORDER:
        row = overall_stats[overall_stats["model"] == model].iloc[0]
        lines.append(f"{model}: Bias={row.bias:.2f}, MAE={row.mae:.2f}, RMSE={row.rmse:.2f}, n={int(row.n)}")
    axes[0].text(0.03, 0.97, "\n".join(lines), transform=axes[0].transAxes, va="top", ha="left", fontsize=8, bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.88))

    abs_data = [samples["gdas_abs_error"].dropna(), samples["era5lagged_abs_error"].dropna()]
    box = axes[1].boxplot(abs_data, tick_labels=MODEL_ORDER, patch_artist=True, widths=0.46, showfliers=False)
    for patch, model in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.65)
    jittered_scatter(axes[1], 1, samples["gdas_abs_error"], MODEL_COLORS["GDAS_Realtime"], seed=21)
    jittered_scatter(axes[1], 2, samples["era5lagged_abs_error"], MODEL_COLORS["ERA5_Lagged"], seed=22)
    axes[1].set_title("(b) Absolute wind speed error", loc="left", fontweight="bold")
    axes[1].set_ylabel("Absolute wind speed error (m/s)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    gdas_rmse = overall_stats.loc[overall_stats["model"] == "GDAS_Realtime", "rmse"].iloc[0]
    era5_rmse = overall_stats.loc[overall_stats["model"] == "ERA5_Lagged", "rmse"].iloc[0]
    change = (era5_rmse - gdas_rmse) / era5_rmse * 100.0 if era5_rmse else np.nan
    change_word = "RMSE improvement" if change >= 0 else "RMSE change"
    lines = []
    for model in MODEL_ORDER:
        row = overall_stats[overall_stats["model"] == model].iloc[0]
        lines.append(f"{model}: MAE={row.mae:.2f}, RMSE={row.rmse:.2f}, n={int(row.n)}")
    lines.append(f"{change_word}: {change:.1f}%")
    axes[1].text(0.03, 0.97, "\n".join(lines), transform=axes[1].transAxes, va="top", ha="left", fontsize=8, bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.88))
    fig.tight_layout()
    return save_figure(fig, "figure_wind_speed_error_overall")


def plot_period_error(period_stats: pd.DataFrame) -> list[Path]:
    periods = ["18-48 h", "48-72 h", "18-72 h"]
    x = np.arange(len(periods))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    fig.suptitle("Wind speed error by forecast period against ICOADS observations", fontsize=13, fontweight="bold")
    for ax, metric, title in [
        (axes[0], "rmse", "(a) RMSE by forecast period"),
        (axes[1], "mae", "(b) MAE by forecast period"),
    ]:
        for offset, model in [(-width / 2, "GDAS_Realtime"), (width / 2, "ERA5_Lagged")]:
            subset = period_stats[period_stats["model"] == model].set_index("forecast_period").reindex(periods)
            values = subset[metric].to_numpy(dtype=float)
            bars = ax.bar(x + offset, values, width=width, color=MODEL_COLORS[model], edgecolor="white", label=model)
            labels = []
            for _, row in subset.iterrows():
                label = f"n={int(row.n)}"
                if bool(row.low_sample):
                    label += "\nlow n"
                labels.append(label)
            annotate_bars(ax, bars, labels, pad=0.035)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, periods)
        ax.set_xlabel("Forecast period")
        ax.set_ylabel(f"{metric.upper()} (m/s)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        current_top = ax.get_ylim()[1]
        ax.set_ylim(0, current_top * 1.20)
    axes[0].legend(loc="upper left", frameon=True)
    fig.text(0.5, -0.01, "low n indicates n < 10.", ha="center", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "figure_wind_error_by_period")


def write_notes(samples: pd.DataFrame, track: pd.DataFrame | None) -> None:
    radius_counts = samples["coarse_radius_bin"].value_counts().reindex(["0-400 km", "400-600 km"], fill_value=0)
    period_counts = {
        "18-48 h": int(((samples["lead_hour_from_case_start"] >= 18) & (samples["lead_hour_from_case_start"] < 48)).sum()),
        "48-72 h": int(((samples["lead_hour_from_case_start"] >= 48) & (samples["lead_hour_from_case_start"] <= 72)).sum()),
        "18-72 h": int(((samples["lead_hour_from_case_start"] >= 18) & (samples["lead_hour_from_case_start"] <= 72)).sum()),
    }
    pre18 = int((samples["lead_hour_from_case_start"] < 18).sum())
    missing_leads = [lead for lead in LEAD_HOURS if int((samples["lead_hour_from_case_start"] == lead).sum()) == 0]
    track_note = "The best track from typhoon_2506_Wipha.csv was overlaid on the sample coverage map." if track is not None else "No best-track coordinates were available, so the sample coverage map does not overlay a track."
    map_note = "Cartopy was used for the sample coverage map." if HAS_CARTOPY else "Cartopy was not available in the rendering environment, so the sample coverage map uses ordinary longitude-latitude axes."
    lines = [
        "# Wipha ICOADS Verification Plotting Notes",
        "",
        f"Matched sample file: `{MATCHED_CSV}`",
        "",
        "ICOADS observations are used as the verification reference.",
        "",
        "## Filtering",
        "",
        "- Kept samples with distance_to_typhoon_center_km <= 600.",
        "- Kept lead_hour_from_case_start within 0-72 h.",
        "- Kept samples where obs_wind_speed, gdas_wind_speed, and era5lagged_wind_speed are non-missing.",
        "- Kept wind speeds within 0-75 m/s.",
        f"- Filtered sample count: {len(samples)}.",
        f"- Samples before 18 h: {pre18}. Missing lead hours: {missing_leads}.",
        "",
        "## Sample Counts",
        "",
        f"- 0-400 km: {int(radius_counts['0-400 km'])}.",
        f"- 400-600 km: {int(radius_counts['400-600 km'])}.",
        f"- 18-48 h: {period_counts['18-48 h']}.",
        f"- 48-72 h: {period_counts['48-72 h']}.",
        f"- 18-72 h: {period_counts['18-72 h']}.",
        "",
        "## Figures",
        "",
        "- figure_icoads_sample_coverage: spatial coverage of filtered ICOADS samples, colored by lead time and sized by observed wind speed.",
        "- figure_sample_count_distribution: sample counts by lead hour, coarse radius bin, and forecast period.",
        "- figure_wind_speed_error_overall: overall GDAS_Realtime and ERA5_Lagged wind speed errors against ICOADS.",
        "- figure_wind_error_by_period: period-aggregated RMSE and MAE, with sample counts marked on each bar.",
        "",
        track_note,
        map_note,
        "",
        "Due to the limited number and uneven temporal-spatial distribution of ICOADS samples, the figures support point-based verification only and should not be interpreted as a full validation of the typhoon inner-core wind structure.",
    ]
    NOTES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[Path, str]] = []
    samples = load_and_filter_samples()
    track = load_track()
    overall_stats = compute_overall_statistics(samples)
    period_stats = compute_period_statistics(samples)
    compute_supplementary_lead_statistics(samples)
    write_notes(samples, track)

    generated.append((PLOTTING_SAMPLES_CSV, "filtered plotting samples within 600 km"))
    generated.append((OVERALL_STATS_CSV, "overall wind speed error statistics"))
    generated.append((PERIOD_STATS_CSV, "period wind speed error statistics"))
    generated.append((SUPPLEMENTARY_LEAD_STATS_CSV, "supplementary lead-hour statistics"))
    generated.append((NOTES_MD, "plotting notes and limitations"))

    for path in plot_sample_coverage(samples, track):
        generated.append((path, "ICOADS sample coverage figure"))
    for path in plot_sample_count_distribution(samples):
        generated.append((path, "sample count distribution figure"))
    for path in plot_overall_error(samples, overall_stats):
        generated.append((path, "overall wind speed error figure"))
    for path in plot_period_error(period_stats):
        generated.append((path, "wind speed error by forecast period figure"))

    print("Generated files:")
    for path, purpose in generated:
        print(f"- {path}: {purpose}")
    print(f"Filtered samples: {len(samples)}")


if __name__ == "__main__":
    main()
