from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent / "icoads_202507"
OUT_DIR = ROOT_DIR / "output"

DETAIL_CSV = OUT_DIR / "china_sea_all_platform_records_area_42_103_13_130.csv"
DAILY_COUNTS_CSV = OUT_DIR / "china_sea_daily_counts_area_42_103_13_130_3hourly.csv"
DAILY_COUNTS_PNG = OUT_DIR / "china_sea_daily_counts_area_42_103_13_130_3hourly.png"

TARGET_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

# True includes records within +/-30 minutes of each target UTC hour.
# False keeps only near-exact target hours.
INCLUDE_HALF_HOUR_WINDOW = True
TIME_TOLERANCE_HOURS = 0.5 if INCLUDE_HALF_HOUR_WINDOW else 0.01


def add_target_hour(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    records["datetime_utc"] = pd.to_datetime(records["datetime_utc"], errors="coerce")

    hour_from_time = (
        records["datetime_utc"].dt.hour
        + records["datetime_utc"].dt.minute / 60.0
        + records["datetime_utc"].dt.second / 3600.0
    )

    records["hour_utc"] = pd.to_numeric(records.get("hour_utc"), errors="coerce").fillna(hour_from_time)

    target_hours = np.array(TARGET_HOURS, dtype=float)
    hour_values = records["hour_utc"].to_numpy(dtype=float)
    raw_diff = np.abs(hour_values[:, None] - target_hours[None, :])
    diff = np.minimum(raw_diff, 24.0 - raw_diff)
    nearest_idx = np.nanargmin(np.where(np.isnan(diff), np.inf, diff), axis=1)

    records["target_hour_utc"] = target_hours[nearest_idx].astype(int)
    records["hour_diff"] = np.abs(records["hour_utc"] - records["target_hour_utc"])
    return records


def filter_records(records: pd.DataFrame) -> pd.DataFrame:
    records = add_target_hour(records)

    required_cols = ["datetime_utc", "latitude", "longitude", "wind_dir_deg", "wind_speed_ms"]
    records = records.dropna(subset=required_cols).copy()

    return records[
        records["wind_dir_deg"].between(1, 360)
        & records["wind_speed_ms"].between(0, 75)
        & (records["hour_diff"] <= TIME_TOLERANCE_HOURS)
    ].copy()


def plot_daily_counts(daily_counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))

    ax.bar(
        daily_counts["date"],
        daily_counts["record_count"],
        color="#2f6f9f",
        edgecolor="#1f3f5b",
        linewidth=0.6,
    )

    mean_count = daily_counts["record_count"].mean()
    ax.axhline(
        mean_count,
        color="#c43c39",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {mean_count:.1f}",
    )

    ax.set_title(
        "Daily ICOADS Records in China Sea Area",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Record count")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(frameon=False)

    ax.set_xticks(daily_counts["date"])
    ax.set_xticklabels(
        daily_counts["date"].dt.strftime("%m-%d"),
        rotation=45,
        ha="right",
    )

    subtitle = (
        "AREA=[42,103,13,130], valid wind direction/speed, "
        f"UTC hours={TARGET_HOURS}, "
        f"{'+/-30 min' if INCLUDE_HALF_HOUR_WINDOW else 'exact only'}"
    )
    fig.text(0.125, 0.91, subtitle, fontsize=9, color="#555555")

    fig.tight_layout()
    fig.savefig(DAILY_COUNTS_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not DETAIL_CSV.exists():
        raise FileNotFoundError(f"Detail CSV not found: {DETAIL_CSV}")

    records = pd.read_csv(DETAIL_CSV)
    records = filter_records(records)

    daily_counts = (
        records.assign(date=records["datetime_utc"].dt.floor("D"))
        .groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "record_count"})
        .sort_values("date")
    )

    daily_counts.to_csv(DAILY_COUNTS_CSV, index=False, encoding="utf-8-sig")
    plot_daily_counts(daily_counts)

    print(f"Filtered records: {len(records)}")
    print(f"Days: {len(daily_counts)}")
    print(f"CSV saved: {DAILY_COUNTS_CSV}")
    print(f"Figure saved: {DAILY_COUNTS_PNG}")
    print(daily_counts.to_string(index=False))


if __name__ == "__main__":
    main()
