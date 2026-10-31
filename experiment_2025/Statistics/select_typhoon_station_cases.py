"""Select three chronological 72-hour cases from existing matched station CSVs.

This stage does not read raw observations or NetCDF forecasts. Forecast and
observation values are preserved; inconsistent initialization metadata is
reported rather than rewritten as fixed initialization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
OUTPUT_STEM = "typhoon_three_cases_chronological_beijing"
SELECTED_CASES = (
    ("韦帕", "韦帕 (Wipha, 2025)", "遮浪", "legacy", "2025-07-18 00:00"),
    ("范斯高", "范斯高 (Francisco, 2025)", "大陈", "additional", "2025-07-24 00:00"),
    ("杨柳", "杨柳 (Podul, 2025)", "北礵", "additional", "2025-08-12 00:00"),
)
EXPERIMENTS = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")


def select_cases(legacy: pd.DataFrame, additional: pd.DataFrame, window_hours=72,
                 require_fixed=False) -> tuple[pd.DataFrame, dict]:
    selected = []
    info = []
    for prefix, label, station, source, expected_target in SELECTED_CASES:
        frame = legacy if source == "legacy" else additional
        sub = frame[
            frame["typhoon"].astype(str).str.startswith(prefix)
            & (frame["station"] == station)
            & frame["plot_lead_hour"].between(0, window_hours)
        ].copy().sort_values("station_time_utc", kind="stable")
        if sub.empty:
            raise ValueError(f"No matched records for {prefix}, {station}")
        if sub["station_time_utc"].duplicated().any():
            raise ValueError(f"Duplicate observation times for {prefix}, {station}")
        times = pd.to_datetime(sub["station_time_utc"], errors="raise")
        raw = pd.to_datetime(sub["station_time_raw"], errors="raise")
        if not np.allclose((raw - times).dt.total_seconds(), 8 * 3600):
            raise ValueError(f"Expected Beijing-time source records for {prefix}, {station}")
        starts = times - pd.to_timedelta(sub["plot_lead_hour"], unit="h")
        if starts.isna().any() or starts.nunique() != 1:
            raise ValueError(f"Inconsistent observation window for {prefix}, {station}")
        targets = pd.to_datetime(sub["target_time_utc"], errors="raise")
        valid = pd.to_datetime(sub["model_valid_time_utc"], errors="raise")
        model_lead = (valid - targets).dt.total_seconds() / 3600
        if targets.isna().any() or valid.isna().any():
            raise ValueError(f"Missing initialization or valid times for {prefix}, {station}")
        for column in ("lead_hour", "model_lead_hour"):
            if column in sub and not np.allclose(sub[column], model_lead):
                raise ValueError(f"{column} disagrees with recorded times for {prefix}, {station}")
        is_fixed = targets.nunique() == 1
        verified = targets == pd.Timestamp(expected_target)
        if require_fixed and not verified.all():
            raise ValueError(
                f"{prefix}, {station}: {int((~verified).sum())} records do not use the required "
                f"fixed initialization {expected_target} UTC; provide freshly matched data. "
                "Changing labels cannot fix forecast values."
            )
        sub["typhoon"] = label
        sub["model_lead_hour"] = model_lead.to_numpy()
        sub["lead_hour"] = model_lead.to_numpy()
        sub["initialization_policy"] = "fixed" if is_fixed else "daily_00_utc"
        sub["expected_target_time_utc"] = expected_target
        start = starts.iloc[0]
        selected.append((start, sub))
        info.append({
            "typhoon": label, "station": station,
            "window_start_utc": str(start), "window_end_utc": str(times.max()),
            "window_hours": float(sub["plot_lead_hour"].max()),
            "rows": int(len(sub)),
            "exact_time_rows": int((sub["match_offset_hours"] == 0).sum()),
            "initialization_policy_recorded": "fixed" if is_fixed else "daily_00_utc",
            "expected_target_time_utc": expected_target,
            "fixed_initialization_verified": bool(verified.all()),
            "rows_with_expected_target": int(verified.sum()),
            "target_times_utc": sorted(sub["target_time_utc"].unique().tolist()),
            "model_lead_hours": [float(model_lead.min()), float(model_lead.max())],
        })
    selected.sort(key=lambda item: item[0])
    info.sort(key=lambda item: item["window_start_utc"])
    combined = pd.concat([frame for _, frame in selected], ignore_index=True)
    return combined, {
        "cases": info, "order": "chronological by UTC observation window",
        "window_hours_requested": window_hours,
        "all_fixed_in_recorded_metadata": all(case["fixed_initialization_verified"] for case in info),
        "values_policy": "Existing matched wind values are preserved; no re-matching or temporal interpolation.",
        "comparison_policy": "Inspect each case separately; mixed initialization policies must not be pooled as a fixed-run average.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-csv", type=Path,
                        default=DEFAULT_DATA_ROOT / "typhoon_station_wind_timeseries_2025_beijing.csv")
    parser.add_argument("--additional-csv", type=Path,
                        default=DEFAULT_DATA_ROOT / "two_typhoons_2025_station_wind_timeseries_beijing.csv")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--matched-csv", type=Path, default=None,
                        help="one matched CSV containing all three selected cases")
    parser.add_argument("--window-hours", type=int, choices=(72, 120), default=72)
    parser.add_argument("--require-fixed", action="store_true",
                        help="reject CSVs containing multiple initializations within one case")
    args = parser.parse_args(argv)
    if args.matched_csv is not None:
        args.legacy_csv = args.additional_csv = args.matched_csv
    frame, summary = select_cases(pd.read_csv(args.legacy_csv), pd.read_csv(args.additional_csv),
                                  args.window_hours, args.require_fixed)
    summary["sources"] = {"legacy": str(args.legacy_csv), "additional": str(args.additional_csv)}
    args.data_root.mkdir(parents=True, exist_ok=True)
    path = args.data_root / f"{OUTPUT_STEM}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    summary_path = args.data_root / f"{OUTPUT_STEM}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected CSV: {path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
