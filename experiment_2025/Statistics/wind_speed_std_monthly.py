"""Compute wind-speed standard deviations for selected months."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

STATS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATS_ROOT.parents[1]
if str(STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(STATS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wind_speed_std import normalized_frame, normalized_markdown_table, run


MONTHS = ((1, "January"), (8, "August"))


def raw_markdown(month_frames: dict[str, pd.DataFrame]) -> str:
    lines = ["# 月度10 m风速标准差（m·s⁻¹）", ""]
    for month, frame in month_frames.items():
        lines.extend(
            [
                f"## {month}",
                "",
                "| 时效/h | 观测标准差 | E0 | G0 | E120 | G120 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in frame.itertuples():
            lines.append(
                f"| {int(row.lead_hour)} | {row.observed_std:.3f} | {row.E0:.3f} | "
                f"{row.G0:.3f} | {row.E120:.3f} | {row.G120:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def normalized_monthly_markdown(month_frames: dict[str, pd.DataFrame]) -> str:
    lines = ["# 月度10 m风速标准差与观测标准差之比（无量纲）", ""]
    for month, frame in month_frames.items():
        lines.append(f"## {month}")
        lines.append("")
        lines.append(normalized_markdown_table(normalized_frame(frame)).rstrip())
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icoads-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=STATS_ROOT / "data")
    args = parser.parse_args(argv)

    # Import defaults lazily so this script remains usable from the project root.
    if args.icoads_root is None or args.output_root is None:
        from experiment_2025.Statistics.statistics import DEFAULT_ICOADS_ROOT, DEFAULT_OUTPUT_ROOT

        if args.icoads_root is None:
            args.icoads_root = DEFAULT_ICOADS_ROOT
        if args.output_root is None:
            args.output_root = DEFAULT_OUTPUT_ROOT

    frames: dict[str, pd.DataFrame] = {}
    summaries = {}
    for month, name in MONTHS:
        start = datetime(2025, month, 1)
        end = datetime(2025, 2, 1) - pd.Timedelta(days=1) if month == 1 else datetime(2025, 9, 1) - pd.Timedelta(days=1)
        frame, summary = run(
            argparse.Namespace(
                icoads_root=args.icoads_root,
                output_root=args.output_root,
                start_date=start,
                end_date=end,
            )
        )
        frames[name] = frame
        summaries[name] = summary

    args.data_root.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    ratio_rows = []
    for month, frame in frames.items():
        for row in frame.to_dict(orient="records"):
            raw_rows.append({"month": month, **row})
        for row in normalized_frame(frame).to_dict(orient="records"):
            ratio_rows.append({"month": month, **row})
    pd.DataFrame(raw_rows).to_csv(
        args.data_root / "wind_speed_standard_deviation_january_august.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(ratio_rows).to_csv(
        args.data_root / "wind_speed_standard_deviation_ratio_january_august.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.data_root / "wind_speed_standard_deviation_january_august.md").write_text(
        raw_markdown(frames), encoding="utf-8"
    )
    (args.data_root / "wind_speed_standard_deviation_ratio_january_august.md").write_text(
        normalized_monthly_markdown(frames), encoding="utf-8"
    )
    (args.data_root / "wind_speed_standard_deviation_january_august_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(raw_markdown(frames))
    print(normalized_monthly_markdown(frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
