"""Compute observed and forecast 10 m wind-speed standard deviations."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

STATS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STATS_ROOT.parents[1]
if str(STATS_ROOT) not in sys.path:
    sys.path.insert(0, str(STATS_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_2025.Statistics.statistics import (
    AREA,
    DEFAULT_ICOADS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GridSampler,
    iter_targets,
    load_daily_updated,
    model_path,
)


LEADS = (0, 24, 48, 72, 96, 120)
EXPERIMENT_ORDER = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")
LABELS = {
    "era5_direct": "E0",
    "gdas_direct": "G0",
    "era5_lagged_120h": "E120",
    "gdas_lagged_120h": "G120",
}


def run(args) -> tuple[pd.DataFrame, dict]:
    obs_by_time, obs_info = load_daily_updated(args.icoads_root, AREA)
    first_model = model_path(args.output_root, "era5_direct", datetime(2025, 1, 1), 0)
    if not first_model.exists():
        raise FileNotFoundError(f"Model output not found: {first_model}")
    sampler = GridSampler(first_model, AREA)

    sums = defaultdict(float)
    counts = defaultdict(int)
    missing = []
    for lead in LEADS:
        for target in iter_targets(args.start_date, args.end_date):
            valid_time = target + timedelta(hours=lead)
            observations = obs_by_time.get(valid_time)
            if observations is None:
                continue
            obs_speed = observations["speed"]
            for experiment in EXPERIMENT_ORDER:
                path = model_path(args.output_root, experiment, target, lead)
                if not path.is_file():
                    missing.append(
                        {
                            "experiment": experiment,
                            "target_time": target.strftime("%Y-%m-%d-%H-%M"),
                            "valid_time": valid_time.strftime("%Y-%m-%d-%H-%M"),
                            "lead_hour": lead,
                            "path": str(path),
                        }
                    )
                    continue
                pred_u, pred_v = sampler.sample(path, observations["lat"], observations["lon"])
                pred_speed = np.sqrt(pred_u * pred_u + pred_v * pred_v)
                valid = np.isfinite(pred_speed) & np.isfinite(obs_speed)
                if not np.any(valid):
                    continue
                pred_values = pred_speed[valid]
                obs_values = obs_speed[valid]
                for series, values in ((experiment, pred_values), ("observed", obs_values)):
                    counts[(lead, series)] += int(values.size)
                    sums[(lead, series, "sum")] += float(np.sum(values))
                    sums[(lead, series, "sum2")] += float(np.sum(values * values))

    rows = []
    for lead in LEADS:
        row = {"lead_hour": lead}
        for series, column in (("observed", "observed_std"), *[(exp, LABELS[exp]) for exp in EXPERIMENT_ORDER]):
            n = counts[(lead, series)]
            total = sums[(lead, series, "sum")]
            total2 = sums[(lead, series, "sum2")]
            variance = max(total2 / n - (total / n) ** 2, 0.0) if n else np.nan
            row[column] = np.sqrt(variance)
            row[f"n_{series}"] = n
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "area": AREA,
        "leads": list(LEADS),
        "std_definition": "population standard deviation, ddof=0, matching legacy np.std",
        "experiments": list(EXPERIMENT_ORDER),
        "missing_outputs": missing,
        "observation_info": obs_info,
    }
    return frame, summary


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "### 10 m 风速标准差（m·s⁻¹）",
        "",
        "| 时效/h | 观测标准差 | E0 | G0 | E120 | G120 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {int(row.lead_hour)} | {row.observed_std:.2f} | {row.E0:.2f} | "
            f"{row.G0:.2f} | {row.E120:.2f} | {row.G120:.2f} |"
        )
    return "\n".join(lines) + "\n"


def normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model standard deviations divided by the matching observed std."""
    result = pd.DataFrame({"lead_hour": frame["lead_hour"].astype(int)})
    for column in ("E0", "G0", "E120", "G120"):
        result[column] = frame[column] / frame["observed_std"]
    return result


def normalized_markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "### 10 m 风速标准差与观测标准差之比（无量纲）",
        "",
        "| 时效/h | E0 | G0 | E120 | G120 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples():
        lines.append(
            f"| {int(row.lead_hour)} | {row.E0:.3f} | {row.G0:.3f} | "
            f"{row.E120:.3f} | {row.G120:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icoads-root", type=Path, default=DEFAULT_ICOADS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=STATS_ROOT / "data")
    parser.add_argument("--start-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2025, 1, 1))
    parser.add_argument("--end-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2025, 12, 31))
    args = parser.parse_args(argv)
    frame, summary = run(args)
    args.data_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.data_root / "wind_speed_standard_deviation_by_lead.csv", index=False, encoding="utf-8-sig")
    (args.data_root / "wind_speed_standard_deviation_table.md").write_text(markdown_table(frame), encoding="utf-8")
    ratio = normalized_frame(frame)
    ratio.to_csv(args.data_root / "wind_speed_standard_deviation_ratio_by_lead.csv", index=False, encoding="utf-8-sig")
    (args.data_root / "wind_speed_standard_deviation_ratio_table.md").write_text(
        normalized_markdown_table(ratio), encoding="utf-8"
    )
    summary["table"] = frame.to_dict(orient="records")
    (args.data_root / "wind_speed_standard_deviation_summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(markdown_table(frame))
    print(normalized_markdown_table(ratio))
    print(f"CSV: {args.data_root / 'wind_speed_standard_deviation_by_lead.csv'}")
    print(f"Missing outputs: {len(summary['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
