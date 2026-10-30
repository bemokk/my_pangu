"""Compute wind-speed RMSE and MAE by observed Beaufort wind-force group."""
from __future__ import annotations

import argparse
import json
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

from experiment_2025.Statistics.statistics import (  # noqa: E402
    AREA,
    DEFAULT_ICOADS_ROOT,
    DEFAULT_OUTPUT_ROOT,
    GridSampler,
    iter_targets,
    load_daily_updated,
    model_path,
)


LEADS = (24, 48, 72, 96, 120)
EXPERIMENT_ORDER = ("era5_direct", "gdas_direct", "era5_lagged_120h", "gdas_lagged_120h")
LABELS = {
    "era5_direct": "E0",
    "gdas_direct": "G0",
    "era5_lagged_120h": "E120",
    "gdas_lagged_120h": "G120",
}
BEAUFORT_BINS = (-np.inf, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, np.inf)
BEAUFORT_LABELS = ("<=2", "3", "4", "5", "6", "7", ">=8")


def beaufort_group(speed: np.ndarray) -> np.ndarray:
    codes = np.digitize(speed, BEAUFORT_BINS[1:-1], right=False)
    return np.asarray(BEAUFORT_LABELS, dtype=object)[codes]


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
            groups = beaufort_group(obs_speed)
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
                for group in BEAUFORT_LABELS:
                    take = valid & (groups == group)
                    if not np.any(take):
                        continue
                    error = pred_speed[take] - obs_speed[take]
                    counts[(lead, group, experiment)] += int(error.size)
                    sums[(lead, group, experiment, "sse")] += float(np.sum(error * error))
                    sums[(lead, group, experiment, "sae")] += float(np.sum(np.abs(error)))

    rows = []
    for lead in LEADS:
        for group in BEAUFORT_LABELS:
            row = {"lead_hour": lead, "beaufort_group": group}
            for experiment in EXPERIMENT_ORDER:
                n = counts[(lead, group, experiment)]
                row[f"n_{LABELS[experiment]}"] = n
                row[f"{LABELS[experiment]}_rmse"] = (
                    float(np.sqrt(sums[(lead, group, experiment, "sse")] / n)) if n else np.nan
                )
                row[f"{LABELS[experiment]}_mae"] = (
                    float(sums[(lead, group, experiment, "sae")] / n) if n else np.nan
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "area": AREA,
        "leads": list(LEADS),
        "beaufort_groups": list(BEAUFORT_LABELS),
        "group_definition": "observed 10 m wind speed; bins [0,3.4,5.5,8.0,10.8,13.9,17.2,75] m s^-1",
        "experiments": list(EXPERIMENT_ORDER),
        "missing_outputs": missing,
        "observation_info": obs_info,
    }
    return frame, summary


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
    long_path = args.data_root / "wind_speed_metrics_by_beaufort_2025.csv"
    frame.to_csv(long_path, index=False, encoding="utf-8-sig")
    summary_path = args.data_root / "wind_speed_metrics_by_beaufort_2025_summary.json"
    summary["table"] = frame.to_dict(orient="records")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"CSV: {long_path}")
    print(f"Missing outputs: {len(summary['missing_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
