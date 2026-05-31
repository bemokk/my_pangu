from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

import plot_wind_speed_beaufort_sample_counts as sample_plot  # noqa: E402


def test_load_sample_counts_collapses_duplicate_dataset_counts(tmp_path):
    csv_path = tmp_path / "wind_speed_metrics_by_beaufort.csv"
    rows = [
        {"dataset": "era5_realtime", "lead_hour": 24, "obs_beaufort_group": "<=2", "n": 10},
        {"dataset": "era5_lagged_5d", "lead_hour": 24, "obs_beaufort_group": "<=2", "n": 10},
        {"dataset": "gdas_forecast", "lead_hour": 24, "obs_beaufort_group": "<=2", "n": 10},
        {"dataset": "era5_realtime", "lead_hour": 48, "obs_beaufort_group": "3", "n": 11},
        {"dataset": "era5_lagged_5d", "lead_hour": 72, "obs_beaufort_group": ">=8", "n": 12},
        {"dataset": "era5_realtime", "lead_hour": 12, "obs_beaufort_group": "4", "n": 13},
        {"dataset": "other", "lead_hour": 24, "obs_beaufort_group": "5", "n": 14},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = sample_plot.load_sample_counts(csv_path)

    assert result["lead_hour"].tolist() == [24, 48, 72]
    assert result["obs_beaufort_group"].tolist() == ["<=2", "3", ">=8"]
    assert result["beaufort_code"].tolist() == [0, 1, 6]
    assert result["n"].tolist() == [10, 11, 12]
