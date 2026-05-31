from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

import plot_wind_direction_figures as direction_figures  # noqa: E402


def test_load_direction_metrics_keeps_all_three_datasets_and_3h_leads(tmp_path):
    csv_path = tmp_path / "wind_direction_metrics_by_lead.csv"
    rows = [
        {"dataset": "era5_realtime", "dataset_label": "ERA5 realtime", "lead_hour": 3, "rmse": 10, "mae": 8},
        {"dataset": "era5_lagged_5d", "dataset_label": "ERA5 lagged 5d forecast", "lead_hour": 6, "rmse": 20, "mae": 18},
        {"dataset": "gdas_forecast", "dataset_label": "GDAS forecast", "lead_hour": 72, "rmse": 30, "mae": 28},
        {"dataset": "era5_realtime", "dataset_label": "ERA5 realtime", "lead_hour": 1, "rmse": 40, "mae": 38},
        {"dataset": "other", "dataset_label": "Other", "lead_hour": 3, "rmse": 50, "mae": 48},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = direction_figures.load_direction_metrics(csv_path)

    assert result["dataset"].tolist() == ["era5_realtime", "era5_lagged_5d", "gdas_forecast"]
    assert result["lead_hour"].tolist() == [3, 6, 72]
    assert result["rmse"].tolist() == [10, 20, 30]


def test_load_direction_frequency_keeps_target_leads_and_sector_order(tmp_path):
    csv_path = tmp_path / "wind_direction_frequency_by_lead.csv"
    rows = [
        {
            "dataset": "era5_realtime",
            "dataset_label": "ERA5 realtime",
            "lead_hour": 24,
            "direction_sector": "N",
            "obs_frequency": 0.10,
            "pred_frequency": 0.11,
        },
        {
            "dataset": "era5_realtime",
            "dataset_label": "ERA5 realtime",
            "lead_hour": 24,
            "direction_sector": "NNE",
            "obs_frequency": 0.12,
            "pred_frequency": 0.13,
        },
        {
            "dataset": "gdas_forecast",
            "dataset_label": "GDAS forecast",
            "lead_hour": 72,
            "direction_sector": "NNW",
            "obs_frequency": 0.14,
            "pred_frequency": 0.15,
        },
        {
            "dataset": "era5_realtime",
            "dataset_label": "ERA5 realtime",
            "lead_hour": 12,
            "direction_sector": "NE",
            "obs_frequency": 0.16,
            "pred_frequency": 0.17,
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = direction_figures.load_direction_frequency(csv_path)

    assert result["lead_hour"].tolist() == [24, 24, 72]
    assert result["direction_sector"].tolist() == ["N", "NNE", "NNW"]
    assert result["sector_code"].tolist() == [0, 1, 15]
