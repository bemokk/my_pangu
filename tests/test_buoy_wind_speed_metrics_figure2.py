from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

from plots import plot_wind_speed_metrics_figure2 as speed_plot  # noqa: E402


def test_load_metrics_can_append_lead_zero_rows(monkeypatch, tmp_path):
    csv_path = tmp_path / "wind_speed_metrics_by_lead.csv"
    pd.DataFrame(
        [
            {
                "dataset": "era5_realtime",
                "lead_hour": 3,
                "rmse": 2.0,
                "mae": 1.0,
                "corr": 0.8,
            }
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        speed_plot,
        "build_lead_zero_metric_rows",
        lambda variable: pd.DataFrame(
            [
                {
                    "dataset": "era5_realtime",
                    "dataset_label": "ERA5 realtime",
                    "lead_hour": 0,
                    "variable": variable,
                    "n": 10,
                    "rmse": 1.0,
                    "mae": 0.5,
                    "bias": 0.0,
                    "corr": 0.9,
                    "pred_mean": 3.0,
                    "obs_mean": 3.0,
                    "diff_std": 1.0,
                }
            ]
        ),
    )

    result = speed_plot.load_metrics(csv_path, include_lead_zero=True)

    assert result["lead_hour"].tolist() == [0, 3]
    assert result["rmse"].tolist() == [1.0, 2.0]
