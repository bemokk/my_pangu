from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

from plots import plot_wind_speed_beaufort_metrics as beaufort_plot  # noqa: E402


def test_load_beaufort_metrics_filters_three_datasets_and_target_leads(tmp_path):
    csv_path = tmp_path / "wind_speed_metrics_by_beaufort.csv"
    rows = [
        {
            "dataset": "era5_realtime",
            "dataset_label": "ERA5 realtime",
            "lead_hour": 24,
            "obs_beaufort_group": "<=2",
            "rmse": 1.0,
            "mae": 0.8,
            "n": 10,
        },
        {
            "dataset": "era5_lagged_5d",
            "dataset_label": "ERA5 lagged 5d forecast",
            "lead_hour": 48,
            "obs_beaufort_group": "3",
            "rmse": 2.0,
            "mae": 1.8,
            "n": 11,
        },
        {
            "dataset": "gdas_forecast",
            "dataset_label": "GDAS forecast",
            "lead_hour": 72,
            "obs_beaufort_group": ">=8",
            "rmse": 3.0,
            "mae": 2.8,
            "n": 12,
        },
        {
            "dataset": "era5_realtime",
            "dataset_label": "ERA5 realtime",
            "lead_hour": 12,
            "obs_beaufort_group": "4",
            "rmse": 4.0,
            "mae": 3.8,
            "n": 13,
        },
        {
            "dataset": "other",
            "dataset_label": "Other",
            "lead_hour": 24,
            "obs_beaufort_group": "5",
            "rmse": 5.0,
            "mae": 4.8,
            "n": 14,
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = beaufort_plot.load_beaufort_metrics(csv_path)

    assert result["dataset"].tolist() == ["era5_realtime", "era5_lagged_5d", "gdas_forecast"]
    assert result["lead_hour"].tolist() == [24, 48, 72]
    assert result["beaufort_code"].tolist() == [0, 1, 6]
    assert result["obs_beaufort_group"].tolist() == ["<=2", "3", ">=8"]


def test_style_axis_only_shows_observed_beaufort_label_when_requested():
    fig, (upper_ax, lower_ax) = plt.subplots(2, 1)
    try:
        beaufort_plot.style_axis(upper_ax, "RMSE", show_xlabel=False)
        beaufort_plot.style_axis(lower_ax, "RMSE", show_xlabel=True)

        assert upper_ax.get_xlabel() == ""
        assert lower_ax.get_xlabel() == beaufort_plot.TEXT_LABELS["observed_beaufort"]
    finally:
        plt.close(fig)
