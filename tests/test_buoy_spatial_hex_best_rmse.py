from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

import plot_spatial_hex_best_rmse as spatial_rmse  # noqa: E402


def test_compute_hex_rmse_winners_compares_lagged_era5_and_gdas_only():
    hexes = pd.DataFrame(
        {
            "hex_id": [1, 2],
            "center_lon": [120.0, 121.0],
            "center_lat": [20.0, 21.0],
        }
    )
    records = pd.DataFrame(
        [
            {"lead_hour": 24, "hex_id": 1, "dataset": "era5_realtime", "speed_error_ms": 0.1, "record_id": "a"},
            {"lead_hour": 24, "hex_id": 1, "dataset": "era5_realtime", "speed_error_ms": 1.0, "record_id": "a"},
            {"lead_hour": 24, "hex_id": 1, "dataset": "era5_lagged_5d", "speed_error_ms": 2.0, "record_id": "a"},
            {"lead_hour": 24, "hex_id": 1, "dataset": "era5_lagged_5d", "speed_error_ms": -2.0, "record_id": "b"},
            {"lead_hour": 24, "hex_id": 1, "dataset": "gdas_forecast", "speed_error_ms": 1.0, "record_id": "a"},
            {"lead_hour": 24, "hex_id": 1, "dataset": "gdas_forecast", "speed_error_ms": -1.0, "record_id": "b"},
            {"lead_hour": 24, "hex_id": 2, "dataset": "era5_lagged_5d", "speed_error_ms": 0.5, "record_id": "c"},
            {"lead_hour": 24, "hex_id": 2, "dataset": "gdas_forecast", "speed_error_ms": 0.25, "record_id": "c"},
        ]
    )

    result = spatial_rmse.compute_hex_rmse_winners(
        records,
        hexes,
        lead_hours=[24],
        min_samples_per_dataset=2,
    )

    winning_hex = result[result["hex_id"] == 1].iloc[0]
    assert winning_hex["observation_count"] == 2
    assert "era5_realtime_rmse" not in result.columns
    assert winning_hex["era5_lagged_5d_rmse"] == 2.0
    assert winning_hex["gdas_forecast_rmse"] == 1.0
    assert winning_hex["best_dataset"] == "gdas_forecast"
    assert winning_hex["best_rmse_margin"] == 1.0

    sparse_hex = result[result["hex_id"] == 2].iloc[0]
    assert sparse_hex["best_dataset"] == "insufficient_data"
    assert sparse_hex["dataset_count_with_min_samples"] == 0
