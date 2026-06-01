import pandas as pd

from Buoy.plots.plot_wipha_case_analysis import (
    angular_difference_deg,
    circular_mean_deg,
    haversine_km,
    select_shortest_lead_forecasts,
)


def test_angular_difference_wraps_to_shortest_path():
    assert angular_difference_deg(350, 10) == 20
    assert angular_difference_deg(10, 350) == -20
    assert angular_difference_deg(180, 0) == -180


def test_circular_mean_handles_north_wrap():
    value = circular_mean_deg([350, 10])
    assert min(abs(value), abs(value - 360)) < 1e-9


def test_haversine_one_degree_equator_is_about_111_km():
    assert 110 <= haversine_km(0, 0, 1, 0) <= 112


def test_select_shortest_lead_forecasts_keeps_one_row_per_dataset_time():
    df = pd.DataFrame(
        {
            "platform_id": ["A", "A", "A"],
            "datetime_utc": pd.to_datetime(["2025-07-17 03:00"] * 3),
            "dataset": ["gdas_forecast", "gdas_forecast", "era5_lagged_5d"],
            "lead_hour": [6, 3, 3],
            "pred_speed_ms": [8.0, 7.0, 6.0],
            "pred_dir_deg": [90.0, 80.0, 70.0],
        }
    )
    out = select_shortest_lead_forecasts(df)
    assert len(out) == 2
    assert out[out["dataset"] == "gdas_forecast"].iloc[0]["lead_hour"] == 3
