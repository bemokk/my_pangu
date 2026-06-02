import importlib

import pandas as pd

from Buoy.plots.plot_wipha_track_buoy_locations import (
    load_wipha_track_csv,
    select_six_hour_track_points,
    track_time_label_annotation,
)
from Buoy.plots.plot_wipha_buoy_wind_timeseries import (
    TIMESERIES_FORECAST_INIT_TIMES,
    select_fixed_init_timeseries_samples,
    timeseries_time_axis_bounds,
)
from Buoy.plots.plot_wipha_buoy_wind_statistics_table import prepare_statistics_input
from Buoy.plots.plot_wipha_virtual_station_radius_test import (
    VIRTUAL_STATIONS,
    geodesic_circle_points,
)
from Buoy.plots.wipha_case_common import (
    SELECTED_PLATFORMS,
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


def test_split_wipha_plot_modules_are_importable():
    modules = [
        "Buoy.plots.plot_wipha_track_buoy_locations",
        "Buoy.plots.plot_wipha_buoy_wind_timeseries",
        "Buoy.plots.plot_wipha_buoy_wind_statistics_table",
        "Buoy.plots.plot_wipha_track_forecast_error",
        "Buoy.plots.plot_wipha_case_analysis",
    ]
    for module in modules:
        assert importlib.import_module(module)


def test_selected_wipha_platforms_are_best_coverage_pair():
    assert SELECTED_PLATFORMS == ["EVH28KM", "YRZSQRB"]


def test_load_wipha_track_csv_normalizes_local_track_columns(tmp_path):
    csv_path = tmp_path / "typhoon_2506_Wipha.csv"
    csv_path.write_text(
        "tc_num,name_cn,name_en,dateUTC,dateCST,vmax,grade,latTC,lonTC,mslp,attr\n"
        "2506,韦帕,Wipha,202507170000,202507170800,15,热带低压,14.6,127.2,1000,analysis\n",
        encoding="utf-8",
    )

    track = load_wipha_track_csv(csv_path)

    assert list(track.columns) == [
        "tc_num",
        "name_cn",
        "name_en",
        "datetime_utc",
        "vmax_ms",
        "grade",
        "lon",
        "lat",
        "mslp_hpa",
        "attr",
        "source",
    ]
    assert track.loc[0, "datetime_utc"] == pd.Timestamp("2025-07-17 00:00:00")
    assert track.loc[0, "lon"] == 127.2
    assert track.loc[0, "lat"] == 14.6


def test_select_six_hour_track_points_keeps_synoptic_hours():
    track = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(
                [
                    "2025-07-17 00:00",
                    "2025-07-17 03:00",
                    "2025-07-17 06:00",
                    "2025-07-17 09:00",
                    "2025-07-17 12:00",
                ]
            ),
            "lon": [127.2, 126.5, 125.8, 125.0, 124.3],
            "lat": [14.6, 14.9, 15.2, 15.6, 16.0],
        }
    )

    selected = select_six_hour_track_points(track)

    assert selected["datetime_utc"].dt.hour.tolist() == [0, 6, 12]
    assert selected["lon"].tolist() == [127.2, 125.8, 124.3]


def test_track_time_labels_in_late_landfall_window_use_land_arrow_labels():
    annotation = track_time_label_annotation(pd.Timestamp("2025-07-20 12:00"), lon=112.0, lat=21.7)

    assert annotation["xy"] == (112.0, 21.7)
    assert annotation["xytext"][1] >= 22.3
    assert annotation["arrowprops"]["arrowstyle"] == "->"
    assert annotation["textcoords"] == "data"


def test_track_time_labels_in_late_landfall_window_are_placed_on_land_side():
    for timestamp in pd.date_range("2025-07-20 12:00", "2025-07-22 00:00", freq="6h"):
        annotation = track_time_label_annotation(timestamp, lon=112.0, lat=21.7)

        assert annotation["xytext"][1] >= 22.3
        assert "arrowprops" in annotation


def test_track_time_labels_outside_late_landfall_window_use_nearby_labels():
    annotation = track_time_label_annotation(pd.Timestamp("2025-07-20 06:00"), lon=113.5, lat=21.8)

    assert annotation["xy"] == (113.5, 21.8)
    assert annotation["xytext"] == (113.65, 21.92)
    assert "arrowprops" not in annotation


def test_select_fixed_init_timeseries_samples_keeps_requested_starts_and_three_hour_times():
    raw = pd.DataFrame(
        {
            "dataset": [
                "gdas_forecast",
                "gdas_forecast",
                "gdas_forecast",
                "era5_lagged_5d",
                "era5_lagged_5d",
                "era5_lagged_5d",
            ],
            "pred_start_time": [
                "2025-07-18-00-00",
                "2025-07-17-00-00",
                "2025-07-18-00-00",
                "2025-07-13-00-00",
                "2025-07-12-00-00",
                "2025-07-13-00-00",
            ],
            "datetime_utc": pd.to_datetime(
                [
                    "2025-07-18 03:00",
                    "2025-07-18 03:00",
                    "2025-07-18 04:00",
                    "2025-07-18 06:00",
                    "2025-07-18 06:00",
                    "2025-07-18 07:00",
                ]
            ),
            "platform_id": ["EVH28KM"] * 6,
            "lead_hour": [3, 3, 4, 6, 6, 7],
        }
    )

    selected = select_fixed_init_timeseries_samples(raw)

    assert selected[["dataset", "pred_start_time", "datetime_utc", "lead_hour"]].to_dict("records") == [
        {
            "dataset": "gdas_forecast",
            "pred_start_time": TIMESERIES_FORECAST_INIT_TIMES["gdas_forecast"],
            "datetime_utc": pd.Timestamp("2025-07-18 03:00"),
            "lead_hour": 3,
        },
        {
            "dataset": "era5_lagged_5d",
            "pred_start_time": TIMESERIES_FORECAST_INIT_TIMES["era5_lagged_5d"],
            "datetime_utc": pd.Timestamp("2025-07-18 06:00"),
            "lead_hour": 6,
        },
    ]


def test_timeseries_time_axis_bounds_use_actual_fixed_init_data_range():
    obs = pd.DataFrame({"datetime_utc": pd.to_datetime(["2025-07-18 06:00", "2025-07-20 12:00"])})
    merged = pd.DataFrame({"datetime_utc": pd.to_datetime(["2025-07-18 03:00", "2025-07-21 00:00"])})

    x_min, x_max = timeseries_time_axis_bounds(obs, merged)

    assert x_min == pd.Timestamp("2025-07-18 03:00")
    assert x_max == pd.Timestamp("2025-07-21 00:00")


def test_statistics_table_uses_fixed_init_timeseries_samples():
    merged = prepare_statistics_input()

    assert set(merged["platform_id"]) == {"EVH28KM", "YRZSQRB"}
    assert set(merged.loc[merged["dataset"] == "gdas_forecast", "pred_start_time"]) == {
        TIMESERIES_FORECAST_INIT_TIMES["gdas_forecast"]
    }
    assert set(merged.loc[merged["dataset"] == "era5_lagged_5d", "pred_start_time"]) == {
        TIMESERIES_FORECAST_INIT_TIMES["era5_lagged_5d"]
    }


def test_wipha_virtual_station_radius_settings_are_fixed():
    by_id = {station["station_id"]: station for station in VIRTUAL_STATIONS}

    assert by_id["VS1"]["lon"] == 120.0
    assert by_id["VS1"]["lat"] == 20.0
    assert by_id["VS1"]["radius_km"] == 300.0
    assert by_id["VS2"]["lon"] == 113.0
    assert by_id["VS2"]["lat"] == 21.0
    assert by_id["VS2"]["radius_km"] == 400.0


def test_geodesic_circle_points_are_closed():
    points = geodesic_circle_points(120.0, 20.0, 300.0, n_points=13)

    assert len(points) == 13
    assert points[0] == points[-1]
