import numpy as np
import pandas as pd
import xarray as xr

from wind_wave.era5 import (
    Region,
    align_wind_to_wave_grid,
    direction_degrees_to_unit,
    find_data_var,
    normalize_time_coord,
    parse_region,
    select_region,
)


def test_find_data_var_accepts_short_and_long_names():
    ds = xr.Dataset(
        {
            "10m_u_component_of_wind": (
                ("time", "latitude", "longitude"),
                np.zeros((1, 2, 2)),
            ),
            "swh": (("time", "latitude", "longitude"), np.ones((1, 2, 2))),
        }
    )

    assert (
        find_data_var(ds, ["u10", "10m_u_component_of_wind"])
        == "10m_u_component_of_wind"
    )
    assert (
        find_data_var(ds, ["swh", "significant_height_of_combined_wind_waves_and_swell"])
        == "swh"
    )


def test_normalize_time_coord_renames_valid_time():
    ds = xr.Dataset(
        {"u10": (("valid_time",), np.array([1.0]))},
        coords={"valid_time": pd.date_range("2025-01-01", periods=1, freq="h")},
    )

    normalized = normalize_time_coord(ds)

    assert "time" in normalized.dims
    assert "valid_time" not in normalized.dims


def test_direction_degrees_to_unit_wraps_around():
    sin_v, cos_v = direction_degrees_to_unit(np.array([0.0, 90.0, 360.0]))

    np.testing.assert_allclose(sin_v, np.array([0.0, 1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(cos_v, np.array([1.0, 0.0, 1.0]), atol=1e-6)


def test_align_wind_to_wave_grid_subsets_finer_wind_grid():
    wind = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.zeros((1, 5, 5)))},
        coords={
            "time": pd.date_range("2025-01-01", periods=1),
            "latitude": np.array([90.0, 89.75, 89.5, 89.25, 89.0]),
            "longitude": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        },
    )
    wave = xr.Dataset(
        {"swh": (("time", "latitude", "longitude"), np.zeros((1, 3, 3)))},
        coords={
            "time": pd.date_range("2025-01-01", periods=1),
            "latitude": np.array([90.0, 89.5, 89.0]),
            "longitude": np.array([0.0, 0.5, 1.0]),
        },
    )

    aligned = align_wind_to_wave_grid(wind, wave)

    assert aligned.sizes["latitude"] == 3
    assert aligned.sizes["longitude"] == 3
    np.testing.assert_allclose(aligned["latitude"].values, wave["latitude"].values)
    np.testing.assert_allclose(aligned["longitude"].values, wave["longitude"].values)


def test_parse_region_uses_south_north_west_east_order():
    region = parse_region("5,45,95,150")

    assert region == Region(south=5.0, north=45.0, west=95.0, east=150.0)


def test_select_region_handles_descending_latitude_and_ascending_longitude():
    ds = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.zeros((1, 11, 14)))},
        coords={
            "time": pd.date_range("2025-01-01", periods=1),
            "latitude": np.arange(50.0, -5.0, -5.0),
            "longitude": np.arange(90.0, 160.0, 5.0),
        },
    )

    selected = select_region(ds, Region(south=15.0, north=40.0, west=105.0, east=135.0))

    np.testing.assert_allclose(selected["latitude"].values, np.array([40.0, 35.0, 30.0, 25.0, 20.0, 15.0]))
    np.testing.assert_allclose(selected["longitude"].values, np.array([105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0]))
