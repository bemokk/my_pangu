import numpy as np
import pandas as pd
import xarray as xr

from wind_wave.era5 import (
    direction_degrees_to_unit,
    find_data_var,
    normalize_time_coord,
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
