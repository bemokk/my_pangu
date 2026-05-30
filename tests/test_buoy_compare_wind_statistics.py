from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

import compare_buoy_wind_statistics as compare  # noqa: E402


def test_surface_wind_sampler_reuses_open_dataset(monkeypatch, tmp_path):
    nc_path = tmp_path / "wind10.nc"
    times = pd.to_datetime(["2025-08-01 00:00", "2025-08-01 03:00"])
    ds = xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), np.ones((2, 2, 2), dtype=np.float32)),
            "v10": (("valid_time", "latitude", "longitude"), np.ones((2, 2, 2), dtype=np.float32) * 2),
        },
        coords={
            "valid_time": times,
            "latitude": [10.0, 11.0],
            "longitude": [120.0, 121.0],
        },
    )
    ds.to_netcdf(nc_path)

    real_open_dataset = xr.open_dataset
    opened = []

    def counting_open_dataset(*args, **kwargs):
        opened.append(args[0])
        return real_open_dataset(*args, **kwargs)

    monkeypatch.setattr(compare.xr, "open_dataset", counting_open_dataset)

    records = pd.DataFrame({"latitude": [10.5], "longitude": [120.5]})
    with compare.SurfaceWindSampler(nc_path) as sampler:
        sampler.sample(records, valid_time=datetime(2025, 8, 1, 0))
        sampler.sample(records, valid_time=datetime(2025, 8, 1, 3))

    assert opened == [nc_path]


def test_build_matches_reuses_realtime_era5_sampler(monkeypatch, tmp_path):
    nc_path = tmp_path / "model_input" / "multi_time_point" / "wind10.nc"
    nc_path.parent.mkdir(parents=True)
    times = pd.to_datetime(["2025-08-01 03:00", "2025-08-01 06:00"])
    ds = xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), np.ones((2, 2, 2), dtype=np.float32)),
            "v10": (("valid_time", "latitude", "longitude"), np.ones((2, 2, 2), dtype=np.float32) * 2),
        },
        coords={
            "valid_time": times,
            "latitude": [10.0, 11.0],
            "longitude": [120.0, 121.0],
        },
    )
    ds.to_netcdf(nc_path)

    monkeypatch.setattr(compare, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(compare, "ERA5_REALTIME_WIND10_NC", nc_path)
    monkeypatch.setattr(compare, "DATASETS", [compare.DatasetConfig("era5_realtime", "ERA5 realtime")])

    real_open_dataset = xr.open_dataset
    opened = []

    def counting_open_dataset(*args, **kwargs):
        opened.append(args[0])
        return real_open_dataset(*args, **kwargs)

    monkeypatch.setattr(compare.xr, "open_dataset", counting_open_dataset)

    records = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(["2025-08-01 03:00", "2025-08-01 06:00"]),
            "latitude": [10.5, 10.5],
            "longitude": [120.5, 120.5],
            "obs_speed_ms": [1.0, 1.0],
            "obs_dir_deg": [90.0, 90.0],
        }
    )
    matched, missing, missing_observations = compare.build_matches(
        records,
        lead_hours=[3, 6],
        target_start=datetime(2025, 8, 1, 0),
        target_end=datetime(2025, 8, 1, 0),
    )

    assert opened == [nc_path]
    assert len(matched) == 2
    assert missing.empty
    assert missing_observations.empty
