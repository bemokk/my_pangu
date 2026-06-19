from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wind_wave.convert_zarr import _write_zarr_group, convert_grib_to_zarr


def test_convert_grib_to_zarr_writes_wind_and_wave_groups(tmp_path):
    grib_path = tmp_path / "source.grib"
    grib_path.write_bytes(b"fake")
    store_path = tmp_path / "source.zarr"

    def fake_opener(path: Path, short_name: str) -> xr.Dataset:
        assert path == grib_path
        times = pd.to_datetime(["2016-01-01T00:00", "2016-01-01T01:00"])
        if short_name in {"10u", "10v"}:
            lat = np.array([6.0, 5.0])
            lon = np.array([95.0, 96.0])
        else:
            lat = np.array([6.0, 5.0])
            lon = np.array([95.0, 96.0])
        values = np.full((len(times), len(lat), len(lon)), len(short_name), dtype=np.float32)
        return xr.Dataset(
            {short_name: (("time", "latitude", "longitude"), values)},
            coords={"time": times, "latitude": lat, "longitude": lon},
        )

    result = convert_grib_to_zarr(
        grib_path=grib_path,
        output_store=store_path,
        region_text="5,6,95,96",
        time_chunk=1,
        overwrite=False,
        opener=fake_opener,
    )

    assert result == store_path
    wind = xr.open_zarr(store_path, group="wind", chunks=None, consolidated=True)
    wave = xr.open_zarr(store_path, group="wave", chunks=None, consolidated=True)
    try:
        assert set(wind.data_vars) == {"u10", "v10"}
        assert wind.sizes == {"time": 2, "latitude": 2, "longitude": 2}
        assert set(wave.data_vars) == {"swh", "mwp", "mwd"}
        assert wave.sizes == {"time": 2, "latitude": 2, "longitude": 2}
    finally:
        wind.close()
        wave.close()


def test_write_zarr_group_forces_local_dask_scheduler(tmp_path, monkeypatch):
    import dask
    import dask.base

    original_get_scheduler = dask.base.get_scheduler

    def guarded_get_scheduler(*args, **kwargs):
        if dask.config.get("scheduler", None) != "synchronous":
            raise RuntimeError("distributed scheduler probe")
        return original_get_scheduler(*args, **kwargs)

    monkeypatch.setattr(dask.base, "get_scheduler", guarded_get_scheduler)
    ds = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.ones((2, 1, 1), dtype=np.float32))},
        coords={"time": pd.date_range("2016-01-01", periods=2, freq="h"), "latitude": [5.0], "longitude": [95.0]},
    )

    with dask.config.set(scheduler=None):
        _write_zarr_group(ds, tmp_path / "scheduler.zarr", "wind", "w", time_chunk=1)

    written = xr.open_zarr(tmp_path / "scheduler.zarr", group="wind", chunks=None, consolidated=True)
    try:
        assert set(written.data_vars) == {"u10"}
    finally:
        written.close()
