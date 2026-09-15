from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

from plots import plot_wind_compare as wind_compare  # noqa: E402


class _DatasetContext:
    def __init__(self, dataset: xr.Dataset) -> None:
        self.dataset = dataset

    def __enter__(self) -> xr.Dataset:
        return self.dataset

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dataset.close()


def _surface_dataset() -> xr.Dataset:
    lat = np.asarray([30.0, 20.0, 10.0], dtype=np.float32)
    lon = np.asarray([105.0, 117.5, 130.0], dtype=np.float32)
    shape = (1, lat.size, lon.size)
    return xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
            "v10": (("valid_time", "latitude", "longitude"), np.ones(shape, dtype=np.float32) * 2),
        },
        coords={
            "valid_time": [np.datetime64("2025-07-18T00:00")],
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_plot_wind_compare_uses_noninteractive_matplotlib_backend():
    assert wind_compare.plt.get_backend().lower() == "agg"


def test_load_surface_wind10_components_opens_netcdf_without_backend_autodiscovery(monkeypatch, tmp_path):
    calls = []

    def fake_open_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _DatasetContext(_surface_dataset())

    monkeypatch.setattr(wind_compare.xr, "open_dataset", fake_open_dataset)
    nc_path = tmp_path / "surface.nc"
    nc_path.write_bytes(b"placeholder")

    wind_compare.load_surface_wind10_components(nc_path, wind_compare.AREA)

    assert calls
    assert calls[0][1]["engine"] == "netcdf4"
