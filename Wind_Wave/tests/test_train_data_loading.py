import numpy as np
import pandas as pd
import xarray as xr

from wind_wave.extract import ExtractedPair
from wind_wave.train import _open_pairs


def test_open_single_pair_preserves_native_wind_and_wave_grids(tmp_path, monkeypatch):
    times = pd.date_range("2025-01-01", periods=3, freq="h")
    wind_lat = np.array([11.0, 10.5, 10.0, 9.5, 9.0])
    wind_lon = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
    wave_lat = np.array([11.0, 10.0, 9.0])
    wave_lon = np.array([100.0, 101.0, 102.0])

    wind = xr.Dataset(
        {
            "u10": (
                ("time", "latitude", "longitude"),
                np.zeros((3, 5, 5), dtype=np.float32),
            ),
            "v10": (
                ("time", "latitude", "longitude"),
                np.ones((3, 5, 5), dtype=np.float32),
            ),
        },
        coords={"time": times, "latitude": wind_lat, "longitude": wind_lon},
    )
    wave = xr.Dataset(
        {
            "swh": (
                ("time", "latitude", "longitude"),
                np.zeros((3, 3, 3), dtype=np.float32),
            ),
            "mwp": (
                ("time", "latitude", "longitude"),
                np.ones((3, 3, 3), dtype=np.float32),
            ),
            "mwd": (
                ("time", "latitude", "longitude"),
                np.full((3, 3, 3), 90.0, dtype=np.float32),
            ),
        },
        coords={"time": times, "latitude": wave_lat, "longitude": wave_lon},
    )

    oper_path = tmp_path / "oper.nc"
    wave_path = tmp_path / "wave.nc"
    wind.to_netcdf(oper_path, engine="netcdf4")
    wave.to_netcdf(wave_path, engine="netcdf4")
    pair = ExtractedPair(
        archive=tmp_path / "source.zip",
        extract_dir=tmp_path,
        oper_nc=oper_path,
        wave_nc=wave_path,
    )
    monkeypatch.setattr(
        xr,
        "concat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("single-pair loading must not concatenate")
        ),
    )

    opened_wind, opened_wave = _open_pairs([pair])

    assert opened_wind.sizes["latitude"] == 5
    assert opened_wind.sizes["longitude"] == 5
    assert opened_wave.sizes["latitude"] == 3
    assert opened_wave.sizes["longitude"] == 3


def test_open_multiple_pairs_lazily_combines_and_sorts_time(tmp_path, monkeypatch):
    pairs = []
    for name, start in (("later", "2025-02-01"), ("earlier", "2025-01-01")):
        pair_dir = tmp_path / name
        pair_dir.mkdir()
        times = pd.date_range(start, periods=2, freq="h")
        coords = {
            "valid_time": times,
            "latitude": np.array([11.0, 10.0]),
            "longitude": np.array([100.0, 101.0]),
        }
        wind = xr.Dataset(
            {
                "u10": (
                    ("valid_time", "latitude", "longitude"),
                    np.zeros((2, 2, 2), dtype=np.float32),
                ),
                "v10": (
                    ("valid_time", "latitude", "longitude"),
                    np.ones((2, 2, 2), dtype=np.float32),
                ),
            },
            coords=coords,
        )
        wave = xr.Dataset(
            {
                "swh": (
                    ("valid_time", "latitude", "longitude"),
                    np.zeros((2, 2, 2), dtype=np.float32),
                ),
                "mwp": (
                    ("valid_time", "latitude", "longitude"),
                    np.ones((2, 2, 2), dtype=np.float32),
                ),
                "mwd": (
                    ("valid_time", "latitude", "longitude"),
                    np.full((2, 2, 2), 90.0, dtype=np.float32),
                ),
            },
            coords=coords,
        )
        oper_path = pair_dir / "oper.nc"
        wave_path = pair_dir / "wave.nc"
        wind.to_netcdf(oper_path, engine="netcdf4")
        wave.to_netcdf(wave_path, engine="netcdf4")
        pairs.append(
            ExtractedPair(
                archive=tmp_path / f"{name}.zip",
                extract_dir=pair_dir,
                oper_nc=oper_path,
                wave_nc=wave_path,
            )
        )

    monkeypatch.setattr(
        xr,
        "concat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("multi-pair loading must use lazy multifile opening")
        ),
    )

    opened_wind, opened_wave = _open_pairs(pairs)

    expected_times = pd.DatetimeIndex(
        [
            "2025-01-01T00:00",
            "2025-01-01T01:00",
            "2025-02-01T00:00",
            "2025-02-01T01:00",
        ]
    )
    assert pd.DatetimeIndex(opened_wind["time"].values).equals(expected_times)
    assert pd.DatetimeIndex(opened_wave["time"].values).equals(expected_times)
    assert opened_wind["u10"].chunks is not None
    assert opened_wave["swh"].chunks is not None
