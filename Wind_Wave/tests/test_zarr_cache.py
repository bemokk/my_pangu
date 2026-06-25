from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from wind_wave.zarr_cache import build_zarr_cache, process_single_grib_to_zarr


def _fake_grib_opener(first: Path, second: Path | None = None):
    def fake_opener(path: Path, short_name: str) -> xr.Dataset:
        start = "2016-01-01T00:00"
        if second is not None and path == second:
            start = "2017-01-01T00:00"
        times = pd.date_range(start, periods=3, freq="h")
        lat = np.array([6.0, 5.0], dtype=np.float32)
        lon = np.array([95.0, 96.0], dtype=np.float32)
        fill_values = {"10u": 10.0, "10v": 20.0, "swh": 2.0, "mwp": 8.0, "mwd": 90.0}
        values = np.full((len(times), len(lat), len(lon)), fill_values[short_name], dtype=np.float32)
        return xr.Dataset(
            {short_name: (("time", "latitude", "longitude"), values)},
            coords={"time": times, "latitude": lat, "longitude": lon},
        )

    return fake_opener


def test_process_single_grib_to_zarr_writes_one_year_wind_and_wave(tmp_path):
    grib_path = tmp_path / "2016.grib"
    grib_path.write_bytes(b"fake")
    zarr_dir = tmp_path / "zarr"

    paths = process_single_grib_to_zarr(
        grib_path=grib_path,
        zarr_dir=zarr_dir,
        region_text="5,6,95,96",
        time_chunk=2,
        overwrite=True,
        opener=_fake_grib_opener(grib_path),
    )

    wind = xr.open_zarr(paths.wind_store, chunks=None, consolidated=True)
    wave = xr.open_zarr(paths.wave_store, chunks=None, consolidated=True)
    try:
        assert sorted(wind.data_vars) == ["u10", "v10"]
        assert sorted(wave.data_vars) == ["mwd", "mwd_cos", "mwd_sin", "mwp", "swh"]
        assert wind["latitude"].values.tolist() == [5.0, 6.0]
        assert wave["latitude"].values.tolist() == [5.0, 6.0]
        assert wind.sizes["time"] == 3
        assert wave.sizes["time"] == 3
        np.testing.assert_allclose(wave["mwd_cos"].isel(time=0).values, 0.0, atol=1e-6)
        np.testing.assert_allclose(wave["mwd_sin"].isel(time=0).values, 1.0, atol=1e-6)
    finally:
        wind.close()
        wave.close()


def test_build_zarr_cache_invokes_single_file_worker_for_each_grib(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    first = raw_dir / "2016.grib"
    second = raw_dir / "2017.grib"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    zarr_dir = tmp_path / "zarr"
    metadata_dir = tmp_path / "metadata"
    calls = []

    def fake_worker(grib_path: Path, zarr_dir: Path, region_text: str, time_chunk: int, overwrite: bool, append: bool):
        calls.append((grib_path.name, overwrite, append))
        process_single_grib_to_zarr(
            grib_path=grib_path,
            zarr_dir=zarr_dir,
            region_text=region_text,
            time_chunk=time_chunk,
            overwrite=overwrite,
            append=append,
            opener=_fake_grib_opener(first, second),
        )

    paths = build_zarr_cache(
        raw_dir=raw_dir,
        zarr_dir=zarr_dir,
        metadata_dir=metadata_dir,
        region_text="5,6,95,96",
        history_hours=2,
        lead_hours=(1,),
        time_chunk=2,
        overwrite=True,
        worker_runner=fake_worker,
    )

    assert calls == [("2016.grib", True, False), ("2017.grib", False, True)]
    wind = xr.open_zarr(paths.wind_store, chunks=None, consolidated=True)
    wave = xr.open_zarr(paths.wave_store, chunks=None, consolidated=True)
    try:
        assert wind.sizes["time"] == 6
        assert wave.sizes["time"] == 6
    finally:
        wind.close()
        wave.close()

    sample_t0 = np.load(metadata_dir / "sample_t0_indices.npy")
    train = np.load(metadata_dir / "train_indices.npy")
    val = np.load(metadata_dir / "val_indices.npy")
    test = np.load(metadata_dir / "test_indices.npy")

    assert sample_t0.tolist() == [1, 2, 3, 4]
    assert np.concatenate([train, val, test]).tolist() == sample_t0.tolist()
    assert (metadata_dir / "normalization.json").exists()
    assert (metadata_dir / "grid_wind_025.json").exists()
    assert (metadata_dir / "grid_wave_050.json").exists()
