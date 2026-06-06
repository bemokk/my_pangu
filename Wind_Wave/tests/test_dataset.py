import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wind_wave.dataset import (
    NormalizationStats,
    WindWaveSeq2SeqDataset,
    compute_normalization_stats,
)
from wind_wave.era5 import Region


def make_synthetic_pair():
    times = pd.date_range("2025-01-01", periods=120, freq="h")
    lat = np.arange(50.0, -5.0, -5.0)
    lon = np.arange(90.0, 160.0, 5.0)
    shape = (len(times), len(lat), len(lon))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    wind = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), base),
            "v10": (("time", "latitude", "longitude"), base + 1.0),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    wave = xr.Dataset(
        {
            "swh": (("time", "latitude", "longitude"), base + 2.0),
            "mwp": (("time", "latitude", "longitude"), base + 3.0),
            "mwd": (
                ("time", "latitude", "longitude"),
                np.full(shape, 90.0, dtype=np.float32),
            ),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    return wind, wave


def test_dataset_returns_expected_seq2seq_shapes_with_stride():
    wind, wave = make_synthetic_pair()
    stats = NormalizationStats.identity(
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "cos_mwd", "sin_mwd"),
    )
    ds = WindWaveSeq2SeqDataset(
        wind,
        wave,
        initialization_times=[pd.Timestamp("2025-01-02T00:00")],
        stats=stats,
        spatial_stride=2,
        input_region=Region(south=5.0, north=45.0, west=95.0, east=150.0),
        output_region=Region(south=15.0, north=40.0, west=105.0, east=135.0),
    )

    sample = ds[0]

    assert sample["inputs"].shape == (24, 2, 5, 6)
    assert sample["targets"].shape == (5, 4, 3, 4)
    assert sample["persistence"].shape == sample["targets"].shape
    assert sample["target_times"][0] == "2025-01-02T06:00:00"
    np.testing.assert_allclose(sample["targets"][:, 2].numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(sample["targets"][:, 3].numpy(), 1.0, atol=1e-6)
    for lead_index in range(1, sample["persistence"].shape[0]):
        np.testing.assert_allclose(
            sample["persistence"][lead_index].numpy(),
            sample["persistence"][0].numpy(),
        )
    expected_swh = (
        wave["swh"]
        .sel(time=pd.Timestamp("2025-01-02T00:00"))
        .sel(latitude=slice(40.0, 15.0), longitude=slice(105.0, 135.0))
        .isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
        .values
    )
    np.testing.assert_allclose(sample["persistence"][0, 0].numpy(), expected_swh)


def test_compute_normalization_stats_rejects_all_nan_values():
    wind, wave = make_synthetic_pair()
    wind["u10"][:] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_normalization_stats(
            wind,
            wave,
            initialization_times=[pd.Timestamp("2025-01-02T00:00")],
            spatial_stride=2,
            input_region=Region(south=5.0, north=45.0, west=95.0, east=150.0),
            output_region=Region(south=15.0, north=40.0, west=105.0, east=135.0),
        )
