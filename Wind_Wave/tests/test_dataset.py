import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wind_wave.dataset import (
    NormalizationStats,
    WindWaveSeq2SeqDataset,
    compute_normalization_stats,
)


def make_synthetic_pair():
    times = pd.date_range("2025-01-01", periods=120, freq="h")
    lat = np.linspace(10.0, 8.0, 6)
    lon = np.linspace(120.0, 123.0, 8)
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
            "pp1d": (("time", "latitude", "longitude"), base + 4.0),
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
        target_names=("swh", "mwp", "pp1d", "sin_mwd", "cos_mwd"),
    )
    ds = WindWaveSeq2SeqDataset(
        wind,
        wave,
        initialization_times=[pd.Timestamp("2025-01-02T00:00")],
        stats=stats,
        spatial_stride=2,
    )

    sample = ds[0]

    assert sample["inputs"].shape == (24, 2, 3, 4)
    assert sample["targets"].shape == (5, 5, 3, 4)
    assert sample["target_times"][0] == "2025-01-02T06:00:00"


def test_dataset_uses_mean_period_as_peak_period_fallback_when_peak_missing():
    wind, wave = make_synthetic_pair()
    wave = wave.drop_vars("pp1d")
    stats = NormalizationStats.identity(
        input_names=("u10", "v10"),
        target_names=("swh", "mwp", "pp1d", "sin_mwd", "cos_mwd"),
    )
    ds = WindWaveSeq2SeqDataset(
        wind,
        wave,
        initialization_times=[pd.Timestamp("2025-01-02T00:00")],
        stats=stats,
        spatial_stride=2,
    )

    sample = ds[0]

    np.testing.assert_allclose(sample["targets"][:, 2].numpy(), sample["targets"][:, 1].numpy())


def test_compute_normalization_stats_rejects_all_nan_values():
    wind, wave = make_synthetic_pair()
    wind["u10"][:] = np.nan

    with pytest.raises(ValueError, match="finite"):
        compute_normalization_stats(
            wind,
            wave,
            initialization_times=[pd.Timestamp("2025-01-02T00:00")],
            spatial_stride=2,
        )
