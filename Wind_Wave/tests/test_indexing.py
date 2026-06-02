import pandas as pd

from wind_wave.indexing import build_valid_initialization_times, chronological_split


def test_valid_initialization_times_require_history_and_future_leads():
    times = pd.date_range("2025-01-01T00:00", periods=120, freq="h")

    valid = build_valid_initialization_times(
        wind_times=times,
        wave_times=times,
        history_hours=24,
        lead_hours=(6, 12, 24, 48, 72),
    )

    assert valid[0] == pd.Timestamp("2025-01-01T23:00")
    assert valid[-1] == pd.Timestamp("2025-01-02T23:00")
    assert len(valid) == 25


def test_chronological_split_preserves_order_and_no_overlap():
    times = pd.date_range("2025-01-01", periods=20, freq="h")

    train, val, test = chronological_split(times, train_fraction=0.7, val_fraction=0.15)

    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
    assert train[-1] < val[0] < test[0]
