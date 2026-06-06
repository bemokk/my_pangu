from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


def _to_timestamp_list(values: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    return [pd.Timestamp(value) for value in values]


def build_valid_initialization_times(
    wind_times: Iterable[pd.Timestamp],
    wave_times: Iterable[pd.Timestamp],
    history_hours: int,
    lead_hours: Sequence[int],
) -> list[pd.Timestamp]:
    wind_set = set(_to_timestamp_list(wind_times))
    wave_set = set(_to_timestamp_list(wave_times))
    valid = []

    for t0 in sorted(wind_set & wave_set):
        history = [t0 - pd.Timedelta(hours=offset) for offset in range(history_hours)]
        leads = [t0 + pd.Timedelta(hours=lead) for lead in lead_hours]
        if all(t in wind_set for t in history) and all(t in wave_set for t in leads):
            valid.append(t0)

    return valid


def chronological_split(
    times: Iterable[pd.Timestamp],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    ordered = sorted(_to_timestamp_list(times))
    if len(ordered) < 3:
        raise ValueError("Need at least three times for train, validation, and test splits")

    train_end = max(1, min(len(ordered) - 2, int(len(ordered) * train_fraction)))
    val_end = max(
        train_end + 1,
        min(len(ordered) - 1, train_end + int(len(ordered) * val_fraction)),
    )

    train = ordered[:train_end]
    val = ordered[train_end:val_end]
    test = ordered[val_end:]

    if not train or not val or not test:
        raise ValueError("Chronological split produced an empty train, validation, or test split")

    return train, val, test
