"""Forecast experiment definitions. All datetimes are naive UTC."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .config import DEFAULT_LEADS


DEFAULT_LEAD_HOURS = DEFAULT_LEADS


@dataclass(frozen=True)
class Experiment:
    name: str
    label: str
    source: str
    lag_hours: int

    def initial_time(self, target_time: datetime) -> datetime:
        return target_time - timedelta(hours=self.lag_hours)

    def model_hour(self, lead_hour: int) -> int:
        return self.lag_hours + lead_hour


EXPERIMENTS = {
    item.name: item
    for item in (
        Experiment("gdas_direct", "GDAS直接起报", "gdas", 0),
        Experiment("era5_lagged_120h", "ERA5滞后起报", "era5", 120),
        Experiment("era5_direct", "ERA5直接起报", "era5", 0),
        Experiment("gdas_lagged_120h", "GDAS滞后起报", "gdas", 120),
    )
}


def time_label(value: datetime) -> str:
    return value.strftime("%Y-%m-%d-%H-%M")


def run_directory(output_root: Path, experiment: Experiment, target_time: datetime) -> Path:
    return Path(output_root) / experiment.name / str(target_time.year) / time_label(target_time)


def wind_path(output_root: Path, experiment: Experiment, target_time: datetime, lead_hour: int) -> Path:
    valid_time = target_time + timedelta(hours=lead_hour)
    return run_directory(output_root, experiment, target_time) / str(lead_hour) / f"wind10_{time_label(valid_time)}.nc"
