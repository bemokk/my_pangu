from __future__ import annotations

import sys
from pathlib import Path

BUOY_DIR = Path(__file__).resolve().parents[1]
if str(BUOY_DIR) not in sys.path:
    sys.path.insert(0, str(BUOY_DIR))

from plots.plot_wipha_buoy_wind_statistics_table import generate as generate_statistics_table
from plots.plot_wipha_buoy_wind_timeseries import generate as generate_timeseries
from plots.plot_wipha_track_buoy_locations import generate as generate_track_buoy_locations
from plots.plot_wipha_track_forecast_error import generate as generate_track_forecast_error


def run_workflow() -> list[Path]:
    outputs: list[Path] = []
    outputs.extend(generate_track_buoy_locations())
    outputs.extend(generate_timeseries())
    outputs.extend(generate_statistics_table())
    outputs.extend(generate_track_forecast_error())
    return outputs


def main() -> None:
    print("Wipha case analysis outputs:")
    for path in run_workflow():
        print(path if path.exists() else f"{path} (not written)")


if __name__ == "__main__":
    main()
