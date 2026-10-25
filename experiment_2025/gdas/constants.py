"""Fixed GDAS conventions used by the Pangu model input reader."""

from __future__ import annotations

from pathlib import Path
import os

from ..core.config import GRID_SHAPE, PRESSURE_LEVELS, SURFACE_ORDER, UPPER_ORDER
from ..core.config import GDAS_ROOT

BASE_URL = "https://data.gdex.ucar.edu/d083003/{year}/{yyyymm}/{filename}"
SUPPORTED_CYCLES = (0, 6, 12, 18)
GRID = tuple(GRID_SHAPE)
GRAVITY = 9.80665

SURFACE_TARGETS = {
    "msl": {"shortName": "prmsl", "typeOfLevel": "meanSea", "level": 0},
    "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
    "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
    "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
}
UPPER_SHORT_NAMES = {"gh": "z", "q": "q", "t": "t", "u": "u", "v": "v"}
SURFACE_UNITS = {"msl": "Pa", "u10": "m s-1", "v10": "m s-1", "t2m": "K"}
UPPER_UNITS = {
    "z": "m2 s-2",
    "q": "kg kg-1",
    "t": "K",
    "u": "m s-1",
    "v": "m s-1",
}


def default_raw_root() -> Path:
    """Return the raw archive location independently of the processed inputs."""
    return Path(os.environ.get("PANGU_GDAS_RAW_ROOT", str(GDAS_ROOT / "raw")))

