"""Paths and fixed model-input conventions for the 2025 experiment."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
MODEL_ROOT = PROJECT_ROOT / "models"

# The H: drive is the default data location used for the 2025 ERA5 archive.
ERA5_ROOT = Path(os.environ.get("PANGU_ERA5_ROOT", r"H:\pangu_2025_data\model_input\era5"))
GDAS_ROOT = Path(
    os.environ.get(
        "PANGU_GDAS_ROOT",
        r"H:\pangu_2025_data\model_input\gdas",
    )
)
OUTPUT_ROOT = Path(
    os.environ.get("PANGU_2025_OUTPUT_ROOT", r"H:\pangu_2025_data\model_output\experiment_2025")
)

GRID_SHAPE = (721, 1440)
PRESSURE_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
SURFACE_ORDER = ("msl", "u10", "v10", "t2m")
UPPER_ORDER = ("z", "q", "t", "u", "v")
MODEL_STEPS = (1, 3, 6, 24)
DEFAULT_LEADS = tuple(range(0, 73, 3))
STATE_CACHE_VERSION = 2


def model_path(step: int) -> Path:
    return MODEL_ROOT / f"pangu_weather_{step}.onnx"

