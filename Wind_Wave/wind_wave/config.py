from __future__ import annotations

from pathlib import Path


INPUT_VARIABLES = ("u10", "v10")
TARGET_VARIABLES = ("swh", "mwp", "cos_mwd", "sin_mwd")
DEFAULT_HISTORY_HOURS = 24
DEFAULT_LEAD_HOURS = (6, 12, 24, 48, 72)
DEFAULT_INPUT_REGION = "5,45,95,150"
DEFAULT_OUTPUT_REGION = "15,40,105,135"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "data"


def raw_data_dir(year: str = "2025") -> Path:
    return data_dir() / year


def extracted_data_dir(year: str = "2025") -> Path:
    return data_dir() / "extracted" / year


def grib_raw_data_dir() -> Path:
    return data_dir() / "raw"


def converted_data_dir() -> Path:
    return data_dir() / "converted"


def outputs_dir() -> Path:
    return project_root() / "outputs"
