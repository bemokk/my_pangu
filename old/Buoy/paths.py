from __future__ import annotations

from pathlib import Path


BUOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUOY_DIR.parent

RESULTS_DIR = BUOY_DIR / "results"
CHINA_SEA_RECORDS_DIR = RESULTS_DIR / "china_sea_platform_records"
WIND_MODEL_STATISTICS_DIR = RESULTS_DIR / "wind_model_statistics"
FIXED_BUOY_WIND_DIR = RESULTS_DIR / "fixed_buoy_wind"
FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_CHINA_SEA_DETAIL_CSV = (
    CHINA_SEA_RECORDS_DIR / "china_sea_all_platform_records_area_42_103_13_130.csv"
)
DEFAULT_CHINA_SEA_SUMMARY_CSV = (
    CHINA_SEA_RECORDS_DIR / "china_sea_all_platform_summary_area_42_103_13_130.csv"
)


def icoads_root(year: int, month: int) -> Path:
    return BUOY_DIR / f"icoads_{year}{month:02d}"


def default_icoads_nc_dirs() -> list[Path]:
    return sorted(path / "nc" for path in BUOY_DIR.glob("icoads_*") if (path / "nc").exists())
