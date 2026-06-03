from pathlib import Path

from Buoy.tools import wipha_pressure_eye_picker as picker


def test_manual_eye_csv_path_defaults_to_pressure_eye_check_folder():
    path = picker.default_manual_eye_csv_path()

    assert path.name == "manual_eye_overrides.csv"
    assert path.parent.name == "wipha_pressure_eye_check"


def test_scheme_options_include_gdas_and_era5():
    options = picker.scheme_options()

    assert ("gdas_forecast", "GDAS forecast") in options
    assert ("era5_lagged_5d", "ERA5 lagged 5d forecast") in options


def test_lead_options_cover_every_three_hours():
    options = picker.lead_options()

    assert options[0] == (0, "+00 h / 2025-07-17 00 UTC")
    assert options[-1] == (72, "+72 h / 2025-07-20 00 UTC")


def test_format_click_status_rounds_coordinates():
    status = picker.format_click_status("era5_lagged_5d", 3, 126.12345, 14.98765, Path("manual.csv"))

    assert "ERA5 lagged 5d forecast" in status
    assert "+03 h" in status
    assert "126.123E" in status
    assert "14.988N" in status
    assert "manual.csv" in status
