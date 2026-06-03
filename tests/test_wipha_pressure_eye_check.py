from pathlib import Path

import pytest

from Buoy.plots import plot_wipha_pressure_eye_check as pressure_eye_check


def test_pressure_eye_check_leads_cover_2025071700_to_2025072000_every_three_hours():
    leads = pressure_eye_check.pressure_eye_check_leads()

    assert leads[0] == 0
    assert leads[-1] == 72
    assert leads == list(range(0, 73, 3))


def test_normalize_scheme_accepts_short_aliases():
    assert pressure_eye_check.normalize_scheme("gdas") == "gdas_forecast"
    assert pressure_eye_check.normalize_scheme("era5") == "era5_lagged_5d"
    assert pressure_eye_check.normalize_scheme("era5_lagged") == "era5_lagged_5d"


def test_schemes_to_run_can_select_all_or_one_scheme():
    assert pressure_eye_check.schemes_to_run("all") == ["gdas_forecast", "era5_lagged_5d"]
    assert pressure_eye_check.schemes_to_run("gdas") == ["gdas_forecast"]


def test_output_dir_is_split_by_scheme(tmp_path: Path):
    assert pressure_eye_check.scheme_output_dir(tmp_path, "gdas_forecast") == tmp_path / "gdas_forecast"
    assert pressure_eye_check.scheme_output_dir(tmp_path, "era5_lagged_5d") == tmp_path / "era5_lagged_5d"


def test_normalize_scheme_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported scheme"):
        pressure_eye_check.normalize_scheme("unknown")


def test_load_manual_eye_overrides_accepts_scheme_aliases(tmp_path: Path):
    csv_path = tmp_path / "manual_eye.csv"
    csv_path.write_text(
        "scheme,lead_hour,lon,lat\n"
        "era5,0,126.8,14.7\n"
        "era5_lagged,3,126.1,15.0\n"
        "gdas,6,125.4,15.2\n",
        encoding="utf-8",
    )

    overrides = pressure_eye_check.load_manual_eye_overrides(csv_path)

    assert overrides[("era5_lagged_5d", 0)] == {"lon": 126.8, "lat": 14.7}
    assert overrides[("era5_lagged_5d", 3)] == {"lon": 126.1, "lat": 15.0}
    assert overrides[("gdas_forecast", 6)] == {"lon": 125.4, "lat": 15.2}


def test_load_manual_eye_overrides_rejects_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "manual_eye.csv"
    csv_path.write_text("scheme,lead_hour,lon\nera5,0,126.8\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        pressure_eye_check.load_manual_eye_overrides(csv_path)


def test_apply_manual_eye_override_replaces_center_and_marks_source():
    auto_eye = {"center_lon": 130.0, "center_lat": 10.0, "min_msl_hpa": 999.0, "search_box": "auto"}
    overrides = {("era5_lagged_5d", 0): {"lon": 126.8, "lat": 14.7}}

    manual_eye = pressure_eye_check.apply_manual_eye_override(
        auto_eye,
        overrides,
        scheme="era5_lagged_5d",
        lead_hour=0,
        manual_msl_hpa=997.5,
    )

    assert manual_eye["center_lon"] == 126.8
    assert manual_eye["center_lat"] == 14.7
    assert manual_eye["min_msl_hpa"] == 997.5
    assert manual_eye["manual_override"]


def test_upsert_manual_eye_override_creates_and_replaces_rows(tmp_path: Path):
    csv_path = tmp_path / "manual_eye_overrides.csv"

    pressure_eye_check.upsert_manual_eye_override(csv_path, "era5", 0, 126.8, 14.7)
    pressure_eye_check.upsert_manual_eye_override(csv_path, "gdas", 3, 125.1, 15.2)
    pressure_eye_check.upsert_manual_eye_override(csv_path, "era5_lagged", 0, 126.9, 14.8)

    rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
    assert rows[0] == "scheme,lead_hour,lon,lat"
    assert "era5_lagged_5d,0,126.9,14.8" in rows
    assert "gdas_forecast,3,125.1,15.2" in rows
    assert "era5_lagged_5d,0,126.8,14.7" not in rows


def test_upsert_manual_eye_override_rejects_outside_domain(tmp_path: Path):
    csv_path = tmp_path / "manual_eye_overrides.csv"

    with pytest.raises(ValueError, match="outside plot domain"):
        pressure_eye_check.upsert_manual_eye_override(csv_path, "era5", 0, 140.0, 14.7)
