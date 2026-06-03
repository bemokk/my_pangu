from pathlib import Path

import pandas as pd
import pytest

from Buoy.plots import plot_wipha_pressure_eye_check as pressure_eye_check
from Buoy.plots import plot_wipha_track_forecast_error as track_forecast_error
from Buoy.plots.wipha_case_common import TRACK_INIT


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


def test_load_pressure_eye_track_uses_final_model_eye_columns(tmp_path: Path):
    csv_path = tmp_path / "gdas_forecast_pressure_eye_positions.csv"
    csv_path.write_text(
        "scheme,scheme_label,lead_hour,valid_time,auto_model_eye_lon,auto_model_eye_lat,model_eye_lon,model_eye_lat,model_eye_msl_hpa,manual_override,error\n"
        "gdas_forecast,GDAS forecast,0,2025-07-17 00:00:00,124.0,14.0,127.9,14.1,999.3,True,\n"
        "gdas_forecast,GDAS forecast,3,2025-07-17 03:00:00,125.0,14.5,126.8,14.5,998.7,True,\n",
        encoding="utf-8",
    )

    track = track_forecast_error.load_pressure_eye_track("gdas_forecast", csv_path)

    assert track[["scheme", "lead_hour", "lon", "lat"]].to_dict("records") == [
        {"scheme": "gdas_forecast", "lead_hour": 0, "lon": 127.9, "lat": 14.1},
        {"scheme": "gdas_forecast", "lead_hour": 3, "lon": 126.8, "lat": 14.5},
    ]
    assert track["manual_override"].tolist() == [True, True]


def test_build_tracks_and_errors_loads_pressure_eye_paths(tmp_path: Path):
    gdas_csv = tmp_path / "gdas_forecast_pressure_eye_positions.csv"
    era5_csv = tmp_path / "era5_lagged_5d_pressure_eye_positions.csv"
    header = "scheme,scheme_label,lead_hour,valid_time,model_eye_lon,model_eye_lat,model_eye_msl_hpa,manual_override,error\n"
    gdas_csv.write_text(
        header
        + "gdas_forecast,GDAS forecast,0,2025-07-17 00:00:00,120.0,20.0,999.0,False,\n"
        + "gdas_forecast,GDAS forecast,3,2025-07-17 03:00:00,122.0,21.0,998.0,True,\n",
        encoding="utf-8",
    )
    era5_csv.write_text(
        header
        + "era5_lagged_5d,ERA5 lagged 5d forecast,0,2025-07-17 00:00:00,120.0,20.0,999.0,False,\n"
        + "era5_lagged_5d,ERA5 lagged 5d forecast,3,2025-07-17 03:00:00,121.0,21.0,998.0,False,\n",
        encoding="utf-8",
    )
    real_track = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime([TRACK_INIT, TRACK_INIT + pd.Timedelta(hours=3)]),
            "lon": [120.0, 121.0],
            "lat": [20.0, 21.0],
            "source": ["test", "test"],
        }
    )

    tracks, errors = track_forecast_error.build_tracks_and_errors_from_pressure_eye(
        real_track,
        track_paths={"gdas_forecast": gdas_csv, "era5_lagged_5d": era5_csv},
        write_outputs=False,
    )

    gdas_track = tracks[(tracks["scheme"] == "gdas_forecast") & (tracks["lead_hour"] == 3)].iloc[0]
    gdas_error = errors[(errors["scheme"] == "gdas_forecast") & (errors["lead_hour"] == 3)].iloc[0]
    era5_error = errors[(errors["scheme"] == "era5_lagged_5d") & (errors["lead_hour"] == 3)].iloc[0]

    assert gdas_track["lon"] == 122.0
    assert gdas_track["lat"] == 21.0
    assert gdas_error["track_error_km"] > 100
    assert era5_error["track_error_km"] == 0


def test_track_forecast_error_loads_local_wipha_csv_for_official_track(tmp_path: Path):
    csv_path = tmp_path / "typhoon_2506_Wipha.csv"
    csv_path.write_text(
        "tc_num,name_cn,name_en,dateUTC,dateCST,vmax,grade,latTC,lonTC,mslp,attr\n"
        "2506,韦帕,Wipha,202507170000,202507170800,15,热带低压,14.6,127.2,1000,analysis\n"
        "2506,韦帕,Wipha,202507170300,202507171100,15,热带低压,15.0,126.4,998,analysis\n",
        encoding="utf-8",
    )

    track = track_forecast_error.load_official_wipha_track(csv_path)

    assert track["source"].unique().tolist() == ["typhoon_2506_Wipha.csv"]
    assert track[["datetime_utc", "lon", "lat"]].to_dict("records") == [
        {"datetime_utc": pd.Timestamp("2025-07-17 00:00:00"), "lon": 127.2, "lat": 14.6},
        {"datetime_utc": pd.Timestamp("2025-07-17 03:00:00"), "lon": 126.4, "lat": 15.0},
    ]
