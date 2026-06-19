from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wind_wave.convert_grib import (
    ConvertedYearPair,
    build_year_output_pairs,
    convert_grib_to_yearly_netcdf,
    discover_grib_files,
    parse_years,
)


def test_parse_years_accepts_single_range_and_list():
    assert parse_years("2016:2018") == (2016, 2017, 2018)
    assert parse_years("2016,2018,2020") == (2016, 2018, 2020)


def test_parse_years_rejects_descending_range():
    with pytest.raises(ValueError, match="ascending"):
        parse_years("2024:2016")


def test_discover_grib_files_accepts_grib_and_grib2_suffixes(tmp_path):
    (tmp_path / "a.grib").write_bytes(b"")
    (tmp_path / "b.grib2").write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("x")

    assert discover_grib_files(tmp_path) == [tmp_path / "a.grib", tmp_path / "b.grib2"]


def test_build_year_output_pairs_uses_wind_and_wave_names(tmp_path):
    pairs = build_year_output_pairs(tmp_path, (2016, 2017))

    assert pairs == [
        ConvertedYearPair(year=2016, wind_nc=tmp_path / "2016" / "wind.nc", wave_nc=tmp_path / "2016" / "wave.nc"),
        ConvertedYearPair(year=2017, wind_nc=tmp_path / "2017" / "wind.nc", wave_nc=tmp_path / "2017" / "wave.nc"),
    ]


def test_convert_grib_to_yearly_netcdf_writes_requested_years(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    grib_path = raw_dir / "source.grib"
    grib_path.write_bytes(b"fake")
    output_dir = tmp_path / "converted"

    def fake_opener(path: Path, short_name: str) -> xr.Dataset:
        assert path == grib_path
        times = pd.to_datetime(["2016-01-01T00:00", "2017-01-01T00:00"])
        values = np.full((2, 2, 2), len(short_name), dtype=np.float32)
        return xr.Dataset(
            {short_name: (("time", "latitude", "longitude"), values)},
            coords={
                "time": times,
                "latitude": np.array([6.0, 5.0]),
                "longitude": np.array([95.0, 96.0]),
            },
        )

    pairs = convert_grib_to_yearly_netcdf(
        raw_dir=raw_dir,
        output_root=output_dir,
        years=(2016, 2017),
        region_text="5,6,95,96",
        opener=fake_opener,
    )

    assert [pair.year for pair in pairs] == [2016, 2017]
    with xr.open_dataset(output_dir / "2016" / "wind.nc", engine="netcdf4") as wind:
        assert set(wind.data_vars) == {"u10", "v10"}
        assert wind.sizes["time"] == 1
    with xr.open_dataset(output_dir / "2017" / "wave.nc", engine="netcdf4") as wave:
        assert set(wave.data_vars) == {"swh", "mwp", "mwd"}
        assert wave.sizes["time"] == 1


def test_convert_grib_skips_files_without_requested_short_name(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    first = raw_dir / "wind.grib"
    second = raw_dir / "wave.grib"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    output_dir = tmp_path / "converted"

    def fake_opener(path: Path, short_name: str) -> xr.Dataset:
        if path == second and short_name in {"u10", "v10"}:
            raise KeyError(short_name)
        times = pd.to_datetime(["2016-01-01T00:00"])
        values = np.full((1, 1, 1), len(short_name), dtype=np.float32)
        return xr.Dataset(
            {short_name: (("time", "latitude", "longitude"), values)},
            coords={"time": times, "latitude": [5.0], "longitude": [95.0]},
        )

    convert_grib_to_yearly_netcdf(
        raw_dir=raw_dir,
        output_root=output_dir,
        years=(2016,),
        opener=fake_opener,
    )

    with xr.open_dataset(output_dir / "2016" / "wind.nc", engine="netcdf4") as wind:
        assert set(wind.data_vars) == {"u10", "v10"}
