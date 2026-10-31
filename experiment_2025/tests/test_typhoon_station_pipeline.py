"""Check the station/NetCDF -> CSV -> figure boundary and time matching."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from experiment_2025.Statistics import typhoon_station_wind_timeseries as statistics
from experiment_2025.core.experiments import EXPERIMENTS, wind_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_SCRIPT = PROJECT_ROOT / "experiment_2025/plots/plot_typhoon_station_wind_timeseries.py"


def make_wind_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("latitude", 2)
        ds.createDimension("longitude", 2)
        ds.createVariable("latitude", "f8", ("latitude",))[:] = [20, 10]
        ds.createVariable("longitude", "f8", ("longitude",))[:] = [100, 120]
        ds.createVariable("u10", "f8", ("latitude", "longitude"))[:] = [[1, 3], [3, 5]]
        ds.createVariable("v10", "f8", ("latitude", "longitude"))[:] = 4


@pytest.mark.parametrize("mode", ["utc", "beijing", "both"])
def test_statistics_writes_matched_csv_and_summary(tmp_path, monkeypatch, mode):
    case = {"typhoon": "测试台风", "station": "测试站", "lon": 110.0, "lat": 15.0,
            "start": pd.Timestamp("2025-07-07").to_pydatetime(),
            "end": pd.Timestamp("2025-07-08").to_pydatetime()}
    monkeypatch.setattr(statistics, "CASES", (case,))
    raw_times = ["2025-07-07 00:00", "2025-07-07 07:00", "2025-07-07 08:00", "2025-07-08 00:00"]
    source = pd.DataFrame({"时间": raw_times, "站点": ["测试站"] * 4,
                           "经度": [110] * 4, "纬度": [15] * 4,
                           "风速": [5, 6, 7, 8], "风向": [350, 10, 20, 30]})
    station_file = tmp_path / "stations.csv"
    source.to_csv(station_file, index=False, encoding="utf-8-sig")
    output_root, data_root = tmp_path / "model", tmp_path / "data"
    expected_times = {
        "utc": ["2025-07-07 00:00", "2025-07-07 06:00", "2025-07-07 09:00", "2025-07-08 00:00"],
        "beijing": ["2025-07-06 15:00", "2025-07-07 00:00", "2025-07-07 00:00", "2025-07-07 15:00"],
    }
    missing_paths = set()
    for time_mode, stamps in expected_times.items():
        for stamp in stamps:
            valid = pd.Timestamp(stamp).to_pydatetime()
            target = valid.replace(hour=0)
            lead = valid.hour
            for name, experiment in EXPERIMENTS.items():
                path = wind_path(output_root, experiment, target, lead)
                if name == "gdas_lagged_120h" and stamp == stamps[-1]:
                    missing_paths.add(path)
                else:
                    make_wind_file(path)
    for path in missing_paths:
        if path.exists():
            path.unlink()

    assert statistics.main(["--station-file", str(station_file), "--output-root", str(output_root),
                            "--data-root", str(data_root), "--timezone-mode", mode]) == 0
    summary = json.loads((data_root / "typhoon_station_wind_timeseries_2025_summary.json").read_text(encoding="utf-8"))
    modes = ("utc", "beijing") if mode == "both" else (mode,)
    assert set(summary["modes"]) == set(modes)
    assert summary["initialization_policy"] == "daily_00_utc"
    for time_mode in modes:
        frame = pd.read_csv(data_root / f"typhoon_station_wind_timeseries_2025_{time_mode}.csv")
        assert frame["model_valid_time_utc"].tolist() == expected_times[time_mode]
        offsets = [0, -1, 1, 0] if time_mode == "utc" else [-1, 1, 0, -1]
        assert frame["match_offset_hours"].tolist() == offsets
        assert frame["plot_lead_hour"].tolist() == [0, 7, 8, 24]
        assert frame["obs_speed"].tolist() == [5, 6, 7, 8]
        np.testing.assert_allclose(frame["era5_direct_speed"], 5)
        np.testing.assert_allclose(frame["era5_direct_direction"], 216.86989764584402)
        assert frame["gdas_lagged_120h_speed"].isna().tolist() == [False, False, False, True]
        assert len(summary["modes"][time_mode]["missing_outputs"]) == 1
        assert summary["modes"][time_mode]["cases"][0]["record_count"] == 4
    assert (data_root / "typhoon_station_wind_timeseries_2025.csv").exists() == (mode != "beijing")
    assert not list(tmp_path.rglob("*.png"))


def run_without_imports(script, blocked, *arguments):
    code = """
import importlib.abc, runpy, sys
class RejectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in BLOCKED):
            raise RuntimeError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, RejectImports())
sys.argv = ARGV
runpy.run_path(SCRIPT, run_name='__main__')
"""
    code = (f"BLOCKED = {blocked!r}\nARGV = {[str(script), *map(str, arguments)]!r}\n"
            f"SCRIPT = {str(script)!r}\n" + code)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=60)


def test_statistics_entry_does_not_require_plotting():
    script = PROJECT_ROOT / "experiment_2025/Statistics/typhoon_station_wind_timeseries.py"
    result = run_without_imports(script, ["matplotlib"], "--help")
    assert result.returncode == 0, result.stderr
    assert "--station-file" in result.stdout


def plot_frame():
    # Two different typhoons at the same station must remain separate panels.
    rows = []
    for typhoon in ("台风甲", "台风乙"):
        for hour in (0, 3, 6):
            # Deliberately place the later case first in the CSV.
            raw = pd.Timestamp("2025-07-09" if typhoon == "台风甲" else "2025-07-07") + pd.Timedelta(hours=hour)
            row = {"typhoon": typhoon, "station": "同一站", "longitude": 110, "latitude": 15,
                   "station_time_raw": str(raw), "station_time_utc": str(raw - pd.Timedelta(hours=8)),
                   "plot_lead_hour": hour, "obs_speed": 5 + hour, "obs_direction": 350 if hour == 0 else 10}
            for name in statistics.EXPERIMENTS:
                row[f"{name}_speed"] = 5 + hour
                row[f"{name}_direction"] = 350 if hour == 0 else 10
            rows.append(row)
    return pd.DataFrame(rows)


def test_plot_reads_csv_without_statistics_or_netcdf(tmp_path):
    csv_path = tmp_path / "arbitrary_name.csv"
    plot_frame().to_csv(csv_path, index=False, encoding="utf-8-sig")
    figures = tmp_path / "figures"
    result = run_without_imports(PLOT_SCRIPT, ["netCDF4", "cartopy", "experiment_2025.Statistics"],
                                 "--plot-only", "--matched-csv", csv_path, "--figures-root", figures)
    assert result.returncode == 0, result.stderr
    assert "does not have a glyph" not in result.stderr
    stem = figures / "typhoon_station_wind_timeseries_beijing"
    for suffix in (".png", ".svg", ".pdf"):
        assert stem.with_suffix(suffix).stat().st_size > 1000
    svg = ET.parse(stem.with_suffix(".svg"))
    assert len([element for element in svg.iter() if element.get("id", "").startswith("axes_")]) == 4
    axes = {element.get("id"): element for element in svg.iter() if element.get("id", "").startswith("axes_")}
    assert "台风乙" in "".join(axes["axes_1"].itertext())
    assert "台风甲" in "".join(axes["axes_3"].itertext())
    assert "2025-07-06 16:00" in stem.with_suffix(".svg").read_text(encoding="utf-8")


def test_plot_rejects_wrong_timezone_before_creating_figures(tmp_path):
    from experiment_2025.plots import plot_typhoon_station_wind_timeseries as plot
    with pytest.raises(ValueError, match="differs from CSV"):
        plot.plot_figure(plot_frame(), "utc", tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_fixed_initialization_does_not_change_at_midnight():
    target = pd.Timestamp("2025-07-19 00:00").to_pydatetime()
    valid = pd.Timestamp("2025-07-21 03:00").to_pydatetime()
    actual_target, lead = statistics._target_and_lead(valid, target)
    assert actual_target == target
    assert lead == 51


def test_selection_preserves_values_and_detects_old_initialization():
    from experiment_2025.Statistics.select_typhoon_station_cases import select_cases
    records = []
    for case_number, (name, station, stamp) in enumerate([
        ("韦帕", "遮浪", "2025-07-18"),
        ("范斯高", "大陈", "2025-07-24"),
        ("杨柳", "北礵", "2025-08-12"),
    ]):
        target = pd.Timestamp(stamp)
        for hour in [0, 3, 72]:
            valid = target + pd.Timedelta(hours=hour)
            records.append({"typhoon": name, "station": station,
                            "station_time_raw": str(valid + pd.Timedelta(hours=8)),
                            "station_time_utc": str(valid), "model_valid_time_utc": str(valid),
                            "target_time_utc": str(target), "plot_lead_hour": hour,
                            "model_lead_hour": hour, "match_offset_hours": 0,
                            "obs_speed": case_number + hour / 10})
    frame = pd.DataFrame(records)
    result, info = select_cases(frame.iloc[:3], frame.iloc[3:], require_fixed=True)
    assert result["station"].drop_duplicates().tolist() == ["遮浪", "大陈", "北礵"]
    assert result["obs_speed"].tolist() == frame["obs_speed"].tolist()
    assert info["all_fixed_in_recorded_metadata"]
    old = frame.iloc[:3].copy()
    old.loc[2, "target_time_utc"] = old.loc[2, "station_time_utc"]
    old.loc[2, "model_lead_hour"] = 0
    with pytest.raises(ValueError, match="韦帕.*fixed initialization"):
        select_cases(old, frame.iloc[3:], require_fixed=True)
    preview, info = select_cases(old, frame.iloc[3:])
    assert not info["all_fixed_in_recorded_metadata"]
    assert preview["obs_speed"].tolist() == frame["obs_speed"].tolist()


def test_plot_marks_unverified_initialization_as_preview():
    from experiment_2025.plots.plot_typhoon_station_wind_timeseries import _initialization_label
    frame = pd.DataFrame({"target_time_utc": ["2025-07-18 00:00", "2025-07-19 00:00"],
                          "expected_target_time_utc": ["2025-07-18 00:00"] * 2})
    assert "预览" in _initialization_label(frame)
    frame["target_time_utc"] = "2025-07-18 00:00"
    assert "预览" not in _initialization_label(frame)
    assert "2025-07-18 00:00" in _initialization_label(frame)
