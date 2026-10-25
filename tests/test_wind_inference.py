import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import infer
import main as runner
from experiments import EXPERIMENTS, run_directory, wind_path
from wind_output import input_grid, write_wind_file


T0 = datetime(2025, 1, 1)
SHAPE = (3, 4)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "GRID_SHAPE", SHAPE)
    for experiment in EXPERIMENTS.values():
        directory = infer.input_directory(tmp_path, experiment, T0)
        directory.mkdir(parents=True, exist_ok=True)
        surface = np.arange(4 * np.prod(SHAPE), dtype=np.float32).reshape(4, *SHAPE)
        upper = np.ones((5, 13, *SHAPE), dtype=np.float32)
        np.save(directory / "input_surface.npy", surface)
        np.save(directory / "input_upper.npy", upper)
        with nc.Dataset(directory / "surface.nc", "w") as ds:
            ds.createDimension("latitude", SHAPE[0])
            ds.createDimension("longitude", SHAPE[1])
            ds.createVariable("latitude", "f4", ("latitude",))[:] = [90, 0, -90]
            ds.createVariable("longitude", "f4", ("longitude",))[:] = [0, 90, 180, 270]
    (tmp_path / "models").mkdir()
    for step in (1, 3, 6, 24):
        (tmp_path / "models" / f"pangu_weather_{step}.onnx").write_bytes(b"test model")
    return tmp_path


class Sessions:
    def __init__(self, fail_at=None):
        self.calls = []
        self.created = []
        self.fail_at = fail_at

    def __call__(self, path):
        step = int(path.stem.rsplit("_", 1)[1])
        self.created.append(step)
        owner = self

        class Session:
            def run(self, outputs, inputs):
                owner.calls.append(step)
                if len(owner.calls) == owner.fail_at:
                    raise RuntimeError("simulated interruption")
                # A nonlinear step-dependent surface update distinguishes 24h+6h
                # from ten consecutive 3h updates, catching integration changes.
                return inputs["input"] + step, inputs["input_surface"] + step**2

        return Session()


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_four_experiments_zero_time_wind_and_cache_cleanup(project, name):
    sessions = Sessions()
    outputs = infer.infer_experiment(T0, name, [0, 3, 6, 24, 30, 72], project_root=project, session_factory=sessions)
    experiment = EXPERIMENTS[name]
    source = infer.input_directory(project, experiment, T0)
    original = np.load(source / "input_surface.npy")
    for lead, path in zip([0, 3, 6, 24, 30, 72], outputs):
        with nc.Dataset(path) as ds:
            assert set(ds.variables) == {"latitude", "longitude", "valid_time", "u10", "v10"}
            increment = sum(step**2 for _, step in infer.integration_steps(experiment.model_hour(lead)))
            np.testing.assert_array_equal(ds["u10"][:], original[1] + increment)
            np.testing.assert_array_equal(ds["v10"][:], original[2] + increment)
            np.testing.assert_array_equal(ds["longitude"][:], [0, 90, 180, 270])
            assert ds.model_forecast_hour == experiment.lag_hours + lead
            assert ds.initial_time_utc == (T0 - timedelta(hours=experiment.lag_hours)).isoformat()
            assert ds.valid_time_utc == (T0 + timedelta(hours=lead)).isoformat()
            assert ds["u10"].filters()["zlib"]
    assert not (run_directory(project / "model_output/experiments", experiment, T0) / "timeline_cache").exists()
    assert not list((project / "model_output").rglob("*.npy"))
    np.testing.assert_array_equal(np.load(source / "input_surface.npy"), original)
    count = len(sessions.calls)
    infer.infer_experiment(T0, name, [0, 3, 6, 24, 30, 72], project_root=project, session_factory=sessions)
    assert len(sessions.calls) == count


def test_interruption_keeps_cache_and_resume_reuses_full_state(project):
    experiment = EXPERIMENTS["gdas_lagged_120h"]
    cache = run_directory(project / "model_output/experiments", experiment, T0) / "timeline_cache"
    interrupted = Sessions(fail_at=3)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        infer.infer_experiment(T0, experiment.name, [0, 3], project_root=project, session_factory=interrupted)
    assert cache.exists()
    assert (cache / "state_48.json").exists()
    resumed = Sessions()
    infer.infer_experiment(T0, experiment.name, [0, 3], project_root=project, session_factory=resumed)
    assert resumed.calls == [24, 24, 24, 3]
    assert not cache.exists()


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_default_leads_reuse_each_model_session_once(project, name):
    sessions = Sessions()
    outputs = infer.infer_experiment(T0, name, project_root=project, session_factory=sessions)
    assert len(outputs) == 25
    assert sessions.created == [24, 6, 3]
    assert len(sessions.calls) == (29 if EXPERIMENTS[name].lag_hours else 24)
    original = np.load(infer.input_directory(project, EXPERIMENTS[name], T0) / "input_surface.npy")
    for lead, path in zip(range(0, 73, 3), outputs):
        expected = sum(step**2 for _, step in infer.integration_steps(EXPERIMENTS[name].model_hour(lead)))
        with nc.Dataset(path) as ds:
            np.testing.assert_array_equal(ds["u10"][:], original[1] + expected)


def test_cache_identity_rejects_changed_input_without_deleting_cache(project):
    name = "gdas_direct"
    with pytest.raises(RuntimeError):
        infer.infer_experiment(T0, name, [6], project_root=project, session_factory=Sessions(fail_at=1))
    source = infer.input_directory(project, EXPERIMENTS[name], T0) / "input_surface.npy"
    data = np.load(source)
    data[1] += 100
    np.save(source, data)
    with pytest.raises(RuntimeError, match="changed"):
        infer.infer_experiment(T0, name, [6], project_root=project, session_factory=Sessions())
    assert list((project / "model_output").rglob("manifest.json"))
    infer.infer_experiment(T0, name, [6], project_root=project, overwrite=True, session_factory=Sessions())
    assert not list((project / "model_output").rglob("timeline_cache"))


def test_broken_output_is_rebuilt_and_legacy_data_is_untouched(project):
    legacy = project / "model_output/gdas/2025-01-01-00-00/timeline_cache"
    legacy.mkdir(parents=True)
    sentinel = legacy / "keep.txt"
    sentinel.write_text("legacy data")
    outputs = infer.infer_experiment(T0, "gdas_direct", [3], project_root=project, session_factory=Sessions())
    outputs[-1].write_bytes(b"interrupted NetCDF")
    sessions = Sessions()
    infer.infer_experiment(T0, "gdas_direct", [3], project_root=project, session_factory=sessions)
    assert sessions.calls == [3]
    assert sentinel.read_text() == "legacy data"


def test_write_failure_preserves_checkpoints(project, monkeypatch):
    real_writer = infer.write_wind_file

    def fail_positive(path, surface, latitude, longitude, metadata):
        if metadata["lead_hour"]:
            raise OSError("disk full")
        return real_writer(path, surface, latitude, longitude, metadata)

    monkeypatch.setattr(infer, "write_wind_file", fail_positive)
    with pytest.raises(OSError, match="disk full"):
        infer.infer_experiment(T0, "gdas_direct", [3], project_root=project, session_factory=Sessions())
    assert list((project / "model_output").rglob("state_3.json"))
    monkeypatch.setattr(infer, "write_wind_file", real_writer)
    sessions = Sessions()
    infer.infer_experiment(T0, "gdas_direct", [3], project_root=project, session_factory=sessions)
    assert sessions.calls == []


def test_grid_rejects_longitude_shift(project):
    source = infer.input_directory(project, EXPERIMENTS["gdas_direct"], T0)
    with nc.Dataset(source / "surface.nc", "r+") as ds:
        ds["longitude"][:] += 0.125
    with pytest.raises(ValueError, match="coordinates"):
        input_grid(source, SHAPE)


def test_npy_only_grid_has_no_longitude_shift(tmp_path):
    latitude, longitude = input_grid(tmp_path, (721, 1440))
    assert (latitude[0], latitude[-1]) == (90, -90)
    assert (longitude[0], longitude[-1]) == (0, 359.75)


def test_nonfinite_wind_does_not_replace_existing_file(tmp_path):
    path = tmp_path / "wind.nc"
    path.write_bytes(b"existing")
    surface = np.zeros((4, *SHAPE), dtype=np.float32)
    surface[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        write_wind_file(path, surface, np.array([90, 0, -90]), np.array([0, 90, 180, 270]), {})
    assert path.read_bytes() == b"existing"


def test_cli_dry_run_has_no_writes_and_reports_missing_inputs(project):
    arguments = ["--project-root", str(project), "--start-date", "2025-01-01", "--end-date", "2025-01-01", "--dry-run"]
    assert runner.main(arguments) == 0
    assert not (project / "model_output").exists()
    (infer.input_directory(project, EXPERIMENTS["gdas_direct"], T0) / "input_upper.npy").unlink()
    assert runner.main(arguments) == 1
    assert not (project / "model_output").exists()


def test_cli_continues_other_experiments_and_records_failures(project, monkeypatch):
    calls = []

    def fake_run(target, name, leads, **kwargs):
        calls.append(name)
        if name == "era5_direct":
            raise RuntimeError("test failure")
        return []

    monkeypatch.setattr(infer, "infer_experiment", fake_run)
    assert runner.main(["--project-root", str(project), "--start-date", "2025-01-01", "--end-date", "2025-01-01"]) == 1
    assert calls == list(EXPERIMENTS)
    report_path = next((project / "model_output/experiments/_runs").glob("*.json"))
    rows = json.loads(report_path.read_text(encoding="utf-8"))["tasks"]
    assert [row["status"] for row in rows] == ["complete", "complete", "failed", "complete"]


def test_default_year_and_requested_lead_validation():
    args = runner.build_parser().parse_args([])
    assert (args.start_date, args.end_date) == (datetime(2025, 1, 1), datetime(2025, 12, 31))
    assert args.lead_hours == list(range(0, 73, 3))
    assert infer.normalize_leads([6, 3, 6]) == [0, 3, 6]
    with pytest.raises(ValueError):
        infer.normalize_leads([-3])
