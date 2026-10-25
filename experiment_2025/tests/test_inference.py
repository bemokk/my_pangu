"""Small physical files exercise the archive reader and full run lifecycle."""

from contextlib import nullcontext
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np
import pytest

from experiment_2025.core import input_data, inference
from experiment_2025.core.experiments import EXPERIMENTS, run_directory, time_label
from experiment_2025.cli.main import build_parser, main


T0 = datetime(2025, 1, 1)
SHAPE = (3, 4)
LEVELS = input_data.PRESSURE_LEVELS


def make_archive_file(path, times, upper=False, missing_value=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("valid_time", len(times))
        ds.createDimension("latitude", SHAPE[0])
        ds.createDimension("longitude", SHAPE[1])
        ds.createVariable("latitude", "f4", ("latitude",))[:] = [90, 0, -90]
        ds.createVariable("longitude", "f4", ("longitude",))[:] = np.arange(SHAPE[1]) * 0.25
        time = ds.createVariable("valid_time", "f8", ("valid_time",))
        time.units = "hours since 2024-01-01"
        time[:] = nc.date2num(times, time.units)
        dims = ("valid_time", "latitude", "longitude")
        names = input_data.SURFACE_ORDER
        if upper:
            ds.createDimension("pressure_level", 13)
            ds.createVariable("pressure_level", "f8", ("pressure_level",))[:] = LEVELS
            dims = ("valid_time", "pressure_level", "latitude", "longitude")
            names = input_data.UPPER_ORDER
        for channel, name in enumerate(names):
            variable = ds.createVariable(name, "f4", dims, fill_value=-9999.0)
            variable[:] = channel + np.arange(len(times), dtype=np.float32).reshape((-1,) + (1,) * (len(dims) - 1)) * 10
            if name == "q":
                variable[0, 0, 0, 0] = -0.00001
            if missing_value and channel == 0:
                variable[0, ...] = -9999.0


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(input_data, "GRID_SHAPE", SHAPE)
    monkeypatch.setattr(inference, "GRID_SHAPE", SHAPE)
    era5 = tmp_path / "era5"
    gdas = tmp_path / "gdas"
    for stamp in (T0, T0 - timedelta(hours=120)):
        make_archive_file(era5 / str(stamp.year) / "surface/surface.nc", [stamp])
        make_archive_file(era5 / str(stamp.year) / "upper/quarter.nc", [stamp], upper=True)
        directory = gdas / time_label(stamp)
        directory.mkdir(parents=True)
        np.save(directory / "input_surface.npy", np.arange(4 * np.prod(SHAPE), dtype=np.float32).reshape(4, *SHAPE))
        np.save(directory / "input_upper.npy", np.ones((5, 13, *SHAPE), dtype=np.float32))
    models = tmp_path / "models"
    models.mkdir()
    for step in (1, 3, 6, 24):
        (models / f"pangu_weather_{step}.onnx").write_bytes(b"test model")
    return {"era5_root": era5, "gdas_root": gdas, "output_root": tmp_path / "output", "model_root": models}


class Sessions:
    def __init__(self, fail_at=None):
        self.steps = []
        self.created = []
        self.fail_at = fail_at

    def __call__(self, model):
        step = int(model.stem.rsplit("_", 1)[1])
        self.created.append(step)
        owner = self

        class Session:
            def run(self, _, inputs):
                owner.steps.append(step)
                if owner.fail_at == len(owner.steps):
                    raise RuntimeError("simulated interruption")
                return inputs["input"] + step, inputs["input_surface"] + step**2

        return Session()


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_four_experiments_preserve_greedy_paths_and_copy_zero(paths, name):
    sessions = Sessions()
    with pytest.warns(RuntimeWarning) if name.startswith("era5") else nullcontext():
        outputs = inference.infer_experiment(T0, name, **paths, session_factory=sessions)
    experiment = EXPERIMENTS[name]
    source = input_data.source_for_experiment(experiment, era5_root=paths["era5_root"], gdas_root=paths["gdas_root"])
    with nullcontext():
        initial = source.load(experiment.initial_time(T0))
    assert len(outputs) == 25
    assert sessions.created == [24, 6, 3]
    assert len(sessions.steps) == (29 if experiment.lag_hours else 24)
    for lead, path in zip(range(0, 73, 3), outputs):
        with nc.Dataset(path) as ds:
            assert set(ds.variables) == {"u10", "v10", "latitude", "longitude", "valid_time"}
            delta = sum(step**2 for _, step in inference.integration_steps(experiment.model_hour(lead)))
            np.testing.assert_array_equal(ds["u10"][:], initial.surface[1] + delta)
            np.testing.assert_array_equal(ds["v10"][:], initial.surface[2] + delta)
            assert ds.model_forecast_hour == lead + experiment.lag_hours
            assert ds.initial_time_utc == experiment.initial_time(T0).isoformat()
            assert ds.valid_time_utc == (T0 + timedelta(hours=lead)).isoformat()
            assert ds["u10"].filters()["zlib"]
    assert not list(paths["output_root"].rglob("timeline_cache"))
    assert not list(paths["output_root"].rglob("*.npy"))
    with nullcontext():
        inference.infer_experiment(T0, name, **paths, session_factory=sessions)
    assert len(sessions.steps) == (29 if experiment.lag_hours else 24)


def test_interruption_resume_and_cleanup(paths):
    name = "gdas_lagged_120h"
    with pytest.raises(RuntimeError, match="interruption"):
        inference.infer_experiment(T0, name, [0, 3], **paths, session_factory=Sessions(fail_at=3))
    cache = run_directory(paths["output_root"], EXPERIMENTS[name], T0) / "timeline_cache"
    assert (cache / "state_0048h.json").exists()
    sessions = Sessions()
    inference.infer_experiment(T0, name, [0, 3], **paths, session_factory=sessions)
    assert sessions.steps == [24, 24, 24, 3]
    assert not cache.exists()


def test_output_failure_preserves_state_and_resume_avoids_model(paths, monkeypatch):
    original_writer = inference.write_wind_file

    def writer(path, surface, latitude, longitude, metadata):
        if metadata["lead_hour"]:
            raise OSError("disk full")
        original_writer(path, surface, latitude, longitude, metadata)

    monkeypatch.setattr(inference, "write_wind_file", writer)
    with pytest.raises(OSError, match="disk full"):
        inference.infer_experiment(T0, "gdas_direct", [3], **paths, session_factory=Sessions())
    assert list(paths["output_root"].rglob("state_0003h.json"))
    monkeypatch.setattr(inference, "write_wind_file", original_writer)
    sessions = Sessions()
    inference.infer_experiment(T0, "gdas_direct", [3], **paths, session_factory=sessions)
    assert not sessions.steps


def test_input_changes_reject_cache_and_overwrite_rebuilds(paths):
    with pytest.raises(RuntimeError):
        inference.infer_experiment(T0, "gdas_direct", [3], **paths, session_factory=Sessions(fail_at=1))
    path = paths["gdas_root"] / time_label(T0) / "input_surface.npy"
    np.save(path, np.load(path) + 1)
    with pytest.raises(RuntimeError, match="identity changed"):
        inference.infer_experiment(T0, "gdas_direct", [3], **paths, session_factory=Sessions())
    inference.infer_experiment(T0, "gdas_direct", [3], **paths, overwrite=True, session_factory=Sessions())
    assert not list(paths["output_root"].rglob("timeline_cache"))


def test_archive_preserves_negative_q_and_rejects_missing_time(paths):
    archive = input_data.Era5Archive(paths["era5_root"])
    with pytest.warns(RuntimeWarning, match="negative"):
        state = archive.load(T0)
    assert state.upper[1, 0, 0, 0] == np.float32(-0.00001)
    assert state.negative_specific_humidity_count == 1
    with pytest.raises(FileNotFoundError, match="exact time"):
        archive.load(T0 + timedelta(days=1))


def test_archive_rejects_duplicate_times_and_missing_values(paths):
    extra = paths["era5_root"] / "2025/upper/duplicate.nc"
    make_archive_file(extra, [T0], upper=True)
    with pytest.raises(ValueError, match="Duplicate"):
        input_data.Era5Archive(paths["era5_root"]).available_times(2025)
    extra.unlink()
    make_archive_file(paths["era5_root"] / "2025/surface/surface.nc", [T0], missing_value=True)
    with pytest.raises(ValueError, match="Missing values"):
        input_data.Era5Archive(paths["era5_root"]).load(T0)


def test_grid_and_pressure_levels_are_validated(paths):
    upper = paths["era5_root"] / "2025/upper/quarter.nc"
    with nc.Dataset(upper, "r+") as ds:
        ds["pressure_level"][0] = 1000.1
    with pytest.raises(ValueError, match="pressure levels"):
        input_data.Era5Archive(paths["era5_root"]).load(T0)
    with nc.Dataset(upper, "r+") as ds:
        ds["pressure_level"][0] = 1000
        ds["longitude"][:] += 0.125
    with pytest.raises(ValueError, match="Grid mismatch"):
        input_data.Era5Archive(paths["era5_root"]).load(T0)


def test_cli_dry_run_checks_gdas_and_era5_without_writes(paths):
    arguments = ["--start-date", "2025-01-01", "--end-date", "2025-01-01"]
    for key, value in paths.items():
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
    assert main(arguments + ["--dry-run"]) == 0
    assert not paths["output_root"].exists()
    (paths["gdas_root"] / time_label(T0) / "input_upper.npy").unlink()
    assert main(arguments + ["--dry-run"]) == 1
    assert not paths["output_root"].exists()


def test_defaults_and_invalid_leads():
    args = build_parser().parse_args([])
    assert args.start_date == datetime(2025, 1, 1)
    assert args.end_date == datetime(2025, 12, 31)
    assert args.lead_hours == list(range(0, 73, 3))
    assert inference.normalize_leads([6, 3, 6]) == [0, 3, 6]
    with pytest.raises(ValueError):
        inference.normalize_leads([-1])


def test_corrupt_output_rebuild_and_one_hour_paths(paths):
    sessions = Sessions()
    outputs = inference.infer_experiment(T0, "gdas_direct", [1, 4, 8], **paths, session_factory=sessions)
    assert len(outputs) == 4
    initial = np.load(paths["gdas_root"] / time_label(T0) / "input_surface.npy")
    for lead, path in zip([0, 1, 4, 8], outputs):
        with nc.Dataset(path) as ds:
            expected = sum(step**2 for _, step in inference.integration_steps(lead))
            np.testing.assert_array_equal(ds["v10"][:], initial[2] + expected)
    outputs[-1].write_bytes(b"truncated")
    sessions = Sessions()
    inference.infer_experiment(T0, "gdas_direct", [1, 4, 8], **paths, session_factory=sessions)
    assert sessions.steps == [6, 1, 1]


def test_overwrite_discards_partial_checkpoints(paths):
    with pytest.raises(RuntimeError):
        inference.infer_experiment(T0, "gdas_lagged_120h", [0, 3], **paths, session_factory=Sessions(fail_at=3))
    sessions = Sessions()
    inference.infer_experiment(T0, "gdas_lagged_120h", [0, 3], **paths, overwrite=True, session_factory=sessions)
    assert sessions.steps == [24, 24, 24, 24, 24, 3]


def test_120_hour_schedule_records_timings_and_preserves_values(paths):
    timings = {}
    sessions = Sessions()
    outputs = inference.infer_experiment(T0, "gdas_direct", range(0, 121, 3),
                                         **paths, session_factory=sessions, timings=timings)
    assert len(outputs) == 41
    assert timings["model_steps"] == 40
    assert timings["outputs_written"] == 41 and timings["outputs_reused"] == 0
    assert sessions.created == [24, 6, 3]
    assert timings["total_seconds"] >= sum(timings[key] for key in (
        "input_seconds", "model_load_seconds", "inference_seconds", "cache_seconds", "output_seconds"))
    original = np.load(paths["gdas_root"] / time_label(T0) / "input_surface.npy")
    for lead, path in zip(range(0, 121, 3), outputs):
        with nc.Dataset(path) as ds:
            expected = sum(step**2 for _, step in inference.integration_steps(lead))
            np.testing.assert_array_equal(ds["u10"][:], original[1] + expected)
    assert not list(paths["output_root"].rglob("timeline_cache"))


def test_main_writes_task_and_batch_timings(paths, monkeypatch):
    import json
    from experiment_2025.cli import main as entry

    def fake_inference(target, name, leads, **kwargs):
        kwargs["timings"].update(inference_seconds=0.1, outputs_written=41)
        return []

    monkeypatch.setattr(entry, "infer_experiment", fake_inference)
    assert main(["--start-date", "2025-01-01", "--end-date", "2025-01-01", "--experiments", "era5_direct",
                 "--output-root", str(paths["output_root"])]) == 0
    report = json.loads(next((paths["output_root"] / "_runs").glob("*.json")).read_text(encoding="utf-8"))
    assert report["tasks"][0]["timings"]["inference_seconds"] == 0.1
    assert report["elapsed_seconds"] >= report["tasks"][0]["elapsed_seconds"] >= 0


