import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from experiment_2025.core import input_data
from experiment_2025.gdas import cli, constants, download, pipeline
from experiment_2025.gdas.paths import raw_path, prepared_directory


PAYLOAD = b"GRIB" + b"\x00\x00\x00\x02" + (28).to_bytes(8, "big") + b"abcdefgh7777"


@pytest.mark.parametrize("stamp", [datetime(2024, 12, 27), datetime(2025, 1, 1)])
def test_download_adopts_flat_valid_file_into_year_without_get(tmp_path, stamp):
    legacy = tmp_path / download.build_filename(stamp)
    legacy.write_bytes(PAYLOAD)

    class Response:
        status_code = 200
        headers = {"Content-Length": str(len(PAYLOAD))}

        def raise_for_status(self):
            pass

        def close(self):
            pass

    class Session:
        def head(self, *args, **kwargs):
            return Response()

        def get(self, *args, **kwargs):
            pytest.fail("complete local GRIB2 must not be downloaded again")

    result = download.download_cycle(stamp, tmp_path, session=Session())
    assert result.path == raw_path(tmp_path, stamp)
    assert result.skipped and result.path.read_bytes() == PAYLOAD
    assert not legacy.exists()


def test_new_input_reader_uses_year_and_still_supports_flat(tmp_path, monkeypatch):
    monkeypatch.setattr(input_data, "GRID_SHAPE", (3, 4))
    stamp = datetime(2025, 1, 1)
    year_dir = prepared_directory(tmp_path, stamp)
    year_dir.mkdir(parents=True)
    np.save(year_dir / "input_upper.npy", np.ones((5, 13, 3, 4), dtype=np.float32))
    np.save(year_dir / "input_surface.npy", np.ones((4, 3, 4), dtype=np.float32))
    reader = input_data.NpyDirectorySource(tmp_path)
    assert reader.load(stamp).source_files[0].parent == year_dir
    flat = tmp_path / stamp.strftime("%Y-%m-%d-%H-%M")
    year_dir.rename(flat)
    assert reader.load(stamp).source_files[0].parent == flat


def test_default_paths_and_daily_interval(monkeypatch):
    monkeypatch.delenv("PANGU_GDAS_RAW_ROOT", raising=False)
    assert constants.default_raw_root() == constants.GDAS_ROOT / "raw"
    args = cli.build_parser().parse_args(["--start", "2024-12-27", "--end", "2025-12-31"])
    assert args.interval_hours == 24


def test_plan_has_no_network_or_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "download_cycle", lambda *a, **k: pytest.fail("network call"))
    monkeypatch.setattr(pipeline, "convert_grib", lambda *a, **k: pytest.fail("conversion"))
    assert cli.main(["run", "--start", "2024-12-27", "--end", "2025-12-31",
                     "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "inputs"),
                     "--dry-run"]) == 0
    assert not list(tmp_path.iterdir())
    text = capsys.readouterr().out
    assert "370 cycles" in text
    assert "2024" in text and "2025" in text


def test_batch_records_success_and_failure_and_retries_missing_later(tmp_path, monkeypatch):
    calls = []

    def convert(path, root, **kwargs):
        calls.append(path)
        if "20250101" in path.name:
            raise ValueError("missing field")
        return prepared_directory(root, datetime(2025, 1, 2))

    monkeypatch.setattr(pipeline, "convert_grib", convert)
    with pytest.raises(RuntimeError, match="1 GDAS cycles failed"):
        pipeline.process_existing(datetime(2025, 1, 1), datetime(2025, 1, 2), raw_root=tmp_path / "raw",
                                  output_root=tmp_path / "inputs", continue_on_error=True)
    report = json.loads(next((tmp_path / "inputs/_runs").glob("*.json")).read_text(encoding="utf-8"))
    assert report["completed"] == 1 and report["failed"] == 1
    assert [row["status"] for row in report["tasks"]] == ["failed", "complete"]
    assert calls[0] == raw_path(tmp_path / "raw", datetime(2025, 1, 1))


def test_stop_on_error_still_saves_report(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("bad source")

    monkeypatch.setattr(pipeline, "convert_grib", fail)
    with pytest.raises(OSError):
        pipeline.process_existing(datetime(2025, 1, 1), datetime(2025, 1, 2), raw_root=tmp_path / "raw",
                                  output_root=tmp_path / "inputs")
    report = json.loads(next((tmp_path / "inputs/_runs").glob("*.json")).read_text(encoding="utf-8"))
    assert len(report["tasks"]) == 1 and report["failed"] == 1

