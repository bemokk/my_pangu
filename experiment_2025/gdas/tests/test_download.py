"""HTTP edge cases use deterministic responses and structurally valid GRIB2."""

from datetime import datetime, timezone, timedelta

import pytest

from experiment_2025.gdas import download
from experiment_2025.gdas import cli


PAYLOAD = b"GRIB" + b"\x00\x00\x00\x02" + (28).to_bytes(8, "big") + b"abcdefgh7777"


class Response:
    def __init__(self, payload=b"", status=200, headers=None, fail=False):
        self.payload = payload
        self.status_code = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(payload))}
        self.fail = fail
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.payload
        if self.fail:
            raise IOError("connection lost")

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses, size=len(PAYLOAD)):
        self.responses = iter(responses)
        self.size = size
        self.requests = []

    def head(self, url, **kwargs):
        return Response(headers={} if self.size is None else {"Content-Length": str(self.size)})

    def get(self, url, headers, **kwargs):
        self.requests.append(headers)
        return next(self.responses)


@pytest.mark.parametrize("remote_total", ["28", "*"])
def test_resume_and_skip(tmp_path, remote_total):
    path = tmp_path / "data.grib2"
    path.with_suffix(".grib2.part").write_bytes(PAYLOAD[:10])
    response = Response(PAYLOAD[10:], 206, {"Content-Length": "18", "Content-Range": f"bytes 10-27/{remote_total}"})
    session = Session([response])
    result = download.download_file(session, "url", path, attempts=1)
    assert not result.skipped and path.read_bytes() == PAYLOAD
    assert session.requests == [{"Range": "bytes=10-"}] and response.closed
    assert download.download_file(session, "url", path).skipped


def test_ignored_range_restarts(tmp_path):
    path = tmp_path / "data.grib2"
    path.with_suffix(".grib2.part").write_bytes(PAYLOAD[:10])
    download.download_file(Session([Response(PAYLOAD)]), "url", path, attempts=1)
    assert path.read_bytes() == PAYLOAD


def test_bounded_ranges_complete_large_download(tmp_path):
    path = tmp_path / "data.grib2"
    session = Session([
        Response(PAYLOAD[:10], 206, {"Content-Length": "10", "Content-Range": "bytes 0-9/28"}),
        Response(PAYLOAD[10:20], 206, {"Content-Length": "10", "Content-Range": "bytes 10-19/*"}),
        Response(PAYLOAD[20:], 206, {"Content-Length": "8", "Content-Range": "bytes 20-27/28"}),
    ])
    download.download_file(session, "url", path, attempts=1, range_chunk_bytes=10)
    assert path.read_bytes() == PAYLOAD
    assert session.requests == [{"Range": "bytes=0-9"}, {"Range": "bytes=10-19"}, {"Range": "bytes=20-"}]


def test_ignored_bounded_range_accepts_full_response(tmp_path):
    path = tmp_path / "data.grib2"
    download.download_file(Session([Response(PAYLOAD)]), "url", path, attempts=1, range_chunk_bytes=10)
    assert path.read_bytes() == PAYLOAD


def test_wrong_bounded_range_rejected(tmp_path):
    path = tmp_path / "data.grib2"
    response = Response(PAYLOAD[:20], 206, {"Content-Length": "20", "Content-Range": "bytes 0-19/28"})
    with pytest.raises(RuntimeError, match="segment"):
        download.download_file(Session([response]), "url", path, attempts=1, range_chunk_bytes=10)
    assert not path.exists()


def test_interrupted_segments_shrink_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    unit = 1024 * 1024
    length = 2 * unit + 20
    payload = b"GRIB" + b"\x00\x00\x00\x02" + length.to_bytes(8, "big") + b"x" * (length - 20) + b"7777"
    session = Session([
        Response(b"", 206, {"Content-Length": str(2 * unit), "Content-Range": f"bytes 0-{2 * unit - 1}/{length}"}, fail=True),
        Response(payload[:unit], 206, {"Content-Length": str(unit), "Content-Range": f"bytes 0-{unit - 1}/{length}"}),
        Response(payload[unit:2 * unit], 206, {"Content-Length": str(unit), "Content-Range": f"bytes {unit}-{2 * unit - 1}/{length}"}),
        Response(payload[2 * unit:], 206, {"Content-Length": "20", "Content-Range": f"bytes {2 * unit}-{length - 1}/{length}"}),
    ], size=length)
    path = tmp_path / "data.grib2"
    download.download_file(session, "url", path, attempts=2, range_chunk_bytes=2 * unit)
    assert path.read_bytes() == payload
    assert session.requests[:2] == [{"Range": f"bytes=0-{2 * unit - 1}"}, {"Range": f"bytes=0-{unit - 1}"}]


def test_stream_interruption_resumes_next_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    path = tmp_path / "data.grib2"
    session = Session([
        Response(PAYLOAD[:10], headers={"Content-Length": "28"}, fail=True),
        Response(PAYLOAD[10:], 206, {"Content-Length": "18", "Content-Range": "bytes 10-27/28"}),
    ])
    download.download_file(session, "url", path, attempts=2)
    assert path.read_bytes() == PAYLOAD
    assert session.requests == [{}, {"Range": "bytes=10-"}]


@pytest.mark.parametrize("content_range", ["", "bytes 0-17/28", "bytes 10-27/99"])
def test_invalid_range_does_not_append(tmp_path, content_range):
    path = tmp_path / "data.grib2"
    part = path.with_suffix(".grib2.part")
    part.write_bytes(PAYLOAD[:10])
    response = Response(PAYLOAD[10:], 206, {"Content-Length": "18", "Content-Range": content_range})
    with pytest.raises(RuntimeError, match="failed after"):
        download.download_file(Session([response]), "url", path, attempts=1)
    assert not path.exists() and part.read_bytes() == PAYLOAD[:10]


def test_short_body_without_head_length_not_published(tmp_path):
    path = tmp_path / "data.grib2"
    response = Response(PAYLOAD[:10], headers={"Content-Length": "28"})
    with pytest.raises(RuntimeError, match="incomplete"):
        download.download_file(Session([response], size=None), "url", path, attempts=1)
    assert not path.exists()


def test_html_body_not_published(tmp_path):
    path = tmp_path / "data.grib2"
    payload = b"<html>login page</html>"
    with pytest.raises(RuntimeError, match="GRIB2"):
        download.download_file(Session([Response(payload)], size=len(payload)), "url", path, attempts=1)
    assert not path.exists()


def test_completed_and_oversized_partial(tmp_path):
    path = tmp_path / "data.grib2"
    part = path.with_suffix(".grib2.part")
    part.write_bytes(PAYLOAD)
    assert download.download_file(Session([]), "url", path).skipped
    path.unlink()
    part.write_bytes(PAYLOAD * 2)
    session = Session([Response(PAYLOAD)])
    download.download_file(session, "url", path, attempts=1)
    assert path.read_bytes() == PAYLOAD and session.requests == [{}]


def test_unavailable_is_not_retried(tmp_path):
    with pytest.raises(download.GDASCycleUnavailable):
        download.download_file(Session([Response(status=404)]), "url", tmp_path / "missing", attempts=1)


def test_time_validation_and_utc_conversion():
    start = datetime(2025, 1, 1)
    assert len(list(download.iter_cycles(start, start + timedelta(days=1)))) == 5
    aware = datetime(2025, 1, 1, 8, tzinfo=timezone(timedelta(hours=8)))
    assert list(download.iter_cycles(aware, aware)) == [start]
    assert cli._parse_time("2025-01-01T08:00+08:00") == start
    for a, b, step in ((start, start - timedelta(days=1), 6), (start, start, 1),
                       (start.replace(hour=1), start.replace(hour=1), 6),
                       (start.replace(microsecond=1), start, 6)):
        with pytest.raises(ValueError):
            list(download.iter_cycles(a, b, step))
    with pytest.raises(ValueError):
        download.parse_init_time("gdas1.fnl0p25.2025010100.f06.grib2")


def test_cli_download_does_not_convert(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "download_range", lambda *args, **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(cli, "download_and_prepare", lambda *args, **kwargs: pytest.fail("conversion called"))
    assert cli.main(["download", "--start", "2025-01-01", "--end", "2025-01-01", "--raw-root", str(tmp_path)]) == 0
    assert calls[0]["raw_root"] == tmp_path


def test_continue_on_error_attempts_remaining_cycles(tmp_path, monkeypatch):
    from experiment_2025.gdas import pipeline

    calls = []
    def convert(path, *args, **kwargs):
        calls.append(path)
        raise ValueError("missing field")
    monkeypatch.setattr(pipeline, "convert_grib", convert)
    with pytest.raises(RuntimeError, match="2 GDAS cycles failed"):
        pipeline.process_existing(datetime(2025, 1, 1), datetime(2025, 1, 2), interval_hours=24,
                                  raw_root=tmp_path, continue_on_error=True)
    assert len(calls) == 2
