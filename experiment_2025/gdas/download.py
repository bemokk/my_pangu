"""Reliable, resumable downloads of GDAS/FNL GRIB2 cycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import time
from typing import Iterable

from .constants import BASE_URL, SUPPORTED_CYCLES


class GDASCycleUnavailable(RuntimeError):
    """The requested cycle is not currently published by the remote archive."""


@dataclass(frozen=True)
class DownloadResult:
    init_time: datetime | None
    url: str
    path: Path
    skipped: bool = False


def normalize_time(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("GDAS times must be datetime instances")
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    if value.microsecond:
        raise ValueError("GDAS cycle times must not include microseconds")
    return value


def iter_cycles(start: datetime, end: datetime, interval_hours: int = 6) -> Iterable[datetime]:
    start = normalize_time(start)
    end = normalize_time(end)
    if end < start:
        raise ValueError("end must not be earlier than start")
    if not isinstance(interval_hours, int) or interval_hours < 1 or interval_hours % 6:
        raise ValueError("GDAS interval_hours must be a positive multiple of 6")
    if start.minute or start.second or end.minute or end.second:
        raise ValueError("GDAS cycle times must be on an exact hour")
    if start.hour not in SUPPORTED_CYCLES or end.hour not in SUPPORTED_CYCLES:
        raise ValueError(f"GDAS cycles are limited to UTC hours {SUPPORTED_CYCLES}")
    current = start
    while current <= end:
        if current.hour not in SUPPORTED_CYCLES:
            raise ValueError(f"interval produced unsupported cycle {current:%Y-%m-%d %H:00} UTC")
        yield current
        current += timedelta(hours=interval_hours)


def build_filename(init_time: datetime, forecast_hour: int = 0) -> str:
    if forecast_hour != 0:
        raise ValueError("Only GDAS/FNL analysis files (f00) are supported")
    stamp = normalize_time(init_time).strftime("%Y%m%d%H")
    return f"gdas1.fnl0p25.{stamp}.f{int(forecast_hour):02d}.grib2"


def build_url(init_time: datetime, *, base_url: str = BASE_URL, forecast_hour: int = 0) -> str:
    init_time = normalize_time(init_time)
    base_url = base_url or BASE_URL
    return base_url.format(
        year=init_time.strftime("%Y"),
        yyyymm=init_time.strftime("%Y%m"),
        filename=build_filename(init_time, forecast_hour),
    )


def make_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Pangu-experiment-2025-GDAS-downloader/1.0", "Accept-Encoding": "identity"})
    return session


def _raise_unavailable(response, url: str, init_time: datetime | None = None) -> None:
    if response.status_code in (404, 410):
        label = f" for {init_time:%Y-%m-%d %H:00} UTC" if init_time else ""
        raise GDASCycleUnavailable(
            f"GDAS cycle{label} is unavailable (HTTP {response.status_code}): {url}"
        )
    if response.status_code in (401, 403):
        raise GDASCycleUnavailable(f"GDAS archive rejected access (HTTP {response.status_code}): {url}")


def probe_remote_file(session, url: str, *, init_time: datetime | None = None, timeout=(20, 60)) -> int | None:
    if init_time is not None:
        init_time = normalize_time(init_time)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if init_time > now:
            raise GDASCycleUnavailable(f"GDAS cycle is in the future: {init_time:%Y-%m-%d %H:00} UTC")
    response = None
    try:
        response = session.head(url, timeout=timeout, allow_redirects=True)
        _raise_unavailable(response, url, init_time)
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        return int(value) if value else None
    except GDASCycleUnavailable:
        raise
    except Exception:
        # Some mirrors do not implement HEAD; GET will still validate the file.
        return None
    finally:
        if response is not None:
            response.close()


def validate_grib(path: Path) -> None:
    """Check all GRIB2 message boundaries without decoding the fields."""
    size = path.stat().st_size
    offset = 0
    with path.open("rb") as stream:
        while offset < size:
            stream.seek(offset)
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB" or header[7] != 2:
                raise ValueError(f"Invalid GRIB2 header at byte {offset}: {path}")
            length = int.from_bytes(header[8:16], "big")
            if length < 20 or offset + length > size:
                raise ValueError(f"Truncated GRIB2 message at byte {offset}: {path}")
            stream.seek(offset + length - 4)
            if stream.read(4) != b"7777":
                raise ValueError(f"Invalid GRIB2 terminator at byte {offset}: {path}")
            offset += length
    if not size:
        raise ValueError(f"Empty GRIB2 file: {path}")


def _download_once(session, url: str, out_path: Path, remote_size: int | None, *,
                   timeout=(20, 180), range_chunk_bytes: int = 8 * 1024 * 1024) -> bool | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists() and (remote_size is None or out_path.stat().st_size == remote_size):
        try:
            validate_grib(out_path)
            return True
        except ValueError:
            pass
    if out_path.exists() and not part_path.exists():
        out_path.replace(part_path)

    downloaded = part_path.stat().st_size if part_path.exists() else 0
    if remote_size is not None and downloaded == remote_size and part_path.exists():
        try:
            validate_grib(part_path)
            part_path.replace(out_path)
            return True
        except ValueError:
            part_path.unlink()
            downloaded = 0
    if remote_size is not None and downloaded > remote_size:
        part_path.unlink()
        downloaded = 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    segment_end = None
    if remote_size is not None and remote_size - downloaded > range_chunk_bytes:
        segment_end = downloaded + range_chunk_bytes - 1
        headers = {"Range": f"bytes={downloaded}-{segment_end}"}
    mode = "ab" if downloaded else "wb"
    response = session.get(url, stream=True, timeout=timeout, headers=headers)
    try:
        _raise_unavailable(response, url)
        if response.status_code == 416 and downloaded:
            part_path.unlink()
            raise IOError("Local partial file exceeds the remote range; restarting")
        if response.status_code == 200 and downloaded:
            downloaded = 0
            mode = "wb"
        response.raise_for_status()
        response_end = remote_size
        if response.status_code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
            if match is None or int(match[1]) != downloaded:
                raise IOError("Server returned an invalid Content-Range; refusing to append")
            range_end = int(match[2])
            range_total = (remote_size or range_end + 1) if match[3] == "*" else int(match[3])
            if range_end < downloaded or range_end >= range_total or (remote_size is not None and range_total != remote_size):
                raise IOError("Content-Range disagrees with the requested GDAS file")
            if segment_end is not None and range_end != segment_end:
                raise IOError("Content-Range does not match the requested segment")
            remote_size = range_total
            response_end = range_end + 1
        elif response.status_code != 200:
            raise IOError(f"Unexpected GDAS download status: {response.status_code}")
        content_length = response.headers.get("Content-Length")
        total = int(content_length) + downloaded if content_length else remote_size
        if response_end is not None and total is not None and total != response_end:
            raise IOError("Content-Length disagrees with the requested GDAS response size")
        try:
            from tqdm import tqdm
            progress = tqdm(total=remote_size or total, initial=downloaded, unit="B", unit_scale=True, unit_divisor=1024,
                            desc=out_path.name, ascii=True, dynamic_ncols=True)
        except ImportError:
            progress = None
        try:
            with part_path.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
        finally:
            if progress is not None:
                progress.close()
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    expected_size = response_end if response_end is not None else total
    if expected_size is not None and part_path.stat().st_size != expected_size:
        raise IOError(f"incomplete GDAS download: {part_path.stat().st_size} != {expected_size} bytes")
    if remote_size is not None and part_path.stat().st_size < remote_size:
        return None
    try:
        validate_grib(part_path)
    except ValueError:
        part_path.unlink()
        raise
    part_path.replace(out_path)
    return False


def download_file(session, url: str, out_path: Path, *, remote_size: int | None = None,
                  attempts: int = 5, timeout=(20, 180),
                  range_chunk_bytes: int = 8 * 1024 * 1024) -> DownloadResult:
    out_path = Path(out_path)
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if not isinstance(range_chunk_bytes, int) or range_chunk_bytes < 1:
        raise ValueError("range_chunk_bytes must be a positive integer")
    remote_size = remote_size if remote_size is not None else probe_remote_file(session, url, timeout=timeout)
    last_error = None
    attempt = 0
    active_chunk_bytes = range_chunk_bytes
    while attempt < attempts:
        try:
            skipped = _download_once(session, url, out_path, remote_size, timeout=timeout,
                                      range_chunk_bytes=active_chunk_bytes)
            if skipped is None:
                attempt = 0
                continue
            return DownloadResult(None, url, out_path, skipped=skipped)
        except GDASCycleUnavailable:
            raise
        except Exception as exc:
            attempt += 1
            active_chunk_bytes = max(min(1024 * 1024, active_chunk_bytes), active_chunk_bytes // 2)
            last_error = exc
            if attempt == attempts:
                break
            print(f"[GDAS] download interrupted: {exc}; retry {attempt + 1}/{attempts}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GDAS download failed after {attempts} attempts: {last_error}")


def download_cycle(init_time: datetime, raw_root: Path, *, session=None, base_url: str = BASE_URL,
                   forecast_hour: int = 0, attempts: int = 5) -> DownloadResult:
    init_time = normalize_time(init_time)
    if init_time.minute or init_time.second or init_time.hour not in SUPPORTED_CYCLES:
        raise ValueError(f"GDAS cycle must be one of UTC hours {SUPPORTED_CYCLES} with zero minutes and seconds")
    base_url = base_url or BASE_URL
    url = build_url(init_time, base_url=base_url, forecast_hour=forecast_hour)
    from .paths import raw_path

    path = raw_path(raw_root, init_time)
    # Adopt previously downloaded flat files (including partial downloads).
    # Never overwrite a file already present in the organized archive.
    legacy = Path(raw_root) / path.name
    for old, new in ((legacy, path), (legacy.with_suffix(".grib2.part"), path.with_suffix(".grib2.part"))):
        if old.exists() and not new.exists():
            if old == legacy:
                validate_grib(old)
            new.parent.mkdir(parents=True, exist_ok=True)
            old.replace(new)
    own_session = session is None
    session = make_session() if own_session else session
    try:
        remote_size = probe_remote_file(session, url, init_time=init_time)
        result = download_file(session, url, path, remote_size=remote_size, attempts=attempts)
        return DownloadResult(init_time, url, result.path, result.skipped)
    finally:
        if own_session:
            session.close()


def parse_init_time(path: Path) -> datetime:
    match = re.fullmatch(r"gdas1\.fnl0p25\.(\d{10})\.f00\.grib2", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse GDAS initialization time from {Path(path).name}")
    stamp = datetime.strptime(match.group(1), "%Y%m%d%H")
    if stamp.hour not in SUPPORTED_CYCLES:
        raise ValueError(f"Unsupported GDAS cycle: {stamp}")
    return stamp

