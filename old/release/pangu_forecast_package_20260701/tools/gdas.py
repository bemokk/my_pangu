"""GDAS/FNL download and GRIB2-to-Pangu conversion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "model_input" / "raw" / "gdas"
BASE_URL = "https://data.gdex.ucar.edu/d083003/{year}/{yyyymm}/{filename}"
SUPPORTED_CYCLES = (0, 6, 12, 18)


class GDASCycleUnavailable(RuntimeError):
    """The requested GDAS cycle is in the future or not published remotely."""


def timerange(start: datetime, end: datetime, interval_hours: int):
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=interval_hours)


def _make_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Pangu-GDAS-FNL-downloader/1.1"})
    return session


def _probe_remote_file(session, url: str, init_time: datetime) -> int | None:
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    if init_time > now_utc:
        raise GDASCycleUnavailable(
            f"请求时次：{init_time:%Y-%m-%d %H:00} UTC\n"
            f"原因：该时次晚于当前 UTC 时间 {now_utc:%Y-%m-%d %H:%M}，数据尚未生成。"
        )

    response = session.head(url, timeout=(20, 60), allow_redirects=True)
    if response.status_code in {404, 410}:
        raise GDASCycleUnavailable(
            f"请求时次：{init_time:%Y-%m-%d %H:00} UTC\n"
            f"原因：远端返回 HTTP {response.status_code}，该 GDAS 时次可能尚未更新。\n"
            f"地址：{url}"
        )
    if response.status_code in {401, 403}:
        raise GDASCycleUnavailable(
            f"请求时次：{init_time:%Y-%m-%d %H:00} UTC\n"
            f"原因：远端返回 HTTP {response.status_code}，服务器拒绝访问。\n"
            f"地址：{url}"
        )
    response.raise_for_status()
    value = response.headers.get("Content-Length")
    return int(value) if value else None


def _download_file(
    session,
    url: str,
    out_path: Path,
    remote_size: int | None,
) -> None:
    from tqdm import tqdm

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists() and remote_size and out_path.stat().st_size == remote_size:
        print(f"[GDAS] 跳过完整文件：{out_path.name}")
        return
    if out_path.exists() and not part_path.exists():
        out_path.replace(part_path)

    downloaded = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    mode = "ab" if downloaded else "wb"
    with session.get(url, stream=True, timeout=(20, 180), headers=headers) as response:
        if response.status_code in {404, 410}:
            raise GDASCycleUnavailable(
                f"下载开始前远端文件消失（HTTP {response.status_code}）：{url}"
            )
        if response.status_code == 200 and downloaded:
            downloaded = 0
            mode = "wb"
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        total = int(length) + downloaded if length else remote_size
        with part_path.open(mode) as stream, tqdm(
            total=total,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=out_path.name,
            ascii=True,
            dynamic_ncols=True,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
                    progress.update(len(chunk))
    if remote_size is not None and part_path.stat().st_size != remote_size:
        raise IOError(
            f"下载不完整：已下载 {part_path.stat().st_size} 字节，"
            f"远端文件应为 {remote_size} 字节"
        )
    part_path.replace(out_path)


def _download_with_retries(
    session,
    url: str,
    out_path: Path,
    remote_size: int | None,
    attempts: int = 5,
) -> None:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            _download_file(session, url, out_path, remote_size)
            return
        except GDASCycleUnavailable:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(2 ** attempt, 30)
            print(
                f"[GDAS] 下载中断：{exc}\n"
                f"       {delay} 秒后断点续传，第 {attempt + 1}/{attempts} 次尝试"
            )
            time.sleep(delay)
    raise RuntimeError(f"连续 {attempts} 次下载失败：{last_error}")


def _convert(grib_file: Path) -> None:
    tools_dir = Path(__file__).resolve().parent
    for script_name in ["gdas_surface.py", "gdas_upper.py"]:
        subprocess.run(
            [sys.executable, str(tools_dir / script_name), "--grib_file", str(grib_file)],
            check=True,
        )


def download_and_prepare(
    start: datetime,
    end: datetime,
    interval_hours: int = 6,
    raw_dir: Path | None = None,
) -> list[Path]:
    """Download 00/06/12/18 UTC GDAS/FNL cycles and create Pangu inputs."""
    if start.hour not in SUPPORTED_CYCLES or end.hour not in SUPPORTED_CYCLES:
        raise ValueError(f"GDAS 时次只能是 {SUPPORTED_CYCLES} UTC")
    if start.minute or end.minute or start.second or end.second:
        raise ValueError("GDAS 时次的分和秒必须为 0")
    if interval_hours < 1 or interval_hours % 6 != 0:
        raise ValueError("GDAS 时间间隔必须是 6 的正整数倍")

    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    session = _make_session()
    prepared: list[Path] = []

    for init_time in timerange(start, end, interval_hours):
        if init_time.hour not in SUPPORTED_CYCLES:
            raise ValueError(
                f"时间间隔产生了不支持的 GDAS 时次：{init_time:%Y-%m-%d %H:%M}"
            )
        stamp = init_time.strftime("%Y%m%d%H")
        filename = f"gdas1.fnl0p25.{stamp}.f00.grib2"
        url = BASE_URL.format(
            year=init_time.strftime("%Y"),
            yyyymm=init_time.strftime("%Y%m"),
            filename=filename,
        )
        grib_file = raw_dir / filename

        print(f"[GDAS] 检查 {init_time:%Y-%m-%d %H:00} UTC 是否已更新", flush=True)
        remote_size = _probe_remote_file(session, url, init_time)
        print(f"[GDAS] 下载 {init_time:%Y-%m-%d %H:00} UTC", flush=True)
        _download_with_retries(session, url, grib_file, remote_size)
        print(f"[GDAS] 转换 {grib_file.name}", flush=True)
        _convert(grib_file)
        prepared.append(
            PROJECT_ROOT
            / "model_input"
            / "single_time_point"
            / "gdas"
            / init_time.strftime("%Y-%m-%d-%H-%M")
        )

    return prepared
