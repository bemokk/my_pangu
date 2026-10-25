"""Orchestration for downloading, converting, and validating GDAS cycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
import json
import time

from .constants import BASE_URL, default_raw_root
from .convert import convert_grib
from .download import DownloadResult, download_cycle, iter_cycles, make_session
from .paths import existing_raw_path, raw_path, prepared_directory


@dataclass(frozen=True)
class PreparedCycle:
    init_time: datetime
    grib_path: Path
    output_dir: Path
    downloaded: bool


def prepare_cycle(init_time: datetime, *, raw_root: Path | None = None, output_root: Path | None = None,
                  session=None, base_url: str = BASE_URL, overwrite: bool = False) -> PreparedCycle:
    from ..core.config import GDAS_ROOT

    raw_root = Path(raw_root) if raw_root is not None else default_raw_root()
    output_root = Path(output_root) if output_root is not None else Path(GDAS_ROOT)
    result: DownloadResult = download_cycle(init_time, raw_root, session=session, base_url=base_url or BASE_URL)
    output_dir = convert_grib(result.path, output_root, overwrite=overwrite)
    return PreparedCycle(result.init_time, result.path, output_dir, not result.skipped)


def _batch(action, cycles, operation, report_root, continue_on_error):
    outputs, failures = [], []
    report_path = Path(report_root) / "_runs" / f"{action}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    report = {"action": action, "cycles_requested": len(cycles), "tasks": []}
    begin = time.perf_counter()
    for index, stamp in enumerate(cycles, 1):
        print(f"[GDAS] {index}/{len(cycles)} {action} {stamp:%Y-%m-%d %H:00} UTC", flush=True)
        row = {"init_time_utc": stamp.isoformat()}
        started = time.perf_counter()
        try:
            path = operation(stamp)
            outputs.append(path)
            row.update(status="complete", path=str(path))
        except Exception as exc:
            failures.append((stamp, exc))
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[GDAS] failed {stamp}: {exc}", flush=True)
            if not continue_on_error:
                raise
        except BaseException as exc:
            row.update(status="interrupted", error=type(exc).__name__)
            raise
        finally:
            row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            report["tasks"].append(row)
            report.update(elapsed_seconds=round(time.perf_counter()-begin, 3),
                          completed=len(outputs), failed=len(failures))
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_suffix(".json.part")
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(report_path)
    print(f"[GDAS] report: {report_path}", flush=True)
    if failures:
        summary = "; ".join(f"{stamp:%Y-%m-%d %H:00}: {error}" for stamp, error in failures)
        raise RuntimeError(f"{len(failures)} GDAS cycles failed: {summary}")
    return outputs


def download_and_prepare(start: datetime, end: datetime, *, interval_hours: int = 24,
                         raw_root: Path | None = None, output_root: Path | None = None,
                         session=None, base_url: str = BASE_URL, overwrite: bool = False,
                         continue_on_error: bool = False) -> list[Path]:
    from ..core.config import GDAS_ROOT

    cycles = list(iter_cycles(start, end, interval_hours))
    with (make_session() if session is None else nullcontext(session)) as active_session:
        return _batch("run", cycles, lambda stamp: prepare_cycle(
            stamp, raw_root=raw_root, output_root=output_root, session=active_session,
            base_url=base_url, overwrite=overwrite).output_dir,
            output_root if output_root is not None else GDAS_ROOT, continue_on_error)


def download_range(start: datetime, end: datetime, *, interval_hours: int = 24,
                   raw_root: Path | None = None, session=None, base_url: str = BASE_URL,
                   continue_on_error: bool = False) -> list[Path]:
    """Download only; this operation does not require pygrib or netCDF4."""
    raw_root = Path(raw_root) if raw_root is not None else default_raw_root()
    cycles = list(iter_cycles(start, end, interval_hours))
    with (make_session() if session is None else nullcontext(session)) as active_session:
        return _batch("download", cycles, lambda stamp: download_cycle(
            stamp, raw_root, session=active_session, base_url=base_url).path,
            raw_root, continue_on_error)


def process_existing(start: datetime, end: datetime, *, interval_hours: int = 24,
                     raw_root: Path | None = None, output_root: Path | None = None,
                     overwrite: bool = False, continue_on_error: bool = False) -> list[Path]:
    """Convert already downloaded GRIB2 files without making network requests."""

    raw_root = Path(raw_root) if raw_root is not None else default_raw_root()
    from ..core.config import GDAS_ROOT
    output_root = Path(output_root) if output_root is not None else Path(GDAS_ROOT)
    return _batch("process", list(iter_cycles(start, end, interval_hours)),
                  lambda stamp: convert_grib(existing_raw_path(raw_root, stamp), output_root, overwrite=overwrite),
                  output_root, continue_on_error)


def plan(start, end, *, interval_hours=24, raw_root=None, output_root=None, **unused):
    """Show paths and existing files without downloads, conversions or writes."""
    from ..core.config import GDAS_ROOT

    raw_root = Path(raw_root) if raw_root is not None else default_raw_root()
    output_root = Path(output_root) if output_root is not None else GDAS_ROOT
    cycles = list(iter_cycles(start, end, interval_hours))
    raw_existing = sum(existing_raw_path(raw_root, stamp).is_file() for stamp in cycles)
    prepared = sum(all((prepared_directory(output_root, stamp) / name).is_file()
                       for name in ("surface.nc", "upper.nc", "input_surface.npy", "input_upper.npy", "prepared.json"))
                   for stamp in cycles)
    print(f"[GDAS] {len(cycles)} cycles, interval={interval_hours}h, UTC {cycles[0]} .. {cycles[-1]}")
    print(f"[GDAS] existing raw={raw_existing}, prepared directories={prepared} (existence check only)")
    for stamp in dict.fromkeys((cycles[0], cycles[-1])):
        print(f"  raw: {raw_path(raw_root, stamp)}")
        print(f"  inputs: {prepared_directory(output_root, stamp)}")
    return []

