"""Convert daily ERA5 NetCDF to Pangu NPY inputs using the legacy channel order."""

import argparse
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..core.config import ERA5_ROOT, PRESSURE_LEVELS, SURFACE_ORDER, UPPER_ORDER
from ..core.experiments import time_label
from ..core.input_data import Era5Archive


def matches_array(path: Path, expected: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        actual = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            return (actual.dtype == expected.dtype and actual.shape == expected.shape
                    and all(np.array_equal(a, b) for a, b in zip(actual, expected)))
        finally:
            del actual
    except (OSError, ValueError, EOFError):
        return False


def save_verified(path: Path, expected: np.ndarray) -> str:
    """Publish an NPY only after its full contents exactly match the NC values."""
    if matches_array(path, expected):
        return "verified_existing"
    temporary = path.with_suffix(".npy.part")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, expected, allow_pickle=False)
        if not matches_array(temporary, expected):
            raise RuntimeError(f"NPY readback differs from NetCDF: {temporary}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "created"


def convert(root: Path, start: datetime, end: datetime):
    if start > end:
        raise ValueError("start-date must not be later than end-date")
    root = root.resolve()
    days = [start + timedelta(days=i) for i in range((end-start).days + 1)]
    # Preflight before writing: both daily source files must exist for every day.
    for stamp in days:
        directory = root / str(stamp.year) / time_label(stamp)
        for name in ("surface.nc", "upper.nc"):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
    report = {
        "start_utc": start.isoformat(), "end_utc": end.isoformat(),
        "surface_order": list(SURFACE_ORDER), "upper_order": list(UPPER_ORDER),
        "pressure_levels_hpa": list(PRESSURE_LEVELS), "dtype": "float32", "days": [],
    }
    report_path = root / f"npy_conversion_{start:%Y%m%d}_{end:%Y%m%d}.json"
    reader = Era5Archive(root)
    begin = time.perf_counter()
    for index, stamp in enumerate(days, 1):
        directory = root / str(stamp.year) / time_label(stamp)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ERA5 q has .*negative grid values.*", category=RuntimeWarning)
            state = reader.load(stamp)
        surface = directory / "input_surface.npy"
        upper = directory / "input_upper.npy"
        row = {
            "time_utc": stamp.isoformat(),
            "surface": save_verified(surface, state.surface),
            "upper": save_verified(upper, state.upper),
            "negative_q_count": state.negative_specific_humidity_count,
            "q_min_kg_kg": float(state.upper[1].min()),
            "bytes": surface.stat().st_size + upper.stat().st_size,
        }
        report["days"].append(row)
        report["elapsed_seconds"] = round(time.perf_counter()-begin, 3)
        report["total_bytes"] = sum(day["bytes"] for day in report["days"])
        temporary = report_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        del state
        if index == 1 or index % 10 == 0 or index == len(days):
            print(f"{index}/{len(days)} {stamp:%Y-%m-%d} verified; elapsed={report['elapsed_seconds']:.1f}s", flush=True)
    print(f"COMPLETE {len(days)} days, {len(days)*2} NPY, {report['total_bytes']/1e9:.3f} GB; report={report_path}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ERA5_ROOT)
    parser.add_argument("--start-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2024, 12, 27))
    parser.add_argument("--end-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d"), default=datetime(2025, 12, 31))
    args = parser.parse_args(argv)
    convert(args.root, args.start_date, args.end_date)


if __name__ == "__main__":
    main()

