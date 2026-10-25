"""Split raw ERA5 archives to year/time/surface.nc and upper.nc losslessly."""

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np

from ..core.config import ERA5_ROOT, SURFACE_ORDER, UPPER_ORDER
from ..core.input_data import _time_labels
from ..core.experiments import time_label


def valid_daily(path, stamp, names, identity):
    if not path.exists():
        return False
    try:
        with nc.Dataset(path) as ds:
            return (getattr(ds, "split_source_identity", "") == identity
                    and _time_labels(ds) == [stamp]
                    and all(name in ds.variables for name in names)
                    and getattr(ds, "split_complete", 0) == 1)
    except (OSError, ValueError, RuntimeError):
        return False


def split_file(source, root):
    kind = source.parent.name
    names = SURFACE_ORDER if kind == "surface" else UPPER_ORDER
    stat = source.stat()
    identity = json.dumps({"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}, sort_keys=True)
    with nc.Dataset(source) as raw:
        times = _time_labels(raw)
        if len(times) != len(set(times)):
            raise ValueError(f"Duplicate times: {source}")
        for name in names:
            if name not in raw.variables:
                raise ValueError(f"Missing {name}: {source}")
        targets = [root / str(stamp.year) / time_label(stamp) / f"{kind}.nc" for stamp in times]
        pending = [i for i, path in enumerate(targets) if not valid_daily(path, times[i], names, identity)]
        print(f"FILE {source.name} {kind}: total={len(times)} pending={len(pending)}", flush=True)
        if not pending:
            return targets
        handles = {}
        hashes = {(index, name): hashlib.sha256() for index in pending for name in names}
        try:
            for index in pending:
                path = targets[index]
                path.parent.mkdir(parents=True, exist_ok=True)
                ds = nc.Dataset(path.with_suffix(".nc.part"), "w", format="NETCDF4")
                handles[index] = ds
                ds.setncatts({key: raw.getncattr(key) for key in raw.ncattrs() if key not in ("split_complete", "split_source_identity")})
                ds.split_source_identity = identity
                for name, dim in raw.dimensions.items():
                    ds.createDimension(name, 1 if name in ("time", "valid_time") else len(dim))
                for name, var in raw.variables.items():
                    if name not in names and name not in raw.dimensions and name not in ("expver", "number"):
                        continue
                    kwargs = {}
                    if name in names:
                        kwargs.update(zlib=True, complevel=1, shuffle=True)
                        kwargs["chunksizes"] = tuple(1 if dim in ("valid_time", "time", "pressure_level", "level") else min(raw.dimensions[dim].size, 180) for dim in var.dimensions)
                    if "_FillValue" in var.ncattrs():
                        kwargs["fill_value"] = var.getncattr("_FillValue")
                    output = ds.createVariable(name, var.datatype, var.dimensions, **kwargs)
                    if name in names:
                        output.set_var_chunk_cache(1024 * 1024, 1009, 0.75)
                    output.setncatts({key: var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
                    if name not in names:
                        key = tuple(slice(index, index + 1) if dim in ("time", "valid_time") else slice(None) for dim in var.dimensions)
                        output[...] = var[key] if key else var[...]
            # Match the source time/pressure chunk sizes to avoid decompressing
            # an entire quarterly chunk separately for every daily file.
            for name in names:
                var = raw[name]
                chunks = var.chunking()
                tblock = chunks[0] if isinstance(chunks, list) else 1
                pblock = chunks[1] if kind == "upper" and isinstance(chunks, list) else 1
                for ti in range(0, len(times), tblock):
                    indices = [i for i in pending if ti <= i < min(ti + tblock, len(times))]
                    if not indices:
                        continue
                    levels = raw.dimensions["pressure_level"].size if kind == "upper" else 1
                    for pi in range(0, levels, pblock):
                        key = (slice(ti, ti + tblock), slice(pi, pi + pblock), slice(None), slice(None)) if kind == "upper" else (slice(ti, ti + tblock), slice(None), slice(None))
                        block = var[key]
                        if np.ma.getmaskarray(block).any() or not np.isfinite(block).all():
                            raise ValueError(f"Missing/non-finite source values: {source} {name} time={ti} level={pi}")
                        for index in indices:
                            dest = handles[index][name]
                            dest_key = (slice(None), slice(pi, pi + pblock), slice(None), slice(None)) if kind == "upper" else (slice(None), slice(None), slice(None))
                            dest[dest_key] = block[index - ti:index - ti + 1]
                            hashes[index, name].update(np.ascontiguousarray(block[index - ti:index - ti + 1]).tobytes())
                        del block
                print(f"  written {name}", flush=True)
            # Read every written variable back before publishing the daily file.
            for index in pending:
                ds = handles.pop(index)
                try:
                    for name in names:
                        values = ds[name][:]
                        if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
                            raise ValueError(f"Daily validation failed: {targets[index]} {name}")
                        if hashlib.sha256(np.ascontiguousarray(values).tobytes()).digest() != hashes[index, name].digest():
                            raise ValueError(f"Source/daily value mismatch: {targets[index]} {name}")
                    ds.split_complete = 1
                finally:
                    ds.close()
                targets[index].with_suffix(".nc.part").replace(targets[index])
                if not valid_daily(targets[index], times[index], names, identity):
                    raise RuntimeError(f"Daily metadata validation failed: {targets[index]}")
        finally:
            for ds in handles.values():
                ds.close()
        return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ERA5_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    files = sorted((root / "raw").rglob("*.nc"))
    if not files:
        raise FileNotFoundError(root / "raw")
    expected = {datetime(2024, 12, 27) + timedelta(days=i) for i in range(370)}
    coverage = {kind: set() for kind in ("surface", "upper")}
    for source in files:
        kind = source.parent.name
        if kind not in coverage:
            raise ValueError(f"Expected raw/<year>/surface or upper: {source}")
        with nc.Dataset(source) as ds:
            times = _time_labels(ds)
        if coverage[kind].intersection(times):
            raise ValueError(f"Overlapping source times: {source}")
        coverage[kind].update(times)
    for kind, times in coverage.items():
        if times != expected:
            raise ValueError(f"{kind} coverage missing={sorted(expected-times)} extra={sorted(times-expected)}")
    outputs = []
    for source in files:
        outputs.extend(split_file(source, root))
    report = {"dates": 370, "daily_files": len(outputs), "bytes": sum(path.stat().st_size for path in outputs), "raw_files": [str(path) for path in files]}
    (root / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

