from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np

from experiment_2025.tools.split_era5 import split_file
from experiment_2025.core.input_data import Era5Archive
from experiment_2025.core import input_data


def test_split_preserves_values_time_chunks_and_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(input_data, "GRID_SHAPE", (3, 4))
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(3)]
    originals = {}
    for kind, names in (("surface", ("msl", "u10", "v10", "t2m")), ("upper", ("z", "q", "t", "u", "v"))):
        path = tmp_path / "raw/2025" / kind / "all.nc"
        path.parent.mkdir(parents=True)
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("valid_time", 3)
            ds.createDimension("latitude", 3)
            ds.createDimension("longitude", 4)
            t = ds.createVariable("valid_time", "i8", ("valid_time",))
            t.units = "hours since 2025-01-01"
            t[:] = [0, 24, 48]
            ds.createVariable("latitude", "f8", ("latitude",))[:] = [90, 0, -90]
            ds.createVariable("longitude", "f8", ("longitude",))[:] = np.arange(4) * .25
            dims = ("valid_time", "latitude", "longitude")
            shape = (3, 3, 4)
            chunks = (2, 3, 4)
            if kind == "upper":
                ds.createDimension("pressure_level", 13)
                ds.createVariable("pressure_level", "f8", ("pressure_level",))[:] = input_data.PRESSURE_LEVELS
                dims = ("valid_time", "pressure_level", "latitude", "longitude")
                shape = (3, 13, 3, 4)
                chunks = (2, 2, 3, 4)
            for name in names:
                values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 7
                ds.createVariable(name, "f4", dims, zlib=True, chunksizes=chunks)[:] = values
                originals[kind, name] = values
        outputs = split_file(path, tmp_path)
        for i, output in enumerate(outputs):
            with nc.Dataset(output) as ds:
                assert ds.split_complete == 1
                for name in names:
                    np.testing.assert_array_equal(ds[name][:], originals[kind, name][i:i+1])
        before = [path.stat().st_mtime_ns for path in outputs]
        split_file(path, tmp_path)
        assert [path.stat().st_mtime_ns for path in outputs] == before
    reader = Era5Archive(tmp_path)
    assert reader.available_times(2025) == set(dates)
    state = reader.load(dates[-1])
    np.testing.assert_array_equal(state.upper[1], originals['upper', 'q'][-1])
    np.testing.assert_array_equal(state.surface[1], originals['surface', 'u10'][-1])


