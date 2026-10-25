"""Small GRIB messages exercise channel ordering and real output roundtrips."""

from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np
import pytest

from experiment_2025.gdas import convert


STAMP = datetime(2025, 7, 1)
SHAPE = (3, 4)


class Message:
    def __init__(self, short, level_type, level, value):
        self.shortName = short
        self.typeOfLevel = level_type
        self.level = level
        self.analDate = STAMP
        self.validDate = STAMP
        self.values = np.arange(12, dtype=np.float32).reshape(SHAPE) + value
        self.lat = np.array([-90, 0, 90], dtype=np.float32)
        self.lon = np.array([180, 270, 0, 90], dtype=np.float32)

    def latlons(self):
        lon, lat = np.meshgrid(self.lon, self.lat)
        return lat, lon

    def __getitem__(self, key):
        return self.metadata[key]


class Grib:
    def __init__(self, messages):
        self.messages = messages
        self.closed = False

    def __iter__(self):
        return iter(self.messages)

    def close(self):
        self.closed = True


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "gdas1.fnl0p25.2025070100.f00.grib2"
    path.write_bytes(b"test source")
    messages = [Message(target["shortName"], target["typeOfLevel"], target["level"], 100 * i)
                for i, target in enumerate(convert.SURFACE_TARGETS.values())]
    messages += [Message(short, "isobaricInhPa", level, 1000 * i + level)
                 for i, short in enumerate(("gh", "q", "t", "u", "v")) for level in convert.PRESSURE_LEVELS]
    handles = []
    class Pygrib:
        @staticmethod
        def open(path):
            handle = Grib(messages)
            handles.append(handle)
            return handle
    monkeypatch.setattr(convert, "_pygrib", lambda: Pygrib)
    return path, messages, handles


def test_full_conversion_roundtrip_order_and_resume(source, tmp_path):
    path, messages, handles = source
    root = tmp_path / "inputs"
    directory = convert.convert_grib(path, root, expected_shape=SHAPE)
    assert directory == root / "2025" / "2025-07-01-00-00"
    surface = np.load(directory / "input_surface.npy")
    upper = np.load(directory / "input_upper.npy")
    expected = messages[0].values[::-1, [2, 3, 0, 1]]
    assert surface.shape == (4, *SHAPE) and upper.shape == (5, 13, *SHAPE)
    np.testing.assert_array_equal(surface[0], expected)
    np.testing.assert_array_equal(upper[0, 0], (expected + 1000) * np.float32(convert.GRAVITY))
    np.testing.assert_array_equal(upper[1, 0], expected + 2000)
    with nc.Dataset(directory / "upper.nc") as ds:
        np.testing.assert_array_equal(ds["pressure_level"][:], convert.PRESSURE_LEVELS)
        assert ds["z"].units == "m2 s-2" and ds["z"].filters()["zlib"]
    assert all(handle.closed for handle in handles)
    old_mtime = (directory / "upper.nc").stat().st_mtime_ns
    convert.convert_grib(path, root, expected_shape=SHAPE)
    assert (directory / "upper.nc").stat().st_mtime_ns == old_mtime
    assert len(handles) == 2


def test_corrupt_output_and_changed_source_rebuild(source, tmp_path):
    path, messages, handles = source
    root = tmp_path / "inputs"
    directory = convert.convert_grib(path, root, expected_shape=SHAPE)
    (directory / "input_upper.npy").write_bytes(b"broken")
    convert.convert_grib(path, root, expected_shape=SHAPE)
    assert np.load(directory / "input_upper.npy").shape == (5, 13, *SHAPE)
    path.write_bytes(b"updated source")
    convert.convert_grib(path, root, expected_shape=SHAPE)
    assert len(handles) == 6


@pytest.mark.parametrize("problem", ["missing", "mask", "time", "grid"])
def test_bad_fields_do_not_publish(source, tmp_path, problem):
    path, messages, handles = source
    if problem == "missing":
        messages.pop()
    elif problem == "mask":
        messages[-1].values = np.ma.array(messages[-1].values, mask=np.ones(SHAPE, dtype=bool))
    elif problem == "time":
        messages[-1].validDate += timedelta(hours=6)
    else:
        messages[-1].lat[1] += 1
    root = tmp_path / "inputs"
    with pytest.raises(ValueError):
        convert.convert_grib(path, root, expected_shape=SHAPE)
    assert not list(root.rglob("prepared.json"))
    assert all(handle.closed for handle in handles)


def test_wrong_global_grid_rejected():
    lat = np.linspace(90, -90, 721, dtype=np.float32)
    lon = np.arange(1440, dtype=np.float32) * 0.25 + 0.125
    with pytest.raises(ValueError, match="0.25-degree"):
        convert._check_grid(lat, lon, (721, 1440))


def test_write_failure_is_recoverable(source, tmp_path, monkeypatch):
    path, messages, handles = source
    root = tmp_path / "inputs"
    original = convert.write_upper
    monkeypatch.setattr(convert, "write_upper", lambda *args: (_ for _ in ()).throw(IOError("disk full")))
    with pytest.raises(IOError, match="disk full"):
        convert.convert_grib(path, root, expected_shape=SHAPE)
    assert not list(root.rglob("prepared.json"))
    monkeypatch.setattr(convert, "write_upper", original)
    directory = convert.convert_grib(path, root, expected_shape=SHAPE)
    assert (directory / "prepared.json").exists()


def masked_height_message():
    message = Message("gh", "isobaricInhPa", 250, 0)
    original = np.array([9998.8, 9998.92, 9999, 9999.08, 9999.2, 10100,
                         9800, 9900, 10200, 10300, 10400, 10500], dtype=np.float64).reshape(SHAPE)
    message.values = np.ma.masked_values(original, 9999)
    message.metadata = {"numberOfMissing": 0, "bitmapPresent": 0, "missingValueManagementUsed": 0,
                        "numberOfDataPoints": 12, "numberOfValues": 12}
    return message, original


def test_false_missing_height_mask_preserves_original_values():
    message, original = masked_height_message()
    assert np.ma.getmaskarray(message.values).sum() == 3
    with pytest.warns(RuntimeWarning, match="preserving 3 original values"):
        actual = convert._values(message)
    np.testing.assert_array_equal(actual, original.astype(np.float32))
    assert actual.dtype == np.float32
    assert np.ma.getmaskarray(message.values).sum() == 3  # no in-place changes


@pytest.mark.parametrize("key,value", [
    ("numberOfMissing", 1), ("bitmapPresent", 1),
    ("missingValueManagementUsed", 1), ("missingValueManagementUsed", 2),
    ("numberOfValues", 11), ("numberOfDataPoints", 13),
    ("numberOfDataPoints", 13.0),
])
def test_real_or_inconsistent_missing_metadata_retains_mask(key, value):
    message, original = masked_height_message()
    message.metadata[key] = value
    actual = convert._values(message)
    assert np.isnan(actual[np.ma.getmaskarray(message.values)]).all()
    assert not np.isfinite(actual).all()


@pytest.mark.parametrize("key", ["numberOfMissing", "bitmapPresent", "missingValueManagementUsed",
                                  "numberOfDataPoints", "numberOfValues"])
def test_unknown_missing_metadata_keeps_mask(key):
    message, _ = masked_height_message()
    del message.metadata[key]
    assert np.isnan(convert._values(message)[np.ma.getmaskarray(message.values)]).all()


def test_restored_values_are_not_interpolated_before_geopotential_conversion(source, tmp_path):
    path, messages, _ = source
    message, original = masked_height_message()
    index = next(i for i, m in enumerate(messages) if m.shortName == "gh" and m.level == 250)
    messages[index] = message
    with pytest.warns(RuntimeWarning, match="preserving"):
        directory = convert.convert_grib(path, tmp_path / "inputs", expected_shape=SHAPE)
    actual = np.load(directory / "input_upper.npy")[0, list(convert.PRESSURE_LEVELS).index(250)]
    expected = original.astype(np.float32)[::-1, [2, 3, 0, 1]] * np.float32(convert.GRAVITY)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("missing_key,value", [("bitmapPresent", 1), ("missingValueManagementUsed", 1),
                                               ("numberOfMissing", 1)])
def test_true_missing_fields_still_block_publication(source, tmp_path, missing_key, value):
    path, messages, _ = source
    message, _ = masked_height_message()
    message.metadata[missing_key] = value
    index = next(i for i, m in enumerate(messages) if m.shortName == "gh" and m.level == 250)
    messages[index] = message
    with pytest.raises(ValueError, match="non-finite"):
        convert.convert_grib(path, tmp_path / "inputs", expected_shape=SHAPE)
    assert not list((tmp_path / "inputs").rglob("prepared.json"))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_still_block_publication_after_unmask(source, tmp_path, bad):
    path, messages, _ = source
    message, _ = masked_height_message()
    message.values.data[0, 0] = bad
    index = next(i for i, m in enumerate(messages) if m.shortName == "gh" and m.level == 250)
    messages[index] = message
    with pytest.warns(RuntimeWarning, match="preserving"):
        with pytest.raises(ValueError, match="non-finite"):
            convert.convert_grib(path, tmp_path / "inputs", expected_shape=SHAPE)
    assert not list((tmp_path / "inputs").rglob("prepared.json"))
