"""Natural Earth land and lake mask used by experiment_2025 statistics."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import covers, points


LOCAL_LAND_ZIP = Path(__file__).resolve().parents[2] / "old" / "src" / "ne_10m_land.zip"
LOCAL_LAKES_ZIP = Path(__file__).resolve().parent / "ne_10m_lakes.zip"


def land_source_path() -> str:
    if LOCAL_LAND_ZIP.exists():
        return str(LOCAL_LAND_ZIP)
    return shpreader.natural_earth(resolution="10m", category="physical", name="land")


def lakes_source_path() -> str:
    if LOCAL_LAKES_ZIP.exists():
        return str(LOCAL_LAKES_ZIP)
    return shpreader.natural_earth(resolution="10m", category="physical", name="lakes")


@lru_cache(maxsize=8)
def load_land_union(lon_min: float, lat_min: float, lon_max: float, lat_max: float):
    area = box(lon_min, lat_min, lon_max, lat_max)
    reader = shpreader.Reader(land_source_path())
    geometries = [geometry for geometry in reader.geometries() if geometry.intersects(area)]
    if not geometries:
        return box(0, 0, 0, 0).difference(box(0, 0, 0, 0))
    return unary_union(geometries)


@lru_cache(maxsize=8)
def load_lake_union(lon_min: float, lat_min: float, lon_max: float, lat_max: float):
    area = box(lon_min, lat_min, lon_max, lat_max)
    reader = shpreader.Reader(lakes_source_path())
    geometries = [geometry for geometry in reader.geometries() if geometry.intersects(area)]
    if not geometries:
        return box(0, 0, 0, 0).difference(box(0, 0, 0, 0))
    return unary_union(geometries)


def records_on_land_mask(lon: np.ndarray, lat: np.ndarray, land_union, lake_union=None) -> np.ndarray:
    """Return a boolean mask; boundaries are land and Natural Earth lakes are water."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    result = np.zeros(lon.shape, dtype=bool)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if np.any(valid):
        point_values = points(lon[valid], lat[valid])
        result[valid] = np.asarray(covers(land_union, point_values), dtype=bool)
        if lake_union is not None:
            result[valid] &= ~np.asarray(covers(lake_union, point_values), dtype=bool)
    return result
