from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LAND_ZIP = PROJECT_ROOT / "src" / "ne_10m_land.zip"


def land_source_path() -> str:
    if LOCAL_LAND_ZIP.exists():
        return str(LOCAL_LAND_ZIP)
    return shpreader.natural_earth(resolution="10m", category="physical", name="land")


@lru_cache(maxsize=16)
def load_land_union(lon_min: float, lat_min: float, lon_max: float, lat_max: float):
    area_polygon = box(lon_min, lat_min, lon_max, lat_max)
    reader = shpreader.Reader(land_source_path())
    land_geoms = [geom for geom in reader.geometries() if geom.intersects(area_polygon)]
    if not land_geoms:
        return box(0, 0, 0, 0).difference(box(0, 0, 0, 0))
    return unary_union(land_geoms)


def records_on_land_mask(
    records: pd.DataFrame,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.Series:
    if records.empty:
        return pd.Series(False, index=records.index)

    land_union = load_land_union(lon_min, lat_min, lon_max, lat_max)
    prepared_land = prep(land_union)

    coords = records[[lon_col, lat_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mask = [
        bool(prepared_land.covers(Point(lon, lat))) if np.isfinite(lon) and np.isfinite(lat) else False
        for lon, lat in coords
    ]
    return pd.Series(mask, index=records.index)


def filter_ocean_records(
    records: pd.DataFrame,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
):
    land_mask = records_on_land_mask(
        records,
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
        lat_col=lat_col,
        lon_col=lon_col,
    )
    return records.loc[~land_mask].copy(), int(land_mask.sum())
