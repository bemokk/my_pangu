from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Buoy"))

from plots import plot_china_sea_hex_counts as hex_counts  # noqa: E402


def test_select_hex_label_point_uses_center_when_center_is_clear_ocean():
    hexagon = hex_counts.regular_hexagon(120.0, 20.0, 1.0)
    ocean_area = hexagon.difference(
        box(120.6, 19.8, 121.1, 20.2)
    )

    label_point = hex_counts.select_hex_label_point(hexagon, ocean_area)

    assert label_point is not None
    assert label_point.x == pytest.approx(hexagon.centroid.x)
    assert label_point.y == pytest.approx(hexagon.centroid.y)


def test_select_hex_label_point_keeps_boundary_hexagon_inside_safe_extent():
    hexagon = hex_counts.regular_hexagon(hex_counts.LON_MIN, 20.0, 1.0)
    ocean_area = box(
        hex_counts.LON_MIN,
        hex_counts.LAT_MIN,
        hex_counts.LON_MAX,
        hex_counts.LAT_MAX,
    )

    label_point = hex_counts.select_hex_label_point(hexagon, ocean_area)

    assert label_point is not None
    assert label_point.x >= hex_counts.LON_MIN + hex_counts.LABEL_EDGE_MARGIN_DEG
