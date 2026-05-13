"""Web Mercator tile math and pixel-space helpers.

All imagery sources we use (Esri World Imagery, USGS basemaps, most ArcGIS
REST tile services) serve Web Mercator (EPSG:3857) tiles at standard
z/x/y addressing with 256 px tiles. This module isolates that math so the
rest of the code can stay in lat/lon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TILE_SIZE = 256
EARTH_CIRCUMFERENCE_M = 2 * math.pi * 6378137.0  # WGS84 equatorial circumference


def lonlat_to_tile_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Fractional tile coordinates (not floored) for a lon/lat at a zoom level."""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_xy_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Inverse of lonlat_to_tile_xy, returns (lon, lat) for fractional tile coords."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return lon, math.degrees(lat_rad)


def ground_resolution_m_per_px(lat: float, zoom: int) -> float:
    """Approximate ground resolution (meters per pixel) at a latitude and zoom."""
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (TILE_SIZE * 2 ** zoom)


def zoom_for_resolution(lat: float, target_m_per_px: float) -> int:
    """Smallest integer zoom whose ground resolution is <= target_m_per_px."""
    z = 0
    while ground_resolution_m_per_px(lat, z) > target_m_per_px and z < 22:
        z += 1
    return z


@dataclass(frozen=True)
class TileWindow:
    """A rectangular tile window covering some bbox, plus the sub-pixel offset
    of a reference lon/lat (typically the intersection center) within it."""

    zoom: int
    x0: int  # left tile x (inclusive)
    y0: int  # top tile y (inclusive)
    x1: int  # right tile x (exclusive)
    y1: int  # bottom tile y (exclusive)
    center_px: tuple[float, float]  # pixel coords of the original lon/lat in the stitched image

    @property
    def width_px(self) -> int:
        return (self.x1 - self.x0) * TILE_SIZE

    @property
    def height_px(self) -> int:
        return (self.y1 - self.y0) * TILE_SIZE


def tile_window_for_center(
    lon: float, lat: float, zoom: int, size_px: int
) -> TileWindow:
    """Compute the tile window that contains a size_px x size_px crop centered
    on (lon, lat) at the given zoom."""
    cx, cy = lonlat_to_tile_xy(lon, lat, zoom)
    half = size_px / 2.0 / TILE_SIZE  # in tiles
    x0 = int(math.floor(cx - half))
    y0 = int(math.floor(cy - half))
    x1 = int(math.floor(cx + half)) + 1
    y1 = int(math.floor(cy + half)) + 1
    cx_px = (cx - x0) * TILE_SIZE
    cy_px = (cy - y0) * TILE_SIZE
    return TileWindow(zoom, x0, y0, x1, y1, (cx_px, cy_px))
