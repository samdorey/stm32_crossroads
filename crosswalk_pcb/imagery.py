"""Fetch aerial imagery as stitched, center-cropped images from public tile services.

Sources are referenced by short name. Add new sources by extending SOURCES.
Tiles are cached on disk to keep iteration fast and avoid hammering services.

Note: Esri World Imagery and USGS basemaps are publicly accessible without a key
but their terms of service apply (no-key endpoints are intended for low-volume,
non-commercial use). For anything beyond prototyping, use a paid plan with a key.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from .tiles import TILE_SIZE, TileWindow, ground_resolution_m_per_px, tile_window_for_center

USER_AGENT = "crosswalk-pcb-prototype/0.1 (https://github.com/samdorey/stm32_crossroads)"


@dataclass(frozen=True)
class ImagerySource:
    name: str
    url_template: str  # uses {z} {x} {y} and optionally {token}
    max_zoom: int
    tile_size_px: int = 256  # Mapbox @2x serves 512, most others 256
    token_env: str | None = None  # name of env var holding access token
    description: str = ""


SOURCES: dict[str, ImagerySource] = {
    # Mapbox Satellite. With @2x we get 512px tiles, doubling effective resolution.
    # Free tier is 750k raster tile requests/month, plenty for prototyping.
    "mapbox_satellite": ImagerySource(
        name="mapbox_satellite",
        url_template=(
            "https://api.mapbox.com/v4/mapbox.satellite/"
            "{z}/{x}/{y}@2x.jpg90?access_token={token}"
        ),
        max_zoom=22,
        tile_size_px=512,
        token_env="MAPBOX_TOKEN",
        description="Mapbox Satellite (commercial mosaic, ~30 cm in metros) — @2x tiles",
    ),
    # Mosaic of best-available high-res imagery in metros (often Maxar/Vexcel,
    # ~30 cm or better). Free for low-volume use under Esri's basemap terms.
    "esri_world": ImagerySource(
        name="esri_world",
        url_template=(
            "https://services.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        max_zoom=19,
        description="Esri World Imagery (Maxar/Vexcel mosaic in metros)",
    ),
    # USGS basemap fed primarily by NAIP (60 cm CONUS, sometimes 30 cm).
    "usgs_imagery": ImagerySource(
        name="usgs_imagery",
        url_template=(
            "https://basemap.nationalmap.gov/arcgis/rest/services/"
            "USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
        ),
        max_zoom=19,
        description="USGS Imagery basemap (NAIP-based)",
    ),
    # SFGIS public ArcGIS server. Endpoint exists; the exact service name has
    # changed over the years. Verify against https://sfplanninggis.org or the
    # current SFGIS portal listing before relying on this in production.
    "sf_data_2022": ImagerySource(
        name="sf_data_2022",
        url_template=(
            "https://sfplanninggis.org/arcgis/rest/services/Imagery/"
            "Aerial_2022/MapServer/tile/{z}/{y}/{x}"
        ),
        max_zoom=21,
        description="DataSF 2022 aerial (~7.5 cm). URL may need updating.",
    ),
}


def _cache_path(cache_dir: Path, source: str, z: int, x: int, y: int) -> Path:
    return cache_dir / source / f"{z}" / f"{x}_{y}.png"


def _resolve_token(source: ImagerySource) -> str | None:
    if source.token_env is None:
        return None
    token = os.environ.get(source.token_env)
    if not token:
        raise RuntimeError(
            f"source {source.name!r} requires env var {source.token_env}; not set"
        )
    return token


def _format_url(source: ImagerySource, z: int, x: int, y: int) -> str:
    fmt: dict[str, object] = {"z": z, "x": x, "y": y}
    token = _resolve_token(source)
    if token is not None:
        fmt["token"] = token
    return source.url_template.format(**fmt)


def fetch_tile(
    source: ImagerySource,
    z: int,
    x: int,
    y: int,
    cache_dir: Path | None,
    session: requests.Session,
    timeout: float = 15.0,
) -> Image.Image:
    """Fetch a single tile (sized per source.tile_size_px). Cached if cache_dir set."""
    if cache_dir is not None:
        cp = _cache_path(cache_dir, source.name, z, x, y)
        if cp.exists():
            return Image.open(cp).convert("RGB")
    url = _format_url(source, z, x, y)
    resp = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    if not resp.headers.get("content-type", "").startswith("image/"):
        # Some ArcGIS servers return a JSON error with HTTP 200. Surface that.
        raise RuntimeError(
            f"non-image response from {source.name}: "
            f"{resp.headers.get('content-type')!r} body={resp.text[:200]!r}"
        )
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    if img.size != (source.tile_size_px, source.tile_size_px):
        # Be strict — if a source returns a different size than declared, that
        # breaks our stitching math. Surface it clearly.
        raise RuntimeError(
            f"{source.name} returned tile of size {img.size}, "
            f"expected {(source.tile_size_px, source.tile_size_px)}"
        )
    if cache_dir is not None:
        cp = _cache_path(cache_dir, source.name, z, x, y)
        cp.parent.mkdir(parents=True, exist_ok=True)
        img.save(cp, "PNG")
    return img


@dataclass
class FetchResult:
    source: str
    image: Image.Image
    zoom: int
    m_per_px: float  # ground resolution at the image's center latitude
    center_px: tuple[float, float]  # pixel coords of the requested lon/lat within image


def fetch_centered(
    source_name: str,
    lon: float,
    lat: float,
    zoom: int,
    size_px: int = 512,
    cache_dir: Path | None = None,
    session: requests.Session | None = None,
    polite_delay_s: float = 0.05,
) -> FetchResult:
    """Fetch and stitch a size_px x size_px image centered on (lon, lat).

    size_px is in the source's native pixel grid (so @2x sources at 512 px/tile
    give double effective resolution for the same zoom).
    """
    if source_name not in SOURCES:
        raise KeyError(f"unknown imagery source: {source_name}")
    source = SOURCES[source_name]
    if zoom > source.max_zoom:
        raise ValueError(
            f"zoom {zoom} exceeds {source.name} max_zoom {source.max_zoom}"
        )
    ts = source.tile_size_px
    # tile_window_for_center uses TILE_SIZE=256 internally. Scale size_px when
    # the source uses larger native tiles so we end up with the right pixel crop.
    size_px_in_native = size_px
    size_px_for_window = size_px_in_native * 256 // ts
    window = tile_window_for_center(lon, lat, zoom, size_px_for_window)
    sess = session or requests.Session()
    canvas = Image.new("RGB", ((window.x1 - window.x0) * ts, (window.y1 - window.y0) * ts))
    for dy, ty in enumerate(range(window.y0, window.y1)):
        for dx, tx in enumerate(range(window.x0, window.x1)):
            tile = fetch_tile(source, zoom, tx, ty, cache_dir, sess)
            canvas.paste(tile, (dx * ts, dy * ts))
            if polite_delay_s:
                time.sleep(polite_delay_s)
    # window.center_px is in TILE_SIZE=256 coords. Rescale to native px.
    cx_native = window.center_px[0] * ts / 256
    cy_native = window.center_px[1] * ts / 256
    left = int(round(cx_native - size_px_in_native / 2))
    top = int(round(cy_native - size_px_in_native / 2))
    crop = canvas.crop((left, top, left + size_px_in_native, top + size_px_in_native))
    center_in_crop = (cx_native - left, cy_native - top)
    # m_per_px in the native pixel grid: standard formula uses 256-px tiles,
    # so we divide by the per-tile scale factor.
    m_per_px = ground_resolution_m_per_px(lat, zoom) * 256.0 / ts
    return FetchResult(
        source=source.name,
        image=crop,
        zoom=zoom,
        m_per_px=m_per_px,
        center_px=center_in_crop,
    )


def stable_intersection_id(lon: float, lat: float) -> str:
    """Deterministic short id for a lat/lon, useful for output filenames."""
    s = f"{lon:.6f},{lat:.6f}"
    return hashlib.md5(s.encode()).hexdigest()[:10]
