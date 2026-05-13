"""Persistent cache of footprint-to-crosswalk matches.

Accumulates results across runs and areas. The goal is to eventually find
a crosswalk match for every footprint in the library.

Cache file: JSON with structure:
{
  "matches": {
    "FOOTPRINT_NAME": {
      "fp_counts": [N, E, S, W],
      "crosswalk": {
        "lat": ..., "lon": ...,
        "cw_counts": [N, E, S, W],
        "area": "sf-sunset",
        "iou": 0.047,
        "rotation_k": 1
      }
    },
    ...
  },
  "scanned_areas": ["sf-sunset", "sf-soma", ...],
  "library_size": 250
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CACHE_PATH = Path("data/match_cache.json")


@dataclass
class CachedMatch:
    footprint_name: str
    fp_counts: tuple[int, int, int, int]
    lat: float
    lon: float
    cw_counts: tuple[int, int, int, int]
    area: str
    iou: float
    rotation_k: int


def load_cache(path: Path = CACHE_PATH) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"matches": {}, "scanned_areas": [], "library_size": 0}


def save_cache(cache: dict, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cache, f, indent=2)


def add_match(
    cache: dict,
    footprint_name: str,
    fp_counts: tuple[int, int, int, int],
    lat: float,
    lon: float,
    cw_counts: tuple[int, int, int, int],
    area: str,
    iou: float,
    rotation_k: int,
) -> None:
    """Add or update a match. Keeps the highest-IoU match per footprint."""
    existing = cache["matches"].get(footprint_name)
    if existing is None or iou > existing["crosswalk"]["iou"]:
        cache["matches"][footprint_name] = {
            "fp_counts": list(fp_counts),
            "crosswalk": {
                "lat": lat,
                "lon": lon,
                "cw_counts": list(cw_counts),
                "area": area,
                "iou": iou,
                "rotation_k": rotation_k,
            },
        }


def mark_area_scanned(cache: dict, area: str) -> None:
    if area not in cache["scanned_areas"]:
        cache["scanned_areas"].append(area)


def matched_footprint_names(cache: dict) -> set[str]:
    return set(cache["matches"].keys())


def progress_summary(cache: dict) -> str:
    n_matched = len(cache["matches"])
    n_total = cache.get("library_size", 0)
    n_areas = len(cache["scanned_areas"])
    areas = ", ".join(cache["scanned_areas"]) if cache["scanned_areas"] else "(none)"
    if n_total > 0:
        return (f"{n_matched}/{n_total} footprints matched "
                f"({n_matched/n_total*100:.0f}%), "
                f"{n_areas} areas scanned: {areas}")
    return f"{n_matched} footprints matched, {n_areas} areas scanned: {areas}"
