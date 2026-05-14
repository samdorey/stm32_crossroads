"""Scan an area for crosswalks that could serve as real PCB footprints.

Chains: OSM query → imagery fetch → detection → exact count matching
against the KiCad library. Results are appended to the persistent cache
so progress accumulates across runs.

Usage:
    python scripts/scan_area.py --area sf-sunset --library /path/to/Amodo.pretty
    python scripts/scan_area.py --area sf-all --top 200 --library /path/to/Amodo.pretty

Check progress:
    python scripts/scan_area.py --status --library /path/to/Amodo.pretty
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosswalk_pcb.cache import (
    add_match, load_cache, mark_area_scanned, matched_footprint_names,
    progress_summary, save_cache, CACHE_PATH,
)
from crosswalk_pcb.detect import (
    align_image_to_roads, build_road_mask, detect_stripes_periodic,
    draw_overlay, extract_stripes,
    find_paint_mask, find_paint_mask_adaptive, find_paint_mask_contrast,
    group_stripes_into_arrays,
)
from crosswalk_pcb.imagery import SOURCES, fetch_centered, stable_intersection_id
from crosswalk_pcb.kicad import load_library
from crosswalk_pcb.match import (
    crosswalk_side_counts, find_exact_matches, footprint_side_counts,
    render_aligned_comparison, _rotate_counts,
)
from crosswalk_pcb.osm import build_query, extract_junctions, rank_candidates, run_overpass

# Same area definitions as find_intersections.py.
AREAS: dict[str, tuple[float, float, float, float]] = {
    # San Francisco
    "sf-soma":      (37.770, -122.420, 37.790, -122.395),
    "sf-mission":   (37.745, -122.430, 37.770, -122.405),
    "sf-sunset":    (37.755, -122.510, 37.775, -122.470),
    "sf-richmond":  (37.770, -122.510, 37.790, -122.470),
    "sf-all":       (37.708, -122.515, 37.815, -122.355),
    # New York City
    "nyc-midtown":  (40.748, -73.990, 40.762, -73.975),  # Times Sq to Grand Central
    "nyc-chelsea":  (40.740, -74.002, 40.752, -73.990),  # Chelsea / Flatiron
    "nyc-les":      (40.715, -73.995, 40.725, -73.982),  # Lower East Side
    "nyc-uws":      (40.775, -73.985, 40.790, -73.970),  # Upper West Side
    "nyc-bk-slope": (40.670, -73.985, 40.682, -73.972),  # Park Slope, Brooklyn
    # Detroit — grid is nearly N-S aligned
    "det-midtown":  (42.330, -83.055, 42.345, -83.040),  # Midtown Detroit
    "det-corktown": (42.325, -83.075, 42.340, -83.055),  # Corktown / Michigan Ave
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=sorted(AREAS.keys()))
    ap.add_argument("--library", required=True, help="path to .pretty directory")
    ap.add_argument("--top", type=int, default=300,
                    help="max candidates to process per area")
    ap.add_argument("--source", default="mapbox_satellite")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--detect", default="global",
                    choices=["global", "adaptive", "contrast", "periodic"],
                    help="stripe detection method: global threshold, "
                         "adaptive local contrast, color-contrast "
                         "(paint-on-asphalt), or periodic autocorrelation")
    ap.add_argument("--road-mask", action="store_true",
                    help="mask out non-road areas using OSM leg bearings "
                         "before detection (reduces false positives)")
    ap.add_argument("--status", action="store_true",
                    help="just print progress and unmatched footprints, then exit")
    ap.add_argument("--cache", default=str(CACHE_PATH))
    args = ap.parse_args()

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    lib = load_library(Path(args.library))
    cache["library_size"] = len(lib)
    print(f"library: {len(lib)} footprints")

    if args.status:
        print(progress_summary(cache))
        matched = matched_footprint_names(cache)
        unmatched = [fp.name for fp in lib if fp.name not in matched]
        print(f"\nunmatched ({len(unmatched)}):")
        for name in sorted(unmatched)[:30]:
            c = footprint_side_counts(
                next(fp for fp in lib if fp.name == name))
            print(f"  {name:45s}  pads={c}")
        if len(unmatched) > 30:
            print(f"  ... and {len(unmatched) - 30} more")
        return 0

    if not args.area:
        ap.error("--area is required unless --status")

    # Build set of footprint count patterns we still need (skip already matched).
    already_matched = matched_footprint_names(cache)
    needed_patterns: set[tuple[int, int, int, int]] = set()
    for fp in lib:
        if fp.name in already_matched:
            continue
        c = footprint_side_counts(fp)
        for k in range(4):
            needed_patterns.add(_rotate_counts(c, k))
    print(f"already matched: {len(already_matched)}, "
          f"still need: {len(lib) - len(already_matched)} footprints "
          f"({len(needed_patterns)} count patterns)")

    # OSM query.
    bbox = AREAS[args.area]
    overpass_cache = Path("data/overpass_cache")
    overpass_cache.mkdir(parents=True, exist_ok=True)
    label = args.area
    cache_file = overpass_cache / f"{label}.json"
    query = build_query(bbox)
    print(f"querying OSM for {label}...")
    data = run_overpass(query, cache_path=cache_file)
    junctions = extract_junctions(data)
    candidates = rank_candidates(junctions, min_perpendicularity=0.5,
                                 min_legs_with_crossing=2)
    print(f"  {len(candidates)} candidates after OSM filter")
    candidates = candidates[:args.top]

    src = SOURCES[args.source]
    zoom = min(args.zoom, src.max_zoom)
    tile_cache = Path("data/tile_cache")
    out_dir = Path("data/exact_matches")
    out_dir.mkdir(parents=True, exist_ok=True)

    new_matches = 0
    for i, cand in enumerate(candidates):
        lon, lat = cand.lon, cand.lat
        sid = stable_intersection_id(lon, lat)
        try:
            res = fetch_centered(args.source, lon=lon, lat=lat, zoom=zoom,
                                 size_px=args.size, cache_dir=tile_cache)
        except Exception:
            continue
        bgr = cv2.cvtColor(np.array(res.image.convert("RGB")), cv2.COLOR_RGB2BGR)
        bearings = [float(b) for b in cand.leg_bearings]
        if bearings:
            bgr, _ = align_image_to_roads(bgr, bearings)
        aerial_bgr = bgr.copy()
        center = (bgr.shape[1] / 2, bgr.shape[0] / 2)
        road_mask = None
        if args.road_mask and bearings:
            road_mask = build_road_mask(
                bgr.shape, bearings, center,
                road_half_width_px=12.0 / res.m_per_px,   # ~12m half-width
                length_px=bgr.shape[0] * 0.45,
                margin_px=5.0 / res.m_per_px,             # ~5m margin
                center_radius_px=20.0 / res.m_per_px,     # ~20m radius
            )
        try:
            if args.detect == "periodic":
                stripes = detect_stripes_periodic(
                    bgr, bearings, res.m_per_px, center,
                    road_mask=road_mask,
                )
            else:
                if args.detect == "contrast":
                    mask = find_paint_mask_contrast(bgr)
                elif args.detect == "adaptive":
                    mask = find_paint_mask_adaptive(bgr)
                else:
                    mask = find_paint_mask(bgr)
                if road_mask is not None:
                    mask = cv2.bitwise_and(mask, road_mask)
                stripes = extract_stripes(mask, m_per_px=res.m_per_px)
            arrays = group_stripes_into_arrays(stripes)
        except Exception:
            continue
        cw_counts = crosswalk_side_counts(arrays, center)

        # Quick check: does this count pattern match anything we still need?
        if cw_counts not in needed_patterns:
            continue

        all_stripes = [s for a in arrays for s in a.stripes]
        det_overlay = draw_overlay(aerial_bgr, arrays)
        matches = find_exact_matches(arrays, center, lib)
        for m in matches:
            if m.footprint_name in already_matched:
                continue
            add_match(cache, m.footprint_name, m.fp_counts,
                      lat, lon, cw_counts, label, m.iou, m.rotation_k)
            already_matched.add(m.footprint_name)
            new_matches += 1
            # Save 5-panel comparison image.
            fp = next(f for f in lib if f.name == m.footprint_name)
            comp = render_aligned_comparison(
                fp, all_stripes, center, m.rotation_k,
                aerial_bgr=aerial_bgr,
                detection_overlay_bgr=det_overlay,
            )
            if comp is not None:
                safe_name = m.footprint_name.replace("/", "_").replace(" ", "_")
                cv2.imwrite(str(out_dir / f"{safe_name}.png"), comp)
            print(f"  NEW: {m.footprint_name}  pads={m.fp_counts}  "
                  f"cw={cw_counts}  IoU={m.iou:.3f}  ({lat:.5f},{lon:.5f})")

        if (i + 1) % 50 == 0:
            print(f"  ... processed {i+1}/{len(candidates)}, "
                  f"{new_matches} new matches so far")

    mark_area_scanned(cache, label)
    save_cache(cache, cache_path)
    print(f"\n{label}: {new_matches} new matches found")
    print(progress_summary(cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
