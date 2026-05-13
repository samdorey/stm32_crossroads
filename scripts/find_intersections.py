"""Query OSM (Overpass) for candidate intersections in a bbox, filter by
perpendicularity + presence of marked crossings, and write a JSON list.

Usage:
    python scripts/find_intersections.py --area sf-soma
    python scripts/find_intersections.py --bbox 37.748,-122.430,37.792,-122.395
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosswalk_pcb.osm import (
    build_query,
    extract_junctions,
    rank_candidates,
    run_overpass,
)

# Pre-defined bboxes (south, west, north, east). Add more as needed.
AREAS: dict[str, tuple[float, float, float, float]] = {
    # Small, dense bbox useful for first-pass testing.
    "sf-soma":      (37.770, -122.420, 37.790, -122.395),
    "sf-mission":   (37.745, -122.430, 37.770, -122.405),
    "sf-sunset":    (37.755, -122.510, 37.775, -122.470),
    "sf-richmond":  (37.770, -122.510, 37.790, -122.470),
    # Whole peninsula chunk (slow but exhaustive).
    "sf-all":       (37.708, -122.515, 37.815, -122.355),
}


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 4 comma-separated floats: s,w,n,e")
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--area", choices=sorted(AREAS.keys()))
    g.add_argument("--bbox", type=parse_bbox)
    ap.add_argument("--out", default="data/candidates")
    ap.add_argument("--cache", default="data/overpass_cache",
                    help="dir for caching Overpass responses; speeds re-runs")
    ap.add_argument("--min-perpendicularity", type=float, default=0.5)
    ap.add_argument("--min-legs-with-crossing", type=int, default=2)
    ap.add_argument("--limit", type=int, default=200,
                    help="max number of candidates to write")
    args = ap.parse_args()

    bbox = AREAS[args.area] if args.area else args.bbox
    label = args.area or "custom"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{label}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}.json"

    print(f"bbox={bbox} (cache={cache_path})")
    query = build_query(bbox)
    print("running Overpass query...")
    data = run_overpass(query, cache_path=cache_path)

    junctions = extract_junctions(data)
    print(f"  extracted {len(junctions)} junctions with >=3 legs")

    ranked = rank_candidates(
        junctions,
        min_perpendicularity=args.min_perpendicularity,
        min_legs_with_crossing=args.min_legs_with_crossing,
    )
    print(f"  {len(ranked)} pass filters "
          f"(perp>={args.min_perpendicularity}, legs_with_crossing>={args.min_legs_with_crossing})")

    candidates = []
    for j in ranked[: args.limit]:
        candidates.append({
            "osm_node_id": j.osm_node_id,
            "lon": j.lon, "lat": j.lat,
            "n_legs": j.n_legs,
            "leg_bearings_deg": [round(b, 1) for b in j.leg_bearings],
            "perpendicularity": round(j.perpendicularity_score(), 3),
            "n_legs_with_crossing": j.n_legs_with_crossing(),
            "crossings": [
                {"osm_id": c.osm_id, "lon": c.lon, "lat": c.lat, "type": c.crossing_type}
                for c in j.crossings
            ],
        })

    out_path = out_dir / f"{label}.json"
    out_path.write_text(json.dumps({"bbox": list(bbox), "candidates": candidates}, indent=2))
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
