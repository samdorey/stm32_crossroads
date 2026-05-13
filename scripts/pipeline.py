"""End-to-end: OSM filter -> imagery fetch -> stripe detection -> PCB scoring.

Reads a candidates JSON written by `find_intersections.py`, fetches imagery
for each candidate from the chosen source, runs the detector + scorer, and
writes a sorted JSON + an HTML thumbnail gallery to make ranking easy to
eyeball.

Usage:
    MAPBOX_TOKEN=pk.xxx python scripts/pipeline.py \
        --candidates data/candidates/sf-soma.json \
        --source mapbox_satellite --zoom 19 --size 768 \
        --top 50
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosswalk_pcb.detect import (
    draw_overlay,
    extract_stripes,
    find_paint_mask,
    group_stripes_into_arrays,
)
from crosswalk_pcb.imagery import SOURCES, fetch_centered, stable_intersection_id
from crosswalk_pcb.score import score_match


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


HTML_HEADER = """<!doctype html>
<html><head><meta charset="utf-8"><title>Crosswalk-PCB ranking</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 16px; background: #111; color: #ddd; }
h1 { font-size: 18px; }
.row { display: flex; gap: 12px; margin: 16px 0; border-bottom: 1px solid #333; padding-bottom: 16px; }
.row img { width: 320px; height: auto; image-rendering: pixelated; }
.meta { font-size: 13px; line-height: 1.5; }
.meta .score { font-size: 22px; font-weight: bold; color: #ffd166; }
.meta a { color: #6cf; }
.meta code { background: #222; padding: 2px 4px; border-radius: 3px; }
</style></head><body>
<h1>Crosswalk-PCB candidates (ranked by similarity)</h1>
"""

HTML_FOOTER = "</body></html>\n"


def _osm_link(lat: float, lon: float, zoom: int = 19) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="JSON written by find_intersections.py")
    ap.add_argument("--source", default="mapbox_satellite", choices=sorted(SOURCES.keys()))
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--size", type=int, default=768,
                    help="output image size in source's native pixel grid")
    ap.add_argument("--top", type=int, default=100,
                    help="process the first N OSM-ranked candidates")
    ap.add_argument("--out", default="data/pipeline_out")
    ap.add_argument("--tile-cache", default="data/tile_cache")
    args = ap.parse_args()

    src = SOURCES[args.source]
    args.zoom = min(args.zoom, src.max_zoom)

    with open(args.candidates) as f:
        data = json.load(f)
    candidates = data["candidates"][: args.top]
    print(f"running {len(candidates)} candidates through source={args.source} z={args.zoom}")

    out_dir = Path(args.out)
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    tile_cache = Path(args.tile_cache)

    results = []
    for cand in candidates:
        lon, lat = cand["lon"], cand["lat"]
        sid = stable_intersection_id(lon, lat)
        try:
            res = fetch_centered(args.source, lon=lon, lat=lat, zoom=args.zoom,
                                 size_px=args.size, cache_dir=tile_cache)
        except Exception as e:
            print(f"  {sid} {lat:.5f},{lon:.5f}: fetch FAIL: {e!r}")
            continue
        bgr = pil_to_bgr(res.image)
        try:
            mask = find_paint_mask(bgr)
            stripes = extract_stripes(mask, m_per_px=res.m_per_px)
            arrays = group_stripes_into_arrays(stripes)
            match = score_match(arrays, image_center=(bgr.shape[1] / 2, bgr.shape[0] / 2))
        except Exception:
            print(f"  {sid} detect/score crashed:\n{traceback.format_exc()}")
            continue

        # Persist thumb + overlay so the HTML report can show them.
        thumb_path = out_dir / "thumbs" / f"{sid}.jpg"
        overlay_path = out_dir / "overlays" / f"{sid}.jpg"
        bgr_to_pil(bgr).save(thumb_path, quality=85)
        bgr_to_pil(draw_overlay(bgr, arrays)).save(overlay_path, quality=85)

        rec = {
            "osm_node_id": cand["osm_node_id"],
            "lat": lat, "lon": lon,
            "stable_id": sid,
            "osm_perpendicularity": cand["perpendicularity"],
            "n_legs_osm": cand["n_legs"],
            "n_legs_with_crossing_osm": cand["n_legs_with_crossing"],
            "n_stripes_detected": len(stripes),
            "n_arrays_detected": len(arrays),
            "match": match.as_dict(),
            "thumb": str(thumb_path.relative_to(out_dir)),
            "overlay": str(overlay_path.relative_to(out_dir)),
            "m_per_px": round(res.m_per_px, 3),
        }
        results.append(rec)
        print(f"  {sid} {lat:.5f},{lon:.5f}  "
              f"sim={match.similarity:.3f}  class={match.package_class}  "
              f"stripes={len(stripes)}  arrays={len(arrays)}")

    results.sort(key=lambda r: r["match"]["similarity"], reverse=True)

    (out_dir / "ranked.json").write_text(json.dumps(results, indent=2))
    html_path = out_dir / "index.html"
    with html_path.open("w") as f:
        f.write(HTML_HEADER)
        for r in results:
            sim = r["match"]["similarity"]
            cls = r["match"]["package_class"]
            f.write('<div class="row">\n')
            f.write(f'  <img src="{r["thumb"]}" alt="thumb"/>\n')
            f.write(f'  <img src="{r["overlay"]}" alt="overlay"/>\n')
            f.write('  <div class="meta">\n')
            f.write(f'    <div class="score">{sim:.2f}</div>\n')
            f.write(f'    <div><b>class:</b> {cls}</div>\n')
            f.write(f'    <div><b>OSM node:</b> {r["osm_node_id"]} ')
            f.write(f'(<a href="{_osm_link(r["lat"], r["lon"])}">map</a>)</div>\n')
            f.write(f'    <div>stripes={r["n_stripes_detected"]} arrays={r["n_arrays_detected"]}</div>\n')
            f.write(f'    <div>sides={r["match"]["n_sides"]} '
                    f'pitch_reg={r["match"]["pitch_regularity"]:.2f} '
                    f'aspect={r["match"]["body_aspect_ratio"]:.2f}</div>\n')
            f.write(f'    <div><code>{r["lat"]:.5f},{r["lon"]:.5f}</code>  '
                    f'{r["m_per_px"]} m/px</div>\n')
            f.write('  </div>\n')
            f.write('</div>\n')
        f.write(HTML_FOOTER)

    print(f"\nwrote {len(results)} results -> {out_dir/'ranked.json'}")
    print(f"open file://{html_path.resolve()} to browse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
