"""Fetch a handful of known SF intersections from each available imagery
source and write a labeled side-by-side strip per intersection.

Useful for deciding which source(s) actually resolve crosswalk stripes well
enough for detection. Run this once, eyeball the output, pick a primary.

Sources that need a token (e.g. Mapbox) are skipped if the env var is unset.

Usage:
    MAPBOX_TOKEN=pk.xxx python scripts/compare_imagery.py
    MAPBOX_TOKEN=pk.xxx python scripts/compare_imagery.py \
        --zoom 19 --size 768 --sources mapbox_satellite esri_world
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosswalk_pcb.imagery import SOURCES, fetch_centered

# Hand-picked SF intersections expected to have continental ("ladder") crosswalks.
# Lat/lon are approximate; adjust if you want pixel-perfect centering.
TEST_INTERSECTIONS = [
    ("market_4th",        37.7843, -122.4072, "Market & 4th (downtown)"),
    ("folsom_8th",        37.7762, -122.4099, "Folsom & 8th (SOMA, rotated grid)"),
    ("mission_24th",      37.7522, -122.4184, "Mission & 24th (BART)"),
    ("lincoln_19th_ave",  37.7649, -122.4761, "Lincoln & 19th Ave (Sunset)"),
    ("california_fillmore", 37.7878, -122.4344, "California & Fillmore (Pac Hts)"),
]


def _label_image(img: Image.Image, lines: list[str]) -> Image.Image:
    """Stamp a small label block on the top-left of img."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    line_h = 14
    pad = 4
    box_h = line_h * len(lines) + 2 * pad
    box_w = max(d.textlength(line) for line in lines) + 2 * pad
    d.rectangle((0, 0, int(box_w), int(box_h)), fill="black")
    for i, line in enumerate(lines):
        d.text((pad, pad + i * line_h), line, fill="white")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=19,
                    help="tile zoom level (Mapbox @2x effectively halves m/px at this zoom)")
    ap.add_argument("--size", type=int, default=512,
                    help="output image size (px, in source's native pixel grid)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="subset of source names to try; default = all that have credentials")
    ap.add_argument("--out", default="data/imagery_compare")
    ap.add_argument("--cache", default="data/tile_cache")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)

    # Pick sources: explicit list or "everything we have creds for".
    candidates = list(SOURCES.keys()) if args.sources is None else list(args.sources)
    usable: list[str] = []
    for name in candidates:
        if name not in SOURCES:
            print(f"  skip {name}: unknown source", file=sys.stderr)
            continue
        s = SOURCES[name]
        if s.token_env is not None and not os.environ.get(s.token_env):
            print(f"  skip {name}: env var {s.token_env} not set")
            continue
        usable.append(name)

    if not usable:
        print("no usable sources; set MAPBOX_TOKEN or use --sources with public sources",
              file=sys.stderr)
        return 1
    print(f"using sources: {usable}")

    for slug, lat, lon, label in TEST_INTERSECTIONS:
        print(f"\n{slug}  ({lat:.4f}, {lon:.4f})  — {label}")
        thumbs: list[Image.Image] = []
        for name in usable:
            s = SOURCES[name]
            z = min(args.zoom, s.max_zoom)
            try:
                res = fetch_centered(
                    name, lon=lon, lat=lat, zoom=z,
                    size_px=args.size, cache_dir=cache_dir,
                )
            except Exception as e:
                print(f"  {name}: FAIL {e!r}")
                continue
            print(f"  {name}: z={res.zoom} {res.m_per_px:.3f} m/px "
                  f"{res.image.size[0]}x{res.image.size[1]}")
            stamped = _label_image(
                res.image,
                [name, f"z={res.zoom} {res.m_per_px:.2f} m/px"],
            )
            thumbs.append(stamped)
        if not thumbs:
            continue
        # Horizontal strip, one source per panel.
        w = sum(t.width for t in thumbs)
        h = max(t.height for t in thumbs)
        strip = Image.new("RGB", (w, h + 24), "white")
        d = ImageDraw.Draw(strip)
        d.text((6, 6), f"{label}  ({lat:.4f}, {lon:.4f})", fill="black")
        x = 0
        for t in thumbs:
            strip.paste(t, (x, 24))
            x += t.width
        path = out_dir / f"{slug}.png"
        strip.save(path)
        print(f"  -> {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
