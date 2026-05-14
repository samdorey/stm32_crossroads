"""Detection parameter tuning workbench.

Runs a fixed set of test intersections through multiple detection configs
and generates an HTML report for visual comparison.

Usage:
    python scripts/detection_workbench.py
    python scripts/detection_workbench.py --out data/workbench
    open data/workbench/index.html
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosswalk_pcb.detect import (
    align_image_to_roads,
    build_road_mask,
    draw_overlay,
    extract_stripes,
    find_paint_mask,
    find_paint_mask_adaptive,
    find_paint_mask_contrast,
    group_stripes_into_arrays,
)
from crosswalk_pcb.imagery import fetch_centered
from crosswalk_pcb.match import crosswalk_side_counts


# ---------- test corpus ----------
# Each entry: (name, lat, lon, approx_bearings, notes)
# Diverse conditions: different cities, paint colors, shadow levels, grid angles.

TEST_INTERSECTIONS = [
    # SF Sunset — yellow paint, clean grid, mild shadow
    ("sf-sunset-good",   37.75885, -122.49939, [1.3, 91.5, 181.3, 271.5],
     "Yellow paint, 4-sided, previously scored 0.75"),
    ("sf-sunset-tssop",  37.77291, -122.47671, [86.7, 266.7, 356.7],
     "Yellow paint, 2-sided T-junction, TSSOP-8 match"),
    # SF Polk Gulch — white paint, slight grid rotation
    ("sf-polk-0",        37.79151, -122.42089, [80.9, 170.9, 260.9, 350.9],
     "White paint, 4-sided, ~9deg rotation"),
    # SF SOMA — diagonal grid, heavy shadow
    ("sf-soma-diag",     37.77623, -122.41472, [35.0, 125.0, 215.0, 305.0],
     "Diagonal grid ~45deg, heavy building shadows"),
    # Detroit — bright concrete, white paint, N-S grid
    ("det-midtown-0",    42.33692, -83.04781,  [0.3, 90.3, 180.3, 270.3],
     "Very bright concrete, white paint, axis-aligned"),
    ("det-midtown-3",    42.33367, -83.05077,  [0.0, 90.0, 180.0, 270.0],
     "Bright concrete, one-sided detection issue"),
    # NYC Brooklyn — rotated grid, some shadow
    ("nyc-bk-0",         40.68061, -73.98096,  [62.8, 152.8, 242.8, 332.8],
     "White paint, ~27deg grid rotation, moderate shadow"),
]


# ---------- detection configs ----------

@dataclass
class DetectConfig:
    name: str
    description: str
    mask_fn_name: str  # 'global', 'adaptive', 'contrast'
    mask_kwargs: dict = field(default_factory=dict)
    use_road_mask: bool = False
    extract_kwargs: dict = field(default_factory=dict)
    group_kwargs: dict = field(default_factory=dict)


CONFIGS = [
    DetectConfig(
        name="baseline",
        description="Global threshold, no road mask (original defaults)",
        mask_fn_name="global",
    ),
    DetectConfig(
        name="baseline+road",
        description="Global threshold + road mask",
        mask_fn_name="global",
        use_road_mask=True,
    ),
    DetectConfig(
        name="global-relaxed",
        description="Global with lower V threshold (160 instead of 180)",
        mask_fn_name="global",
        mask_kwargs={"min_value": 160, "yellow_min_val": 150},
        use_road_mask=True,
    ),
    DetectConfig(
        name="adaptive",
        description="Adaptive local contrast (bm=30) + road mask",
        mask_fn_name="adaptive",
        use_road_mask=True,
    ),
    DetectConfig(
        name="adaptive-loose",
        description="Adaptive with lower margin (bm=22) + road mask",
        mask_fn_name="adaptive",
        mask_kwargs={"brightness_margin": 22.0},
        use_road_mask=True,
    ),
    DetectConfig(
        name="contrast",
        description="Color-contrast (paint-on-asphalt) + road mask",
        mask_fn_name="contrast",
        mask_kwargs={"paint_min_val": 140, "min_asphalt_frac": 0.5},
        use_road_mask=True,
    ),
    DetectConfig(
        name="wider-stripes",
        description="Global+road, wider stripe size range (0.2-1.5m width, 1.0-8m length)",
        mask_fn_name="global",
        use_road_mask=True,
        extract_kwargs={
            "min_stripe_width_m": 0.20,
            "max_stripe_width_m": 1.5,
            "min_stripe_len_m": 1.0,
            "max_stripe_len_m": 8.0,
            "min_aspect_ratio": 2.0,
        },
    ),
    DetectConfig(
        name="loose-grouping",
        description="Global+road, relaxed grouping (angle_tol=20, perp_tol=30, min_count=2)",
        mask_fn_name="global",
        use_road_mask=True,
        group_kwargs={
            "angle_tol_deg": 20.0,
            "perp_tol_px": 30.0,
            "min_count": 2,
        },
    ),
]


# ---------- run one intersection × one config ----------

MASK_FNS = {
    "global": find_paint_mask,
    "adaptive": find_paint_mask_adaptive,
    "contrast": find_paint_mask_contrast,
}


def run_config(
    bgr: np.ndarray,
    bearings: list[float],
    m_per_px: float,
    config: DetectConfig,
) -> dict:
    """Run a detection config on an aligned image. Returns stats + images."""
    center = (bgr.shape[1] / 2, bgr.shape[0] / 2)

    road_mask = None
    if config.use_road_mask and bearings:
        road_mask = build_road_mask(
            bgr.shape, bearings, center,
            road_half_width_px=12.0 / m_per_px,
            length_px=bgr.shape[0] * 0.45,
            margin_px=5.0 / m_per_px,
            center_radius_px=20.0 / m_per_px,
        )

    mask_fn = MASK_FNS[config.mask_fn_name]
    mask = mask_fn(bgr, **config.mask_kwargs)
    if road_mask is not None:
        mask = cv2.bitwise_and(mask, road_mask)

    stripes = extract_stripes(mask, m_per_px=m_per_px, **config.extract_kwargs)
    arrays = group_stripes_into_arrays(stripes, **config.group_kwargs)
    counts = crosswalk_side_counts(arrays, center)
    all_stripes = [s for a in arrays for s in a.stripes]

    overlay = draw_overlay(bgr, arrays)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return {
        "n_stripes": len(stripes),
        "n_in_arrays": len(all_stripes),
        "n_arrays": len(arrays),
        "counts": counts,
        "overlay": overlay,
        "mask": mask_bgr,
    }


# ---------- HTML report ----------

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Detection Workbench</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 12px; background: #111; color: #ddd; }
h1 { font-size: 18px; }
h2 { font-size: 14px; color: #aaa; margin-top: 28px; border-top: 1px solid #333; padding-top: 8px; }
table { border-collapse: collapse; margin: 8px 0; }
th, td { border: 1px solid #333; padding: 4px 6px; text-align: center; vertical-align: top; font-size: 12px; }
th { background: #222; }
td img { width: 220px; display: block; margin: 2px auto; image-rendering: pixelated; }
.good { color: #6f6; }
.bad { color: #f66; }
.cfg-desc { font-size: 10px; color: #888; max-width: 220px; }
</style></head><body>
<h1>Detection Workbench</h1>
"""

HTML_FOOT = "</body></html>\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/workbench")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--zoom", type=int, default=19)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    tile_cache = Path("data/tile_cache")

    html_parts = [HTML_HEAD]

    for ti, (name, lat, lon, bearings, notes) in enumerate(TEST_INTERSECTIONS):
        print(f"\n=== {name} ({lat:.5f}, {lon:.5f}) ===")
        print(f"    {notes}")

        try:
            res = fetch_centered(
                "mapbox_satellite", lon=lon, lat=lat, zoom=args.zoom,
                size_px=args.size, cache_dir=tile_cache,
            )
        except Exception as e:
            print(f"    FETCH FAIL: {e!r}")
            continue

        bgr = cv2.cvtColor(np.array(res.image.convert("RGB")), cv2.COLOR_RGB2BGR)
        bgr, rot = align_image_to_roads(bgr, bearings)
        m_per_px = res.m_per_px

        # Save aerial.
        aerial_path = f"imgs/{name}_aerial.jpg"
        cv2.imwrite(str(out / aerial_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        html_parts.append(f'<h2>{name} — {notes} (rot={rot:.1f}°)</h2>\n')
        html_parts.append("<table><tr><th>aerial</th>")
        for cfg in CONFIGS:
            html_parts.append(f"<th>{cfg.name}</th>")
        html_parts.append("</tr>\n")

        # Row 1: overlay images
        html_parts.append(f'<tr><td><img src="{aerial_path}"/></td>')
        for cfg in CONFIGS:
            result = run_config(bgr, bearings, m_per_px, cfg)
            img_name = f"imgs/{name}_{cfg.name}_overlay.jpg"
            cv2.imwrite(str(out / img_name), result["overlay"],
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            c = result["counts"]
            n = result["n_in_arrays"]
            cls = "good" if n >= 6 else "bad" if n == 0 else ""
            html_parts.append(
                f'<td><img src="{img_name}"/>'
                f'<div class="{cls}">{n} in arrays</div>'
                f'<div>counts={c}</div></td>'
            )
            print(f"    {cfg.name:20s}  arr={n:2d}  counts={c}")
        html_parts.append("</tr>\n")

        # Row 2: mask images
        html_parts.append(f'<tr><td></td>')
        for cfg in CONFIGS:
            result = run_config(bgr, bearings, m_per_px, cfg)
            mask_name = f"imgs/{name}_{cfg.name}_mask.jpg"
            cv2.imwrite(str(out / mask_name), result["mask"],
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            html_parts.append(f'<td><img src="{mask_name}"/>'
                              f'<div class="cfg-desc">{cfg.description}</div></td>')
        html_parts.append("</tr></table>\n")

    html_parts.append(HTML_FOOT)
    html_path = out / "index.html"
    html_path.write_text("".join(html_parts))
    print(f"\n-> {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
