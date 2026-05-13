"""Generate a synthetic 'QFP-like' intersection image (4 continental crosswalks
arranged around a square road body), run it through detect + score, and
verify the pipeline produces high similarity. Used for offline development.

Usage:
    python -m scripts.synthetic_test               # default: writes data/synthetic/*.png
    python -m scripts.synthetic_test --quiet       # just exit 0/1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as 'python scripts/synthetic_test.py' or '-m scripts.synthetic_test'.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crosswalk_pcb.detect import (
    draw_overlay,
    extract_stripes,
    find_paint_mask,
    group_stripes_into_arrays,
)
from crosswalk_pcb.score import score_match


def _draw_stripe(canvas: np.ndarray, cx: float, cy: float,
                 length_px: float, width_px: float, angle_deg: float,
                 color=(245, 245, 245)) -> None:
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    u = np.array([cos_a, sin_a])
    v = np.array([-sin_a, cos_a])
    hu, hv = length_px / 2, width_px / 2
    c = np.array([cx, cy])
    pts = np.array(
        [c + hu * u + hv * v, c - hu * u + hv * v,
         c - hu * u - hv * v, c + hu * u - hv * v],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [pts], color)


def make_synthetic_intersection(
    size_px: int = 512,
    m_per_px: float = 0.15,
    pad_count: int = 8,
    stripe_w_m: float = 0.6,
    stripe_l_m: float = 3.0,
    stripe_pitch_m: float = 1.2,
    leg_offset_m: float = 12.0,
    seed: int = 0,
    add_road_body: bool = True,
    body_aspect: float = 1.0,
) -> np.ndarray:
    """Build a synthetic top-down intersection with continental crosswalks on
    all four sides. Returns a BGR uint8 image."""
    rng = np.random.default_rng(seed)
    # Asphalt: dark gray with mild speckle.
    canvas = rng.integers(60, 95, size=(size_px, size_px, 3), dtype=np.uint8)

    cx, cy = size_px / 2, size_px / 2
    leg_offset_px = leg_offset_m / m_per_px
    stripe_w_px = stripe_w_m / m_per_px
    stripe_l_px = stripe_l_m / m_per_px
    pitch_px = stripe_pitch_m / m_per_px

    # Optional: lighter "road body" rectangles, so the area between the
    # crosswalks looks like the central road space rather than asphalt
    # texture. Helps stress-test paint thresholding.
    if add_road_body:
        road_half = leg_offset_px + stripe_l_px + 5
        road_half_y = road_half * body_aspect
        cv2.rectangle(
            canvas,
            (int(cx - road_half), int(cy - 8)),
            (int(cx + road_half), int(cy + 8)),
            (78, 78, 78), -1,
        )
        cv2.rectangle(
            canvas,
            (int(cx - 8), int(cy - road_half_y)),
            (int(cx + 8), int(cy + road_half_y)),
            (78, 78, 78), -1,
        )

    # Helper: draw an array of `pad_count` stripes centered at (acx, acy),
    # arrayed along `array_axis_deg`, stripes oriented at `stripe_angle_deg`.
    def draw_array(acx, acy, array_axis_deg, stripe_angle_deg):
        ax = math.radians(array_axis_deg)
        ux, uy = math.cos(ax), math.sin(ax)
        half = (pad_count - 1) / 2.0
        for i in range(pad_count):
            offset = (i - half) * pitch_px
            sx = acx + offset * ux
            sy = acy + offset * uy
            _draw_stripe(canvas, sx, sy, stripe_l_px, stripe_w_px,
                         stripe_angle_deg)

    # 4 sides. Coordinates: x grows right, y grows down. body_aspect > 1
    # pushes the N/S crosswalks further from center, producing a rectangular body.
    ns_offset = leg_offset_px * body_aspect
    ew_offset = leg_offset_px
    # North (top): stripes oriented vertically (90°), arrayed horizontally (0°).
    draw_array(cx, cy - ns_offset, array_axis_deg=0.0, stripe_angle_deg=90.0)
    # South (bottom).
    draw_array(cx, cy + ns_offset, array_axis_deg=0.0, stripe_angle_deg=90.0)
    # West (left): stripes oriented horizontally (0°), arrayed vertically (90°).
    draw_array(cx - ew_offset, cy, array_axis_deg=90.0, stripe_angle_deg=0.0)
    # East (right).
    draw_array(cx + ew_offset, cy, array_axis_deg=90.0, stripe_angle_deg=0.0)

    return canvas


def make_t_intersection(**kw) -> np.ndarray:
    """3-leg T variant: same as the QFP synthetic, minus one side. Used to
    verify the scorer differentiates QFP-class from incomplete patterns."""
    size_px = kw.get("size_px", 512)
    m_per_px = kw.get("m_per_px", 0.15)
    img = make_synthetic_intersection(**kw)
    # Wipe the north side by overlaying asphalt.
    cy = size_px / 2
    leg_offset_px = kw.get("leg_offset_m", 12.0) / m_per_px
    stripe_l_px = kw.get("stripe_l_m", 3.0) / m_per_px
    cv2.rectangle(
        img,
        (0, 0),
        (size_px, int(cy - leg_offset_px + stripe_l_px / 2)),
        (75, 75, 75), -1,
    )
    return img


def run_one(image: np.ndarray, m_per_px: float) -> dict:
    """Run the full detect→group→score pipeline on one image and return stats."""
    mask = find_paint_mask(image)
    stripes = extract_stripes(mask, m_per_px=m_per_px)
    arrays = group_stripes_into_arrays(stripes)
    h, w = image.shape[:2]
    match = score_match(arrays, image_center=(w / 2, h / 2))
    return {
        "n_stripes": len(stripes),
        "n_arrays": len(arrays),
        "match": match,
        "mask": mask,
        "arrays": arrays,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic", help="output dir")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_per_px = 0.15

    # --- Case 1: ideal QFP-like intersection.
    qfp = make_synthetic_intersection(m_per_px=m_per_px, pad_count=8, seed=1)
    res_qfp = run_one(qfp, m_per_px)

    # --- Case 2: T-intersection (only 3 sides).
    t3 = make_t_intersection(m_per_px=m_per_px, pad_count=8, seed=2)
    res_t = run_one(t3, m_per_px)

    # --- Case 3: rectangular body (different leg distances).
    skew = make_synthetic_intersection(m_per_px=m_per_px, pad_count=8, seed=3, body_aspect=1.8)
    res_skew = run_one(skew, m_per_px)

    # Persist annotated images for visual inspection.
    cv2.imwrite(str(out_dir / "qfp_input.png"), qfp)
    cv2.imwrite(str(out_dir / "qfp_overlay.png"), draw_overlay(qfp, res_qfp["arrays"]))
    cv2.imwrite(str(out_dir / "t3_input.png"), t3)
    cv2.imwrite(str(out_dir / "t3_overlay.png"), draw_overlay(t3, res_t["arrays"]))
    cv2.imwrite(str(out_dir / "skew_input.png"), skew)
    cv2.imwrite(str(out_dir / "skew_overlay.png"), draw_overlay(skew, res_skew["arrays"]))

    report = {
        "qfp": {**{k: v for k, v in res_qfp.items() if k in ("n_stripes", "n_arrays")},
                "match": res_qfp["match"].as_dict()},
        "t3": {**{k: v for k, v in res_t.items() if k in ("n_stripes", "n_arrays")},
               "match": res_t["match"].as_dict()},
        "skew": {**{k: v for k, v in res_skew.items() if k in ("n_stripes", "n_arrays")},
                 "match": res_skew["match"].as_dict()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    if not args.quiet:
        for name, r in report.items():
            print(f"\n=== {name} ===")
            print(f"  stripes detected: {r['n_stripes']}")
            print(f"  arrays found:     {r['n_arrays']}")
            print(f"  match:            {r['match']['summary']}")
            print(f"  similarity:       {r['match']['similarity']:.3f}")

    # Pass/fail checks. These are the "did the detection logic actually work"
    # gates we want to enforce on every change.
    qfp_match = res_qfp["match"]
    t_match = res_t["match"]
    ok = True
    if qfp_match.n_sides < 4:
        print(f"FAIL: QFP case found only {qfp_match.n_sides} sides", file=sys.stderr)
        ok = False
    if qfp_match.similarity < 0.45:
        print(f"FAIL: QFP similarity too low: {qfp_match.similarity:.3f}", file=sys.stderr)
        ok = False
    if qfp_match.package_class != "QFP":
        print(f"FAIL: QFP case classified as {qfp_match.package_class}", file=sys.stderr)
        ok = False
    if t_match.similarity >= qfp_match.similarity:
        print(f"FAIL: T-intersection sim {t_match.similarity:.3f} "
              f">= QFP sim {qfp_match.similarity:.3f}", file=sys.stderr)
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
