"""Match crosswalk intersections against KiCad footprints via image comparison.

Both footprint pads and crosswalk stripes are rendered as thin marks on a
normalized canvas. Each element keeps its CENTER position and LONG-AXIS
length (scaled to fit), but the SHORT-AXIS width is fixed to a constant
pixel thickness. This focuses the IoU comparison on the spatial pattern —
number of marks, spacing, distribution along edges, body aspect — rather
than the duty cycle (pad width / pitch vs stripe width / pitch).

We try 4 rotations (0/90/180/270) of the footprint to find the best
alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .detect import Stripe, StripeArray
from .kicad import Footprint, Pad

CANVAS = 512
MARGIN = 0.10
MARK_PX = 5  # fixed short-axis width in rendered pixels


# ---------- rendering ----------

def _fit_to_canvas(
    xs: list[float], ys: list[float],
    canvas: int, margin: float,
) -> tuple[float, float, float]:
    """Compute (scale, off_x, off_y) that fits a set of points into the
    canvas with margin. Scale is uniform (preserves aspect ratio)."""
    if not xs:
        return 1.0, 0.0, 0.0
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = x_max - x_min or 1e-6
    span_y = y_max - y_min or 1e-6
    usable = canvas * (1.0 - 2 * margin)
    scale = min(usable / span_x, usable / span_y)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    off_x = canvas / 2 - cx * scale
    off_y = canvas / 2 - cy * scale
    return scale, off_x, off_y


def _draw_mark(
    img: np.ndarray,
    cx: float, cy: float,
    long_dim: float, angle_deg: float,
    scale: float, off_x: float, off_y: float,
    mark_px: int = MARK_PX,
) -> None:
    """Draw a thin filled rectangle: long axis at `angle_deg`, short axis
    fixed to `mark_px` pixels regardless of scale."""
    px = cx * scale + off_x
    py = cy * scale + off_y
    pl = long_dim * scale       # long axis in pixels
    pw = float(mark_px)         # short axis fixed

    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    u = np.array([cos_a, sin_a])  # unit vector along long axis
    v = np.array([-sin_a, cos_a]) # unit vector along short axis
    c = np.array([px, py])
    hl, hw = pl / 2, pw / 2
    pts = np.array([
        c + hl * u + hw * v,
        c - hl * u + hw * v,
        c - hl * u - hw * v,
        c + hl * u - hw * v,
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)


def _pad_long_axis(p: Pad) -> tuple[float, float]:
    """Return (long_dimension_mm, angle_deg) for a pad, accounting for
    the pad's KiCad rotation. The long axis is max(width, height); if
    width >= height the base angle is 0° (horizontal), else 90° (vertical).
    The pad's (at ... ANGLE) rotation is added on top."""
    if p.width >= p.height:
        return p.width, 0.0 + p.angle
    else:
        return p.height, 90.0 + p.angle


def render_footprint(fp: Footprint, canvas: int = CANVAS) -> np.ndarray:
    """Render footprint pads as thin marks on a black canvas.

    Each pad's long axis is rendered at scale, accounting for the pad's
    rotation from its (at X Y ANGLE) attribute. The short axis is
    replaced with MARK_PX."""
    img = np.zeros((canvas, canvas), dtype=np.uint8)
    # Fit based on pad centers + half the long-axis extent (rotated).
    extents_x: list[float] = []
    extents_y: list[float] = []
    for p in fp.pads:
        long_dim, angle = _pad_long_axis(p)
        hl = long_dim / 2
        cos_a = abs(math.cos(math.radians(angle)))
        sin_a = abs(math.sin(math.radians(angle)))
        extents_x += [p.x - hl * cos_a, p.x + hl * cos_a]
        extents_y += [p.y - hl * sin_a, p.y + hl * sin_a]
    scale, off_x, off_y = _fit_to_canvas(extents_x, extents_y, canvas, MARGIN)

    for p in fp.pads:
        long_dim, angle = _pad_long_axis(p)
        _draw_mark(img, p.x, p.y, long_dim, angle, scale, off_x, off_y)
    return img


def _fp_transform(fp: Footprint, canvas: int = CANVAS) -> tuple[float, float, float]:
    """Compute the fit transform for a footprint (shared by render + label fns)."""
    extents_x: list[float] = []
    extents_y: list[float] = []
    for p in fp.pads:
        long_dim, angle = _pad_long_axis(p)
        hl = long_dim / 2
        cos_a = abs(math.cos(math.radians(angle)))
        sin_a = abs(math.sin(math.radians(angle)))
        extents_x += [p.x - hl * cos_a, p.x + hl * cos_a]
        extents_y += [p.y - hl * sin_a, p.y + hl * sin_a]
    return _fit_to_canvas(extents_x, extents_y, canvas, MARGIN)


def _stripe_transform(stripes: list[Stripe], canvas: int = CANVAS) -> tuple[float, float, float]:
    """Compute the fit transform for stripes (shared by render + label fns)."""
    extents_x: list[float] = []
    extents_y: list[float] = []
    for s in stripes:
        cos_a = math.cos(math.radians(s.angle_deg))
        sin_a = math.sin(math.radians(s.angle_deg))
        hl = s.length_px / 2
        extents_x += [s.cx - hl * abs(cos_a), s.cx + hl * abs(cos_a)]
        extents_y += [s.cy - hl * abs(sin_a), s.cy + hl * abs(sin_a)]
    return _fit_to_canvas(extents_x, extents_y, canvas, MARGIN)


def render_stripes(stripes: list[Stripe], canvas: int = CANVAS) -> np.ndarray:
    """Render crosswalk stripes as thin marks on a black canvas."""
    img = np.zeros((canvas, canvas), dtype=np.uint8)
    if not stripes:
        return img
    scale, off_x, off_y = _stripe_transform(stripes, canvas)
    for s in stripes:
        _draw_mark(img, s.cx, s.cy, s.length_px, s.angle_deg,
                   scale, off_x, off_y)
    return img


# ---------- labeled rendering for 1:1 matching ----------

def _render_single_mark(
    canvas_size: int, cx: float, cy: float, long_dim: float,
    angle_deg: float, scale: float, off_x: float, off_y: float,
) -> np.ndarray:
    """Render a single mark as a binary mask."""
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    _draw_mark(mask, cx, cy, long_dim, angle_deg, scale, off_x, off_y)
    return mask


def render_footprint_masks(
    fp: Footprint, canvas: int = CANVAS, rotation_k: int = 0,
) -> list[np.ndarray]:
    """Render each pad as a separate binary mask. Applies rotation_k * 90° CCW."""
    scale, off_x, off_y = _fp_transform(fp, canvas)
    masks = []
    for p in fp.pads:
        long_dim, angle = _pad_long_axis(p)
        m = _render_single_mark(canvas, p.x, p.y, long_dim, angle,
                                scale, off_x, off_y)
        if rotation_k:
            m = np.rot90(m, rotation_k)
        masks.append(m)
    return masks


def render_stripe_masks(
    stripes: list[Stripe], canvas: int = CANVAS,
) -> list[np.ndarray]:
    """Render each stripe as a separate binary mask."""
    if not stripes:
        return []
    scale, off_x, off_y = _stripe_transform(stripes, canvas)
    masks = []
    for s in stripes:
        m = _render_single_mark(canvas, s.cx, s.cy, s.length_px,
                                s.angle_deg, scale, off_x, off_y)
        masks.append(m)
    return masks


def check_1to1(
    fp: Footprint,
    stripes: list[Stripe],
    rotation_k: int,
    canvas: int = CANVAS,
) -> tuple[bool, list[tuple[int, int]]]:
    """Check if every pad has exactly one overlapping stripe and no stripe
    is shared by multiple pads.

    Returns (is_valid, pairs) where pairs is a list of (pad_idx, stripe_idx)
    for successful mappings."""
    pad_masks = render_footprint_masks(fp, canvas, rotation_k)
    stripe_masks = render_stripe_masks(stripes, canvas)

    if not pad_masks or not stripe_masks:
        return False, []

    # Build overlap matrix: overlap[i][j] = True if pad i overlaps stripe j.
    n_pads = len(pad_masks)
    n_stripes = len(stripe_masks)
    overlap = np.zeros((n_pads, n_stripes), dtype=bool)
    for i, pm in enumerate(pad_masks):
        pm_bool = pm > 0
        for j, sm in enumerate(stripe_masks):
            if np.any(pm_bool & (sm > 0)):
                overlap[i, j] = True

    # Check: every pad must overlap exactly one stripe.
    pairs: list[tuple[int, int]] = []
    for i in range(n_pads):
        matched = np.where(overlap[i])[0]
        if len(matched) != 1:
            return False, []
        pairs.append((i, int(matched[0])))

    # Check: no stripe matched by more than one pad.
    assigned_stripes = [p[1] for p in pairs]
    if len(set(assigned_stripes)) != len(assigned_stripes):
        return False, []

    return True, pairs


# ---------- comparison ----------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union of two binary images."""
    a_bool = a > 127
    b_bool = b > 127
    intersection = np.count_nonzero(a_bool & b_bool)
    union = np.count_nonzero(a_bool | b_bool)
    if union == 0:
        return 0.0
    return intersection / union


def _rotate_90(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate image by k * 90 degrees CCW."""
    return np.rot90(img, k)


def compare(fp_img: np.ndarray, cw_img: np.ndarray) -> tuple[float, int]:
    """Compare footprint and crosswalk images across 4 rotations.
    Returns (best_iou, best_rotation_k)."""
    best_iou = 0.0
    best_k = 0
    for k in range(4):
        rotated = _rotate_90(fp_img, k)
        score = _iou(rotated, cw_img)
        if score > best_iou:
            best_iou = score
            best_k = k
    return best_iou, best_k


# ---------- top-level matching ----------

@dataclass
class MatchResult:
    footprint_name: str
    footprint_path: str
    iou: float
    rotation_k: int
    fp_image: np.ndarray
    cw_image: np.ndarray


def find_best_matches(
    stripes: list[Stripe],
    library: list[Footprint],
    top_n: int = 5,
    canvas: int = CANVAS,
) -> list[MatchResult]:
    """Render the crosswalk once, compare against every footprint by IoU."""
    cw_img = render_stripes(stripes, canvas)
    if np.count_nonzero(cw_img) == 0:
        return []

    results: list[MatchResult] = []
    for fp in library:
        fp_img = render_footprint(fp, canvas)
        if np.count_nonzero(fp_img) == 0:
            continue
        iou, k = compare(fp_img, cw_img)
        results.append(MatchResult(
            footprint_name=fp.name,
            footprint_path=fp.path,
            iou=iou,
            rotation_k=k,
            fp_image=_rotate_90(fp_img, k),
            cw_image=cw_img,
        ))

    results.sort(key=lambda r: -r.iou)
    return results[:top_n]


def render_comparison(match: MatchResult) -> np.ndarray:
    """Side-by-side: footprint | crosswalk | overlay (green=FP, red=CW, white=both)."""
    fp = match.fp_image
    cw = match.cw_image
    h, w = fp.shape

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    fp_bool = fp > 127
    cw_bool = cw > 127
    overlay[fp_bool & ~cw_bool] = (0, 180, 0)
    overlay[cw_bool & ~fp_bool] = (0, 0, 180)
    overlay[fp_bool & cw_bool] = (255, 255, 255)

    fp_bgr = cv2.cvtColor(fp, cv2.COLOR_GRAY2BGR)
    cw_bgr = cv2.cvtColor(cw, cv2.COLOR_GRAY2BGR)
    strip = np.hstack([fp_bgr, cw_bgr, overlay])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(strip, f"FP: {match.footprint_name}", (4, 16), font, 0.45, (0, 200, 200), 1)
    cv2.putText(strip, "crosswalk", (w + 4, 16), font, 0.45, (0, 200, 200), 1)
    cv2.putText(strip, f"IoU={match.iou:.3f}", (2 * w + 4, 16), font, 0.45, (0, 200, 200), 1)

    return strip


# ---------- count-based exact matching ----------

def _classify_pad_side(p: Pad, cx: float, cy: float) -> str:
    """N/E/S/W based on pad center position relative to footprint centroid."""
    dx = p.x - cx
    dy = p.y - cy
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    else:
        return "S" if dy > 0 else "N"


def footprint_side_counts(fp: Footprint) -> tuple[int, int, int, int]:
    """Count pads per side (N, E, S, W) for a footprint."""
    cx = sum(p.x for p in fp.pads) / len(fp.pads)
    cy = sum(p.y for p in fp.pads) / len(fp.pads)
    counts = {"N": 0, "E": 0, "S": 0, "W": 0}
    for p in fp.pads:
        counts[_classify_pad_side(p, cx, cy)] += 1
    return (counts["N"], counts["E"], counts["S"], counts["W"])


def crosswalk_side_counts(
    arrays: list[StripeArray],
    image_center: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Count stripes per side (N, E, S, W) from detected arrays.
    Uses the same quadrant assignment as score.py."""
    counts = {"N": 0, "E": 0, "S": 0, "W": 0}
    for arr in arrays:
        dx = arr.centroid[0] - image_center[0]
        dy = arr.centroid[1] - image_center[1]
        if abs(dy) >= abs(dx):
            side = "N" if dy < 0 else "S"
        else:
            side = "E" if dx > 0 else "W"
        counts[side] += arr.n_stripes
    return (counts["N"], counts["E"], counts["S"], counts["W"])


def _rotate_counts(counts: tuple[int, int, int, int], k: int) -> tuple[int, int, int, int]:
    """Rotate NESW counts by k * 90° CCW.
    k=1: what was East becomes North, South→East, West→South, North→West."""
    n, e, s, w = counts
    for _ in range(k % 4):
        n, e, s, w = e, s, w, n
    return (n, e, s, w)


def _counts_match(
    fp_counts: tuple[int, int, int, int],
    cw_counts: tuple[int, int, int, int],
) -> tuple[bool, int]:
    """Check if footprint counts match crosswalk counts at any rotation.
    Returns (matches, best_rotation_k)."""
    for k in range(4):
        if _rotate_counts(fp_counts, k) == cw_counts:
            return True, k
    return False, 0


@dataclass
class ExactMatch(MatchResult):
    """A match where per-side stripe counts exactly equal pad counts."""
    fp_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    cw_counts: tuple[int, int, int, int] = (0, 0, 0, 0)


def find_exact_matches(
    arrays: list[StripeArray],
    image_center: tuple[float, float],
    library: list[Footprint],
    canvas: int = CANVAS,
) -> list[ExactMatch]:
    """Find footprints whose per-side pad counts exactly match the crosswalk's
    per-side stripe counts (at some rotation). Then rank by IoU.

    This is the strict filter: could this crosswalk literally be used as a
    footprint? Every stripe must correspond 1:1 to a pad spatially.

    Two-stage filter:
      1. Per-side counts must match exactly (at some rotation).
      2. Each pad must overlap exactly one stripe in the rendered image."""
    cw_counts = crosswalk_side_counts(arrays, image_center)

    all_stripes = [s for a in arrays for s in a.stripes]
    cw_img = render_stripes(all_stripes, canvas)
    if np.count_nonzero(cw_img) == 0:
        return []

    results: list[ExactMatch] = []
    for fp in library:
        fp_counts = footprint_side_counts(fp)
        matches, count_rot_k = _counts_match(fp_counts, cw_counts)
        if not matches:
            continue
        # Stage 2: check 1:1 spatial overlap.
        is_valid, pairs = check_1to1(fp, all_stripes, count_rot_k, canvas)
        if not is_valid:
            continue
        # Passed both filters — compute IoU for ranking.
        fp_img = render_footprint(fp, canvas)
        if np.count_nonzero(fp_img) == 0:
            continue
        rotated_fp = _rotate_90(fp_img, count_rot_k)
        iou = _iou(rotated_fp, cw_img)
        results.append(ExactMatch(
            footprint_name=fp.name,
            footprint_path=fp.path,
            iou=iou,
            rotation_k=count_rot_k,
            fp_image=rotated_fp,
            cw_image=cw_img,
            fp_counts=fp_counts,
            cw_counts=cw_counts,
        ))

    results.sort(key=lambda r: -r.iou)
    return results
