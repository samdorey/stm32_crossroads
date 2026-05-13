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
    _draw_rect_real(img, cx, cy, long_dim, mark_px / scale,
                    angle_deg, scale, off_x, off_y)


def _draw_rect_real(
    img: np.ndarray,
    cx: float, cy: float,
    w: float, h: float,
    angle_deg: float,
    scale: float, off_x: float, off_y: float,
) -> None:
    """Draw a filled rectangle at real dimensions (both axes scaled).
    w = extent along angle_deg, h = extent perpendicular."""
    px = cx * scale + off_x
    py = cy * scale + off_y
    pw = w * scale
    ph = h * scale

    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    u = np.array([cos_a, sin_a])
    v = np.array([-sin_a, cos_a])
    c = np.array([px, py])
    hw, hh = pw / 2, ph / 2
    pts = np.array([
        c + hw * u + hh * v,
        c - hw * u + hh * v,
        c - hw * u - hh * v,
        c + hw * u - hh * v,
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


def _pair_by_side(
    fp: Footprint,
    stripes: list[Stripe],
    cw_center: tuple[float, float],
    rotation_k: int,
) -> list[tuple[Pad, Stripe]] | None:
    """Pair pads with stripes by sorting per side and matching by rank.

    rotation_k rotates the footprint's side assignment so that its N side
    aligns with the crosswalk's N side. Returns None if any side has a
    count mismatch."""
    # Classify pads into NESW.
    cx_fp = sum(p.x for p in fp.pads) / len(fp.pads)
    cy_fp = sum(p.y for p in fp.pads) / len(fp.pads)
    pad_sides: dict[str, list[Pad]] = {"N": [], "E": [], "S": [], "W": []}
    for p in fp.pads:
        pad_sides[_classify_pad_side(p, cx_fp, cy_fp)].append(p)

    # Rotate pad side labels by rotation_k.
    labels = ["N", "E", "S", "W"]
    rotated_labels = labels[rotation_k:] + labels[:rotation_k]
    # rotated_labels[i] is the original side that maps to labels[i] after rotation.
    # e.g. rotation_k=1: rotated_labels = [E, S, W, N]
    # meaning: what was E is now N, S→E, W→S, N→W.
    pad_sides_rotated: dict[str, list[Pad]] = {}
    for i, new_label in enumerate(labels):
        old_label = rotated_labels[i]
        pad_sides_rotated[new_label] = pad_sides[old_label]

    # Classify stripes into NESW.
    stripe_sides: dict[str, list[Stripe]] = {"N": [], "E": [], "S": [], "W": []}
    for s in stripes:
        dx = s.cx - cw_center[0]
        dy = s.cy - cw_center[1]
        if abs(dy) >= abs(dx):
            side = "N" if dy < 0 else "S"
        else:
            side = "E" if dx > 0 else "W"
        stripe_sides[side].append(s)

    # Sort each side by position along the edge and pair by rank.
    pairs: list[tuple[Pad, Stripe]] = []
    for label in labels:
        pads = pad_sides_rotated[label]
        ss = stripe_sides[label]
        if len(pads) != len(ss):
            return None
        if not pads:
            continue
        # N/S sides: sort by x. E/W sides: sort by y.
        if label in ("N", "S"):
            pads.sort(key=lambda p: p.x)
            ss.sort(key=lambda s: s.cx)
        else:
            pads.sort(key=lambda p: p.y)
            ss.sort(key=lambda s: s.cy)
        for pad, stripe in zip(pads, ss):
            pairs.append((pad, stripe))
    return pairs


def _solve_similarity_transform(
    src: np.ndarray, dst: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Find uniform scale + rotation + translation mapping src points to dst.

    src, dst: (N, 2) arrays of corresponding points.
    Returns (scale, angle_rad, tx, ty) or None if degenerate.

    Uses the closed-form Umeyama solution for similarity transforms."""
    n = src.shape[0]
    if n < 2:
        return None
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src_centered = src - src_c
    dst_centered = dst - dst_c
    # Cross-covariance.
    H = src_centered.T @ dst_centered / n
    U, S, Vt = np.linalg.svd(H)
    # Handle reflection.
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0 if d > 0 else -1.0])
    R = Vt.T @ D @ U.T
    # Uniform scale: ratio of dst spread to src spread.
    src_var = np.sum(src_centered ** 2) / n
    if src_var < 1e-12:
        return None
    scale = np.trace(np.diag(S) @ D) / src_var
    # Translation.
    t = dst_c - scale * R @ src_c
    angle = math.atan2(R[1, 0], R[0, 0])
    return float(scale), float(angle), float(t[0]), float(t[1])


def check_1to1(
    fp: Footprint,
    stripes: list[Stripe],
    cw_center: tuple[float, float],
    rotation_k: int,
    canvas: int = CANVAS,
) -> tuple[bool, list[tuple[int, int]], float]:
    """Check if every pad has exactly one overlapping stripe using a
    best-fit similarity transform (uniform scale + rotation + translate).

    Steps:
      1. Pair pads and stripes by side + rank order.
      2. Compute the similarity transform mapping stripe centroids → pad centroids.
      3. Render footprint pads in footprint-mm space.
      4. Transform stripe positions/angles into that same space and render.
      5. Check every pad mask overlaps exactly one stripe mask (1:1).

    Returns (is_valid, pairs_as_indices, residual_error)."""
    pairs = _pair_by_side(fp, stripes, cw_center, rotation_k)
    if pairs is None:
        return False, [], float("inf")

    # Build point correspondences: stripe centroid → pad centroid.
    src = np.array([[s.cx, s.cy] for _, s in pairs])
    dst = np.array([[p.x, p.y] for p, _ in pairs])
    result = _solve_similarity_transform(src, dst)
    if result is None:
        return False, [], float("inf")
    scale, angle, tx, ty = result

    # Compute residual: mean distance between transformed stripe centers and pad centers.
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    transformed = np.column_stack([
        scale * (cos_a * src[:, 0] - sin_a * src[:, 1]) + tx,
        scale * (sin_a * src[:, 0] + cos_a * src[:, 1]) + ty,
    ])
    residuals = np.sqrt(np.sum((transformed - dst) ** 2, axis=1))
    mean_residual = float(residuals.mean())

    # Render everything in footprint-mm space.
    # Compute the footprint transform to canvas.
    fp_scale, fp_off_x, fp_off_y = _fp_transform(fp, canvas)

    # Render each pad as an individual mask.
    pad_masks: list[np.ndarray] = []
    for p in fp.pads:
        long_dim, pad_angle = _pad_long_axis(p)
        m = _render_single_mark(canvas, p.x, p.y, long_dim, pad_angle,
                                fp_scale, fp_off_x, fp_off_y)
        pad_masks.append(m)

    # Render each stripe transformed into footprint-mm space.
    angle_deg = math.degrees(angle)
    stripe_masks: list[np.ndarray] = []
    for _, s in pairs:
        # Transform stripe center.
        tx_s = scale * (cos_a * s.cx - sin_a * s.cy) + tx
        ty_s = scale * (sin_a * s.cx + cos_a * s.cy) + ty
        # Transform stripe dimensions and angle.
        long_dim = s.length_px * scale
        stripe_angle = s.angle_deg + angle_deg
        m = _render_single_mark(canvas, tx_s, ty_s, long_dim, stripe_angle,
                                fp_scale, fp_off_x, fp_off_y)
        stripe_masks.append(m)

    # Render pads and transformed stripes at REAL dimensions in mm space,
    # then check 1:1 pixel overlap. This is the honest test: does each
    # pad's actual rectangle overlap exactly one stripe's actual rectangle?
    fp_scale, fp_off_x, fp_off_y = _fp_transform(fp, canvas)

    # Render each pad individually at real mm dimensions.
    pad_masks: list[np.ndarray] = []
    for p in fp.pads:
        long_dim, pad_angle = _pad_long_axis(p)
        short_dim = min(p.width, p.height)
        m = np.zeros((canvas, canvas), dtype=np.uint8)
        _draw_rect_real(m, p.x, p.y, long_dim, short_dim, pad_angle,
                        fp_scale, fp_off_x, fp_off_y)
        pad_masks.append(m)

    # Render each stripe transformed into mm space at real dimensions.
    stripe_masks: list[np.ndarray] = []
    for _, s in pairs:
        sx = scale * (cos_a * s.cx - sin_a * s.cy) + tx
        sy = scale * (sin_a * s.cx + cos_a * s.cy) + ty
        s_long = s.length_px * scale  # length in mm after transform
        s_short = s.width_px * scale  # width in mm after transform
        s_angle = s.angle_deg + angle_deg
        m = np.zeros((canvas, canvas), dtype=np.uint8)
        _draw_rect_real(m, sx, sy, s_long, s_short, s_angle,
                        fp_scale, fp_off_x, fp_off_y)
        stripe_masks.append(m)

    # Check 1:1 overlap.
    n_pads = len(pad_masks)
    n_stripes = len(stripe_masks)
    overlap = np.zeros((n_pads, n_stripes), dtype=bool)
    for i, pm in enumerate(pad_masks):
        pm_bool = pm > 0
        for j, sm in enumerate(stripe_masks):
            if np.any(pm_bool & (sm > 0)):
                overlap[i, j] = True

    idx_pairs: list[tuple[int, int]] = []
    for i in range(n_pads):
        matched = np.where(overlap[i])[0]
        if len(matched) != 1:
            return False, [], mean_residual
        idx_pairs.append((i, int(matched[0])))

    assigned = [p[1] for p in idx_pairs]
    if len(set(assigned)) != len(assigned):
        return False, [], mean_residual

    return True, idx_pairs, mean_residual


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


def render_aligned_comparison(
    fp: Footprint,
    stripes: list[Stripe],
    cw_center: tuple[float, float],
    rotation_k: int,
    canvas: int = CANVAS,
    aerial_bgr: np.ndarray | None = None,
    detection_overlay_bgr: np.ndarray | None = None,
) -> np.ndarray | None:
    """Render footprint and crosswalk stripes in a SHARED coordinate space.

    The crosswalk stripes are transformed into the footprint's mm space
    via the similarity transform computed from paired correspondences.

    If aerial_bgr and/or detection_overlay_bgr are provided, they are
    prepended as extra panels: [aerial] [detection] [footprint] [aligned CW] [overlap].

    Returns a multi-panel BGR image, or None if pairing fails."""
    pairs = _pair_by_side(fp, stripes, cw_center, rotation_k)
    if pairs is None or len(pairs) < 2:
        return None

    # Compute similarity transform: stripe coords → footprint mm coords.
    src = np.array([[s.cx, s.cy] for _, s in pairs])
    dst = np.array([[p.x, p.y] for p, _ in pairs])
    result = _solve_similarity_transform(src, dst)
    if result is None:
        return None
    scale_t, angle_t, tx, ty = result
    cos_a = math.cos(angle_t)
    sin_a = math.sin(angle_t)
    angle_deg_t = math.degrees(angle_t)

    # Render footprint at real pad dimensions in mm space.
    fp_scale, fp_off_x, fp_off_y = _fp_transform(fp, canvas)
    fp_img = np.zeros((canvas, canvas), dtype=np.uint8)
    for p in fp.pads:
        long_dim, pad_angle = _pad_long_axis(p)
        short_dim = min(p.width, p.height)
        _draw_rect_real(fp_img, p.x, p.y, long_dim, short_dim, pad_angle,
                        fp_scale, fp_off_x, fp_off_y)

    # Render stripes transformed into mm space at real dimensions.
    cw_img = np.zeros((canvas, canvas), dtype=np.uint8)
    all_stripes = [s for _, s in pairs]
    for s in all_stripes:
        sx = scale_t * (cos_a * s.cx - sin_a * s.cy) + tx
        sy = scale_t * (sin_a * s.cx + cos_a * s.cy) + ty
        s_long = s.length_px * scale_t
        s_short = s.width_px * scale_t
        s_angle = s.angle_deg + angle_deg_t
        _draw_rect_real(cw_img, sx, sy, s_long, s_short, s_angle,
                        fp_scale, fp_off_x, fp_off_y)

    # Build 5-panel image:
    # [aerial tile] [aerial + detection overlay] [footprint] [crosswalk aligned] [overlap]
    h, w = fp_img.shape
    overlap_img = np.zeros((h, w, 3), dtype=np.uint8)
    fp_bool = fp_img > 127
    cw_bool = cw_img > 127
    overlap_img[fp_bool & ~cw_bool] = (0, 180, 0)
    overlap_img[cw_bool & ~fp_bool] = (0, 0, 180)
    overlap_img[fp_bool & cw_bool] = (255, 255, 255)

    fp_bgr = cv2.cvtColor(fp_img, cv2.COLOR_GRAY2BGR)
    cw_bgr = cv2.cvtColor(cw_img, cv2.COLOR_GRAY2BGR)

    # If aerial imagery and detection overlay are provided, prepend them.
    panels = [fp_bgr, cw_bgr, overlap_img]
    labels = [f"FP: {fp.name}", "crosswalk (aligned)", "overlap"]

    if aerial_bgr is not None:
        # Resize aerial to match canvas.
        aerial_resized = cv2.resize(aerial_bgr, (w, h))
        panels.insert(0, aerial_resized)
        labels.insert(0, "aerial")
    if detection_overlay_bgr is not None:
        det_resized = cv2.resize(detection_overlay_bgr, (w, h))
        panels.insert(1 if aerial_bgr is not None else 0, det_resized)
        labels.insert(1 if aerial_bgr is not None else 0, "detection")

    strip = np.hstack(panels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x_off = 0
    for i, label in enumerate(labels):
        cv2.putText(strip, label, (x_off + 4, 16), font, 0.4, (0, 200, 200), 1)
        x_off += w

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
        # Stage 2: check 1:1 spatial overlap via similarity transform.
        is_valid, pairs, residual = check_1to1(
            fp, all_stripes, image_center, count_rot_k, canvas)
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
