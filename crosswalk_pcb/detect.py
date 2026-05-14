"""Detect continental crosswalk stripes in an aerial image and group them
into 'arrays' that correspond to the legs of an intersection.

Pipeline:
  1. Threshold for bright + low-saturation pixels (crosswalk paint).
  2. Find connected components; keep ones that look like stripes
     (rectangular, right aspect ratio, plausible area).
  3. PCA each stripe to get its centerline + orientation.
  4. Cluster stripes into arrays:
       - similar orientation, and
       - centroids colinear along the perpendicular to that orientation.
  5. Per array: pin count, pitch (mean / CV), array length, mean stripe size,
     centroid + main axis.

The output of this module is consumed by score.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------- thresholding ----------

def find_paint_mask(
    image_bgr: np.ndarray,
    min_value: int = 180,
    max_sat_white: int = 80,
    yellow_hue_lo: int = 15,
    yellow_hue_hi: int = 35,
    yellow_min_sat: int = 60,
    yellow_min_val: int = 170,
) -> np.ndarray:
    """Return a uint8 mask of likely crosswalk-paint pixels.

    Detects both white paint (high value, low saturation) and yellow paint
    (hue in the yellow band, moderate-to-high saturation, high value).
    Many US cities (SF, NYC, etc.) use yellow for continental crosswalks.

    OpenCV HSV ranges: H [0,180], S [0,255], V [0,255].
    Yellow paint in aerial imagery typically lands at H ~20-30, S 80-200, V 170+.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white_mask = (val >= min_value) & (sat <= max_sat_white)
    yellow_mask = (
        (hue >= yellow_hue_lo) & (hue <= yellow_hue_hi)
        & (sat >= yellow_min_sat)
        & (val >= yellow_min_val)
    )
    return ((white_mask | yellow_mask).astype(np.uint8) * 255)


def find_paint_mask_tophat(
    image_bgr: np.ndarray,
    stripe_width_px: int = 8,
    element_scale: float = 3.0,
    threshold_fraction: float = 0.25,
    max_sat_white: int = 120,
    yellow_hue_lo: int = 12,
    yellow_hue_hi: int = 38,
    yellow_min_sat: int = 35,
    min_absolute_val: int = 90,
) -> np.ndarray:
    """White top-hat paint mask.

    The white top-hat (morphological opening subtracted from original)
    extracts bright features *smaller* than the structuring element.
    Crosswalk stripes (width ~0.5m = ~4-8px at zoom 19) are extracted
    while large bright areas (concrete, rooftops) are suppressed.

    The structuring element is an ellipse whose minor axis is
    stripe_width_px * element_scale. This lets stripe-sized bright
    features through while rejecting anything wider.

    After top-hat, we threshold at a fraction of the local max response,
    combined with a color filter (white or yellow paint).

    This is the key advantage over global thresholds: concrete that is
    uniformly bright produces near-zero top-hat response, while paint
    stripes (bright on dark) produce a strong response."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    gray = val.astype(np.float32)

    # Structuring element: must be larger than stripe width to preserve them.
    se_size = max(3, int(stripe_width_px * element_scale)) | 1  # ensure odd
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))

    # White top-hat: original - opening.  Extracts bright features < se.
    tophat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_TOPHAT, se)

    # Threshold: pixels where the top-hat response exceeds a fraction of
    # the local maximum.  Use a large blur to estimate local max.
    if tophat.max() == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    # Adaptive threshold: compare to a large local max.
    blur_size = se_size * 5 | 1
    local_max = cv2.dilate(tophat, np.ones((blur_size, blur_size), np.uint8))
    local_max = cv2.GaussianBlur(local_max.astype(np.float32),
                                 (blur_size, blur_size), 0)
    local_max = np.maximum(local_max, 1.0)

    # Simple threshold on tophat value.
    th_val = max(10, int(tophat.max() * threshold_fraction))
    bright = tophat >= th_val

    # Also require minimum absolute brightness.
    bright = bright & (val >= min_absolute_val)

    # Color filter: white or yellow.
    white = sat <= max_sat_white
    yellow = (
        (hue >= yellow_hue_lo) & (hue <= yellow_hue_hi)
        & (sat >= yellow_min_sat)
    )
    return ((bright & (white | yellow)).astype(np.uint8) * 255)


def find_paint_mask_multithresh(
    image_bgr: np.ndarray,
    thresholds: tuple[int, ...] = (140, 160, 180, 200),
    max_sat_white: int = 100,
    yellow_hue_lo: int = 12,
    yellow_hue_hi: int = 38,
    yellow_min_sat: int = 40,
    min_stripe_area_px: int = 30,
    max_stripe_area_px: int = 2000,
) -> np.ndarray:
    """Multi-threshold paint mask with connected-component filtering.

    Runs the global threshold at multiple brightness levels and keeps
    components that appear as stripe-shaped at ANY threshold level.
    This handles shadows (lower threshold captures dim paint) and bright
    concrete (higher thresholds filter out diffuse brightness -- only
    compact bright features survive the area filter).

    The key insight: at V=140, shadowed paint appears but so does some
    concrete; at V=200, only the brightest paint survives. A stripe-shaped
    component at any level is likely paint."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Color filter.
    white = sat <= max_sat_white
    yellow = (
        (hue >= yellow_hue_lo) & (hue <= yellow_hue_hi)
        & (sat >= yellow_min_sat)
    )
    color_ok = white | yellow

    combined = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    for thresh in thresholds:
        mask = ((val >= thresh) & color_ok).astype(np.uint8) * 255
        # Morphological cleanup.
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        # Keep only stripe-sized components.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            h = stats[lbl, cv2.CC_STAT_HEIGHT]
            aspect = max(w, h) / max(min(w, h), 1)
            if min_stripe_area_px <= area <= max_stripe_area_px and aspect >= 2.0:
                combined[labels == lbl] = 255

    return combined


def find_paint_mask_canny_lines(
    image_bgr: np.ndarray,
    canny_lo: int = 30,
    canny_hi: int = 100,
    hough_threshold: int = 15,
    min_line_length_px: int = 8,
    max_line_gap_px: int = 12,
    stripe_width_range_px: tuple[int, int] = (2, 22),
    angle_tolerance_deg: float = 20.0,
    min_pair_length_px: int = 8,
    dilate_px: int = 3,
    max_lines: int = 800,
) -> np.ndarray:
    """Edge-based stripe detection using Canny + Hough line segments.

    Detects edges, finds line segments, then pairs parallel segments that
    are stripe-width apart. For each valid pair, draws a filled rotated
    rectangle (proper stripe shape) between the two line midpoints.

    Then dilates the result to create connected blobs suitable for
    the connected-component extraction pipeline."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length_px,
        maxLineGap=max_line_gap_px,
    )
    if lines is None:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    segments = lines.reshape(-1, 4)
    dx = (segments[:, 2] - segments[:, 0]).astype(np.float64)
    dy = (segments[:, 3] - segments[:, 1]).astype(np.float64)
    angles = np.degrees(np.arctan2(dy, dx))
    lengths = np.sqrt(dx**2 + dy**2)
    mid_x = (segments[:, 0] + segments[:, 2]) / 2.0
    mid_y = (segments[:, 1] + segments[:, 3]) / 2.0

    # Cap the number of lines to keep the O(n^2) pairing tractable.
    if len(segments) > max_lines:
        # Keep the longest lines.
        order = np.argsort(-lengths)[:max_lines]
        segments = segments[order]
        dx = dx[order]; dy = dy[order]
        angles = angles[order]; lengths = lengths[order]
        mid_x = mid_x[order]; mid_y = mid_y[order]

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    n = len(segments)
    min_w, max_w = stripe_width_range_px

    norms = np.column_stack([-dy, dx])
    safe_lengths = np.maximum(lengths, 1e-6)
    norms = norms / safe_lengths[:, None]
    dirs = np.column_stack([dx, dy]) / safe_lengths[:, None]

    for i in range(n):
        if lengths[i] < min_pair_length_px:
            continue
        for j in range(i + 1, n):
            if lengths[j] < min_pair_length_px:
                continue
            adiff = abs(angles[i] - angles[j]) % 180
            adiff = min(adiff, 180 - adiff)
            if adiff > angle_tolerance_deg:
                continue
            dmid = np.array([mid_x[j] - mid_x[i], mid_y[j] - mid_y[i]])
            perp_dist = abs(float(dmid @ norms[i]))
            if not (min_w <= perp_dist <= max_w):
                continue
            along_dist = abs(float(dmid @ dirs[i]))
            # Allow midpoints to be offset along the line direction
            # up to the full length of the longer segment (not 0.7x).
            if along_dist > max(lengths[i], lengths[j]) * 1.2:
                continue
            # Draw a proper rotated rectangle centered between the pair.
            cx = (mid_x[i] + mid_x[j]) / 2.0
            cy = (mid_y[i] + mid_y[j]) / 2.0
            avg_len = (lengths[i] + lengths[j]) / 2.0
            avg_angle = angles[i]  # use one line's angle
            cos_a = math.cos(math.radians(avg_angle))
            sin_a = math.sin(math.radians(avg_angle))
            u = np.array([cos_a, sin_a])
            v = np.array([-sin_a, cos_a])
            c = np.array([cx, cy])
            hl, hw = avg_len / 2, perp_dist / 2
            corners = np.array([
                c + hl * u + hw * v,
                c - hl * u + hw * v,
                c - hl * u - hw * v,
                c + hl * u - hw * v,
            ], dtype=np.int32)
            cv2.fillPoly(mask, [corners], 255)

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(mask, k)

    return mask


def find_paint_mask_ensemble(
    image_bgr: np.ndarray,
    use_tophat: bool = True,
    use_adaptive: bool = True,
    use_global: bool = True,
    min_votes: int = 2,
    tophat_kwargs: dict | None = None,
    adaptive_kwargs: dict | None = None,
    global_kwargs: dict | None = None,
) -> np.ndarray:
    """Ensemble paint mask: combine multiple detection methods via voting.

    A pixel is marked as paint if at least min_votes methods agree.
    This is robust because each method has different failure modes:
    - Global fails on bright concrete (false positives)
    - Adaptive fails on very uniform areas
    - Top-hat naturally rejects large bright areas

    With min_votes=2, a pixel needs agreement from 2/3 methods."""
    votes = np.zeros(image_bgr.shape[:2], dtype=np.int32)

    if use_global:
        kw = global_kwargs or {}
        m = find_paint_mask(image_bgr, **kw)
        votes += (m > 0).astype(np.int32)

    if use_adaptive:
        kw = adaptive_kwargs or {}
        m = find_paint_mask_adaptive(image_bgr, **kw)
        votes += (m > 0).astype(np.int32)

    if use_tophat:
        kw = tophat_kwargs or {}
        m = find_paint_mask_tophat(image_bgr, **kw)
        votes += (m > 0).astype(np.int32)

    return ((votes >= min_votes).astype(np.uint8) * 255)


def find_paint_mask_tophat_adaptive(
    image_bgr: np.ndarray,
    stripe_width_px: int = 8,
    element_scale: float = 3.0,
    brightness_margin: float = 20.0,
    min_absolute_val: int = 90,
    max_sat_white: int = 120,
    yellow_hue_lo: int = 12,
    yellow_hue_hi: int = 38,
    yellow_min_sat: int = 35,
) -> np.ndarray:
    """Combined top-hat + adaptive local contrast.

    Uses top-hat to extract features smaller than the structuring element,
    THEN applies adaptive thresholding on the top-hat response itself.
    This double filtering:
    1. Top-hat removes large bright areas (concrete, rooftops)
    2. Adaptive threshold on top-hat adapts to local response intensity

    This is the most robust combination: concrete gives near-zero top-hat,
    while paint stripes give strong top-hat. Even in shadow, the relative
    brightness difference (paint vs asphalt) produces top-hat response."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Top-hat to extract small bright features.
    se_size = max(3, int(stripe_width_px * element_scale)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    tophat = cv2.morphologyEx(val, cv2.MORPH_TOPHAT, se)

    # Adaptive threshold on the top-hat response.
    tophat_f = tophat.astype(np.float32)
    block = se_size * 5 | 1
    local_mean = cv2.GaussianBlur(tophat_f, (block, block), 0)
    bright = tophat_f > (local_mean + brightness_margin)

    # Absolute brightness floor.
    bright = bright & (val >= min_absolute_val)

    # Color filter.
    white = sat <= max_sat_white
    yellow = (
        (hue >= yellow_hue_lo) & (hue <= yellow_hue_hi)
        & (sat >= yellow_min_sat)
    )
    return ((bright & (white | yellow)).astype(np.uint8) * 255)


def find_paint_mask_adaptive(
    image_bgr: np.ndarray,
    block_size: int = 51,
    brightness_margin: float = 30.0,
    min_absolute_val: int = 100,
    max_sat_white: int = 120,
    yellow_hue_lo: int = 12,
    yellow_hue_hi: int = 38,
    yellow_min_sat: int = 40,
) -> np.ndarray:
    """Local-contrast paint mask. A pixel is "paint" if:
      1. It is brighter than its local neighborhood by brightness_margin, AND
      2. Its absolute brightness is above min_absolute_val (rejects dark
         edges like tree shadows that are locally bright), AND
      3. It has a paint-like color (white or yellow).

    Handles shadows: a stripe in shadow at V=130 on asphalt at V=100 still
    has +30 local contrast and V>90, so it passes.

    block_size must be odd; controls the neighborhood radius (~6m at 0.12 m/px
    with block_size=51)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2].astype(np.float32)
    # Local mean brightness.
    local_mean = cv2.GaussianBlur(val, (block_size, block_size), 0)
    bright_relative = (val - local_mean) > brightness_margin
    bright_absolute = val >= min_absolute_val
    # Color check — slightly relaxed thresholds since shadow paint is dimmer.
    white = sat <= max_sat_white
    yellow = (
        (hue >= yellow_hue_lo) & (hue <= yellow_hue_hi)
        & (sat >= yellow_min_sat)
    )
    return ((bright_relative & bright_absolute & (white | yellow)).astype(np.uint8) * 255)


def find_paint_mask_contrast(
    image_bgr: np.ndarray,
    neighborhood_size: int = 31,
    min_asphalt_frac: float = 0.35,
    asphalt_max_val: int = 160,
    asphalt_max_sat: int = 70,
    paint_min_val: int = 110,
    paint_max_sat_white: int = 120,
    yellow_hue_lo: int = 12,
    yellow_hue_hi: int = 38,
    yellow_min_sat: int = 35,
) -> np.ndarray:
    """Color-contrast paint mask. A pixel is "stripe" if:
      1. It is paint-colored (white or yellow), AND
      2. Its local neighborhood is predominantly asphalt.

    This is robust to shadows (asphalt stays dark+neutral even in shadow,
    paint stays relatively bright+neutral) and rejects rooftops/cars
    (their neighborhoods aren't asphalt).

    neighborhood_size controls the kernel radius for the asphalt fraction
    computation (~4m at 0.12 m/px with size=31)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Asphalt: dark-to-medium brightness, low saturation, any hue.
    asphalt = (val <= asphalt_max_val) & (sat <= asphalt_max_sat)

    # Local asphalt fraction: what fraction of the neighborhood is asphalt?
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (neighborhood_size, neighborhood_size))
    kernel_norm = kernel.astype(np.float32) / kernel.sum()
    local_asphalt = cv2.filter2D(asphalt.astype(np.float32), -1, kernel_norm)
    on_road = local_asphalt >= min_asphalt_frac

    # Paint: white or yellow (relaxed — shadow paint can be V=110+).
    white = (val >= paint_min_val) & (sat <= paint_max_sat_white)
    yellow = (
        (hue >= yellow_hue_lo) & (hue <= yellow_hue_hi)
        & (sat >= yellow_min_sat)
        & (val >= paint_min_val)
    )
    paint = white | yellow

    return ((paint & on_road).astype(np.uint8) * 255)


# ---------- road mask ----------

def build_road_mask(
    image_shape: tuple[int, ...],
    leg_bearings_deg: list[float],
    center_px: tuple[float, float],
    road_half_width_px: float = 100.0,
    length_px: float = 400.0,
    margin_px: float = 40.0,
    center_radius_px: float = 150.0,
) -> np.ndarray:
    """Build a binary mask of the road corridors radiating from the
    intersection center, plus a circle covering the intersection itself.

    road_half_width_px + margin_px determines how wide the mask is on
    each side of the road centerline. center_radius_px covers the
    intersection area where crosswalks sit between the legs."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    thickness = int(2 * (road_half_width_px + margin_px))
    cx, cy = int(center_px[0]), int(center_px[1])
    # Circle covering the intersection center area.
    cv2.circle(mask, (cx, cy), int(center_radius_px), 255, -1)
    # Road corridors radiating outward.
    for bearing in leg_bearings_deg:
        angle_rad = math.radians(bearing)
        dx = math.sin(angle_rad) * length_px
        dy = -math.cos(angle_rad) * length_px
        end = (int(cx + dx), int(cy + dy))
        cv2.line(mask, (cx, cy), end, 255, thickness)
    return mask


# ---------- periodic stripe detection ----------

def detect_stripes_periodic(
    image_bgr: np.ndarray,
    leg_bearings_deg: list[float],
    m_per_px: float,
    center_px: tuple[float, float],
    road_mask: np.ndarray | None = None,
    min_pitch_m: float = 0.8,
    max_pitch_m: float = 2.0,
    stripe_width_m: float = 0.6,
    search_depth_m: float = 5.0,
    search_width_m: float = 8.0,
    min_stripes: int = 3,
    min_peak_strength: float = 0.15,
) -> list:
    """Detect crosswalk stripes by looking for periodic bright/dark patterns
    along each road leg.

    For each leg bearing, samples a strip of pixels in the road direction
    at the expected crosswalk location, computes the autocorrelation, and
    extracts stripe positions from periodic peaks.

    Returns a list of Stripe objects (same as extract_stripes)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    cx, cy = center_px
    min_pitch_px = min_pitch_m / m_per_px
    max_pitch_px = max_pitch_m / m_per_px
    search_depth_px = search_depth_m / m_per_px
    search_width_px = search_width_m / m_per_px
    stripe_width_px = stripe_width_m / m_per_px

    all_stripes: list[Stripe] = []

    for bearing in leg_bearings_deg:
        angle_rad = math.radians(bearing)
        # Road direction unit vector (from center outward).
        road_dx = math.sin(angle_rad)
        road_dy = -math.cos(angle_rad)
        # Perpendicular (along-crosswalk) unit vector.
        perp_dx = road_dy
        perp_dy = -road_dx

        # Sample a rectangular strip at the expected crosswalk location.
        # The strip is centered at (center + search_depth along road),
        # oriented along the perpendicular direction.
        strip_center_x = cx + road_dx * search_depth_px
        strip_center_y = cy + road_dy * search_depth_px

        # Sample along the perpendicular at multiple offsets along the road
        # direction, then average. This gives a 1D brightness profile.
        n_perp = int(search_width_px)
        n_road = max(3, int(stripe_width_px * 0.5))  # average over a few road-direction pixels
        profile = np.zeros(n_perp, dtype=np.float32)
        count = 0
        for rd in range(-n_road // 2, n_road // 2 + 1):
            for i in range(n_perp):
                t = i - n_perp / 2
                px = int(strip_center_x + perp_dx * t + road_dx * rd)
                py = int(strip_center_y + perp_dy * t + road_dy * rd)
                if 0 <= px < w and 0 <= py < h:
                    if road_mask is None or road_mask[py, px] > 0:
                        profile[i] += gray[py, px]
                        count += 1
            count = max(count, 1)  # avoid div by zero in sparse areas
        profile /= max(n_road, 1)

        if len(profile) < int(min_pitch_px * 3):
            continue

        # Remove DC / low-frequency trend.
        kernel = int(max_pitch_px * 2) | 1  # ensure odd
        if kernel < len(profile):
            baseline = cv2.GaussianBlur(profile.reshape(1, -1), (kernel, 1), 0).flatten()
            profile_hp = profile - baseline
        else:
            profile_hp = profile - profile.mean()

        # Autocorrelation.
        n = len(profile_hp)
        autocorr = np.correlate(profile_hp, profile_hp, mode="full")
        autocorr = autocorr[n - 1:]  # keep non-negative lags only
        if autocorr[0] > 0:
            autocorr /= autocorr[0]

        # Find peaks in autocorrelation at valid pitch range.
        lo = int(min_pitch_px)
        hi = min(int(max_pitch_px), len(autocorr) - 1)
        if lo >= hi:
            continue
        peak_region = autocorr[lo:hi + 1]
        if len(peak_region) == 0 or peak_region.max() < min_peak_strength:
            continue
        best_lag = lo + int(np.argmax(peak_region))

        # Found a periodic signal. Now extract individual stripe positions
        # by finding local maxima in the high-pass profile at ~best_lag spacing.
        # Use a simple peak finder: local max within half-pitch windows.
        half_lag = best_lag // 2
        peaks: list[int] = []
        for start in range(0, n - half_lag, best_lag):
            window = profile_hp[start:start + best_lag]
            if len(window) == 0:
                continue
            local_max_idx = start + int(np.argmax(window))
            if profile_hp[local_max_idx] > profile_hp.std() * 0.3:
                peaks.append(local_max_idx)

        if len(peaks) < min_stripes:
            continue

        # Convert peak positions to image coordinates.
        for pk in peaks:
            t = pk - n_perp / 2
            sx = strip_center_x + perp_dx * t
            sy = strip_center_y + perp_dy * t
            # Stripe is oriented along the road direction (perpendicular to
            # the crosswalk array direction).
            stripe_angle = math.degrees(math.atan2(road_dy, road_dx))
            stripe_length_px = search_depth_px * 0.6  # approximate
            all_stripes.append(Stripe(
                cx=sx, cy=sy,
                angle_deg=stripe_angle,
                length_px=stripe_length_px,
                width_px=stripe_width_px,
                area_px=int(stripe_length_px * stripe_width_px),
                contour=np.array([[int(sx), int(sy)]], dtype=np.int32),
            ))

    return all_stripes


# ---------- image alignment ----------

def align_image_to_roads(
    image_bgr: np.ndarray, leg_bearings_deg: list[float]
) -> tuple[np.ndarray, float]:
    """Rotate image so the road closest to a cardinal direction becomes
    exactly axis-aligned. Returns (rotated_image, rotation_degrees).

    leg_bearings_deg are compass bearings (0=up/north, CW) from the OSM
    junction data. We find whichever leg is closest to any multiple of 90°
    and rotate the image to close that gap.

    cv2.getRotationMatrix2D uses CCW-positive angles, and our bearings are
    CW-from-north in image coords (y-down). A leg at bearing B needs
    rotation = -(B - nearest_cardinal) in cv2's convention."""
    best_dev = 180.0
    for b in leg_bearings_deg:
        for card in (0, 90, 180, 270):
            diff = ((b - card + 180) % 360) - 180  # signed, [-180, 180)
            if abs(diff) < abs(best_dev):
                best_dev = diff
    rotation_deg = -best_dev
    h, w = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        image_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, rotation_deg


# ---------- stripe extraction ----------

@dataclass
class Stripe:
    cx: float
    cy: float
    angle_deg: float    # orientation of long axis, [-90, 90)
    length_px: float    # along long axis
    width_px: float     # along short axis
    area_px: int
    contour: np.ndarray = field(repr=False)


def _pca_oriented_box(pts: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """PCA of (N,2) pixel coords. Returns (cx, cy, angle_deg, len, width)
    or None for degenerate blobs (all collinear, singular covariance)."""
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    if cov.ndim < 2 or not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvecs)):
        return None
    # eigh returns ascending eigenvalues -> long axis is the last column.
    long_axis = eigvecs[:, -1]
    # Project onto axes, compute spans.
    proj_long = centered @ long_axis
    proj_short = centered @ eigvecs[:, 0]
    if not (np.all(np.isfinite(proj_long)) and np.all(np.isfinite(proj_short))):
        return None
    length = proj_long.max() - proj_long.min()
    width = proj_short.max() - proj_short.min()
    # Note: pixel y grows downward, so angle math is in image coords. We use
    # 'angle of long axis in image coords' and normalize to [-90, 90).
    angle = math.degrees(math.atan2(long_axis[1], long_axis[0]))
    if angle >= 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return float(mean[0]), float(mean[1]), float(angle), float(length), float(width)


def extract_stripes(
    mask: np.ndarray,
    m_per_px: float,
    min_stripe_len_m: float = 1.5,
    max_stripe_len_m: float = 6.0,
    min_stripe_width_m: float = 0.25,
    max_stripe_width_m: float = 1.2,
    min_aspect_ratio: float = 2.5,
) -> list[Stripe]:
    """Extract stripe-like components from a binary paint mask."""
    # Open with a small kernel to remove speckle, then close to glue gaps from JPEG noise.
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_open, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    stripes: list[Stripe] = []
    min_area = max(8, int((min_stripe_len_m / m_per_px) * (min_stripe_width_m / m_per_px) * 0.5))
    max_area = int((max_stripe_len_m / m_per_px) * (max_stripe_width_m / m_per_px) * 2.0)

    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        ys, xs = np.where(labels == lbl)
        if len(xs) < 6:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        result = _pca_oriented_box(pts)
        if result is None:
            continue
        cx, cy, angle, length, width = result
        if width < 1.0:
            continue
        aspect = length / max(width, 1e-6)
        length_m = length * m_per_px
        width_m = width * m_per_px
        if not (min_stripe_len_m <= length_m <= max_stripe_len_m):
            continue
        if not (min_stripe_width_m <= width_m <= max_stripe_width_m):
            continue
        if aspect < min_aspect_ratio:
            continue
        # Build a contour for visualization.
        contour = np.column_stack([xs, ys]).astype(np.int32)
        stripes.append(
            Stripe(
                cx=cx, cy=cy, angle_deg=angle,
                length_px=length, width_px=width, area_px=area,
                contour=contour,
            )
        )
    return stripes


# ---------- grouping stripes into arrays ----------

@dataclass
class StripeArray:
    """A linear array of parallel stripes — the crosswalk on one leg."""
    stripes: list[Stripe]
    centroid: tuple[float, float]
    stripe_angle_deg: float    # orientation of individual stripes' long axis
    array_angle_deg: float     # orientation along which stripe centroids fall
    n_stripes: int = 0
    pitch_px: float = 0.0
    pitch_cv: float = 1.0      # CV of inter-stripe spacing; 0 = perfectly regular
    length_px: float = 0.0     # total span along array axis
    bearing_from_center_deg: float = 0.0  # set by score.py from image center

    def __post_init__(self):
        self.n_stripes = len(self.stripes)


def _wrap_angle_diff(a: float, b: float) -> float:
    """Smallest absolute diff between two angles modulo 180°."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def group_stripes_into_arrays(
    stripes: list[Stripe],
    angle_tol_deg: float = 12.0,
    perp_tol_px: float = 18.0,
    min_count: int = 3,
) -> list[StripeArray]:
    """Group stripes into linear arrays of parallel siblings.

    Two stripes are 'array-mates' if:
      - their long-axis orientations are within angle_tol_deg, AND
      - the line through both centroids is roughly perpendicular to the stripe
        orientation (within perp_tol_px of being a clean perpendicular array).
    """
    if not stripes:
        return []

    # Build adjacency by stripe orientation cluster first.
    angle_clusters: list[list[int]] = []
    for i, s in enumerate(stripes):
        placed = False
        for c in angle_clusters:
            ref = stripes[c[0]]
            if _wrap_angle_diff(s.angle_deg, ref.angle_deg) <= angle_tol_deg:
                c.append(i)
                placed = True
                break
        if not placed:
            angle_clusters.append([i])

    arrays: list[StripeArray] = []
    for cluster in angle_clusters:
        if len(cluster) < min_count:
            continue
        # Within an angle cluster, separate into sub-arrays by perpendicular position.
        # Project each stripe centroid onto the axis perpendicular to the mean stripe
        # angle, then run a 1-D clustering by gap continuity.
        ref_angle = float(np.mean([stripes[i].angle_deg for i in cluster]))
        ax = math.radians(ref_angle)
        # Unit vector along stripe long axis:
        u = np.array([math.cos(ax), math.sin(ax)])
        # Perpendicular axis (which is the array axis):
        v = np.array([-u[1], u[0]])

        positions = []  # (stripe_idx, projection_along_v, projection_along_u)
        for i in cluster:
            s = stripes[i]
            p = np.array([s.cx, s.cy])
            positions.append((i, float(p @ v), float(p @ u)))
        # Sort by projection along v (array axis).
        positions.sort(key=lambda t: t[1])

        # Now we want to group by "near-collinear along v" — i.e. their u-projections
        # should not differ by much (the array is a tight line, not a scattered blob).
        # We greedy-cluster: start a new sub-array whenever the next stripe's
        # u-projection is more than perp_tol_px from the running median.
        sub: list[list[tuple[int, float, float]]] = []
        for p in positions:
            placed = False
            for grp in sub:
                med_u = float(np.median([q[2] for q in grp]))
                if abs(p[2] - med_u) <= perp_tol_px:
                    grp.append(p)
                    placed = True
                    break
            if not placed:
                sub.append([p])

        for grp in sub:
            if len(grp) < min_count:
                continue
            # Sort by along-array projection again (in case order got mixed).
            grp.sort(key=lambda t: t[1])
            grp_stripes = [stripes[t[0]] for t in grp]
            # pitch
            v_positions = np.array([t[1] for t in grp])
            gaps = np.diff(v_positions)
            pitch_px = float(np.mean(gaps)) if len(gaps) else 0.0
            pitch_cv = float(np.std(gaps) / (abs(pitch_px) + 1e-6)) if len(gaps) else 1.0
            length_px = float(v_positions.max() - v_positions.min())
            cx = float(np.mean([s.cx for s in grp_stripes]))
            cy = float(np.mean([s.cy for s in grp_stripes]))
            # array_angle_deg is the bearing of v in image coords.
            array_angle = math.degrees(math.atan2(v[1], v[0]))
            if array_angle >= 90:
                array_angle -= 180
            if array_angle < -90:
                array_angle += 180
            arrays.append(
                StripeArray(
                    stripes=grp_stripes,
                    centroid=(cx, cy),
                    stripe_angle_deg=ref_angle,
                    array_angle_deg=array_angle,
                    pitch_px=abs(pitch_px),
                    pitch_cv=pitch_cv,
                    length_px=length_px,
                )
            )
    return arrays


# ---------- visualization ----------

def draw_overlay(image_bgr: np.ndarray, arrays: list[StripeArray]) -> np.ndarray:
    """Render an overlay showing detected stripes (green rotated boxes) and
    array centroids (red dot + label) on top of the input image."""
    out = image_bgr.copy()
    for i, arr in enumerate(arrays):
        for s in arr.stripes:
            # Draw a rotated rectangle matching the stripe's PCA orientation.
            cos_a = math.cos(math.radians(s.angle_deg))
            sin_a = math.sin(math.radians(s.angle_deg))
            u = np.array([cos_a, sin_a])   # along long axis
            v = np.array([-sin_a, cos_a])   # along short axis
            c = np.array([s.cx, s.cy])
            hl, hw = s.length_px / 2, s.width_px / 2
            corners = np.array([
                c + hl * u + hw * v,
                c - hl * u + hw * v,
                c - hl * u - hw * v,
                c + hl * u - hw * v,
            ], dtype=np.int32)
            cv2.polylines(out, [corners], isClosed=True, color=(0, 200, 0), thickness=1)
        cx, cy = arr.centroid
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(out, f"#{i} n={arr.n_stripes} cv={arr.pitch_cv:.2f}",
                    (int(cx) + 6, int(cy) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return out
