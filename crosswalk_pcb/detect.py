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


def find_paint_mask_adaptive(
    image_bgr: np.ndarray,
    block_size: int = 51,
    brightness_margin: float = 25.0,
    max_sat_white: int = 120,
    yellow_hue_lo: int = 12,
    yellow_hue_hi: int = 38,
    yellow_min_sat: int = 40,
) -> np.ndarray:
    """Local-contrast paint mask. A pixel is "paint" if it is brighter than
    its local neighborhood by at least `brightness_margin` AND has a
    paint-like color (white or yellow).

    Handles shadows: a stripe in shadow at V=130 on asphalt at V=100 still
    has a +30 local contrast, so it passes. The global method would miss it
    because V=130 < 180.

    block_size must be odd; controls the neighborhood radius (~6m at 0.12 m/px
    with block_size=51)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2].astype(np.float32)
    # Local mean brightness.
    local_mean = cv2.GaussianBlur(val, (block_size, block_size), 0)
    bright_relative = (val - local_mean) > brightness_margin
    # Color check — slightly relaxed thresholds since shadow paint is dimmer.
    white = sat <= max_sat_white
    yellow = (
        (hue >= yellow_hue_lo) & (hue <= yellow_hue_hi)
        & (sat >= yellow_min_sat)
    )
    return ((bright_relative & (white | yellow)).astype(np.uint8) * 255)


# ---------- road mask ----------

def build_road_mask(
    image_shape: tuple[int, ...],
    leg_bearings_deg: list[float],
    center_px: tuple[float, float],
    road_half_width_px: float = 80.0,
    length_px: float = 400.0,
    margin_px: float = 30.0,
) -> np.ndarray:
    """Build a binary mask of the road corridors radiating from the
    intersection center. Uses OSM leg bearings to draw thick lines
    outward from center.

    road_half_width_px + margin_px determines how wide the mask is on
    each side of the road centerline. At 0.12 m/px, 80px ≈ 10m which
    covers a typical 2-lane road with some buffer."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    thickness = int(2 * (road_half_width_px + margin_px))
    cx, cy = int(center_px[0]), int(center_px[1])
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
    """Render an overlay showing detected stripes (green) and array centroids
    (red dot + label) on top of the input image. Useful for debugging."""
    out = image_bgr.copy()
    for i, arr in enumerate(arrays):
        for s in arr.stripes:
            pts = s.contour
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            cv2.rectangle(out, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                          (0, 200, 0), 1)
        cx, cy = arr.centroid
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(out, f"#{i} n={arr.n_stripes} cv={arr.pitch_cv:.2f}",
                    (int(cx) + 6, int(cy) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return out
