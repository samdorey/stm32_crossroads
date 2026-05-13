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

def find_paint_mask(image_bgr: np.ndarray, min_value: int = 190, max_sat: int = 80) -> np.ndarray:
    """Return a uint8 mask of likely-paint pixels (bright + low saturation).

    Tuning notes: in shaded portions of an intersection paint will be dimmer;
    too-low min_value picks up concrete/sand. Watch for false positives on
    light-colored vehicle roofs (they're high-value too)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    return ((val >= min_value) & (sat <= max_sat)).astype(np.uint8) * 255


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


def _pca_oriented_box(pts: np.ndarray) -> tuple[float, float, float, float, float]:
    """PCA of (N,2) pixel coords. Returns (cx, cy, angle_deg, len, width)."""
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues -> long axis is the last column.
    long_axis = eigvecs[:, -1]
    # Project onto axes, compute spans.
    proj_long = centered @ long_axis
    proj_short = centered @ eigvecs[:, 0]
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
        cx, cy, angle, length, width = _pca_oriented_box(pts)
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
