"""Score how 'PCB component footprint'-like an intersection's crosswalk
pattern is, given detected StripeArrays.

We classify into:
  - QFP   : 4 arrays at ~90° around the body (4-sided pad array)
  - SOIC  : 2 arrays on opposite sides
  - DIP   : same as SOIC but tighter pitch / fewer "pins" — we don't really
            distinguish them from imagery alone, so we treat SOIC and DIP as
            one bucket and let the user disambiguate visually.
  - QFN   : would be 4-sided with no gaps; we ignore this since paint
            crosswalks always leave gaps.
  - Other : doesn't cleanly fit, low score.

Sub-scores are all in [0, 1]; final similarity is a weighted product so a
weak component drags the whole thing down (which matches the user intent of
"how *closely* does this look like a footprint, not just kinda").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .detect import StripeArray


@dataclass
class SideStats:
    """One 'side' of the package: arrays clustered at a similar bearing."""
    bearing_deg: float           # mean bearing from image center, 0=up/north
    arrays: list[StripeArray]
    total_stripes: int
    mean_pitch_px: float
    pitch_cv: float              # CV of all inter-stripe gaps on this side
    distance_from_center_px: float


@dataclass
class PackageMatch:
    package_class: str           # 'QFP' | 'SOIC' | 'Other'
    similarity: float            # [0, 1]
    sides: list[SideStats]
    n_sides: int
    body_aspect_ratio: float     # 1.0 = square, perfect QFP-like
    side_count_match: float
    pitch_regularity: float
    pin_count_consistency: float
    summary: str

    def as_dict(self) -> dict:
        return {
            "package_class": self.package_class,
            "similarity": round(self.similarity, 4),
            "n_sides": self.n_sides,
            "body_aspect_ratio": round(self.body_aspect_ratio, 3),
            "side_count_match": round(self.side_count_match, 3),
            "pitch_regularity": round(self.pitch_regularity, 3),
            "pin_count_consistency": round(self.pin_count_consistency, 3),
            "summary": self.summary,
            "sides": [
                {
                    "bearing_deg": round(s.bearing_deg, 1),
                    "n_stripes": s.total_stripes,
                    "mean_pitch_px": round(s.mean_pitch_px, 2),
                    "pitch_cv": round(s.pitch_cv, 3),
                    "distance_from_center_px": round(s.distance_from_center_px, 1),
                }
                for s in self.sides
            ],
        }


def _bearing_from_center(cx: float, cy: float, center: tuple[float, float]) -> float:
    """Bearing of (cx, cy) from `center` in image coords, 0° = up, clockwise."""
    dx = cx - center[0]
    dy = -(cy - center[1])  # flip so up=positive
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0


def _angular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _assign_quadrant(cx: float, cy: float, center: tuple[float, float]) -> str:
    """Fixed 4-quadrant assignment: N / E / S / W.

    After align_image_to_roads(), roads are axis-aligned, so array centroids
    fall cleanly into one of four quadrants relative to image center."""
    dx = cx - center[0]
    dy = cy - center[1]  # positive = below center in image coords
    if abs(dy) >= abs(dx):
        return "N" if dy < 0 else "S"
    else:
        return "E" if dx > 0 else "W"


def _build_sides_by_quadrant(
    arrays: list[StripeArray], center: tuple[float, float]
) -> list[SideStats]:
    """Assign each array to a fixed N/E/S/W quadrant and merge into SideStats.

    Much simpler than the old bearing-cluster approach; relies on the image
    being pre-rotated so roads are axis-aligned."""
    buckets: dict[str, list[tuple[StripeArray, float]]] = {
        "N": [], "E": [], "S": [], "W": [],
    }
    # Fixed cardinal bearings for each quadrant.
    CARDINAL = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}

    for arr in arrays:
        d = math.hypot(arr.centroid[0] - center[0], arr.centroid[1] - center[1])
        q = _assign_quadrant(arr.centroid[0], arr.centroid[1], center)
        arr.bearing_from_center_deg = CARDINAL[q]
        buckets[q].append((arr, d))

    sides: list[SideStats] = []
    for q in ("N", "E", "S", "W"):
        items = buckets[q]
        if not items:
            continue
        arrs = [a for a, _ in items]
        total = sum(a.n_stripes for a in arrs)
        # Weighted-average pitch stats across arrays on this side.
        pitches = []
        for a in arrs:
            if a.n_stripes >= 2:
                pitches.extend([a.pitch_px] * (a.n_stripes - 1))
        mean_p = sum(pitches) / len(pitches) if pitches else 0.0
        var = (sum((p - mean_p) ** 2 for p in pitches) / len(pitches)) if pitches else 0.0
        cv = math.sqrt(var) / (mean_p + 1e-6) if pitches else 1.0
        mean_d = sum(d for _, d in items) / len(items)
        sides.append(SideStats(
            bearing_deg=CARDINAL[q],
            arrays=arrs,
            total_stripes=total,
            mean_pitch_px=mean_p,
            pitch_cv=cv,
            distance_from_center_px=mean_d,
        ))
    return sides


def score_match(
    arrays: list[StripeArray],
    image_center: tuple[float, float],
    min_stripes_per_side: int = 3,
    max_distance_from_center_px: float = 300.0,
) -> PackageMatch:
    """Compute the PCB-similarity for one intersection's detection."""
    # Pre-filter: drop arrays too far from center (likely rooftop / parking lot noise).
    near_arrays = [
        a for a in arrays
        if math.hypot(a.centroid[0] - image_center[0],
                      a.centroid[1] - image_center[1]) <= max_distance_from_center_px
    ]
    sides = _build_sides_by_quadrant(near_arrays, image_center)
    # Drop trivially-small sides.
    sides = [s for s in sides if s.total_stripes >= min_stripes_per_side]
    sides.sort(key=lambda s: -s.total_stripes)
    n_sides = len(sides)

    if n_sides == 0:
        return PackageMatch(
            package_class="Other", similarity=0.0, sides=[], n_sides=0,
            body_aspect_ratio=0.0, side_count_match=0.0,
            pitch_regularity=0.0, pin_count_consistency=0.0,
            summary="no stripe arrays detected",
        )

    # --- side count contribution: prefer 4, then 2, then 3, then 1.
    # We avoid hard-classifying into a single package here; instead each side count
    # gets a different ceiling score and we let the final similarity reflect it.
    if n_sides >= 4:
        side_count_match = 1.0
        package_class = "QFP"
    elif n_sides == 2:
        # Are they roughly opposite?
        ang = _angular_diff(sides[0].bearing_deg, sides[1].bearing_deg)
        side_count_match = max(0.0, 1.0 - abs(ang - 180.0) / 30.0) * 0.85
        package_class = "SOIC"
    elif n_sides == 3:
        # 3-leg T: in PCB world this is rare (almost no packages have 3-sided pads).
        side_count_match = 0.45
        package_class = "Other"
    else:
        side_count_match = 0.25
        package_class = "Other"

    # --- pitch regularity: average of (1 - pitch_cv) across sides, clipped.
    pitch_reg_per_side = [max(0.0, 1.0 - 2.0 * s.pitch_cv) for s in sides]
    pitch_regularity = sum(pitch_reg_per_side) / len(pitch_reg_per_side)

    # --- pin count consistency: for opposing sides, how similar are the counts?
    # Use sqrt of ratio so 6-vs-9 gets ~0.82 instead of 0.67 — real crosswalks
    # rarely have perfectly matching stripe counts on opposing sides.
    pin_count_consistency = 1.0
    if n_sides >= 4:
        ring = sorted(sides, key=lambda s: s.bearing_deg)
        pairs = [(ring[0], ring[2])]
        if len(ring) >= 4:
            pairs.append((ring[1], ring[3]))
        ratios = []
        for a, b in pairs:
            lo, hi = sorted([a.total_stripes, b.total_stripes])
            ratios.append(math.sqrt(lo / max(hi, 1)))
        pin_count_consistency = sum(ratios) / len(ratios)
    elif n_sides == 2:
        lo, hi = sorted([sides[0].total_stripes, sides[1].total_stripes])
        pin_count_consistency = math.sqrt(lo / max(hi, 1))

    # --- body aspect ratio: distance between opposing sides; closer to 1.0 = square.
    body_aspect_ratio = 0.0
    if n_sides >= 4:
        ring = sorted(sides, key=lambda s: s.bearing_deg)
        d02 = ring[0].distance_from_center_px + ring[2].distance_from_center_px
        d13 = (ring[1].distance_from_center_px + ring[3].distance_from_center_px) \
              if len(ring) >= 4 else d02
        if d02 > 0 and d13 > 0:
            body_aspect_ratio = min(d02, d13) / max(d02, d13)
    elif n_sides == 2:
        body_aspect_ratio = 1.0  # not meaningful, neutral

    # --- pin-count "richness": more stripes = more PCB-like up to a point.
    total = sum(s.total_stripes for s in sides)
    richness = min(1.0, total / 32.0)  # 32 stripes total → "rich enough"

    # Compose. Weighted product so weak components hurt.
    components = {
        "side_count_match": side_count_match,
        "pitch_regularity": pitch_regularity,
        "pin_count_consistency": pin_count_consistency,
        "body_aspect_ratio": max(body_aspect_ratio, 0.5),  # don't zero out for 2-sided
        "richness": max(richness, 0.4),
    }
    similarity = 1.0
    for v in components.values():
        similarity *= v

    summary = (
        f"{package_class}: {n_sides} sides, "
        f"{total} stripes, pitch_cv≈{1.0 - pitch_regularity:.2f}, "
        f"aspect={body_aspect_ratio:.2f}"
    )

    return PackageMatch(
        package_class=package_class,
        similarity=similarity,
        sides=sides,
        n_sides=n_sides,
        body_aspect_ratio=body_aspect_ratio,
        side_count_match=side_count_match,
        pitch_regularity=pitch_regularity,
        pin_count_consistency=pin_count_consistency,
        summary=summary,
    )
