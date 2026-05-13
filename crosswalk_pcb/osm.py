"""OpenStreetMap (Overpass API) queries and intersection geometry.

The job of this module is to take a bbox, find road junctions where:
  - roads meet at roughly perpendicular angles, and
  - there are marked crossings on multiple legs.

We then hand a ranked list of junctions to the imagery + detection pipeline.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

# Roads we'd expect to have pedestrian crossings. Excludes motorway/trunk
# (no pedestrians) and most service roads (too small / private).
CROSSING_ROAD_HIGHWAYS = (
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "living_street",
    "tertiary_link",
    "secondary_link",
)

EARTH_R_M = 6371008.8


# ---------- math helpers ----------

def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two lon/lat points, in meters."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


def initial_bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial bearing from (1) to (2), degrees clockwise from north [0, 360)."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference between two bearings, in [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def cluster_bearings(bearings: list[float], tol_deg: float = 15.0) -> list[float]:
    """Cluster bearings within tol_deg and return the cluster mean bearings.
    Uses a simple greedy 1-D clustering on the circle."""
    if not bearings:
        return []
    sorted_bearings = sorted(bearings)
    clusters: list[list[float]] = [[sorted_bearings[0]]]
    for b in sorted_bearings[1:]:
        if angle_diff_deg(b, clusters[-1][-1]) <= tol_deg:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    # Merge wraparound: if first and last cluster are within tol, merge them.
    if len(clusters) > 1 and angle_diff_deg(clusters[0][0], clusters[-1][-1]) <= tol_deg:
        merged = clusters[-1] + clusters[0]
        clusters = [merged] + clusters[1:-1]
    # Compute circular means.
    means = []
    for c in clusters:
        # Convert to unit vectors and average to handle wraparound.
        sx = sum(math.sin(math.radians(b)) for b in c)
        sy = sum(math.cos(math.radians(b)) for b in c)
        means.append((math.degrees(math.atan2(sx, sy)) + 360.0) % 360.0)
    return means


# ---------- data types ----------

@dataclass
class Crossing:
    osm_id: int
    lon: float
    lat: float
    crossing_type: str  # value of `crossing` or `crossing_ref`


@dataclass
class Junction:
    osm_node_id: int
    lon: float
    lat: float
    leg_bearings: list[float] = field(default_factory=list)  # degrees, sorted
    crossings: list[Crossing] = field(default_factory=list)

    @property
    def n_legs(self) -> int:
        return len(self.leg_bearings)

    def perpendicularity_score(self) -> float:
        """0-1 score for how 'PCB-like-orthogonal' the junction's legs are.

        4 legs at exactly 90° apart -> 1.0
        3 legs with one pair at 180° and a third at 90° -> 1.0
        Score decays linearly with the average deviation from these patterns.
        """
        bs = self.leg_bearings
        if len(bs) < 3:
            return 0.0
        # Adjacent gaps around the compass, sorted.
        sb = sorted(bs)
        gaps = [(sb[(i + 1) % len(sb)] - sb[i]) % 360.0 for i in range(len(sb))]
        if len(bs) == 4:
            # Ideal: 90, 90, 90, 90
            err = sum(abs(g - 90.0) for g in gaps) / 4.0
            return max(0.0, 1.0 - err / 30.0)  # 0 at 30° avg error
        if len(bs) == 3:
            # Ideal: one gap at 180°, two at 90°.
            sg = sorted(gaps)
            err = (abs(sg[0] - 90.0) + abs(sg[1] - 90.0) + abs(sg[2] - 180.0)) / 3.0
            return max(0.0, 1.0 - err / 30.0)
        return 0.0

    def n_legs_with_crossing(self, radius_m: float = 40.0) -> int:
        """Number of legs that have at least one marked crossing within radius_m
        and pointing roughly along the leg bearing (±25°)."""
        leg_has = [False] * len(self.leg_bearings)
        for c in self.crossings:
            d = haversine_m(self.lon, self.lat, c.lon, c.lat)
            if d > radius_m:
                continue
            b = initial_bearing_deg(self.lon, self.lat, c.lon, c.lat)
            for i, leg_b in enumerate(self.leg_bearings):
                if angle_diff_deg(b, leg_b) <= 25.0:
                    leg_has[i] = True
                    break
        return sum(leg_has)


# ---------- Overpass query and parsing ----------

def build_query(bbox: tuple[float, float, float, float], timeout_s: int = 90) -> str:
    """bbox is (south, west, north, east) in degrees."""
    s, w, n, e = bbox
    road_filter = "|".join(CROSSING_ROAD_HIGHWAYS)
    return f"""
[out:json][timeout:{timeout_s}];
(
  node["highway"="crossing"]["crossing"~"^(marked|zebra|uncontrolled|traffic_signals)$"]({s},{w},{n},{e});
  node["highway"="crossing"]["crossing_ref"="zebra"]({s},{w},{n},{e});
  way["highway"~"^({road_filter})$"]({s},{w},{n},{e});
);
out body;
>;
out skel qt;
""".strip()


def run_overpass(query: str, cache_path: Path | None = None) -> dict:
    """Run an Overpass query, trying mirrors on failure. Cache to disk if asked."""
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as f:
            return json.load(f)
    last_err: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                timeout=180,
                headers={"User-Agent": "crosswalk-pcb-prototype/0.1"},
            )
            r.raise_for_status()
            data = r.json()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w") as f:
                    json.dump(data, f)
            return data
        except Exception as e:
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(f"all Overpass endpoints failed: {last_err}")


def extract_junctions(overpass_json: dict, min_legs: int = 3) -> list[Junction]:
    """Find junction nodes (shared by 2+ relevant ways) and compute leg bearings.
    Then attach nearby marked crossings to each junction."""
    nodes: dict[int, tuple[float, float]] = {}  # id -> (lon, lat)
    ways: list[tuple[int, list[int]]] = []  # (way_id, node_ids)
    crossings: list[Crossing] = []

    for el in overpass_json.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
            tags = el.get("tags") or {}
            if tags.get("highway") == "crossing":
                ctype = tags.get("crossing") or tags.get("crossing_ref") or "unknown"
                if ctype in {"marked", "zebra", "uncontrolled", "traffic_signals"}:
                    crossings.append(
                        Crossing(
                            osm_id=el["id"],
                            lon=el["lon"],
                            lat=el["lat"],
                            crossing_type=ctype,
                        )
                    )
        elif el["type"] == "way":
            tags = el.get("tags") or {}
            if tags.get("highway") in CROSSING_ROAD_HIGHWAYS:
                ways.append((el["id"], el["nodes"]))

    # Count how many ways each node belongs to AND for each junction node,
    # collect the bearings of the way leaving the node.
    node_way_count: dict[int, int] = {}
    node_outgoing_bearings: dict[int, list[float]] = {}
    for way_id, node_ids in ways:
        if len(node_ids) < 2:
            continue
        for i, nid in enumerate(node_ids):
            node_way_count[nid] = node_way_count.get(nid, 0) + 1
            # Compute bearing to a neighbor at least ~20m away for stability.
            if nid not in nodes:
                continue
            lon, lat = nodes[nid]
            # walk forward
            for j in range(i + 1, len(node_ids)):
                nb = nodes.get(node_ids[j])
                if nb is None:
                    continue
                if haversine_m(lon, lat, nb[0], nb[1]) >= 15.0:
                    node_outgoing_bearings.setdefault(nid, []).append(
                        initial_bearing_deg(lon, lat, nb[0], nb[1])
                    )
                    break
            # walk backward
            for j in range(i - 1, -1, -1):
                nb = nodes.get(node_ids[j])
                if nb is None:
                    continue
                if haversine_m(lon, lat, nb[0], nb[1]) >= 15.0:
                    node_outgoing_bearings.setdefault(nid, []).append(
                        initial_bearing_deg(lon, lat, nb[0], nb[1])
                    )
                    break

    junctions: list[Junction] = []
    for nid, count in node_way_count.items():
        if count < 2 or nid not in node_outgoing_bearings:
            continue
        bearings = cluster_bearings(node_outgoing_bearings[nid], tol_deg=15.0)
        if len(bearings) < min_legs:
            continue
        lon, lat = nodes[nid]
        junctions.append(
            Junction(
                osm_node_id=nid,
                lon=lon,
                lat=lat,
                leg_bearings=sorted(bearings),
            )
        )

    # Attach crossings within 50m of each junction.
    for j in junctions:
        for c in crossings:
            if haversine_m(j.lon, j.lat, c.lon, c.lat) <= 50.0:
                j.crossings.append(c)

    return junctions


def rank_candidates(
    junctions: list[Junction],
    min_perpendicularity: float = 0.5,
    min_legs_with_crossing: int = 2,
) -> list[Junction]:
    """Filter and sort junctions by a combined OSM-only pre-score."""
    out = []
    for j in junctions:
        perp = j.perpendicularity_score()
        if perp < min_perpendicularity:
            continue
        n_cross_legs = j.n_legs_with_crossing()
        if n_cross_legs < min_legs_with_crossing:
            continue
        out.append((perp, n_cross_legs, j))
    out.sort(key=lambda t: (t[1], t[0]), reverse=True)
    return [j for _, _, j in out]
