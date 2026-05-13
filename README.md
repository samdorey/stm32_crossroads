# crosswalk-pcb

Finds American intersections whose continental crosswalk markings most closely
resemble surface-mount PCB component footprints (Quad Flat Package, Small
Outline Integrated Circuit, etc.).

## Pipeline

```
OSM (Overpass API)          Aerial imagery (Mapbox/Esri)
    |                              |
    v                              v
find_intersections.py        fetch_centered()
    |                              |
    | candidates.json              | 768x768 crop per intersection
    v                              v
pipeline.py ───────────────────────┘
    |
    ├─ find_paint_mask()     HSV threshold: white (V≥180, S≤80) OR
    |                        yellow (H 15-35, S≥60, V≥170)
    |
    ├─ extract_stripes()     Connected components → PCA orientation →
    |                        filter by metric dimensions (0.25-1.2m wide,
    |                        1.5-6m long, aspect ratio ≥2.5)
    |
    ├─ group_stripes_into_arrays()   Cluster by:
    |                                 1. stripe orientation (±12°)
    |                                 2. centroid collinearity along
    |                                    perpendicular axis (±18 px)
    |                                Minimum 3 stripes per array.
    |
    ├─ score_match()         Per intersection:
    |   ├─ cluster arrays by bearing from image center (±35°) → "sides"
    |   ├─ drop arrays >300 px from center (rooftop noise)
    |   ├─ classify: 4 sides → QFP, 2 opposite → SOIC, else → Other
    |   └─ similarity = product of:
    |       • side_count_match     (4 sides = 1.0, 2 opposite = 0.85)
    |       • pitch_regularity     (1 − 2·CV, averaged over sides)
    |       • pin_count_consistency (√(min/max) of opposing-side stripe counts)
    |       • body_aspect_ratio    (min/max of opposing-side center distances)
    |       • richness             (min(1, total_stripes / 32))
    |
    └─ ranked.json + index.html gallery
```

## OSM pre-filter

Queries Overpass for nodes tagged `highway=crossing` with `crossing` ∈
{marked, zebra, uncontrolled, traffic_signals} plus all road ways in the
bounding box. Builds a road graph, identifies junction nodes (shared by ≥2
ways), clusters outgoing bearings into legs, and scores perpendicularity:

- 4-leg junction: mean |gap − 90°| across consecutive bearing gaps
- 3-leg T-junction: deviation from the 90°/90°/180° pattern

Candidates must have perpendicularity ≥0.5 and marked crossings on ≥2 legs
within 50 m of the junction node.

## Imagery sources

| Source | Resolution at z19 | Auth | Notes |
|---|---|---|---|
| `mapbox_satellite` | ~0.12 m/px (@2x tiles) | `MAPBOX_TOKEN` env var or `.env` | Best metro resolution. Free tier: 750k tiles/month. |
| `esri_world` | ~0.24 m/px | None | Maxar/Vexcel mosaic. Free for low-volume non-commercial. |
| `usgs_imagery` | ~0.60 m/px (NAIP) | None | 404s above z16 in practice. |

Tiles are cached on disk under `data/tile_cache/` to avoid re-fetching.

## Usage

```bash
pip install -r requirements.txt
# Put your Mapbox public token in .env (gitignored):
echo "MAPBOX_TOKEN=pk.xxx" > .env

# 1. Compare imagery sources (eyeball quality)
python scripts/compare_imagery.py --zoom 19 --size 768

# 2. Find OSM candidate intersections
python scripts/find_intersections.py --area sf-sunset
# or: python scripts/find_intersections.py --bbox 37.755,-122.51,37.775,-122.47

# 3. Run full pipeline
python scripts/pipeline.py \
    --candidates data/candidates/sf-sunset.json \
    --source mapbox_satellite --zoom 19 --size 768 --top 50

# 4. Browse results
open data/pipeline_out/index.html

# Offline sanity check (no network)
python -m scripts.synthetic_test
```

Pre-defined bounding boxes: `sf-soma`, `sf-mission`, `sf-sunset`, `sf-richmond`, `sf-all`.

## Current results (SF Sunset, top 3)

| Similarity | Class | Stripes | Sides | Location |
|---|---|---|---|---|
| 0.753 | QFP | 41 | 4 (7/9/9/7) | 37.7589, -122.4994 |
| 0.675 | QFP | 34 | 4 (6/6/9/7) | 37.7633, -122.4836 |
| 0.656 | QFP | 39 | 4 (7/7/8/9) | 37.7569, -122.5014 |

## Known limitations

- **Yellow paint**: Detected via HSV hue band (15-35). Some worn yellow stripes
  in shadow fall outside this range.
- **False positives**: White rooftop edges, parking lines, and bus-lane markings
  can be picked up as stripes. The 300 px distance filter mitigates but doesn't
  eliminate.
- **OSM junction model**: Requires a single shared node for junction detection.
  Intersections mapped as areas or with separate way-end nodes are missed.
- **Shadows**: Building shadows reduce paint brightness below threshold on one
  side, causing asymmetric stripe counts.
- **SOMA-style rotated grids**: Detection works at any angle, but scoring is
  dragged down by non-crosswalk road markings (bike lane chevrons, etc.) common
  in denser neighborhoods.
