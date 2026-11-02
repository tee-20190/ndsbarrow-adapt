"""card_base (2026-10-23, CONTEXT-1 R2): strict modern-surface mask S for D1 lead selection, from OpenStreetMap.

CONTEXT-1 stage 0 (work/context1_2026-10-23/RESULT_STAGE0.md) measured on 653 Dutch EVAL barrows: S masks 1 record
(0.15 %) and 10.1 % of D1's matched-workload proposals (20 of 21 roundabout-island cards), while the NL-LEADS-1 A1
mask (40 m around roads and cycleways, all military/cemetery/garden land) hides 118 records (18.1 %). S replaces A1
for unfrozen D1 lead selection; A1/S-a style classes may still be shown to readers as flags, never as drops.
S masks a point only on surfaces where a surviving mound cannot sit at that point:
  island    inside an enclosed roundabout face of at most 5,000 m2       ring_way  <= 6 m from a roundabout way
  building  <= 3 m from a building        major  <= 8 m from motorway..tertiary (+links)
  minor     <= 4 m from unclassified/residential/service/living_street       rail  <= 5 m from an active railway
  poly      inside industrial/commercial/retail/construction/landfill/quarry/railway/garages/depot landuse,
            pitch/stadium/track/sports_centre/miniature_golf leisure, or amenity=parking
Never masked: cycleways, paths, tracks, residential landuse, cemeteries, military land, golf courses, gardens,
farmyards (barrows survive in such places). Rule code copied from work/context1_2026-10-23/ctx_stage0.py (sha256
74b64d00...). Distances are computed in the point's UTM zone.

  python -B strict_mask.py <cases.json> <out.json>     cases = [{"code", "lat", "lon"}, ...]; one request per 20
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
MINOR = {"unclassified", "residential", "service", "living_street"}
RAIL = {"rail", "light_rail", "tram", "subway", "narrow_gauge", "preserved", "monorail", "funicular"}
S_LANDUSE = {"industrial", "commercial", "retail", "construction", "landfill", "quarry", "railway", "garages", "depot"}
S_LEISURE = {"pitch", "stadium", "track", "sports_centre", "miniature_golf"}
S_DIST = {"ring": 6.0, "building": 3.0, "major": 8.0, "minor": 4.0, "rail": 5.0}
ISLAND_MAX_M2 = 5000.0
AROUND_M = 60
ORDER = (("island", None), ("ring_way", S_DIST["ring"]), ("building", S_DIST["building"]),
         ("major", S_DIST["major"]), ("minor", S_DIST["minor"]), ("rail", S_DIST["rail"]), ("poly", None))


def _re(values: set[str]) -> str:
    return "^(" + "|".join(sorted(values)) + ")$"


def body(lat: float, lon: float) -> str:
    """Per-point Overpass statements for overpass_batch.fetch_points: ways near the point, plus enclosing areas."""
    a = f"around:{AROUND_M},{lat},{lon}"
    return (f'is_in({lat},{lon})->.a;'
            f'(way({a})["highway"~"{_re(MAJOR | MINOR)}"];way({a})["junction"="roundabout"];way({a})["building"];'
            f'way({a})["railway"~"{_re(RAIL)}"];way({a})["landuse"~"{_re(S_LANDUSE)}"];'
            f'way({a})["leisure"~"{_re(S_LEISURE)}"];way({a})["amenity"="parking"];'
            f'way(pivot.a)["landuse"~"{_re(S_LANDUSE)}"];way(pivot.a)["leisure"~"{_re(S_LEISURE)}"];'
            f'way(pivot.a)["amenity"="parking"];way(pivot.a)["building"];);out tags geom;')


def layers(elements: list[dict], to_xy: Callable[[float, float], tuple[float, float]]) -> dict[str, list]:
    lay: dict[str, list] = {k: [] for k in ("ring_way", "building", "major", "minor", "rail", "poly")}
    for e in elements:
        geom, t = e.get("geometry") or [], e.get("tags", {})
        if len(geom) < 2:
            continue
        xy = [to_xy(g["lon"], g["lat"]) for g in geom]
        closed = xy[0] == xy[-1] and len(xy) >= 4
        if t.get("junction") == "roundabout":
            lay["ring_way"].append(LineString(xy))
        if "building" in t:
            lay["building"].append(Polygon(xy).buffer(0) if closed else LineString(xy))
        if t.get("highway") in MAJOR:
            lay["major"].append(LineString(xy))
        elif t.get("highway") in MINOR:
            lay["minor"].append(LineString(xy))
        if t.get("railway") in RAIL:
            lay["rail"].append(LineString(xy))
        if closed and (t.get("landuse") in S_LANDUSE or t.get("leisure") in S_LEISURE or t.get("amenity") == "parking"):
            lay["poly"].append(Polygon(xy).buffer(0))
    faces = list(polygonize(unary_union(lay["ring_way"]))) if lay["ring_way"] else []
    lay["island"] = [f for f in faces if f.area <= ISLAND_MAX_M2]
    return lay


def rule(lay: dict[str, list]) -> Callable[[Point], str | None]:
    trees = {k: STRtree(v) for k, v in lay.items() if v}

    def masked(pt: Point) -> str | None:
        for name, d in ORDER:
            if name not in trees:
                continue
            hit = (trees[name].query(pt, predicate="intersects") if d is None
                   else trees[name].query(pt, predicate="dwithin", distance=d))
            if len(hit):
                return name
        return None
    return masked


def utm(lat: float, lon: float) -> Transformer:
    zone = int((lon + 180) // 6) + 1
    return Transformer.from_crs(4326, (32600 if lat >= 0 else 32700) + zone, always_xy=True)


def flag(elements: list[dict], lat: float, lon: float) -> str | None:
    tf = utm(lat, lon)
    return rule(layers(elements, tf.transform))(Point(*tf.transform(lon, lat)))


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from overpass_batch import fetch_points
    cases = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    replies = fetch_points({c["code"]: (c["lat"], c["lon"]) for c in cases}, body)
    out = [{"code": c["code"], "s_mask": flag(replies[c["code"]].get("elements", []), c["lat"], c["lon"])}
           for c in cases]
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"cases": len(out), "masked": sum(bool(o["s_mask"]) for o in out)}))


if __name__ == "__main__":
    main()
