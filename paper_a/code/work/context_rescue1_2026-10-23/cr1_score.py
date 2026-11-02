"""CONTEXT-RESCUE-1 shared evaluation definition, part 2: the scorer, used by CONTEXT-RESCUE-1 and D1-SA stage B.

For one model's proposals (decoder objects with easting/northing in EPSG:25832 and OT-1's rank fields):
  1. Assign each proposal to the 5 km core holding its centroid (half-open squares); proposals outside the 210
     evaluation cores are dropped.
  2. Strict mask S-v1 (work/card_base/strict_mask.py `layers` and `rule`, unchanged; points and geometry in UTM 32N
     as `flag` uses them): masked proposals are removed. The OSM elements come from one snapshot per core
     (`core_query`): every way meeting S-v1's tag filters within the core + 60 m, plus the ways of the areas
     enclosing the core centre. Snapshots are fetched once, before any inference, and shared by every arm. Replies
     carrying an Overpass `remark` are rejected (CR0-6).
  3. Rank each core's proposals with OT-1's rank_key (scot_score, unchanged).
  4. Keep the core's top B (variant "plain") or PROX(B): the top ceil(0.7 B), then the best-ranked further proposals
     of the same core within 200 m of a kept one, then the next-ranked if fewer remain (variant "prox").
  5. One-to-one maximum matching at 15 m between all kept proposals and all truth records of the 210 cores together
     (scot_score.edges / max_matching, unchanged).
Output: the matched truth record ids and the per-core kept counts.

  python -B cr1_score.py selftest      (WSL venv: shapely, pyproj, scipy; no network, no terrain)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIV = ROOT / "outputs/context_rescue1_2026-10-23_PRIVATE"
OT_SCORE = ROOT / "work/open_transfer_2026-10-22/ot_score.py"
CARD = ROOT / "work/card_base"
CORE_M, REACH_M, R_PROX_M, F_FILL, AROUND_M = 5000, 15.0, 200.0, 0.3, 60.0
PINS = {"EVAL_CORES_PRIVATE.json": "1d82dee832af4e3251010308334028a62cab9585257a4f07b68955e255dfe54a",
        "EVAL_TRUTH_PRIVATE.json": "fe9575b91082511487f27c4e8ac40d3ec2a5bedb65c059c2d62e43ff9aee6317"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name: str) -> dict:
    p = PRIV / name
    if sha(p) != PINS[name]:
        raise SystemExit(f"{name} differs from its pin")
    return json.loads(p.read_text(encoding="utf-8"))


def scot():
    spec = importlib.util.spec_from_file_location("ot_score", OT_SCORE)
    ots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ots)
    return ots.scorer(raw=True)            # checks scot_score.py against OT-1's pin


def core_key(x: float, y: float) -> tuple[int, int]:
    return int(x // CORE_M), int(y // CORE_M)


def _lonlat():
    from pyproj import Transformer
    return Transformer.from_crs(25832, 4326, always_xy=True)


def core_query(i: int, j: int) -> str:
    """Overpass QL for one core's S-v1 snapshot (tag filters as strict_mask.body)."""
    sys.path.insert(0, str(CARD))
    import strict_mask as SM
    tr = _lonlat()
    x0, y0, x1, y1 = i * CORE_M - AROUND_M, j * CORE_M - AROUND_M, (i + 1) * CORE_M + AROUND_M, (j + 1) * CORE_M + AROUND_M
    lons, lats = zip(*(tr.transform(x, y) for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))))
    bb = f"{min(lats):.7f},{min(lons):.7f},{max(lats):.7f},{max(lons):.7f}"
    clon, clat = tr.transform((i + 0.5) * CORE_M, (j + 0.5) * CORE_M)
    r = SM._re
    return (f"[out:json][timeout:180];is_in({clat:.7f},{clon:.7f})->.a;"
            f'(way({bb})["highway"~"{r(SM.MAJOR | SM.MINOR)}"];way({bb})["junction"="roundabout"];'
            f'way({bb})["building"];way({bb})["railway"~"{r(SM.RAIL)}"];way({bb})["landuse"~"{r(SM.S_LANDUSE)}"];'
            f'way({bb})["leisure"~"{r(SM.S_LEISURE)}"];way({bb})["amenity"="parking"];'
            f'way(pivot.a)["landuse"~"{r(SM.S_LANDUSE)}"];way(pivot.a)["leisure"~"{r(SM.S_LEISURE)}"];'
            f'way(pivot.a)["amenity"="parking"];way(pivot.a)["building"];);out tags geom;')


def check_reply(res: dict) -> list[dict]:
    """A snapshot reply is usable only without an Overpass remark (timeouts and runtime errors set one)."""
    if res.get("remark"):
        raise RuntimeError(f"Overpass remark, reply rejected: {res['remark'][:200]}")
    return res.get("elements", [])


def mask_rule(elements: list[dict]) -> Callable[[float, float], str | None]:
    """S-v1 for points of one core: (x, y) in EPSG:25832 -> mask reason or None (strict_mask.flag's frames)."""
    from pyproj import Transformer
    from shapely.geometry import Point
    sys.path.insert(0, str(CARD))
    import strict_mask as SM
    to_utm = Transformer.from_crs(4326, 32632, always_xy=True)
    rule = SM.rule(SM.layers(elements, to_utm.transform))
    tr = _lonlat()

    def masked(x: float, y: float) -> str | None:
        lon, lat = tr.transform(x, y)
        return rule(Point(*to_utm.transform(round(lon, 7), round(lat, 7))))
    return masked


def prox(ranked: list[dict], B: int) -> list[dict]:
    """PROX(R = 200 m, f = 0.3) on one core's ranked, unmasked proposals (as dev_baseline_nl.prox)."""
    from scipy.spatial import cKDTree
    keep = ranked[:math.ceil((1 - F_FILL) * B)]
    rest = ranked[len(keep):]
    if not keep or not rest:
        return keep
    tree = cKDTree([[p["easting"], p["northing"]] for p in keep])
    near = [p for p in rest if tree.query([p["easting"], p["northing"]])[0] <= R_PROX_M]
    fill = near[:B - len(keep)]
    if len(fill) < B - len(keep):
        ids = {p["id"] for p in fill}
        fill += [p for p in rest if p["id"] not in ids][:B - len(keep) - len(fill)]
    return keep + fill


def select(props: list[dict], budgets: dict[tuple[int, int], int], masks: dict, variant: str, S) -> dict:
    """Steps 1-4: {core: kept proposals}."""
    by: dict[tuple[int, int], list[dict]] = {}
    for p in props:
        k = core_key(p["easting"], p["northing"])
        if k in budgets:
            by.setdefault(k, []).append(p)
    out = {}
    for k, B in budgets.items():
        ranked = sorted((p for p in by.get(k, []) if not masks[k](p["easting"], p["northing"])), key=S.rank_key)
        out[k] = ranked[:B] if variant == "plain" else prox(ranked, B)
    return out


def score(props: list[dict], cores: list[dict], records: list[dict], masks: dict, variant: str, S) -> dict:
    budgets = {(c["i"], c["j"]): c["B"] for c in cores}
    kept = select(props, budgets, masks, variant, S)
    flat = [p for k in sorted(kept) for p in kept[k]]
    sites = [{"e": r["x"], "n": r["y"]} for r in records]
    hit = S.max_matching(S.edges(flat, sites, REACH_M), len(sites))
    return {"variant": variant, "matched_ids": sorted(records[j]["id"] for j in hit),
            "kept_per_core": {f"{i}_{j}": len(v) for (i, j), v in sorted(kept.items())}, "kept_total": len(flat)}


def selftest() -> dict:
    S = scot()
    cores = [{"i": 100, "j": 1160, "B": 3}, {"i": 101, "j": 1160, "B": 2}]
    recs = [{"id": "r1", "x": 500010.0, "y": 5800010.0}, {"id": "r2", "x": 504995.0, "y": 5801000.0},
            {"id": "r3", "x": 505020.0, "y": 5801000.0}, {"id": "r4", "x": 503000.0, "y": 5803000.0}]

    def pr(pid, x, y, s):
        return {"id": pid, "easting": x, "northing": y, "score": s, "mean_probability": s, "area_m2": 50.0}
    props = [pr("a", 500012.0, 5800008.0, 0.9), pr("b", 505001.0, 5801000.0, 0.8),   # b sits in core 101, 6 m from r2
             pr("c", 505021.0, 5801001.0, 0.7), pr("d", 503005.0, 5803000.0, 0.2),
             pr("e", 502000.0, 5802000.0, 0.6), pr("f", 502100.0, 5802000.0, 0.5),
             pr("m", 500011.0, 5800011.0, 0.95), pr("z", 515000.0, 5801000.0, 0.99)]   # z: outside the cores
    masks = {(100, 1160): lambda x, y: "building" if (x, y) == (500011.0, 5800011.0) else None,
             (101, 1160): lambda x, y: None}
    plain = score(props, cores, recs, masks, "plain", S)
    px = score(props, cores, recs, masks, "prox", S)
    # core 100 unmasked ranking: a .9, e .6, f .5, d .2 -> plain keeps a, e, f; prox keeps ceil(2.1) = 3: a, e, f
    # core 101 ranking: b .8, c .7 -> both kept; b matches r2 across the core boundary, c matches r3
    kept = select(props, {(c["i"], c["j"]): c["B"] for c in cores}, masks, "plain", S)
    checks = {"masked_m_dropped": [p["id"] for p in kept[(100, 1160)]] == ["a", "e", "f"],
              "outside_dropped": plain["kept_total"] == 5,
              "plain_hits": plain["matched_ids"] == ["r1", "r2", "r3"], "prox_equal_here": px["matched_ids"] == ["r1", "r2", "r3"]}
    ranked = sorted([pr(f"p{k}", 500000.0 + 300 * k, 5800000.0, 1 - k / 10) for k in range(6)]
                    + [pr("near", 500150.0, 5800050.0, 0.05)], key=S.rank_key)
    pk = [p["id"] for p in prox(ranked, 4)]
    checks["prox_rule"] = pk == ["p0", "p1", "p2", "near"]          # ceil(2.8) = 3 kept, then the 200 m neighbour
    sys.path.insert(0, str(ROOT / "work/context_rescue0_2026-10-23"))
    import dev_baseline_nl as DB
    checks["prox_matches_dev_baseline"] = [p["id"] for p in DB.prox(ranked, 4)] == pk
    q = core_query(100, 1160)
    checks["query_has_filters"] = all(s in q for s in ('"junction"="roundabout"', "pivot.a", "out tags geom"))
    try:
        check_reply({"remark": "runtime error: Query timed out", "elements": []})
        checks["remark_rejected"] = False
    except RuntimeError:
        checks["remark_rejected"] = True
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        print(json.dumps(selftest(), indent=1))
