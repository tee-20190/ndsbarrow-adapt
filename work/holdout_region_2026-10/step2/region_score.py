"""HOLDOUT-REGION-1 Step 2a (a): region-manifest scorer.

Ports score_holdout.ranked_pool to a region manifest and reuses the frozen score_holdout.recall_at unchanged, so the
allowance logic (first background proposal beyond floor(d x background core km2)), the 15 m reach, cur.nms (60 m,
score desc then id) and cur.rank_key are the frozen English ones. Stage-2 score() is not used (88 English parts).

Manifest (schema region_manifest_v1):
  tiles: [{id, role: positive|background, cluster, core: [[x0, y0, x1, y1], ...]}]   half-open boxes; a proposal
         counts for a tile when x0 <= easting < x1 and y0 <= northing < y1 for any of its boxes (merged tiles list
         several boxes); background core area = union area of its boxes.
  parts: [{id, class: barrow|enclosure, home: tile id, geometry: GeoJSON}]         primary parts only.
Objects: {tile id: [{id, easting, northing, score, mean_probability, area_m2, ...}]}; every tile needs an entry.
Geometry contract (check_geometry; fails closed with ManifestGeometryError, repair 2026-09-30): each tile has one
  non-empty cluster label (its super-block owner) and finite boxes with x0 < x1, y0 < y1; cores of distinct tiles are
  disjoint whatever their roles (overlapping positive tiles are merged into one multi-box tile, packet "Evaluation
  geometry"); summed background area must equal the geometric union of all background boxes. overlap_policy
  "frozen_english_holdout" is set only by english_manifest(): it accepts exactly the recorded English overlaps below.

score_files(manifest_path, proposals_path) scores a PROPOSALS file and refuses one made on another manifest (sha256).

  python -B work/holdout_region_2026-10/step2/region_score.py english   # regression on recorded English objects
(WSL adaf-h2h-venv, repo root)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in ("work/tta_ensemble_2026-09-24", "work/next_experiment_2026-09-23", "work/campaign_2026-09-22"):
    sys.path.insert(0, str(ROOT / _p))
import r3_curation as cur  # noqa: E402  frozen: nms, rank_key, window_bounds, EDGE_M
import score_holdout as shd  # noqa: E402  frozen: recall_at, REACH_M, DENSITIES

REACH_M = shd.REACH_M
DENSITIES = shd.DENSITIES
DESCRIPTIVE_DENSITIES = shd.DESCRIPTIVE_DENSITIES
CLASSES = ("barrow", "enclosure")
SUPERBLOCK_M = 20000.0


def in_core(o: dict, tile: dict) -> bool:
    e, n = o["easting"], o["northing"]
    return any(x0 <= e < x1 and y0 <= n < y1 for x0, y0, x1, y1 in tile["core"])


def core_area_km2(tile: dict) -> float:
    return unary_union([box(*b) for b in tile["core"]]).area / 1e6


class ManifestGeometryError(ValueError):
    """Evaluation cores are malformed or not uniquely owned; the manifest must not be scored."""


DISJOINT, FROZEN_ENGLISH = "disjoint", "frozen_english_holdout"
# Frozen English TRAIN-holdout windows (score_holdout.setup sums negative-window inner areas, no union): exactly these
# six negative pairs overlap, so 0.197147 of 33.380179 km2 is counted twice (union 33.183032 km2); 250 positive pairs
# overlap too, each within one 20 km super-block; no positive/negative pair overlaps (checked 2026-09-30). The recorded
# English primaries depend on that convention, so english_manifest() keeps it; no other manifest may.
ENGLISH_BACKGROUND_OVERLAPS = frozenset(frozenset(p) for p in (
    ("NHLE-1016911-p0-neg2", "NHLE-1017706-p0-neg0"), ("NHLE-1016911-p0-neg2", "NHLE-1010402-p0-neg0"),
    ("NHLE-1017706-p0-neg0", "NHLE-1010402-p0-neg0"), ("NHLE-1011694-p0-neg2", "NHLE-1002680-p0-neg2"),
    ("NHLE-1007694-p0-neg2", "NHLE-1008144-p0-neg2"), ("NHLE-1015149-p0-neg2", "NHLE-1016215-p2-neg2")))
ENGLISH_DOUBLE_COUNTED_KM2 = 0.197147


def core_boxes(tiles: list[dict]) -> list[tuple[int, tuple[float, ...]]]:
    """(tile index, box) for every core box; one super-block label per tile and well-formed half-open boxes."""
    out = []
    for i, t in enumerate(tiles):
        if not isinstance(t.get("cluster"), str) or not t["cluster"]:
            raise ManifestGeometryError(f"tile {t['id']}: cluster must be one non-empty super-block label")
        if not t["core"]:
            raise ManifestGeometryError(f"tile {t['id']}: empty core")
        for b in t["core"]:
            v = tuple(float(x) for x in b)
            if len(v) != 4 or not all(map(math.isfinite, v)) or not (v[0] < v[2] and v[1] < v[3]):
                raise ManifestGeometryError(f"tile {t['id']}: core box {b} is not finite with x0 < x1 and y0 < y1")
            out.append((i, v))
    return out


def core_overlaps(boxes: list[tuple[int, tuple[float, ...]]]) -> set[tuple[int, int]]:
    """Tile index pairs (i < j) of distinct tiles whose half-open cores share a point (touching edges do not)."""
    geoms = [box(*b) for _, b in boxes]
    found = set()
    for a, c in zip(*STRtree(geoms).query(geoms, predicate="intersects").tolist()):
        (i, p), (j, q) = boxes[a], boxes[c]
        if i < j and max(p[0], q[0]) < min(p[2], q[2]) and max(p[1], q[1]) < min(p[3], q[3]):
            found.add((i, j))
    return found


def check_geometry(manifest: dict) -> dict:
    """Fail closed unless evaluation cores are uniquely owned (module docstring); returns the geometry record."""
    tiles, policy = manifest["tiles"], manifest.get("overlap_policy", DISJOINT)
    if policy not in (DISJOINT, FROZEN_ENGLISH):
        raise ManifestGeometryError(f"unknown overlap_policy {policy!r}")
    boxes = core_boxes(tiles)
    kinds: dict[str, list[tuple[str, str]]] = {}
    for i, j in sorted(core_overlaps(boxes)):
        kinds.setdefault("/".join(sorted({tiles[i]["role"], tiles[j]["role"]})), []).append(
            (tiles[i]["id"], tiles[j]["id"]))
    summed = sum(core_area_km2(t) for t in tiles if t["role"] == "background")
    union = unary_union([box(*b) for i, b in boxes if tiles[i]["role"] == "background"]).area / 1e6
    record = {"overlap_policy": policy, "background_sum_km2": summed, "background_union_km2": union,
              "overlapping_pairs": {k: len(v) for k, v in sorted(kinds.items())}}
    if policy == FROZEN_ENGLISH:
        bg_pairs = frozenset(map(frozenset, kinds.get("background", [])))
        if ("background/positive" in kinds or bg_pairs != ENGLISH_BACKGROUND_OVERLAPS
                or not math.isclose(summed - union, ENGLISH_DOUBLE_COUNTED_KM2, abs_tol=1e-6)):
            raise ManifestGeometryError("overlap_policy frozen_english_holdout accepts only the recorded English "
                                        f"background overlaps; found {record['overlapping_pairs']}")
        return record
    if kinds:
        detail = "; ".join(f"{k}: {len(v)} pair(s), e.g. {v[0][0]} / {v[0][1]}" for k, v in sorted(kinds.items()))
        raise ManifestGeometryError(f"evaluation cores of distinct tiles overlap ({detail}): background cores must be "
                                    "disjoint from every other core, and overlapping positive tiles must be merged")
    if not math.isclose(summed, union, rel_tol=1e-9, abs_tol=1e-9):
        raise ManifestGeometryError(f"summed background area {summed!r} km2 != geometric union {union!r} km2")
    return record


def setup(manifest: dict) -> dict:
    """score_holdout.setup equivalent: the dict shape recall_at consumes, built from a manifest (geometry checked)."""
    tiles = manifest["tiles"]
    index = {t["id"]: i for i, t in enumerate(tiles)}
    if len(index) != len(tiles):
        raise ValueError("duplicate tile id")
    if {t["role"] for t in tiles} - {"positive", "background"}:
        raise ValueError("tile role must be positive or background")
    geometry = check_geometry(manifest)
    neg_area = {i: core_area_km2(t) for i, t in enumerate(tiles) if t["role"] == "background"}
    parts = {q["id"]: q for q in manifest["parts"]}
    if len(parts) != len(manifest["parts"]):
        raise ValueError("duplicate part id")
    home = {pid: index[q["home"]] for pid, q in parts.items()}
    if any(tiles[i]["role"] != "positive" for i in home.values()):
        raise ValueError("a part's home must be a positive tile")
    eligible = sorted(parts)
    klass = {pid: parts[pid]["class"] for pid in eligible}
    if set(klass.values()) - set(CLASSES):
        raise ValueError("unknown class")
    tree = STRtree([shape(parts[pid]["geometry"]).buffer(REACH_M) for pid in eligible])
    return {"wins": tiles, "index": index, "eligible": eligible, "home": home, "klass": klass, "tree": tree,
            "neg_area": neg_area, "cluster": [t["cluster"] for t in tiles], "geometry": geometry}


def ranked_pool(objects: dict[str, list[dict]], s: dict) -> list[dict]:
    """score_holdout.ranked_pool with the tile core in place of cur.inner (same NMS, rank key and hits)."""
    rows = []
    for t in s["wins"]:
        rows += [dict(o, window=t["id"]) for o in objects[t["id"]] if in_core(o, t)]
    pool = sorted(cur.nms(rows), key=cur.rank_key)
    for r in pool:
        wi = s["index"][r["window"]]
        r["wi"] = wi
        r["neg"] = wi in s["neg_area"]
        r["hits"] = [s["eligible"][i] for i in s["tree"].query(Point(r["easting"], r["northing"]),
                                                                 predicate="intersects")]
    return pool


def class_recall(hit: set, s: dict, k: str, wmult: np.ndarray | None = None) -> float:
    w = np.ones(len(s["wins"])) if wmult is None else wmult
    members = [p for p in s["eligible"] if s["klass"][p] == k]
    den = sum(w[s["home"][p]] for p in members)
    return float("nan") if den == 0 else sum(w[s["home"][p]] for p in members if p in hit) / den


def arm_summary(pool: list[dict], s: dict) -> dict:
    """score_holdout.arm_summary plus per-class recall and per-class primary."""
    area = sum(s["neg_area"].values())
    nneg = sum(r["neg"] for r in pool)
    out = {"pool": len(pool), "negative_proposals": nneg, "negative_per_km2": nneg / area if area else None,
           "eligible_parts": len(s["eligible"]),
           "parts_by_class": {k: sum(v == k for v in s["klass"].values()) for k in CLASSES}}
    for d in DENSITIES + DESCRIPTIVE_DENSITIES + (None,):
        rec, hit = shd.recall_at(pool, s, d)
        out["all" if d is None else f"d{d}"] = {
            "recall": rec, "parts": len(hit),
            "by_class": {k: sum(s["klass"][p] == k for p in hit) for k in CLASSES},
            "class_recall": {k: class_recall(hit, s, k) for k in CLASSES}}
    out["primary"] = float(np.mean([out[f"d{d}"]["recall"] for d in DENSITIES]))
    out["class_primary"] = {k: float(np.mean([out[f"d{d}"]["class_recall"][k] for d in DENSITIES]))
                            for k in CLASSES}
    return out


def score(manifest: dict, objects: dict[str, list[dict]]) -> dict:
    s = setup(manifest)
    return arm_summary(ranked_pool(objects, s), s)


class ManifestHashError(ValueError):
    """Proposals were produced on another manifest; they must not be scored against this one."""


def score_files(manifest_path: Path, proposals_path: Path) -> dict:
    """score() of a PROPOSALS file ({"manifest_sha256", "objects", ...}) on a manifest file. Refuses (fails closed)
    unless the proposals' manifest_sha256 equals the sha256 of the manifest file's bytes (added 2026-09-30 at the
    chain critic's request; score() on in-memory dicts is unchanged for the English regression)."""
    import hashlib
    data = Path(manifest_path).read_bytes()
    p = json.loads(Path(proposals_path).read_text(encoding="utf-8"))
    if p.get("manifest_sha256") != hashlib.sha256(data).hexdigest():
        raise ManifestHashError(f"{proposals_path}: proposals were made on another manifest (sha256 differs)")
    return score(json.loads(data), p["objects"])


# ---------------------------------------------------------------- English adapter (regression only)
def superblock_id(x: float, y: float) -> str:
    return f"SB_{int(x // SUPERBLOCK_M)}_{int(y // SUPERBLOCK_M)}"


def english_manifest(s_eng: dict | None = None) -> dict:
    """Region manifest from the frozen English TRAIN-holdout setup: tile core = window inner (EDGE_M), cluster = 20 km
    super-block of the window centre, parts = eligible primary parts at their home windows (in memory; private).
    Its windows overlap, so it carries overlap_policy frozen_english_holdout (see ENGLISH_BACKGROUND_OVERLAPS)."""
    s_eng = shd.setup() if s_eng is None else s_eng
    parts_geom = {q["id"]: q["geom"] for q in shd.r2_masks.rebuild_parts()[0]}
    tiles = []
    for w in s_eng["wins"]:
        x0, y0, x1, y1 = cur.window_bounds(w)
        tiles.append({"id": w["id"], "role": "positive" if w["role"] != "negative" else "background",
                      "cluster": superblock_id((x0 + x1) / 2, (y0 + y1) / 2),
                      "core": [[x0 + cur.EDGE_M, y0 + cur.EDGE_M, x1 - cur.EDGE_M, y1 - cur.EDGE_M]]})
    parts = [{"id": pid, "class": s_eng["klass"][pid], "home": s_eng["wins"][s_eng["home"][pid]]["id"],
              "geometry": mapping(parts_geom[pid])} for pid in s_eng["eligible"]]
    return {"schema": "region_manifest_v1", "overlap_policy": FROZEN_ENGLISH, "tiles": tiles, "parts": parts}


def english_recorded_objects() -> dict[str, dict]:
    """Recorded English TRAIN-holdout objects (B0 path) of the models whose primaries are recorded."""
    def rd(p: str) -> dict:
        return json.loads((ROOT / p).read_text(encoding="utf-8"))["objects"]
    exp1 = rd("outputs/counterfactual_2026-09-28/holdout/HOLDOUT_OBJECTS_2026092800_PRIVATE.json")
    adapt = rd("outputs/amazon_adapt_2026-09-28/HOLDOUT_OBJECTS_PRIVATE.json")
    rep = rd("outputs/amazon_adapt_rep_2026-09-28/HOLDOUT_OBJECTS_PRIVATE.json")
    return {"CONT": exp1["CONT"], "C_EXP1": exp1["C"], "C": adapt["C"], "A_BON": adapt["A_BON"],
            "A_RIB": adapt["A_RIB"], "A_FN3": adapt["A_FN3"], "C_REP": rep["C"], "A_RIB_REP": rep["A_RIB"],
            "CONT_ADAPT": adapt["CONT"], "CONT_REP": rep["CONT"]}


# recorded 4-dp primaries: amazon_adapt RESULT.md:101,109 and amazon_adapt_rep RESULT.md:47-50
RECORDED_PRIMARY = {"CONT": 0.7688, "C_EXP1": 0.7668, "C": 0.7872, "A_BON": 0.7339, "A_RIB": 0.7787,
                    "A_FN3": 0.7833, "C_REP": 0.7681, "A_RIB_REP": 0.7477}
RECORDED_BY_DENSITY = {"C_REP": (0.7213, 0.7727, 0.8103), "A_RIB_REP": (0.6779, 0.7648, 0.8004),
                       "C_EXP1": (0.7134, 0.7747, 0.8123), "CONT": (0.7312, 0.7747, 0.8004)}


def english_regression() -> dict:
    s_eng = shd.setup()
    man = english_manifest(s_eng)
    s = setup(man)
    out = {}
    for name, objs in english_recorded_objects().items():
        mine = arm_summary(ranked_pool(objs, s), s)
        frozen = shd.arm_summary(shd.ranked_pool(objs, s_eng), s_eng)
        out[name] = {"primary": mine["primary"], "frozen_primary": frozen["primary"],
                     "by_density": [mine[f"d{d}"]["recall"] for d in DENSITIES],
                     "class_primary": mine["class_primary"], "recorded": RECORDED_PRIMARY.get(name)}
    return {"eligible_parts": len(s["eligible"]), "background_km2": sum(s["neg_area"].values()),
            "clusters": len(set(s["cluster"])), "models": out}


if __name__ == "__main__":
    if sys.argv[1:] != ["english"]:
        raise SystemExit(__doc__)
    print(json.dumps(english_regression(), indent=1))
