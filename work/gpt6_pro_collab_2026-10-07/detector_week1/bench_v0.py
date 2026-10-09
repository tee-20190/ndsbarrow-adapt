"""Task 3 (detector_week1): benchmark v0 sensitivity rescore of saved Oldenburg (ADAPT-NDS-1 CONFIRM) and Verden
(FRESH-1) proposals. Legacy numbers are recomputed with the frozen region_score beside the v0 numbers; nothing in
the project is overwritten. Aggregates only (no ids, no coordinates) go to BENCH_V0.json beside this file.

v0 rules: entities = individual register barrows (the register defines no barrow groups: objektnummer is unique per
Gemarkung for every Grabhuegel record; Grabhuegelfeld records stay IGNORE as in the legacy manifests); a proposal can
match an entity when it lies within 15 m of the entity footprint polygon (region_score's buffered tree); one-to-one
maximum-cardinality bipartite matching between the top-K proposals and the entities; K = floor(b x total scored area)
counting ALL in-core proposals (positive and background tiles), b = 0.5, 1, 2 per km2. Pools: NMS60 = the frozen ranked
pool (in-core, 60 m greedy NMS, rank_key); RAW = every in-core proposal in rank_key order (no NMS).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/gpt6_pro_collab_2026-10-07/detector_week1/bench_v0.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from shapely.geometry import Point, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "work/holdout_region_2026-10/step2"))
sys.path.insert(0, str(ROOT / "work/fusion_2026-10-03"))
import region_score as rs  # noqa: E402  frozen, read-only
import fuse_eval as fe  # noqa: E402  FUSION-1 INTERLEAVE, read-only
shd, cur = rs.shd, rs.cur

B = (0.5, 1.0, 2.0)
VS = "outputs/fresh_region_2026-10-03/step2/score"
OS = "outputs/adapt_nds_2026-10-05/score/confirm"
SEEDS = ("2026092802", "2026092803", "2026092804")
REGIONS = {
    "verden": {"individual": "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json",
               "group": "outputs/fresh_region_2026-10-03/step0/GROUP_MANIFEST_PRIVATE.json",
               "dir": VS, "models": ["ADAF_AO", "ADAF_barrow", "CONT", "J_2026092803"],
               "deploy": ("ADAF_AO", "J_2026092803")},
    "oldenburg": {"individual": "outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json",
                  "group": "outputs/adapt_nds_2026-10-05/confirm/run/GROUP_MANIFEST_PRIVATE.json",
                  "dir": OS, "models": ["ADAF_AO", "ADAF_barrow", "CONT"] + [f"A_NDS_{s}" for s in SEEDS]
                  + [f"C_NDS_{s}" for s in SEEDS] + [f"J_{s}" for s in SEEDS], "deploy": None},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(unit_dir: Path, name: str) -> dict:
    return json.loads((unit_dir / f"PROPOSALS_{name}_PRIVATE.json").read_text(encoding="utf-8"))["objects"]


def raw_pool(objs: dict, s: dict) -> list[dict]:
    rows = []
    for t in s["wins"]:
        rows += [dict(o, window=t["id"]) for o in objs[t["id"]] if rs.in_core(o, t)]
    return sorted(rows, key=cur.rank_key)


def match_count(rows: list[dict], s: dict) -> int:
    """Maximum-cardinality one-to-one matching of rows (proposals) to eligible parts within 15 m of the footprint."""
    r_idx, c_idx = [], []
    for i, r in enumerate(rows):
        for j in s["tree"].query(Point(r["easting"], r["northing"]), predicate="intersects"):
            r_idx.append(i)
            c_idx.append(int(j))
    if not r_idx:
        return 0
    g = csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(len(rows), len(s["eligible"])))
    m = maximum_bipartite_matching(g, perm_type="column")
    return int((m >= 0).sum())


def legacy(objs: dict, s: dict) -> dict:
    pool = rs.ranked_pool(objs, s)
    rec = {str(d): shd.recall_at(pool, s, d)[0] for d in B}
    rec["primary"] = float(np.mean([rec[str(d)] for d in B]))
    return rec


def v0(objs: dict, s: dict, area: float) -> dict:
    n = len(s["eligible"])
    out = {}
    for tag, pool in (("NMS60", rs.ranked_pool(objs, s)), ("RAW", raw_pool(objs, s))):
        rec = {}
        for b in B:
            k = math.floor(b * area + 1e-9)
            rec[str(b)] = match_count(pool[:k], s) / n
        rec["primary"] = float(np.mean([rec[str(b)] for b in B]))
        rec["supply"] = len(pool)
        out[tag] = rec
    return out


def deploy_objs(a: dict, b: dict, s: dict) -> dict:
    return fe.fused_objects(fe.order("INTERLEAVE", fe.own_pool(a, s, "A"), fe.own_pool(b, s, "S")), s)


def run(name: str, spec: dict) -> dict:
    ip, gp = ROOT / spec["individual"], ROOT / spec["group"]
    mi, mg = json.loads(ip.read_text(encoding="utf-8")), json.loads(gp.read_text(encoding="utf-8"))
    si, sg = rs.setup(mi), rs.setup(mg)
    area = unary_union([box(*b) for t in mi["tiles"] for b in t["core"]]).area / 1e6
    out = {"individual_manifest_sha256": sha(ip), "group_manifest_sha256": sha(gp), "entities": len(si["eligible"]),
           "groups": len(sg["eligible"]), "total_scored_km2": area,
           "background_km2": float(sum(si["neg_area"].values())),
           "K": {str(b): math.floor(b * area + 1e-9) for b in B}, "systems": {}}
    objs_i = {m: load(ROOT / spec["dir"] / "individual", m) for m in spec["models"]}
    objs_g = {m: load(ROOT / spec["dir"] / "group", m) for m in spec["models"]}
    if spec["deploy"]:
        a, b = spec["deploy"]
        objs_i["DEPLOY"] = deploy_objs(objs_i[a], objs_i[b], si)
        objs_g["DEPLOY"] = deploy_objs(objs_g[a], objs_g[b], sg)
    for m in objs_i:
        out["systems"][m] = {"legacy_group": legacy(objs_g[m], sg), "legacy_individual": legacy(objs_i[m], si),
                             "v0_individual": v0(objs_i[m], si, area)}
        print(name, m, json.dumps({k: round(v["primary"], 4) for k, v in
                                   {"lg": out["systems"][m]["legacy_group"], "li": out["systems"][m]["legacy_individual"],
                                    "v0nms": out["systems"][m]["v0_individual"]["NMS60"],
                                    "v0raw": out["systems"][m]["v0_individual"]["RAW"]}.items()}), flush=True)
    return out


def main() -> None:
    res = {n: run(n, sp) for n, sp in REGIONS.items()}
    (HERE / "BENCH_V0.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
