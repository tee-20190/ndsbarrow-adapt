"""Task 1 (detector_week1): all-specialist miss decomposition + union / oracle ceilings on open development sets.

Read-only reuse of work/gap_analysis_2026-10-03/gap_region.py (imported unchanged: part_flags, load_objects, miss_kind)
and the frozen HR1 region_score (15 m reach, 60 m greedy NMS, background-only budget floor(d x background km2)).
Writes aggregate counts only (no part ids, no coordinates) to UNION_DECOMP.json beside this file.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/gpt6_pro_collab_2026-10-07/detector_week1/decomp.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from shapely.geometry import Point, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "work/gap_analysis_2026-10-03"))
import gap_region as gr  # noqa: E402  unchanged
rs, shd, cur = gr.rs, gr.shd, gr.cur

DENS = (0.5, 1.0, 2.0)
JC = "outputs/joint_campaign_2026-10-03/chain_runs"
S1, S4 = f"{JC}/run_s1_20261003T093950Z/v1/results", f"{JC}/run_s4_20261003T192115Z/v1/results"
SSTAR = "J_2026092803"  # FRESH-1 freeze sstar (work/fresh_region_2026-10-03/step2/freeze/FREEZE.json)
VS = "outputs/fresh_region_2026-10-03/step2/score"

SETS = {
    "england_holdout": {"manifest": None, "models": {
        "ADAF_AO": f"{S1}/refs_holdout/HOLDOUT_OBJECTS_ADAF_AO_PRIVATE.json",
        "ADAF_barrow": f"{S1}/refs_holdout/HOLDOUT_OBJECTS_ADAF_barrow_PRIVATE.json",
        "ADAF_enclosure": f"{S1}/refs_holdout/HOLDOUT_OBJECTS_ADAF_enclosure_PRIVATE.json",
        "ADAF_ringfort": f"{S1}/refs_holdout/HOLDOUT_OBJECTS_ADAF_ringfort_PRIVATE.json",
        "CONT": f"{S1}/refs_holdout/HOLDOUT_OBJECTS_CONT_PRIVATE.json",
        "S*": f"{S4}/runs/{SSTAR}/HOLDOUT_OBJECTS_PRIVATE.json"},
        "legacy": {  # the inputs GAP_TABLE.md used (REFERENCE_SOURCES.json role recorded_crosscheck)
        "ADAF_AO_gap": "outputs/campaign_2026-09-22/holdout_diag/AO_holdout/OBJECTS_PRIVATE.json",
        "CONT_gap": "outputs/counterfactual_2026-09-28/holdout/HOLDOUT_OBJECTS_2026092800_PRIVATE.json#CONT"}},
    "wales_dev": {"manifest": "work/holdout_region_2026-10/chain/candidates/DEV_MANIFEST_CANDIDATE.json", "models": {
        "ADAF_AO": "outputs/holdout_region_2026-10/chain_runs/t2_s1_20261001T003208Z/v1/results/PROPOSALS_ADAF_AO_PRIVATE.json",
        "ADAF_barrow": f"{S1}/wales/PROPOSALS_ADAF_barrow_PRIVATE.json",
        "ADAF_enclosure": f"{S1}/wales/PROPOSALS_ADAF_enclosure_PRIVATE.json",
        "ADAF_ringfort": f"{S1}/wales/PROPOSALS_ADAF_ringfort_PRIVATE.json",
        "CONT": "outputs/holdout_region_2026-10/chain_runs/t2_s1_20261001T003208Z/v1/results/PROPOSALS_CONT_PRIVATE.json",
        "S*": f"{S4}/wales/PROPOSALS_{SSTAR}_PRIVATE.json"}},
    "ireland_dev": {"manifest": "outputs/hr3_ireland_2026-10-02/step2/DEV_MANIFEST_PRIVATE.json", "models": {
        "ADAF_AO": "outputs/hr3_ireland_2026-10-02/step2/chain_runs/run_s1_20261002T235042Z/v1/results/PROPOSALS_ADAF_AO_PRIVATE.json",
        "ADAF_barrow": f"{S1}/ireland/PROPOSALS_ADAF_barrow_PRIVATE.json",
        "ADAF_enclosure": f"{S1}/ireland/PROPOSALS_ADAF_enclosure_PRIVATE.json",
        "ADAF_ringfort": f"{S1}/ireland/PROPOSALS_ADAF_ringfort_PRIVATE.json",
        "CONT": "outputs/hr3_ireland_2026-10-02/step2/chain_runs/run_s1_20261002T235042Z/v1/results/PROPOSALS_CONT_PRIVATE.json",
        "S*": f"{S4}/ireland/PROPOSALS_{SSTAR}_PRIVATE.json"}},
}
for unit in ("group", "individual"):
    SETS[f"verden_{unit}"] = {"manifest": f"outputs/fresh_region_2026-10-03/step0/{unit.upper()}_MANIFEST_PRIVATE.json",
                              "models": {"ADAF_AO": f"{VS}/{unit}/PROPOSALS_ADAF_AO_PRIVATE.json",
                                         "ADAF_barrow": f"{VS}/{unit}/PROPOSALS_ADAF_barrow_PRIVATE.json",
                                         "CONT": f"{VS}/{unit}/PROPOSALS_CONT_PRIVATE.json",
                                         "S*": f"{VS}/{unit}/PROPOSALS_{SSTAR}_PRIVATE.json"}}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidates(objs: dict, s: dict, tag: str) -> list[dict]:
    """Every in-core pre-NMS proposal of one model with the parts it reaches (15 m) and its charge class."""
    out = []
    for t in s["wins"]:
        for o in objs.get(t["id"], []):
            if rs.in_core(o, t):
                hits = {s["eligible"][i] for i in s["tree"].query(Point(o["easting"], o["northing"]),
                                                                   predicate="intersects")}
                out.append({"x": o["easting"], "y": o["northing"], "hits": hits,
                            "neg": s["index"][t["id"]] in s["neg_area"], "src": tag})
    return out


def greedy_cover(cands: list[dict], budget_free: bool, allow_neg: float, k_total: float, nms_m: float | None) -> int:
    """Greedy max coverage over hitting proposals (a lower bound on the oracle optimum). budget_free: positive-tile
    proposals cost nothing and background-tile ones count against allow_neg (frozen convention); otherwise every
    proposal costs 1 against k_total. nms_m: chosen proposals must be > nms_m apart, so the frozen NMS keeps them all."""
    pool = [c for c in cands if c["hits"]]
    covered, n_chosen, neg_used = set(), 0, 0
    while pool:
        if not budget_free and n_chosen + 1 > k_total:
            break
        best, gain = None, 0
        for c in pool:
            g = len(c["hits"] - covered)
            if g <= gain or (budget_free and c["neg"] and neg_used + 1 > allow_neg):
                continue
            best, gain = c, g
        if best is None:
            break
        n_chosen += 1
        covered |= best["hits"]
        neg_used += best["neg"]
        pool = [c for c in pool if c["hits"] - covered and (nms_m is None or
                math.hypot(c["x"] - best["x"], c["y"] - best["y"]) > nms_m)]
    return len(covered)


def topk_all_charged(objs: dict, s: dict, k: int) -> int:
    """A model's own frozen ranked pool (NMS 60 m) truncated at its first k proposals, every proposal charged."""
    pool = rs.ranked_pool(objs, s)
    return len({p for r in pool[:k] for p in r["hits"]})


def run_set(name: str, spec: dict) -> dict:
    if spec["manifest"] is None:
        m, msha = rs.english_manifest(), "english_manifest()"
    else:
        mp = ROOT / spec["manifest"]
        m, msha = json.loads(mp.read_text(encoding="utf-8")), sha(mp)
    s = rs.setup(m)
    n = len(s["eligible"])
    bg = float(sum(s["neg_area"].values()))
    scored_union = unary_union([box(*b) for t in m["tiles"] for b in t["core"]]).area / 1e6
    out = {"manifest_sha256": msha, "eligible_parts": n, "background_km2": bg, "scored_km2_union": scored_union,
           "by_class_n": {k: sum(v == k for v in s["klass"].values()) for k in rs.CLASSES}, "inputs": {}, "models": {}}
    flags, cands, objs_all = {}, [], {}
    models = dict(spec["models"], **spec.get("legacy", {}))
    for mn, ps in models.items():
        path, _, key = ps.partition("#")
        p = ROOT / path
        if not p.is_file():
            out["inputs"][mn] = {"path": ps, "status": "MISSING"}
            continue
        out["inputs"][mn] = {"path": ps, "sha256": sha(p)}
        objs = gr.load_objects(p, key, s)
        f = gr.part_flags(objs, s)
        flags[mn], objs_all[mn] = f, objs
        kinds = {str(d): {"HIT": 0, "RANK": 0, "NMS": 0, "MISSING": 0} for d in DENS}
        by_cls = {k: {str(d): {"HIT": 0, "RANK": 0, "NMS": 0, "MISSING": 0} for d in DENS} for k in rs.CLASSES}
        for pid in s["eligible"]:
            for d in DENS:
                kd = gr.miss_kind(pid, f, d)
                kinds[str(d)][kd] += 1
                by_cls[s["klass"][pid]][str(d)][kd] += 1
        out["models"][mn] = {"kinds": kinds, "kinds_by_class": by_cls, "pre_nms_parts": len(f["pre"]),
                             "post_nms_parts": len(f["post"]), "pool": f["pool"], "neg_per_km2": f["neg_per_km2"],
                             "recall": {str(d): len(f["hits"][d]) / n for d in DENS},
                             "topk_all_charged_parts": {str(d): topk_all_charged(objs, s, math.floor(d * scored_union + 1e-9))
                                                        for d in DENS}}
        if mn in spec["models"]:
            cands += candidates(objs, s, mn)
    main = [x for x in spec["models"] if x in flags]
    pre_u = set().union(*(flags[x]["pre"] for x in main))
    post_u = set().union(*(flags[x]["post"] for x in main))
    union = {"models": main, "pre_nms_any": len(pre_u), "post_nms_any": len(post_u),
             "pre_nms_by_class": {k: sum(s["klass"][p] == k for p in pre_u) for k in rs.CLASSES},
             "hit_any_at_d": {str(d): len(set().union(*(flags[x]["hits"][d] for x in main))) for d in DENS},
             "no_model_pre_nms": n - len(pre_u)}
    oracle = {}
    for d in DENS:
        allow = math.floor(d * bg + 1e-9)
        k = math.floor(d * scored_union + 1e-9)
        oracle[str(d)] = {
            "frozen_charge_upper": len(pre_u),
            "frozen_charge_greedy_nms_feasible": greedy_cover(cands, True, allow, math.inf, rs.cur.NMS_M),
            "all_charged_K": k,
            "all_charged_upper": min(len(pre_u), sum(sorted((len(c["hits"]) for c in cands), reverse=True)[:k])),
            "all_charged_greedy_nms_feasible": greedy_cover(cands, False, math.inf, k, rs.cur.NMS_M),
            "all_charged_greedy_no_nms": greedy_cover(cands, False, math.inf, k, None)}
    out["union"], out["oracle"] = union, oracle
    return out


def main() -> None:
    only = sys.argv[1:]
    res = {}
    for name, spec in SETS.items():
        if only and name not in only:
            continue
        res[name] = run_set(name, spec)
        print(name, json.dumps({"n": res[name]["eligible_parts"], "union": res[name]["union"]}), flush=True)
    tag = "_".join(only) if only else "ALL"
    (HERE / f"UNION_DECOMP_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
