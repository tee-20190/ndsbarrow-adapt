"""STAGE1-D1-V0 scoring (FREEZE.md, commit 1ea377ca). Runs only after step0.py wrote STEP0.json with step0 = PASS.

Benchmark v0 on the Step 0 eligible subset (all 717 Verden tiles / 728 individual barrows; Step 0 found no training
overlap). Logic copied from work/gpt6_pro_collab_2026-10-07/detector_week1/bench_v0.py (raw_pool, match_count, v0;
unchanged there): one-to-one maximum-cardinality matching, 15 m footprint reach (region_score tree), K = floor(b x
total scored km2) counting every in-core proposal, b = 0.5 / 1 / 2. RAW = every in-core proposal in rank_key order
(score, mean probability, area desc, proposal id asc; stable tile order); NMS60 = frozen region_score.ranked_pool.
Legacy IGNORE handling: unchanged (manifest geometry as built). Aggregates only go to RESULT.json.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/stage1_d1_v0_2026-10-07/score_v0.py
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "work/holdout_region_2026-10/step2"))
import region_score as rs  # noqa: E402  frozen, read-only
cur = rs.cur

B = (0.5, 1.0, 2.0)
SEEDS = tuple(range(2026092805, 2026092815))
IND = ROOT / "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json"
INF = ROOT / "outputs/fresh_region_2026-10-03/step2/INFER_MANIFEST_PRIVATE.json"
ADAF = ROOT / "outputs/fresh_region_2026-10-03/step2/score/individual/PROPOSALS_ADAF_barrow_PRIVATE.json"
ADAF_CKPT = "51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216"
S3 = ROOT / "outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z"
WEEK1 = ROOT / "work/gpt6_pro_collab_2026-10-07/detector_week1/BENCH_V0.json"
GATE = {"b": "0.5", "decoder": "RAW", "mean_diff_min": 0.030, "positive_min": 8}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def tile_keys(m: dict) -> list:
    return [[t["id"], t["role"], t["cluster"], t["core"]] for t in m["tiles"]]


def raw_pool(objs: dict, s: dict) -> list[dict]:
    rows = []
    for t in s["wins"]:
        rows += [dict(o, window=t["id"]) for o in objs[t["id"]] if rs.in_core(o, t)]
    return sorted(rows, key=cur.rank_key)


def match_count(rows: list[dict], s: dict) -> int:
    r_idx, c_idx = [], []
    for i, r in enumerate(rows):
        for j in s["tree"].query(Point(r["easting"], r["northing"]), predicate="intersects"):
            r_idx.append(i)
            c_idx.append(int(j))
    if not r_idx:
        return 0
    g = csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(len(rows), len(s["eligible"])))
    return int((maximum_bipartite_matching(g, perm_type="column") >= 0).sum())


def v0(objs: dict, s: dict, area: float) -> dict:
    n, out = len(s["eligible"]), {}
    for tag, pool in (("RAW", raw_pool(objs, s)), ("NMS60", rs.ranked_pool(objs, s))):
        rec = {"supply": len(pool)}
        for b in B:
            k = math.floor(b * area + 1e-9)
            m = match_count(pool[:k], s)
            rec[str(b)] = {"K": k, "matches": m, "recall": m / n, "ceiling_c": min(1.0, k / n),
                           "per_100_slots": 100.0 * m / k, "slots_filled": min(k, len(pool))}
        rec["primary"] = float(np.mean([rec[str(b)]["recall"] for b in B]))
        out[tag] = rec
    return out


def load_checked(p: Path, name: str, want_sha: str | None, ckpt: str, inf_sha: str, tiles: set, bad: list) -> dict:
    if want_sha and sha(p) != want_sha:
        bad.append(f"{name}: proposals sha256 differs from the session pin")
    d = rj(p)
    if d.get("model") != name or d.get("checkpoint_sha256") != ckpt:
        bad.append(f"{name}: model or checkpoint differs")
    if d.get("manifest_sha256") != inf_sha and d.get("inference_manifest_sha256") != inf_sha:
        bad.append(f"{name}: not made on the Verden inference manifest")
    if set(d["objects"]) != tiles:
        bad.append(f"{name}: tiles differ from the manifest")
    return d["objects"]


def systems(ind: dict, inf: dict, bad: list) -> tuple[dict, dict]:
    inf_sha, tiles = sha(INF), {t["id"] for t in ind["tiles"]}
    sv = rj(S3 / "SESSION_VERIFIED.json")
    if sv["status"] != "PASS":
        bad.append("session 3 not verified PASS")
    objs, prov = {}, {}
    objs["ADAF_barrow"] = load_checked(ADAF, "ADAF_barrow", None, ADAF_CKPT, inf_sha, tiles, bad)
    prov["ADAF_barrow"] = {"path": str(ADAF.relative_to(ROOT)), "sha256": sha(ADAF), "checkpoint_sha256": ADAF_CKPT}
    for s in SEEDS:
        name = f"D1F1_BARROW_L_{s}"
        key = f"dev/PROPOSALS_{name}_PRIVATE.json"
        p = S3 / "v1/results" / key
        objs[name] = load_checked(p, name, sv["proposals"][key], sv["trained"][name], inf_sha, tiles, bad)
        prov[name] = {"path": str(p.relative_to(ROOT)), "sha256": sv["proposals"][key],
                      "checkpoint_sha256": sv["trained"][name]}
    return objs, prov


def gate(res: dict) -> dict:
    ref = res["ADAF_barrow"][GATE["decoder"]][GATE["b"]]["recall"]
    diffs = {s: res[f"D1F1_BARROW_L_{s}"][GATE["decoder"]][GATE["b"]]["recall"] - ref for s in SEEDS}
    mean = float(np.mean(list(diffs.values())))
    pos = sum(d > 0 for d in diffs.values())
    ok = mean >= GATE["mean_diff_min"] - 1e-12 and pos >= GATE["positive_min"]
    return {**GATE, "adaf_barrow_recall": ref, "per_run_diff": {str(k): v for k, v in diffs.items()},
            "mean_diff": mean, "positive_diffs": pos, "verdict": "PASS" if ok else "FAIL"}


def summaries(res: dict) -> dict:
    out = {}
    for dec in ("RAW", "NMS60"):
        for b in B:
            ref = res["ADAF_barrow"][dec][str(b)]["recall"]
            d = [res[f"D1F1_BARROW_L_{s}"][dec][str(b)]["recall"] - ref for s in SEEDS]
            out[f"{dec}_{b}"] = {"adaf_barrow": ref, "mean_run": ref + float(np.mean(d)), "mean_diff": float(np.mean(d)),
                                 "min_diff": min(d), "max_diff": max(d), "positive": sum(x > 0 for x in d)}
    return out


def main() -> None:
    step0 = rj(HERE / "STEP0.json")
    if step0["step0"] != "PASS":
        raise SystemExit("Step 0 is not PASS: NOT_EVALUABLE, nothing is scored")
    ind, inf = rj(IND), rj(INF)
    if tile_keys(ind) != tile_keys(inf):
        raise SystemExit("individual and inference manifests differ in tiles")
    s = rs.setup(ind)
    area = unary_union([box(*b) for t in ind["tiles"] for b in t["core"]]).area / 1e6
    if len(s["eligible"]) != step0["eligible_subset"]["entities"]:
        raise SystemExit("eligible entity count differs from Step 0")
    bad: list[str] = []
    objs, prov = systems(ind, inf, bad)
    if bad:
        raise SystemExit(f"refused: {bad}")
    res = {m: v0(o, s, area) for m, o in objs.items()}
    for m, r in res.items():
        print(m, r["RAW"]["supply"], [r["RAW"][str(b)]["matches"] for b in B],
              [r["NMS60"][str(b)]["matches"] for b in B], flush=True)
    wk = rj(WEEK1)["verden"]["systems"]["ADAF_barrow"]["v0_individual"]
    def here(dec: str, k: str) -> float:
        v = res["ADAF_barrow"][dec][k]
        return v["recall"] if isinstance(v, dict) else v

    repro = {dec: {k: {"week1": wk[dec][k], "here": here(dec, k)} for k in ("0.5", "1.0", "2.0", "primary", "supply")}
             for dec in ("RAW", "NMS60")}
    repro_ok = all(abs(v["week1"] - v["here"]) < 1e-12 for d in repro.values() for v in d.values())
    out = {"schema": "stage1_d1_v0_result_v1", "freeze": "work/stage1_d1_v0_2026-10-07/FREEZE.md (commit 1ea377ca)",
           "individual_manifest_sha256": sha(IND), "inference_manifest_sha256": sha(INF),
           "region_score_sha256": sha(Path(rs.__file__)), "entities_N": len(s["eligible"]), "total_scored_km2": area,
           "K": {str(b): math.floor(b * area + 1e-9) for b in B}, "provenance": prov, "systems": res,
           "summaries": summaries(res), "gate": gate(res),
           "week1_reproduction": {"identical": repro_ok, "values": repro, "source": str(WEEK1.relative_to(ROOT)),
                                  "source_sha256": sha(WEEK1)}}
    (HERE / "RESULT.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"gate": out["gate"], "repro_identical": repro_ok}, indent=1))


if __name__ == "__main__":
    main()
