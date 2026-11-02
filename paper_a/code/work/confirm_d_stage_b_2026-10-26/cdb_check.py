"""CONFIRM-D stage B: independent recount + FREEZE §8 supporting report. Writes CHECK_B_K<K>.json (counts only).

Recount from the private hit sets, without cr1_analysis / cd_analysis: Δ_k = (|D_k ∩ pop| − |A_k ∩ pop|) / N, Δ = mean Δ_k;
core-cluster variance of the seed-mean difference, se² = J/(J−1) · Σ_j (d̄_j − n_j·Δ)² / N² with d̄_j = core j's mean
(over k) hit difference; var_seed = max(var(Δ_k, ddof 1) / K, 2·(0.013)²/K) (SCALED) or max(·, SEED_FLOOR) (constant).
Descriptive: kept counts and underfilled cores at 2/km² + PROX, the recipe contrast at 10 and 25 m reach, and D_101
re-decoded on the A100 VM (H0 cores) against stage A's local decode, in hits.

  python -B cdb_check.py K
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cdb_score as CS  # noqa: E402

A1, DS = CS.A1, CS.DS
FLOOR_CONST = 2 * 0.013 ** 2 / 3


def recount(xs: list[set], ys: list[set], pop: list[dict]) -> dict:
    N, K = len(pop), len(xs)
    dk = [(sum(r["id"] in x for r in pop) - sum(r["id"] in y for r in pop)) / N for x, y in zip(xs, ys)]
    delta = sum(dk) / K
    d, n = defaultdict(float), defaultdict(int)
    for r in pop:
        d[r["core"]] += sum((r["id"] in x) - (r["id"] in y) for x, y in zip(xs, ys)) / K
        n[r["core"]] += 1
    J = len(n)
    se2 = J / (J - 1) * sum((d[c] - n[c] * delta) ** 2 for c in n) / N ** 2
    obs = statistics.variance(dk) / K
    out = {}
    for tag, floor in (("scaled", 2 * 0.013 ** 2 / K), ("constant", FLOOR_CONST)):
        hw = 1.96 * math.sqrt(se2 + max(obs, floor))
        out[tag] = [100 * delta, 100 * (delta - hw), 100 * (delta + hw)]
    return dict(out, sd_delta_k_points=100 * math.sqrt(obs * K), delta_k_points=[round(100 * v, 3) for v in dk])


def main(K: int) -> dict:
    res = json.loads((HERE / f"RESULT_B_K{K}.json").read_text(encoding="utf-8"))
    hp = CS.BPRIV / f"B_HITS_K{K}_PRIVATE.json"
    hits = {k: set(v) for k, v in json.loads(hp.read_text(encoding="utf-8")).items()}
    cores, recs, masks = DS.inputs("JW")
    for r in recs:
        r["id"] = int(r["id"])
    H = {int(k): v for k, v in json.loads((DS.PRIV / "E0_H_DK_PRIVATE.json").read_text())["H"].items()}
    pops = A1.populations(recs, H)
    seeds = res["seeds"]
    xs, ys = [hits[f"D_{k}|prox|2"] for k in seeds], [hits[f"A_{k}|prox|2"] for k in seeds]
    out = {"hits_sha_matches": hashlib.sha256(hp.read_bytes()).hexdigest() == res["private_hits_sha256"], "recount": {}}
    for p in ("all", *CS.GUARDS, "small"):
        mine = recount(xs, ys, pops[p])
        rec = res["B"]["contrast"][p]
        match = all(abs(u - v) < 1e-9 for t in ("scaled", "constant") for u, v in
                    zip(mine[t], (rec[t]["delta_points"], rec[t]["lower_points"], rec[t]["upper_points"])))
        out["recount"][p] = {**{t: [round(v, 4) for v in mine[t]] for t in ("scaled", "constant")},
                             "sd_delta_k_points": round(mine["sd_delta_k_points"], 3),
                             "delta_k_points": mine["delta_k_points"], "match": match}
    budgets = {(c["i"], c["j"]): c["B"] for c in cores}
    kept = {}
    for m in [f"{a}_{k}" for k in seeds for a in ("D", "A")]:
        sel = DS.S.select(CS.proposals(m)[0], budgets, masks, "prox", DS.S.scot())
        kept[m] = {"kept": sum(len(s) for s in sel.values()), "budget": sum(budgets.values()),
                   "underfilled_cores": sum(len(sel.get(k, [])) < B for k, B in budgets.items())}
    out["kept_2km2_prox"] = kept
    reach = {}
    for rm in (10.0, 25.0):
        h, _ = CS.hits_for([f"{a}_{k}" for k in seeds for a in ("D", "A")], cores, recs, masks, rm)
        xr, yr = [h[(f"D_{k}", "prox", 2)] for k in seeds], [h[(f"A_{k}", "prox", 2)] for k in seeds]
        reach[f"R{int(rm)}"] = [round(v, 3) for v in recount(xr, yr, pops["all"])["scaled"]]
    out["reach_descriptive_scaled"] = reach
    out["d101_cross_device"] = cross_device(cores, recs, masks, res["reach_m"])
    out["all_recount_match"] = out["hits_sha_matches"] and all(v["match"] for v in out["recount"].values())
    out["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE / f"CHECK_B_K{K}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


def cross_device(cores: list[dict], recs: list[dict], masks: dict, reach: float) -> dict:
    """D_101 decoded on the A100 VM (H0 cores) vs stage A's RTX 2060 decode on the same cores, 2/km² + PROX."""
    parts = CS.BPRIV / "eval_vm/decode/phase0/parts/D_101"
    ids = {p.stem for p in parts.glob("*.json")}
    vm = [o for p in sorted(parts.glob("*.json")) for o in json.loads(p.read_text(encoding="utf-8"))["objects"]]
    local = [o for o in CS.proposals("D_101")[0] if o["cluster"] in ids]
    sub = [c for c in cores if f"DK_{c['i']}_{c['j']}" in ids]
    keys = {(c["i"], c["j"]) for c in sub}
    rs = [r for r in recs if r["core"] in keys]
    cd = [dict(c, B=round(2 * c["land_km2"])) for c in sub]
    old = DS.S.REACH_M
    DS.S.REACH_M = reach
    try:
        hv = set(DS.S.score(vm, cd, rs, masks, "prox", DS.S.scot())["matched_ids"])
        hl = set(DS.S.score(local, cd, rs, masks, "prox", DS.S.scot())["matched_ids"])
    finally:
        DS.S.REACH_M = old
    return {"cores": len(sub), "records": len(rs), "objects_vm": len(vm), "objects_local": len(local),
            "hits_vm": len(hv), "hits_local": len(hl), "hits_symmetric_difference": len(hv ^ hl)}


if __name__ == "__main__":
    r = main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
    print(json.dumps({k: r[k] for k in ("hits_sha_matches", "all_recount_match", "reach_descriptive_scaled",
                                        "d101_cross_device")}, indent=1))
    print(json.dumps({p: v["match"] for p, v in r["recount"].items()}))
