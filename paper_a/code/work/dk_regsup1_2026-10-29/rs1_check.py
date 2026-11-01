"""DK-REGSUP-1 stage A: independent recount (FREEZE.md section 9). Writes CHECK_A.json (counts only).

Recount from the private hit sets without cr1_analysis / cd_analysis: Delta_k = (|A_k n pop| - |C_k n pop|) / N,
Delta = mean_k Delta_k; core-cluster variance of the seed-mean difference, se2 = J/(J-1) sum_j (d_j - n_j Delta)^2 / N^2
with d_j = core j's mean (over k) hit difference; var_seed = max(var(Delta_k, ddof 1) / K, 2 (0.013)^2 / K) (SCALED).
The gate is re-derived from these intervals. Descriptive: kept counts and underfilled cores at 2/km2 + PROX, the
contrast at 10 and 25 m reach, and per-fold contrasts (each fold's held-out records only).

  python -B rs1_check.py
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
import rs1_score as RS  # noqa: E402

DS = RS.DS


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
    hw = 1.96 * math.sqrt(se2 + max(obs, 2 * 0.013 ** 2 / K))
    return {"scaled": [100 * delta, 100 * (delta - hw), 100 * (delta + hw)],
            "sd_delta_k_points": 100 * math.sqrt(obs * K), "delta_k_points": [round(100 * v, 3) for v in dk]}


def main() -> dict:
    res = json.loads((HERE / "RESULT_A.json").read_text(encoding="utf-8"))
    hp = RS.PRIV / "A_HITS_PRIVATE.json"
    hits = {k: set(v) for k, v in json.loads(hp.read_text(encoding="utf-8")).items()}
    cores, recs, masks, pops, reach = RS.inputs()
    seeds = res["seeds"]
    xs, ys = [hits[f"A_{k}|prox|2"] for k in seeds], [hits[f"C_{k}|prox|2"] for k in seeds]
    out = {"hits_sha_matches": hashlib.sha256(hp.read_bytes()).hexdigest() == res["private_hits_sha256"], "recount": {}}
    for p in ("all", *RS.GATE_GUARDS, *RS.DESCRIPTIVE):
        mine = recount(xs, ys, pops[p])
        rec = res["contrast"][p]["scaled"]
        match = all(abs(u - v) < 1e-9 for u, v in zip(mine["scaled"], (rec["delta_points"], rec["lower_points"],
                                                                         rec["upper_points"])))
        out["recount"][p] = {"scaled": [round(v, 4) for v in mine["scaled"]],
                             "sd_delta_k_points": round(mine["sd_delta_k_points"], 3),
                             "delta_k_points": mine["delta_k_points"], "match": match}
    r = out["recount"]
    gate = "PASS" if r["all"]["scaled"][1] > 0 and all(r[p]["scaled"][1] > -2 for p in RS.GATE_GUARDS) else "FAIL"
    out["gate_recount"] = gate
    out["gate_matches"] = gate == res["gate"]["outcome"]
    budgets = {(c["i"], c["j"]): c["B"] for c in cores}
    kept, fo = {}, RS.folds()
    for k in seeds:
        for name, objs in (("A", RS.a_union(k, fo)[0]),
                           ("C", json.loads(RS.c_file(k).read_text(encoding="utf-8"))["objects"])):
            sel = DS.S.select(objs, budgets, masks, "prox", DS.S.scot())
            kept[f"{name}_{k}"] = {"kept": sum(len(s) for s in sel.values()), "budget": sum(budgets.values()),
                                   "underfilled_cores": sum(len(sel.get(c, [])) < B for c, B in budgets.items())}
    out["kept_2km2_prox"] = kept
    reach_rows = {}
    for rm in (10.0, 25.0):
        xa, ya = [], []
        for k in seeds:
            xa.append(RS.hits(RS.a_union(k, fo)[0], cores, recs, masks, rm)[("prox", 2)])
            ya.append(RS.hits(json.loads(RS.c_file(k).read_text(encoding="utf-8"))["objects"], cores, recs, masks,
                              rm)[("prox", 2)])
        reach_rows[f"R{int(rm)}"] = [round(v, 3) for v in recount(xa, ya, pops["all"])["scaled"]]
    out["reach_descriptive_scaled"] = reach_rows
    per_fold = {}
    for f in fo:
        keys = {(int(c.split("_")[1]), int(c.split("_")[2])) for c in f["held_out"]}
        pop = [r for r in pops["all"] if r["core"] in keys]
        per_fold[f"F{f['fold']}"] = {"n": len(pop), "scaled": [round(v, 3) for v in recount(xs, ys, pop)["scaled"]]}
    out["per_fold_descriptive_scaled"] = per_fold
    out["all_recount_match"] = out["hits_sha_matches"] and all(v["match"] for v in r.values()) and out["gate_matches"]
    out["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE / "CHECK_A.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("hits_sha_matches", "gate_recount", "gate_matches", "all_recount_match",
                                        "reach_descriptive_scaled", "per_fold_descriptive_scaled")}, indent=1))
    print(json.dumps({p: v["match"] for p, v in r["recount"].items()}))
