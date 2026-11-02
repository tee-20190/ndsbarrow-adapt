"""CONFIRM-D stage A: independent recount + FREEZE §8 supporting report (kept counts, underfilled cores, reach 10/25 m).

Recount: A1/A2 deltas from the private hit sets by direct counting, and the core-cluster ratio variance by a separate
implementation (not cr1_analysis): se² = J/(J-1) · Σ_j (d_j − n_j·Δ)² / N², d_j = core j's hit difference.
Writes CHECK_A1.json (counts only).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dk_a1 as A  # noqa: E402

DS, M1 = A.DS, A.M1


def ratio_ci(x: set, y: set, recs: list[dict]) -> tuple[float, float, float]:
    N = len(recs)
    d, n = defaultdict(int), defaultdict(int)
    for r in recs:
        d[r["core"]] += (r["id"] in x) - (r["id"] in y)
        n[r["core"]] += 1
    delta = sum(d.values()) / N
    J = len(n)
    se = math.sqrt(J / (J - 1) * sum((d[c] - n[c] * delta) ** 2 for c in n) / N ** 2)
    return 100 * delta, 100 * (delta - 1.96 * se), 100 * (delta + 1.96 * se)


def main() -> dict:
    res = json.loads((HERE / "RESULT_A1.json").read_text())
    hits = {k: set(v) for k, v in json.loads((DS.PRIV / "jw/A1_HITS_PRIVATE.json").read_text()).items()}
    ok_sha = hashlib.sha256((DS.PRIV / "jw/A1_HITS_PRIVATE.json").read_bytes()).hexdigest() == res["private_hits_sha256"]
    cores, recs, masks = DS.inputs("JW")
    for r in recs:
        r["id"] = int(r["id"])
    H = {int(k): v for k, v in json.loads((DS.PRIV / "E0_H_DK_PRIVATE.json").read_text())["H"].items()}
    pops = A.populations(recs, H)
    out = {"hits_sha_matches": ok_sha, "recount": {}}
    for tag, (a, b) in {"A1": ("S_D|prox|2", "incumbent|prox|2"), "A2": ("S_D|prox|2", "S_D1|prox|2")}.items():
        for p in ("all", "large", "isolated", "strong_grouped"):
            mine = ratio_ci(hits[a], hits[b], pops[p])
            rec = res[tag]["contrast"] if p == "all" else res[tag]["guards"][p]
            want = (rec["delta_points"], rec["lower_points"], rec["upper_points"])
            out["recount"][f"{tag}|{p}"] = {"points": [round(v, 4) for v in mine],
                                            "match": all(abs(u - v) < 1e-9 for u, v in zip(mine, want))}
    budgets = {(c["i"], c["j"]): c["B"] for c in cores}
    P = {m: DS.props("JW", m) for m in ("D_101", "D_202", "D1_05", "D1_06", "ADAF_barrow")}
    systems = {"S_D": M1.merged(P["D_101"], P["D_202"], masks, budgets), "incumbent": P["D1_05"],
               "S_D1": M1.merged(P["D1_05"], P["D1_06"], masks, budgets), "ADAF_barrow": P["ADAF_barrow"]}
    kept = {}
    for name, objs in systems.items():
        for v in ("plain", "prox"):
            sel = DS.S.select(objs, budgets, masks, v, DS.S.scot())
            under = [k for k, B in budgets.items() if len(sel.get(k, [])) < B]
            kept[f"{name}|{v}"] = {"kept": sum(len(s) for s in sel.values()), "budget": sum(budgets.values()),
                                   "underfilled_cores": len(under)}
    out["kept_2km2"] = kept
    reach = {}
    for rm in (10.0, 25.0):
        xs = A.score_all({k: systems[k] for k in ("S_D", "incumbent")}, cores, recs, masks, rm)
        x, y = xs[("S_D", "prox", 2)], xs[("incumbent", "prox", 2)]
        reach[f"R{int(rm)}"] = {"S_D": len(x), "incumbent": len(y), "A1_points": [round(v, 3) for v in ratio_ci(x, y, recs)]}
    out["reach_descriptive"] = reach
    out["all_recount_match"] = ok_sha and all(v["match"] for v in out["recount"].values())
    out["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE / "CHECK_A1.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
