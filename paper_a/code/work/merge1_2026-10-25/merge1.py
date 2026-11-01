"""MERGE-1 (FREEZE.md, ad8aa987f): equal-budget interleave of two proposal lists per core; scored with cr1_score unchanged."""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/context_rescue1_2026-10-23/eval_score"))
sys.path.insert(0, str(ROOT / "work/context_rescue1_2026-10-23"))
import cr1_eval_score as E
import cr1_analysis as AN
from scipy.spatial import cKDTree

S = E.S
SC = S.scot()
DEDUP_M = 15.0


def load(path: str) -> list[dict]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))["objects"]


def by_core(props: list[dict], masks: dict, budgets: dict) -> dict:
    out: dict = {}
    for p in props:
        k = S.core_key(p["easting"], p["northing"])
        if k in budgets and not masks[k](p["easting"], p["northing"]):
            out.setdefault(k, []).append(p)
    return {k: sorted(v, key=SC.rank_key) for k, v in out.items()}


def interleave(x: list[dict], y: list[dict]) -> list[dict]:
    order, pts = [], []
    for i in range(max(len(x), len(y))):
        for lst in (x, y):
            if i < len(lst):
                p = lst[i]
                if pts and cKDTree(pts).query([p["easting"], p["northing"]])[0] <= DEDUP_M:
                    continue
                order.append(p)
                pts.append([p["easting"], p["northing"]])
    return order


def merged(a: list[dict], b: list[dict], masks: dict, budgets: dict) -> list[dict]:
    ca, cb = by_core(a, masks, budgets), by_core(b, masks, budgets)
    out = []
    for k in set(ca) | set(cb):
        for i, p in enumerate(interleave(ca.get(k, []), cb.get(k, []))):
            out.append({**p, "id": f"m{k[0]}_{k[1]}_{i}", "score": 1.0 - i * 1e-6, "mean_probability": 0.0, "area_m2": 0.0})
    return out


def main() -> None:
    base = E.load_eval_base()
    budgets = {(c["i"], c["j"]): c["B"] for c in base["cores"]}
    inp = json.loads((ROOT / "outputs/dense_terrain1_2026-10-25_PRIVATE/score/INPUTS_FULL.json").read_text())["phase0"]
    D = [load(inp[f"D_{s}"]) for s in ("101", "202", "303")]
    D1 = [load(f"outputs/subtle1_2026-10-23_PRIVATE/infer/LS/PROPOSALS_D1_0{s}_PRIVATE.json") for s in "567"]
    arms = {"MIX": [(D[k], D1[k]) for k in range(3)], "DD": [(D[k], D[(k + 1) % 3]) for k in range(3)],
            "D1D1": [(D1[k], D1[(k + 1) % 3]) for k in range(3)]}
    hits: dict = {}
    for name, pairs in arms.items():
        for v in ("plain", "prox"):
            hits[(name, v)] = [S.score(merged(a, b, base["masks"], budgets), base["cores"], base["records"], base["masks"], v, SC)["matched_ids"]
                               for a, b in pairs]
    for name, lists in (("D", D), ("D1", D1)):
        for v in ("plain", "prox"):
            hits[(name, v)] = [S.score(p, base["cores"], base["records"], base["masks"], v, SC)["matched_ids"] for p in lists]
    st = E.strata(base["records"], base["H"])
    big = {r["id"] for r in base["records"] if r.get("area_m2", 0) >= 200}
    pops = {"all": base["records"], "large": [r for r in base["records"] if r["id"] in big],
            "isolated": st["isolated"], "strong_grouped": st["strong_grouped"]}
    pt = lambda c: {k: round(100 * c[k], 2) for k in ("delta", "lower", "upper")} | {"by_seed": [round(100 * d, 2) for d in c["delta_by_seed"]]}
    N = len(base["records"])
    res = {"recall_points": {f"{a}|{v}": round(100 * sum(map(len, h)) / 3 / N, 2) for (a, v), h in hits.items()}}
    C = {}
    C["PRIMARY MIX+PROX - DD+PROX"] = pt(AN.contrast(hits[("MIX", "prox")], hits[("DD", "prox")], pops["all"]))
    for x, y in (("MIX", "D1D1"), ("MIX", "D"), ("MIX", "D1"), ("DD", "D"), ("D", "D1")):
        for v in ("prox", "plain"):
            C[f"{x}+{v} - {y}+{v}"] = pt(AN.contrast(hits[(x, v)], hits[(y, v)], pops["all"]))
    for s in ("large", "isolated", "strong_grouped"):
        C[f"GUARD {s}: MIX+PROX - DD+PROX"] = pt(AN.contrast(hits[("MIX", "prox")], hits[("DD", "prox")], pops[s]))
    res["contrasts_points"] = C
    res["primary_pass"] = C["PRIMARY MIX+PROX - DD+PROX"]["lower"] > 0
    res["guards_pass"] = {s: C[f"GUARD {s}: MIX+PROX - DD+PROX"]["lower"] > -2 for s in ("large", "isolated", "strong_grouped")}
    (ROOT / "work/merge1_2026-10-25/RESULT.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
