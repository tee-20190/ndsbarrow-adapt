"""CONFIRM-D stage A scoring (FREEZE §3, §4, §6, §7, §8): A1 primary, A2 fixed-sequence secondary, descriptive tables.

Systems (FREEZE §3): S_D = merge1.merged(D_101, D_202) + PROX; incumbent = D1_05 + PROX; S_D1 = merge1.merged(D1_05,
D1_06) + PROX. Primary workload 2/km² + PROX at the §5 reach (REACH_DECISION.json). Contrasts: cd_analysis.fixed_contrast
(no seed term); guards: large (area >= 200 m²), isolated (200 m group < 3), strong_grouped (group >= 3 and H >= 2T,
E0_H_DK). Labels: cd_analysis.a1_label / guard_label. Everything else is descriptive.

  python -B dk_a1.py JW                         # writes RESULT_A1.json (counts only) + private hits
  python -B dk_a1.py DEV --plumbing             # pipeline test on DEV with stand-in members (no claim; not written)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work/merge1_2026-10-25"))
import cd_analysis as CA  # noqa: E402
import dk_score as DS  # noqa: E402
import merge1 as M1  # noqa: E402

T_FAINT = 0.3290754481743007
DENSITIES = (1, 2, 5, 20)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def populations(recs: list[dict], H: dict) -> dict[str, list[dict]]:
    g = [r for r in recs if r["group200"] >= 3]
    return {"all": recs,
            "small": [r for r in recs if r["area_m2"] is not None and r["area_m2"] < 200],
            "large": [r for r in recs if r["area_m2"] is not None and r["area_m2"] >= 200],
            "size_unknown": [r for r in recs if r["area_m2"] is None],
            "isolated": [r for r in recs if r["group200"] < 3],
            "grouped": g,
            "strong_grouped": [r for r in g if H.get(r["id"]) is not None and H[r["id"]] >= 2 * T_FAINT]}


def systems(tag: str, plumbing: bool, masks: dict, budgets: dict) -> dict[str, list[dict]]:
    if plumbing:  # DEV has only D1_05 and ADAF: stand-ins exercise the merge path, nothing is claimed
        a, b = DS.props(tag, "D1_05"), DS.props(tag, "ADAF_barrow")
        return {"S_D": M1.merged(a, b, masks, budgets), "incumbent": a, "S_D1": M1.merged(b, a, masks, budgets),
                "ADAF_barrow": b}
    P = {m: DS.props(tag, m) for m in ("D_101", "D_202", "D_303", "D1_05", "D1_06", "D1_07", "ADAF_barrow")}
    out = {"S_D": M1.merged(P["D_101"], P["D_202"], masks, budgets), "incumbent": P["D1_05"],
           "S_D1": M1.merged(P["D1_05"], P["D1_06"], masks, budgets)}
    out.update({m: P[m] for m in ("D_101", "D_202", "D_303", "D1_07", "ADAF_barrow")})
    return out


def fc(x: set, y: set, pop: list[dict]) -> dict:
    """fixed_contrast, or an EMPTY record (interval unbounded, so a guard on it is UNRESOLVED) for an empty population."""
    return CA.fixed_contrast(x, y, pop) if pop else {"n": 0, "status": "EMPTY", "lower": float("-inf"),
                                                      "upper": float("inf")}


def score_all(sys_: dict, cores: list[dict], recs: list[dict], masks: dict, reach: float) -> dict:
    hits = {}
    for name, objs in sys_.items():
        for d in DENSITIES:
            cd = [dict(c, B=round(d * c["land_km2"])) for c in cores]
            for v in ("plain", "prox"):
                old = DS.S.REACH_M
                DS.S.REACH_M = reach
                try:
                    hits[(name, v, d)] = set(DS.S.score(objs, cd, recs, masks, v, DS.S.scot())["matched_ids"])
                finally:
                    DS.S.REACH_M = old
    return hits


def main(tag: str, plumbing: bool = False) -> dict:
    cores, recs, masks = DS.inputs(tag)
    for r in recs:
        r["id"] = int(r["id"])
    H = {}
    if not plumbing:
        H = {int(k): v for k, v in json.loads((DS.PRIV / "E0_H_DK_PRIVATE.json").read_text())["H"].items()}
    reach = json.loads((HERE / "REACH_DECISION.json").read_text())["primary_reach_m"]
    budgets = {(c["i"], c["j"]): c["B"] for c in cores}
    sys_ = systems(tag, plumbing, masks, budgets)
    hits = score_all(sys_, cores, recs, masks, reach)
    pops = populations(recs, H)
    N = {k: len(v) for k, v in pops.items()}
    rec = {f"{n}|{v}|{d}": {p: round(100 * sum(r["id"] in hits[(n, v, d)] for r in pop) / len(pop), 3) if pop else None
                            for p, pop in pops.items()} for (n, v, d) in hits}
    x, y, z = hits[("S_D", "prox", 2)], hits[("incumbent", "prox", 2)], hits[("S_D1", "prox", 2)]
    a1 = CA.fixed_contrast(x, y, pops["all"])
    g1 = {p: fc(x, y, pops[p]) for p in ("large", "isolated", "strong_grouped")}
    out = {"packet": "CONFIRM-D stage A (FREEZE 3cd5a9682; addenda SEQ-1)", "tag": tag, "plumbing_only": plumbing,
           "reach_m": reach, "cores": len(cores), "N": N, "B_total_2km2": sum(c["B"] for c in cores),
           "A1": {"contrast": a1, "guards": {p: dict(c, label=CA.guard_label(c)) for p, c in g1.items()},
                  "label": CA.a1_label(a1, g1)},
           "A1_block10km": CA.block_contrast(x, y, pops["all"], 10_000), "recall_points": rec}
    if a1["lower"] > 0:
        a2 = CA.fixed_contrast(x, z, pops["all"])
        g2 = {p: fc(x, z, pops[p]) for p in ("large", "isolated", "strong_grouped")}
        out["A2"] = {"contrast": a2, "guards": {p: dict(c, label=CA.guard_label(c)) for p, c in g2.items()},
                     "label": CA.a1_label(a2, g2)}
    else:
        out["A2"] = {"status": "NOT_TESTED (A1 did not pass; fixed sequence)"}
    out["code_sha256"] = sha(Path(__file__))
    if not plumbing:
        priv = DS.PRIV / "jw/A1_HITS_PRIVATE.json"
        priv.write_text(json.dumps({f"{n}|{v}|{d}": sorted(h) for (n, v, d), h in hits.items()}) + "\n")
        out["private_hits_sha256"] = sha(priv)
        (HERE / "RESULT_A1.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                            newline="\n")
    return out


if __name__ == "__main__":
    r = main(sys.argv[1], "--plumbing" in sys.argv)
    print(json.dumps({k: r[k] for k in ("tag", "plumbing_only", "reach_m", "cores", "N", "A1", "A2")}, indent=1,
                     default=float)[:4000])
