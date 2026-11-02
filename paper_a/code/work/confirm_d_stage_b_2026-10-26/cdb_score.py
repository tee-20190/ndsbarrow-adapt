"""CONFIRM-D stage B scoring (FREEZE §4, §6 B, §7 B, §8): the D recipe vs the D1 recipe over K paired training seeds.

Pairs k = (D_k, A_k), k in {101, ..., 909}. D_101/202/303 proposals are stage A's (jw/decode/phase0, FREEZE §9: decoded
once); A_101..909 and D_404..909 are this packet's VM decodes (dt1_decode.py unchanged, assembled over all 171 cores).
Scoring is stage A's: dk_score.inputs("JW") (PREP_JW cores, records, S-v1 masks), cr1_score.score unchanged, PROX at
2/km² of land, primary reach REACH_DECISION.json (15 m). Populations: dk_a1.populations (guards large, isolated,
strong_grouped). Contrast: cd_analysis.recipe_contrast(D hits, A hits, pop, "scaled") with the constant-floor interval
alongside; labels cd_analysis.guard_label / a1_label (§7 B: PASS iff lower bound > 0 under the SCALED floor; guards as A1).

  python -B cdb_score.py K         # K = 9 (or 6 = k <= 606, the §9 fallback) -> RESULT_B.json (counts only) + private hits
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A_DIR = ROOT / "work/confirm_d_2026-10-25"
sys.path.insert(0, str(A_DIR))
import cd_analysis as CA  # noqa: E402
import dk_a1 as A1  # noqa: E402
import dk_score as DS  # noqa: E402

BPRIV = ROOT / "outputs/confirm_d_stage_b_2026-10-26_PRIVATE"
B_DEC = BPRIV / "eval_vm/decode/phase0"
A_DEC = DS.PRIV / "jw/decode/phase0"
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
GUARDS = ("large", "isolated", "strong_grouped")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def proposals(model: str) -> tuple[list[dict], str]:
    """Stage A's file for D_101/202/303 and the D1 / ADAF references; else this packet's assembled VM decode."""
    stage_a = model in ("D_101", "D_202", "D_303", "D1_05", "D1_06", "D1_07")
    p = (A_DEC if stage_a else B_DEC) / f"PROPOSALS_{model}_PRIVATE.json"
    return json.loads(p.read_text(encoding="utf-8"))["objects"], sha(p)


def hits_for(models: list[str], cores: list[dict], recs: list[dict], masks: dict, reach: float) -> tuple[dict, dict]:
    hits, files = {}, {}
    for m in models:
        objs, files[m] = proposals(m)
        for d in (2, 20):
            cd = [dict(c, B=round(d * c["land_km2"])) for c in cores]
            for v in ("prox", "plain"):
                if d == 20 and v == "prox":
                    continue
                old = DS.S.REACH_M
                DS.S.REACH_M = reach
                try:
                    r = DS.S.score(objs, cd, recs, masks, v, DS.S.scot())
                finally:
                    DS.S.REACH_M = old
                hits[(m, v, d)] = set(r["matched_ids"])
    return hits, files


def pct(h: set, pop: list[dict]) -> float | None:
    return round(100 * sum(r["id"] in h for r in pop) / len(pop), 3) if pop else None


def contrasts(xs: list[set], ys: list[set], pops: dict) -> dict:
    out = {}
    for p in ("all", *GUARDS, "small"):
        if not pops[p]:
            out[p] = {"n": 0, "status": "EMPTY"}
            continue
        sc, co = CA.recipe_contrast(xs, ys, pops[p], "scaled"), CA.recipe_contrast(xs, ys, pops[p], "constant")
        out[p] = {"scaled": sc, "constant": co, "guard_label_scaled": CA.guard_label(sc)}
    return out


def main(K: int) -> dict:
    seeds = SEEDS[:K]
    cores, recs, masks = DS.inputs("JW")
    for r in recs:
        r["id"] = int(r["id"])
    H = {int(k): v for k, v in json.loads((DS.PRIV / "E0_H_DK_PRIVATE.json").read_text())["H"].items()}
    reach = json.loads((A_DIR / "REACH_DECISION.json").read_text())["primary_reach_m"]
    pops = A1.populations(recs, H)
    models = [f"{a}_{k}" for k in seeds for a in ("D", "A")] + ["D1_05", "D1_06", "D1_07"]
    hits, files = hits_for(models, cores, recs, masks, reach)
    xs = [hits[(f"D_{k}", "prox", 2)] for k in seeds]
    ys = [hits[(f"A_{k}", "prox", 2)] for k in seeds]
    c = contrasts(xs, ys, pops)
    label = CA.a1_label(c["all"]["scaled"], {p: c[p]["scaled"] for p in GUARDS if "scaled" in c[p]})
    plain = contrasts([hits[(f"D_{k}", "plain", 2)] for k in seeds], [hits[(f"A_{k}", "plain", 2)] for k in seeds],
                      {"all": pops["all"], **{p: [] for p in (*GUARDS, "small")}})["all"]
    r0 = [hits[(f"D1_0{i}", "prox", 2)] for i in (5, 6, 7)]
    by_index = CA.recipe_contrast(xs[:3], r0, pops["all"], "constant")
    per_seed = {f"{a}_{k}": {f"{v}_{d}": pct(hits[(f"{a}_{k}", v, d)], pops["all"]) for v, d in (("prox", 2), ("plain", 2),
                                                                                              ("plain", 20))}
                for k in seeds for a in ("D", "A")}
    per_seed.update({f"D1_0{i}": {"prox_2": pct(hits[(f"D1_0{i}", "prox", 2)], pops["all"])} for i in (5, 6, 7)})
    d9 = sum(per_seed[f"D_{k}"]["prox_2"] for k in seeds) / K
    out = {"packet": "CONFIRM-D stage B (FREEZE 3cd5a9682 §6 B, §7 B; STAGE_B_NOTES fb8deb34)", "K": K,
           "seeds": list(seeds), "reach_m": reach, "cores": len(cores), "N": {k: len(v) for k, v in pops.items()},
           "B_total_2km2": sum(round(2 * c_["land_km2"]) for c_ in cores),
           "B": {"contrast": c, "label": label, "plain_2km2_all": plain},
           "descriptive": {"per_seed_recall_points": per_seed,
                           "D_by_index_minus_R0_05_06_07_constant": by_index,
                           "mean_D_K_minus_mean_R0_points": d9 - sum(per_seed[f"D1_0{i}"]["prox_2"] for i in (5, 6, 7)) / 3,
                           "per_stratum_recall_points": {m: {p: pct(hits[(m, "prox", 2)], pop) for p, pop in pops.items()}
                                                         for m in models}},
           "proposal_files_sha256": files, "code_sha256": sha(Path(__file__)),
           "cd_analysis_sha256": sha(A_DIR / "cd_analysis.py")}
    priv = BPRIV / f"B_HITS_K{K}_PRIVATE.json"
    priv.write_text(json.dumps({f"{m}|{v}|{d}": sorted(h) for (m, v, d), h in hits.items()}) + "\n", encoding="utf-8")
    out["private_hits_sha256"] = sha(priv)
    (HERE / f"RESULT_B_K{K}.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                             newline="\n")
    return out


if __name__ == "__main__":
    r = main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
    a = r["B"]["contrast"]["all"]
    print(json.dumps({"K": r["K"], "label": r["B"]["label"], **{k: a["scaled"][k] for k in ("delta_points", "lower_points",
                                                                                           "upper_points")},
                      "constant": [a["constant"]["lower_points"], a["constant"]["upper_points"]],
                      "guards": {p: r["B"]["contrast"][p].get("guard_label_scaled") for p in GUARDS}}, indent=1))
