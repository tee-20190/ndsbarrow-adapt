"""DK-REGSUP-1 stage A scoring (FREEZE.md sections 7-8): A - C over K = 15 seed pairs, cross-fitted over 3 folds.

A_k's proposals are the union over folds f of A_k^f's objects on fold f's held-out cores: every DK-JW core is decoded by
exactly one A_k^f, the one that never read terrain within 10 km of it. C_k is the seed's D-recipe run on all 171 cores:
D_101..303 (CONFIRM-D stage A decode), D_404..909 (stage B VM decode), C_1010..1515 (this packet). Scoring is CONFIRM-D
stage B's (cdb_score): dk_score.inputs("JW") (PREP_JW cores, the frozen 5,273 records, S-v1 masks), cr1_score.score
unchanged, PROX at 2/km2 of land (per core), reach 15 m (REACH_DECISION.json), populations dk_a1.populations.
Contrast cd_analysis.recipe_contrast(A hits, C hits, pop, "scaled"): Delta = mean_k Delta_k, 95 % interval with the
5 km core-cluster term and var_seed = max(var(Delta_k)/K, 2 * (1.3 pts)^2 / K).
Gate (FREEZE section 8): PASS iff all-records lower > 0 and the large and isolated lower bounds > -2 points.

  python -B rs1_score.py assemble DIR SET -> dt1_decode.assemble (unchanged) over DIR/phase0/parts for MODELS_<SET> /
                                             PREP_<SET> (vmin/), then each model's proposals, with the PREP core list
                                             added as "cores", into decode/phase0/PROPOSALS_<model>_PRIVATE.json
  python -B rs1_score.py score            -> RESULT_A.json (counts only) + private hits; needs every A_k^f and C_k
  python -B rs1_score.py smoke            end-to-end on the real FREEZE, before any DK-REGSUP-1 output exists:
         S1 stand-in A_k^f = D_k's objects split by fold, C_k = CONFIRM-D stage-B A_k (D1 recipe), K = 9: must
            reproduce RESULT_B_K9.json's B contrast exactly (all populations, both floors) -> the fold union is lossless
         S2 identity A = C = D_k (K = 9): Delta 0 and gate FAIL
         -> SMOKE_SCORE.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CDB = ROOT / "work/confirm_d_stage_b_2026-10-26"
sys.path.insert(0, str(CDB))
import cdb_score as CS  # noqa: E402  stage-B scorer (unchanged): proposals, hits_for's scoring call, populations

CA, A1, DS = CS.CA, CS.A1, CS.DS
PRIV = ROOT / "outputs/dk_regsup1_2026-10-29_PRIVATE"
DEC = PRIV / "decode/phase0"
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515)
FOLDS = (0, 1, 2)
GATE_GUARDS = ("large", "isolated")
DESCRIPTIVE = ("strong_grouped", "small")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def folds() -> list[dict]:
    return json.loads((PRIV / "FOLDS_PRIVATE.json").read_text(encoding="utf-8"))["folds"]


def c_file(k: int) -> Path:
    if k in (101, 202, 303):
        return CS.A_DEC / f"PROPOSALS_D_{k}_PRIVATE.json"
    if k <= 909:
        return CS.B_DEC / f"PROPOSALS_D_{k}_PRIVATE.json"
    return DEC / f"PROPOSALS_C_{k}_PRIVATE.json"


def a_union(k: int, fo: list[dict], src: dict | None = None) -> tuple[list[dict], dict]:
    """A_k's cross-fitted proposals; src[(fold, k)] overrides the file (smoke stand-ins)."""
    objs, files = [], {}
    for f in fo:
        held = set(f["held_out"])
        if src is not None:
            got = src[(f["fold"], k)]
        else:
            p = DEC / f"PROPOSALS_A_F{f['fold']}_{k}_PRIVATE.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            got, files[f"A_F{f['fold']}_{k}"] = d["objects"], sha(p)
            if sorted(d.get("cores", [])) != sorted(held):
                raise SystemExit(f"A_F{f['fold']}_{k}: decoded cores differ from fold {f['fold']}'s held-out cores")
        if any(o["cluster"] not in held for o in got):
            raise SystemExit(f"A_F{f['fold']}_{k}: an object lies outside its fold's held-out cores")
        objs += got
    return objs, files


def hits(objs: list[dict], cores: list[dict], recs: list[dict], masks: dict, reach: float) -> dict:
    out = {}
    for d, v in ((2, "prox"), (2, "plain"), (20, "plain")):
        cd = [dict(c, B=round(d * c["land_km2"])) for c in cores]
        old = DS.S.REACH_M
        DS.S.REACH_M = reach
        try:
            out[(v, d)] = set(DS.S.score(objs, cd, recs, masks, v, DS.S.scot())["matched_ids"])
        finally:
            DS.S.REACH_M = old
    return out


def contrasts(xs: list[set], ys: list[set], pops: dict) -> dict:
    out = {}
    for p in ("all", *GATE_GUARDS, *DESCRIPTIVE):
        sc, co = CA.recipe_contrast(xs, ys, pops[p], "scaled"), CA.recipe_contrast(xs, ys, pops[p], "constant")
        out[p] = {"scaled": sc, "constant": co, "guard_label_scaled": CA.guard_label(sc)}
    return out


def gate(c: dict) -> dict:
    allp = c["all"]["scaled"]["lower"] > 0
    guards = {p: c[p]["scaled"]["lower"] > CA.GUARD for p in GATE_GUARDS}
    return {"all_lower_gt_0": allp, **{f"{p}_lower_gt_minus2": v for p, v in guards.items()},
            "outcome": "PASS" if allp and all(guards.values()) else "FAIL"}


def inputs() -> tuple[list[dict], list[dict], dict, dict, float]:
    cores, recs, masks = DS.inputs("JW")
    for r in recs:
        r["id"] = int(r["id"])
    H = {int(k): v for k, v in json.loads((DS.PRIV / "E0_H_DK_PRIVATE.json").read_text())["H"].items()}
    reach = json.loads((CS.A_DIR / "REACH_DECISION.json").read_text())["primary_reach_m"]
    return cores, recs, masks, A1.populations(recs, H), reach


def evaluate(seeds: tuple, a_src: dict | None, c_files: dict, ctx: tuple) -> dict:
    cores, recs, masks, pops, reach = ctx
    fo = folds()
    H, files = {}, {}
    for k in seeds:
        objs, fa = a_union(k, fo, a_src)
        files.update(fa)
        H[f"A_{k}"] = hits(objs, cores, recs, masks, reach)
        cp = c_files[k]
        H[f"C_{k}"] = hits(json.loads(cp.read_text(encoding="utf-8"))["objects"], cores, recs, masks, reach)
        files[f"C_{k}"] = sha(cp)
    xs = [H[f"A_{k}"][("prox", 2)] for k in seeds]
    ys = [H[f"C_{k}"][("prox", 2)] for k in seeds]
    c = contrasts(xs, ys, pops)
    plain = CA.recipe_contrast([H[f"A_{k}"][("plain", 2)] for k in seeds], [H[f"C_{k}"][("plain", 2)] for k in seeds],
                               pops["all"], "scaled")
    pct = {m: {f"{v}_{d}": round(100 * sum(r["id"] in h for r in pops["all"]) / len(pops["all"]), 3)
               for (v, d), h in hh.items()} for m, hh in H.items()}
    strata = {m: {p: round(100 * sum(r["id"] in hh[("prox", 2)] for r in pop) / len(pop), 3)
                  for p, pop in pops.items() if pop} for m, hh in H.items()}
    return {"seeds": list(seeds), "K": len(seeds), "reach_m": reach, "cores": len(cores),
            "N": {k: len(v) for k, v in pops.items()}, "contrast": c, "gate": gate(c),
            "a1_label": CA.a1_label(c["all"]["scaled"], {p: c[p]["scaled"] for p in GATE_GUARDS}),
            "plain_2km2_all_scaled": plain, "per_model_recall_points": pct, "per_stratum_prox2": strata,
            "proposal_files_sha256": files, "hits": H}


def score() -> dict:
    ctx = inputs()
    r = evaluate(SEEDS, None, {k: c_file(k) for k in SEEDS}, ctx)
    hp = PRIV / "A_HITS_PRIVATE.json"
    hp.write_text(json.dumps({f"{m}|{v}|{d}": sorted(h) for m, hh in r.pop("hits").items() for (v, d), h in hh.items()})
                  + "\n", encoding="utf-8")
    r.update({"packet": "DK-REGSUP-1 stage A (FREEZE.md)", "private_hits_sha256": sha(hp),
              "code_sha256": sha(Path(__file__)), "cdb_score_sha256": sha(CDB / "cdb_score.py"),
              "cd_analysis_sha256": sha(CS.A_DIR / "cd_analysis.py"), "folds_private_sha256": sha(PRIV / "FOLDS_PRIVATE.json")})
    (HERE / "RESULT_A.json").write_text(json.dumps(r, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    return r


def assemble(dec_dir: Path, set_name: str) -> dict:
    sys.path.insert(0, str(ROOT / "work/dense_terrain1_2026-10-25"))
    import dt1_decode as DTD
    if sha(Path(DTD.__file__)) != "af88392ab734fc4959da470ee1ae93ef1d1153e2bdae408fdb2b6cd3ec47bb40":
        raise SystemExit("dt1_decode.py differs from its pin")
    specs = json.loads((PRIV / f"vmin/MODELS_{set_name}.json").read_text(encoding="utf-8"))["models"]
    prep = json.loads((PRIV / f"vmin/PREP_{set_name}.json").read_text(encoding="utf-8"))
    n = DTD.assemble(specs, prep, dec_dir, 0.0, None)
    cores = [c["id"] for c in prep["clusters"]]
    DEC.mkdir(parents=True, exist_ok=True)
    out = {}
    for m in specs:
        d = json.loads((dec_dir / "phase0" / f"PROPOSALS_{m}_PRIVATE.json").read_text(encoding="utf-8"))
        d["cores"] = cores
        p = DEC / f"PROPOSALS_{m}_PRIVATE.json"
        p.write_text(json.dumps(d) + "\n", encoding="utf-8", newline="\n")
        out[m] = {"n": n[m], "sha256": sha(p)}
    s = dec_dir / "phase0" / "DECODE_SUMMARY.json"
    (DEC / f"DECODE_SUMMARY_{set_name}.json").write_bytes(s.read_bytes())
    return out


def standin(seeds: tuple, src_file) -> dict:
    """{(fold, k): objects of the seed's file restricted to the fold's held-out cores}."""
    out = {}
    for k in seeds:
        objs = json.loads(src_file(k).read_text(encoding="utf-8"))["objects"]
        for f in folds():
            held = set(f["held_out"])
            out[(f["fold"], k)] = [o for o in objs if o["cluster"] in held]
    return out


def smoke() -> dict:
    ctx = inputs()
    k9 = SEEDS[:9]
    ref = json.loads((CDB / "RESULT_B_K9.json").read_text(encoding="utf-8"))["B"]["contrast"]
    s1 = evaluate(k9, standin(k9, c_file), {k: CS.B_DEC / f"PROPOSALS_A_{k}_PRIVATE.json" for k in k9}, ctx)
    s1.pop("hits")
    diffs = {}
    for p in ("all", "large", "isolated", "strong_grouped", "small"):
        for t in ("scaled", "constant"):
            diffs[f"{p}_{t}"] = max(abs(s1["contrast"][p][t][q] - ref[p][t][q])
                                    for q in ("delta_points", "lower_points", "upper_points"))
    s2 = evaluate(k9, standin(k9, c_file), {k: c_file(k) for k in k9}, ctx)
    s2.pop("hits")
    out = {"S1_reproduces_stage_B": {"max_abs_diff_points": diffs, "pass": max(diffs.values()) < 1e-9,
                                     "delta_points": s1["contrast"]["all"]["scaled"]["delta_points"],
                                     "interval_points": [s1["contrast"]["all"]["scaled"]["lower_points"],
                                                         s1["contrast"]["all"]["scaled"]["upper_points"]],
                                     "gate": s1["gate"]},
           "S2_identity": {"delta_points": s2["contrast"]["all"]["scaled"]["delta_points"], "gate": s2["gate"],
                           "pass": s2["contrast"]["all"]["scaled"]["delta_points"] == 0
                           and s2["gate"]["outcome"] == "FAIL"},
           "folds_private_sha256": sha(PRIV / "FOLDS_PRIVATE.json"), "code_sha256": sha(Path(__file__)),
           "utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())}
    out["status"] = "PASS" if out["S1_reproduces_stage_B"]["pass"] and out["S2_identity"]["pass"] else "FAIL"
    (HERE / "SMOKE_SCORE.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                           newline="\n")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "assemble" and len(sys.argv) == 4:
        print(json.dumps(assemble(Path(sys.argv[2]), sys.argv[3]), indent=1))
        raise SystemExit(0)
    if cmd not in ("score", "smoke"):
        raise SystemExit(__doc__)
    r = score() if cmd == "score" else smoke()
    if cmd == "score":
        a = r["contrast"]["all"]["scaled"]
        print(json.dumps({"K": r["K"], "gate": r["gate"], "delta": a["delta_points"], "lower": a["lower_points"],
                          "upper": a["upper_points"], "guards": {p: r["contrast"][p]["guard_label_scaled"]
                                                                  for p in GATE_GUARDS}}, indent=1))
    else:
        print(json.dumps(r, indent=1, default=float))
