"""ADAPT-NDS-1 scorer (CPU; WSL adaf-h2h-venv): the packet's outcome rules, exactly.

  python -B score_adapt.py score FREEZE TOP1 TOP2 TOP3 [--draws N --seed S]   (tests only change --draws / --seed)

Metric: FRESH-1's benchmark primary (barrow-group recall at 0.5, 1 and 2 background proposals per km2, their mean;
HR1 region_score.py and hier_uncertainty.py byte-identical, pinned by the freeze; reach 15 m, NMS 60 m).
  Primary 1 (CONFIRM), ADAPT effect: A - C, three seeds paired, hier_uncertainty.hierarchical_ci (cluster x seed
    bootstrap, 2,000 draws, seed 20261005). ADAPT_GAIN if the mean >= +0.03 and lo95 > 0; ADAPT_HARM if the mean <=
    -0.03; else NULL.
  Primary 2 (CONFIRM), milestone: A (3-seed mean) - ADAF barrow (each seed paired with the one ADAF barrow run).
    BEATS_ADAF_BARROW if the mean >= +0.02 and lo95 > 0; TRAILS if hi95 < 0; else NOT_DISTINGUISHED.
  Comparisons use a tolerance of 1e-9 (FRESH-1 score_fresh EPS): mean >= t - 1e-9, lo95 > 1e-9, hi95 < -1e-9.
Secondary (descriptive): the same contrasts on DEV (Verden; FRESH-1's zero-shot references by hash, A and C inferred
here); English retention A - C on the English TRAIN holdout (each run's in-training holdout objects; same bootstrap);
Welsh and Irish DEV for seed 2026092802 (A and C, point values); register-height strata on the individual manifest;
INTERLEAVE(ADAF barrow, A) and INTERLEAVE(ADAF AO, A) (FUSION-1 fuse_eval; ADAF is model A); A - J, C - J and the
references against ADAF barrow. Forecasts are scored at the result.
Proposals are made on inference manifests without labels (the VMs never held CONFIRM or DEV labels) and rebound here
to the group / individual manifests after a tile-identity check, as FRESH-1's scorer does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nds_common as N  # noqa: E402

sys.path.insert(0, str(N.HR1_STEP2))
import hier_uncertainty as hu  # noqa: E402  frozen
import region_score as rs  # noqa: E402  frozen

sys.path.insert(0, str(N.FUSION))
import fuse_eval as fe  # noqa: E402  FUSION-1 INTERLEAVE (bound by sha256)

shd = rs.shd
EPS = 1e-9
SCHEMA = "adapt_nds_score_v1"
SCORE_DIR = N.PRIV / "score"
RESULT = HERE / "RESULT.json"
# Operative forecasts: PACKET.md "Forecast revision before any data (00:43Z)"; the first-stated values are co-reported.
FORECASTS = {"ADAPT_GAIN": 0.60, "BEATS_ADAF_BARROW": 0.30, "TRAILS": 0.20, "ENGLISH_RETENTION_BELOW_-0.010": 0.30,
             "WITHIN_120_CU": 0.85}
FORECASTS_FIRST_STATED = {"ADAPT_GAIN": 0.70, "BEATS_ADAF_BARROW": 0.40, "TRAILS": 0.15,
                          "ENGLISH_RETENTION_BELOW_-0.010": 0.30, "WITHIN_120_CU": 0.85}
ATTRIBUTION = [N.ATTRIBUTION["terrain"], N.ATTRIBUTION["register"],
               "ADAF AO and barrow HRNets: Čož, Corns, Curran, Kocev, Kokalj 2026, J. Archaeol. Sci. Rep. 71:105733; "
               "weights CC BY-SA 4.0 (Zenodo 10.5281/zenodo.15848662), code Apache-2.0"]


# ---------------------------------------------------------------- outcome rules (packet "Measures and outcome rule")
def adapt_label(mean: float, lo95: float) -> str:
    if mean >= N.GAIN_MEAN - EPS and lo95 > EPS:
        return "ADAPT_GAIN"
    if mean <= N.HARM_MEAN + EPS:
        return "ADAPT_HARM"
    return "NULL"


def milestone_label(mean: float, lo95: float, hi95: float) -> str:
    if mean >= N.BEAT_MEAN - EPS and lo95 > EPS:
        return "BEATS_ADAF_BARROW"
    if hi95 < -EPS:
        return "TRAILS"
    return "NOT_DISTINGUISHED"


# ---------------------------------------------------------------- scoring primitives (JOINT-1 / FRESH-1 forms)
def region_setup(manifest: dict) -> dict:
    s = rs.setup(manifest)
    labels = sorted(set(s["cluster"]))
    cidx = {c: i for i, c in enumerate(labels)}
    bg = np.zeros(len(labels))
    for i, a in s["neg_area"].items():
        bg[cidx[s["cluster"][i]]] += a
    return {"s": s, "cidx": cidx, "bg": bg, "n_clusters": len(labels), "eligible_parts": len(s["eligible"]),
            "background_km2": float(bg.sum()), "tiles": {t["id"] for t in s["wins"]}, "manifest": manifest}


def score_model(objs: dict, ctx: dict) -> dict:
    s = ctx["s"]
    pool = rs.ranked_pool(objs, s)
    summ = rs.arm_summary(pool, s)
    arrays = hu.from_pool(pool, s, ctx["cidx"])
    if abs(float(hu.primary(arrays, np.ones((1, ctx["n_clusters"])), ctx["bg"])[0]) - summ["primary"]) > 1e-12:
        raise SystemExit("interval arrays disagree with region_score (internal check)")
    return {"primary": summ["primary"], "by_density": {k: summ[k]["recall"] for k in summ if k.startswith("d")},
            "all": summ["all"]["recall"], "negative_per_km2": summ["negative_per_km2"], "pool": summ["pool"],
            "arrays": arrays, "hits": {k: sorted(shd.recall_at(pool, s, d)[1])
                                       for k, d in (("d0.5", 0.5), ("d1.0", 1.0), ("d2.0", 2.0))}}


def ci_draws(a0: list, ak: list, bg: np.ndarray, draws: int, seed: int, chunk: int = 500) -> np.ndarray:
    """hier_uncertainty.hierarchical_ci's draws of the mean paired delta (JOINT-1 score_joint.ci_draws)."""
    n_c, n_p = len(bg), len(a0)
    rng = np.random.default_rng(seed)
    w_all = hu.cluster_weights(rng, n_c, draws)
    pairs = rng.integers(0, n_p, size=(draws, n_p))
    deltas = np.empty((draws, n_p))
    for s0 in range(0, draws, chunk):
        w = w_all[s0:s0 + chunk]
        for p in range(n_p):
            deltas[s0:s0 + chunk, p] = hu.primary(ak[p], w, bg) - hu.primary(a0[p], w, bg)
    if not np.all(np.isfinite(deltas)):
        raise ValueError("non-finite delta in a draw; fail closed")
    return np.take_along_axis(deltas, pairs, axis=1).mean(axis=1)


def contrast(xs: list[dict], refs: list[dict], ctx: dict, draws: int = N.BOOT_DRAWS, seed: int = N.BOOT_SEED) -> dict:
    """Paired (refs[p], xs[p]); hierarchical_ci's interval, checked against replicated draws."""
    a0, ak = [r["arrays"] for r in refs], [x["arrays"] for x in xs]
    ci = hu.hierarchical_ci(a0, ak, ctx["bg"], draws=draws, seed=seed)
    means = ci_draws(a0, ak, ctx["bg"], draws, seed)
    lo, hi = np.quantile(means, [0.025, 0.975])
    d = [x["primary"] - r["primary"] for x, r in zip(xs, refs)]
    if abs(lo - ci["lo95"]) > 1e-12 or abs(hi - ci["hi95"]) > 1e-12 or abs(ci["point_mean_delta"] - np.mean(d)) > 1e-9:
        raise SystemExit("replicated draws disagree with hierarchical_ci (internal check)")
    return {"mean": float(np.mean(d)), "lo95": ci["lo95"], "hi95": ci["hi95"], "median": ci["median"],
            "per_seed": d, "draws": draws, "seed": seed, "n_clusters": ci["n_clusters"], "n_pairs": ci["n_pairs"]}


def fused(objs_adaf: dict, objs_x: dict, ctx: dict, tag: str = "X") -> dict:
    """INTERLEAVE by FUSION-1: ADAF's own pool is model A (first), the other model is B (FRESH-1 fused_objects)."""
    s = ctx["s"]
    return fe.fused_objects(fe.order("INTERLEAVE", fe.own_pool(objs_adaf, s, "A"), fe.own_pool(objs_x, s, tag)), s)


def strata(x: dict, ctx: dict) -> dict:
    """Recall per register-height stratum on the individual manifest (FRESH-1 score_fresh.strata)."""
    parts = {p["id"]: p.get("stratum", "unknown") for p in ctx["manifest"]["parts"]}
    out = {}
    for k in ("d0.5", "d1.0", "d2.0"):
        hit = set(x["hits"][k])
        out[k] = {st: {"n": sum(1 for v in parts.values() if v == st),
                       "recall": sum(1 for p, v in parts.items() if v == st and p in hit)
                       / max(1, sum(1 for v in parts.values() if v == st))} for st in sorted(set(parts.values()))}
    return out


def public(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("arrays", "hits")}


def mean_primary(xs: list[dict]) -> float:
    return float(np.mean([x["primary"] for x in xs]))


# ---------------------------------------------------------------- binding
def refuse(msgs: list[str]) -> None:
    raise SystemExit("score refused:\n  " + "\n  ".join(msgs))


def load_freeze(path: Path) -> dict:
    lock = N.read_json(Path(path).with_name("FREEZE_LOCK.json"))
    sha = N.sha256_file(path)
    if sha != lock["freeze_sha256"]:
        refuse(["FREEZE.json differs from FREEZE_LOCK.json"])
    f = N.read_json(path)
    f["_sha256"] = sha
    return f


def static_violations(f: dict) -> list[str]:
    keys = ("score_adapt", "region_score", "hier_uncertainty", "fuse_eval", "dev_group_manifest",
            "dev_individual_manifest", "dev_infer_manifest")
    return [f"{k} differs from the freeze" for k in keys
            if N.sha256_file(N.ROOT / f["pins"][k]["path"]) != f["pins"][k]["sha256"]]


def confirm_binding(f: dict, v: list) -> dict:
    lock = N.read_json(N.ROOT / f["confirm"]["build_lock"])
    if lock["freeze_sha256"] != f["_sha256"]:
        v.append("CONFIRM_BUILD_LOCK.json was made under another freeze")
    bad = [p for p, h in lock["files"].items() if N.sha256_file(N.ROOT / p) != h]
    if bad:
        v.append(f"CONFIRM build files changed: {bad}")
    return {k: Path(N.ROOT / p) for k, p in (("group", f["confirm"]["group_manifest"]),
                                            ("individual", f["confirm"]["individual_manifest"]),
                                            ("infer", f["confirm"]["infer_manifest"]))}


def session_record(top: Path) -> dict:
    p = Path(top) / "SESSION_VERIFIED.json"
    return N.read_json(p) if p.is_file() else {}


def check_sessions(f: dict, tops: list[Path]) -> list[str]:
    v = []
    for i, top in enumerate(tops, 1):
        sv = session_record(top)
        if sv.get("status") != "PASS" or sv.get("freeze_sha256") != f["_sha256"] or sv.get("index") != i:
            v.append(f"{Path(top).name}: SESSION_VERIFIED is not PASS for session {i} under this freeze")
    return v


def load_objects(path: Path, name: str, msha: str, ckpt: str, tiles: set, v: list) -> dict:
    if not path.is_file():
        v.append(f"{path.name}: missing")
        return {}
    d = N.read_json(path)
    if d.get("model") != name or d.get("manifest_sha256") != msha or d.get("checkpoint_sha256") != ckpt:
        v.append(f"{path.name}: model, inference manifest or weights differ from the binding")
    if set(d.get("objects", {})) != tiles:
        v.append(f"{path.name}: tiles differ from the inference manifest")
    return d.get("objects", {})


def tile_keys(m: dict) -> list:
    return [[t["id"], t["role"], t["cluster"], t["core"]] for t in m["tiles"]]


def rebind(name: str, objs: dict, ckpt: str, inf: dict, inf_sha: str, unit_path: Path, out_dir: Path) -> Path:
    """FRESH-1 score_fresh.rebind: proposals made on the inference manifest, written for a scoring manifest with
    identical tiles (id, role, cluster, scored boxes)."""
    unit = N.read_json(unit_path)
    if tile_keys(inf) != tile_keys(unit):
        refuse([f"{unit_path.name}: tiles differ from the inference manifest; proposals cannot be rebound"])
    p = out_dir / f"PROPOSALS_{name}_PRIVATE.json"
    N.write_json(p, {"model": name, "checkpoint_sha256": ckpt, "manifest_sha256": N.sha256_file(unit_path),
                     "inference_manifest_sha256": inf_sha, "objects": objs}, indent=None)
    return p


def weights_of(f: dict, tops: list[Path]) -> dict[str, str]:
    """name -> checkpoint sha256: references from the freeze, trained runs from each SESSION_VERIFIED.json."""
    out = {n: m["sha256"] for n, m in f["models"].items()}
    for top in tops:
        out.update(session_record(top).get("trained", {}))
    return out


def gather(f: dict, tops: list[Path], v: list) -> dict:
    cb = confirm_binding(f, v)
    w = weights_of(f, tops)
    inf_c = N.read_json(cb["infer"])
    inf_d = N.read_json(N.ROOT / f["pins"]["dev_infer_manifest"]["path"])
    sc_tiles, dv_tiles = {t["id"] for t in inf_c["tiles"]}, {t["id"] for t in inf_d["tiles"]}
    out = {"confirm": {}, "dev": {}, "eng": {}, "wales": {}, "ireland": {}, "w": w, "cb": cb, "inf_c": inf_c,
           "inf_d": inf_d}
    for i, top in enumerate(tops, 1):
        res = Path(top) / "v1" / "results"
        for name in session_record(top).get("confirm_models", []):
            out["confirm"][name] = load_objects(res / "confirm" / f"PROPOSALS_{name}_PRIVATE.json", name,
                                                N.sha256_file(cb["infer"]), w.get(name, "?"), sc_tiles, v)
        for name in session_record(top).get("dev_models", []):
            out["dev"][name] = load_objects(res / "dev" / f"PROPOSALS_{name}_PRIVATE.json", name,
                                            f["pins"]["dev_infer_manifest"]["sha256"], w.get(name, "?"), dv_tiles, v)
        for name in session_record(top).get("trained", {}):
            p = res / "runs" / name / "HOLDOUT_OBJECTS_PRIVATE.json"
            if p.is_file():
                out["eng"][name] = N.read_json(p)["objects"]
        for reg in ("wales", "ireland"):           # optional seed-02 Welsh/Irish DEV pass (session 3 when scheduled)
            for name in session_record(top).get("wi_models", []):
                p = res / reg / f"PROPOSALS_{name}_PRIVATE.json"
                if p.is_file():
                    out[reg][name] = N.read_json(p)["objects"]
    s1 = N.ROOT / f["dev_references"]["top"]
    for name, h in f["dev_references"]["proposals"].items():
        p = s1 / "v1" / "harvest" / "main" / f"PROPOSALS_{name}_PRIVATE.json"
        if not p.is_file() or N.sha256_file(p) != h:
            v.append(f"FRESH-1 DEV reference {name}: proposals missing or changed")
            continue
        out["dev"][name] = N.read_json(p)["objects"]
    return out


# ---------------------------------------------------------------- the analysis
def names(f: dict) -> dict:
    return {"A": [N.run_name("A", s) for s in N.SEEDS], "C": [N.run_name("C", s) for s in N.SEEDS],
            "J": [N.j_name(s) for s in N.SEEDS]}


def score_unit(objs: dict, ctx: dict) -> dict:
    return {n: score_model(o, ctx) for n, o in objs.items()}


def region_block(sc: dict, nm: dict, ctx: dict, draws: int, seed: int, adaf: dict) -> dict:
    """The two packet contrasts and the descriptive ones on one region (CONFIRM or DEV)."""
    A, C, J = [sc[n] for n in nm["A"]], [sc[n] for n in nm["C"]], [sc[n] for n in nm["J"] if n in sc]
    barrow = sc["ADAF_barrow"]
    p1 = contrast(A, C, ctx, draws, seed)
    p2 = contrast(A, [barrow] * len(A), ctx, draws, seed)
    out = {"systems": {n: public(x) for n, x in sc.items()},
           "means": {"A": mean_primary(A), "C": mean_primary(C), "J": mean_primary(J) if J else None,
                     "ADAF_barrow": barrow["primary"], "ADAF_AO": sc["ADAF_AO"]["primary"],
                     "CONT": sc["CONT"]["primary"]},
           "adapt_effect_A_minus_C": p1 | {"label": adapt_label(p1["mean"], p1["lo95"])},
           "milestone_A_minus_ADAF_barrow": p2 | {"label": milestone_label(p2["mean"], p2["lo95"], p2["hi95"])},
           "descriptive": {}}
    d = out["descriptive"]
    if len(J) == len(A):
        d["A_minus_J"] = contrast(A, J, ctx, draws, seed)
        d["C_minus_J"] = contrast(C, J, ctx, draws, seed)
        d["J_minus_ADAF_barrow"] = contrast(J, [barrow] * len(J), ctx, draws, seed)
    for ref in ("ADAF_AO", "CONT"):
        d[f"{ref}_minus_ADAF_barrow"] = contrast([sc[ref]], [barrow], ctx, draws, seed)
    for tag, adaf_name in (("INTERLEAVE_ADAF_barrow_A", "ADAF_barrow"), ("INTERLEAVE_ADAF_AO_A", "ADAF_AO")):
        xs = [score_model(fused(adaf[adaf_name], adaf[n], ctx), ctx) for n in nm["A"]]
        d[tag] = {"per_seed": [x["primary"] for x in xs], "mean": mean_primary(xs),
                  "minus_ADAF_barrow": contrast(xs, [barrow] * len(xs), ctx, draws, seed)}
    return out


def strip_ci(o):
    if isinstance(o, dict):
        return {k: strip_ci(v) for k, v in o.items() if k not in ("arrays", "hits", "draws_array")}
    if isinstance(o, list):
        return [strip_ci(v) for v in o]
    return o


def analyse(freeze: Path, tops: list[Path], draws: int = N.BOOT_DRAWS, seed: int = N.BOOT_SEED) -> dict:
    f = load_freeze(freeze)
    v = static_violations(f) + check_sessions(f, tops)
    got = gather(f, tops, v)
    nm = names(f)
    need_c = nm["A"] + nm["C"] + nm["J"] + ["ADAF_barrow", "ADAF_AO", "CONT"]
    need_d = nm["A"] + nm["C"] + ["ADAF_barrow", "ADAF_AO", "CONT"]
    v += [f"CONFIRM proposals of {n} missing" for n in need_c if n not in got["confirm"]]
    v += [f"DEV proposals of {n} missing" for n in need_d if n not in got["dev"]]
    if v:
        refuse(v)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    res = {"schema": SCHEMA, "utc": N.utc(), "freeze_sha256": f["_sha256"], "confirm_district": f["confirm"]["district"],
           "rules": {"adapt": "ADAPT_GAIN if mean >= +0.03 and lo95 > 0; ADAPT_HARM if mean <= -0.03; else NULL",
                     "milestone": "BEATS_ADAF_BARROW if mean >= +0.02 and lo95 > 0; TRAILS if hi95 < 0; else "
                                  "NOT_DISTINGUISHED", "bootstrap": f"hier_uncertainty.hierarchical_ci, cluster x "
                                  f"seed, {draws} draws, seed {seed}", "tolerance": EPS}}
    cb, inf_c_sha = got["cb"], N.sha256_file(got["cb"]["infer"])
    for unit in ("group", "individual"):
        for n, o in got["confirm"].items():
            rebind(n, o, got["w"][n], got["inf_c"], inf_c_sha, cb[unit], SCORE_DIR / "confirm" / unit)
    ctx_c = {u: region_setup(N.read_json(cb[u])) for u in ("group", "individual")}
    sc_c = {u: score_unit(got["confirm"], ctx_c[u]) for u in ctx_c}
    res["confirm"] = region_block(sc_c["group"], nm, ctx_c["group"], draws, seed, got["confirm"])
    res["confirm"]["individual_unit"] = {n: public(x) for n, x in sc_c["individual"].items()}
    res["confirm"]["height_strata"] = {n: strata(x, ctx_c["individual"]) for n, x in sc_c["individual"].items()}
    res["confirm"]["context"] = {"groups": ctx_c["group"]["eligible_parts"], "clusters": ctx_c["group"]["n_clusters"],
                                 "background_km2": ctx_c["group"]["background_km2"]}
    ctx_d = region_setup(N.read_json(N.ROOT / f["pins"]["dev_group_manifest"]["path"]))
    for n, o in got["dev"].items():
        rebind(n, o, got["w"].get(n, "fresh1"), got["inf_d"], f["pins"]["dev_infer_manifest"]["sha256"],
               N.ROOT / f["pins"]["dev_group_manifest"]["path"], SCORE_DIR / "dev")
    sc_d = score_unit(got["dev"], ctx_d)
    res["dev"] = region_block(sc_d, nm, ctx_d, draws, seed, got["dev"]) | {
        "note": "development evidence: Verden was evaluated by FRESH-1 (zero-shot references reused by hash)"}
    res["english_retention"] = english(got["eng"], nm, draws, seed)
    res["wales_ireland_seed_2026092802"] = regional_dev(got, nm)
    res["primary_outcomes"] = {"ADAPT": res["confirm"]["adapt_effect_A_minus_C"]["label"],
                               "MILESTONE": res["confirm"]["milestone_A_minus_ADAF_barrow"]["label"]}
    res["cu_spent"] = cu_spent()
    res["forecasts"] = forecasts(res, res["cu_spent"])
    res["attribution"] = ATTRIBUTION
    out = strip_ci(res)
    N.write_json(RESULT, out)
    return out


def english(eng: dict, nm: dict, draws: int, seed: int) -> dict:
    if not all(n in eng for n in nm["A"] + nm["C"]):
        return {"status": "MISSING", "have": sorted(eng)}
    ctx = region_setup(rs.english_manifest())
    A = [score_model(eng[n], ctx) for n in nm["A"]]
    C = [score_model(eng[n], ctx) for n in nm["C"]]
    ci = contrast(A, C, ctx, draws, seed)
    return {"A": [x["primary"] for x in A], "C": [x["primary"] for x in C], "A_minus_C": ci,
            "below_minus_0.010": ci["mean"] < -0.010}


def regional_dev(got: dict, nm: dict) -> dict:
    """Seed 2026092802 only, point values (descriptive): A - C on HR1 Wales DEV and HR3 Ireland DEV."""
    sys.path.insert(0, str(N.J1_CHAIN))
    import joint_common as J  # noqa: E402  frozen paths of the Welsh and Irish DEV manifests
    out = {}
    a, c = nm["A"][0], nm["C"][0]
    for reg, path in (("wales", J.W_DEV), ("ireland", J.I_DEV)):
        if a not in got[reg] or c not in got[reg]:
            out[reg] = {"status": "NOT_RUN"}
            continue
        ctx = region_setup(N.read_json(path))
        xa, xc = score_model(got[reg][a], ctx), score_model(got[reg][c], ctx)
        out[reg] = {"A": xa["primary"], "C": xc["primary"], "A_minus_C": xa["primary"] - xc["primary"]}
    return out


def cu_spent(path: Path = N.CU_SPEND) -> float | None:
    """The chain's recorded spend (driver CU_SPEND.jsonl; the last row per attempt, as the driver's cu_plan)."""
    if not path.is_file():
        return None
    per = {}
    for x in path.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            per[r["attempt"]] = r
    return round(sum(float(r.get("spent", 0.0)) for r in per.values()), 2)


def forecasts(res: dict, cu_total: float | None = None) -> dict:
    """Brier scores of the operative (00:43Z revision) forecasts; first-stated values co-reported. WITHIN_120_CU is
    scored when the driver's CU ledger total is given."""
    occurred = {"ADAPT_GAIN": res["primary_outcomes"]["ADAPT"] == "ADAPT_GAIN",
                "BEATS_ADAF_BARROW": res["primary_outcomes"]["MILESTONE"] == "BEATS_ADAF_BARROW",
                "TRAILS": res["primary_outcomes"]["MILESTONE"] == "TRAILS",
                "ENGLISH_RETENTION_BELOW_-0.010": res["english_retention"].get("below_minus_0.010"),
                "WITHIN_120_CU": None if cu_total is None else cu_total <= N.CU_CEILING}
    return {k: {"p": p, "p_first_stated": FORECASTS_FIRST_STATED[k], "occurred": occurred[k],
                "brier": None if occurred[k] is None else round((p - float(occurred[k])) ** 2, 4)}
            for k, p in FORECASTS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score",))
    ap.add_argument("freeze", type=Path)
    ap.add_argument("tops", type=Path, nargs=3)
    ap.add_argument("--draws", type=int, default=N.BOOT_DRAWS)
    ap.add_argument("--seed", type=int, default=N.BOOT_SEED)
    a = ap.parse_args()
    out = analyse(a.freeze, list(a.tops), a.draws, a.seed)
    print(json.dumps({"primary_outcomes": out["primary_outcomes"],
                      "adapt": {k: out["confirm"]["adapt_effect_A_minus_C"][k] for k in ("mean", "lo95", "hi95")},
                      "milestone": {k: out["confirm"]["milestone_A_minus_ADAF_barrow"][k]
                                    for k in ("mean", "lo95", "hi95")}}, indent=1))


if __name__ == "__main__":
    main()
