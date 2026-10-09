"""STAGE2-D1-CONTROLS scoring and the frozen gate (PACKET.md "Endpoint and gate"). Local CPU only; labels never travel.

Runs only after session 1's SESSION_VERIFIED.json is PASS under this packet's freeze, i.e. after all ten prescribed
controls are complete and verified; with fewer, nothing archaeological is scored and RESULT.json records INCOMPLETE
(ROUND4_REPLY.md R4-1 last table row; R4-2: no score enters any execution decision).

Primary scorer: benchmark v0 on Verden exactly as STAGE1-D1-V0 scored it (work/stage1_d1_v0_2026-10-07/score_v0.py
functions imported read-only: one-to-one maximum-cardinality matching, 15 m footprint reach, K = floor(b x 422.84 km2)
counting every in-core proposal, RAW = every in-core proposal in rank_key order, NMS60 = region_score.ranked_pool,
legacy IGNORE handling), on the same 717 tiles / 728 individual barrows.

Systems: ADAF barrow (FRESH-1 proposals, unchanged), the ten D1 runs (run_s3 proposals, pinned by D1's
SESSION_VERIFIED), the ten equal-update nonlocal mixed-class replay controls (this packet's session 1).

Gate (RAW, b = 0.5), decided on unrounded values in integer matches (N = 728):
  d_s = matches(D1_s) - matches(CTL_s);  PASS iff 100 x sum_s d_s >= 3 x N x 10 (mean >= +0.030 inclusive, i.e. >=
  21.84 matches; with integers sum >= 219) AND #{s: d_s > 0} >= 8 (strictly positive; a zero difference is not
  positive). Otherwise FAIL.
Clear advantage over the prespecified replay control on Verden: PASS AND the primary paired 2 km block-bootstrap lower
95 % bound (boot_s2.py, frozen, 5,000 replicates) > 0, unrounded, strict; sensitivity analyses disclosed, never
substituted.
Outcome wording: Pro's R4-1 table (OUTCOMES below; the row is chosen by outcome_row, prespecified).
Sensitivity (prespecified): evaluator v0.1 (work/evaluator_v0_1_2026-10-07/, role-based IGNORE rule; ROUND4_REPLY.md
R4-6) re-scores the same proposals only if that directory holds a FREEZE receipt at scoring time; otherwise the
sensitivity is refused (recorded) and the primary is unaffected.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/stage2_d1_controls_2026-10-07/score_s2.py FREEZE S1_TOP
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2_common as S  # noqa: E402

N = S.N
sys.path.insert(0, str(S.ROOT / "work/stage1_d1_v0_2026-10-07"))
import score_v0 as V0  # noqa: E402  STAGE1-D1-V0 (completed, read-only): v0, load_checked, tile_keys, rs, IND, INF
import score_match as SM  # noqa: E402  ADAPT-MATCH-0 (frozen): dev_context, load_objects, SA.score_model (FRESH-1)
import boot_s2 as BS  # noqa: E402  frozen paired bootstrap (this packet)
sys.path.insert(0, str(S.CHAIN))

BOOT = HERE / "BOOT_S2.json"
V01_DIR = S.ROOT / "work/evaluator_v0_1_2026-10-07"
V01_RECEIPTS = ("FREEZE.json", "FREEZE.md", "freeze/FREEZE.json")
V01_ENTRY_DEFAULT = ("score_v0_1.py", "score_systems")
EPS = 1e-12

# Pro, ROUND4_REPLY.md R4-1 "Retain the numerical gate; rename the inferential claim", outcome table (verbatim rows)
OUTCOMES = {
    "pass_clear": ("Practical gate passes and primary joint lower bound is positive",
                   "The selected local barrow-training package exceeded the specified replay package by the reported "
                   "amount and met both predefined criteria."),
    "pass_not_clear": ("Practical gate passes but interval criterion fails",
                       "The practical replication criterion was met; the predefined uncertainty criterion was not."),
    "positive_below_threshold": ("Mean improvement is positive but below three points",
                                 "A positive estimated advantage was observed, but it did not meet the "
                                 "material-improvement threshold."),
    "size_without_consistency": ("Mean is at least three points but fewer than 8/10 differences are positive",
                                 "The average improvement met the size threshold but failed the run-consistency "
                                 "criterion."),
    "neither_leads": ("Neither treatment clearly leads",
                      "The study did not demonstrate a material incremental advantage over this replay package. Both "
                      "may nevertheless improve on unchanged ADAF barrow."),
    "replay_better": ("Replay performs better",
                      "The specified replay package performed better on Verden under this recipe. Stage 1’s "
                      "improvement over unchanged ADAF barrow still stands."),
    "pass_with_collapsed_controls": ("Gate passes with collapsed controls",
                                     "The protocol-level comparison passed, including the prespecified training "
                                     "failures; the difference includes replay instability."),
    "incomplete": ("Ten prescribed runs are not completed within the cap",
                   "The experiment is incomplete under its registered gate. Do not substitute a smaller denominator.")}


def collapse_reasons(n_props: int, primary: float, det_on_end: bool) -> list[str]:
    """RECIPE.md "Collapsed definition", BARROW_L reference (ADAF barrow zero-shot FRESH-1 primary 0.5556)."""
    out = []
    if n_props == 0:
        out.append("a_zero_proposals")
    if primary < 0.5 * S.REFERENCE_PRIMARY:
        out.append("b_primary_below_half_reference")
    if det_on_end:
        out.append("c_detector_on_at_end")
    return out


def systems(s1_top: Path, freeze_sha: str, bad: list) -> tuple[dict, dict, dict]:
    ind = V0.rj(V0.IND)
    inf_sha, tiles = V0.sha(V0.INF), {t["id"] for t in ind["tiles"]}
    objs, prov, ctl_meta = {}, {}, {}
    objs["ADAF_barrow"] = V0.load_checked(V0.ADAF, "ADAF_barrow", None, V0.ADAF_CKPT, inf_sha, tiles, bad)
    prov["ADAF_barrow"] = {"path": N.rel(V0.ADAF), "sha256": V0.sha(V0.ADAF)}
    d1v = V0.rj(S.ROOT / S.D1_S3_TOP / "SESSION_VERIFIED.json")
    if N.sha256_file(S.ROOT / S.D1_S3_TOP / "SESSION_VERIFIED.json") != S.D1_S3_VERIFIED_SHA256:
        bad.append("D1 session-3 SESSION_VERIFIED.json differs from its pin")
    s2v = V0.rj(s1_top / "SESSION_VERIFIED.json")
    for seed in S.SEEDS:
        for name, top, sv in ((S.d1_name(seed), S.ROOT / S.D1_S3_TOP, d1v), (S.run_name(seed), s1_top, s2v)):
            key = f"dev/PROPOSALS_{name}_PRIVATE.json"
            p = top / "v1/results" / key
            ckpt = sv.get("trained", {}).get(name, "?")
            if name.startswith("D1") and ckpt != S.D1_FINAL[seed]:
                bad.append(f"{name}: D1 checkpoint differs from the pairing table")
            objs[name] = V0.load_checked(p, name, sv.get("proposals", {}).get(key), ckpt, inf_sha, tiles, bad)
            prov[name] = {"path": N.rel(p), "sha256": sv.get("proposals", {}).get(key), "checkpoint_sha256": ckpt}
        ctl_meta[S.run_name(seed)] = {"detector": s2v.get("detectors", {}).get(S.run_name(seed), {}),
                                      "checkpoint_sha256": s2v.get("trained", {}).get(S.run_name(seed))}
    return objs, prov, ctl_meta


def completeness(s1_top: Path, freeze_sha: str) -> list[str]:
    """Empty when all ten prescribed controls are complete and verified under this freeze (SESSION_VERIFIED PASS,
    barrier PASS, ten trained hashes, ten verified proposal files)."""
    p = s1_top / "SESSION_VERIFIED.json"
    sv = V0.rj(p) if p.is_file() else {}
    why = []
    if sv.get("status") != "PASS" or sv.get("freeze_sha256") != freeze_sha or sv.get("index") != 1:
        why.append("stage-2 session 1 is not verified PASS under this freeze")
    trained = sv.get("trained", {})
    missing = [S.run_name(s) for s in S.SEEDS if not trained.get(S.run_name(s))]
    if missing:
        why.append(f"{len(missing)} of the ten prescribed controls are not complete and verified")
    if sv.get("barrier", {}).get("status") != "PASS":
        why.append("the first-run technical barrier is not recorded PASS")
    return why


def collapse_table(objs: dict, ctl_meta: dict) -> dict:
    """FRESH-1 group primary (ADAPT-MATCH-0 scoring path, as score_d1) and the collapse reasons of every control."""
    ctx = SM.dev_context()
    out = {}
    for seed in S.SEEDS:
        name = S.run_name(seed)
        primary = SM.SA.score_model(objs[name], ctx)["primary"]
        n_props = sum(len(x) for x in objs[name].values())
        reasons = collapse_reasons(n_props, primary, bool(ctl_meta[name]["detector"].get("on_at_end")))
        out[name] = {"fresh1_primary": primary, "proposals": n_props, "collapse_reasons": reasons,
                     "collapsed": bool(reasons)}
    return out


def recall(res: dict, name: str, dec: str, b: str) -> float:
    return res[name][dec][b]["recall"]


def paired(res: dict, dec: str, b: str, seeds: tuple = S.SEEDS) -> dict:
    dm = {s: res[S.d1_name(s)][dec][b]["matches"] - res[S.run_name(s)][dec][b]["matches"] for s in seeds}
    d = {s: recall(res, S.d1_name(s), dec, b) - recall(res, S.run_name(s), dec, b) for s in seeds}
    return {"per_seed": {str(k): v for k, v in d.items()}, "per_seed_matches": {str(k): v for k, v in dm.items()},
            "mean": float(np.mean(list(d.values()))), "sum_matches": int(sum(dm.values())),
            "positive": sum(v > 0 for v in dm.values()), "min": min(d.values()), "max": max(d.values()),
            "n": len(d)}


def gate_decision(sum_matches: int, positive: int, n_entities: int, n_pairs: int = S.GATE["n_pairs"]) -> dict:
    """Integer form of mean(d) >= 0.030 (inclusive) and >= 8 strictly positive differences."""
    size_ok = 100 * sum_matches >= round(100 * S.GATE["mean_diff_min"]) * n_entities * n_pairs
    cons_ok = positive >= S.GATE["positive_min"]
    return {"size_criterion": size_ok, "consistency_criterion": cons_ok,
            "threshold_matches_mean": S.GATE["mean_diff_min"] * n_entities,
            "mean_matches": sum_matches / n_pairs, "verdict": "PASS" if size_ok and cons_ok else "FAIL"}


def gate(res: dict, n_entities: int) -> dict:
    p = paired(res, S.GATE["decoder"], S.GATE["b"])
    g = gate_decision(p["sum_matches"], p["positive"], n_entities)
    float_ok = p["mean"] >= S.GATE["mean_diff_min"] - EPS and p["positive"] >= S.GATE["positive_min"]
    if float_ok != (g["verdict"] == "PASS"):
        raise SystemExit("integer and float gate forms disagree")
    return {**S.GATE, **p, **g}


def clear_label(g: dict, boot: dict | None) -> dict:
    if boot is None or "primary" not in boot:
        return {"label": "NOT_EVALUATED", "reason": "paired bootstrap not computed (incomplete or invalid pairs)"}
    lo = boot["primary"]["lower95"]
    if g["verdict"] != "PASS":
        return {"label": "NOT_APPLICABLE_GATE_FAILED", "lower95": lo, "upper95": boot["primary"]["upper95"]}
    return {"label": "CLEAR_ADVANTAGE_OVER_REPLAY_CONTROL" if boot["primary"]["clear"] else "PASS_NOT_CLEAR",
            "lower95": lo, "upper95": boot["primary"]["upper95"], "sensitivity": boot["sensitivity_raw_0.5"],
            "sensitivity_sign_differs": any((v["lower95"] > 0) != boot["primary"]["clear"]
                                            for v in boot["sensitivity_raw_0.5"].values())}


def outcome_row(g: dict | None, clear: dict, collapsed: int, upper95: float | None) -> list[str]:
    """Prespecified mapping to Pro's R4-1 rows. 'Replay performs better' = mean < 0 with the primary interval's upper
    bound < 0; 'neither clearly leads' = mean <= 0 otherwise. The collapsed-controls row is added to a PASS."""
    if g is None:
        return ["incomplete"]
    if g["verdict"] == "PASS":
        rows = ["pass_clear" if clear["label"] == "CLEAR_ADVANTAGE_OVER_REPLAY_CONTROL" else "pass_not_clear"]
        return rows + (["pass_with_collapsed_controls"] if collapsed else [])
    if g["size_criterion"]:
        return ["size_without_consistency"]
    if g["sum_matches"] > 0:
        return ["positive_below_threshold"]
    if g["sum_matches"] < 0 and upper95 is not None and upper95 < 0:
        return ["replay_better"]
    return ["neither_leads"]


def three_way(res: dict) -> dict:
    """D1 - ADAF barrow, control - ADAF barrow, D1 - control: every decoder and budget (10-seed means, ranges)."""
    out = {}
    for dec in ("RAW", "NMS60"):
        for b in map(str, S.B):
            ref = recall(res, "ADAF_barrow", dec, b)
            d1 = [recall(res, S.d1_name(s), dec, b) for s in S.SEEDS]
            ct = [recall(res, S.run_name(s), dec, b) for s in S.SEEDS]
            out[f"{dec}_{b}"] = {"adaf_barrow": ref, "d1_mean": float(np.mean(d1)), "control_mean": float(np.mean(ct)),
                                 "d1_minus_adaf": float(np.mean(d1)) - ref, "control_minus_adaf": float(np.mean(ct)) - ref,
                                 "control_positive_vs_adaf": sum(x - ref > 0 for x in ct),
                                 "d1_minus_control": paired(res, dec, b)}
    return out


# ---------------------------------------------------------------- evaluator v0.1 sensitivity (prespecified hook)
def _rel(p: Path) -> str:
    return N.rel(p) if Path(p).resolve().is_relative_to(S.ROOT) else str(p)


def v01_sensitivity(objs: dict) -> dict:
    """Runs only with a FREEZE receipt in work/evaluator_v0_1_2026-10-07/. Interface: the module named by FREEZE.json
    "entrypoint" (default score_v0_1.py) exposes FUNCTION(objs) -> {system: {"RAW": {"0.5": {"matches", "recall"}}}}
    (FREEZE.json "function", default score_systems). Records receipt and module sha256. Never alters the primary."""
    receipts = [V01_DIR / r for r in V01_RECEIPTS if (V01_DIR / r).is_file()]
    if not receipts:
        return {"status": "REFUSED_NO_FREEZE", "reason": f"no FREEZE receipt in {_rel(V01_DIR)} at scoring time"}
    rec = {"receipts_sha256": {_rel(p): N.sha256_file(p) for p in receipts}}
    fj = V01_DIR / "FREEZE.json"
    meta = N.read_json(fj) if fj.is_file() else {}
    entry = V01_DIR / str(meta.get("entrypoint", V01_ENTRY_DEFAULT[0]))
    fn = str(meta.get("function", V01_ENTRY_DEFAULT[1]))
    if not entry.is_file():
        return {**rec, "status": "REFUSED_INTERFACE_ABSENT", "reason": f"{_rel(entry)} missing"}
    rec["entrypoint"] = {"path": _rel(entry), "sha256": N.sha256_file(entry), "function": fn}
    pins = meta.get("pins") or meta.get("files") or {}
    if isinstance(pins, dict):
        drift = [k for k, v in pins.items() if isinstance(v, str) and len(v) == 64 and (S.ROOT / k).is_file()
                 and N.sha256_file(S.ROOT / k) != v]
        if drift:
            return {**rec, "status": "REFUSED_FREEZE_DRIFT", "reason": f"files differ from the v0.1 freeze: {drift[:3]}"}
    spec = importlib.util.spec_from_file_location("evaluator_v0_1_entry", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, fn):
        return {**rec, "status": "REFUSED_INTERFACE_ABSENT", "reason": f"{fn} not defined"}
    res = getattr(mod, fn)(objs)
    n_ent = res.get("_entities_N") if isinstance(res, dict) else None
    p = paired(res, "RAW", "0.5")
    out = {**rec, "status": "COMPUTED", "label": "prespecified sensitivity (evaluator v0.1); not the primary",
           "raw_0.5": p}
    if n_ent:
        out["gate_form"] = gate_decision(p["sum_matches"], p["positive"], int(n_ent))
    return out


# ---------------------------------------------------------------- scoring
def incomplete_record(f: dict, s1_top: Path, why: list[str]) -> dict:
    out = {"schema": "s2_result_v2", "utc": N.utc(), "freeze_sha256": f["_sha256"], "session_top": N.rel(s1_top),
           "verdict": "INCOMPLETE", "why": why, "outcome": [OUTCOMES["incomplete"]],
           "note": "no archaeological score of any control is computed; no smaller denominator"}
    N.write_json(S.RESULT, out)
    return out


def score(freeze: Path, s1_top: Path) -> dict:
    import sessions_s2 as SS2
    f = SS2.load_freeze(freeze)
    why = completeness(s1_top, f["_sha256"])
    if why:
        return incomplete_record(f, s1_top, why)
    ind, inf = V0.rj(V0.IND), V0.rj(V0.INF)
    if V0.tile_keys(ind) != V0.tile_keys(inf):
        raise SystemExit("individual and inference manifests differ in tiles")
    s = V0.rs.setup(ind)
    from shapely.geometry import box
    from shapely.ops import unary_union
    area = unary_union([box(*b) for t in ind["tiles"] for b in t["core"]]).area / 1e6
    bad: list[str] = []
    objs, prov, meta = systems(s1_top, f["_sha256"], bad)
    if bad:
        raise SystemExit(f"scoring refused (technical invalidity; recovery policy applies): {bad[:6]}")
    res = {m: V0.v0(o, s, area) for m, o in objs.items()}
    n_ent = len(s["eligible"])
    stage1 = N.read_json(S.ROOT / S.STAGE1_RESULT)
    repro = {m: all(abs(res[m][d][b]["recall"] - stage1["systems"][m][d][b]["recall"]) < EPS
                    for d in ("RAW", "NMS60") for b in map(str, S.B))
             for m in ["ADAF_barrow"] + [S.d1_name(x) for x in S.SEEDS]}
    col = collapse_table(objs, meta)
    g = gate(res, n_ent)
    pairs = [(S.d1_name(x), S.run_name(x)) for x in S.SEEDS]
    boot = BS.run(objs, s, ind, pairs, res)
    N.write_json(BOOT, {"schema": "s2_boot_v1", "freeze_sha256": f["_sha256"], **boot})
    clear = clear_label(g, boot)
    n_col = sum(v["collapsed"] for v in col.values())
    rows = outcome_row(g, clear, n_col, boot["primary"]["upper95"])
    healthy = tuple(x for x in S.SEEDS if not col[S.run_name(x)]["collapsed"])
    out = {"schema": "s2_result_v2", "utc": N.utc(), "freeze_sha256": f["_sha256"], "session_top": N.rel(s1_top),
           "entities_N": n_ent, "total_scored_km2": area,
           "K": {str(b): math.floor(b * area + 1e-9) for b in S.B}, "provenance": prov, "systems": res,
           "stage1_reproduction": {"identical": all(repro.values()), "per_system": repro},
           "collapse": col, "gate": g, "clear_advantage_over_replay_control": clear,
           "outcome": [OUTCOMES[r] for r in rows], "outcome_keys": rows,
           "boot": {"path": N.rel(BOOT), "sha256": N.sha256_file(BOOT), "primary": boot["primary"],
                    "sensitivity_raw_0.5": boot["sensitivity_raw_0.5"]},
           "three_way": three_way(res),
           "sensitivity_excluding_collapsed_controls": (paired(res, "RAW", "0.5", healthy) if healthy else None),
           "sensitivity_evaluator_v0_1": v01_sensitivity(objs),
           "note": "collapsed controls are results and stay in the gate; the exclusion line is descriptive only"}
    N.write_json(S.RESULT, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("freeze", type=Path)
    ap.add_argument("s1_top", type=Path)
    a = ap.parse_args()
    r = score(a.freeze, a.s1_top)
    print(json.dumps({k: r.get(k) for k in ("verdict", "gate", "clear_advantage_over_replay_control", "outcome_keys")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
