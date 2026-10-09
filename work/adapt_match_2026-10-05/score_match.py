"""ADAPT-MATCH-0 scorer (CPU; WSL adaf-h2h-venv): the packet's measures on Verden DEV, exactly.

  python -B score_match.py inputs [--a-top I=TOP ...]       # reference proposals present? exit 70 while any is missing
  python -B score_match.py zero-shot OUT.json                # FRESH-1's zero-shot primaries reproduced (no new data)
  python -B score_match.py score FREEZE TOP1 TOP2 [--draws N --seed S]   (tests only change --draws / --seed)

Metric: FRESH-1's benchmark primary (barrow-group recall at 0.5, 1 and 2 background proposals per km2, their mean) on
FRESH-1's frozen Verden group manifest: HR1 region_score.py and hier_uncertainty.py byte-identical, through ADAPT-NDS-1
score_adapt's primitives (region_setup, score_model, contrast, ci_draws, fused; imported, never edited). Reach 15 m,
NMS 60 m, 119 clusters.
  Main contrast: A (3-seed mean) - the better adapted baseline (each a 3-seed mean). Cluster x seed bootstrap: 2,000
    draws, seed 20261005, the cluster weights and seed resamples of hier_uncertainty.hierarchical_ci (one stream, seeds
    paired by index: A_NDS_<s>, AO_L_<s>, BARROW_L_<s>); inside every draw the baseline with the higher resampled mean
    is selected again. A_AHEAD if mean >= +0.03 and lo95 > 0; A_BEHIND if hi95 < 0; else NOT_DISTINGUISHED
    (tolerance 1e-9 as FRESH-1 / ADAPT-NDS-1). Descriptive development evidence, not a claim.
  Also reported: A - AO_L and A - BARROW_L (hierarchical_ci); AO_L - ADAF AO zero-shot and BARROW_L - ADAF barrow
    zero-shot (each seed paired with the one zero-shot run); INTERLEAVE(BARROW_L, A) per seed (FUSION-1 fuse_eval;
    BARROW_L first); seed SDs (ddof 1); the zero-shot gap (best zero-shot ADAF - S*) and the adapted gap; forecasts.
Reference proposals are reused by hash, never recomputed: A_NDS_<seed> from ADAPT-NDS-1's verified sessions
(TOP/v1/results/dev/PROPOSALS_A_NDS_<seed>_PRIVATE.json against that session's SESSION_VERIFIED.json, pinned by this
packet's freeze) and FRESH-1's zero-shot ADAF AO, ADAF barrow and S* (J_2026092803) from its session 1 harvest.
Proposals are made on the label-free inference manifest and scored on the group manifest after a tile-identity check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import match_common as M  # noqa: E402

N = M.N
import score_adapt as SA  # noqa: E402  frozen ADAPT-NDS-1 scorer primitives (puts region_score, hu, fuse_eval on path)

hu = SA.hu
REFUSE_EXIT = 70
SCHEMA = "adapt_match_score_v1"


# ---------------------------------------------------------------- outcome label (packet "Measures")
def main_label(mean: float, lo95: float, hi95: float) -> str:
    if mean >= M.AHEAD_MEAN - M.EPS and lo95 > M.EPS:
        return "A_AHEAD"
    if hi95 < -M.EPS:
        return "A_BEHIND"
    return "NOT_DISTINGUISHED"


def refuse(msgs: list[str]) -> None:
    print("score refused:\n  " + "\n  ".join(msgs), file=sys.stderr)
    raise SystemExit(REFUSE_EXIT)


# ---------------------------------------------------------------- reference proposals (by hash)
def a_tops(explicit: dict[int, Path] | None = None) -> tuple[dict[int, Path], list[str]]:
    """ADAPT-NDS-1 session TOPs with a PASS SESSION_VERIFIED.json under its freeze ba07548b (one per index)."""
    explicit, out, notes = explicit or {}, {}, []
    for i in (1, 2, 3):
        cands = [explicit[i]] if i in explicit else sorted(M.NDS_RUNS.glob(f"run_s{i}_*"))
        ok = []
        for top in cands:
            sv = Path(top) / "SESSION_VERIFIED.json"
            rec = N.read_json(sv) if sv.is_file() else {}
            if rec.get("status") == "PASS" and rec.get("index") == i and rec.get("freeze_sha256") == M.NDS_FREEZE_SHA256:
                ok.append(Path(top))
        if len(ok) == 1:
            out[i] = ok[0]
        elif not ok:
            notes.append(f"ADAPT-NDS-1 session {i}: no verified session yet ({len(cands)} attempt dirs)")
        else:
            notes.append(f"ADAPT-NDS-1 session {i}: {len(ok)} verified sessions; name one with --a-top {i}=TOP")
    return out, notes


def expected_a_paths() -> dict[str, dict]:
    """Where ADAPT-NDS-1 writes A's Verden proposals (chain_lib_adapt.sh do_dev -> lgln_pass dev; driver TOP naming;
    sessions_adapt.verify records the hash) and where the hash is read from."""
    rel = M.NDS_RUNS.relative_to(M.ROOT).as_posix()
    return {M.a_name(s): {"session": i, "proposals": f"{rel}/run_s{i}_<UTC stamp>/v1/results/dev/"
                                                      f"PROPOSALS_{M.a_name(s)}_PRIVATE.json",
                          "harvested_copy": f"{rel}/run_s{i}_<UTC stamp>/v1/harvest/dev/"
                                            f"PROPOSALS_{M.a_name(s)}_PRIVATE.json",
                          "hash_from": f"{rel}/run_s{i}_<UTC stamp>/SESSION_VERIFIED.json proposals"
                                       f"['dev/PROPOSALS_{M.a_name(s)}_PRIVATE.json'] (status PASS, index {i}, "
                                       f"freeze ba07548b)",
                          "attempt_dirs_now": sorted(p.name for p in M.NDS_RUNS.glob(f"run_s{i}_*"))}
            for i, s in enumerate(M.SEEDS, 1)}


def a_reference(i: int, top: Path) -> tuple[dict | None, str | None]:
    seed = M.SEEDS[i - 1]
    name = M.a_name(seed)
    sv_path = top / "SESSION_VERIFIED.json"
    sv = N.read_json(sv_path)
    key = f"dev/PROPOSALS_{name}_PRIVATE.json"
    p = top / "v1" / "results" / key
    want, ckpt = sv.get("proposals", {}).get(key), sv.get("trained", {}).get(name)
    if not p.is_file() or not want or not ckpt:
        return None, f"{name}: DEV proposals or their SESSION_VERIFIED record missing in {N.rel(top)}"
    if N.sha256_file(p) != want:
        return None, f"{name}: DEV proposals differ from SESSION_VERIFIED.json"
    return {"name": name, "seed": seed, "path": N.rel(p), "sha256": want, "checkpoint_sha256": ckpt,
            "session_verified": N.rel(sv_path), "session_verified_sha256": N.sha256_file(sv_path)}, None


def a_references(explicit: dict[int, Path] | None = None) -> tuple[dict[str, dict], list[str]]:
    tops, missing = a_tops(explicit)
    out = {}
    for i, top in sorted(tops.items()):
        rec, why = a_reference(i, top)
        if rec:
            out[rec["name"]] = rec
        else:
            missing.append(why)
    return out, missing


def zero_shot_references() -> tuple[dict[str, dict], list[str]]:
    top = M.ROOT / M.FRESH1_S1
    sv = N.read_json(top / "SESSION_VERIFIED.json")
    out, bad = {}, []
    for name, want in M.FRESH1_PROPOSALS.items():
        key = f"main/PROPOSALS_{name}_PRIVATE.json"
        p = top / "v1" / "harvest" / key
        if sv.get("proposals", {}).get(key) != want or not p.is_file() or N.sha256_file(p) != want:
            bad.append(f"FRESH-1 zero-shot {name}: proposals missing or changed")
            continue
        out[name] = {"name": name, "path": N.rel(p), "sha256": want}
    return out, bad


def check_inputs(explicit: dict[int, Path] | None = None) -> dict:
    """Every reference the score needs: A at the three seeds and the three zero-shot models. Refuses (exit 70) with
    the list of what is missing; ADAPT-NDS-1 sessions 2-3 had not run when this was built."""
    a, missing = a_references(explicit)
    z, bad = zero_shot_references()
    need = [M.a_name(s) for s in M.SEEDS]
    missing += [f"{n}: not available" for n in need if n not in a and not any(m.startswith(n) for m in missing)]
    rec = {"a_references": a, "zero_shot": z, "missing": missing + bad, "status": "READY" if not (missing or bad)
           else "REFUSED", "utc": N.utc()}
    if missing or bad:
        print(json.dumps(rec, indent=1))
        refuse(missing + bad)
    return rec


# ---------------------------------------------------------------- scoring context
def dev_context(f: dict | None = None) -> dict:
    """FRESH-1's Verden group manifest (scoring) and the inference manifest the proposals were made on."""
    pins = (f or {}).get("pins", {})
    paths = {k: M.ROOT / (pins[k]["path"] if k in pins else v) for k, v in
             {**M.DEV_LABELS, "dev_infer_manifest": M.DEV_DATA["dev_infer_manifest"]}.items()}
    inf, grp = N.read_json(paths["dev_infer_manifest"]), N.read_json(paths["dev_group_manifest"])
    if SA.tile_keys(inf) != SA.tile_keys(grp):
        refuse(["the Verden inference and group manifests differ in tiles; proposals cannot be rebound"])
    ctx = SA.region_setup(grp)
    ctx["infer_sha256"] = N.sha256_file(paths["dev_infer_manifest"])
    ctx["infer_tiles"] = {t["id"] for t in inf["tiles"]}
    return ctx


def load_objects(path: Path, name: str, want_sha: str | None, ckpt: str | None, ctx: dict, v: list) -> dict:
    if not path.is_file():
        v.append(f"{name}: {N.rel(path)} missing")
        return {}
    if want_sha and N.sha256_file(path) != want_sha:
        v.append(f"{name}: proposals differ from their pinned sha256")
    d = N.read_json(path)
    if d.get("model") != name or d.get("manifest_sha256") != ctx["infer_sha256"]:
        v.append(f"{name}: model or inference manifest differs from the binding")
    if ckpt and d.get("checkpoint_sha256") != ckpt:
        v.append(f"{name}: weights differ from the binding")
    if set(d.get("objects", {})) != ctx["infer_tiles"]:
        v.append(f"{name}: tiles differ from the inference manifest")
    return d.get("objects", {})


def zero_shot(out: Path | None = None) -> dict:
    """FRESH-1's recorded group primaries of ADAF AO, ADAF barrow and S*, reproduced from its hashed proposals."""
    z, bad = zero_shot_references()
    if bad:
        refuse(bad)
    ctx = dev_context()
    rec_fresh = N.read_json(M.ROOT / M.FRESH1_RESULT)
    rows, v = {}, []
    for name, r in z.items():
        objs = load_objects(M.ROOT / r["path"], name, r["sha256"], None, ctx, v)
        got = SA.score_model(objs, ctx)["primary"]
        want = rec_fresh["benchmark"]["group"][name]["primary"]
        rows[name] = {"primary": got, "fresh1_recorded": want, "reproduced": abs(got - want) <= 1e-12}
    units = rec_fresh["units"]["group"]
    res = {"models": rows, "n_clusters": ctx["n_clusters"], "eligible_parts": ctx["eligible_parts"],
           "background_km2": ctx["background_km2"], "fresh1_units": units,
           "status": "PASS" if not v and all(r["reproduced"] for r in rows.values()) and
           ctx["n_clusters"] == units["n_clusters"] == 119 and ctx["eligible_parts"] == units["eligible_parts"]
           else "FAIL", "violations": v, "utc": N.utc()}
    if out is not None:
        N.write_json(out, res)
    return res


# ---------------------------------------------------------------- the main contrast: baseline re-selected per draw
def reselect(a: list[dict], ao: list[dict], br: list[dict], ctx: dict, draws: int = M.BOOT_DRAWS,
             seed: int = M.BOOT_SEED, chunk: int = 500) -> dict:
    """A (seed mean) - max(AO_L mean, BARROW_L mean), the max taken inside each cluster x seed draw."""
    bg, n_p = ctx["bg"], len(a)
    if not n_p or len(ao) != n_p or len(br) != n_p:
        raise ValueError("need three equal, non-empty seed lists")
    rng = np.random.default_rng(seed)                       # hierarchical_ci's stream: cluster weights, then pairs
    w_all = hu.cluster_weights(rng, len(bg), draws)
    pairs = rng.integers(0, n_p, size=(draws, n_p))
    arms = {"A": a, "AO_L": ao, "BARROW_L": br}
    prim = {k: np.empty((draws, n_p)) for k in arms}
    for s0 in range(0, draws, chunk):
        w = w_all[s0:s0 + chunk]
        for k, xs in arms.items():
            for p in range(n_p):
                prim[k][s0:s0 + chunk, p] = hu.primary(xs[p]["arrays"], w, bg)
    if not all(np.all(np.isfinite(v)) for v in prim.values()):
        raise ValueError("non-finite primary in a draw; fail closed")
    means = {k: np.take_along_axis(v, pairs, axis=1).mean(axis=1) for k, v in prim.items()}
    pick_barrow = means["BARROW_L"] >= means["AO_L"]
    delta = means["A"] - np.where(pick_barrow, means["BARROW_L"], means["AO_L"])
    check_reselect(delta, pick_barrow, arms, bg, draws, seed)
    point = {k: float(np.mean([x["primary"] for x in xs])) for k, xs in arms.items()}
    better = "BARROW_L" if point["BARROW_L"] >= point["AO_L"] else "AO_L"
    mean = point["A"] - point[better]
    lo, hi = (float(q) for q in np.quantile(delta, [0.025, 0.975]))
    return {"mean": mean, "lo95": lo, "hi95": hi, "median": float(np.median(delta)), "better_baseline": better,
            "means": point, "selected_share": {"BARROW_L": float(pick_barrow.mean()),
                                               "AO_L": float(1 - pick_barrow.mean())},
            "per_seed": {k: [x["primary"] for x in xs] for k, xs in arms.items()}, "draws": draws, "seed": seed,
            "n_clusters": len(bg), "n_pairs": n_p, "label": main_label(mean, lo, hi)}


def check_reselect(delta: np.ndarray, pick_barrow: np.ndarray, arms: dict, bg: np.ndarray, draws: int,
                   seed: int) -> None:
    """Internal check: in each draw the delta equals hierarchical_ci's draw of A minus the baseline selected there."""
    a = [x["arrays"] for x in arms["A"]]
    for k, sel in (("BARROW_L", pick_barrow), ("AO_L", ~pick_barrow)):
        d = SA.ci_draws([x["arrays"] for x in arms[k]], a, bg, draws, seed)
        if sel.any() and float(np.max(np.abs(delta[sel] - d[sel]))) > 1e-12:
            raise SystemExit(f"re-selected draws disagree with hierarchical_ci's A - {k} draws (internal check)")


# ---------------------------------------------------------------- the analysis
def names() -> dict:
    return {"A": [M.a_name(s) for s in M.SEEDS], "AO_L": [M.run_name("AO_L", s) for s in M.SEEDS],
            "BARROW_L": [M.run_name("BARROW_L", s) for s in M.SEEDS]}


def seed_sd(xs: list[dict]) -> float:
    return float(np.std([x["primary"] for x in xs], ddof=1))


def analysis(sc: dict, objs: dict, ctx: dict, draws: int, seed: int) -> dict:
    nm = names()
    A, AO, BR = ([sc[n] for n in nm[k]] for k in ("A", "AO_L", "BARROW_L"))
    zs = {k: sc[k] for k in M.ZERO_SHOT}
    main = reselect(A, AO, BR, ctx, draws, seed)
    il = [SA.score_model(SA.fused(objs[b], objs[a], ctx, tag="B"), ctx) for b, a in zip(nm["BARROW_L"], nm["A"])]
    best_zs = max(zs["ADAF_AO"]["primary"], zs["ADAF_barrow"]["primary"])
    return {
        "main_A_minus_better_adapted_baseline": main,
        "A_minus_each_baseline": {k: SA.contrast(A, xs, ctx, draws, seed) for k, xs in (("AO_L", AO), ("BARROW_L", BR))},
        "baseline_minus_zero_shot_start": {
            "AO_L_minus_ADAF_AO": SA.contrast(AO, [zs["ADAF_AO"]] * len(AO), ctx, draws, seed),
            "BARROW_L_minus_ADAF_barrow": SA.contrast(BR, [zs["ADAF_barrow"]] * len(BR), ctx, draws, seed)},
        "INTERLEAVE_BARROW_L_A": {"per_seed": [x["primary"] for x in il], "mean": SA.mean_primary(il),
                                  "minus_A": SA.contrast(il, A, ctx, draws, seed),
                                  "minus_BARROW_L": SA.contrast(il, BR, ctx, draws, seed)},
        "seed_sd": {"A": seed_sd(A), "AO_L": seed_sd(AO), "BARROW_L": seed_sd(BR)},
        "zero_shot": {k: x["primary"] for k, x in zs.items()},
        "gaps": {"zero_shot_gap_best_ADAF_minus_S_STAR": best_zs - zs[M.S_STAR]["primary"],
                 "adapted_gap_best_baseline_minus_A": -main["mean"],
                 "adapted_gap_smaller": -main["mean"] < best_zs - zs[M.S_STAR]["primary"],
                 "note": "descriptive (hypothesis, no direction assumed); S* = FRESH-1's J_2026092803, A = 3-seed mean"},
        "systems": {n: SA.public(x) for n, x in sc.items()}}


def forecasts(res: dict) -> dict:
    zs_barrow = res["zero_shot"]["ADAF_barrow"]
    br_mean = res["main_A_minus_better_adapted_baseline"]["means"]["BARROW_L"]
    label = res["main_A_minus_better_adapted_baseline"]["label"]
    occurred = {"BARROW_L_MEAN_GT_ADAF_BARROW_ZERO_SHOT": br_mean > zs_barrow, "A_BEHIND": label == "A_BEHIND",
                "A_AHEAD": label == "A_AHEAD"}
    out = {k: {"p": p, "occurred": occurred[k], "brier": round((p - float(occurred[k])) ** 2, 4)}
           for k, p in M.FORECASTS.items()}
    out["BARROW_L_MEAN_GT_ADAF_BARROW_ZERO_SHOT"]["threshold"] = zs_barrow
    out["BARROW_L_MEAN_GT_ADAF_BARROW_ZERO_SHOT"]["packet_rounded_threshold_0.556_exceeded"] = br_mean > 0.556
    return out


# ---------------------------------------------------------------- binding (freeze, sessions)
def load_freeze(path: Path) -> dict:
    lock = N.read_json(Path(path).with_name("FREEZE_LOCK.json"))
    sha = N.sha256_file(path)
    if sha != lock["freeze_sha256"]:
        refuse(["FREEZE.json differs from FREEZE_LOCK.json"])
    f = N.read_json(path)
    f["_sha256"] = sha
    return f


SCORER_PINS = ("score_match", "score_adapt", "match_common", "region_score", "hier_uncertainty", "fuse_eval",
               "dev_group_manifest", "dev_individual_manifest", "dev_infer_manifest", "fresh1_result")


def static_violations(f: dict) -> list[str]:
    return [f"{k} differs from the freeze" for k in SCORER_PINS
            if N.sha256_file(M.ROOT / f["pins"][k]["path"]) != f["pins"][k]["sha256"]]


def session_record(top: Path) -> dict:
    p = Path(top) / "SESSION_VERIFIED.json"
    return N.read_json(p) if p.is_file() else {}


def check_sessions(f: dict, tops: list[Path]) -> list[str]:
    v = []
    for i, top in enumerate(tops, 1):
        sv = session_record(top)
        if sv.get("status") != "PASS" or sv.get("freeze_sha256") != f["_sha256"] or sv.get("index") != i or \
                sv.get("arm") != M.session_arm(i):
            v.append(f"{Path(top).name}: SESSION_VERIFIED is not PASS for session {i} under this freeze")
    return v


def gather(f: dict, tops: list[Path], ctx: dict, v: list) -> dict:
    objs = {}
    for top in tops:
        sv = session_record(top)
        for name, ckpt in sv.get("trained", {}).items():
            key = f"dev/PROPOSALS_{name}_PRIVATE.json"
            objs[name] = load_objects(Path(top) / "v1" / "results" / key, name, sv.get("proposals", {}).get(key), ckpt,
                                      ctx, v)
    for name, r in f["a_references"].items():
        objs[name] = load_objects(M.ROOT / r["path"], name, r["sha256"], r["checkpoint_sha256"], ctx, v)
        if N.sha256_file(M.ROOT / r["session_verified"]) != r["session_verified_sha256"]:
            v.append(f"{name}: ADAPT-NDS-1 SESSION_VERIFIED.json changed after the freeze")
    for name, r in f["zero_shot"].items():
        objs[name] = load_objects(M.ROOT / r["path"], name, r["sha256"], None, ctx, v)
    return objs


def score(freeze: Path, tops: list[Path], draws: int = M.BOOT_DRAWS, seed: int = M.BOOT_SEED,
          out: Path | None = None) -> dict:
    f = load_freeze(freeze)
    v = static_violations(f) + check_sessions(f, tops)
    if len(tops) != len(M.SESSIONS):
        v.append(f"{len(tops)} session TOPs given; the design has {len(M.SESSIONS)}")
    ctx = dev_context(f)
    objs = gather(f, tops, ctx, v)
    nm = names()
    need = nm["A"] + nm["AO_L"] + nm["BARROW_L"] + list(M.ZERO_SHOT)
    v += [f"DEV proposals of {n} missing" for n in need if not objs.get(n)]
    if v:
        refuse(v)
    sc = {n: SA.score_model(objs[n], ctx) for n in need}
    res = {"schema": SCHEMA, "utc": N.utc(), "freeze_sha256": f["_sha256"], "region": "Verden (DEV; development "
           "evidence: previously evaluated by FRESH-1)", "unit": "barrow group",
           "context": {"clusters": ctx["n_clusters"], "groups": ctx["eligible_parts"],
                       "background_km2": ctx["background_km2"]},
           "rules": {"main": "A_AHEAD if mean >= +0.03 and lo95 > 0; A_BEHIND if hi95 < 0; else NOT_DISTINGUISHED",
                     "bootstrap": f"cluster x seed, {draws} draws, seed {seed}; better baseline re-selected per draw",
                     "tolerance": M.EPS}}
    res |= analysis(sc, objs, ctx, draws, seed)
    res["outcome"] = res["main_A_minus_better_adapted_baseline"]["label"]
    res["forecasts"] = forecasts(res)
    res["cu_spent"] = SA.cu_spent(M.CU_SPEND)
    res["references"] = {"a": f["a_references"], "zero_shot": f["zero_shot"]}
    res["attribution"] = list(M.ATTRIBUTION.values())
    clean = SA.strip_ci(res)
    N.write_json(out or M.RESULT, clean)
    return clean


def parse_tops(specs: list[str]) -> dict[int, Path]:
    out = {}
    for s in specs or []:
        i, _, p = s.partition("=")
        out[int(i)] = Path(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("inputs", "zero-shot", "score"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--a-top", action="append", default=[])
    ap.add_argument("--draws", type=int, default=M.BOOT_DRAWS)
    ap.add_argument("--seed", type=int, default=M.BOOT_SEED)
    a = ap.parse_args()
    if a.cmd == "inputs":
        r = check_inputs(parse_tops(a.a_top))
        print(json.dumps({"status": r["status"], "a": sorted(r["a_references"]), "zero_shot": sorted(r["zero_shot"])}))
    elif a.cmd == "zero-shot":
        r = zero_shot(Path(a.args[0]) if a.args else None)
        print(json.dumps({k: r[k] for k in ("status", "n_clusters", "models")}, indent=1))
        if r["status"] != "PASS":
            raise SystemExit(1)
    else:
        r = score(Path(a.args[0]), [Path(x) for x in a.args[1:]], a.draws, a.seed)
        m = r["main_A_minus_better_adapted_baseline"]
        print(json.dumps({"outcome": r["outcome"], **{k: m[k] for k in ("mean", "lo95", "hi95", "better_baseline")}},
                         indent=1))


if __name__ == "__main__":
    main()
