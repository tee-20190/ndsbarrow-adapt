"""STAGE2-D1-CONTROLS trainer (PACKET.md "Design"): the matched nonlocal-replay control of each D1 BARROW_L run.

Identical to D1-COLLAPSE-1's confirmation runs (imported read-only from d1_train / match_train, frozen f323e17e /
351b5961): the start (ADAF barrow tar, size and sha256 checked: match_train.start_weights / fresh_model), the frozen
BatchNorm train mode, the fresh AdamW (lr 1e-4, wd 1e-4), recipe F1 (d1_train.update: BCE + 1 - soft Dice, BCE only on
empty-target windows; lr d1_train.lr_at("F1", t) = nds_schedule.lr_at = cosine 1e-4 -> 1e-6 over 5,116 updates, no
warm-up), batch 1, fp32, TF32 off, torch.manual_seed(seed), the online collapse detector (d1_train.Detector; records
only), dense per-update curves, the loader (adapt_train AdaptSteps + loader: void policy V, 512 px crops, +-160 px
offsets, dihedral), 5,116 updates in two epochs numbered 4 and 5, positive at even and background at odd steps, final
weights only (no checkpoint selection), resumable per epoch.
The one substantive difference: the steps. Every step is ADAPT-NDS-1 arm C's step at the same seed and update index
(s2_schedule: J's English plan for epochs 4-5, HR1 A_W30 Welsh pairs, HR3 A_I30 Irish pairs; nds_train.schedule("C")).
No Lower Saxony table, window or mask is read; TRAIN_RESULT.json records local_labels_read = false.

  python -B s2_train.py pin-plans                                    # CPU: SUBSETS_S2 + ENGLISH_E45_<seed> (private)
  python -B s2_train.py plan-check SEED OUT.json [--ls-table T]      # CPU: composition, alternation, C identity
  python -B s2_train.py step0 OUT.json                               # HR1/JOINT-1 step-0 environment gate (as D1)
  python -B s2_train.py train SEED --w-win W --w-mask M --i-win W --i-mask M --run-root R
  python -B s2_train.py smoke N OUT.json --w-win W --w-mask M --i-win W --i-mask M --scratch S [--device cpu]
                              [--run-root R]                          (staged windows only; SMOKE_SEED)
  python -B s2_train.py detector CURVES.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2_common as S  # noqa: E402
import s2_schedule as SS  # noqa: E402

N = S.N
import d1_train as T1  # noqa: E402  D1-COLLAPSE-1 (frozen): update (F1), lr_at, Detector, detector_from_curves
import match_train as MT  # noqa: E402  ADAPT-MATCH-0 (frozen): fresh_model / start_weights (ADAF barrow tar)
from s2_schedule import JT1, NT, SCH, plans, rt  # noqa: E402
from nds_train import at, pt, wt  # noqa: E402  frozen recipe modules as ADAPT-NDS-1 imports them


# ---------------------------------------------------------------- pinned plans (CPU, before the freeze)
def pin_plans() -> dict:
    """SUBSETS_S2_PRIVATE.json and the ten English epoch 4-5 plans (write_same: identical bytes on a rerun)."""
    w_t, i_t = N.read_json(J_W()), N.read_json(J_I())
    sub = SS.subsets(w_t, i_t)
    N.write_same(S.SUBSETS_S2, {"schema": "s2_subsets_v1", "rule": "default_rng([seed, 30]).choice(sorted(monuments), "
                                "30, replace=False) (JOINT-1 audit_mixture.subset)", "recorded_seeds_reproduced":
                                list(S.SUBSET_RECORDED_SEEDS), **sub}, indent=None)
    _, fit_pos, fit_neg, _ = plans.load_windows()
    out = {"subsets": {"path": N.rel(S.SUBSETS_S2), "sha256": N.sha256_file(S.SUBSETS_S2)}, "english": {}}
    for seed in S.SEEDS:
        p = S.english_plan_path(seed)
        N.write_same(p, SS.english_record(seed, fit_pos, fit_neg), indent=None)
        out["english"][str(seed)] = {"path": N.rel(p), "sha256": N.sha256_file(p)}
    return out


def J_W() -> Path:
    return SS.J.W_FIT


def J_I() -> Path:
    return SS.J.I_FIT


def load_subsets() -> dict:
    """The pinned extended subsets, re-derived by the rule and compared (fails closed)."""
    rec = N.read_json(S.SUBSETS_S2)
    want = SS.subsets(N.read_json(J_W()), N.read_json(J_I()))
    if {k: rec[k] for k in ("wales", "ireland")} != want:
        raise SystemExit("SUBSETS_S2_PRIVATE.json differs from the JOINT-1 subset rule")
    return want


# ---------------------------------------------------------------- context
def context(seed: int, paths: dict) -> dict:
    NT.check_constants()
    windows, fit_pos, fit_neg, _ = plans.load_windows()
    rt.verify_masks(windows)
    ctx = SS.control_context(paths, load_subsets())
    real = SS.english_real(seed, fit_pos, fit_neg, pinned=S.english_plan_path(seed))
    return {"ctx": ctx, "windows": windows, "real": real}


def epoch_steps(seed: int, epoch: int, c: dict) -> tuple[list, dict, list[dict]]:
    real, wmap = c["real"][epoch], SS.control_map(seed, epoch, c["ctx"])
    steps = SS.control_steps(seed, epoch, real, wmap, c["ctx"]["rr"]["rows"], c["windows"])
    bad = [r["step"] for r in steps if r["role"] != SS.ROLES[r["step"] % 2] or r["kind"] == "lower_saxony"]
    if bad:
        raise SystemExit(f"epoch {epoch}: {len(bad)} steps break the alternation or read Lower Saxony")
    return real, wmap, steps


def epoch_pass(seed: int, epoch: int, c: dict, name: str, started: float, device: str = "cuda",
               limit: int | None = None) -> tuple[dict, list[dict]]:
    """d1_train.epoch_pass with C's step map (English plan + Welsh/Irish map) in place of the all-local plan."""
    t0 = time.monotonic()
    rows = c["ctx"]["rr"]["rows"]
    real, wmap, steps = epoch_steps(seed, epoch, c)
    n = SCH.STEPS_PER_EPOCH if limit is None else limit
    real_n, wmap_n = real[:n], {s: v for s, v in wmap.items() if s < n}
    sums, recs, det = T1.new_sums(), [], c["detector"]
    kinds = {k: 0 for k in SCH.KINDS}
    for s, x, y, _ in at.loader(at.AdaptSteps(c["windows"], real_n, rows, wmap_n)):
        st = steps[int(s)]
        role, t = st["role"], st["t"]
        r = T1.update(c["model"], c["opt"], S.D1_RECIPE, t, x, y, device)
        rec = {"t": t, "epoch": epoch, "step": int(s), "role": role, "kind": st["kind"], **r}
        was_on = det.on
        det.push(t, role, rec)
        rec["detector_on"] = det.on
        recs.append(rec)
        kinds[st["kind"]] += 1
        if r["skipped"]:
            sums["skipped"] += 1
        else:
            for m in ("loss_old", "loss_used", "dice", "sum_p"):
                sums[f"{role}_{m}"] += r[m]
            sums[f"n_{role}"] += 1
        if int(s) % at.LOG_EVERY == 0 or det.on != was_on:
            print(json.dumps({"run": name, "epoch": epoch, "step": int(s), "t": t, "role": role, "kind": st["kind"],
                              "lr": r["lr"], **({k: r[k] for k in ("loss_old", "dice", "sum_p", "z_max")}
                                                if not r["skipped"] else {}),
                              "detector_on": det.on, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    row = {"epoch": epoch, "recipe": S.D1_RECIPE, "lr_end": c["opt"].param_groups[0]["lr"],
           "skipped_all_ignore": sums["skipped"], "epoch_s": time.monotonic() - t0,
           "updates": {k: sum(1 for r in steps[:n] if r["role"] == k) for k in SS.ROLES},
           "kind_updates": kinds, "lower_saxony_updates": 0,
           **{f"mean_{k}_loss": (sums[f"{k}_loss_old"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None) for k in SS.ROLES},
           **{f"mean_{k}_{m}": (sums[f"{k}_{m}"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None)
              for k in SS.ROLES for m in ("loss_used", "dice", "sum_p")},
           "detector_on_at_epoch_end": det.on, "draw_digest": SS.draw_digest(wmap, rows),
           "steps_digest": N.digest([[r["t"], r["kind"], r["uid"], r["dy"], r["dx"], r["k"]] for r in steps])}
    return row, recs


def optimizer_counter(opt) -> dict:
    """AdamW's own per-parameter step counters (records only; read after training, no effect on it). The barrier
    requires min = max = executed updates (5,116 minus skipped all-IGNORE updates)."""
    steps = [int(float(st["step"])) for st in opt.state.values() if "step" in st]
    return {"min": min(steps) if steps else 0, "max": max(steps) if steps else 0, "params_with_state": len(steps),
            "params_total": sum(len(g["params"]) for g in opt.param_groups)}


def plan_summary(seed: int, c: dict) -> dict:
    out = {"seed": seed, "epochs": []}
    for e in S.EPOCHS:
        _, wmap, steps = epoch_steps(seed, e, c)
        out["epochs"].append({"epoch": e, "draw_digest": SS.draw_digest(wmap, c["ctx"]["rr"]["rows"]),
                              "steps_digest": N.digest([[r["t"], r["kind"], r["uid"], r["dy"], r["dx"], r["k"]]
                                                        for r in steps])})
    return out


# ---------------------------------------------------------------- training (VM, GPU)
def train(seed: int, paths: dict, run_root: Path) -> dict:
    import torch
    name = S.run_name(seed)
    c = context(seed, paths)
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    last, log_path, curves = run / "last.pt", run / "epochs.jsonl", run / "curves.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return N.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    c["model"], c["opt"] = MT.fresh_model(S.ARM)
    torch.manual_seed(seed)
    done = wt._resume(c["model"], c["opt"], last)
    if done == 0:
        for p in (log_path, curves):
            p.unlink(missing_ok=True)
    c["detector"] = T1.Detector()
    if done:
        for ln in curves.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            c["detector"].push(r["t"], r["role"], r)
    for epoch in S.EPOCHS:
        if epoch < done:
            continue
        row, recs = epoch_pass(seed, epoch, c, name, started)
        T1.append_lines(curves, recs)
        if epoch == S.EPOCHS[-1]:
            final = run / "final.pt"
            pt.save({"state_dict": c["model"].state_dict(), "epoch": epoch, "seed": seed, "arm": S.ARM,
                     "recipe": S.D1_RECIPE, "control": "nds_c_replay", "start": S.START["sha256"]}, final)
            row["final_sha256"] = rt.sha(final)
        pt.save({"state_dict": c["model"].state_dict(), "optimizer": c["opt"].state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        T1.append_lines(log_path, [row])
        print(json.dumps({"run": name, **row}), flush=True)
    result = result_record(seed, run, c, started)
    N.write_json(run / "TRAIN_RESULT.json", result)
    last.unlink(missing_ok=True)
    return result


def result_record(seed: int, run: Path, c: dict, started: float) -> dict:
    import torch
    name, final, curves = run.name, run / "final.pt", run / "curves.jsonl"
    rows_log = [json.loads(x) for x in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    det = T1.detector_from_curves(curves)
    if det != c["detector"].record():
        raise SystemExit("detector: online record differs from the curves file")
    return {"run": name, "recipe": S.D1_RECIPE, "arm": S.ARM, "seed": seed, "paired_d1_run": S.d1_name(seed),
            "status": "COMPLETE", "final": str(final), "final_sha256": rt.sha(final),
            "final_bytes": final.stat().st_size, "epochs": rows_log, "detector": det,
            "curves_sha256": N.sha256_file(curves), "plan_summary": plan_summary(seed, c),
            "start": {"registry_id": S.START["registry_id"], "path": S.START["path"], "sha256": S.START["sha256"]},
            "updates": S.N_UPDATES, "lower_saxony_updates": 0, "local_labels_read": False,
            "optimizer_step_counter": optimizer_counter(c["opt"]),
            "skipped_all_ignore": sum(r["skipped_all_ignore"] for r in rows_log),
            "steps": "ADAPT-NDS-1 arm C's step map at this seed (English plan epochs 4-5, HR1 A_W30, HR3 A_I30)",
            "lr": "nds_schedule.lr_at (cosine 1e-4 -> 1e-6 over 5,116 updates, no warm-up) = d1_train.lr_at('F1')",
            "loss": "F1: BCE + 1 - Dice; BCE only on empty targets (d1_train.update)",
            "tables_sha256": {"w_table": N.sha256_file(J_W()), "i_table": N.sha256_file(J_I()),
                              "english_manifest": plans.MANIFEST_SHA256, "subsets_s2": N.sha256_file(S.SUBSETS_S2),
                              "english_plan": N.sha256_file(S.english_plan_path(seed))},
            "script_sha256": N.sha256_file(Path(__file__)), "schedule_sha256": N.sha256_file(Path(SS.__file__)),
            "common_sha256": N.sha256_file(Path(S.__file__)), "d1_train_sha256": N.sha256_file(Path(T1.__file__)),
            "nds_train_sha256": N.sha256_file(Path(NT.__file__)),
            "nds_schedule_sha256": N.sha256_file(Path(SCH.__file__)),
            "frozen_trainer_sha256": N.sha256_file(Path(at.__file__)), "wall_s": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "attribution": S.ATTRIBUTION, **(wt.env_record() if torch.cuda.is_available() else {})}


# ---------------------------------------------------------------- plan-check (CPU, no forward)
def plan_check(seed: int, out: Path, ls_table: Path | None = None) -> dict:
    """Composition of the control at SEED (and, with --ls-table, of the paired D1 run); C identity at seeds 02-04."""
    NT.check_constants()
    windows, fit_pos, fit_neg, _ = plans.load_windows()
    sub = SS.subsets(N.read_json(J_W()), N.read_json(J_I()))
    ctx = SS.control_context(SS.replay_paths(), sub)
    pinned = S.english_plan_path(seed) if S.english_plan_path(seed).is_file() else None
    real = SS.english_real(seed, fit_pos, fit_neg, pinned=pinned)
    steps, digests, step_digests = [], {}, {}
    for e in S.EPOCHS:
        wmap = SS.control_map(seed, e, ctx)
        st = SS.control_steps(seed, e, real[e], wmap, ctx["rr"]["rows"], windows)
        steps += st
        digests[str(e)] = SS.draw_digest(wmap, ctx["rr"]["rows"])
        step_digests[str(e)] = N.digest([[r["t"], r["kind"], r["uid"], r["dy"], r["dx"], r["k"]] for r in st])
    comp = SS.composition(steps)
    bad = list(comp["violations"])
    kinds = {k: sum(1 for r in steps if r["kind"] == k) for k in SCH.KINDS}
    if kinds != SCH.kind_counts("C")["updates"]:
        bad.append(f"kind counts {kinds} differ from nds_schedule.kind_counts('C')")
    rec = {"schema": "s2_plan_check_v1", "seed": seed, "control": comp, "kind_updates": kinds,
           "epoch_draw_digests": digests, "epoch_steps_digests": step_digests,
           "english_plan_pinned": pinned is not None,
           "steps_digest": N.digest([[r["t"], r["kind"], r["uid"], r["dy"], r["dx"], r["k"]] for r in steps])}
    if seed in S.NDS_C_SEEDS:
        rec["c_identity"] = c_identity(seed, digests, real, bad)
    if ls_table is not None and seed in S.SEEDS:
        ls = SS.ls_context(ls_table)
        rec["d1"] = SS.composition([r for e in S.EPOCHS for r in SS.d1_steps(seed, e, ls)])
    rec |= {"status": "FAIL" if bad else "PASS", "violations": bad, "utc": N.utc()}
    N.write_json(out, rec)
    if bad:
        raise SystemExit(f"plan-check failed: {bad}")
    return rec


def c_identity(seed: int, digests: dict, real: dict, bad: list) -> dict:
    """The generator reproduces ADAPT-NDS-1 C exactly: per-epoch draw digests of C's TRAIN_RESULT and C's pinned
    English plan (seeds 02-04 only; no control trains at these seeds)."""
    top = S.ROOT / S.NDS_C_RUNS[seed]
    tr = N.read_json(top / f"v1/results/runs/C_NDS_{seed}/TRAIN_RESULT.json")
    want = {str(e["epoch"]): e["draw_digest"] for e in tr["plan_summary"]["epochs"]}
    eng = N.read_json(NT.english_plan_path(seed))["epochs"]
    ok_draw, ok_eng = want == digests, all(eng[str(e)] == real[e] for e in S.EPOCHS)
    if not (ok_draw and ok_eng):
        bad.append(f"seed {seed}: generator differs from ADAPT-NDS-1 C (draws {ok_draw}, English plan {ok_eng})")
    return {"c_train_result": N.rel(top / f"v1/results/runs/C_NDS_{seed}/TRAIN_RESULT.json"),
            "draw_digests_equal": ok_draw, "english_plan_equal": ok_eng}


# ---------------------------------------------------------------- smoke (staged windows; production artifact route)
def smoke(n: int, out: Path, paths: dict, device: str = "cuda", run_root: Path | None = None) -> dict:
    """First n updates of epoch 4 at SMOKE_SEED on the staged Welsh/Irish windows (subsets = staged monuments, as
    nds_train.smoke); with run_root the model is saved as RUN_ROOT/SMOKE_<name>/final.pt for the artifact route."""
    import torch
    seed = S.SMOKE_SEED
    NT.check_constants()
    windows, fit_pos, fit_neg, _ = plans.load_windows()
    w_t, i_t = (JT1.staged(N.read_json(t), Path(paths[w]), Path(paths[m]))
                for t, w, m in ((J_W(), "w_win", "w_mask"), (J_I(), "i_win", "i_mask")))
    rr = JT1.region_rows(w_t, Path(paths["w_win"]), Path(paths["w_mask"]), i_t, Path(paths["i_win"]),
                         Path(paths["i_mask"]))
    rr["ls_pos"], rr["ls_bg"] = [], []
    sub = {"smoke": True, "wales": {str(seed): sorted(w_t["monuments"])}, "ireland": {str(seed): sorted(i_t["monuments"])}}
    c = {"ctx": {"rr": rr, "sub": sub}, "windows": windows,
         "real": SS.english_real(seed, fit_pos, fit_neg, pinned=S.english_plan_path(seed))}
    torch.manual_seed(seed)
    c["model"], c["opt"] = MT.fresh_model(S.ARM, device)
    c["detector"] = T1.Detector()
    t0 = time.monotonic()
    row, recs = epoch_pass(seed, S.EPOCHS[0], c, f"SMOKE_{S.run_name(seed)}", t0, device=device, limit=n)
    losses = [None if r["skipped"] else r["loss_used"] for r in recs]
    rec = {"seed": seed, "n": n, "device": device, "row": row, "losses": losses,
           "kinds": {k: sum(1 for r in recs if r["kind"] == k) for k in SCH.KINDS},
           "s_per_update": (time.monotonic() - t0) / max(n, 1), "start_sha256": S.START["sha256"],
           "all_finite": all(v is None or math.isfinite(v) for v in losses)}
    if run_root is not None:
        run = Path(run_root) / f"SMOKE_{S.run_name(seed)}"
        run.mkdir(parents=True, exist_ok=True)
        pt.save({"state_dict": c["model"].state_dict(), "epoch": S.EPOCHS[0], "seed": seed, "arm": S.ARM,
                 "recipe": S.D1_RECIPE, "smoke_updates": n}, run / "final.pt")
        rec["final_sha256"], rec["final_bytes"] = rt.sha(run / "final.pt"), (run / "final.pt").stat().st_size
        T1.append_lines(run / "curves.jsonl", recs)
        N.write_json(run / "TRAIN_RESULT.json", {"run": run.name, "seed": seed, "tag": "smoke", "status": "SMOKE",
                                                "final": str(run / "final.pt"), "final_sha256": rec["final_sha256"],
                                                "final_bytes": rec["final_bytes"], "updates": n,
                                                "optimizer_step_counter": optimizer_counter(c["opt"]),
                                                "curves_sha256": N.sha256_file(run / "curves.jsonl"),
                                                **(wt.env_record() if torch.cuda.is_available() else {})})
    N.write_json(out, rec)
    if not rec["all_finite"]:
        raise SystemExit("smoke: non-finite loss")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pin-plans", "plan-check", "step0", "train", "smoke", "smoke-windows", "detector"))
    ap.add_argument("args", nargs="*")
    for k in ("w-win", "w-mask", "i-win", "i-mask", "run-root", "scratch", "ls-table"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd in ("step0", "train") or (a.cmd == "smoke" and a.device == "cuda"):
        JT1.J.torch_flags()
    paths = {**SS.replay_paths(), **{k: getattr(a, k) for k in ("w_win", "w_mask", "i_win", "i_mask")
                                     if getattr(a, k, None) is not None}}
    if a.cmd == "pin-plans":
        print(json.dumps(pin_plans(), indent=1))
    elif a.cmd == "plan-check":
        r = plan_check(int(a.args[0]), Path(a.args[1]), a.ls_table)
        print(json.dumps({k: r[k] for k in ("seed", "status", "steps_digest")}))
    elif a.cmd == "step0":
        print(json.dumps(JT1.step0(Path(a.args[0]))))
    elif a.cmd == "train":
        r = train(int(a.args[0]), paths, a.run_root)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "wall_s", "detector")}))
    elif a.cmd == "smoke-windows":                      # driver: SMOKE_W_WINDOWS / SMOKE_I_WINDOWS (window ids only)
        ctx = SS.control_context(SS.replay_paths(), load_subsets())
        w = SS.smoke_windows(S.SMOKE_SEED, int(a.args[0]), ctx)
        print(f"SMOKE_W_WINDOWS='{' '.join(w['wales'])}'\nSMOKE_I_WINDOWS='{' '.join(w['ireland'])}'")
    elif a.cmd == "smoke":
        r = smoke(int(a.args[0]), Path(a.args[1]), paths, a.device, a.run_root)
        print(json.dumps({k: r[k] for k in ("seed", "n", "device", "s_per_update", "all_finite", "kinds")}))
    else:
        print(json.dumps(T1.detector_from_curves(Path(a.args[0]))))


if __name__ == "__main__":
    main()
