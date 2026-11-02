"""ENG-ADAPT-1 trainer (SPEC.md section 1): arm EB = public ADAF barrow + recipe F1 on English barrow-only windows.

Identical to D1-COLLAPSE-1's confirmation runs and STAGE2-D1-CONTROLS' trainer (imported read-only): the start (ADAF
barrow tar, size and sha256 checked: match_train.fresh_model), frozen BatchNorm train mode, fresh AdamW (lr 1e-4,
wd 1e-4), recipe F1 (d1_train.update; lr d1_train.lr_at("F1", t) = cosine 1e-4 -> 1e-6 over 5,116 updates), batch 1,
fp32, TF32 off, torch.manual_seed(seed), the online collapse detector (records only), dense per-update curves, the
loader (adapt_train AdaptSteps + loader: void policy V, 512 px crops, +-160 px offsets, dihedral), 5,116 updates in
epochs 4 and 5, positive at even and background at odd steps, final weights only, resumable per epoch.
The one substantive difference: the windows. D1's own draw generator (match_schedule.local_plan) runs over the English
pool (852 barrow-only R2 FIT positives, SPEC section 1) and the 741 R2 FIT negatives; every step is an English step.
No Lower Saxony, Welsh or Irish file is read.

  python -B ea_train.py pin                     # CPU: POOL_PRIVATE.json + EB_PLAN_<seed>_PRIVATE.json, PLAN_PINS.json
  python -B ea_train.py plan-check SEED OUT     # CPU: composition of the seed's plan (masks only)
  python -B ea_train.py composition             # CPU: COMPOSITION.json over all ten seeds
  python -B ea_train.py step0 OUT.json          # environment gate (as D1 / Stage 2)
  python -B ea_train.py train SEED --run-root R
  python -B ea_train.py smoke N OUT.json [--device cpu] [--run-root R]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea_common as EA  # noqa: E402

N, MS, SCH = EA.N, EA.MS, EA.SCH
import d1_train as T1  # noqa: E402  D1-COLLAPSE-1 (frozen): update (F1), Detector, curves helpers
import match_train as MT  # noqa: E402  ADAPT-MATCH-0 (frozen): fresh_model (ADAF barrow tar)
from nds_train import JT1, at, plans, pt, rt, wt  # noqa: E402  frozen recipe modules as ADAPT-NDS-1 imports them
import nds_train as NT  # noqa: E402  check_constants (as D1 / Stage 2 call it before training)


# ---------------------------------------------------------------- pool and plans (CPU)
def pool_and_bg() -> tuple[list[dict], list[int], list[int], int]:
    windows, fit_pos, fit_neg, _ = plans.load_windows()
    eligible, chosen = EA.select_pool(windows, fit_pos)
    return windows, chosen, list(fit_neg), len(eligible)


def plan_lists(seed: int, pool: list[int], bg: list[int]) -> dict[int, list[tuple[int, int, int, int]]]:
    """{epoch: [(window index, dy, dx, k) for step 0..2,557]} from match_schedule.local_plan (D1's generator)."""
    out = {}
    for e in EA.EPOCHS:
        m = MS.local_plan(seed, e, pool, bg)
        out[e] = [tuple(int(v) for v in m[s]) for s in range(EA.STEPS_PER_EPOCH)]
    return out


def pin() -> dict:
    windows, pool, bg, n_eligible = pool_and_bg()
    if len(pool) != EA.POOL_N:
        print(json.dumps({"warning": "pool shortfall", "eligible": n_eligible, "chosen": len(pool)}))
    pool_rec = {"schema": "eng_adapt1_pool_v1", "rule": "SPEC section 1 (barrow-only live parts; sha256 tag order)",
                "tag": EA.POOL_TAG, "eligible": n_eligible, "chosen": len(pool), "background": len(bg),
                "manifest_sha256": plans.MANIFEST_SHA256,
                "pool_ids": [windows[i]["id"] for i in pool], "bg_ids": [windows[i]["id"] for i in bg]}
    N.write_same(EA.POOL_FILE, pool_rec, indent=None)
    pins = {"schema": "eng_adapt1_plan_pins_v1", "utc": N.utc(), "eligible": n_eligible, "pool": len(pool),
            "background": len(bg), "pool_file_sha256": N.sha256_file(EA.POOL_FILE),
            "pool_ids_digest": N.digest(pool_rec["pool_ids"]), "bg_ids_digest": N.digest(pool_rec["bg_ids"]),
            "manifest_sha256": plans.MANIFEST_SHA256, "plans": {}}
    for seed in EA.SEEDS:
        lists = plan_lists(seed, pool, bg)
        rec = {"schema": "eng_adapt1_plan_v1", "seed": seed, "epochs": {str(e): v for e, v in lists.items()}}
        N.write_same(EA.plan_path(seed), rec, indent=None)
        pins["plans"][str(seed)] = {"sha256": N.sha256_file(EA.plan_path(seed)),
                                    "steps_digest": N.digest(rec["epochs"])}
    N.write_json(EA.PLAN_PINS, pins)
    return pins


def load_pool(windows: list[dict]) -> tuple[list[int], list[int]]:
    """The pinned pool, re-derived by the SPEC rule and compared (fails closed)."""
    rec = N.read_json(EA.POOL_FILE)
    _, fit_pos, fit_neg, _ = plans.load_windows()
    _, chosen = EA.select_pool(windows, fit_pos)
    ids = {w["id"]: i for i, w in enumerate(windows)}
    pool, bg = [ids[x] for x in rec["pool_ids"]], [ids[x] for x in rec["bg_ids"]]
    if pool != chosen or bg != list(fit_neg):
        raise SystemExit("POOL_PRIVATE.json differs from the SPEC rule")
    return pool, bg


def load_plan(seed: int, pool: list[int], bg: list[int]) -> dict[int, list[tuple[int, int, int, int]]]:
    lists = plan_lists(seed, pool, bg)
    rec = N.read_json(EA.plan_path(seed))
    if any([tuple(x) for x in rec["epochs"][str(e)]] != lists[e] for e in EA.EPOCHS):
        raise SystemExit(f"pinned plan of {seed} differs from the generator")
    return lists


def steps_of(epoch: int, real: list[tuple[int, int, int, int]], windows: list[dict]) -> list[dict]:
    return [{"step": s, "t": SCH.t_index(epoch, s), "role": EA.ROLES[s % 2], "kind": "english",
             "uid": windows[wi]["id"], "dy": dy, "dx": dx, "k": k} for s, (wi, dy, dx, k) in enumerate(real)]


# ---------------------------------------------------------------- composition (CPU, masks only; descriptive)
def plan_check(seed: int, out: Path | None = None, cache: dict | None = None) -> dict:
    windows, _, _, _ = plans.load_windows()
    pool, bg = load_pool(windows)
    lists = load_plan(seed, pool, bg)
    cache = {} if cache is None else cache
    draws = {"positive": Counter(), "background": Counter()}
    fg, empty_pos, bg_fg_crops, bad = [], 0, 0, []
    pset, bset = set(pool), set(bg)
    for e in EA.EPOCHS:
        for st in steps_of(e, lists[e], windows):
            wi = lists[e][st["step"]][0]
            role = st["role"]
            if wi not in (pset if role == "positive" else bset):
                bad.append(f"step {e}/{st['step']}: {role} step on a window outside its pool")
            draws[role][wi] += 1
            lab = cache.get(wi)
            if lab is None:
                lab = cache[wi] = np.load(EA.ROOT / windows[wi]["mask_path"])["label"]
            c = rt.dihedral(rt.crop(lab, st["dy"], st["dx"]), st["k"])
            valid = c != 255
            pos = int(((c == 1) & valid).sum())
            if role == "positive":
                nv = int(valid.sum())
                fg.append(pos / nv if nv else 0.0)
                empty_pos += pos == 0
            elif pos:
                bg_fg_crops += 1
    q = np.quantile(fg, [0, .05, .25, .5, .75, .95, 1]).tolist() if fg else []
    rec = {"schema": "eng_adapt1_plan_check_v1", "seed": seed,
           "positive_draws_per_window": dict(Counter(draws["positive"].values())),
           "background_draws_per_window": dict(Counter(draws["background"].values())),
           "distinct_positive": len(draws["positive"]), "distinct_background": len(draws["background"]),
           "positive_steps": sum(draws["positive"].values()), "background_steps": sum(draws["background"].values()),
           "positive_foreground_mean": float(np.mean(fg)) if fg else None, "positive_foreground_quantiles": q,
           "empty_positive_crops": int(empty_pos), "background_crops_with_foreground": bg_fg_crops,
           "status": "FAIL" if bad else "PASS", "violations": bad, "utc": N.utc()}
    if out is not None:
        N.write_json(out, rec)
    if bad:
        raise SystemExit(f"plan-check failed: {bad}")
    return rec


def composition() -> dict:
    cache: dict = {}
    per_seed = {str(s): plan_check(s, cache=cache) for s in EA.SEEDS}
    s2 = N.read_json(EA.ROOT / "work/stage2_d1_controls_2026-10-07/COMPOSITION.json")
    rec = {"schema": "eng_adapt1_composition_v1", "utc": N.utc(), "eb_per_seed": per_seed,
           "d1_reference": {"source": "work/stage2_d1_controls_2026-10-07/COMPOSITION.json",
                            "sha256": N.sha256_file(EA.ROOT / "work/stage2_d1_controls_2026-10-07/COMPOSITION.json"),
                            "keys": sorted(s2.keys()) if isinstance(s2, dict) else None,
                            "quoted_per_seed": {str(x): {k: s2["per_seed"]["d1"][str(x)][k]
                                                         for k in ("a_draws_per_window", "cd_crops")}
                                                for x in EA.SEEDS if str(x) in s2.get("per_seed", {}).get("d1", {})}},
           "note": "mask level (void policy V needs terrain); descriptive only, nothing here changes the plan"}
    N.write_json(EA.COMPOSITION, rec)
    return rec


# ---------------------------------------------------------------- training (VM, GPU)
def context(seed: int) -> dict:
    NT.check_constants()
    windows, _, _, _ = plans.load_windows()
    rt.verify_masks(windows)
    pool, bg = load_pool(windows)
    return {"windows": windows, "pool": pool, "bg": bg, "real": load_plan(seed, pool, bg)}


def epoch_pass(seed: int, epoch: int, c: dict, name: str, started: float, device: str = "cuda",
               limit: int | None = None) -> tuple[dict, list[dict]]:
    """d1_train.epoch_pass with the English plan (every step English; no regional map)."""
    t0 = time.monotonic()
    real = c["real"][epoch]
    steps = steps_of(epoch, real, c["windows"])
    n = EA.STEPS_PER_EPOCH if limit is None else limit
    sums, recs, det = T1.new_sums(), [], c["detector"]
    for s, x, y, _ in at.loader(at.AdaptSteps(c["windows"], real[:n], [], {})):
        st = steps[int(s)]
        role, t = st["role"], st["t"]
        r = T1.update(c["model"], c["opt"], EA.RECIPE, t, x, y, device)
        rec = {"t": t, "epoch": epoch, "step": int(s), "role": role, "kind": "english", **r}
        was_on = det.on
        det.push(t, role, rec)
        rec["detector_on"] = det.on
        recs.append(rec)
        if r["skipped"]:
            sums["skipped"] += 1
        else:
            for m in ("loss_old", "loss_used", "dice", "sum_p"):
                sums[f"{role}_{m}"] += r[m]
            sums[f"n_{role}"] += 1
        if int(s) % at.LOG_EVERY == 0 or det.on != was_on:
            print(json.dumps({"run": name, "epoch": epoch, "step": int(s), "t": t, "role": role, "lr": r["lr"],
                              **({k: r[k] for k in ("loss_old", "dice", "sum_p", "z_max")} if not r["skipped"]
                                 else {}), "detector_on": det.on,
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    row = {"epoch": epoch, "recipe": EA.RECIPE, "lr_end": c["opt"].param_groups[0]["lr"],
           "skipped_all_ignore": sums["skipped"], "epoch_s": time.monotonic() - t0,
           "updates": {k: sum(1 for r in steps[:n] if r["role"] == k) for k in EA.ROLES},
           "english_updates": n, "lower_saxony_updates": 0,
           **{f"mean_{k}_loss": (sums[f"{k}_loss_old"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None) for k in EA.ROLES},
           **{f"mean_{k}_{m}": (sums[f"{k}_{m}"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None)
              for k in EA.ROLES for m in ("loss_used", "dice", "sum_p")},
           "detector_on_at_epoch_end": det.on,
           "steps_digest": N.digest([[r["t"], r["uid"], r["dy"], r["dx"], r["k"]] for r in steps])}
    return row, recs


def optimizer_counter(opt) -> dict:
    steps = [int(float(st["step"])) for st in opt.state.values() if "step" in st]
    return {"min": min(steps) if steps else 0, "max": max(steps) if steps else 0, "params_with_state": len(steps),
            "params_total": sum(len(g["params"]) for g in opt.param_groups)}


def train(seed: int, run_root: Path) -> dict:
    import torch
    name = EA.run_name(seed)
    c = context(seed)
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    last, log_path, curves = run / "last.pt", run / "epochs.jsonl", run / "curves.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return N.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    c["model"], c["opt"] = MT.fresh_model(EA.ARM)
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
    for epoch in EA.EPOCHS:
        if epoch < done:
            continue
        row, recs = epoch_pass(seed, epoch, c, name, started)
        T1.append_lines(curves, recs)
        if epoch == EA.EPOCHS[-1]:
            final = run / "final.pt"
            pt.save({"state_dict": c["model"].state_dict(), "epoch": epoch, "seed": seed, "arm": EA.ARM,
                     "recipe": EA.RECIPE, "experiment": "ENG-ADAPT-1", "data": "english_barrow_only",
                     "start": EA.START["sha256"]}, final)
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
    final, curves = run / "final.pt", run / "curves.jsonl"
    rows_log = [json.loads(x) for x in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    det = T1.detector_from_curves(curves)
    if det != c["detector"].record():
        raise SystemExit("detector: online record differs from the curves file")
    return {"run": run.name, "experiment": "ENG-ADAPT-1", "recipe": EA.RECIPE, "arm": EA.ARM, "seed": seed,
            "paired_d1_run": EA.d1_name(seed), "status": "COMPLETE", "final": str(final),
            "final_sha256": rt.sha(final), "final_bytes": final.stat().st_size, "epochs": rows_log, "detector": det,
            "curves_sha256": N.sha256_file(curves),
            "start": {"registry_id": EA.START["registry_id"], "path": EA.START["path"], "sha256": EA.START["sha256"]},
            "updates": EA.N_UPDATES, "lower_saxony_updates": 0, "welsh_updates": 0, "irish_updates": 0,
            "local_labels_read": False, "optimizer_step_counter": optimizer_counter(c["opt"]),
            "skipped_all_ignore": sum(r["skipped_all_ignore"] for r in rows_log),
            "steps": "match_schedule.local_plan over the English barrow-only pool (852) and R2 FIT negatives (741)",
            "lr": "nds_schedule.lr_at (cosine 1e-4 -> 1e-6 over 5,116 updates, no warm-up) = d1_train.lr_at('F1')",
            "loss": "F1: BCE + 1 - Dice; BCE only on empty targets (d1_train.update)",
            "tables_sha256": {"english_manifest": plans.MANIFEST_SHA256, "pool": N.sha256_file(EA.POOL_FILE),
                              "plan": N.sha256_file(EA.plan_path(seed))},
            "script_sha256": N.sha256_file(Path(__file__)), "common_sha256": N.sha256_file(Path(EA.__file__)),
            "d1_train_sha256": N.sha256_file(Path(T1.__file__)), "match_schedule_sha256": N.sha256_file(Path(MS.__file__)),
            "nds_schedule_sha256": N.sha256_file(Path(SCH.__file__)),
            "frozen_trainer_sha256": N.sha256_file(Path(at.__file__)), "wall_s": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "attribution": EA.ATTRIBUTION, **(wt.env_record() if torch.cuda.is_available() else {})}


def smoke(n: int, out: Path, device: str = "cuda", run_root: Path | None = None) -> dict:
    """First n updates of epoch 4 at SMOKE_SEED (production code path; never a scored checkpoint)."""
    import torch
    seed = EA.SMOKE_SEED
    c = context(seed)
    torch.manual_seed(seed)
    c["model"], c["opt"] = MT.fresh_model(EA.ARM, device)
    c["detector"] = T1.Detector()
    t0 = time.monotonic()
    row, recs = epoch_pass(seed, EA.EPOCHS[0], c, f"SMOKE_{EA.run_name(seed)}", t0, device=device, limit=n)
    losses = [None if r["skipped"] else r["loss_used"] for r in recs]
    rec = {"seed": seed, "n": n, "device": device, "row": row, "losses": losses,
           "s_per_update": (time.monotonic() - t0) / max(n, 1), "start_sha256": EA.START["sha256"],
           "all_finite": all(v is None or math.isfinite(v) for v in losses)}
    if run_root is not None:
        run = Path(run_root) / f"SMOKE_{EA.run_name(seed)}"
        run.mkdir(parents=True, exist_ok=True)
        pt.save({"state_dict": c["model"].state_dict(), "epoch": EA.EPOCHS[0], "seed": seed, "arm": EA.ARM,
                 "recipe": EA.RECIPE, "smoke_updates": n}, run / "final.pt")
        rec["final_sha256"], rec["final_bytes"] = rt.sha(run / "final.pt"), (run / "final.pt").stat().st_size
    N.write_json(out, rec)
    if not rec["all_finite"]:
        raise SystemExit("smoke: non-finite loss")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pin", "plan-check", "composition", "step0", "train", "smoke"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--run-root", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd in ("step0", "train") or (a.cmd == "smoke" and a.device == "cuda"):
        JT1.J.torch_flags()
    if a.cmd == "pin":
        p = pin()
        print(json.dumps({k: p[k] for k in ("eligible", "pool", "background", "pool_file_sha256")}))
    elif a.cmd == "plan-check":
        r = plan_check(int(a.args[0]), Path(a.args[1]))
        print(json.dumps({k: r[k] for k in ("seed", "status", "positive_foreground_mean", "empty_positive_crops")}))
    elif a.cmd == "composition":
        r = composition()
        print(json.dumps({s: {k: v[k] for k in ("positive_draws_per_window", "background_draws_per_window",
                                                  "positive_foreground_mean", "empty_positive_crops")}
                          for s, v in r["eb_per_seed"].items()}))
    elif a.cmd == "step0":
        print(json.dumps(JT1.step0(Path(a.args[0]))))
    elif a.cmd == "train":
        r = train(int(a.args[0]), a.run_root)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "wall_s", "detector")}))
    else:
        r = smoke(int(a.args[0]), Path(a.args[1]), a.device, a.run_root)
        print(json.dumps({k: r[k] for k in ("seed", "n", "device", "s_per_update", "all_finite")}))


if __name__ == "__main__":
    main()
