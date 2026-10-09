"""ADAPT-NDS-1 trainer: A (J continued with Lower Saxony register barrows as a fourth region) and C (the equal-update
continuation without Lower Saxony data), both from JOINT-1 J's final weights at the seed (PACKET.md "Arms").

Recipe (JOINT-1's, through its frozen modules, imported read-only): adapt_train AdaptSteps + loader (void policy V,
512 px crops, +-160 px offsets, dihedral), r2_train.masked_loss (BCE + 1 - soft Dice over non-IGNORE pixels), batch 1,
fp32, TF32 off, frozen BatchNorm (r2_train.set_train_mode), a fresh AdamW (lr 1e-4, weight decay 1e-4: the optimizer
is reset as J's was), torch.manual_seed(seed), final weights only. Changes, all from the packet:
  start   J_<seed>/final.pt (DETECTOR_REGISTRY sha256; r2_train.load_model, the loader every J inference uses)
  steps   two J epochs, numbered 4 and 5 (nds_schedule.py): the English plan plans.real_plan(seed, 4|5) (the generator
          of J's PLAN files, pinned per seed in ENGLISH_E45_<seed>_PRIVATE.json), HR1 A_W30's Welsh pairs
          (welsh_train.welsh_plan) and HR3 A_I30's Irish draw streams (joint_train.irish_plan) continued; A replaces
          the pairs with (p - e) % 3 == 0 by Lower Saxony (positive, background) pairs (nds_schedule.ls_plan); C keeps
          J's draws there. Every other step is identical in A and C (window, offsets, dihedral, update index, LR).
  lr      J's cosine (1e-4 -> 1e-6, no warm-up) compressed to 5,116 updates (nds_schedule.lr_at); update() is
          adapt_train.update with that schedule and an explicit device (copied: adapt_train.py sha256 below).
Retention in the final epoch as JOINT-1: r2_train.evaluate holdout losses, final.pt, the English TRAIN-holdout objects
(welsh_train.holdout_objects, 804 windows) and their primary.

  python -B nds_train.py plans                      # ENGLISH_E45_<seed>_PRIVATE.json for the three seeds (CPU)
  python -B nds_train.py plan-check SEED OUT.json --w-table T --i-table T --ls-table T     (CPU; no forward)
  python -B nds_train.py step0 OUT.json             # JOINT-1's step-0 gate (C, seed 2026092800, CONT e41)
  python -B nds_train.py train SEED ARM --w-table T --w-win W --w-mask M --i-table T --i-win W --i-mask M
                               --ls-table T --ls-win W --ls-mask M --run-root R
  python -B nds_train.py smoke ARM N OUT.json ... [--device cpu] [--run-root R]   (staged windows only)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nds_common as N  # noqa: E402
import nds_schedule as SCH  # noqa: E402

sys.path.insert(0, str(N.J1_CHAIN))
import joint_train as JT1  # noqa: E402  JOINT-1 trainer (frozen, read-only): region rows, subsets, Irish draws
from joint_train import at, plans, pt, rt, wt  # noqa: E402  frozen recipe modules exactly as JOINT-1 imports them

PLANS_DIR = N.PRIV / "plans"
LS_REGION = "lower_saxony"


def english_plan_path(seed: int) -> Path:
    return PLANS_DIR / f"ENGLISH_E45_{seed}_PRIVATE.json"


def check_constants() -> None:
    """The schedule module's copies of J's constants must equal the frozen modules' values."""
    bad = []
    if plans.STEPS_PER_EPOCH != SCH.STEPS_PER_EPOCH or plans.N_EPOCHS != 4 or SCH.EPOCHS != (4, 5):
        bad.append("epoch structure")
    if (plans.LR_MAX, plans.LR_MIN) != (SCH.LR_MAX, SCH.LR_MIN) or plans.lr_at(0) != SCH.lr_at(0):
        bad.append("learning-rate end points")
    if (JT1.J.WELSH_RESIDUE, JT1.J.IRISH_RESIDUE) != (SCH.WELSH_RESIDUE, SCH.IRISH_RESIDUE) or rt.OFFSET_PX != \
            SCH.OFFSET_PX or JT1.N_PAIRS != SCH.N_PAIRS:
        bad.append("J residues, pair count or crop offsets")
    rng_a, rng_b = np.random.default_rng([1, 2, 3]), np.random.default_rng([1, 2, 3])
    if [rt.draw(rng_a) for _ in range(5)] != [SCH.draw(rng_b) for _ in range(5)]:
        bad.append("draw")
    if bad:
        raise SystemExit(f"nds_schedule constants differ from the frozen recipe: {bad}")


# ---------------------------------------------------------------- English plans for epochs 4-5 (CPU)
def build_english_plans() -> dict:
    windows, fit_pos, fit_neg, _ = plans.load_windows()
    out = {}
    for seed in N.SEEDS:
        rec = {"schema": "adapt_nds_english_e45_v1", "seed": seed, "generator": "plans.real_plan(seed, epoch, "
               "fit_pos, fit_neg) (work/counterfactual_2026-09-28/plans.py, the generator of J's PLAN files)",
               "plans_sha256": N.sha256_file(Path(plans.__file__)), "manifest_sha256": plans.MANIFEST_SHA256,
               "epochs": {str(e): [list(map(int, r)) for r in plans.real_plan(seed, e, fit_pos, fit_neg)]
                          for e in SCH.EPOCHS}}
        path = english_plan_path(seed)
        N.write_same(path, rec, indent=None)
        out[seed] = {"path": N.rel(path), "sha256": N.sha256_file(path)}
    return out


def english_real(seed: int, windows_ctx: tuple | None = None) -> dict[int, list]:
    """The pinned English plan for epochs 4-5, regenerated and compared (fails closed on any difference)."""
    rec = N.read_json(english_plan_path(seed))
    _, fit_pos, fit_neg, _ = windows_ctx or plans.load_windows()
    out = {}
    for e in SCH.EPOCHS:
        regen = [list(map(int, r)) for r in plans.real_plan(seed, e, fit_pos, fit_neg)]
        if regen != rec["epochs"][str(e)] or len(regen) != SCH.STEPS_PER_EPOCH:
            raise SystemExit(f"English plan of seed {seed} epoch {e} differs from its pinned file")
        out[e] = regen
    return out


# ---------------------------------------------------------------- rows and schedule
def ls_rows(table: dict, win: Path, mask: Path, offset: int) -> tuple[list[dict], list[int], list[int]]:
    rows, mons, bg = wt.welsh_rows(table, win, mask)
    for r in rows:
        r["region"], r["uid"] = LS_REGION, f"{LS_REGION}:{r['id']}"
        N.check_path(r["tif"])
    pos = [mons[w["sam"]][0] + offset for w in table["positive_windows"]]
    return rows, pos, [i + offset for i in bg]


def region_context(arm: str, paths: dict) -> dict:
    """J's Welsh + Irish rows (JOINT-1 region_rows, identical indices in A and C); A appends the Lower Saxony rows."""
    w_table, i_table = N.read_json(paths["w_table"]), N.read_json(paths["i_table"])
    rr = JT1.region_rows(w_table, Path(paths["w_win"]), Path(paths["w_mask"]), i_table, Path(paths["i_win"]),
                         Path(paths["i_mask"]))
    rr["ls_pos"], rr["ls_bg"] = [], []
    if arm == "A":
        ls_table = N.read_json(paths["ls_table"])
        if ls_table.get("role") != "FIT" or "Lower Saxony FIT" not in ls_table.get("region", ""):
            raise N.RefusedTerrain("the Lower Saxony table is not the ADAPT-NDS-1 FIT table")
        rows, pos, bg = ls_rows(ls_table, Path(paths["ls_win"]), Path(paths["ls_mask"]), len(rr["rows"]))
        rr["rows"] = rr["rows"] + rows
        rr["ls_pos"], rr["ls_bg"] = pos, bg
    if len({r["uid"] for r in rr["rows"]}) != len(rr["rows"]):
        raise SystemExit("duplicate regional window")
    return {"rr": rr, "sub": JT1.load_subsets(w_table, i_table, Path(paths.get("subsets") or JT1.J.SUBSETS))}


def j_map(seed: int, epoch: int, rr: dict, sub: dict) -> dict:
    """J's Welsh (HR1 A_W30) and Irish (HR3 A_I30) draws for a continuation epoch (joint_train.epoch_inputs' maps)."""
    wpool, ipool = list(sub["wales"][str(seed)]), list(sub["ireland"][str(seed)])
    if not sub.get("smoke") and (len(wpool) != JT1.J.K or len(ipool) != JT1.J.K):
        raise RuntimeError("subset size is not 30")
    wmap = wt.welsh_plan("A_W30", seed, epoch, rr["w_mon"], wpool, rr["w_bg"])
    imap = JT1.irish_plan(seed, epoch, rr["i_mon"], ipool, rr["i_bg"])
    if set(wmap) & set(imap):
        raise RuntimeError("Welsh and Irish steps collide")
    return {**wmap, **imap}


def schedule(arm: str, seed: int, epoch: int, rr: dict, sub: dict) -> dict:
    ls_map = SCH.ls_plan(seed, epoch, rr["ls_pos"], rr["ls_bg"]) if arm == "A" else {}
    return SCH.compose(arm, epoch, j_map(seed, epoch, rr, sub), ls_map)


def kinds_of(wmap: dict, rows: list[dict]) -> dict[int, str]:
    names = {"wales": "welsh", "ireland": "irish", LS_REGION: LS_REGION}
    out = {s: "english" for s in range(SCH.STEPS_PER_EPOCH)}
    for s, v in wmap.items():
        out[s] = names[rows[v[0]]["region"]]
    return out


def shared_digest(arm: str, epoch: int, real: list, wmap: dict, rows: list[dict]) -> str:
    """Digest of every step outside the Lower Saxony pairs (source, window, offsets, dihedral, update index): equal in
    A and C by construction; recorded per epoch so that the two runs' records can be compared."""
    src = SCH.step_sources(arm, epoch, real, wmap, rows)
    ls = SCH.ls_steps(epoch)
    return N.digest([[s, SCH.t_index(epoch, s), *src[s]] for s in range(SCH.STEPS_PER_EPOCH) if s not in ls])


def plan_summary(arm: str, seed: int, ctx: dict, real: dict) -> dict:
    rr = ctx["rr"]
    out = {"arm": arm, "seed": seed, "epochs": [], "kind_counts": SCH.kind_counts(arm)}
    for e in SCH.EPOCHS:
        wmap = schedule(arm, seed, e, rr, ctx["sub"])
        kinds = kinds_of(wmap, rr["rows"])
        cnt = {k: sum(1 for v in kinds.values() if v == k) for k in SCH.KINDS}
        out["epochs"].append({"epoch": e, "updates": cnt, "shared_digest": shared_digest(arm, e, real[e], wmap,
                                                                                         rr["rows"]),
                              "draw_digest": N.digest({str(s): [rr["rows"][v[0]]["uid"], *v[1:]]
                                                       for s, v in sorted(wmap.items())})})
    return out


# ---------------------------------------------------------------- model and update
def start_weights(seed: int) -> Path:
    rel, sha, size = N.J_START[seed]
    p = N.ROOT / rel
    if not p.is_file() or p.stat().st_size != size or N.sha256_file(p) != sha:
        raise SystemExit(f"start weights of seed {seed} missing or differ from the registry ({rel})")
    return p


def fresh_model(seed: int, device: str = "cuda"):
    """adapt_train.fresh_model with J_<seed>/final.pt as the start: same loader, train mode with frozen BatchNorm, a
    fresh AdamW (lr plans.LR_MAX, weight decay r2_train.WD)."""
    import torch
    model = rt.load_model(start_weights(seed), device=device)
    rt.set_train_mode(model)
    return model, torch.optim.AdamW(model.parameters(), lr=plans.LR_MAX, weight_decay=rt.WD)


def update(model, opt, t: int, x, y, device: str = "cuda") -> dict:
    """adapt_train.update (cf_v1.matched_update's real-crop path) with the compressed schedule nds_schedule.lr_at."""
    lr = SCH.lr_at(t)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    z = model(x[None].to(device, non_blocking=True))["out"][0, 1]
    res = rt.masked_loss(z, y.to(device, non_blocking=True))
    if res is None:
        return {"skipped": True, "lr": lr}
    res[0].backward()
    opt.step()
    return {"skipped": False, "loss": float(res[1] + (1 - res[2])), "lr": lr}


def new_sums() -> dict:
    return {**{f"{k}_loss": 0.0 for k in SCH.KINDS}, **{f"n_{k}": 0 for k in SCH.KINDS}, "skipped": 0}


def epoch_pass(arm: str, seed: int, epoch: int, c: dict, name: str, started: float, device: str = "cuda",
               limit: int | None = None) -> dict:
    t0 = time.monotonic()
    rows = c["ctx"]["rr"]["rows"]
    real, wmap = c["real"][epoch], schedule(arm, seed, epoch, c["ctx"]["rr"], c["ctx"]["sub"])
    if limit is not None:
        real, wmap = real[:limit], {s: v for s, v in wmap.items() if s < limit}
    kinds = kinds_of(wmap, rows)
    sums, losses = new_sums(), []
    for s, x, y, _ in at.loader(at.AdaptSteps(c["windows"], real, rows, wmap)):
        kind = kinds[int(s)]
        r = update(c["model"], c["opt"], SCH.t_index(epoch, int(s)), x, y, device)
        losses.append(None if r["skipped"] else r["loss"])
        if r["skipped"]:
            sums["skipped"] += 1
        else:
            sums[f"{kind}_loss"] += r["loss"]
            sums[f"n_{kind}"] += 1
        if int(s) % at.LOG_EVERY == 0:
            print(json.dumps({"run": name, "epoch": epoch, "step": int(s), "kind": kind, **r,
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    row = {"epoch": epoch, "lr_end": c["opt"].param_groups[0]["lr"], "skipped_all_ignore": sums["skipped"],
           "epoch_s": time.monotonic() - t0,
           **{f"{k}_updates": sums[f"n_{k}"] for k in SCH.KINDS},
           **{f"mean_{k}_loss": (sums[f"{k}_loss"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None) for k in SCH.KINDS},
           "shared_digest": shared_digest(arm, epoch, c["real"][epoch], schedule(arm, seed, epoch, c["ctx"]["rr"],
                                                                                  c["ctx"]["sub"]), rows),
           "draw_digest": N.digest({str(s): [rows[v[0]]["uid"], *v[1:]] for s, v in sorted(wmap.items())})}
    row["losses_digest"] = N.digest(losses) if limit is not None else None
    c["losses"] = losses
    return row


def retention_epoch(row: dict, c: dict, run: Path, name: str, arm: str, seed: int, epoch: int) -> dict:
    """JOINT-1's final-epoch retention: holdout losses, final.pt, English TRAIN-holdout objects and their primary."""
    row.update(rt.evaluate(c["model"], c["windows"], c["hold"]))
    final = run / "final.pt"
    pt.save({"state_dict": c["model"].state_dict(), "epoch": epoch, "seed": seed, "arm": arm,
             "start": N.J_START[seed][1]}, final)
    row["final_sha256"] = rt.sha(final)
    objs = wt.holdout_objects(final, name)
    N.write_json(run / "HOLDOUT_OBJECTS_PRIVATE.json", {"model": name, "checkpoint_sha256": row["final_sha256"],
                                                        "objects": objs}, indent=None)
    retention = wt.holdout_primary(objs)
    row["holdout_primary"] = retention["primary"]
    return retention


def train_context(seed: int, arm: str, paths: dict) -> dict:
    check_constants()
    pt.verify_start()
    wins = plans.load_windows()
    windows, _, _, hold = wins
    rt.verify_masks(windows)
    return {"ctx": region_context(arm, paths), "windows": windows, "hold": hold, "real": english_real(seed, wins)}


def check_run(seed: int, arm: str) -> str:
    if arm not in N.ARMS or seed not in N.SEEDS:
        raise SystemExit(f"train: arm {arm!r} / seed {seed} outside the frozen design")
    return N.run_name(arm, seed)


def train(seed: int, arm: str, paths: dict, run_root: Path) -> dict:
    """Resumable per epoch (last.pt), as JOINT-1."""
    import torch
    name = check_run(seed, arm)
    c = train_context(seed, arm, paths)
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    last, log_path = run / "last.pt", run / "epochs.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return N.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    c["model"], c["opt"] = fresh_model(seed)
    torch.manual_seed(seed)
    done, retention = wt._resume(c["model"], c["opt"], last), None
    for epoch in SCH.EPOCHS:
        if epoch < done:
            continue
        row = epoch_pass(arm, seed, epoch, c, name, started)
        if epoch == SCH.EPOCHS[-1]:
            retention = retention_epoch(row, c, run, name, arm, seed, epoch)
        pt.save({"state_dict": c["model"].state_dict(), "optimizer": c["opt"].state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"run": name, **row}), flush=True)
    rows_log = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if retention is None:
        retention = wt.holdout_primary(N.read_json(run / "HOLDOUT_OBJECTS_PRIVATE.json")["objects"])
    result = result_record(arm, seed, run, rows_log, retention, c, paths, started)
    N.write_json(run / "TRAIN_RESULT.json", result)
    last.unlink(missing_ok=True)
    return result


def result_record(arm: str, seed: int, run: Path, rows_log: list, retention: dict, c: dict, paths: dict,
                  started: float) -> dict:
    import torch
    final = run / "final.pt"
    keys = ("w_table", "i_table") + (("ls_table",) if arm == "A" else ())      # C never reads the LS table
    tables = {k: N.sha256_file(paths[k]) for k in keys if paths.get(k)}
    return {"run": run.name, "arm": arm, "seed": seed, "status": "COMPLETE", "final": str(final),
            "final_sha256": rt.sha(final), "final_bytes": final.stat().st_size, "epochs": rows_log,
            "retention": retention, "plan_summary": plan_summary(arm, seed, c["ctx"], c["real"]),
            "start": {"path": N.J_START[seed][0], "sha256": N.J_START[seed][1]},
            "updates": SCH.N_UPDATES, "lr": "cosine 1e-4 -> 1e-6 over 5,116 updates (J's, compressed; no warm-up)",
            "english_plan_sha256": N.sha256_file(english_plan_path(seed)), "tables_sha256": tables,
            "script_sha256": N.sha256_file(Path(__file__)), "schedule_sha256": N.sha256_file(Path(SCH.__file__)),
            "joint1_train_sha256": N.sha256_file(Path(JT1.__file__)),
            "frozen_trainer_sha256": N.sha256_file(Path(at.__file__)), "wall_s": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "attribution": N.ATTRIBUTION if arm == "A" else {k: N.ATTRIBUTION[k] for k in ("terrain",)},
            **(wt.env_record() if torch.cuda.is_available() else {})}


# ---------------------------------------------------------------- smoke (staged windows; GPU on the VM, CPU locally)
def staged_paths(paths: dict) -> dict:
    """Tables restricted to staged windows (JOINT-1 staged); subsets = the staged monuments (smoke)."""
    out = dict(paths)
    for k, w, m in (("w_table", "w_win", "w_mask"), ("i_table", "i_win", "i_mask"), ("ls_table", "ls_win", "ls_mask")):
        if not paths.get(k):
            continue
        t = JT1.staged(N.read_json(paths[k]), Path(paths[w]), Path(paths[m]))
        p = Path(paths["scratch"]) / f"STAGED_{k}.json"
        N.write_json(p, t, indent=None)
        out[k] = p
    return out


def smoke(arm: str, n: int, out: Path, paths: dict, device: str = "cuda", run_root: Path | None = None) -> dict:
    """First n updates of epoch 4 at seed 2026092802 on staged windows; with run_root the model is saved as
    RUN_ROOT/SMOKE_<arm>/final.pt (train format) for the artifact route."""
    import torch
    seed = N.SEEDS[0]
    sp = staged_paths(paths)
    check_constants()
    pt.verify_start()
    wins = plans.load_windows()
    w_t, i_t = N.read_json(sp["w_table"]), N.read_json(sp["i_table"])
    rr = JT1.region_rows(w_t, Path(sp["w_win"]), Path(sp["w_mask"]), i_t, Path(sp["i_win"]), Path(sp["i_mask"]))
    rr["ls_pos"], rr["ls_bg"] = [], []
    if arm == "A":
        rows, pos, bg = ls_rows(N.read_json(sp["ls_table"]), Path(sp["ls_win"]), Path(sp["ls_mask"]), len(rr["rows"]))
        rr["rows"], rr["ls_pos"], rr["ls_bg"] = rr["rows"] + rows, pos, bg
    sub = {"smoke": True, "wales": {str(seed): sorted(w_t["monuments"])}, "ireland": {str(seed): sorted(i_t["monuments"])}}
    c = {"ctx": {"rr": rr, "sub": sub}, "windows": wins[0], "real": english_real(seed, wins)}
    torch.manual_seed(seed)
    c["model"], c["opt"] = fresh_model(seed, device)
    t0 = time.monotonic()
    row = epoch_pass(arm, seed, SCH.EPOCHS[0], c, f"SMOKE_{arm}", t0, device=device, limit=n)
    rec = {"arm": arm, "seed": seed, "n": n, "device": device, "row": row, "losses": c["losses"],
           "s_per_update": (time.monotonic() - t0) / max(n, 1),
           "all_finite": all(v is None or math.isfinite(v) for v in c["losses"])}
    if run_root is not None:
        run = Path(run_root) / f"SMOKE_{arm}"
        run.mkdir(parents=True, exist_ok=True)
        pt.save({"state_dict": c["model"].state_dict(), "epoch": SCH.EPOCHS[0], "seed": seed, "arm": arm,
                 "smoke_updates": n}, run / "final.pt")
        rec["final_sha256"], rec["final_bytes"] = rt.sha(run / "final.pt"), (run / "final.pt").stat().st_size
        N.write_json(run / "TRAIN_RESULT.json", {"run": run.name, "arm": arm, "seed": seed, "tag": "smoke",
                                                "status": "SMOKE", "final": str(run / "final.pt"),
                                                "final_sha256": rec["final_sha256"],
                                                "final_bytes": rec["final_bytes"]})
    N.write_json(out, rec)
    if not rec["all_finite"]:
        raise SystemExit("smoke: non-finite loss")
    return rec


# ---------------------------------------------------------------- plan-check (CPU, no forward)
def plan_check(seed: int, paths: dict, out: Path) -> dict:
    """A and C schedules of both epochs on the real tables: every step outside the Lower Saxony pairs has the same
    source, window, offsets, dihedral and update index in A and C; the shares equal nds_schedule.kind_counts."""
    check_constants()
    wins = plans.load_windows()
    real = english_real(seed, wins)
    ctx = {arm: region_context(arm, paths) for arm in ("A", "C")}
    rec = {"seed": seed, "epochs": [], "kind_counts": {a: SCH.kind_counts(a) for a in ("A", "C")},
           "replacement_counts": SCH.replacement_counts()}
    totals = {a: {k: 0 for k in SCH.KINDS} for a in ("A", "C")}
    bad = []
    for e in SCH.EPOCHS:
        srcs = {}
        for a in ("A", "C"):
            wmap = schedule(a, seed, e, ctx[a]["rr"], ctx[a]["sub"])
            srcs[a] = SCH.step_sources(a, e, real[e], wmap, ctx[a]["rr"]["rows"])
            for k in kinds_of(wmap, ctx[a]["rr"]["rows"]).values():
                totals[a][k] += 1
        ls = SCH.ls_steps(e)
        diff = [s for s in range(SCH.STEPS_PER_EPOCH) if s not in ls and srcs["A"][s] != srcs["C"][s]]
        ls_bad = [s for s in ls if srcs["A"][s][0] != LS_REGION or srcs["C"][s][0] == LS_REGION]
        bad += [f"epoch {e}: {len(diff)} shared steps differ"] if diff else []
        bad += [f"epoch {e}: {len(ls_bad)} Lower Saxony slots wrong"] if ls_bad else []
        rec["epochs"].append({"epoch": e, "shared_steps": SCH.STEPS_PER_EPOCH - len(ls), "shared_differ": len(diff),
                              "ls_steps": len(ls),
                              "shared_digest_A": N.digest([[s, *srcs["A"][s]] for s in range(SCH.STEPS_PER_EPOCH)
                                                           if s not in ls]),
                              "shared_digest_C": N.digest([[s, *srcs["C"][s]] for s in range(SCH.STEPS_PER_EPOCH)
                                                           if s not in ls])})
    for a in ("A", "C"):
        if totals[a] != SCH.kind_counts(a)["updates"]:
            bad.append(f"{a}: realised update counts {totals[a]} differ from nds_schedule")
    ls_used = {r["uid"] for e in SCH.EPOCHS for s, v in schedule("A", seed, e, ctx["A"]["rr"], ctx["A"]["sub"]).items()
               if ctx["A"]["rr"]["rows"][v[0]]["region"] == LS_REGION for r in [ctx["A"]["rr"]["rows"][v[0]]]}
    rec |= {"realised_updates": totals, "ls_windows_drawn": len(ls_used),
            "ls_windows_available": len(ctx["A"]["rr"]["ls_pos"]) + len(ctx["A"]["rr"]["ls_bg"]),
            "status": "FAIL" if bad else "PASS", "violations": bad, "utc": N.utc(), "attribution": N.ATTRIBUTION}
    N.write_json(out, rec)
    if bad:
        raise SystemExit(f"plan-check failed: {bad}")
    return rec


def parse_paths(a) -> dict:
    keys = ("w_table", "w_win", "w_mask", "i_table", "i_win", "i_mask", "ls_table", "ls_win", "ls_mask", "subsets",
            "scratch")
    return {k: getattr(a, k) for k in keys if getattr(a, k, None) is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plans", "plan-check", "step0", "train", "smoke"))
    ap.add_argument("args", nargs="*")
    for k in ("w-table", "w-win", "w-mask", "i-table", "i-win", "i-mask", "ls-table", "ls-win", "ls-mask",
              "run-root", "subsets", "scratch"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd in ("step0", "train", "smoke") and a.device == "cuda":
        JT1.J.torch_flags()
    paths = parse_paths(a)
    if a.cmd == "plans":
        print(json.dumps(build_english_plans(), indent=1))
    elif a.cmd == "plan-check":
        print(json.dumps({k: v for k, v in plan_check(int(a.args[0]), paths, Path(a.args[1])).items()
                          if k != "epochs"}))
    elif a.cmd == "step0":
        print(json.dumps(JT1.step0(Path(a.args[0]))))
    elif a.cmd == "train":
        r = train(int(a.args[0]), a.args[1], paths, a.run_root)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "wall_s")}))
    else:
        if "scratch" not in paths:
            paths["scratch"] = Path(a.args[2]).parent
        r = smoke(a.args[0], int(a.args[1]), Path(a.args[2]), paths, a.device, a.run_root)
        print(json.dumps({k: r[k] for k in ("arm", "n", "device", "s_per_update", "all_finite")}))


if __name__ == "__main__":
    main()
