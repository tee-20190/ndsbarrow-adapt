"""HOLDOUT-REGION-1 Stage A item 3: Welsh arm trainer on the frozen AMZ-ADAPT-1 recipe (imported, never edited).

Frozen pieces used unchanged: adapt_train.fresh_model (CONT e41 start, fresh AdamW wd 1e-4), adapt_train.update
(cosine LR plans.lr_at per scheduled update, r2_train.masked_loss, batch 1, fp32, frozen BatchNorm), adapt_train
.AdaptSteps + loader (void policy V, 512 px crops, +-160 px offsets, dihedral), adapt_train.replaced_pair, the English
plan plans.load_plan(seed) (10,232 updates), torch.manual_seed(seed), paired_train.save/verify_start, final checkpoint
only. Only the map "scheduled step -> Welsh window draw" differs between arms (packet "Arms"):

  C       no Welsh draws: the English plan of the seed, unchanged
  A_W0    at every pair p with (p - e) % 4 == 0 the background step 2p + 1 is a Welsh FIT background draw; the
          positive step 2p keeps the English positive
  A_W30   at the same pairs step 2p is a Welsh positive (a window centred on a sampled FIT monument of the seed's
          30-monument subset) and step 2p + 1 is the same Welsh background draw as A_W0
  A_WALL  as A_W30 with all eligible FIT monuments
Draws are shared within a training seed: stream default_rng([seed + 20, kind, cycle]) (kind 0 monument, 1
background), offsets/dihedral r2_train.draw(default_rng([seed + 20, 2, epoch, step])), cycling through permutations
continued across epochs as adapt_train.amazon_plan. A monument with several windows gives window `cycle % n`.

Retention inside the final epoch (packet "Steps" 4): the recipe's holdout loss (r2_train.evaluate) plus TRAIN-holdout
objects of final.pt on the 804 holdout windows through the frozen tta_run/holdout_eval path and their primary
(score_holdout). The gate on the five-seed mean is gates.py retention; a run never stops on it.
Step 0: a run whose step 0 is the English first step of seed 2026092800 (C, A_W0) must reproduce the declared
reference within 1e-6 or it stops with exit 96 (the chain's step-0 gate runs first: `step0`).

  python -B welsh_train.py step0 OUT.json
  python -B welsh_train.py train SEED ARM --fit-table T --win-dir W --mask-dir M --run-root R [--tag rerun]
  python -B welsh_train.py smoke ARM N OUT.json --fit-table T --win-dir W --mask-dir M [--run-root R]  (staged only)
  python -B welsh_train.py plan-check SEED --fit-table T                                     (CPU; no forward)
  python -B welsh_train.py holdout-check N OUT.json [--checkpoint PT --name NAME]   (smoke of the retention path)
GPU commands call common.torch_flags() (cuDNN/matmul TF32 off; CUDNN_ALLOW_TF32=1 restores the PyTorch default).
(repo root; Colab: /content/venv/bin/python with NVIDIA_TF32_OVERRIDE=0)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

ROOT = C.ROOT
sys.path.insert(0, str(ROOT / "work/amazon_adapt_2026-09-28"))
import adapt_train as at  # noqa: E402  frozen trainer (also puts r2_train, plans, paired_train on sys.path)
import paired_train as pt  # noqa: E402
import plans  # noqa: E402
import r2_train as rt  # noqa: E402

STEP0_EXIT = 96
HOLDOUT_RAW = ROOT / "outputs/campaign_2026-09-22/train_raw"


# ---------------------------------------------------------------- Welsh pools and draw plan (CPU)
def welsh_rows(table: dict, win_dir: Path, mask_dir: Path) -> tuple[list[dict], dict[str, list[int]], list[int]]:
    """AdaptSteps rows (absolute tif / mask paths), monument -> row indices, background row indices."""
    wins = table["positive_windows"] + table["background_windows"]
    rows = [{"id": w["id"], "tif": str(win_dir / f"{w['id']}.tif"), "mask_path": str(mask_dir / f"{w['id']}.npz"),
             "role": w["role"], "sam": w.get("sam"), "incidental": len(w.get("incidental_positives", []))}
            for w in wins]
    index = {r["id"]: i for i, r in enumerate(rows)}
    monuments = {sam: [index[w] for w in ids] for sam, ids in table["monuments"].items() if all(w in index for w in ids)}
    return rows, monuments, [index[w["id"]] for w in table["background_windows"]]


def arm_monuments(arm: str, seed: int, table: dict) -> list[str]:
    if arm == "A_WALL":
        return list(table["arms"]["A_WALL"])
    if arm == "A_W30":
        return list(table["arms"]["A_W30"][str(seed)])
    return []


def replaced_pairs(epoch: int) -> list[int]:
    return [p for p in range(plans.STEPS_PER_EPOCH // 2) if at.replaced_pair(p, epoch)]


def welsh_plan(arm: str, seed: int, epoch: int, monuments: dict[str, list[int]], pool: list[str],
               bg: list[int]) -> dict[int, tuple[int, int, int, int]]:
    """Scheduled step -> (Welsh row index, dy, dx, k) for one epoch of one arm."""
    if arm not in C.ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    if arm == "C":
        return {}
    draw = seed + C.WELSH_DRAW_OFFSET
    offset = sum(len(replaced_pairs(e)) for e in range(epoch))
    out = {}
    for m, p in enumerate(replaced_pairs(epoch)):
        n = offset + m
        if arm in ("A_W30", "A_WALL"):
            cycle, idx = divmod(n, len(pool))
            sam = pool[int(np.random.default_rng([draw, 0, cycle]).permutation(len(pool))[idx])]
            wins = monuments[sam]
            out[2 * p] = (wins[cycle % len(wins)], *rt.draw(np.random.default_rng([draw, 2, epoch, 2 * p])))
        cycle, idx = divmod(n, len(bg))
        wi = int(np.random.default_rng([draw, 1, cycle]).permutation(bg)[idx])
        out[2 * p + 1] = (wi, *rt.draw(np.random.default_rng([draw, 2, epoch, 2 * p + 1])))
    return out


def epoch_inputs(arm: str, seed: int, epoch: int, plan: dict, table: dict, monuments: dict,
                 bg: list[int]) -> tuple[list, dict]:
    real = plan["real"][epoch]
    if len(real) != plans.STEPS_PER_EPOCH:
        raise RuntimeError("real plan length changed")
    pool = arm_monuments(arm, seed, table)
    if arm in ("A_W30", "A_WALL") and (not pool or any(s not in monuments for s in pool)):
        raise RuntimeError(f"{arm}: monument pool missing or not staged")
    return real, welsh_plan(arm, seed, epoch, monuments, pool, bg)


def plan_summary(arm: str, seed: int, table: dict, plan: dict | None = None) -> dict:
    """Per-epoch Welsh positions and draw digests; unique centred monuments (k) and incidental exposure."""
    rows, monuments, bg = welsh_rows(table, Path("."), Path("."))
    plan = plan or {"real": [[[0, 0, 0, 0]] * plans.STEPS_PER_EPOCH] * plans.N_EPOCHS}
    out, centred, incidental = {"arm": arm, "seed": seed, "epochs": []}, set(), 0
    for e in range(plans.N_EPOCHS):
        _, wmap = epoch_inputs(arm, seed, e, plan, table, monuments, bg)
        pos = sorted(s for s in wmap if rows[wmap[s][0]]["role"] == "positive")
        bgs = sorted(s for s in wmap if rows[wmap[s][0]]["role"] == "background")
        centred |= {rows[wmap[s][0]]["sam"] for s in pos}
        incidental += sum(rows[wmap[s][0]]["incidental"] > 0 for s in pos)
        out["epochs"].append({"epoch": e, "welsh_positive_steps": len(pos), "welsh_background_steps": len(bgs),
                              "draw_digest": C.digest({str(s): [rows[v[0]]["id"], *v[1:]] for s, v in wmap.items()})})
    out.update(unique_centred_monuments=len(centred), positive_draws_with_incidental=incidental,
               welsh_updates=sum(x["welsh_positive_steps"] + x["welsh_background_steps"] for x in out["epochs"]))
    return out


# ---------------------------------------------------------------- GPU paths
def env_record() -> dict:
    import torch
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def step0(out: Path) -> dict:
    """Loss of C's first update for seed 2026092800 (the declared reference configuration); nothing is saved."""
    import torch
    pt.verify_start()
    windows = plans.load_windows()[0]
    real = plans.load_plan(C.STEP0_SEED)["real"][0][:1]
    model, opt = at.fresh_model()
    torch.manual_seed(C.STEP0_SEED)
    r = None
    for s, x, y, _ in at.loader(at.AdaptSteps(windows, real, [], {})):
        r = at.update(model, opt, 0, x, y)
    if r is None or r["skipped"]:
        raise SystemExit("step0: the first sample was skipped")
    rec = {"seed": C.STEP0_SEED, "arm": "C", "step": 0, "loss": r["loss"], "repr": repr(r["loss"]),
           "hex": float.hex(r["loss"]), "reference": C.STEP0_REF, "abs_diff": abs(r["loss"] - C.STEP0_REF),
           **env_record()}
    C.write_json(out, rec)
    return rec


def holdout_objects(final: Path, name: str) -> dict:
    """TRAIN-holdout objects of final.pt: frozen tta_run.objects_multi + holdout_eval.predictor, 804 windows."""
    import torch
    import holdout_eval as he
    import tta_run as tr
    tr.ac.check_frozen()
    (model,), _ = tr.ac.load_models([final])
    predict = he.predictor({name: model})
    torch.set_num_threads(4)
    objs = {w["id"]: tr.objects_multi(HOLDOUT_RAW / f"{w['id']}.tif", predict, [name])["arms"][name]["objects"]
            for w in tr.holdout_windows()}
    del model
    torch.cuda.empty_cache()
    return objs


def holdout_check(n: int, out: Path, checkpoint: Path | None = None, name: str = "CONT") -> dict:
    """Smoke of the in-trainer retention path on the first n holdout windows: the checkpoint (default CONT e41; the
    smoke passes SMOKE_A_WALL/final.pt, a welsh_train checkpoint) through the frozen loader and tta_run path, then
    holdout_primary on those objects with the other windows empty. That primary is a format check, not an estimate."""
    import torch
    import holdout_eval as he
    import tta_run as tr
    ck = checkpoint or pt.START
    tr.ac.check_frozen()
    (model,), _ = tr.ac.load_models([ck])
    predict = he.predictor({name: model})
    t0 = time.monotonic()
    wins = tr.holdout_windows()
    objs = {w["id"]: tr.objects_multi(HOLDOUT_RAW / f"{w['id']}.tif", predict, [name])["arms"][name]["objects"]
            for w in wins[:n]}
    wall = time.monotonic() - t0
    prim = holdout_primary({w["id"]: objs.get(w["id"], []) for w in wins})
    rec = {"checkpoint": str(ck), "checkpoint_sha256": C.sha256_file(ck), "name": name, "windows": len(objs),
           "objects": {k: len(v) for k, v in objs.items()}, "s_per_window": wall / max(len(objs), 1),
           "subset_primary_format_check": prim["primary"], "setup_eligible_parts": prim["eligible_parts"],
           **env_record()}
    C.write_json(out, rec)
    del model
    torch.cuda.empty_cache()
    return rec


def holdout_primary(objs: dict) -> dict:
    import score_holdout as shd
    s = shd.setup()
    a = shd.arm_summary(shd.ranked_pool(objs, s), s)
    return {"primary": a["primary"], "by_density": [a[f"d{d}"]["recall"] for d in shd.DENSITIES],
            "negative_per_km2": a["negative_per_km2"], "eligible_parts": len(s["eligible"])}


def _resume(model, opt, last: Path) -> int:
    import torch
    if not last.exists():
        return 0
    state = torch.load(last, map_location="cuda", weights_only=False)
    model.load_state_dict(state["state_dict"])
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng_cpu"].cpu())
    torch.cuda.set_rng_state(state["rng_cuda"].cpu())
    return state["epoch"] + 1


def _check_step0(seed: int, wmap: dict, loss: float) -> None:
    if seed == C.STEP0_SEED and 0 not in wmap and not abs(loss - C.STEP0_REF) <= C.STEP0_TOL:
        print(json.dumps({"step0_failed": loss, "reference": C.STEP0_REF}), flush=True)
        raise SystemExit(STEP0_EXIT)


def train(seed: int, arm: str, table_path: Path, win_dir: Path, mask_dir: Path, run_root: Path,
          tag: str = "") -> dict:
    import torch
    table = C.read_json(table_path)
    pt.verify_start()
    windows, _, _, hold = plans.load_windows()
    rt.verify_masks(windows)
    plan = plans.load_plan(seed)
    rows, monuments, bg = welsh_rows(table, win_dir, mask_dir)
    name = f"{arm}_{seed}{'_' + tag if tag else ''}"
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    final, last, log_path = run / "final.pt", run / "last.pt", run / "epochs.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return C.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model, opt = at.fresh_model()
    torch.manual_seed(seed)
    first, step0_loss, retention = _resume(model, opt, last), None, None
    for epoch in range(first, plans.N_EPOCHS):
        t0 = time.monotonic()
        real, wmap = epoch_inputs(arm, seed, epoch, plan, table, monuments, bg)
        sums = {"loss": 0.0, "n": 0, "w_loss": 0.0, "n_w": 0, "skipped": 0}
        for s, x, y, is_w in at.loader(at.AdaptSteps(windows, real, rows, wmap)):
            r = at.update(model, opt, epoch * plans.STEPS_PER_EPOCH + int(s), x, y)
            if epoch == 0 and int(s) == 0:
                step0_loss = None if r["skipped"] else r["loss"]
                _check_step0(seed, wmap, step0_loss if step0_loss is not None else math.nan)
            if r["skipped"]:
                sums["skipped"] += 1
                continue
            sums["w_loss" if is_w else "loss"] += r["loss"]
            sums["n_w" if is_w else "n"] += 1
            if int(s) % at.LOG_EVERY == 0:
                print(json.dumps({"run": name, "epoch": epoch, "step": int(s), **r,
                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        row = {"epoch": epoch, "english_updates": sums["n"], "welsh_updates": sums["n_w"],
               "skipped_all_ignore": sums["skipped"], "mean_english_loss": sums["loss"] / max(sums["n"], 1),
               "mean_welsh_loss": sums["w_loss"] / max(sums["n_w"], 1), "lr_end": opt.param_groups[0]["lr"],
               "draw_digest": C.digest({str(s): [rows[v[0]]["id"], *v[1:]] for s, v in wmap.items()}),
               "epoch_s": time.monotonic() - t0}
        if epoch == 0:
            row["step0_loss"] = step0_loss
        if epoch == plans.N_EPOCHS - 1:
            row.update(rt.evaluate(model, windows, hold))
            pt.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": seed, "arm": arm}, final)
            row["final_sha256"] = rt.sha(final)
            objs = holdout_objects(final, name)
            C.write_json(run / "HOLDOUT_OBJECTS_PRIVATE.json", {"model": name, "checkpoint_sha256": row["final_sha256"],
                                                                "objects": objs}, indent=None)
            retention = holdout_primary(objs)
            row["holdout_primary"] = retention["primary"]
        pt.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"run": name, **row}), flush=True)
    rows_log = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if retention is None:
        retention = holdout_primary(C.read_json(run / "HOLDOUT_OBJECTS_PRIVATE.json")["objects"])
    result = {"run": name, "arm": arm, "seed": seed, "tag": tag, "status": "COMPLETE",
              "final": str(final), "final_sha256": rt.sha(final), "epochs": rows_log,
              "step0_loss": rows_log[0].get("step0_loss"), "retention": retention,
              "plan_summary": plan_summary(arm, seed, table, plan),
              "start_sha256": pt.START_SHA256, "plan_sha256": rt.sha(plans.PLANS / f"PLAN_{seed}_PRIVATE.json"),
              "fit_table_sha256": C.sha256_file(table_path), "script_sha256": C.sha256_file(Path(__file__)),
              "frozen_trainer_sha256": C.sha256_file(Path(at.__file__)), "wall_s": time.monotonic() - started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, **env_record()}
    C.write_json(run / "TRAIN_RESULT.json", result)
    last.unlink(missing_ok=True)
    return result


def staged_table(table: dict, win_dir: Path, mask_dir: Path) -> dict:
    """The FIT table restricted to staged windows (Step 2c smoke): pools of whatever was staged."""
    ok = {w["id"] for w in table["positive_windows"] + table["background_windows"]
          if (win_dir / f"{w['id']}.tif").exists() and (mask_dir / f"{w['id']}.npz").exists()}
    mons = {k: [w for w in v if w in ok] for k, v in table["monuments"].items()}
    mons = {k: v for k, v in mons.items() if v}
    return {**table, "positive_windows": [w for w in table["positive_windows"] if w["id"] in ok],
            "background_windows": [w for w in table["background_windows"] if w["id"] in ok], "monuments": mons,
            "arms": {**table["arms"], "A_WALL": sorted(mons),
                     "A_W30": {s: sorted(mons) for s in table["arms"]["A_W30"]}}}


def smoke(arm: str, n: int, out: Path, table_path: Path, win_dir: Path, mask_dir: Path,
          run_root: Path | None = None) -> dict:
    """First n updates of epoch 0 at seed 2026092800 with the staged Welsh windows only. With run_root, the model
    after n updates is saved as RUN_ROOT/SMOKE_<arm>/final.pt (same format as train) for the final-weight route."""
    import torch
    table = staged_table(C.read_json(table_path), win_dir, mask_dir)
    if not table["background_windows"] or (arm in ("A_W30", "A_WALL") and not table["positive_windows"]):
        raise SystemExit("smoke: required Welsh windows are not staged")
    pt.verify_start()
    windows = plans.load_windows()[0]
    seed = C.STEP0_SEED
    rows, monuments, bg = welsh_rows(table, win_dir, mask_dir)
    real, wmap = epoch_inputs(arm, seed, 0, plans.load_plan(seed), table, monuments, bg)
    real, wmap = real[:n], {s: v for s, v in wmap.items() if s < n}
    torch.cuda.reset_peak_memory_stats()
    model, opt = at.fresh_model()
    torch.manual_seed(seed)
    times, stats, prev = {"english": [], "welsh": []}, {"welsh_samples": 0, "welsh_pos_px": 0, "welsh_ignore_px": 0,
                                                         "losses": []}, time.monotonic()
    for s, x, y, is_w in at.loader(at.AdaptSteps(windows, real, rows, wmap)):
        r = at.update(model, opt, int(s), x, y)
        torch.cuda.synchronize()
        now = time.monotonic()
        if int(s) >= 8:
            times["welsh" if is_w else "english"].append(now - prev)
        prev = now
        if is_w:
            stats["welsh_samples"] += 1
            stats["welsh_pos_px"] += int((y == 1).sum())
            stats["welsh_ignore_px"] += int((y == rt.IGNORE).sum())
        stats["losses"].append(None if r["skipped"] else r["loss"])
    rec = {"arm": arm, "seed": seed, "n": n, "welsh_steps": sorted(wmap),
           "s_per_english_update": float(np.mean(times["english"])) if times["english"] else None,
           "s_per_welsh_update": float(np.mean(times["welsh"])) if times["welsh"] else None,
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, **stats, **env_record()}
    rec["all_finite"] = all(v is None or math.isfinite(v) for v in stats["losses"])
    if run_root is not None:
        run = run_root / f"SMOKE_{arm}"
        run.mkdir(parents=True, exist_ok=True)
        pt.save({"state_dict": model.state_dict(), "epoch": 0, "seed": seed, "arm": arm, "smoke_updates": n},
                run / "final.pt")
        rec["final_sha256"], rec["final_bytes"] = rt.sha(run / "final.pt"), (run / "final.pt").stat().st_size
        C.write_json(run / "TRAIN_RESULT.json", {"run": run.name, "arm": arm, "seed": seed, "tag": "smoke",
                                                 "status": "SMOKE", "final": str(run / "final.pt"),
                                                 "final_sha256": rec["final_sha256"]})
    C.write_json(out, rec)
    if not rec["all_finite"] or stats["welsh_samples"] == 0:
        raise SystemExit("smoke: non-finite loss or no Welsh sample")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("step0", "train", "smoke", "plan-check", "holdout-check"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--fit-table", type=Path)
    ap.add_argument("--win-dir", type=Path)
    ap.add_argument("--mask-dir", type=Path)
    ap.add_argument("--run-root", type=Path)
    ap.add_argument("--tag", default="")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--name", default="CONT")
    a = ap.parse_args()
    if a.cmd in ("step0", "train", "smoke", "holdout-check"):
        C.torch_flags()
    if a.cmd == "step0":
        print(json.dumps(step0(Path(a.args[0]))))
    elif a.cmd == "holdout-check":
        print(json.dumps(holdout_check(int(a.args[0]), Path(a.args[1]), a.checkpoint, a.name)))
    elif a.cmd == "train":
        r = train(int(a.args[0]), a.args[1], a.fit_table, a.win_dir, a.mask_dir, a.run_root, a.tag)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "step0_loss", "retention")}))
    elif a.cmd == "smoke":
        print(json.dumps(smoke(a.args[0], int(a.args[1]), Path(a.args[2]), a.fit_table, a.win_dir, a.mask_dir,
                               a.run_root)))
    else:
        table = C.read_json(a.fit_table)
        seed = int(a.args[0])
        print(json.dumps({arm: plan_summary(arm, seed, table) for arm in C.ARMS}, indent=1))


if __name__ == "__main__":
    main()
