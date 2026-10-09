"""JOINT-1 trainer: English control C, joint English + Welsh FIT + Irish FIT model J, and J with ADAF AO output
distillation (JD), on the frozen AMZ-ADAPT recipe (adapt_train imported, never edited).

Frozen pieces used unchanged: adapt_train.fresh_model (CONT e41 start, fresh AdamW wd 1e-4), adapt_train.update
(cosine LR per scheduled update, r2_train.masked_loss, batch 1, fp32, frozen BatchNorm), AdaptSteps + loader (void
policy V, 512 px crops, +-160 px offsets, dihedral), the English plan plans.load_plan(seed) (10,232 updates),
torch.manual_seed(seed), paired_train.save/verify_start, HR1 welsh_train helpers (welsh_rows, welsh_plan,
holdout_objects, holdout_primary, env_record, _resume, step0). Only the map "scheduled step -> regional window draw"
and, for JD, the distillation term differ between arms (PACKET.md "Arms"):

  C    no regional draws: the English plan of the seed, unchanged (HR1's C path)
  J    pair p of epoch e with (p - e) % 4 == 0: HR1 A_W30's Welsh pair (welsh_plan, the seed's 30-monument subset,
       identical draws); (p - e) % 4 == 2: HR3 A_I30's Irish draw streams (default_rng([seed + 20, kind, cycle]),
       offsets default_rng([seed + 20, 2, epoch, step])) on those pairs; other pairs English
  JD   J's schedule; every update adds KD_LAMBDA * BCE(student logit, sigmoid(ADAF AO logit)) over label-background
       pixels (y = 0) only: trusted positives (y = 1) keep their supervised target alone, so a teacher miss never
       overrides them, and IGNORE pixels get neither term (owner clarification 2026-10-03 07:59Z). Teacher = pinned
       ADAF AO (r2_train.load_model(), eval mode, no gradient) on the same input tensor (same crop, dihedral and
       preprocessing; CONT e41 is AO fine-tuned, so both share the input contract and output grid)

Retention in the final epoch: TRAIN-holdout objects of final.pt on the 804 English holdout windows (frozen
tta_run/holdout_eval path) and their primary. Step 0: a run whose step 0 is the English first step of seed 2026092800
must reproduce the declared reference within 1e-6 (exit 96); the chain's own step-0 gate runs first.

  python -B joint_train.py step0 OUT.json
  python -B joint_train.py train SEED ARM --w-table T --w-win W --w-mask M --i-table T --i-win W --i-mask M --run-root R
  python -B joint_train.py smoke ARM N OUT.json --w-table ... --i-table ... [--run-root R]   (staged windows only)
  python -B joint_train.py kd-check N OUT.json --w-table ... --i-table ...                  (staged windows only)
  python -B joint_train.py plan-check SEED --w-table T --i-table T                           (CPU; no forward)
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
import joint_common as J  # noqa: E402

sys.path.insert(0, str(J.HR1_CHAIN))
import welsh_train as wt  # noqa: E402  HR1 trainer module (frozen): helpers only
from welsh_train import at, plans, pt, rt  # noqa: E402  frozen recipe modules as HR1 imports them

STEP0_EXIT = 96
N_PAIRS = plans.STEPS_PER_EPOCH // 2


def pairs_with(epoch: int, residue: int) -> list[int]:
    return [p for p in range(N_PAIRS) if (p - epoch) % 4 == residue]


def load_subsets(w_table: dict, i_table: dict, path: Path = J.SUBSETS) -> dict:
    """Per-seed 30-monument subsets; seeds 00-04 must equal the frozen HR1 A_W30 / HR3 A_I30 subsets."""
    sub = J.read_json(path)
    for s in J.FROZEN_SUBSET_SEEDS:
        if sorted(sub["wales"][str(s)]) != sorted(map(str, w_table["arms"]["A_W30"][str(s)])):
            raise SystemExit(f"Welsh subset of {s} differs from HR1's frozen A_W30 subset")
        if sorted(sub["ireland"][str(s)]) != sorted(map(str, i_table["arms"]["A_I30"][str(s)])):
            raise SystemExit(f"Irish subset of {s} differs from HR3's frozen A_I30 subset")
    return sub


def region_rows(w_table: dict, w_win: Path, w_mask: Path, i_table: dict, i_win: Path, i_mask: Path) -> dict:
    """AdaptSteps rows: Welsh rows then Irish rows (Irish indices offset), monument maps and background indices."""
    wr, wm, wb = wt.welsh_rows(w_table, w_win, w_mask)
    ir, im, ib = wt.welsh_rows(i_table, i_win, i_mask)
    off = len(wr)
    rows = wr + ir
    for reg, part in (("wales", wr), ("ireland", ir)):
        for r in part:
            r["region"], r["uid"] = reg, f"{reg}:{r['id']}"   # both regions number windows FIT-B-NNNN
    if len({r["uid"] for r in rows}) != len(rows) or len({r["tif"] for r in rows}) != len(rows):
        raise SystemExit("duplicate regional window or file")
    return {"rows": rows, "w_mon": wm, "w_bg": wb, "i_mon": {k: [i + off for i in v] for k, v in im.items()},
            "i_bg": [i + off for i in ib]}


def irish_plan(seed: int, epoch: int, monuments: dict[str, list[int]], pool: list[str],
               bg: list[int]) -> dict[int, tuple[int, int, int, int]]:
    """HR3 A_I30's draw streams on the pairs with (p - e) % 4 == 2 (irl_train.irish_plan with shifted pairs)."""
    draw = seed + J.DRAW_OFFSET
    offset = sum(len(pairs_with(e, J.IRISH_RESIDUE)) for e in range(epoch))
    out = {}
    for m, p in enumerate(pairs_with(epoch, J.IRISH_RESIDUE)):
        n = offset + m
        cycle, idx = divmod(n, len(pool))
        mon = pool[int(np.random.default_rng([draw, 0, cycle]).permutation(len(pool))[idx])]
        wins = monuments[mon]
        out[2 * p] = (wins[cycle % len(wins)], *rt.draw(np.random.default_rng([draw, 2, epoch, 2 * p])))
        cycle, idx = divmod(n, len(bg))
        wi = int(np.random.default_rng([draw, 1, cycle]).permutation(bg)[idx])
        out[2 * p + 1] = (wi, *rt.draw(np.random.default_rng([draw, 2, epoch, 2 * p + 1])))
    return out


def epoch_inputs(arm: str, seed: int, epoch: int, plan: dict, rr: dict, sub: dict) -> tuple[list, dict]:
    real = plan["real"][epoch]
    if len(real) != plans.STEPS_PER_EPOCH:
        raise RuntimeError("real plan length changed")
    if arm not in J.ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    if arm == "C":
        return real, {}
    wpool, ipool = list(sub["wales"][str(seed)]), list(sub["ireland"][str(seed)])
    smoke = sub.get("smoke", False)
    if not smoke and (len(wpool) != J.K or len(ipool) != J.K):
        raise RuntimeError("subset size is not 30")
    if any(m not in rr["w_mon"] for m in wpool) or any(m not in rr["i_mon"] for m in ipool):
        raise RuntimeError("a subset monument is not staged")
    wmap = wt.welsh_plan("A_W30", seed, epoch, rr["w_mon"], wpool, rr["w_bg"])     # HR1 A_W30, identical draws
    imap = irish_plan(seed, epoch, rr["i_mon"], ipool, rr["i_bg"])
    if set(wmap) & set(imap):
        raise RuntimeError("Welsh and Irish steps collide")
    return real, {**wmap, **imap}


def kd_term(z, zt, yc):
    """BCE(student logit, sigmoid(teacher logit)) over the label-background pixels; 0 when a crop has none."""
    import torch
    import torch.nn.functional as F
    if z.shape != zt.shape:
        raise RuntimeError("teacher and student output grids differ")
    bg = yc == 0
    if not bool(bg.any()):
        return z.sum() * 0.0
    return F.binary_cross_entropy_with_logits(z[bg], torch.sigmoid(zt[bg]))


def kd_update(model, teacher, opt, t: int, x, y, lam: float) -> dict:
    """adapt_train.update plus lam * kd_term (label-background pixels only)."""
    import torch
    lr = plans.lr_at(t)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    xin = x[None].cuda(non_blocking=True)
    yc = y.cuda(non_blocking=True)
    z = model(xin)["out"][0, 1]
    res = rt.masked_loss(z, yc)
    if res is None:
        return {"skipped": True, "lr": lr}
    with torch.no_grad():
        zt = teacher(xin)["out"][0, 1]
    kd = kd_term(z, zt, yc)
    (res[0] + lam * kd).backward()
    opt.step()
    return {"skipped": False, "loss": float(res[1] + (1 - res[2])), "kd": float(kd), "lr": lr}


def teacher_model():
    import torch
    model = rt.load_model()                            # pinned ADAF AO (md5-checked by adaf_configs), eval mode
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def step0(out: Path) -> dict:
    return wt.step0(out)                               # C's first update, seed 2026092800 (declared reference)


def _check_step0(seed: int, wmap: dict, loss: float) -> None:
    if seed == J.STEP0_SEED and 0 not in wmap and not abs(loss - J.STEP0_REF) <= J.STEP0_TOL:
        print(json.dumps({"step0_failed": loss, "reference": J.STEP0_REF}), flush=True)
        raise SystemExit(STEP0_EXIT)


def plan_summary(arm: str, seed: int, rr: dict, sub: dict, plan: dict | None = None) -> dict:
    plan = plan or {"real": [[[0, 0, 0, 0]] * plans.STEPS_PER_EPOCH] * plans.N_EPOCHS}
    rows = rr["rows"]
    out = {"arm": arm, "seed": seed, "epochs": []}
    centred = {"wales": set(), "ireland": set()}
    for e in range(plans.N_EPOCHS):
        _, wmap = epoch_inputs(arm, seed, e, plan, rr, sub)
        cnt = {f"{reg}_{role}": 0 for reg in ("wales", "ireland") for role in ("positive", "background")}
        for s, v in wmap.items():
            r = rows[v[0]]
            cnt[f"{r['region']}_{r['role']}"] += 1
            if r["role"] == "positive":
                centred[r["region"]].add(r["uid"])
        out["epochs"].append({"epoch": e, **cnt, "english_steps": plans.STEPS_PER_EPOCH - len(wmap),
                              "draw_digest": J.digest({str(s): [rows[v[0]]["uid"], *v[1:]] for s, v in wmap.items()})})
    out["unique_centred_windows"] = {k: len(v) for k, v in centred.items()}
    out["regional_updates"] = sum(sum(v for k, v in x.items() if k.startswith(("wales", "ireland")))
                                  for x in out["epochs"])
    return out


def _tables(a) -> tuple[dict, dict]:
    return J.read_json(a.w_table), J.read_json(a.i_table)


def train(seed: int, arm: str, w_table_path: Path, w_win: Path, w_mask: Path, i_table_path: Path, i_win: Path,
          i_mask: Path, run_root: Path, subsets: Path = J.SUBSETS) -> dict:
    import torch
    if arm not in J.ARMS or seed not in J.SEEDS:
        raise SystemExit(f"train: arm {arm!r} / seed {seed} outside the frozen design")
    w_table, i_table = J.read_json(w_table_path), J.read_json(i_table_path)
    sub = load_subsets(w_table, i_table, subsets)
    pt.verify_start()
    windows, _, _, hold = plans.load_windows()
    rt.verify_masks(windows)
    plan = plans.load_plan(seed)
    rr = region_rows(w_table, w_win, w_mask, i_table, i_win, i_mask)
    name = f"{arm}_{seed}"
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    final, last, log_path = run / "final.pt", run / "last.pt", run / "epochs.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return J.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model, opt = at.fresh_model()
    teacher = teacher_model() if arm == "JD" else None
    torch.manual_seed(seed)
    first, step0_loss, retention = wt._resume(model, opt, last), None, None
    for epoch in range(first, plans.N_EPOCHS):
        t0 = time.monotonic()
        real, wmap = epoch_inputs(arm, seed, epoch, plan, rr, sub)
        sums = {"loss": 0.0, "n": 0, "r_loss": 0.0, "n_r": 0, "kd": 0.0, "n_kd": 0, "skipped": 0}
        for s, x, y, is_r in at.loader(at.AdaptSteps(windows, real, rr["rows"], wmap)):
            t = epoch * plans.STEPS_PER_EPOCH + int(s)
            r = kd_update(model, teacher, opt, t, x, y, J.KD_LAMBDA) if teacher is not None else at.update(model, opt, t, x, y)
            if epoch == 0 and int(s) == 0:
                step0_loss = None if r["skipped"] else r["loss"]
                _check_step0(seed, wmap, step0_loss if step0_loss is not None else math.nan)
            if r["skipped"]:
                sums["skipped"] += 1
                continue
            sums["r_loss" if is_r else "loss"] += r["loss"]
            sums["n_r" if is_r else "n"] += 1
            if "kd" in r:
                sums["kd"] += r["kd"]
                sums["n_kd"] += 1
            if int(s) % at.LOG_EVERY == 0:
                print(json.dumps({"run": name, "epoch": epoch, "step": int(s), **r,
                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        row = {"epoch": epoch, "english_updates": sums["n"], "regional_updates": sums["n_r"],
               "skipped_all_ignore": sums["skipped"], "mean_english_loss": sums["loss"] / max(sums["n"], 1),
               "mean_regional_loss": sums["r_loss"] / max(sums["n_r"], 1),
               "mean_kd": sums["kd"] / sums["n_kd"] if sums["n_kd"] else None, "lr_end": opt.param_groups[0]["lr"],
               "draw_digest": J.digest({str(s): [rr["rows"][v[0]]["uid"], *v[1:]] for s, v in wmap.items()}),
               "epoch_s": time.monotonic() - t0}
        if epoch == 0:
            row["step0_loss"] = step0_loss
        if epoch == plans.N_EPOCHS - 1:
            row.update(rt.evaluate(model, windows, hold))
            pt.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": seed, "arm": arm}, final)
            row["final_sha256"] = rt.sha(final)
            objs = wt.holdout_objects(final, name)
            J.write_json(run / "HOLDOUT_OBJECTS_PRIVATE.json", {"model": name, "checkpoint_sha256": row["final_sha256"],
                                                                "objects": objs}, indent=None)
            retention = wt.holdout_primary(objs)
            row["holdout_primary"] = retention["primary"]
        pt.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"run": name, **row}), flush=True)
    rows_log = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if retention is None:
        retention = wt.holdout_primary(J.read_json(run / "HOLDOUT_OBJECTS_PRIVATE.json")["objects"])
    result = {"run": name, "arm": arm, "seed": seed, "status": "COMPLETE", "final": str(final),
              "final_sha256": rt.sha(final), "final_bytes": final.stat().st_size, "epochs": rows_log,
              "step0_loss": rows_log[0].get("step0_loss"), "retention": retention,
              "plan_summary": plan_summary(arm, seed, rr, sub, plan), "kd_lambda": J.KD_LAMBDA if arm == "JD" else 0.0,
              "start_sha256": pt.START_SHA256, "plan_sha256": rt.sha(plans.PLANS / f"PLAN_{seed}_PRIVATE.json"),
              "w_table_sha256": J.sha256_file(w_table_path), "i_table_sha256": J.sha256_file(i_table_path),
              "subsets_sha256": J.sha256_file(subsets), "script_sha256": J.sha256_file(Path(__file__)),
              "frozen_trainer_sha256": J.sha256_file(Path(at.__file__)), "wall_s": time.monotonic() - started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, **wt.env_record()}
    J.write_json(run / "TRAIN_RESULT.json", result)
    last.unlink(missing_ok=True)
    return result


def staged(table: dict, win: Path, mask: Path) -> dict:
    ok = {w["id"] for w in table["positive_windows"] + table["background_windows"]
          if (win / f"{w['id']}.tif").exists() and (mask / f"{w['id']}.npz").exists()}
    mons = {k: [w for w in v if w in ok] for k, v in table["monuments"].items()}
    return {**table, "positive_windows": [w for w in table["positive_windows"] if w["id"] in ok],
            "background_windows": [w for w in table["background_windows"] if w["id"] in ok],
            "monuments": {k: v for k, v in mons.items() if v}}


def smoke(arm: str, n: int, out: Path, a, run_root: Path | None = None) -> dict:
    """First n updates of epoch 0 at seed 2026092800 on staged regional windows (smoke subsets = staged monuments);
    with run_root the model is saved as RUN_ROOT/SMOKE_<arm>/final.pt (train format) for the artifact route."""
    import torch
    w_table, i_table = (staged(t, wd, md) for t, wd, md in ((J.read_json(a.w_table), a.w_win, a.w_mask),
                                                            (J.read_json(a.i_table), a.i_win, a.i_mask)))
    if not (w_table["background_windows"] and i_table["background_windows"] and w_table["monuments"]
            and i_table["monuments"]):
        raise SystemExit("smoke: required Welsh and Irish windows are not staged")
    seed = J.STEP0_SEED
    sub = {"smoke": True, "wales": {str(seed): sorted(w_table["monuments"])},
           "ireland": {str(seed): sorted(i_table["monuments"])}}
    pt.verify_start()
    windows = plans.load_windows()[0]
    rr = region_rows(w_table, a.w_win, a.w_mask, i_table, a.i_win, a.i_mask)
    real, wmap = epoch_inputs(arm, seed, 0, plans.load_plan(seed), rr, sub)
    real, wmap = real[:n], {s: v for s, v in wmap.items() if s < n}
    torch.cuda.reset_peak_memory_stats()
    model, opt = at.fresh_model()
    teacher = teacher_model() if arm == "JD" else None
    torch.manual_seed(seed)
    times, stats, prev = {"english": [], "regional": []}, {"regional_samples": 0, "losses": [], "kd": []}, time.monotonic()
    for s, x, y, is_r in at.loader(at.AdaptSteps(windows, real, rr["rows"], wmap)):
        r = kd_update(model, teacher, opt, int(s), x, y, J.KD_LAMBDA) if teacher is not None else at.update(model, opt, int(s), x, y)
        torch.cuda.synchronize()
        now = time.monotonic()
        if int(s) >= 8:
            times["regional" if is_r else "english"].append(now - prev)
        prev = now
        stats["regional_samples"] += int(is_r)
        stats["losses"].append(None if r["skipped"] else r["loss"])
        if "kd" in r:
            stats["kd"].append(r["kd"])
    rec = {"arm": arm, "seed": seed, "n": n, "regional_steps": sorted(wmap),
           "s_per_english_update": float(np.mean(times["english"])) if times["english"] else None,
           "s_per_regional_update": float(np.mean(times["regional"])) if times["regional"] else None,
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20, **stats, **wt.env_record()}
    rec["all_finite"] = all(v is None or math.isfinite(v) for v in stats["losses"] + stats["kd"])
    if run_root is not None:
        run = run_root / f"SMOKE_{arm}"
        run.mkdir(parents=True, exist_ok=True)
        pt.save({"state_dict": model.state_dict(), "epoch": 0, "seed": seed, "arm": arm, "smoke_updates": n},
                run / "final.pt")
        rec["final_sha256"], rec["final_bytes"] = rt.sha(run / "final.pt"), (run / "final.pt").stat().st_size
        J.write_json(run / "TRAIN_RESULT.json", {"run": run.name, "arm": arm, "seed": seed, "tag": "smoke",
                                                 "status": "SMOKE", "final": str(run / "final.pt"),
                                                 "final_sha256": rec["final_sha256"], "final_bytes": rec["final_bytes"]})
    J.write_json(out, rec)
    if not rec["all_finite"] or (arm != "C" and stats["regional_samples"] == 0) or (arm == "JD" and not stats["kd"]):
        raise SystemExit("smoke: non-finite loss, no regional sample or no distillation term")
    return rec


def _digest_state(model) -> str:
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _losses(model, opt, teacher, lam: float | None, steps) -> list:
    out = []
    for s, x, y, _ in steps:
        r = at.update(model, opt, int(s), x, y) if lam is None else kd_update(model, teacher, opt, int(s), x, y, lam)
        out.append(None if r["skipped"] else r["loss"])
    return out


def kd_check(n: int, out: Path, a) -> dict:
    """Distillation correctness on staged windows (owner clarification 07:59Z): JD with lambda 0 reproduces J's first n
    updates (|loss difference| <= 1e-4; GPU kernels are not bitwise deterministic), the teacher's weights are
    unchanged by training, and the distillation gradient of the first crop with background pixels is finite and > 0."""
    import torch
    w_table, i_table = (staged(t, wd, md) for t, wd, md in ((J.read_json(a.w_table), a.w_win, a.w_mask),
                                                            (J.read_json(a.i_table), a.i_win, a.i_mask)))
    seed = J.STEP0_SEED
    sub = {"smoke": True, "wales": {str(seed): sorted(w_table["monuments"])},
           "ireland": {str(seed): sorted(i_table["monuments"])}}
    pt.verify_start()
    windows = plans.load_windows()[0]
    rr = region_rows(w_table, a.w_win, a.w_mask, i_table, a.i_win, a.i_mask)
    real, wmap = epoch_inputs("J", seed, 0, plans.load_plan(seed), rr, sub)
    real, wmap = real[:n], {s: v for s, v in wmap.items() if s < n}
    steps = lambda: at.loader(at.AdaptSteps(windows, real, rr["rows"], wmap))  # noqa: E731
    teacher = teacher_model()
    before = _digest_state(teacher)
    losses = {}
    for label, lam in (("J", None), ("JD_lambda0", 0.0)):
        model, opt = at.fresh_model()
        torch.manual_seed(seed)
        losses[label] = _losses(model, opt, teacher, lam, steps())
        del model, opt
    model, _ = at.fresh_model()
    grad, step = None, None
    for s, x, y, _ in steps():
        if not bool((y == 0).any()):
            continue
        xin = x[None].cuda()
        z = model(xin)["out"][0, 1]
        with torch.no_grad():
            zt = teacher(xin)["out"][0, 1]
        model.zero_grad(set_to_none=True)
        kd_term(z, zt, y.cuda()).backward()
        grad = float(torch.sqrt(sum((q.grad.detach() ** 2).sum() for q in model.parameters() if q.grad is not None)))
        step = int(s)
        break
    diffs = [abs(u - v) for u, v in zip(losses["J"], losses["JD_lambda0"]) if u is not None and v is not None]
    rec = {"n": n, "regional_steps": sorted(wmap), "losses": losses, "max_abs_loss_diff": max(diffs) if diffs else None,
           "teacher_unchanged": before == _digest_state(teacher), "teacher_sha": before, "kd_grad_norm": grad,
           "kd_grad_step": step, **wt.env_record()}
    same_skips = [u is None for u in losses["J"]] == [v is None for v in losses["JD_lambda0"]]
    rec["pass"] = bool(diffs and same_skips and rec["max_abs_loss_diff"] <= 1e-4 and rec["teacher_unchanged"]
                       and grad is not None and math.isfinite(grad) and grad > 0)
    J.write_json(out, rec)
    if not rec["pass"]:
        raise SystemExit("kd-check failed: lambda 0 differs from J, teacher changed or no finite distillation gradient")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("step0", "train", "smoke", "kd-check", "plan-check"))
    ap.add_argument("args", nargs="*")
    for k in ("w-table", "w-win", "w-mask", "i-table", "i-win", "i-mask", "run-root", "subsets"):
        ap.add_argument(f"--{k}", type=Path)
    a = ap.parse_args()
    if a.cmd in ("step0", "train", "smoke", "kd-check"):
        J.torch_flags()
    if a.cmd == "step0":
        print(json.dumps(step0(Path(a.args[0]))))
    elif a.cmd == "train":
        r = train(int(a.args[0]), a.args[1], a.w_table, a.w_win, a.w_mask, a.i_table, a.i_win, a.i_mask, a.run_root,
                  a.subsets or J.SUBSETS)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "step0_loss", "retention")}))
    elif a.cmd == "smoke":
        print(json.dumps(smoke(a.args[0], int(a.args[1]), Path(a.args[2]), a, a.run_root)))
    elif a.cmd == "kd-check":
        r = kd_check(int(a.args[0]), Path(a.args[1]), a)
        print(json.dumps({k: r[k] for k in ("max_abs_loss_diff", "teacher_unchanged", "kd_grad_norm", "pass")}))
    else:
        w_table, i_table = _tables(a)
        sub = load_subsets(w_table, i_table, a.subsets or J.SUBSETS)
        rr = region_rows(w_table, Path("w_win"), Path("w_mask"), i_table, Path("i_win"), Path("i_mask"))
        seed = int(a.args[0])
        print(json.dumps({arm: plan_summary(arm, seed, rr, sub) for arm in ("J", "JD")}, indent=1))


if __name__ == "__main__":
    main()
