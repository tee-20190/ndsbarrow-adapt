"""D1-COLLAPSE-1 trainer (PACKET.md): ADAPT-MATCH-0's AO_L / BARROW_L setting with a recipe switch, dense per-update
logging and the online collapse detector.

Unchanged from ADAPT-MATCH-0 (imported read-only from match_train, frozen 351b5961): the Lower Saxony rows and pools
(ls_context), the all-local plan (match_schedule.local_plan via match_train.schedule), the draw digest, the start tars
(start_weights, size and sha256 checked), the model loader and frozen-BatchNorm train mode and the fresh AdamW
(match_train.fresh_model), the loader (adapt_train AdaptSteps + loader: void policy V, 512 px crops, +-160 px
offsets, dihedral, batch 1), t_index, save, resume and environment records, fp32 with TF32 off.

Recipes (the only differences):
  OLD  loss BCE + 1 - soft Dice on every window; lr nds_schedule.lr_at (cosine 1e-4 -> 1e-6, no warm-up). Probe P0.
  F1   as OLD, but a window whose valid target has no positive pixel is scored with BCE only.
  F2   F1 + lr = (cosine 5e-5 -> 1e-6 over 5,116 updates) x min(1, (t + 1) / 256).
Per update (curves.jsonl): role, lr, BCE, Dice, sum p, sum g, valid pixels, max / mean logit, the analytic
logit-space BCE and Dice gradient sums and L2 norms (detached; the Dice term's even when the recipe omits it), the
old-form loss (BCE + 1 - Dice) and the loss used. Online detector (PACKET): at t >= 255 over the trailing 256 updates
ON iff mean positive-window Dice < 0.05 and mean background-window sum p < 1.0; it records, it never intervenes.

  python -B d1_train.py plan-check SEED OUT.json --ls-table T --ls-win W --ls-mask M      (CPU; no forward)
  python -B d1_train.py step0 OUT.json                     # HR1/JOINT-1 step-0 environment gate (as ADAPT-MATCH-0)
  python -B d1_train.py train NAME --ls-table T --ls-win W --ls-mask M --run-root R       (NAME = D1<recipe>_<arm>_<seed>)
  python -B d1_train.py detector CURVES.jsonl              # recompute the detector record from a curves file
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_common as D  # noqa: E402

N = D.N
import match_train as MT  # noqa: E402  ADAPT-MATCH-0 trainer (frozen): rows, plan, starts, model
import match_schedule as MS  # noqa: E402
from match_train import JT1, NT, SCH, at, pt, rt, wt  # noqa: E402  frozen recipe modules as ADAPT-MATCH-0 uses them

STAT_KEYS = ("bce", "dice", "sum_p", "sum_g", "n_valid", "z_max", "z_mean", "g_bce_sum", "g_dice_sum", "g_bce_l2",
             "g_dice_l2")


# ---------------------------------------------------------------- recipe pieces (pure where possible)
def lr_at(recipe: str, t: int) -> float:
    if recipe in ("OLD", "F1"):
        return SCH.lr_at(t)
    if recipe != "F2":
        raise ValueError(recipe)
    if not 0 <= t < SCH.N_UPDATES:
        raise ValueError("update index outside the schedule")
    cos = SCH.LR_MIN + 0.5 * (D.F2_LR_MAX - SCH.LR_MIN) * (1 + math.cos(math.pi * t / (SCH.N_UPDATES - 1)))
    return cos * min(1.0, (t + 1) / D.WARMUP_UPDATES)


def uses_dice(recipe: str, sum_g: float) -> bool:
    return recipe == "OLD" or sum_g > 0


def terms(z, y):
    """BCE, soft Dice (r2_train.masked_loss's exact expressions) and detached diagnostics; None when all IGNORE."""
    import torch
    import torch.nn.functional as F
    valid = y != rt.IGNORE
    if not bool(valid.any()):
        return None
    zv = z[valid]
    g = (y[valid] == 1).float()
    bce = F.binary_cross_entropy_with_logits(zv, g)
    p = torch.sigmoid(zv)
    sp, sg, inter = p.sum(), g.sum(), (p * g).sum()
    dice = (2 * inter + 1.0) / (sp + sg + 1.0)
    with torch.no_grad():
        pd, n = p.detach(), zv.numel()
        den = sp.detach() + sg + 1.0
        g_bce = (pd - g) / n
        g_dice = -(2 * g * den - (2 * inter.detach() + 1.0)) / den ** 2 * pd * (1 - pd)
        stats = torch.stack([bce.detach(), dice.detach(), sp.detach(), sg, torch.tensor(float(n), device=z.device),
                             zv.detach().max(), zv.detach().mean(), g_bce.sum(), g_dice.sum(),
                             g_bce.norm(), g_dice.norm()]).tolist()
    return bce, dice, dict(zip(STAT_KEYS, stats))


def update(model, opt, recipe: str, t: int, x, y, device: str = "cuda") -> dict:
    """nds_train.update with the recipe's loss and learning rate; the same zero_grad / forward / backward / step."""
    lr = lr_at(recipe, t)
    for grp in opt.param_groups:
        grp["lr"] = lr
    opt.zero_grad(set_to_none=True)
    z = model(x[None].to(device, non_blocking=True))["out"][0, 1]
    res = terms(z, y.to(device, non_blocking=True))
    if res is None:
        return {"skipped": True, "lr": lr}
    bce, dice, st = res
    with_dice = uses_dice(recipe, st["sum_g"])
    loss = bce + (1 - dice) if with_dice else bce
    loss.backward()
    opt.step()
    return {"skipped": False, "lr": lr, "loss_old": st["bce"] + 1 - st["dice"],
            "loss_used": st["bce"] + (1 - st["dice"] if with_dice else 0.0), "dice_used": with_dice, **st}


# ---------------------------------------------------------------- online collapse detector
class Detector:
    """Trailing DETECT_WINDOW updates; ON iff mean positive Dice < DETECT_DICE and mean background sum p < DETECT_MASS.
    Skipped updates occupy their slot but contribute to neither mean. Defined from t = DETECT_WINDOW - 1."""

    def __init__(self) -> None:
        self.buf: deque = deque(maxlen=D.DETECT_WINDOW)
        self.first_on, self.n_on, self.on, self.t = None, 0, False, -1

    def push(self, t: int, role: str, rec: dict) -> bool:
        if t != self.t + 1:
            raise ValueError(f"detector: update {t} after {self.t}")
        self.t = t
        self.buf.append(None if rec.get("skipped") else (role, rec["dice"], rec["sum_p"]))
        self.on = False
        if len(self.buf) == D.DETECT_WINDOW:
            pos = [r[1] for r in self.buf if r and r[0] == "positive"]
            bg = [r[2] for r in self.buf if r and r[0] == "background"]
            self.on = bool(pos and bg and sum(pos) / len(pos) < D.DETECT_DICE and sum(bg) / len(bg) < D.DETECT_MASS)
        if self.on:
            self.n_on += 1
            if self.first_on is None:
                self.first_on = t
        return self.on

    def record(self) -> dict:
        return {"window": D.DETECT_WINDOW, "dice_below": D.DETECT_DICE, "background_mass_below": D.DETECT_MASS,
                "first_on": self.first_on, "updates_on": self.n_on, "on_at_end": self.on, "last_update": self.t}


def detector_from_curves(path: Path) -> dict:
    det = Detector()
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            det.push(r["t"], r["role"], r)
    return det.record()


# ---------------------------------------------------------------- epoch and run
def new_sums() -> dict:
    return {f"{k}_{m}": 0.0 for k in MS.ROLES for m in ("loss_old", "loss_used", "dice", "sum_p")} | \
        {f"n_{k}": 0 for k in MS.ROLES} | {"skipped": 0}


def epoch_pass(recipe: str, arm: str, seed: int, epoch: int, c: dict, name: str, started: float,
               device: str = "cuda", limit: int | None = None) -> tuple[dict, list[dict]]:
    """match_train.epoch_pass with the recipe's update; returns the epoch row and the per-update records."""
    t0 = time.monotonic()
    rows = c["ctx"]["rows"]
    wmap = MT.schedule(seed, epoch, c["ctx"])
    n = MS.STEPS_PER_EPOCH if limit is None else limit
    wmap = {s: v for s, v in wmap.items() if s < n}
    if set(wmap) != set(range(n)):
        raise SystemExit("the all-local plan does not cover every step")
    sums, recs, det = new_sums(), [], c["detector"]
    for s, x, y, _ in at.loader(at.AdaptSteps([], [None] * n, rows, wmap)):
        role = rows[wmap[int(s)][0]]["role"]
        t = SCH.t_index(epoch, int(s))
        r = update(c["model"], c["opt"], recipe, t, x, y, device)
        rec = {"t": t, "epoch": epoch, "step": int(s), "role": role, **r}
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
                              **({k: r[k] for k in ("loss_old", "dice", "sum_p", "z_max")} if not r["skipped"] else {}),
                              "detector_on": det.on, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    row = {"epoch": epoch, "recipe": recipe, "lr_end": c["opt"].param_groups[0]["lr"],
           "skipped_all_ignore": sums["skipped"], "epoch_s": time.monotonic() - t0,
           "updates": MT.role_counts(wmap, rows), "lower_saxony_updates": len(wmap),
           **{f"mean_{k}_loss": (sums[f"{k}_loss_old"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None) for k in MS.ROLES},
           **{f"mean_{k}_{m}": (sums[f"{k}_{m}"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None)
              for k in MS.ROLES for m in ("loss_used", "dice", "sum_p")},
           "detector_on_at_epoch_end": det.on, "draw_digest": MT.draw_digest(wmap, rows)}
    return row, recs


def parse_name(name: str) -> dict:
    for i in D.SESSIONS:
        for r in D.session_runs(i):
            if r["name"] == name:
                return r
    raise SystemExit(f"{name} is not a D1 run")


def append_lines(path: Path, recs: list[dict]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def train(name: str, paths: dict, run_root: Path) -> dict:
    """Resumable per epoch (last.pt + the epoch's curves), as match_train.train; final weights only."""
    import torch
    spec = parse_name(name)
    recipe, arm, seed = spec["recipe"], spec["arm"], spec["seed"]
    NT.check_constants()
    c = {"ctx": MT.ls_context(paths)}
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    last, log_path, curves = run / "last.pt", run / "epochs.jsonl", run / "curves.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return N.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    c["model"], c["opt"] = MT.fresh_model(arm)
    torch.manual_seed(seed)
    done = wt._resume(c["model"], c["opt"], last)
    if done == 0:
        for p in (log_path, curves):
            p.unlink(missing_ok=True)
    c["detector"] = Detector()
    if done:                                               # replay the finished epochs' curves into the detector
        for ln in curves.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            c["detector"].push(r["t"], r["role"], r)
    for epoch in MS.EPOCHS:
        if epoch < done:
            continue
        row, recs = epoch_pass(recipe, arm, seed, epoch, c, name, started)
        append_lines(curves, recs)
        if epoch == MS.EPOCHS[-1]:
            final = run / "final.pt"
            pt.save({"state_dict": c["model"].state_dict(), "epoch": epoch, "seed": seed, "arm": arm,
                     "recipe": recipe, "start": D.STARTS[arm]["sha256"]}, final)
            row["final_sha256"] = rt.sha(final)
        pt.save({"state_dict": c["model"].state_dict(), "optimizer": c["opt"].state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        append_lines(log_path, [row])
        print(json.dumps({"run": name, **row}), flush=True)
    rows_log = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    det = detector_from_curves(curves)
    if det != c["detector"].record():
        raise SystemExit("detector: online record differs from the curves file")
    final = run / "final.pt"
    result = {"run": name, "recipe": recipe, "arm": arm, "seed": seed, "status": "COMPLETE", "final": str(final),
              "final_sha256": rt.sha(final), "final_bytes": final.stat().st_size, "epochs": rows_log,
              "detector": det, "curves_sha256": N.sha256_file(curves),
              "plan_summary": MT.plan_summary(seed, c["ctx"]),
              "start": {"registry_id": D.STARTS[arm]["registry_id"], "path": D.STARTS[arm]["path"],
                        "sha256": D.STARTS[arm]["sha256"]},
              "updates": MS.N_UPDATES, "lower_saxony_updates": MS.N_UPDATES,
              "lr": {"OLD": "nds_schedule.lr_at (cosine 1e-4 -> 1e-6, no warm-up)", "F1": "as OLD",
                     "F2": "cosine 5e-5 -> 1e-6 x min(1, (t + 1) / 256)"}[recipe],
              "loss": {"OLD": "BCE + 1 - Dice on every window", "F1": "BCE + 1 - Dice; BCE only on empty targets",
                       "F2": "as F1"}[recipe],
              "tables_sha256": {"ls_table": N.sha256_file(paths["ls_table"])},
              "script_sha256": N.sha256_file(Path(__file__)), "common_sha256": N.sha256_file(Path(D.__file__)),
              "match_train_sha256": N.sha256_file(Path(MT.__file__)),
              "schedule_sha256": N.sha256_file(Path(MS.__file__)), "nds_schedule_sha256": N.sha256_file(Path(SCH.__file__)),
              "frozen_trainer_sha256": N.sha256_file(Path(at.__file__)), "wall_s": time.monotonic() - started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
              "attribution": D.ATTRIBUTION, **(wt.env_record() if torch.cuda.is_available() else {})}
    N.write_json(run / "TRAIN_RESULT.json", result)
    last.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------- plan-check (CPU, no forward)
def plan_check(seed: int, paths: dict, out: Path) -> dict:
    """match_train.plan_check's checks for a D1 seed (the plan is recipe-independent and identical for both arms)."""
    if seed not in D.SEEDS:
        raise SystemExit(f"plan-check: seed {seed} outside the D1 design")
    NT.check_constants()
    ctx = MT.ls_context(paths)
    rows, pool, bg = ctx["rows"], ctx["pos"], ctx["bg"]
    plan = MS.run_plan(seed, pool, bg)
    bad = []
    for e in MS.EPOCHS:
        wmap = plan[e]
        if set(wmap) != set(range(MS.STEPS_PER_EPOCH)):
            bad.append(f"epoch {e}: not every step is a Lower Saxony step")
        wrong = [s for s, v in wmap.items() if rows[v[0]]["role"] != MS.ROLES[s % 2] or rows[v[0]]["region"] != MT.LS_REGION]
        bad += [f"epoch {e}: {len(wrong)} steps break the positive/background alternation"] if wrong else []
    t_idx = sorted(SCH.t_index(e, s) for e in MS.EPOCHS for s in plan[e])
    bad += [] if t_idx == list(range(MS.N_UPDATES)) else ["update indices are not 0..5,115 once each"]
    bad += MS.pass_complete(plan, pool, bg)
    exp = MS.exposure(plan, pool, bg)
    want = MS.expected_exposure(len(pool), len(bg))
    bad += [f"{r}: exposure {exp[r]['histogram']} != {want[r]}" for r in MS.ROLES if exp[r]["histogram"] != want[r]]
    summ = MT.plan_summary(seed, ctx)
    rec = {"schema": "d1_plan_check_v1", "seed": seed, "status": "FAIL" if bad else "PASS", "violations": bad,
           "updates": MS.N_UPDATES, "pool": {"positive": len(pool), "background": len(bg)}, "exposure": exp,
           "lr": {r: {"t0": lr_at(r, 0), "t255": lr_at(r, 255), "t_mid": lr_at(r, MS.N_UPDATES // 2),
                      "t_last": lr_at(r, MS.N_UPDATES - 1)} for r in D.RECIPES},
           "plan_summary": summ, "plan_digest": N.digest([e["draw_digest"] for e in summ["epochs"]]),
           "ls_table_sha256": N.sha256_file(paths["ls_table"]), "utc": N.utc(), "attribution": D.ATTRIBUTION}
    N.write_json(out, rec)
    if bad:
        raise SystemExit(f"plan-check failed: {bad}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan-check", "step0", "train", "detector"))
    ap.add_argument("args", nargs="*")
    for k in ("ls-table", "ls-win", "ls-mask", "run-root"):
        ap.add_argument(f"--{k}", type=Path)
    a = ap.parse_args()
    if a.cmd in ("step0", "train"):
        JT1.J.torch_flags()
    paths = {k: getattr(a, k) for k in ("ls_table", "ls_win", "ls_mask") if getattr(a, k, None) is not None}
    if a.cmd == "plan-check":
        r = plan_check(int(a.args[0]), paths, Path(a.args[1]))
        print(json.dumps({k: r[k] for k in ("seed", "status", "plan_digest")}))
    elif a.cmd == "step0":
        print(json.dumps(JT1.step0(Path(a.args[0]))))
    elif a.cmd == "train":
        r = train(a.args[0], paths, a.run_root)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "wall_s", "detector")}))
    else:
        print(json.dumps(detector_from_curves(Path(a.args[0]))))


if __name__ == "__main__":
    main()
