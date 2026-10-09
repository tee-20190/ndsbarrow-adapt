"""PACKET.md: background scale-up (NEG) against a matched control, both continuing CONT e41 with the ANNEAL recipe.

Recipe = work/anneal_2026-09-25/anneal_train.py, ANNEAL arm, unchanged and imported where possible: start CONT e41,
fresh AdamW (lr 1e-4, wd 1e-4) through anneal_train.ScheduledAdamW with anneal_train.schedule("ANNEAL", 10,232)
(cosine to zero), batch 1, fp32, frozen BatchNorm, r2_train inputs / crops / dihedral flips / BCE + soft-Dice loss,
plan epochs 42-45 of the R2 seed sequence (r2_train.epoch_plan), weights after every epoch, declared final
checkpoint e45. Only the fitting-negative pool differs:
  ORIGINAL  the 741 fitting negatives of R2_MASKS_MANIFEST_PRIVATE.json (CTRL arm and the determinism gate)
  NEG       ORIGINAL plus the verified new fitting negatives of the full-cohort split (combined manifest, frozen at
            FREEZE stage B before any NEG step)
Every plan keeps one positive then one negative step (1:1). Positive windows, their crops and flips, and the crop and
flip draws of every negative step are identical across pools; only the window that fills a negative step changes
(r2_train.negative_at over the pool in window-id order). train_steps below is r2_train.train_steps' own code object
with only its `loader` global rebound to a loader that also finds the new raw windows.

  python -B work/bgscale_2026-09-26/bg_train.py gate   # Phase B gate 1: ORIGINAL pool, first 1,001 plan steps of
                                                       # epoch 42 against ANNEAL's recorded losses (exact equality)
  python -B work/bgscale_2026-09-26/bg_train.py NEG    # NEG arm, epochs 42-45
  python -B work/bgscale_2026-09-26/bg_train.py CTRL   # fresh control arm; refused unless the gate did not PASS
(WSL /home/tee-/adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
sys.path.insert(0, str(ROOT / "work/aim_train_audit_2026-09-24"))
sys.path.insert(0, str(ROOT / "work/anneal_2026-09-25"))
import anneal_train as at  # noqa: E402
import cont_train as ct  # noqa: E402
import r2_train as rt  # noqa: E402

OUT = ROOT / "outputs/bgscale_2026-09-26"
NEG_RAW = OUT / "neg_raw"
NEG_MASKS = OUT / "neg_masks"
COMBINED = OUT / "BGSCALE_MASKS_MANIFEST_PRIVATE.json"
FREEZE = HERE / "FREEZE.json"
GATE_FILE = OUT / "GATE_DETERMINISM.json"
POOL_PLAN = HERE / "NEG_POOL_PLAN_PRIVATE.json"
START, START_SHA256 = at.START, at.START_SHA256
ANNEAL_E45 = ROOT / "outputs/anneal_2026-09-25/training/ANNEAL/epoch45.pt"
ANNEAL_E45_SHA256 = "64e4ebfa0ba311687ad918e6c19cc2943e55973f3be5664294c2475973386bd1"
ANNEAL_LOG = ROOT / "outputs/anneal_2026-09-25/train_ANNEAL.log"
R2_MANIFEST = ct.MANIFESTS["CONT"]
PINNED = {  # frozen upstream code and inputs this trainer relies on (hashes from their own FREEZE files)
    "work/campaign_2026-09-22/r2_train.py": ct.R2_TRAIN_SHA256,
    "work/aim_train_audit_2026-09-24/cont_train.py": "572502ce41b522d11c0dd894e6d3113591d32c595843898f9327e6ab446b05bb",
    "work/anneal_2026-09-25/anneal_train.py": "53dda3edef167a45d45db32663d3e9ad5f69f4c2e1b9a77648579de70514f728",
    "work/campaign_2026-09-22/R2_MASKS_MANIFEST_PRIVATE.json":
        "f44a36bd83eb73083f87ff434efed135fcb35f0c398e942c28192ceccecb3acc",
}
FIRST_EPOCH, N_EPOCHS = at.FIRST_EPOCH, at.N_EPOCHS  # 42, 4
FINAL_EPOCH = FIRST_EPOCH + N_EPOCHS - 1  # declared checkpoint e45
N_FIT_POS = 1279
TOTAL_STEPS = N_EPOCHS * 2 * N_FIT_POS  # 10,232 optimizer steps, as ANNEAL
GATE_STEPS = 1001  # plan steps 0..1000 of epoch 42: ANNEAL logged losses at steps 0, 200, ..., 1000 (six records)
ARMS = {"NEG": "NEG", "CTRL": "ORIGINAL"}
LEDGER = OUT / "GPU_TIME.json"
CEILING_S = 5.0 * 3600
MIN_FREE_BYTES = at.MIN_FREE_BYTES  # 3 GB floor before each checkpoint write (storage_guard is the outer guard)


def sha(path: Path) -> str:
    return rt.sha(path)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_pins() -> None:
    bad = [p for p, h in PINNED.items() if sha(ROOT / p) != h]
    if sha(START) != START_SHA256:
        bad.append("CONT e41")
    if bad:
        raise RuntimeError(f"frozen inputs changed: {bad}")


# ---------------------------------------------------------------- GPU ledger (5.0 h ceiling, every Phase B stage)
def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"runs": [], "ceiling_s": CEILING_S}


def charge(kind: str, wall_s: float, note: dict) -> None:
    led = ledger()
    led["runs"].append({"kind": kind, "wall_s": wall_s,
                        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **note})
    led["total_wall_s"] = sum(r["wall_s"] for r in led["runs"])
    led["ceiling_s"] = CEILING_S
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, indent=2), encoding="utf-8")
    tmp.replace(LEDGER)


def remaining_s() -> float:
    return CEILING_S - sum(r["wall_s"] for r in ledger()["runs"])


# ---------------------------------------------------------------- pools
def stage_b() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8")) if FREEZE.exists() else {}
    if "stage_B" not in freeze:
        raise RuntimeError("FREEZE stage B (pool, masks, cache) is missing; no NEG step may run")
    return freeze["stage_B"]


def check_combined(windows: list[dict], fit_pos: list[int], hold: list[int]) -> None:
    """Combined manifest = R2 manifest window records unchanged, plus planned new fitting negatives only."""
    ref, ref_pos, _, ref_hold = ct.load_windows(R2_MANIFEST)
    by_id = {w["id"]: w for w in windows}
    if len(by_id) != len(windows):
        raise RuntimeError("duplicate window ids in the combined manifest")
    if any(by_id.get(w["id"]) != w for w in ref):
        raise RuntimeError("an R2 window record differs in the combined manifest")
    planned = {w["id"] for w in json.loads(POOL_PLAN.read_text(encoding="utf-8"))["windows"]}
    ref_ids = {w["id"] for w in ref}
    new = [w for w in windows if w["id"] not in ref_ids]
    if not new or any(w["role"] != "negative" or w["holdout"] or w["id"] not in planned for w in new):
        raise RuntimeError("combined manifest adds something other than planned fitting negatives")
    if any(not w.get("raw_path", "").startswith("outputs/bgscale_2026-09-26/neg_raw/") for w in new):
        raise RuntimeError("new negative without a neg_raw path")
    ids = lambda ws, ix: [ws[i]["id"] for i in ix]  # noqa: E731
    if ids(windows, fit_pos) != ids(ref, ref_pos) or ids(windows, hold) != ids(ref, ref_hold):
        raise RuntimeError("positive or holdout windows differ from R2")


def load_pool(pool: str) -> tuple[list[dict], list[int], list[int], list[int]]:
    if pool == "ORIGINAL":
        windows, fit_pos, fit_neg, hold = ct.load_windows(R2_MANIFEST)
        ct.check_matched(windows)
    elif pool == "NEG":
        if sha(COMBINED) != stage_b()["sha256"]["outputs/bgscale_2026-09-26/BGSCALE_MASKS_MANIFEST_PRIVATE.json"]:
            raise RuntimeError("combined manifest differs from FREEZE stage B")
        windows, fit_pos, fit_neg, hold = ct.load_windows(COMBINED)
        check_combined(windows, fit_pos, hold)
    else:
        raise SystemExit(f"unknown pool {pool}")
    if any(windows[i]["holdout"] for i in fit_pos + fit_neg) or not all(windows[i]["holdout"] for i in hold):
        raise RuntimeError("holdout isolation violated")
    if len(fit_pos) != N_FIT_POS:
        raise RuntimeError(f"expected {N_FIT_POS} fitting positives, found {len(fit_pos)}")
    return windows, fit_pos, fit_neg, hold


def plan_ids(windows: list[dict], fit_pos: list[int], fit_neg: list[int], epoch: int) -> list[tuple]:
    """r2_train.epoch_plan with window indices replaced by ids (for pool comparisons)."""
    return [(windows[wi]["id"], dy, dx, k) for wi, dy, dx, k in rt.epoch_plan(epoch, fit_pos, fit_neg)]


def index_windows(records: list[dict]) -> tuple[list[dict], list[int], list[int], list[int]]:
    """cont_train.load_windows' sort and index rules for an in-memory window list."""
    windows = sorted(records, key=lambda w: w["id"])
    fit_pos = [i for i, w in enumerate(windows) if w["role"] == "positive" and not w["holdout"]]
    fit_neg = [i for i, w in enumerate(windows) if w["role"] == "negative" and not w["holdout"]]
    hold = [i for i, w in enumerate(windows) if w["holdout"]]
    return windows, fit_pos, fit_neg, hold


def planned_neg_pool() -> tuple[list[dict], list[int], list[int], list[int]]:
    """The NEG pool as it would stand if every planned window verifies (ids only; for Phase A analysis and tests)."""
    ref = json.loads(R2_MANIFEST.read_text(encoding="utf-8"))["windows"]
    new = [{"id": w["id"], "role": "negative", "holdout": False}
           for w in json.loads(POOL_PLAN.read_text(encoding="utf-8"))["windows"]]
    return index_windows(ref + new)


def draw_stats(windows: list[dict], fit_pos: list[int], fit_neg: list[int], original: set[str]) -> dict:
    """Which negative windows the four planned epochs actually draw (every odd plan step is a negative step)."""
    drawn = [windows[p[0]]["id"] for e in range(FIRST_EPOCH, FINAL_EPOCH + 1)
             for p in rt.epoch_plan(e, fit_pos, fit_neg)[1::2]]
    count: dict[str, int] = {}
    for i in drawn:
        count[i] = count.get(i, 0) + 1
    return {"pool": len(fit_neg), "negative_steps": len(drawn), "positive_steps": len(drawn),
            "unique_negatives_drawn": len(count), "unique_from_original_741": sum(i in original for i in count),
            "unique_new": sum(i not in original for i in count),
            "steps_on_original_741": sum(v for i, v in count.items() if i in original),
            "max_draws_per_window": max(count.values()), "windows_never_drawn": len(fit_neg) - len(count)}


# ---------------------------------------------------------------- loader and training steps
def raw_path(w: dict) -> Path:
    """R2 windows keep r2_train.RAW; new negatives carry their own repo-relative raw_path."""
    return ROOT / w["raw_path"] if "raw_path" in w else rt.RAW / f"{w['id']}.tif"


class BgSteps(rt.Steps):
    """r2_train.Steps with the raw window resolved per window; every other operation is r2_train's."""

    def __getitem__(self, i: int):
        wi, dy, dx, k = self.plan[i]
        w = self.windows[wi]
        img = rt.relief_window(raw_path(w))
        lab = np.load(ROOT / w["mask_path"])["label"]
        if img.shape != lab.shape:
            raise ValueError(f"{w['id']}: input {img.shape} vs mask {lab.shape}")
        img, lab = rt.dihedral(rt.crop(img, dy, dx), k), rt.dihedral(rt.crop(lab, dy, dx), k)
        return rt.to_input(img), torch.from_numpy(lab), wi


def loader(windows: list[dict], plan: list[tuple], workers: int = rt.WORKERS) -> DataLoader:
    return DataLoader(BgSteps(windows, plan), batch_size=1, shuffle=False, num_workers=workers,
                      pin_memory=True, prefetch_factor=4 if workers else None)


# r2_train.train_steps' own code object; only the name `loader` resolves to the loader above.
train_steps = types.FunctionType(rt.train_steps.__code__, dict(vars(rt), loader=loader), "train_steps")


def make_optimizer(model: torch.nn.Module) -> at.ScheduledAdamW:
    return at.ScheduledAdamW(model.parameters(), at.schedule("ANNEAL", TOTAL_STEPS), lr=rt.LR, weight_decay=rt.WD)


def save(model, opt, epoch: int, path: Path, with_optimizer: bool) -> None:
    if shutil.disk_usage(path.parent).free < MIN_FREE_BYTES:
        raise SystemExit(f"free disk below {MIN_FREE_BYTES / 1e9:.0f} GB; stop before writing {path.name}")
    payload = {"state_dict": model.state_dict(), "epoch": epoch}
    if with_optimizer:
        payload["optimizer"] = opt.state_dict()
        payload["opt_steps"] = opt.n
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------- determinism gate (reuse ANNEAL e45 as control?)
def recorded_gate_records(path: Path = ANNEAL_LOG, n_steps: int = GATE_STEPS) -> list[dict]:
    """ANNEAL's step records of its first epoch (e42) with plan step < n_steps, in log order."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if "epoch" in d:
            break
        if "step" in d and d["step"] < n_steps:
            out.append({k: d[k] for k in ("step", "lr", "loss", "bce", "dice")})
    return out


def compare_gate(recorded: list[dict], ours: list[dict]) -> dict:
    """Exact equality of step, learning rate, loss, BCE and Dice at every recorded step (no tolerance)."""
    rows, ok = [], len(recorded) == len(ours) and len(recorded) > 0
    for r, o in zip(recorded, ours):
        same = all(r[k] == o.get(k) for k in ("step", "lr", "loss", "bce", "dice"))
        ok &= same
        rows.append({"step": r["step"], "exact": same, "recorded_loss": r["loss"], "loss": o.get("loss"),
                     "abs_diff_loss": abs(r["loss"] - o["loss"]) if o.get("loss") is not None else None,
                     "lr_equal": r["lr"] == o.get("lr")})
    return {"verdict": "PASS" if ok else "FAIL", "compared": len(rows), "expected": len(recorded), "rows": rows}


def gate() -> dict:
    verify_pins()
    if GATE_FILE.exists():
        return json.loads(GATE_FILE.read_text(encoding="utf-8"))
    recorded = recorded_gate_records()
    if [r["step"] for r in recorded] != list(range(0, GATE_STEPS, 200)):
        raise RuntimeError("ANNEAL log does not hold the expected epoch-42 step records")
    windows, fit_pos, fit_neg, hold = load_pool("ORIGINAL")
    rt.verify_masks(windows)
    if remaining_s() <= 0:
        raise SystemExit("GPU ceiling reached")
    started = time.monotonic()
    model = rt.load_model(START)
    opt = make_optimizer(model)
    ours: list[dict] = []

    def log(d: dict) -> None:
        rec = {"arm": "GATE", "lr": opt.param_groups[0]["lr"], **d}
        ours.append({k: rec.get(k) for k in ("step", "lr", "loss", "bce", "dice")})
        print(json.dumps(rec), flush=True)

    try:
        n, mean_loss, stopped = train_steps(model, opt, windows, rt.epoch_plan(FIRST_EPOCH, fit_pos, fit_neg)[:GATE_STEPS],
                                            remaining_s(), started, log)
    finally:
        charge("gate", time.monotonic() - started, {"steps": GATE_STEPS})
    res = compare_gate(recorded, ours)
    res.update({"gate_steps": GATE_STEPS, "optimizer_steps": n, "stopped_by_ceiling": stopped,
                "control": "ANNEAL" if res["verdict"] == "PASS" else "CTRL",
                "rule": "PASS only if every recorded ANNEAL step record in plan steps 0..1000 of epoch 42 (steps 0, 200, "
                        "..., 1000) is reproduced exactly (step, lr, loss, bce, dice); PASS reuses ANNEAL e45 as the "
                        "control, anything else runs the fresh CTRL arm",
                "anneal_log_sha256": sha(ANNEAL_LOG), "script_sha256": sha(Path(__file__)),
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()})
    GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    GATE_FILE.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def control_arm() -> str:
    if not GATE_FILE.exists():
        raise RuntimeError("determinism gate has not run")
    return json.loads(GATE_FILE.read_text(encoding="utf-8"))["control"]


# ---------------------------------------------------------------- training
def train(arm: str) -> dict:
    if arm not in ARMS:
        raise SystemExit(__doc__)
    verify_pins()
    if arm == "CTRL" and control_arm() != "CTRL":
        raise SystemExit("the determinism gate passed: ANNEAL e45 is the control; CTRL is not run")
    windows, fit_pos, fit_neg, hold = load_pool(ARMS[arm])
    rt.verify_masks(windows)
    total = N_EPOCHS * 2 * len(fit_pos)
    if total != TOTAL_STEPS:
        raise RuntimeError("step count differs from ANNEAL")
    run = OUT / "training" / arm
    run.mkdir(parents=True, exist_ok=True)
    budget = remaining_s()
    if budget <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    started = time.monotonic()
    model = rt.load_model(START)
    opt = make_optimizer(model)
    last, log_path = run / "last.pt", run / "epochs.jsonl"
    first = FIRST_EPOCH
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=True)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        opt.n = int(state["opt_steps"])
        first = state["epoch"] + 1
    log = lambda d: print(json.dumps({"arm": arm, "lr": opt.param_groups[0]["lr"], **d}), flush=True)  # noqa: E731
    stopped = False
    try:
        for epoch in range(first, FIRST_EPOCH + N_EPOCHS):
            t0 = time.monotonic()
            n, mean_loss, stopped = train_steps(model, opt, windows, rt.epoch_plan(epoch, fit_pos, fit_neg),
                                                budget, started, log)
            if stopped:
                break
            ev = rt.evaluate(model, windows, hold)
            ck = run / f"epoch{epoch}.pt"
            save(model, opt, epoch, ck, with_optimizer=False)
            save(model, opt, epoch, last, with_optimizer=True)
            row = {"epoch": epoch, "steps": n, "opt_steps_total": opt.n, "lr_end": opt.param_groups[0]["lr"],
                   "train_loss": mean_loss, **ev, "checkpoint": ck.relative_to(ROOT).as_posix(),
                   "checkpoint_sha256": sha(ck), "epoch_s": time.monotonic() - t0,
                   "run_elapsed_s": time.monotonic() - started}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            log(row)
    finally:
        charge(f"train_{arm}", time.monotonic() - started, {"stopped_by_ceiling": stopped})
    manifest = COMBINED if ARMS[arm] == "NEG" else R2_MANIFEST
    result = {"arm": arm, "pool": ARMS[arm], "status": "CEILING_REACHED" if stopped else "COMPLETE",
              "planned_opt_steps": total, "opt_steps": opt.n, "fit_negatives": len(fit_neg),
              "manifest": manifest.relative_to(ROOT).as_posix(), "manifest_sha256": sha(manifest),
              "start_sha256": START_SHA256, "script_sha256": sha(Path(__file__)),
              "gpu_s_total": sum(r["wall_s"] for r in ledger()["runs"])}
    (run / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "gate":
        print(json.dumps({k: v for k, v in gate().items() if k != "rows"}))
    elif cmd in ARMS:
        print(json.dumps(train(cmd)))
    else:
        raise SystemExit(__doc__)
