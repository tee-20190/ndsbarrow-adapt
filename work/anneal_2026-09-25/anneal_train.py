"""PACKET.md: matched continuation of CONT e41 that differs only in the learning-rate schedule. One arm per call.

Everything but the per-step learning rate is r2_train.py (inputs, crops, dihedral flips, loss, frozen BatchNorm,
sampling plan) and cont_train.py's continuation recipe (fresh AdamW, lr 1e-4, wd 1e-4, batch 1, fp32). Plan epochs
42-45 of the R2 seed sequence (4 x 2,558 steps), identical order and data in both arms.
  CONST   lr = 1e-4 at every optimizer step
  ANNEAL  lr = 1e-4 * 0.5 * (1 + cos(pi * n / N)), n = optimizer steps taken, N = 4 * 2,558 (cosine to zero)
Both arms use the same optimizer wrapper, so the code path is identical. Weights are saved after every epoch;
last.pt (with optimizer state and step count) makes a run resumable. GPU wall time goes to the packet ledger.

  python -B work/anneal_2026-09-25/anneal_train.py CONST
  python -B work/anneal_2026-09-25/anneal_train.py ANNEAL
(WSL adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
sys.path.insert(0, str(ROOT / "work/aim_train_audit_2026-09-24"))
import cont_train as ct  # noqa: E402
import r2_train as rt  # noqa: E402

OUT = ROOT / "outputs/anneal_2026-09-25"
START = ROOT / "outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt"
START_SHA256 = "91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383"
FIRST_EPOCH, N_EPOCHS = 42, 4
LEDGER = OUT / "GPU_TIME.json"
CEILING_S = 3.5 * 3600
ARMS = ("CONST", "ANNEAL")


def schedule(arm: str, total: int):
    if arm == "CONST":
        return lambda n: rt.LR
    if arm == "ANNEAL":
        return lambda n: rt.LR * 0.5 * (1 + math.cos(math.pi * min(n, total) / total))
    raise SystemExit(f"unknown arm {arm}")


class ScheduledAdamW(torch.optim.AdamW):
    """AdamW whose learning rate is set from the step count immediately before every step."""

    def __init__(self, params, lr_at, **kw) -> None:
        super().__init__(params, **kw)
        self.lr_at, self.n = lr_at, 0

    def step(self, closure=None):
        lr = self.lr_at(self.n)
        for g in self.param_groups:
            g["lr"] = lr
        self.n += 1
        return super().step(closure)


def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"runs": [], "ceiling_s": CEILING_S}


def charge(kind: str, wall_s: float, note: dict) -> None:
    led = ledger()
    led["runs"].append({"kind": kind, "wall_s": wall_s,
                        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **note})
    led["total_wall_s"] = sum(r["wall_s"] for r in led["runs"])
    led["ceiling_s"] = CEILING_S
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2), encoding="utf-8")


def remaining_s() -> float:
    return CEILING_S - sum(r["wall_s"] for r in ledger()["runs"])


MIN_FREE_BYTES = 3 * 10**9  # C: is shared and nearly full; stop cleanly instead of filling it


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


def train(arm: str) -> dict:
    if arm not in ARMS:
        raise SystemExit(__doc__)
    if ct.sha(START) != START_SHA256 or ct.sha(Path(rt.__file__)) != ct.R2_TRAIN_SHA256:
        raise RuntimeError("start checkpoint or r2_train.py changed")
    windows, fit_pos, fit_neg, hold = ct.load_windows(ct.MANIFESTS["CONT"])
    ct.check_matched(windows)
    rt.verify_masks(windows)
    if any(windows[i]["holdout"] for i in fit_pos + fit_neg):
        raise RuntimeError("holdout isolation violated")
    total = N_EPOCHS * 2 * len(fit_pos)
    run = OUT / "training" / arm
    run.mkdir(parents=True, exist_ok=True)
    budget = remaining_s()
    if budget <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    started = time.monotonic()
    model = rt.load_model(START)
    opt = ScheduledAdamW(model.parameters(), schedule(arm, total), lr=rt.LR, weight_decay=rt.WD)
    last, log_path = run / "last.pt", run / "epochs.jsonl"
    first = FIRST_EPOCH
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=True)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        opt.n = int(state["opt_steps"])
        first = state["epoch"] + 1
    log = lambda d: print(json.dumps({"arm": arm, "lr": opt.param_groups[0]["lr"], **d}), flush=True)
    stopped = False
    try:
        for epoch in range(first, FIRST_EPOCH + N_EPOCHS):
            t0 = time.monotonic()
            n, mean_loss, stopped = rt.train_steps(model, opt, windows, rt.epoch_plan(epoch, fit_pos, fit_neg),
                                                   budget, started, log)
            if stopped:
                break
            ev = rt.evaluate(model, windows, hold)
            ck = run / f"epoch{epoch}.pt"
            save(model, opt, epoch, ck, with_optimizer=False)
            save(model, opt, epoch, last, with_optimizer=True)
            row = {"epoch": epoch, "steps": n, "opt_steps_total": opt.n, "lr_end": opt.param_groups[0]["lr"],
                   "train_loss": mean_loss, **ev, "checkpoint": ck.relative_to(ROOT).as_posix(),
                   "checkpoint_sha256": ct.sha(ck), "epoch_s": time.monotonic() - t0,
                   "run_elapsed_s": time.monotonic() - started}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            log(row)
    finally:
        charge(f"train_{arm}", time.monotonic() - started, {"stopped_by_ceiling": stopped})
    result = {"arm": arm, "status": "CEILING_REACHED" if stopped else "COMPLETE", "planned_opt_steps": total,
              "opt_steps": opt.n, "manifest_sha256": ct.sha(ct.MANIFESTS["CONT"]), "start_sha256": START_SHA256,
              "script_sha256": ct.sha(Path(__file__)), "gpu_s_total": sum(r["wall_s"] for r in ledger()["runs"])}
    (run / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    print(json.dumps(train(sys.argv[1])))
