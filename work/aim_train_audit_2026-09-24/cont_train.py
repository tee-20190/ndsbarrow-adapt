"""PREREG.md B: matched continuation of frozen R2 (epoch 38) for 6 epochs (plan epochs 40-45), one arm per call.

Everything but the mask manifest is r2_train.py: inputs, crops, dihedral flips, loss, BN handling, sampling plan.
Fresh AdamW (best.pt holds no optimizer state). Weights are saved after every epoch; last.pt (with optimizer) makes
the run resumable. GPU wall time is charged to a ledger shared by every PREREG B stage (4.5 h ceiling).

  python -B work/aim_train_audit_2026-09-24/cont_train.py CONT
  python -B work/aim_train_audit_2026-09-24/cont_train.py AIMX
(WSL adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
import r2_train as rt  # noqa: E402

OUT = ROOT / "outputs/aim_train_audit_2026-09-24"
START = ROOT / "outputs/campaign_2026-09-22/training/R2_adaf_hrnet_slrm/best.pt"
START_SHA256 = "fbd1cbacdf10c2d2806ec6dc7662196ebeed37416b908bad8f076df3113e7392"
R2_TRAIN_SHA256 = "11ef821bce4ff00927880005a658002409ac32742aa28272b0720ac849bc5ddd"
MANIFESTS = {"CONT": ROOT / "work/campaign_2026-09-22/R2_MASKS_MANIFEST_PRIVATE.json",
             "AIMX": OUT / "AIMX_MASKS_MANIFEST_PRIVATE.json"}
FIRST_EPOCH, N_EPOCHS = 40, 6
LEDGER = OUT / "GPU_TIME.json"
CEILING_S = 4.5 * 3600


def sha(path: Path) -> str:
    return rt.sha(path)


def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"runs": [], "ceiling_s": CEILING_S}


def charge(kind: str, wall_s: float, note: dict) -> None:
    led = ledger()
    led["runs"].append({"kind": kind, "wall_s": wall_s, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        **note})
    led["total_wall_s"] = sum(r["wall_s"] for r in led["runs"])
    led["ceiling_s"] = CEILING_S
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2), encoding="utf-8")


def remaining_s() -> float:
    return CEILING_S - sum(r["wall_s"] for r in ledger()["runs"])


def load_windows(manifest: Path) -> tuple[list[dict], list[int], list[int], list[int]]:
    """r2_train.load_windows for an explicit manifest (same sort and index rules)."""
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if m["summary"]["gate"] != "PASS":
        raise RuntimeError("mask gate did not pass")
    windows = sorted(m["windows"], key=lambda w: w["id"])
    fit_pos = [i for i, w in enumerate(windows) if w["role"] == "positive" and not w["holdout"]]
    fit_neg = [i for i, w in enumerate(windows) if w["role"] == "negative" and not w["holdout"]]
    hold = [i for i, w in enumerate(windows) if w["holdout"]]
    return windows, fit_pos, fit_neg, hold


def check_matched(windows: list[dict]) -> None:
    """Both arms must index the same windows, roles and holdout flags; only fitting mask files may differ."""
    ref, _, _, _ = load_windows(MANIFESTS["CONT"])
    if [(w["id"], w["role"], w["holdout"]) for w in windows] != [(w["id"], w["role"], w["holdout"]) for w in ref]:
        raise RuntimeError("arm manifest does not match the R2 window index")
    if any(w["holdout"] and w["mask_sha256"] != r["mask_sha256"] for w, r in zip(windows, ref)):
        raise RuntimeError("holdout masks must stay unchanged")


def save(model, opt, epoch: int, path: Path, with_optimizer: bool) -> None:
    payload = {"state_dict": model.state_dict(), "epoch": epoch}
    if with_optimizer:
        payload["optimizer"] = opt.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(arm: str) -> dict:
    if sha(START) != START_SHA256 or sha(Path(rt.__file__)) != R2_TRAIN_SHA256:
        raise RuntimeError("frozen start checkpoint or r2_train.py changed")
    windows, fit_pos, fit_neg, hold = load_windows(MANIFESTS[arm])
    check_matched(windows)
    rt.verify_masks(windows)
    run = OUT / "training" / arm
    run.mkdir(parents=True, exist_ok=True)
    budget = remaining_s()
    if budget <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    started = time.monotonic()
    model = rt.load_model(START)
    opt = torch.optim.AdamW(model.parameters(), lr=rt.LR, weight_decay=rt.WD)
    last, log_path = run / "last.pt", run / "epochs.jsonl"
    first = FIRST_EPOCH
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=True)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        first = state["epoch"] + 1
    log = lambda d: print(json.dumps({"arm": arm, **d}), flush=True)
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
            row = {"epoch": epoch, "steps": n, "train_loss": mean_loss, **ev, "checkpoint": ck.relative_to(ROOT).as_posix(),
                   "checkpoint_sha256": sha(ck), "epoch_s": time.monotonic() - t0,
                   "run_elapsed_s": time.monotonic() - started}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            log(row)
    finally:
        charge(f"train_{arm}", time.monotonic() - started, {"stopped_by_ceiling": stopped})
    result = {"arm": arm, "status": "CEILING_REACHED" if stopped else "COMPLETE",
              "manifest_sha256": sha(MANIFESTS[arm]), "start_sha256": START_SHA256,
              "script_sha256": sha(Path(__file__)), "gpu_s_total": sum(r["wall_s"] for r in ledger()["runs"])}
    (run / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    print(json.dumps(train(sys.argv[1])))
