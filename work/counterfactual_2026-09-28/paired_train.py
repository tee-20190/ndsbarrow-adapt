"""Experiment 1 v2 paired training (PROTOCOL.md sections 4 and 5.5): one arm per process.

Both arms: CONT e41 start, fresh AdamW (wd 1e-4), batch 1, fp32, frozen BatchNorm, r2_train real inputs / crops /
dihedral flips / masked BCE + soft Dice, plans.real_plan epochs 0-3, cosine lr per scheduled update (plans.lr_at).
Every fourth scheduled update carries an auxiliary example (plans.aux_scheduled). cf_v1.counterfactual.matched_update
performs the real and auxiliary backwards and ONE optimizer step; the control's auxiliary loss has zero gradient.

  python -B work/counterfactual_2026-09-28/paired_train.py smoke            # 32 updates per arm + 16-window inference
  python -B work/counterfactual_2026-09-28/paired_train.py train SEED ARM   # ARM = C | T (resumable)
(WSL /home/tee-/adaf-h2h-venv, repo root)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cf2  # noqa: E402
import ledger as lg  # noqa: E402
import plans  # noqa: E402

cf1, rt = cf2.cf1, cf2.rt
START = ROOT / "outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt"
START_SHA256 = "91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383"
AUX_WEIGHT = 0.25
ARMS = ("C", "T")
WORKERS, PREFETCH = 3, 4
MIN_FREE_BYTES = 3 * 10**9
LOG_EVERY = 200
SCORING_ESTIMATE_S, DIAGNOSTIC_ESTIMATE_S = 2.7 * 3600, 0.3 * 3600  # PROTOCOL.md 11 planning figures


# ---------------------------------------------------------------- data
def aux_tensors(windows: list[dict], ex: dict) -> tuple[torch.Tensor, ...]:
    """Clean and implanted model inputs, target and outside masks for one planned auxiliary example."""
    w = windows[ex["wi"]]
    if w["id"] != ex["window_id"]:
        raise RuntimeError("plan window index does not match its id")
    dtm, profile = cf2.read_native(rt.RAW / f"{w['id']}.tif")
    blocked = cf2.blocked_native(dtm, np.load(ROOT / w["mask_path"])["label"])
    aug, target, influence = cf2.apply(dtm, cf2.implant_from_dict(ex["implant"]), blocked)
    clean_in, finite, _ = cf2.relief_from_array(dtm, profile)
    aug_in, _, _ = cf2.relief_from_array(aug, profile)
    outside = finite & ~cf2.up2(influence) & ~cf2.up2(blocked)
    k = ex["k"]

    def tf(a: np.ndarray) -> np.ndarray:
        return rt.dihedral(cf2.crop_centre(a), k)

    tgt, out = tf(cf2.up2(target)), tf(outside)
    if not tgt.any() or not out.any() or (tgt & out).any():
        raise RuntimeError(f"aux example on {w['id']}: empty or overlapping target / outside after crop")
    return rt.to_input(tf(clean_in)), rt.to_input(tf(aug_in)), torch.from_numpy(tgt), torch.from_numpy(out)


class PairedSteps(Dataset):
    """r2_train.Steps' real example (same calls, same order) plus the scheduled auxiliary example, if any."""

    def __init__(self, windows: list[dict], real: list[list[int]], aux: dict[int, dict]) -> None:
        self.windows, self.real, self.aux = windows, real, aux

    def __len__(self) -> int:
        return len(self.real)

    def __getitem__(self, s: int):
        wi, dy, dx, k = self.real[s]
        w = self.windows[wi]
        img = rt.relief_window(rt.RAW / f"{w['id']}.tif")
        lab = np.load(ROOT / w["mask_path"])["label"]
        if img.shape != lab.shape:
            raise ValueError(f"{w['id']}: input {img.shape} vs mask {lab.shape}")
        img, lab = rt.dihedral(rt.crop(img, dy, dx), k), rt.dihedral(rt.crop(lab, dy, dx), k)
        ex = self.aux.get(s)
        return s, rt.to_input(img), torch.from_numpy(lab), wi, (aux_tensors(self.windows, ex) if ex else None)


def loader(windows: list[dict], real: list[list[int]], aux: dict[int, dict]) -> DataLoader:
    return DataLoader(PairedSteps(windows, real, aux), batch_size=1, shuffle=False, num_workers=WORKERS,
                      pin_memory=True, prefetch_factor=PREFETCH, collate_fn=lambda b: b[0])


# ---------------------------------------------------------------- model helpers
def verify_start() -> None:
    if rt.sha(START) != START_SHA256:
        raise RuntimeError("CONT e41 changed")
    cur = json.loads((ROOT / "CURRENT_DETECTOR.json").read_text(encoding="utf-8-sig"))
    if START_SHA256 not in json.dumps(cur):
        raise RuntimeError("CURRENT_DETECTOR.json does not name CONT e41's hash")


def state_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_pair() -> tuple[torch.nn.Module, torch.nn.Module]:
    model, teacher = rt.load_model(START), rt.load_model(START)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    rt.set_train_mode(model)
    return model, teacher


def set_lr(opt: torch.optim.Optimizer, t: int) -> float:
    lr = plans.lr_at(t)
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def step(model, teacher, opt, arm: str, t: int, batch) -> dict:
    s, real_x, real_y, wi, aux = batch
    lr = set_lr(opt, t)
    kw = {}
    if aux is not None:
        clean_x, aug_x, tgt, outside = aux
        kw = {"clean_x": clean_x[None].cuda(non_blocking=True), "implanted_x": aug_x[None].cuda(non_blocking=True),
              "target": tgt.cuda(non_blocking=True), "outside": outside.cuda(non_blocking=True)}
    res = cf1.matched_update(model, teacher, opt, real_x[None].cuda(non_blocking=True), real_y.cuda(non_blocking=True),
                             rt.masked_loss, treatment=(arm == "T"), aux_weight=AUX_WEIGHT, **kw)
    return {"step": int(s), "t": t, "lr": lr, "aux": aux is not None, **res}


def save(payload: dict, path: Path) -> None:
    if shutil.disk_usage(path.parent).free < MIN_FREE_BYTES:
        raise SystemExit(f"free disk below {MIN_FREE_BYTES / 1e9:.0f} GB; stop before writing {path.name}")
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------- training
def run_dir(seed: int, arm: str) -> Path:
    return plans.OUT / "training" / f"{seed}_{arm}"


def train(seed: int, arm: str) -> dict:
    if arm not in ARMS:
        raise SystemExit(__doc__)
    stage = "pilot" if seed == plans.PILOT_SEED else "full"
    lg.verify_frozen(stage)
    verify_start()
    windows, _, _, hold = plans.load_windows()
    rt.verify_masks(windows)
    plan = plans.load_plan(seed)
    run = run_dir(seed, arm)
    run.mkdir(parents=True, exist_ok=True)
    final, last, log_path = run / "final.pt", run / "last.pt", run / "epochs.jsonl"
    failed = run / "TRAIN_FAILED.json"
    if failed.exists():
        raise SystemExit(f"{run.name}: an earlier process recorded a failure ({failed.name}); this arm is INVALID")
    if (run / "TRAIN_RESULT.json").exists():
        return json.loads((run / "TRAIN_RESULT.json").read_text(encoding="utf-8"))
    budget = lg.remaining_s(stage)
    if budget <= 0:
        raise SystemExit(f"{stage} GPU ceiling reached; INCOMPLETE")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model, teacher = load_pair()
    teacher_digest = state_digest(teacher)
    opt = torch.optim.AdamW(model.parameters(), lr=plans.LR_MAX, weight_decay=rt.WD)
    torch.manual_seed(seed)  # same dropout stream in both arms (identical forward sequences)
    first = 0
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())  # map_location moved the byte tensors to the GPU
        torch.cuda.set_rng_state(state["rng_cuda"].cpu())
        first = state["epoch"] + 1
    stopped = False
    try:
        for epoch in range(first, plans.N_EPOCHS):
            t0 = time.monotonic()
            aux = {int(s): ex for s, ex in plan["aux"][str(epoch)].items()}
            real = plan["real"][epoch]
            if len(real) != plans.STEPS_PER_EPOCH:
                raise RuntimeError("real plan length changed")
            sums = {"real": 0.0, "aux": 0.0, "n_real": 0, "n_aux": 0, "skipped": 0}
            for batch in loader(windows, real, aux):
                if time.monotonic() - started >= budget:
                    stopped = True
                    break
                t = epoch * plans.STEPS_PER_EPOCH + int(batch[0])
                r = step(model, teacher, opt, arm, t, batch)
                if r["skipped_all_ignore"]:
                    sums["skipped"] += 1
                else:
                    sums["real"] += r["real_loss"]
                    sums["n_real"] += 1
                    if r["aux"]:
                        sums["aux"] += r["aux_loss"]
                        sums["n_aux"] += 1
                if r["step"] % LOG_EVERY == 0:
                    print(json.dumps({"seed": seed, "arm": arm, "epoch": epoch, **r,
                                      "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
            if stopped:
                break
            if state_digest(teacher) != teacher_digest:  # checked before any save, so a rerun cannot skip it
                failed.write_text(json.dumps({"epoch": epoch, "reason": "teacher weights changed"}), encoding="utf-8")
                raise RuntimeError("teacher weights changed during training; arm INVALID")
            row = {"epoch": epoch, "updates": sums["n_real"], "skipped_all_ignore": sums["skipped"],
                   "aux_updates": sums["n_aux"], "mean_real_loss": sums["real"] / max(sums["n_real"], 1),
                   "mean_aux_loss": sums["aux"] / max(sums["n_aux"], 1), "lr_end": opt.param_groups[0]["lr"],
                   "epoch_s": time.monotonic() - t0}
            if epoch == plans.N_EPOCHS - 1:
                row.update(rt.evaluate(model, windows, hold))
                save({"state_dict": model.state_dict(), "epoch": epoch, "seed": seed, "arm": arm}, final)
                row["final_sha256"] = rt.sha(final)
            save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                  "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"seed": seed, "arm": arm, **row}), flush=True)
    finally:
        lg.charge(stage, f"train_{seed}_{arm}", time.monotonic() - started,
                  {"stopped_by_ceiling": stopped, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
    if stopped:
        return {"seed": seed, "arm": arm, "status": "CEILING_REACHED"}
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    result = {"seed": seed, "arm": arm, "status": "COMPLETE", "final": final.relative_to(ROOT).as_posix(),
              "final_sha256": rt.sha(final), "epochs": rows, "teacher_unchanged": True,
              "start_sha256": START_SHA256, "plan_sha256": rt.sha(plans.PLANS / f"PLAN_{seed}_PRIVATE.json"),
              "script_sha256": rt.sha(Path(__file__)), "torch": torch.__version__,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark}
    (run / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    last.unlink()
    return result


# ---------------------------------------------------------------- GPU smoke (PROTOCOL.md 6)
def smoke(n_updates: int = 32) -> dict:
    lg.verify_frozen("pilot")
    verify_start()
    windows, _, _, hold = plans.load_windows()
    plan = plans.load_plan(plans.PILOT_SEED)
    real = plan["real"][0][:n_updates]
    aux = {int(s): ex for s, ex in plan["aux"]["0"].items() if int(s) < n_updates}
    out: dict = {"n_updates": n_updates, "aux_updates": len(aux), "arms": {}}
    started = time.monotonic()
    try:
        for arm in ARMS:
            torch.cuda.reset_peak_memory_stats()
            model, teacher = load_pair()
            opt = torch.optim.AdamW(model.parameters(), lr=plans.LR_MAX, weight_decay=rt.WD)
            torch.manual_seed(plans.PILOT_SEED)
            times = {"real": [], "aux": []}
            t_prev = time.monotonic()
            for batch in loader(windows, real, aux):
                r = step(model, teacher, opt, arm, int(batch[0]), batch)
                torch.cuda.synchronize()
                now = time.monotonic()
                if r["step"] >= 8:  # after loader warm-up
                    times["aux" if r["aux"] else "real"].append(now - t_prev)
                t_prev = now
            out["arms"][arm] = {"s_per_real_update": float(np.mean(times["real"])),
                                "s_per_aux_update": float(np.mean(times["aux"])),
                                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
            del model, teacher, opt
            torch.cuda.empty_cache()
        out["inference"] = smoke_inference(windows, hold[:16])
    finally:
        lg.charge("pilot", "smoke", time.monotonic() - started, {"n_updates": n_updates})
    a = out["arms"]["T"]
    per_arm = (plans.TOTAL_UPDATES - plans.TOTAL_UPDATES // 4) * a["s_per_real_update"] + \
        (plans.TOTAL_UPDATES // 4) * a["s_per_aux_update"] + 60.0
    out["projected_train_s_per_arm"] = per_arm
    out["planning_scoring_s"], out["planning_diagnostic_s"] = SCORING_ESTIMATE_S, DIAGNOSTIC_ESTIMATE_S
    out["projected_pilot_total_s"] = lg.charged_s("pilot") + 2 * per_arm + SCORING_ESTIMATE_S + DIAGNOSTIC_ESTIMATE_S
    out["fits_ceiling"] = out["projected_pilot_total_s"] <= lg.CEILINGS_S["pilot"]
    out["limit_allocated_mib"] = 5.2 * 1024
    out["status"] = "PASS" if max(v["peak_allocated_mib"] for v in out["arms"].values()) <= out["limit_allocated_mib"] \
        else "OVER_MEMORY"
    (plans.OUT / "SMOKE.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def smoke_inference(windows: list[dict], idx: list[int]) -> dict:
    import score_pairs as sp
    tr = sp.tr
    models = sp.load_models({"CONT": START})
    predict = sp.he.predictor(models)
    t0 = time.monotonic()
    for i in idx:
        tr.objects_multi(rt.RAW / f"{windows[i]['id']}.tif", predict, ["CONT"])
    return {"windows": len(idx), "s_per_window_one_model": (time.monotonic() - t0) / len(idx)}


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "smoke":
        print(json.dumps(smoke(), indent=1))
    elif cmd == "train" and len(sys.argv) == 4:
        print(json.dumps(train(int(sys.argv[2]), sys.argv[3]), indent=1))
    else:
        raise SystemExit(__doc__)
