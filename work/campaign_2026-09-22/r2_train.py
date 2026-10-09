"""Amendment 2 D: R2 training path — fine-tune the pinned ADAF AO HRNet SLRM checkpoint on catalogue masks.

Inputs are adapter.py's preprocessing (bilinear 0.5 m view, SLRM radius 20 cells, normalize_image('slrm',
-0.5, 0.5, 'value'), NaN -> 0, three identical channels), recomputed per step; 0.5 m inputs are never cached.
Loss: BCE with logits on output channel 1 over non-ignored pixels + soft Dice on sigmoid(channel 1) with
smoothing 1.0, equal weights (ADAF infers with sigmoid(out)[:, 1] >= 0.5). BatchNorm running statistics stay at
the checkpoint values (BN modules in eval mode); every parameter trains. AdamW lr 1e-4, wd 1e-4, batch 1, fp32.

  python r2_train.py reproduce         # PREREG L115: R2-path eval load reproduces adapter output (<= 1e-4)
  python r2_train.py smoke STEPS       # memory/time smoke on STEPS training steps + 16 holdout windows
  python r2_train.py train             # full run (resumable), 40-epoch max, 12 GPU-hour shared cap
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import adaf_configs  # noqa: E402

adapter = adaf_configs.adapter
MASKS = HERE / "R2_MASKS_MANIFEST_PRIVATE.json"
RAW = ROOT / "outputs/campaign_2026-09-22/train_raw"
OUT = ROOT / "outputs/campaign_2026-09-22/training/R2_adaf_hrnet_slrm"
GPU_LEDGER = OUT / "GPU_TIME.json"
SEED, CROP, OFFSET_PX, RADIUS, IGNORE = 20260923, 512, 160, 20, 255
LR = WD = 1e-4
MAX_EPOCHS, CAP_S, WORKERS = 40, 12 * 3600, 3


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- inputs and augmentation
def relief_window(raw_path: Path) -> np.ndarray:
    """adapter.probability_raster's per-pixel input for a whole raw window (identical values)."""
    with rasterio.open(raw_path) as original:
        src = adapter.source_view(original)
        try:
            a = src.read(1, masked=True).filled(np.nan).astype("float32")
        finally:
            if src is not original:
                src.close()
    data = np.full((a.shape[0] + 2 * RADIUS, a.shape[1] + 2 * RADIUS), np.nan, dtype="float32")
    data[RADIUS:-RADIUS, RADIUS:-RADIUS] = a
    slrm = adapter.rvt.vis.slrm(data, radius_cell=RADIUS)
    norm = adapter.normalize_image("slrm", slrm, -.5, .5, "value")[RADIUS:-RADIUS, RADIUS:-RADIUS]
    return np.nan_to_num(norm, nan=0.).astype("float32")


def crop(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """512 px crop whose centre is offset (dy, dx) px from the window centre (+-160 px = +-80 m)."""
    h, w = arr.shape[:2]
    y0, x0 = h // 2 - CROP // 2 + dy, w // 2 - CROP // 2 + dx
    if not (0 <= y0 and y0 + CROP <= h and 0 <= x0 and x0 + CROP <= w):
        raise ValueError(f"crop outside window: {(h, w, dy, dx)}")
    return arr[y0:y0 + CROP, x0:x0 + CROP]


def dihedral(arr: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: rotate k % 4 quarter turns in the (row, col) plane, then flip columns if k >= 4."""
    out = np.rot90(arr, k % 4, axes=(0, 1))
    if k >= 4:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def to_input(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.repeat(img[None, :, :], 3, axis=0))


class Steps(Dataset):
    def __init__(self, windows: list[dict], plan: list[tuple[int, int, int, int]]) -> None:
        self.windows, self.plan = windows, plan

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int):
        wi, dy, dx, k = self.plan[i]
        w = self.windows[wi]
        img = relief_window(RAW / f"{w['id']}.tif")
        lab = np.load(ROOT / w["mask_path"])["label"]
        if img.shape != lab.shape:
            raise ValueError(f"{w['id']}: input {img.shape} vs mask {lab.shape}")
        img, lab = dihedral(crop(img, dy, dx), k), dihedral(crop(lab, dy, dx), k)
        return to_input(img), torch.from_numpy(lab), wi


# ---------------------------------------------------------------- sampling plan (deterministic, resumable)
def draw(rng: np.random.Generator) -> tuple[int, int, int]:
    dy, dx = (int(v) for v in rng.integers(-OFFSET_PX, OFFSET_PX + 1, size=2))
    return dy, dx, int(rng.integers(0, 8))


def negative_at(g: int, fit_neg: list[int]) -> int:
    """Seeded negative cycle: the g-th negative overall (continues across epochs)."""
    cycle, pos = divmod(g, len(fit_neg))
    return int(np.random.default_rng([SEED, 1, cycle]).permutation(fit_neg)[pos])


def epoch_plan(epoch: int, fit_pos: list[int], fit_neg: list[int]) -> list[tuple[int, int, int, int]]:
    rng = np.random.default_rng([SEED, 0, epoch])
    plan = []
    for i, wi in enumerate(rng.permutation(fit_pos)):
        plan.append((int(wi), *draw(rng)))
        plan.append((negative_at(epoch * len(fit_pos) + i, fit_neg), *draw(rng)))
    return plan


# ---------------------------------------------------------------- model, loss
def load_model(state_path: Path | None = None, device: str = "cuda") -> torch.nn.Module:
    """Pinned AO checkpoint (md5-verified) through the official AiTLAS HRNet definition."""
    _, path, _ = adaf_configs.resolve("AO") if state_path is None else ("", state_path, None)
    (model,), _ = adaf_configs.load_models([path], device=device)
    return model


def set_train_mode(model: torch.nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()  # batch 1: keep checkpoint running statistics


def masked_loss(logit: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, float, float] | None:
    valid = label != IGNORE
    if not bool(valid.any()):
        return None
    z = logit[valid]
    g = (label[valid] == 1).float()
    bce = F.binary_cross_entropy_with_logits(z, g)
    p = torch.sigmoid(z)
    dice = (2 * (p * g).sum() + 1.0) / (p.sum() + g.sum() + 1.0)
    return bce + (1 - dice), float(bce), float(dice)


# ---------------------------------------------------------------- data index
def verify_masks(windows: list[dict]) -> None:
    bad = [w["id"] for w in windows if sha(ROOT / w["mask_path"]) != w["mask_sha256"]]
    if bad:
        raise RuntimeError(f"{len(bad)} mask files differ from the manifest, e.g. {bad[:3]}")


def load_windows() -> tuple[list[dict], list[int], list[int], list[int]]:
    manifest = json.loads(MASKS.read_text(encoding="utf-8"))
    if manifest["summary"]["gate"] != "PASS":
        raise RuntimeError("mask gate did not pass")
    windows = sorted(manifest["windows"], key=lambda w: w["id"])
    fit_pos = [i for i, w in enumerate(windows) if w["role"] == "positive" and not w["holdout"]]
    fit_neg = [i for i, w in enumerate(windows) if w["role"] == "negative" and not w["holdout"]]
    hold = [i for i, w in enumerate(windows) if w["holdout"]]
    return windows, fit_pos, fit_neg, hold


def loader(windows: list[dict], plan: list[tuple], workers: int = WORKERS) -> DataLoader:
    return DataLoader(Steps(windows, plan), batch_size=1, shuffle=False, num_workers=workers,
                      pin_memory=True, prefetch_factor=4 if workers else None)


@torch.no_grad()
def evaluate(model: torch.nn.Module, windows: list[dict], hold: list[int]) -> dict:
    model.eval()
    per = []
    for x, y, wi in loader(windows, [(i, 0, 0, 0) for i in hold]):
        out = model(x.cuda(non_blocking=True))["out"][0, 1]
        res = masked_loss(out, y[0].cuda())
        if res is not None:
            role = windows[int(wi)]["role"]
            per.append((float(res[0]), role))
    set_train_mode(model)
    losses = np.array([v for v, _ in per])
    return {"holdout_loss": float(losses.mean()), "n": len(per),
            "positive_loss": float(np.mean([v for v, r in per if r == "positive"])),
            "negative_loss": float(np.mean([v for v, r in per if r == "negative"]))}


# ---------------------------------------------------------------- GPU-time ledger (shared 12 h cap)
def ledger() -> dict:
    return json.loads(GPU_LEDGER.read_text(encoding="utf-8")) if GPU_LEDGER.exists() else {"runs": []}


def charged() -> float:
    return sum(r["wall_s"] for r in ledger()["runs"])


def charge(kind: str, wall_s: float, note: dict) -> None:
    led = ledger()
    led["runs"].append({"kind": kind, "wall_s": wall_s, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                   time.gmtime()), **note})
    led["total_wall_s"] = sum(r["wall_s"] for r in led["runs"])
    led["cap_s"] = CAP_S
    GPU_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    GPU_LEDGER.write_text(json.dumps(led, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- PREREG L115 reproduction check
def fixed_tile(windows: list[dict], fit: list[int]) -> dict:
    """The fixed TRAIN tile: first fitting window in SHA-256 order of its verified raw file."""
    return min((windows[i] for i in fit), key=lambda w: sha(RAW / f"{w['id']}.tif"))


def reproduce() -> dict:
    windows, fit_pos, fit_neg, _ = load_windows()
    w = fixed_tile(windows, fit_pos + fit_neg)
    img = crop(relief_window(RAW / f"{w['id']}.tif"), 0, 0)
    ref_model, _, _ = adapter.load_model(device="cuda")
    reference = adapter.predict(ref_model, img)
    del ref_model
    model = load_model()
    model.eval()
    with torch.inference_mode():
        ours = torch.sigmoid(model(to_input(img)[None].cuda())["out"])[0, 1].cpu().numpy()
    err = float(np.max(np.abs(ours - reference)))
    result = {"status": "PASS" if err <= 1e-4 else "FAIL", "tile_id": w["id"], "max_abs_error": err,
              "tolerance": 1e-4, "r2_train_sha256": sha(Path(__file__))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPRODUCTION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------- training
def train_steps(model, opt, windows, plan, budget_s: float, started: float, log) -> tuple[int, float, bool]:
    """Run one plan; returns (steps done, mean loss, stopped_by_cap)."""
    set_train_mode(model)
    total, n = 0.0, 0
    for step, (x, y, wi) in enumerate(loader(windows, plan)):
        if time.monotonic() - started >= budget_s:
            return n, total / max(n, 1), True
        res = masked_loss(model(x.cuda(non_blocking=True))["out"][0, 1], y[0].cuda(non_blocking=True))
        if res is None:
            continue
        opt.zero_grad(set_to_none=True)
        res[0].backward()
        opt.step()
        total, n = total + float(res[0]), n + 1
        if step % 200 == 0:
            log({"step": step, "loss": float(res[0]), "bce": res[1], "dice": res[2],
                 "elapsed_s": time.monotonic() - started})
    return n, total / max(n, 1), False


def save(model, opt, epoch: int, best: float, path: Path, with_optimizer: bool) -> None:
    payload = {"state_dict": model.state_dict(), "epoch": epoch, "best_holdout_loss": best}
    if with_optimizer:
        payload["optimizer"] = opt.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def smoke(steps: int) -> dict:
    windows, fit_pos, fit_neg, hold = load_windows()
    verify_masks(windows)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = load_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    plan = epoch_plan(0, fit_pos, fit_neg)[:steps]
    t0 = time.monotonic()
    n, mean_loss, _ = train_steps(model, opt, windows, plan, float("inf"), started, lambda d: print(json.dumps(d)))
    train_s = time.monotonic() - t0
    t1 = time.monotonic()
    ev = evaluate(model, windows, hold[:16])
    eval_s = time.monotonic() - t1
    wall = time.monotonic() - started
    result = {"steps": n, "mean_loss": mean_loss, "train_s": train_s, "s_per_step": train_s / max(n, 1),
              "eval_windows": 16, "eval_s": eval_s, "eval": ev, "wall_s": wall,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "limit_mib": 5.5 * 1024,
              "fit_positive": len(fit_pos), "fit_negative": len(fit_neg), "holdout": len(hold),
              "epoch_steps": 2 * len(fit_pos), "r2_train_sha256": sha(Path(__file__))}
    result["status"] = "PASS" if result["peak_reserved_mib"] <= result["limit_mib"] else "OVER_MEMORY"
    charge("smoke", wall, {"steps": n, "status": result["status"]})
    (OUT / "SMOKE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train() -> dict:
    windows, fit_pos, fit_neg, hold = load_windows()
    verify_masks(windows)
    if any(windows[i]["holdout"] for i in fit_pos + fit_neg) or not all(windows[i]["holdout"] for i in hold):
        raise RuntimeError("holdout isolation violated")
    budget = CAP_S - charged()
    started = time.monotonic()
    model = load_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    last, best_path, log_path = OUT / "last.pt", OUT / "best.pt", OUT / "epochs.jsonl"
    epoch0, best = 0, float("inf")
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=True)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        epoch0, best = state["epoch"] + 1, state["best_holdout_loss"]
    log = lambda d: print(json.dumps(d), flush=True)
    stopped = False
    try:
        for epoch in range(epoch0, MAX_EPOCHS):
            t0 = time.monotonic()
            n, mean_loss, stopped = train_steps(model, opt, windows, epoch_plan(epoch, fit_pos, fit_neg),
                                                budget, started, log)
            if stopped:
                break
            ev = evaluate(model, windows, hold)
            improved = ev["holdout_loss"] < best
            best = min(best, ev["holdout_loss"])
            if improved:
                save(model, opt, epoch, best, best_path, with_optimizer=False)
            save(model, opt, epoch, best, last, with_optimizer=True)
            row = {"epoch": epoch, "steps": n, "train_loss": mean_loss, **ev, "improved": improved,
                   "epoch_s": time.monotonic() - t0, "run_elapsed_s": time.monotonic() - started}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            log(row)
    finally:
        charge("train", time.monotonic() - started, {"stopped_by_cap": stopped})
    result = {"status": "CAP_REACHED" if stopped else "COMPLETE", "best_holdout_loss": best,
              "best_sha256": sha(best_path) if best_path.exists() else None, "charged_s": charged()}
    (OUT / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    cmd = sys.argv[1]
    torch.backends.cudnn.benchmark = False
    if cmd == "reproduce":
        print(json.dumps(reproduce()))
    elif cmd == "smoke":
        print(json.dumps(smoke(int(sys.argv[2]))))
    elif cmd == "train":
        print(json.dumps(train()))
    else:
        raise SystemExit(__doc__)
