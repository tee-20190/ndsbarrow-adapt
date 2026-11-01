"""AMZ-ADAPT-1 matched continuation trainer (PACKET.md): CONT e41 -> 10,232 updates, English plan of Experiment 1's
pilot seed; the adapted arms replace whole English (positive, negative) step pairs p = s // 2 with
(p - e) % 4 == 0 (1,279 pairs = 2,558 steps, 25 % of draws) by an (Amazon positive, Amazon background) pair from the
arm's training groups, so every epoch keeps C's 1:1 positive:negative mix.

Arms: C (English only), A_BON / A_RIB / A_FN3 (Amazon draws from the two ORNL groups other than the named one).
Void policy V (both arms, training and inference): model input NaN -> 0.5 (0 m relief) instead of 0.0; pixels whose
model input is non-finite carry the ignore label (255). English TRAIN windows hold no voids, so V leaves every English
sample unchanged. Everything else is Experiment 1's recipe: fresh AdamW (wd 1e-4), batch 1, fp32, frozen BatchNorm,
cosine LR 1e-4 -> 1e-6 per update (plans.lr_at), r2_train.masked_loss, final checkpoint only.

  python -B work/amazon_adapt_2026-09-28/adapt_train.py smoke ARM N    # N updates, timing + memory, no save
  python -B work/amazon_adapt_2026-09-28/adapt_train.py train ARM      # full run (resumable per epoch)
(WSL, adaf-h2h-venv, repo root; launched through archive_gate + storage_guard by run_pilot.sh)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("work/campaign_2026-09-22", "work/counterfactual_2026-09-28", "work/counterfactual_2026-09-28/cf_v1"):
    sys.path.insert(0, str(ROOT / p))
import r2_train as rt  # noqa: E402  (frozen: relief, crop, dihedral, loss, model loading, holdout loss)
import plans  # noqa: E402  (frozen: Experiment 1 plan, windows, LR schedule)
import paired_train as pt  # noqa: E402  (frozen: START, verify_start, save with disk check)

OUT = ROOT / "outputs/amazon_adapt_2026-09-28"
WINDOWS = OUT / "WINDOWS_PRIVATE.json"
SEED = plans.PILOT_SEED            # 2026092800: Experiment 1's pilot plan (English draws identical to its C arm)
AMZ_SEED = 2026092810              # Amazon draw stream (positions fixed by the English plan)
FILL = 0.5                         # void policy V: 0 m relief
ARMS = {"C": None, "A_BON": ("RIB", "FN3"), "A_RIB": ("BON", "FN3"), "A_FN3": ("BON", "RIB")}
WORKERS, PREFETCH, LOG_EVERY = 3, 4, 500


def relief_v(tif: Path) -> np.ndarray:
    """rt.relief_window with void policy V: identical transform, NaN -> 0.5; NaN left for the caller's mask."""
    import rasterio
    with rasterio.open(tif) as original:
        src = rt.adapter.source_view(original)
        try:
            a = src.read(1, masked=True).filled(np.nan).astype("float32")
        finally:
            if src is not original:
                src.close()
    r = rt.RADIUS
    data = np.full((a.shape[0] + 2 * r, a.shape[1] + 2 * r), np.nan, dtype="float32")
    data[r:-r, r:-r] = a
    slrm = rt.adapter.rvt.vis.slrm(data, radius_cell=r)
    return rt.adapter.normalize_image("slrm", slrm, -.5, .5, "value")[r:-r, r:-r].astype("float32")


def apply_v(norm: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bad = ~np.isfinite(norm)
    lab = lab.copy()
    lab[bad] = rt.IGNORE
    return np.where(bad, np.float32(FILL), norm).astype("float32"), lab


def amazon_windows(arm: str) -> tuple[list[dict], list[int], list[int]]:
    rows = json.loads(WINDOWS.read_text(encoding="utf-8"))["windows"]
    groups = ARMS[arm] or ()
    pos = [i for i, w in enumerate(rows) if w["group"] in groups and w["role"] == "positive"]
    bg = [i for i, w in enumerate(rows) if w["group"] in groups and w["role"] == "background"]
    return rows, pos, bg


def replaced_pair(p: int, epoch: int) -> bool:
    """English step pair p = (2p positive, 2p + 1 negative) is replaced in the adapted arms."""
    return (p - epoch) % 4 == 0


def amazon_plan(epoch: int, pairs: list[int], pos: list[int], bg: list[int],
                offset: int) -> dict[int, tuple[int, int, int, int]]:
    """Replaced step -> (Amazon window index, dy, dx, k). Pair n overall (offset + position in this epoch) puts the
    n-th positive draw at step 2p and the n-th background draw at step 2p + 1; each kind cycles through a seeded
    permutation (new permutation per cycle); offsets and dihedral as rt.draw."""
    out = {}
    for m, p in enumerate(pairs):
        n = offset + m
        for kind, pool, s in ((0, pos, 2 * p), (1, bg, 2 * p + 1)):
            cycle, idx = divmod(n, len(pool))
            wi = int(np.random.default_rng([AMZ_SEED, kind, cycle]).permutation(pool)[idx])
            dy, dx, k = rt.draw(np.random.default_rng([AMZ_SEED, 2, epoch, s]))
            out[s] = (wi, dy, dx, k)
    return out


class AdaptSteps(Dataset):
    def __init__(self, windows: list[dict], real: list, amz_rows: list[dict], amz: dict) -> None:
        self.windows, self.real, self.amz_rows, self.amz = windows, real, amz_rows, amz

    def __len__(self) -> int:
        return len(self.real)

    def __getitem__(self, s: int):
        if s in self.amz:
            wi, dy, dx, k = self.amz[s]
            w = self.amz_rows[wi]
            norm = relief_v(ROOT / w["tif"])
            lab = np.load(ROOT / w["mask_path"])["label"]
            src = w["id"]
        else:
            wi, dy, dx, k = self.real[s]
            w = self.windows[wi]
            norm = relief_v(rt.RAW / f"{w['id']}.tif")
            lab = np.load(ROOT / w["mask_path"])["label"]
            src = w["id"]
        if norm.shape != lab.shape:
            raise ValueError(f"{src}: input {norm.shape} vs mask {lab.shape}")
        img, lab = apply_v(norm, lab)
        img, lab = rt.dihedral(rt.crop(img, dy, dx), k), rt.dihedral(rt.crop(lab, dy, dx), k)
        return s, rt.to_input(img), torch.from_numpy(lab), s in self.amz


def loader(ds: AdaptSteps) -> DataLoader:
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=WORKERS, pin_memory=True,
                      prefetch_factor=PREFETCH, collate_fn=lambda b: b[0])


def epoch_inputs(arm: str, epoch: int, plan: dict, amz_rows: list[dict], pos: list[int], bg: list[int]):
    real = plan["real"][epoch]
    if len(real) != plans.STEPS_PER_EPOCH:
        raise RuntimeError("real plan length changed")
    if ARMS[arm] is None:
        return real, {}
    n_pairs = plans.STEPS_PER_EPOCH // 2
    pairs = [p for p in range(n_pairs) if replaced_pair(p, epoch)]
    offset = sum(sum(1 for p in range(n_pairs) if replaced_pair(p, e)) for e in range(epoch))
    return real, amazon_plan(epoch, pairs, pos, bg, offset)


def update(model, opt, t: int, x: torch.Tensor, y: torch.Tensor) -> dict:
    """cf_v1.matched_update's real-crop path without the auxiliary forwards (identical parameter update)."""
    lr = plans.lr_at(t)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    z = model(x[None].cuda(non_blocking=True))["out"][0, 1]
    res = rt.masked_loss(z, y.cuda(non_blocking=True))
    if res is None:
        return {"skipped": True, "lr": lr}
    res[0].backward()
    opt.step()
    return {"skipped": False, "loss": float(res[1] + (1 - res[2])), "lr": lr}


def run_dir(arm: str) -> Path:
    return OUT / "training" / arm


def fresh_model() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = rt.load_model(pt.START)
    rt.set_train_mode(model)
    return model, torch.optim.AdamW(model.parameters(), lr=plans.LR_MAX, weight_decay=rt.WD)


def train(arm: str) -> dict:
    if arm not in ARMS:
        raise SystemExit(__doc__)
    pt.verify_start()
    windows, _, _, hold = plans.load_windows()
    rt.verify_masks(windows)
    plan = plans.load_plan(SEED)
    amz_rows, pos, bg = amazon_windows(arm)
    run = run_dir(arm)
    run.mkdir(parents=True, exist_ok=True)
    final, last, log_path = run / "final.pt", run / "last.pt", run / "epochs.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return json.loads((run / "TRAIN_RESULT.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model, opt = fresh_model()
    torch.manual_seed(SEED)
    first = 0
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state(state["rng_cuda"].cpu())
        first = state["epoch"] + 1
    for epoch in range(first, plans.N_EPOCHS):
        t0 = time.monotonic()
        real, amz = epoch_inputs(arm, epoch, plan, amz_rows, pos, bg)
        sums = {"loss": 0.0, "n": 0, "amz_loss": 0.0, "n_amz": 0, "skipped": 0}
        for s, x, y, is_amz in loader(AdaptSteps(windows, real, amz_rows, amz)):
            r = update(model, opt, epoch * plans.STEPS_PER_EPOCH + int(s), x, y)
            if r["skipped"]:
                sums["skipped"] += 1
                continue
            key = "amz_loss" if is_amz else "loss"
            sums[key] += r["loss"]
            sums["n_amz" if is_amz else "n"] += 1
            if int(s) % LOG_EVERY == 0:
                print(json.dumps({"arm": arm, "epoch": epoch, "step": int(s), **r,
                                  "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        row = {"epoch": epoch, "english_updates": sums["n"], "amazon_updates": sums["n_amz"],
               "skipped_all_ignore": sums["skipped"], "mean_english_loss": sums["loss"] / max(sums["n"], 1),
               "mean_amazon_loss": sums["amz_loss"] / max(sums["n_amz"], 1), "lr_end": opt.param_groups[0]["lr"],
               "epoch_s": time.monotonic() - t0}
        if epoch == plans.N_EPOCHS - 1:
            row.update(rt.evaluate(model, windows, hold))
            pt.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": SEED, "arm": arm}, final)
            row["final_sha256"] = rt.sha(final)
        pt.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"arm": arm, **row}), flush=True)
    rows = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    result = {"arm": arm, "status": "COMPLETE", "final": final.relative_to(ROOT).as_posix(),
              "final_sha256": rt.sha(final), "epochs": rows, "start_sha256": pt.START_SHA256,
              "plan_sha256": rt.sha(plans.PLANS / f"PLAN_{SEED}_PRIVATE.json"),
              "windows_sha256": rt.sha(WINDOWS), "script_sha256": rt.sha(Path(__file__)),
              "wall_s": time.monotonic() - started, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "torch": torch.__version__, "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark}
    (run / "TRAIN_RESULT.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    last.unlink()
    return result


def smoke(arm: str, n: int) -> dict:
    """First n updates of epoch 0 (no save): timing, memory, Amazon-sample validity."""
    pt.verify_start()
    windows, _, _, _ = plans.load_windows()
    plan = plans.load_plan(SEED)
    amz_rows, pos, bg = amazon_windows(arm)
    real, amz = epoch_inputs(arm, 0, plan, amz_rows, pos, bg)
    real, amz = real[:n], {s: v for s, v in amz.items() if s < n}
    torch.cuda.reset_peak_memory_stats()
    model, opt = fresh_model()
    torch.manual_seed(SEED)
    times, stats = {"eng": [], "amz": []}, {"amz_pos_px": 0, "amz_ignore_px": 0, "amz_samples": 0, "losses": []}
    prev = time.monotonic()
    for s, x, y, is_amz in loader(AdaptSteps(windows, real, amz_rows, amz)):
        r = update(model, opt, int(s), x, y)
        torch.cuda.synchronize()
        now = time.monotonic()
        if int(s) >= 8:
            times["amz" if is_amz else "eng"].append(now - prev)
        prev = now
        if is_amz:
            stats["amz_samples"] += 1
            stats["amz_pos_px"] += int((y == 1).sum())
            stats["amz_ignore_px"] += int((y == rt.IGNORE).sum())
        stats["losses"].append(None if r["skipped"] else round(r["loss"], 5))
    out = {"arm": arm, "n": n, "s_per_english_update": float(np.mean(times["eng"])) if times["eng"] else None,
           "s_per_amazon_update": float(np.mean(times["amz"])) if times["amz"] else None,
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, **stats}
    (OUT / f"SMOKE_{arm}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "smoke" and len(sys.argv) == 4:
        print(json.dumps(smoke(sys.argv[2], int(sys.argv[3])), indent=1))
    elif cmd == "train" and len(sys.argv) == 3:
        print(json.dumps(train(sys.argv[2]), indent=1))
    else:
        raise SystemExit(__doc__)
