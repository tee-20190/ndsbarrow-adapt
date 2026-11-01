"""DENSE-TERRAIN-1 F1 trainer (FREEZE.md Sections 2-4): D1's collapse-safe F1 fine-tune with two differences only,
(a) the model per arm (dt1_model.build: A = D1's 3-channel model object; B/G/D = the summed-weight 4-channel stem, G/D
from an SSL encoder) and (b) for B/G/D the three added input channels, cut from the window's native products with the
same crop and dihedral as the legacy channel.

Unchanged from D1 (imported read-only, never edited): the Lower Saxony FIT table, rows and pools
(match_train.ls_context), D1's all-local plan per seed (match_train.schedule), the start (public ADAF barrow tar),
recipe F1 (d1_train.update), fresh AdamW lr 1e-4 wd 1e-4 on every parameter, frozen BatchNorm, batch 1, fp32 with TF32
off, torch.manual_seed(seed) after the model is built, the online collapse detector (records only), per-update curves,
resume per epoch, final weights only. The legacy channel is adapt_train.AdaptSteps' own item (void policy V).

  python -B dt1_train.py train SEED --arm A|B|G|D --ls-table T --ls-win W --ls-mask M [--prod P] [--encoder E] --run-root R
  python -B dt1_train.py plan SEED OUT.json --ls-table T --ls-win W --ls-mask M     (CPU: per-epoch draw digests)
  python -B dt1_train.py smoke N OUT.json --arm A|B --ls-table T --ls-win W --ls-mask M [--prod P] [--device cuda|cpu]
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work/d1_collapse_2026-10-06"))
import d1_train as T1  # noqa: E402  D1-COLLAPSE-1 (frozen)
from d1_train import D, MS, MT, N, NT, SCH, at, pt, rt, wt  # noqa: E402
import dt1_channels as CH  # noqa: E402
import dt1_model as DM  # noqa: E402

EXPERIMENT = "DENSE-TERRAIN-1"
SEEDS = (101, 202, 303)
START_ARM, RECIPE = "BARROW_L", "F1"
ATTRIBUTION = {**D.ATTRIBUTION,
               "dt1_weights": "DENSE-TERRAIN-1 runs are adaptations of public ADAF barrow (CC BY-SA 4.0) fine-tuned with "
                              "recipe F1 on Lower Saxony register labels and LGLN DGM1 (CC BY 4.0) inputs; any shared "
                              "copy stays CC BY-SA 4.0 with this attribution"}


def run_name(arm: str, seed: int) -> str:
    if arm not in DM.ARMS or seed not in SEEDS:
        raise SystemExit(f"run outside DENSE-TERRAIN-1: arm {arm} seed {seed}")
    return f"DT1{arm}_F1_BARROW_L_{seed}"


def extra_channels(prod_path: Path, dy: int, dx: int, k: int) -> np.ndarray:
    """(3, 512, 512): asinh(r2/.1), asinh(r32/1), v for the crop (dy, dx) and dihedral k of the window."""
    with np.load(prod_path) as z:
        prod = {key: z[key] for key in ("r2", "r32", "support")}
    ch = CH.view_channels(prod)[:, 2:-2, 2:-2]                 # window + 1 border cell -> the window's 0.5 m grid
    return np.stack([rt.dihedral(rt.crop(c, dy, dx), k) for c in ch]).astype(np.float32)


class DT1Steps(at.AdaptSteps):
    """adapt_train.AdaptSteps over D1's all-local plan; for B/G/D the item gains the three added channels."""

    def __init__(self, rows: list[dict], wmap: dict, n: int, arm: str, prod_dir: Path | None) -> None:
        super().__init__([], [None] * n, rows, wmap)
        if DM.CHANNELS[arm] == 4 and prod_dir is None:
            raise SystemExit(f"arm {arm} needs --prod")
        self.arm, self.prod_dir = arm, prod_dir

    def __getitem__(self, s: int):
        item = super().__getitem__(s)
        if DM.CHANNELS[self.arm] == 3:
            return item
        import torch
        step, x, y, flag = item
        wi, dy, dx, k = self.amz[s]
        ex = extra_channels(self.prod_dir / f"{self.amz_rows[wi]['id']}.npz", dy, dx, k)
        return step, torch.cat([x[:1], torch.from_numpy(ex)], 0), y, flag


def epoch_pass(arm: str, seed: int, epoch: int, c: dict, name: str, started: float, device: str = "cuda",
               limit: int | None = None) -> tuple[dict, list[dict]]:
    """d1_train.epoch_pass (recipe F1, BARROW_L) over DT1Steps; c["detector"] None = detector off (smoke)."""
    t0 = time.monotonic()
    rows = c["ctx"]["rows"]
    wmap = MT.schedule(seed, epoch, c["ctx"])
    n = MS.STEPS_PER_EPOCH if limit is None else limit
    wmap = {s: v for s, v in wmap.items() if s < n}
    if set(wmap) != set(range(n)):
        raise SystemExit("the all-local plan does not cover every step")
    sums, recs, det = T1.new_sums(), [], c["detector"]
    for s, x, y, _ in at.loader(DT1Steps(rows, wmap, n, arm, c.get("prod"))):
        role = rows[wmap[int(s)][0]]["role"]
        t = SCH.t_index(epoch, int(s))
        r = T1.update(c["model"], c["opt"], RECIPE, t, x, y, device)
        rec = {"t": t, "epoch": epoch, "step": int(s), "role": role, **r}
        was_on = det.on if det else None
        if det is not None:
            det.push(t, role, rec)
        rec["detector_on"] = det.on if det else None
        recs.append(rec)
        if r["skipped"]:
            sums["skipped"] += 1
        else:
            for m in ("loss_old", "loss_used", "dice", "sum_p"):
                sums[f"{role}_{m}"] += r[m]
            sums[f"n_{role}"] += 1
        if int(s) % at.LOG_EVERY == 0 or (det is not None and det.on != was_on):
            print(json.dumps({"run": name, "epoch": epoch, "step": int(s), "t": t, "role": role, "lr": r["lr"],
                              **({k: r[k] for k in ("loss_old", "dice", "sum_p", "z_max")} if not r["skipped"] else {}),
                              "detector_on": rec["detector_on"], "elapsed_s": round(time.monotonic() - started, 1)}),
                  flush=True)
    row = {"epoch": epoch, "recipe": RECIPE, "arm": arm, "lr_end": c["opt"].param_groups[0]["lr"],
           "skipped_all_ignore": sums["skipped"], "epoch_s": time.monotonic() - t0,
           "updates": MT.role_counts(wmap, rows), "lower_saxony_updates": len(wmap),
           **{f"mean_{k}_loss": (sums[f"{k}_loss_old"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None) for k in MS.ROLES},
           **{f"mean_{k}_{m}": (sums[f"{k}_{m}"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None)
              for k in MS.ROLES for m in ("loss_used", "dice", "sum_p")},
           "detector_on_at_epoch_end": det.on if det else None, "draw_digest": MT.draw_digest(wmap, rows)}
    return row, recs


def stem_record(model) -> dict:
    w = model.backbone.conv1.weight.detach()
    return {"in_channels": int(w.shape[1]),
            "added_channel_weight_l2": float(w[:, 1:].norm()) if w.shape[1] == 4 else None}


def train(arm: str, seed: int, paths: dict, run_root: Path) -> dict:
    import torch
    name = run_name(arm, seed)
    NT.check_constants()
    c = {"ctx": MT.ls_context(paths), "prod": paths.get("prod")}
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    last, log_path, curves = run / "last.pt", run / "epochs.jsonl", run / "curves.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return N.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    c["model"], c["opt"] = DM.build(arm, "cuda", paths.get("encoder"))
    torch.manual_seed(seed)
    done = wt._resume(c["model"], c["opt"], last)
    if done == 0:
        for p in (log_path, curves):
            p.unlink(missing_ok=True)
    c["detector"] = T1.Detector()
    if done:
        for ln in curves.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            c["detector"].push(r["t"], r["role"], r)
    for epoch in MS.EPOCHS:
        if epoch < done:
            continue
        row, recs = epoch_pass(arm, seed, epoch, c, name, started)
        row["stem"] = stem_record(c["model"])
        T1.append_lines(curves, recs)
        if epoch == MS.EPOCHS[-1]:
            final = run / "final.pt"
            pt.save({"state_dict": c["model"].state_dict(), "epoch": epoch, "seed": seed, "arm": START_ARM,
                     "dt1_arm": arm, "channels": DM.CHANNELS[arm], "recipe": RECIPE, "experiment": EXPERIMENT,
                     "start": D.STARTS[START_ARM]["sha256"]}, final)
            row["final_sha256"] = rt.sha(final)
        pt.save({"state_dict": c["model"].state_dict(), "optimizer": c["opt"].state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        T1.append_lines(log_path, [row])
        print(json.dumps({"run": name, **row}), flush=True)
    result = result_record(arm, seed, run, c, paths, started)
    N.write_json(run / "TRAIN_RESULT.json", result)
    last.unlink(missing_ok=True)
    return result


def code_hashes() -> dict:
    mods = {"dt1_train.py": Path(__file__), "dt1_model.py": Path(DM.__file__), "dt1_channels.py": Path(CH.__file__),
            "d1_train.py": Path(T1.__file__), "match_train.py": Path(MT.__file__), "match_schedule.py": Path(MS.__file__),
            "nds_schedule.py": Path(SCH.__file__), "adapt_train.py": Path(at.__file__), "r2_train.py": Path(rt.__file__)}
    return {k: N.sha256_file(p) for k, p in mods.items()}


def result_record(arm: str, seed: int, run: Path, c: dict, paths: dict, started: float) -> dict:
    import torch
    final, curves = run / "final.pt", run / "curves.jsonl"
    rows_log = [json.loads(x) for x in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    det = T1.detector_from_curves(curves)
    if det != c["detector"].record():
        raise SystemExit("detector: online record differs from the curves file")
    return {"run": run.name, "experiment": EXPERIMENT, "recipe": RECIPE, "arm": arm, "seed": seed,
            "channels": DM.CHANNELS[arm], "status": "COMPLETE", "final": str(final), "final_sha256": rt.sha(final),
            "final_bytes": final.stat().st_size, "epochs": rows_log, "detector": det,
            "collapsed_by_detector": bool(det["on_at_end"]), "curves_sha256": N.sha256_file(curves),
            "plan_summary": MT.plan_summary(seed, c["ctx"]), "stem_final": stem_record(c["model"]),
            "encoder": (str(paths["encoder"]), N.sha256_file(paths["encoder"])) if paths.get("encoder") else None,
            "start": {"registry_id": D.STARTS[START_ARM]["registry_id"], "path": D.STARTS[START_ARM]["path"],
                      "sha256": D.STARTS[START_ARM]["sha256"]},
            "updates": MS.N_UPDATES, "lower_saxony_updates": MS.N_UPDATES,
            "lr": "d1_train.lr_at('F1') = nds_schedule.lr_at (cosine 1e-4 -> 1e-6 over 5,116 updates, no warm-up)",
            "loss": "F1: BCE + 1 - Dice; BCE only on empty targets (d1_train.update)",
            "tables_sha256": {"ls_table": N.sha256_file(paths["ls_table"])}, "code_sha256": code_hashes(),
            "wall_s": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "attribution": ATTRIBUTION, **(wt.env_record() if torch.cuda.is_available() else {})}


def plan(seed: int, paths: dict, out: Path) -> dict:
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} outside DENSE-TERRAIN-1")
    ctx = MT.ls_context(paths)
    res = {"experiment": EXPERIMENT, "seed": seed, "plan_summary": MT.plan_summary(seed, ctx),
           "code_sha256": code_hashes(), "utc": N.utc()}
    res["plan_digest"] = N.digest(res["plan_summary"])
    N.write_json(out, res)
    return res


def smoke(arm: str, n: int, out: Path, paths: dict, device: str = "cuda") -> dict:
    """The first n updates of epoch 4 of seed 101 on the production code path. Never saved, never scored."""
    import torch
    seed = SEEDS[0]
    NT.check_constants()
    c = {"ctx": MT.ls_context(paths), "prod": paths.get("prod")}
    c["model"], c["opt"] = DM.build(arm, device, paths.get("encoder"))
    torch.manual_seed(seed)
    c["detector"] = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    row, recs = epoch_pass(arm, seed, MS.EPOCHS[0], c, f"SMOKE_{run_name(arm, seed)}", t0, device=device, limit=n)
    wall = time.monotonic() - t0
    losses = [None if r["skipped"] else r["loss_used"] for r in recs]
    rec = {"experiment": EXPERIMENT, "arm": arm, "seed": seed, "n": n, "device": device, "row": row, "losses": losses,
           "s_per_update_incl_startup": wall / max(n, 1), "all_finite": all(v is None or math.isfinite(v) for v in losses),
           "stem_end": stem_record(c["model"]), "code_sha256": code_hashes(), "utc": N.utc(),
           "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None}
    rec["status"] = "PASS" if rec["all_finite"] and len(recs) == n else "FAIL"
    N.write_json(out, rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "plan", "smoke"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--arm", choices=DM.ARMS)
    for k in ("ls-table", "ls-win", "ls-mask", "prod", "encoder", "run-root"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "train" or (a.cmd == "smoke" and a.device == "cuda"):
        T1.JT1.J.torch_flags()
    paths = {k: getattr(a, k) for k in ("ls_table", "ls_win", "ls_mask", "prod", "encoder") if getattr(a, k) is not None}
    if a.cmd == "train":
        r = train(a.arm, int(a.args[0]), paths, a.run_root)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "wall_s", "detector", "stem_final")}))
    elif a.cmd == "plan":
        r = plan(int(a.args[0]), paths, Path(a.args[1]))
        print(json.dumps({"seed": r["seed"], "plan_digest": r["plan_digest"]}))
    else:
        r = smoke(a.arm, int(a.args[0]), Path(a.args[1]), paths, a.device)
        print(json.dumps({k: r[k] for k in ("status", "arm", "n", "s_per_update_incl_startup", "all_finite",
                                            "peak_allocated_mib")}))


if __name__ == "__main__":
    main()
