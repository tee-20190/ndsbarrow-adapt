"""Experiment 1 v2 plans (PROTOCOL.md sections 3, 4, 5.4): roughness table, background groups, real and auxiliary
plans for the pilot seed and the eight full-test seeds. Written before any training; hashed in FREEZE.json.

  python -B work/counterfactual_2026-09-28/plans.py roughness   # ROUGHNESS_PRIVATE.json (CPU, parallel)
  python -B work/counterfactual_2026-09-28/plans.py build       # plans/PLAN_<seed>_PRIVATE.json for all 9 seeds
(WSL /home/tee-/adaf-h2h-venv, repo root)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work/aim_train_audit_2026-09-24"))
import cf2  # noqa: E402
import cont_train as ct  # noqa: E402

rt = cf2.rt
OUT = ROOT / "outputs/counterfactual_2026-09-28"
PLANS = OUT / "plans"
ROUGHNESS = OUT / "ROUGHNESS_PRIVATE.json"
MANIFEST = ct.MANIFESTS["CONT"]
MANIFEST_SHA256 = "f44a36bd83eb73083f87ff434efed135fcb35f0c398e942c28192ceccecb3acc"
PILOT_SEED = 2026092800
FULL_SEEDS = tuple(range(2026092801, 2026092809))
N_EPOCHS, N_FIT_POS = 4, 1279
STEPS_PER_EPOCH = 2 * N_FIT_POS
TOTAL_UPDATES = N_EPOCHS * STEPS_PER_EPOCH  # 10,232
LR_MAX, LR_MIN = 1e-4, 1e-6
MIN_PER_GROUP = 20
WINDOW_ATTEMPTS, EXAMPLE_ATTEMPTS = 32, 128


def sha(path: Path) -> str:
    return rt.sha(path)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lr_at(t: int) -> float:
    """Cosine per scheduled update t = 0..10,231, from 1e-4 to 1e-6 (PROTOCOL.md 4)."""
    if not 0 <= t < TOTAL_UPDATES:
        raise ValueError("scheduled update index out of range")
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t / (TOTAL_UPDATES - 1)))


def aux_scheduled(step: int, epoch: int) -> bool:
    return (step - epoch) % 4 == 0


def load_windows() -> tuple[list[dict], list[int], list[int], list[int]]:
    if sha(MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("R2 mask manifest changed")
    windows, fit_pos, fit_neg, hold = ct.load_windows(MANIFEST)
    if len(fit_pos) != N_FIT_POS or len(fit_neg) != 741 or len(hold) != 804:
        raise RuntimeError("unexpected R2 cohort sizes")
    if any(windows[i]["holdout"] for i in fit_pos + fit_neg) or not all(windows[i]["holdout"] for i in hold):
        raise RuntimeError("holdout isolation violated")
    return windows, fit_pos, fit_neg, hold


# ---------------------------------------------------------------- real plans: r2_train.epoch_plan, explicit seed
def negative_at(seed: int, g: int, fit_neg: list[int]) -> int:
    cycle, pos = divmod(g, len(fit_neg))
    return int(np.random.default_rng([seed, 1, cycle]).permutation(fit_neg)[pos])


def real_plan(seed: int, epoch: int, fit_pos: list[int], fit_neg: list[int]) -> list[tuple[int, int, int, int]]:
    rng = np.random.default_rng([seed, 0, epoch])
    plan = []
    for i, wi in enumerate(rng.permutation(fit_pos)):
        plan.append((int(wi), *rt.draw(rng)))
        plan.append((negative_at(seed, epoch * len(fit_pos) + i, fit_neg), *rt.draw(rng)))
    return plan


# ---------------------------------------------------------------- roughness (unclipped SLRM MAD, central 256 m)
def _mad_of(window_id: str) -> tuple[str, float]:
    dtm, profile = cf2.read_native(rt.RAW / f"{window_id}.tif")
    _, _, unclipped = cf2.relief_from_array(dtm, profile)
    return window_id, cf2.roughness(unclipped)


def roughness() -> dict:
    windows, _, fit_neg, hold = load_windows()
    ids = sorted({windows[i]["id"] for i in fit_neg} | {windows[i]["id"] for i in hold if windows[i]["role"] == "negative"})
    t0 = time.monotonic()
    with Pool(6) as pool:
        mads = dict(pool.map(_mad_of, ids, chunksize=8))
    res = {"rule": "MAD of unclipped SLRM (pinned 1 m -> 0.5 m view, radius 20 cells) over the central 512 x 512 "
                   "0.5 m cells, finite cells only; fitting negatives and TRAIN-holdout negatives",
           "mad_m": {k: mads[k] for k in ids}, "n": len(ids), "wall_s": time.monotonic() - t0,
           "manifest_sha256": MANIFEST_SHA256}
    OUT.mkdir(parents=True, exist_ok=True)
    ROUGHNESS.write_text(json.dumps(res, indent=0), encoding="utf-8")
    return {"n": len(ids), "wall_s": res["wall_s"]}


def groups(ids: list[str], mads: dict[str, float]) -> list[list[str]]:
    """Eligible (MAD >= 5 mm) ids sorted by (MAD, id), cut into four near-equal groups (lowest first)."""
    elig = sorted((i for i in ids if mads[i] >= cf2.MIN_MAD_M), key=lambda i: (mads[i], i))
    out = [list(g) for g in np.array_split(np.array(elig, dtype=object), 4)]
    if min(len(g) for g in out) < MIN_PER_GROUP:
        raise RuntimeError("a roughness group has fewer than 20 eligible windows")
    return out


def fitting_groups(windows: list[dict], fit_neg: list[int]) -> tuple[list[list[str]], dict]:
    mads = json.loads(ROUGHNESS.read_text(encoding="utf-8"))["mad_m"]
    ids = [windows[i]["id"] for i in fit_neg]
    g = groups(ids, mads)
    return g, {"eligible": sum(len(x) for x in g), "excluded_below_5mm": sum(mads[i] < cf2.MIN_MAD_M for i in ids),
               "group_sizes": [len(x) for x in g],
               "group_mad_ranges_m": [[round(mads[x[0]], 5), round(mads[x[-1]], 5)] for x in g]}


# ---------------------------------------------------------------- auxiliary plans
def blocked_for(window: dict) -> np.ndarray:
    dtm, _ = cf2.read_native(rt.RAW / f"{window['id']}.tif")
    return cf2.blocked_native(dtm, np.load(ROOT / window["mask_path"])["label"])


def aux_example(seed: int, epoch: int, i: int, grp: list[list[str]], blocked: dict[str, np.ndarray],
                table: list[dict]) -> dict:
    rng = np.random.default_rng([seed, 2, epoch, i])
    g = int(rng.integers(4))
    wid = grp[g][int(rng.integers(len(grp[g])))]
    geom, meta = cf2.draw_geometry(rng, table)
    k = int(rng.integers(8))
    rect = cf2.feasible_rectangle(geom)
    if rect is None:
        raise RuntimeError(f"seed {seed} epoch {epoch} aux {i}: geometry cannot fit the crop")
    used = 0
    while used < EXAMPLE_ATTEMPTS:
        placed, n, dz = cf2.place(rng, geom, blocked[wid], min(WINDOW_ATTEMPTS, EXAMPLE_ATTEMPTS - used), rect)
        used += n
        if placed is not None:
            return {"window_id": wid, "group": g, "k": k, "implant": cf2.implant_to_dict(placed), **meta,
                    "attempts": used, "dz_min_m": float(dz.min()), "dz_max_m": float(dz.max()),
                    "peak_to_trough_m": float(dz.max() - dz.min())}
        wid = grp[g][int(rng.integers(len(grp[g])))]
    raise RuntimeError(f"seed {seed} epoch {epoch} aux {i}: no placement in {EXAMPLE_ATTEMPTS} attempts; preflight stops")


def build_seed(seed: int, windows: list[dict], fit_pos: list[int], fit_neg: list[int], grp: list[list[str]],
               blocked: dict[str, np.ndarray], table: list[dict]) -> dict:
    by_id = {w["id"]: i for i, w in enumerate(windows)}
    real = [real_plan(seed, e, fit_pos, fit_neg) for e in range(N_EPOCHS)]
    aux: dict[str, dict[str, dict]] = {}
    for e in range(N_EPOCHS):
        aux[str(e)] = {}
        for s in range(STEPS_PER_EPOCH):
            if aux_scheduled(s, e):
                ex = aux_example(seed, e, (s - e) // 4, grp, blocked, table)
                ex["wi"] = by_id[ex["window_id"]]
                aux[str(e)][str(s)] = ex
    return {"seed": seed, "epochs": N_EPOCHS, "real": [[list(p) for p in ep] for ep in real], "aux": aux,
            "n_aux": sum(len(v) for v in aux.values())}


def build() -> dict:
    windows, fit_pos, fit_neg, _ = load_windows()
    grp, gstats = fitting_groups(windows, fit_neg)
    by_id = {w["id"]: w for w in windows}
    blocked = {wid: blocked_for(by_id[wid]) for g in grp for wid in g}
    table = cf2.design_table()
    PLANS.mkdir(parents=True, exist_ok=True)
    summary = {"groups": gstats, "design_sizes_sha256": sha(cf2.DESIGN), "seeds": {}}
    for seed in (PILOT_SEED, *FULL_SEEDS):
        t0 = time.monotonic()
        plan = build_seed(seed, windows, fit_pos, fit_neg, grp, blocked, table)
        text = json.dumps(plan, separators=(",", ":"))
        path = PLANS / f"PLAN_{seed}_PRIVATE.json"
        path.write_text(text, encoding="utf-8")
        fams: dict[str, int] = {}
        for ep in plan["aux"].values():
            for ex in ep.values():
                fams[ex["family"] + "/" + ex["subtype"]] = fams.get(ex["family"] + "/" + ex["subtype"], 0) + 1
        summary["seeds"][str(seed)] = {"sha256": sha(path), "n_aux": plan["n_aux"], "families": fams,
                                       "max_attempts": max(ex["attempts"] for ep in plan["aux"].values()
                                                           for ex in ep.values()),
                                       "wall_s": round(time.monotonic() - t0, 1)}
        print(json.dumps({"seed": seed, **summary["seeds"][str(seed)]}), flush=True)
    (OUT / "PLANS_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_plan(seed: int) -> dict:
    return json.loads((PLANS / f"PLAN_{seed}_PRIVATE.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"roughness": roughness, "build": build}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    print(json.dumps(fn(), indent=1))
