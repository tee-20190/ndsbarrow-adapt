"""Experiment 1 v2 real-landscape scoring (PROTOCOL.md 8.4 G-R, 8.6, 9): shared C/T inference passes, frozen scorers.

Inference is tta_run's frozen tiling / validity masking / vectorising with holdout_eval.predictor (one forward per
model per tile), the half-open interior filter and one writer per arm directory, as bgscale's bg_eval.py.
Guards: SCREEN source hashes (adaf_configs.source_for); CONT e41 re-inferred on the first SCREEN block and on the
first byte-matching stage-2 block must reproduce its recorded objects; every cached stage-2 mosaic must match the
bgscale cache receipt; CONT e41 must reproduce its recorded holdout objects in every holdout window. Static CONT's
reference objects: recorded SCREEN proposals, bgscale's stage-2 CONT objects on the same cached bytes, recorded
holdout objects. Scoring: bg_eval's frozen SCREEN / stage-2 scorer functions (screen_scorer.build_pool, match_greedy,
K = 323 / 647 / 1,294 and 909 / 1,817 / 3,635), the stage-2 independent recount, score_holdout, aim_analyze.

  python -B work/counterfactual_2026-09-28/score_pairs.py screen SEED    # GPU
  python -B work/counterfactual_2026-09-28/score_pairs.py stage2 SEED    # GPU
  python -B work/counterfactual_2026-09-28/score_pairs.py holdout SEED   # GPU
  python -B work/counterfactual_2026-09-28/score_pairs.py score SEED     # CPU -> results/SCORES_<SEED>_PRIVATE.json
(WSL /home/tee-/adaf-h2h-venv, repo root)
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("work/campaign_2026-09-22", "work/tta_ensemble_2026-09-24", "work/aim_train_audit_2026-09-24",
          "work/anneal_2026-09-25", "work/stage2_2026-09-24", "work/aim_2026-09-24", "work/bgscale_2026-09-26"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(HERE))
import bg_eval as be  # noqa: E402  (frozen bgscale scorer functions; nothing in it is run as a stage)
import holdout_eval as he  # noqa: E402
import ledger as lg  # noqa: E402
import plans  # noqa: E402
import stage2_run as s2  # noqa: E402
import tta_run as tr  # noqa: E402

OUT = plans.OUT
START = ROOT / "outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt"
START_SHA256 = "91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383"
CACHE = ROOT / "outputs/bgscale_2026-09-26/stage2_cache"
CACHE_RECEIPT = ROOT / "outputs/bgscale_2026-09-26/STAGE2_CACHE_RECEIPT.json"
CONT_STAGE2_ON_CACHE = ROOT / "outputs/bgscale_2026-09-26/stage2/CONT"
RECORDED_CONT_SCREEN = be.RECORDED_CONT_SCREEN
RECORDED_STAGE2 = be.RECORDED_STAGE2
RECORDED_CONT_HOLDOUT = ROOT / "outputs/aim_train_audit_2026-09-24/HOLDOUT_OBJECTS_CONT_PRIVATE.json"
DENS = be.DENS
SATURATED = 0.9999
REAL_GATE_MIN = -5  # PROTOCOL.md 8.4 G-R (A1)


def read(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def stage_of(seed: int) -> str:
    return "pilot" if seed == plans.PILOT_SEED else "full"


def arm_checkpoints(seed: int) -> dict[str, Path]:
    out = {}
    for arm in ("C", "T"):
        res = read(plans.OUT / "training" / f"{seed}_{arm}" / "TRAIN_RESULT.json")
        path = ROOT / res["final"]
        if res["status"] != "COMPLETE" or plans.sha(path) != res["final_sha256"]:
            raise RuntimeError(f"{seed} {arm}: training incomplete or final weights changed")
        out[arm] = path
    return out


def load_models(ck: dict[str, Path]) -> dict:
    tr.ac.check_frozen()
    names = sorted(ck)
    models, _ = tr.ac.load_models([ck[n] for n in names])
    return dict(zip(names, models))


def screen_dir(seed: int, arm: str) -> Path:
    return OUT / "screen" / str(seed) / arm


def stage2_dir(seed: int, arm: str) -> Path:
    return OUT / "stage2" / str(seed) / arm


def write_props(path: Path, arm: str, block_id: str, ck_sha: str, rows: list[dict], rec: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"status": "INFERENCE_COMPLETE", "config": arm, "block_id": block_id, "checkpoint_sha256": ck_sha,
            "proposal_count": len(rows), "objects": rows, "valid_km2": rec["valid_km2"], "patches": rec["patches"],
            "map_sha256": rec["arms"][arm]["map_sha256"], **extra}
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")


def _budget(stage: str, started: float) -> None:
    if time.monotonic() - started >= lg.remaining_s(stage):
        raise SystemExit(f"{stage} GPU ceiling reached mid-pass; INCOMPLETE")


# ---------------------------------------------------------------- inference (GPU)
def infer_screen(seed: int) -> dict:
    import torch
    stage = stage_of(seed)
    lg.verify_frozen(stage)
    ck = arm_checkpoints(seed)
    ck_sha = {a: plans.sha(p) for a, p in ck.items()}
    blocks = tr.ac.screen_blocks()
    first = blocks[0]["id"]
    started, done = time.monotonic(), 0
    models = load_models({**ck, "CONT": START})
    pred = {True: he.predictor(models), False: he.predictor({a: models[a] for a in ck})}
    torch.set_num_threads(4)
    try:
        for b in blocks:
            bid = b["id"]
            targets = {a: screen_dir(seed, a) / "proposals" / f"{bid}.json" for a in ck}
            if all(t.exists() for t in targets.values()):
                continue
            _budget(stage, started)
            guard = bid == first
            names = sorted(models) if guard else sorted(ck)
            t0 = time.monotonic()
            rec = tr.objects_multi(tr.ac.source_for(bid), pred[guard], names, interior=b["interior"])
            rows = {a: [dict(p, source_block=bid) for p in rec["arms"][a]["objects"] if tr.ac.half_open(p, b["interior"])]
                    for a in names}
            if guard and not tr.same_objects(rows["CONT"], read(RECORDED_CONT_SCREEN / "proposals" / f"{bid}.json")["objects"]):
                raise RuntimeError(f"{bid}: CONT e41 does not reproduce its recorded SCREEN objects; stop")
            for a in ck:
                write_props(targets[a], a, bid, ck_sha[a], rows[a], rec, {"wall_seconds": time.monotonic() - t0,
                                                                          "cont_guard_block": guard})
            done += 1
            print(json.dumps({"block": bid, "guard": guard, "counts": {a: len(rows[a]) for a in names},
                              "wall_s": round(time.monotonic() - t0, 1)}), flush=True)
    finally:
        lg.charge(stage, f"screen_{seed}", time.monotonic() - started, {"blocks_run": done})
    return {"seed": seed, "blocks_run": done, "cont_guard_block": first}


def cache_receipt() -> dict:
    rec = read(CACHE_RECEIPT)
    if rec["status"] != "COMPLETE":
        raise RuntimeError("stage-2 cache receipt is not COMPLETE")
    return rec["per_block"]


def infer_stage2(seed: int) -> dict:
    import torch
    stage = stage_of(seed)
    lg.verify_frozen(stage)
    per = cache_receipt()
    ck = arm_checkpoints(seed)
    ck_sha = {a: plans.sha(p) for a, p in ck.items()}
    blocks = s2.blocks()
    repro = next(b["id"] for b in blocks if per[b["id"]]["matches_cont_recorded"])
    started, done = time.monotonic(), 0
    models = load_models({**ck, "CONT": START})
    pred = {True: he.predictor(models), False: he.predictor({a: models[a] for a in ck})}
    torch.set_num_threads(4)
    try:
        for b in blocks:
            bid = b["id"]
            targets = {a: stage2_dir(seed, a) / "proposals" / f"{bid}.json" for a in ck}
            if all(t.exists() for t in targets.values()):
                continue
            _budget(stage, started)
            src = CACHE / f"{bid}.tif"
            if plans.sha(src) != per[bid]["dtm_sha256"]:
                raise RuntimeError(f"{bid}: cached mosaic changed")
            guard = bid == repro
            names = sorted(models) if guard else sorted(ck)
            t0 = time.monotonic()
            rec = tr.objects_multi(src, pred[guard], names, interior=b["interior"])
            rows = {a: [dict(p, source_block=bid) for p in rec["arms"][a]["objects"] if tr.ac.half_open(p, b["interior"])]
                    for a in names}
            if guard and not tr.same_objects(rows["CONT"], read(RECORDED_STAGE2 / "CONT" / f"{bid}.json")["objects"]):
                raise RuntimeError(f"{bid}: CONT e41 does not reproduce its recorded stage-2 objects; stop")
            for a in ck:
                write_props(targets[a], a, bid, ck_sha[a], rows[a], rec,
                            {"source_sha256": per[bid]["dtm_sha256"], "wall_seconds": time.monotonic() - t0,
                             "cont_guard_block": guard})
            done += 1
            print(json.dumps({"block": bid, "guard": guard, "counts": {a: len(rows[a]) for a in names},
                              "wall_s": round(time.monotonic() - t0, 1)}), flush=True)
    finally:
        lg.charge(stage, f"stage2_{seed}", time.monotonic() - started, {"blocks_run": done})
    return {"seed": seed, "blocks_run": done, "cont_guard_block": repro}


def holdout_path(seed: int) -> Path:
    return OUT / "holdout" / f"HOLDOUT_OBJECTS_{seed}_PRIVATE.json"


def infer_holdout(seed: int) -> dict:
    import torch
    stage = stage_of(seed)
    lg.verify_frozen(stage)
    ck = arm_checkpoints(seed)
    recorded = read(RECORDED_CONT_HOLDOUT)["objects"]["e41"]
    started = time.monotonic()
    models = load_models({**ck, "CONT": START})
    predict = he.predictor(models)
    torch.set_num_threads(4)
    objs: dict[str, dict] = {a: {} for a in models}
    wins = tr.holdout_windows()
    try:
        for w in wins:
            _budget(stage, started)
            rec = tr.objects_multi(be.TRAIN_RAW / f"{w['id']}.tif", predict, sorted(models))
            if not tr.same_objects(rec["arms"]["CONT"]["objects"], recorded[w["id"]]):
                raise RuntimeError(f"{w['id']}: CONT e41 does not reproduce its recorded holdout objects; stop")
            for a in models:
                objs[a][w["id"]] = rec["arms"][a]["objects"]
    finally:
        lg.charge(stage, f"holdout_{seed}", time.monotonic() - started, {"windows": len(objs["CONT"])})
    receipt = {"windows": len(wins), "checkpoints": {a: plans.sha(p) for a, p in {**ck, "CONT": START}.items()},
               "wall_s": time.monotonic() - started, "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    holdout_path(seed).parent.mkdir(parents=True, exist_ok=True)
    holdout_path(seed).write_text(json.dumps({"receipt": receipt, "objects": objs}), encoding="utf-8")
    return receipt


# ---------------------------------------------------------------- secondary decoder (A7)
def nms_secondary(rows: list[dict], radius: float = 60.0) -> list[dict]:
    """score_staged.global_nms with suppression order (-score, -mean_probability, id); otherwise identical."""
    return _nms(rows, radius, lambda d: (-float(d["score"]), -float(d["mean_probability"]), str(d["id"])))


def _nms(rows: list[dict], radius: float, key) -> list[dict]:
    radius2 = radius * radius
    bins: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    kept = []
    for row in sorted(rows, key=key):
        x, y = float(row["easting"]), float(row["northing"])
        bx, by = math.floor(x / radius), math.floor(y / radius)
        near = (p for nx in range(bx - 1, bx + 2) for ny in range(by - 1, by + 2) for p in bins.get((nx, ny), ()))
        if any((x - px) ** 2 + (y - py) ** 2 <= radius2 for px, py in near):
            continue
        kept.append(row)
        bins[(bx, by)].append((x, y))
    return kept


class _SecondaryStage:
    def __init__(self, stage) -> None:
        self._stage = stage

    def __getattr__(self, name: str):
        return getattr(self._stage, name)

    @staticmethod
    def global_nms(rows: list[dict]) -> list[dict]:
        return nms_secondary(rows)


@contextlib.contextmanager
def secondary_decoder():
    import score_screen as sc
    import screen_scorer as ss
    if sc.ss is not ss:
        raise RuntimeError("score_screen uses a different screen_scorer module object")
    original = ss.frozen_inputs

    def patched():
        data, blocks, stage = original()
        return data, blocks, _SecondaryStage(stage)

    ss.frozen_inputs = patched
    try:
        yield
    finally:
        ss.frozen_inputs = original


# ---------------------------------------------------------------- scoring (CPU)
def cont_stage2_rows() -> dict[str, list[dict]]:
    per = cache_receipt()
    rows = {}
    for b in s2.blocks():
        body = read(CONT_STAGE2_ON_CACHE / "proposals" / f"{b['id']}.json")
        if body["source_sha256"] != per[b["id"]]["dtm_sha256"]:
            raise RuntimeError(f"{b['id']}: bgscale CONT objects are not on the cached bytes")
        if body["checkpoint_sha256"] != START_SHA256:
            raise RuntimeError(f"{b['id']}: bgscale CONT objects do not come from CONT e41")
        rows[b["id"]] = [dict(o, source_block=b["id"]) for o in body["objects"]]
    return rows


def real_tables(seed: int) -> tuple[dict, dict]:
    sdirs = {"C": screen_dir(seed, "C"), "T": screen_dir(seed, "T"), "CONT": RECORDED_CONT_SCREEN}
    sh = be.screen_hits(sdirs)
    if tuple(sh["CONT"][d]["H"] for d in DENS) != be.EXPECT_SCREEN["CONT"]:
        raise SystemExit("recorded CONT e41 SCREEN proposals no longer score 39/45/52")
    th = be.stage2_hits({"C": be.stage2_rows(stage2_dir(seed, "C")), "T": be.stage2_rows(stage2_dir(seed, "T")),
                         "CONT": cont_stage2_rows(),
                         "CONT_recorded": be.stage2_rows(RECORDED_STAGE2 / "CONT", recorded_layout=True)})
    if tuple(th["CONT_recorded"][d]["H"] for d in DENS) != be.EXPECT_STAGE2["CONT"]:
        raise SystemExit("recorded CONT e41 stage-2 proposals no longer score 50/55/59")
    return sh, th


def saturation(seed: int) -> dict:
    out = {}
    for arm in ("C", "T"):
        scores = [float(o["score"]) for d in (screen_dir(seed, arm), stage2_dir(seed, arm))
                  for f in sorted((d / "proposals").glob("*.json")) for o in read(f)["objects"]]
        out[arm] = {"proposals": len(scores), "saturated_share": float(np.mean(np.array(scores) >= SATURATED))}
    return out


def holdout_table(seed: int) -> dict:
    import score_holdout as shd
    rec = read(holdout_path(seed))
    s = shd.setup()
    pools = {n: shd.ranked_pool(o, s) for n, o in rec["objects"].items()}
    summ = {n: shd.arm_summary(p, s) for n, p in pools.items()}
    return {n: {"primary": round(v["primary"], 4), **{f"d{d}": round(v[f"d{d}"]["recall"], 4) for d in shd.DENSITIES},
                "negative_window_proposals_per_km2_(workload_proxy)": round(v["negative_per_km2"], 3)}
            for n, v in sorted(summ.items())}


def score(seed: int) -> dict:
    sh, th = real_tables(seed)
    pooled = be.pooled_table(sh, th, ["C", "T", "CONT"])
    delta = be.pooled_delta(sh, th, "T", "C")
    with secondary_decoder():
        sh2, th2 = real_tables(seed)
    res = {"seed": seed,
           "pooled": pooled,
           "T_minus_C": {d: int(delta[d].sum()) for d in DENS},
           "T_minus_CONT": {d: pooled["T"][d]["H"] - pooled["CONT"][d]["H"] for d in DENS},
           "by_block_T_minus_C_d0.5": delta["d0.5"].tolist(),
           "blocks": sh["_blocks"] + th["_blocks"],
           "strata": ["SCREEN"] * len(sh["_blocks"]) + ["STAGE2"] * len(th["_blocks"]),
           "gate_G_R": {"statistic": int(delta["d0.5"].sum()), "min": REAL_GATE_MIN,
                        "pass": int(delta["d0.5"].sum()) >= REAL_GATE_MIN},
           "secondary_decoder_pooled": be.pooled_table(sh2, th2, ["C", "T", "CONT"]),
           "saturation": saturation(seed),
           "recount": be.recount({a: screen_dir(seed, a) for a in ("C", "T")}, {a: stage2_dir(seed, a) for a in ("C", "T")}),
           "aim_screen_(preservation_proxy)": be.aim_yield({"C": screen_dir(seed, "C"), "T": screen_dir(seed, "T"),
                                                            "CONT": RECORDED_CONT_SCREEN}, "T", "C"),
           "holdout": holdout_table(seed) if holdout_path(seed).exists() else "INCOMPLETE",
           "ids": {part: {a: {d: v[a][d]["ids"] for d in DENS} for a in ("C", "T", "CONT")}
                   for part, v in (("screen", sh), ("stage2", th))}}
    for a in ("C", "T"):
        rc = res["recount"]
        if [rc[f"stage1_{a}"][f"K{k}"] for k in (323, 1294)] != [sh[a]["d0.5"]["H"], sh[a]["d2.0"]["H"]] or \
                [rc[f"stage2_{a}"][f"K{k}"] for k in (909, 3635)] != [th[a]["d0.5"]["H"], th[a]["d2.0"]["H"]]:
            res["recount_disagrees"] = True
    path = OUT / "results" / f"SCORES_{seed}_PRIVATE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return {k: v for k, v in res.items() if k not in ("ids", "by_block_T_minus_C_d0.5", "blocks", "strata")}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"screen": infer_screen, "stage2": infer_stage2, "holdout": infer_holdout, "score": score}.get(cmd)
    if fn is None or len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(fn(int(sys.argv[2])), indent=1))
