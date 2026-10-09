"""PACKET.md holdout stage: every saved epoch of both arms on the 804 selection-holdout windows, then the primary.

One pass: the SLRM tile input is computed once per window and all eight checkpoints predict on it (identity only),
through tta_run's copy of the frozen tiling, validity masking and vectorising. Scoring is score_holdout.py's
(506 parts; recall at 0.5 / 1 / 2 negative-window proposals per km2).
Primary (declared): ANNEAL e45 - CONST e45 holdout primary recall; SUPPORTED >= +0.010, HARM <= -0.010, else NULL.
Paired window bootstrap (positive and negative windows resampled separately, 2,000 draws, seed 20260925).

  python -B work/anneal_2026-09-25/anneal_holdout.py          # inference + scoring
  python -B work/anneal_2026-09-25/anneal_holdout.py score    # rescore saved objects
(WSL adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work/tta_ensemble_2026-09-24"))
sys.path.insert(0, str(ROOT / "work/aim_train_audit_2026-09-24"))
sys.path.insert(0, str(HERE))
import anneal_train as at  # noqa: E402
import holdout_eval as he  # noqa: E402
import score_holdout as sh  # noqa: E402
import tta_run as tr  # noqa: E402

OUT = at.OUT
FINAL = f"e{at.FIRST_EPOCH + at.N_EPOCHS - 1}"
BOOT_N, BOOT_SEED = 2000, 20260925


def checkpoints() -> dict[str, Path]:
    ck = {}
    for arm in at.ARMS:
        rows = [json.loads(l) for l in (OUT / "training" / arm / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        if [r["epoch"] for r in rows] != list(range(at.FIRST_EPOCH, at.FIRST_EPOCH + at.N_EPOCHS)):
            raise SystemExit(f"{arm}: training incomplete")
        for r in rows:
            p = ROOT / r["checkpoint"]
            if at.ct.sha(p) != r["checkpoint_sha256"]:
                raise RuntimeError(f"{r['checkpoint']} changed since training")
            ck[f"{arm}_e{r['epoch']}"] = p
    return ck


def infer() -> dict:
    ck = checkpoints()
    if at.remaining_s() <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    started = time.monotonic()
    tr.ac.check_frozen()
    names = sorted(ck)
    models, _ = tr.ac.load_models([ck[n] for n in names])
    predict = he.predictor(dict(zip(names, models)))
    torch.set_num_threads(4)
    objs = {n: {} for n in names}
    wins = tr.holdout_windows()
    try:
        for w in wins:
            if time.monotonic() - started >= at.remaining_s():
                raise SystemExit("GPU ceiling reached mid-holdout; comparison INCOMPLETE")
            rec = tr.objects_multi(ROOT / "outputs/campaign_2026-09-22/train_raw" / f"{w['id']}.tif", predict, names)
            for n in names:
                objs[n][w["id"]] = rec["arms"][n]["objects"]
    finally:
        at.charge("holdout", time.monotonic() - started, {"windows": len(objs[names[0]]), "checkpoints": names})
    receipt = {"windows": len(wins), "checkpoints": {n: at.ct.sha(p) for n, p in ck.items()},
               "wall_s": time.monotonic() - started, "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    (OUT / "HOLDOUT_OBJECTS_PRIVATE.json").write_text(json.dumps({"receipt": receipt, "objects": objs}),
                                                      encoding="utf-8")
    return receipt


def paired_bootstrap(pa: list[dict], pb: list[dict], s: dict) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    pos = [i for i in range(len(s["wins"])) if i not in s["neg_area"]]
    neg = sorted(s["neg_area"])
    diffs = []
    for _ in range(BOOT_N):
        w = np.zeros(len(s["wins"]))
        np.add.at(w, rng.choice(pos, len(pos)), 1)
        np.add.at(w, rng.choice(neg, len(neg)), 1)
        a = np.mean([sh.recall_at(pa, s, d, w)[0] for d in sh.DENSITIES])
        b = np.mean([sh.recall_at(pb, s, d, w)[0] for d in sh.DENSITIES])
        diffs.append(a - b)
    return {"median": float(np.median(diffs)), "q05": float(np.quantile(diffs, .05)),
            "q95": float(np.quantile(diffs, .95)), "share_gt_0": float(np.mean(np.array(diffs) > 0))}


def score() -> dict:
    rec = json.loads((OUT / "HOLDOUT_OBJECTS_PRIVATE.json").read_text(encoding="utf-8"))
    s = sh.setup()
    pools = {n: sh.ranked_pool(o, s) for n, o in rec["objects"].items()}
    summ = {n: sh.arm_summary(p, s) for n, p in pools.items()}
    table = {n: {"primary": round(v["primary"], 4), **{f"d{d}": round(v[f"d{d}"]["recall"], 4) for d in sh.DENSITIES},
                 "all": round(v["all"]["recall"], 4),
                 "enclosure_d0.5_1_2": [v[f"d{d}"]["by_class"]["enclosure"] for d in sh.DENSITIES],
                 "barrow_d0.5_1_2": [v[f"d{d}"]["by_class"]["barrow"] for d in sh.DENSITIES],
                 "neg_per_km2": round(v["negative_per_km2"], 2)} for n, v in sorted(summ.items())}
    a, b = f"ANNEAL_{FINAL}", f"CONST_{FINAL}"
    delta = summ[a]["primary"] - summ[b]["primary"]
    outcome = "SUPPORTED" if delta >= .010 - 1e-12 else "HARM" if delta <= -.010 + 1e-12 else "NULL"
    result = {"eligible_parts": len(s["eligible"]), "table": table, "primary_pair": [a, b],
              "primary_delta": round(delta, 4), "primary_outcome": outcome,
              "rule": "ANNEAL final - CONST final holdout primary: SUPPORTED >= +0.010, HARM <= -0.010, else NULL",
              "bootstrap_90": paired_bootstrap(pools[a], pools[b], s),
              "reference_recorded": {"CONT_e41": 0.7688, "CONT_e40_45_mean": 0.7387, "R2_e38": 0.7372},
              "receipt": rec["receipt"]}
    (OUT / "HOLDOUT_RESULT.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps(infer()))
    r = score()
    print(json.dumps({k: r[k] for k in ("table", "primary_delta", "primary_outcome", "bootstrap_90")}, indent=1))
