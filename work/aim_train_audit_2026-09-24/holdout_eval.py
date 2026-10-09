"""PREREG.md B selection: holdout primary recall for every saved epoch of one arm, then the declared selection.

One pass over the 804 holdout windows; the SLRM tile input is computed once and every checkpoint predicts on it
(identity only), through tta_run's copy of the frozen tiling, validity masking and vectorising. Scoring is
score_holdout.py's (506 parts, recall at 0.5 / 1 / 2 negative-window proposals per km²).

  python -B work/aim_train_audit_2026-09-24/holdout_eval.py ARM          # CONT or AIMX
  python -B work/aim_train_audit_2026-09-24/holdout_eval.py ARM score    # rescore saved objects only
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
sys.path.insert(0, str(HERE))
import cont_train as ct  # noqa: E402
import score_holdout as sh  # noqa: E402
import tta_run as tr  # noqa: E402

OUT = ct.OUT


def checkpoints(arm: str) -> dict[str, Path]:
    rows = [json.loads(l) for l in (OUT / "training" / arm / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if [r["epoch"] for r in rows] != list(range(ct.FIRST_EPOCH, ct.FIRST_EPOCH + ct.N_EPOCHS)):
        raise SystemExit(f"{arm}: training incomplete")
    ck = {f"e{r['epoch']}": ROOT / r["checkpoint"] for r in rows}
    for r in rows:
        if ct.sha(ROOT / r["checkpoint"]) != r["checkpoint_sha256"]:
            raise RuntimeError(f"{r['checkpoint']} changed since training")
    return ck


def predictor(models: dict[str, torch.nn.Module]):
    def predict(a: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.tensor(np.repeat(a[None, :, :], 3, axis=0), dtype=torch.float32, device="cuda")[None]
        with torch.inference_mode():
            return {k: torch.sigmoid(m(x)["out"])[0, 1].cpu().numpy() for k, m in models.items()}
    return predict


def infer(arm: str) -> dict:
    ck = checkpoints(arm)
    if ct.remaining_s() <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    started = time.monotonic()
    tr.ac.check_frozen()
    names = sorted(ck)
    models, _ = tr.ac.load_models([ck[n] for n in names])
    predict = predictor(dict(zip(names, models)))
    torch.set_num_threads(4)
    objs = {n: {} for n in names}
    wins = tr.holdout_windows()
    try:
        for w in wins:
            rec = tr.objects_multi(ROOT / "outputs/campaign_2026-09-22/train_raw" / f"{w['id']}.tif", predict, names)
            for n in names:
                objs[n][w["id"]] = rec["arms"][n]["objects"]
    finally:
        ct.charge(f"holdout_{arm}", time.monotonic() - started, {"windows": len(wins), "checkpoints": names})
    receipt = {"arm": arm, "windows": len(wins), "checkpoints": {n: ct.sha(p) for n, p in ck.items()},
               "wall_s": time.monotonic() - started, "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    (OUT / f"HOLDOUT_OBJECTS_{arm}_PRIVATE.json").write_text(json.dumps({"receipt": receipt, "objects": objs}),
                                                              encoding="utf-8")
    return receipt


def score(arm: str) -> dict:
    rec = json.loads((OUT / f"HOLDOUT_OBJECTS_{arm}_PRIVATE.json").read_text(encoding="utf-8"))
    s = sh.setup()
    summ = {n: sh.arm_summary(sh.ranked_pool(o, s), s) for n, o in rec["objects"].items()}
    best = max(summ, key=lambda n: (round(summ[n]["primary"], 12), int(n[1:])))  # tie -> later epoch
    table = {n: {"primary": round(v["primary"], 4), **{f"d{d}": round(v[f"d{d}"]["recall"], 4) for d in sh.DENSITIES},
                 "enclosure_d0.5_1_2": [v[f"d{d}"]["by_class"]["enclosure"] for d in sh.DENSITIES],
                 "neg_per_km2": round(v["negative_per_km2"], 2)} for n, v in sorted(summ.items())}
    result = {"arm": arm, "eligible_parts": len(s["eligible"]), "table": table, "selected": best,
              "selected_checkpoint": rec["receipt"]["checkpoints"][best],
              "rule": "max primary recall (mean at 0.5/1/2 negative proposals per km2); tie -> later epoch",
              "receipt": rec["receipt"]}
    (OUT / f"HOLDOUT_SELECTION_{arm}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    arm = sys.argv[1]
    if len(sys.argv) < 3:
        print(json.dumps(infer(arm)))
    r = score(arm)
    print(json.dumps({k: r[k] for k in ("arm", "table", "selected")}, indent=1))
