"""JOINT-1 chain: shared paths, seeds, arms and small helpers (no side effects at import).

Design: work/joint_campaign_2026-10-03/PACKET.md. Reused unchanged: the AMZ-ADAPT recipe (adapt_train.py), HR1's
welsh_train helpers, Welsh staging/inference (HR1 chain) and Irish staging/inference (HR3 chain), the frozen English
holdout path (tta_run / holdout_eval) and the HR1 scorer (region_score.py / hier_uncertainty.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent              # work/joint_campaign_2026-10-03/chain
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[1]
HR1 = ROOT / "work/holdout_region_2026-10"
HR1_CHAIN = HR1 / "chain"
HR1_STEP2 = HR1 / "step2"
HR3_CHAIN = ROOT / "work/hr3_ireland_2026-10-02/step2/chain"
PRIV = ROOT / "outputs/joint_campaign_2026-10-03"   # private outputs (coordinates): never committed

# Frozen regional inputs (bound by HR1 FREEZE_T2 and HR3 FREEZE a67bbb37)
W_FIT = HR1_CHAIN / "candidates/FIT_TABLES_CANDIDATE.json"
W_DEV = HR1_CHAIN / "candidates/DEV_MANIFEST_CANDIDATE.json"
W_PLAN = HR1_CHAIN / "candidates/STAGING_PLAN_CANDIDATE.json"
I_FIT = ROOT / "outputs/hr3_ireland_2026-10-02/step2/FIT_TABLES_PRIVATE.json"
I_DEV = ROOT / "outputs/hr3_ireland_2026-10-02/step2/DEV_MANIFEST_PRIVATE.json"
I_PLAN = ROOT / "outputs/hr3_ireland_2026-10-02/step2/STAGING_PLAN_PRIVATE.json"
SUBSETS = PRIV / "audit/SUBSETS_PRIVATE.json"       # audit_mixture.py: default_rng([seed, 30]) subsets per seed
AUDIT = PRIV / "audit/AUDIT.json"

SCREEN_SEEDS = (2026092800, 2026092801)
CONFIRM_SEEDS = (2026092802, 2026092803, 2026092804, 2026092805, 2026092806)
SEEDS = SCREEN_SEEDS + CONFIRM_SEEDS
FROZEN_SUBSET_SEEDS = (2026092800, 2026092801, 2026092802, 2026092803, 2026092804)
ARMS = ("C", "J", "JD")
WELSH_RESIDUE, IRISH_RESIDUE = 0, 2                # pair p of epoch e is Welsh if (p - e) % 4 == 0, Irish if == 2
DRAW_OFFSET = 20                                    # HR1 / HR3 regional draw streams: training seed + 20
KD_LAMBDA = 1.0                                     # J+D: + 1.0 * BCE(student logit, sigmoid(ADAF AO logit))
K = 30
STEP0_SEED = 2026092800
STEP0_REF = 1.0099719719146378
STEP0_TOL = 1e-6
CU_CEILING = 200.0
VM_RESERVE_GB = 20.0
BOOT_DRAWS, BOOT_SEED = 2000, 20261003
SELECT_TIE = 0.010


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path | str, obj: Any, indent: int | None = 1) -> None:
    """Atomic LF write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=indent)
        f.write("\n")
    tmp.replace(path)


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def torch_flags() -> dict:
    """TF32 off at the PyTorch level (NVIDIA_TF32_OVERRIDE=0 in the VM job), as HR1/HR3 torch_flags."""
    import os
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = os.environ.get("CUDNN_ALLOW_TF32", "0") == "1"
    return {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE")}
