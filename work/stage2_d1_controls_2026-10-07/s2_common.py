"""STAGE2-D1-CONTROLS (PACKET.md, draft 2026-10-07): shared paths, design constants, pairing table and helpers.

No side effects at import. Reuses, read-only by import (never edited):
  D1-COLLAPSE-1 (freeze f323e17e): d1_common (STARTS, DEV data, detector constants), d1_train (F1 loss, schedule,
    detector, update) and, through it, ADAPT-MATCH-0 (match_train: start loading, fresh AdamW, frozen BatchNorm;
    match_schedule: N_UPDATES, epochs, roles) and ADAPT-NDS-1 (freeze ba07548b: nds_schedule, nds_train: arm C's
    English/Welsh/Irish step map, region rows, loader).
  JOINT-1 (joint_common: W/I FIT tables and plans, the 30-monument subsets and their rule).
Private files (plans, subsets, plan checks, proposals) live under outputs/stage2_d1_controls_2026-10-07/; code and
coordinate-free receipts under work/stage2_d1_controls_2026-10-07/.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # work/stage2_d1_controls_2026-10-07
ROOT = HERE.parents[1]
D1_DIR = ROOT / "work/d1_collapse_2026-10-06"                 # D1-COLLAPSE-1 (frozen): import only
if str(D1_DIR) not in sys.path:
    sys.path.insert(0, str(D1_DIR))
import d1_common as D1  # noqa: E402  (also puts ADAPT-MATCH-0 and ADAPT-NDS-1 on sys.path)

N = D1.N                                                       # ADAPT-NDS-1 nds_common: hashing, JSON, digest, utc
MC = D1.MC                                                     # ADAPT-MATCH-0 match_common
NDS = D1.NDS
J1_CHAIN = N.J1_CHAIN
if str(J1_CHAIN) not in sys.path:
    sys.path.insert(0, str(J1_CHAIN))
import joint_common as J  # noqa: E402  JOINT-1 (frozen): W_FIT, W_PLAN, I_FIT, I_PLAN, SUBSETS, K

PRIV = ROOT / "outputs/stage2_d1_controls_2026-10-07"          # private: never committed
PLANS_DIR = PRIV / "plans"
WEIGHTS_DIR = PRIV / "weights"
CHAIN_RUNS = PRIV / "chain_runs"
PACKET = HERE / "PACKET.md"
CHAIN = HERE / "chain"
FREEZE_DIR = HERE / "freeze"
FREEZE_FILE = FREEZE_DIR / "FREEZE.json"
CU_PROJECTION = HERE / "CU_PROJECTION.json"
CU_SPEND = CHAIN / "CU_SPEND.jsonl"
RESULT = HERE / "RESULT.json"
VM_ROOT = "/content/DB/outputs/stage2_d1_controls_2026-10-07/vm"

# ---------------------------------------------------------------- the paired D1 runs (STAGE1-D1-V0 PASS)
D1_S3_TOP = "outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z"
D1_S3_VERIFIED_SHA256 = "b93ac07c1d79ae68d3b6db3d6d150b7d6f77ad18b83c4f03846e9a898d25bdc4"
D1_FREEZE_SHA256 = "f323e17e58e996c3f9ea62c5d3a684ea4e05d25be6dc6f76063b4ee05ed86ae6"
D1_TRAIN_SHA256 = "2e232c7759f718f77c5a97443e7b8c6784904f8785d5ae27dc4c11159c3c804e"
D1_RECIPE = "F1"
STAGE1_RESULT = "work/stage1_d1_v0_2026-10-07/RESULT.json"
STAGE1_RESULT_SHA256 = "3818ac2d9db21c1c14159b8f66b536c8d38a9b254b2387c58fc945174c3a9aa9"
SEEDS = D1.CONFIRM_SEEDS                                       # 2026092805 ... 2026092814, one control per D1 run
D1_FINAL = {                                                   # run_s3 SESSION_VERIFIED.json "trained"
    2026092805: "d9ece1fb023d2daf72ab0e5f789b5b89a27af73a28307e6260e4ee2fb5ac501e",
    2026092806: "f291694980fe24843e027b8ad846da08c4f407a0d75c1f3600ee0c5ef4f0bade",
    2026092807: "d705204634fc35751371c37e7fea85600acdd12e2eae15ed22aaf90cf7d73511",
    2026092808: "61ae1cb053a35a6bc427eefeaf7614d299dd8772e7c0d5eb99a47f35ae0762b5",
    2026092809: "38d14080dc98962ea79f31c0fbbfedc96f9bfd121cc1ef82003fe0d3b610563d",
    2026092810: "6b294645c62a074f864b6fe2ab8df4e9796e69b25a9abd6993093b53b78f21a5",
    2026092811: "ed12f627ec5975fb7cfde6565f9af6eab7b30b7f89423f49da97c1001bf69229",
    2026092812: "5437f281701203d9295e1f3b1da9a80261e2681cdecfe94b51ee4c7e54a815eb",
    2026092813: "3a1c2643ef26a336ba8915780328f1af27bf32a52b24905cb3ccb3f19c18a044",
    2026092814: "24db9082f55fe2f8cbd45e265b6164d73ec01823858c3ebd4630fe6b0b5f0858"}

# ---------------------------------------------------------------- the control (PACKET.md "Design")
ARM = "BARROW_L"                                               # start = ADAF barrow tar (D1.STARTS["BARROW_L"])
START = D1.STARTS[ARM]                                         # sha256 51b679fa..., 861,530,273 B
REFERENCE_PRIMARY = D1.REFERENCE[ARM]                          # 0.5556: ADAF barrow zero-shot FRESH-1 primary
NDS_ARM = "C"                                                  # ADAPT-NDS-1's equal-update control arm: its step map
NDS_C_SEEDS = (2026092802, 2026092803, 2026092804)             # generator identity check against C's recorded digests
NDS_C_RUNS = {2026092802: "outputs/adapt_nds_2026-10-05/chain_runs/run_s1_20261005T043612Z",
              2026092803: "outputs/adapt_nds_2026-10-05/chain_runs/run_s2_20261005T092024Z",
              2026092804: "outputs/adapt_nds_2026-10-05/chain_runs/run_s3_20261005T124915Z"}
EPOCHS = (4, 5)                                                # C's epochs (nds_schedule.EPOCHS)
N_UPDATES = 5116
SUBSET_TAG = 30                                                # JOINT-1 audit_mixture.subset: default_rng([seed, 30])
SUBSET_RECORDED_SEEDS = tuple(range(2026092800, 2026092807))   # SUBSETS_PRIVATE.json holds 00-06
W_TABLE_SHA256 = "65c494fd53386957ed1d84c3231221dd5afccb143c3b17383e2072c7854acd15"   # = C's TRAIN_RESULT pins
I_TABLE_SHA256 = "9ef37686b732ebf280408f36f36ed088598628a40a6fc25348d1e219319f44fa"
SUBSETS_SHA256 = "bfdc2cb706acc52838ba8149c8877d33abc558839f0a1bda5c0e4656861c9487"
SMOKE_SEED, SMOKE_UPDATES = 2026092805, 64

# sessions: 0 = GPU smoke on the production artifact route (one control, SMOKE_UPDATES updates, forced transfer fault);
# 1 = the ten controls (train in seed order, each WEIGHTS_READY) + one shared Verden DEV pass of the ten.
SESSIONS = {0: {"kind": "smoke", "seeds": (SMOKE_SEED,), "infer": False},
            1: {"kind": "run", "seeds": SEEDS, "infer": True}}
PREVIOUS = {1: 0}                                              # session 1 needs session 0's SMOKE_VERIFIED.json (PASS)
CU_CEILING = 60.0                                              # hard: cumulative over sessions 0 + 1
LOCAL_RESERVE_GB, PAGEFILE_HEADROOM_GB, WEIGHT_DEFER_GB = 20.0, 2.0, 22.0
WEIGHT_BYTES = 319_970_247                                     # D1 final.pt size (same state-dict layout)

# ---------------------------------------------------------------- the gate (PACKET.md "Endpoint and gate")
GATE = {"region": "verden", "benchmark": "v0", "decoder": "RAW", "b": "0.5", "mean_diff_min": 0.030,
        "positive_min": 8, "n_pairs": 10}
B = (0.5, 1.0, 2.0)

ATTRIBUTION = {
    **D1.ATTRIBUTION,
    "replay_data": "English R2 windows and NHLE masks, Welsh (HR1) and Irish (HR3) FIT windows and masks exactly as "
                   "JOINT-1 / ADAPT-NDS-1 used them; attribution and licence statements as recorded by those packets "
                   "(Irish: 'Contains Irish Public Sector Data (Geological Survey) licensed under CC BY 4.0'; NMS SMR "
                   "CC BY 4.0; HR3 PACKET.md); modified",
    "control_weights": "S2CTL runs are adaptations of the ADAF barrow weights (CC BY-SA 4.0); any shared copy stays "
                       "CC BY-SA 4.0 with this attribution"}


def run_name(seed: int) -> str:
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} outside the ten D1 confirmation seeds")
    return f"S2CTL_BARROW_{seed}"


def d1_name(seed: int) -> str:
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} outside the ten D1 confirmation seeds")
    return f"D1F1_BARROW_L_{seed}"


def english_plan_path(seed: int) -> Path:
    return PLANS_DIR / f"ENGLISH_E45_{seed}_PRIVATE.json"


SUBSETS_S2 = PLANS_DIR / "SUBSETS_S2_PRIVATE.json"            # recorded 00-06 + rule-derived 07-14 (subset_rule)


def session_runs(index: int) -> list[dict]:
    if index not in SESSIONS:
        raise ValueError(f"no stage-2 session {index}")
    s = SESSIONS[index]
    return [{"name": run_name(seed), "seed": seed, "infer": s["infer"], "kind": s["kind"]} for seed in s["seeds"]]


def free_gb(path: Path | str = ROOT) -> float:
    return D1.free_gb(path)
