"""ADAPT-MATCH-0 (PACKET.md, draft 2026-10-05): shared paths, constants, refusals and small helpers.

No side effects at import. Private files (window ids, proposals, plan checks) live under outputs/adapt_match_2026-10-05/;
code and coordinate-free records under work/adapt_match_2026-10-05/.
Reused read-only (imported with python -B, never edited): ADAPT-NDS-1 nds_common (hashing, JSON, path refusals),
nds_schedule (A's epoch structure, draw streams and compressed learning-rate schedule), nds_train (the update and the
Lower Saxony rows), nds_reader (Lower Saxony staging), score_adapt (FRESH-1 scoring primitives), chain/package_results.sh;
FRESH-1 lgln_reader / fresh_gates (DEV staging and consume) and its frozen scorer (HR1 region_score.py,
hier_uncertainty.py; FUSION-1 fuse_eval.py for INTERLEAVE); HR1 / HR3 / JOINT-1 chain modules as ADAPT-NDS-1 uses them.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # work/adapt_match_2026-10-05
ROOT = HERE.parents[1]
NDS = ROOT / "work/adapt_nds_2026-10-05"                       # ADAPT-NDS-1 (frozen, ba07548b): import only
if str(NDS) not in sys.path:
    sys.path.insert(0, str(NDS))
import nds_common as N  # noqa: E402  frozen helpers: sha256_file, read_json, write_json, digest, utc, rel, check_path

PRIV = ROOT / "outputs/adapt_match_2026-10-05"                  # private: never committed
PLANS_DIR = PRIV / "plans"
SCORE_DIR = PRIV / "score"
WEIGHTS_DIR = PRIV / "weights"                                  # verified local copies of AO_L / BARROW_L final weights
CHAIN_RUNS = PRIV / "chain_runs"                                # one TOP per session attempt (driver)
PACKET = HERE / "PACKET.md"
CHAIN = HERE / "chain"
FREEZE_DIR = HERE / "freeze"
FREEZE_FILE = FREEZE_DIR / "FREEZE.json"
CU_PROJECTION = HERE / "CU_PROJECTION.json"
CU_SPEND = CHAIN / "CU_SPEND.jsonl"
CU_LEDGER = CHAIN / "CU_LEDGER.jsonl"
SMOKE_RECEIPT = HERE / "SMOKE_RECEIPT.json"
RESULT = HERE / "RESULT.json"
VM_ROOT = "/content/DB/outputs/adapt_match_2026-10-05/vm"

# ---------------------------------------------------------------- ADAPT-NDS-1 (reference arm A; FIT data, by hash)
NDS_PRIV = ROOT / "outputs/adapt_nds_2026-10-05"
NDS_FREEZE = NDS / "freeze/FREEZE.json"
NDS_PINS = NDS / "freeze/PINS_FREEZE.json"
NDS_FREEZE_SHA256 = "ba07548bc3c67a0ee05bd20c19cdd6ed826c914bd55523940372320dfef2675b"
NDS_RESULT = NDS / "RESULT.json"                               # ADAPT-NDS-1 scored (packet "Starts after")
NDS_RUNS = NDS_PRIV / "chain_runs"                             # run_s<i>_<stamp>/SESSION_VERIFIED.json
LS_REL = "outputs/adapt_nds_2026-10-05/fit"
LS_DATA = {"ls_fit": f"{LS_REL}/FIT_TABLE_PRIVATE.json",              # 852 positive + 852 background windows, labels
           "ls_plan": f"{LS_REL}/FIT_STAGING_PINNED_PRIVATE.json",     # DGM1 tile pins (HEAD md5 = ETag)
           "ls_masks": f"{LS_REL}/FIT_INSIDE_MASKS_PRIVATE.npz"}       # FIT district union masks

# ---------------------------------------------------------------- FRESH-1 (DEV = Verden; zero-shot references)
FRESH_STEP2 = ROOT / "work/fresh_region_2026-10-03/step2"
FRESH_OUT, FRESH_S0 = "outputs/fresh_region_2026-10-03/step2", "outputs/fresh_region_2026-10-03/step0"
DEV_DATA = {"dev_infer_manifest": f"{FRESH_OUT}/INFER_MANIFEST_PRIVATE.json",        # label-free; travels to the VM
            "dev_enc_manifest": f"{FRESH_OUT}/ENCLOSURE_MANIFEST_PRIVATE.json",
            "dev_plan": f"{FRESH_OUT}/STAGING_PLAN_PRIVATE.json",
            "dev_masks": f"{FRESH_OUT}/INSIDE_MASKS_PRIVATE.npz",
            "dev_tile_index": f"{FRESH_S0}/TILE_INDEX_PRIVATE.json",
            "dev_tile_ledger": f"{FRESH_S0}/TILE_LEDGER_PRIVATE.jsonl"}
DEV_LABELS = {"dev_group_manifest": f"{FRESH_S0}/GROUP_MANIFEST_PRIVATE.json",        # local only (scorer)
              "dev_individual_manifest": f"{FRESH_S0}/INDIVIDUAL_MANIFEST_PRIVATE.json"}
FRESH1_S1 = "outputs/fresh_region_2026-10-03/step2/chain_runs/run_s1_20261004T174340Z"
FRESH1_RESULT = "work/fresh_region_2026-10-03/step2/STEP2_RESULT.json"
S_STAR = "J_2026092803"                                        # FRESH-1 STEP2_RESULT.json sstar.name
ZERO_SHOT = ("ADAF_AO", "ADAF_barrow", S_STAR)
FRESH1_PROPOSALS = {                                           # FRESH-1 SESSION_VERIFIED main/ hashes (= ADAPT-NDS-1
    "ADAF_AO": "458010e6992fa5c502297d1351f23ba03f9b6e93de4799806a196f4076f05e48",   # freeze dev_references)
    "ADAF_barrow": "c5d790c8f7fe26872ee74ec88cd54aa6046428f8a0338279100b4135e564d2f9",
    S_STAR: "f8f52d5eb60ff40a9446ccee7ca4cba9528d227abeab091cb2f2cabb75fc0047"}

# ---------------------------------------------------------------- design (packet "Arms")
SEEDS = (2026092802, 2026092803, 2026092804)
ARMS = ("AO_L", "BARROW_L")
SESSIONS = {1: "AO_L", 2: "BARROW_L"}                          # sequential: AO_L verified before BARROW_L
ZERO_SHOT_OF = {"AO_L": "ADAF_AO", "BARROW_L": "ADAF_barrow"}  # each adapted arm's zero-shot start
STARTS = {                                                     # DETECTOR_REGISTRY.json ADAF-AO / ADAF-barrow; FRESH-1 freeze
    "AO_L": {"registry_id": "ADAF-AO", "md5": "fba3667a886b7b55d6e745f836951919", "bytes": 861530273,
             "path": "work/adaf_head_to_head_2026-09-19/weights/"
                     "AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
             "sha256": "a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb"},
    "BARROW_L": {"registry_id": "ADAF-barrow", "md5": "8c2cf34634c1bc45874a324a92d2096b", "bytes": 861530273,
                 "path": "work/campaign_2026-09-22/adaf_weights/"
                         "barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
                 "sha256": "51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216"}}
N_UPDATES = 5116                                               # A's total (two J epochs of 2,558)
WEIGHT_BYTES = 319_970_119                                     # an HRNet state-dict final.pt (J family, registry size)

# ---------------------------------------------------------------- measures (packet "Measures")
BOOT_SEED, BOOT_DRAWS = 20261005, 2000
AHEAD_MEAN = 0.03                                              # A_AHEAD: mean >= +0.03 and lo95 > 0; A_BEHIND: hi95 < 0
EPS = 1e-9                                                     # FRESH-1 / ADAPT-NDS-1 comparison tolerance
FORECASTS = {"BARROW_L_MEAN_GT_ADAF_BARROW_ZERO_SHOT": 0.60, "A_BEHIND": 0.35, "A_AHEAD": 0.20}
A_NAME = "A_NDS_{seed}"                                        # ADAPT-NDS-1 nds_common.run_name("A", seed)

# ---------------------------------------------------------------- resources (packet "Resources and procedure")
CU_CEILING = 60.0                                              # "about 40 CU in total; at most 60"
CU_EXPECTED = 40.0
LOCAL_WEIGHT_GB = 2.0                                          # six final weights = 1.92 GB
LOCAL_PEAK_GB = 2.5                                            # declared launch peak (storage_guard, brief)
START_FREE_GB = 27.0                                           # C: free >= 20 GB reserve + 2 GB + 5 GB page-file headroom
WEIGHT_DEFER_GB = 22.0                                         # a weight download waits while C: free - bytes < 20 + 2 GB
VM_RESERVE_GB = 20.0

ATTRIBUTION = {
    "terrain": N.ATTRIBUTION["terrain"],
    "register": N.ATTRIBUTION["register"],
    "adaf_weights": "ADAF AO and barrow HRNets: Čož, Corns, Curran, Kocev, Kokalj 2026, J. Archaeol. Sci. Rep. "
                    "71:105733; weights CC BY-SA 4.0 (Zenodo 10.5281/zenodo.15848662), code Apache-2.0",
    "adapted_weights": "AO_L and BARROW_L are adaptations of the ADAF weights above (modified by continued training "
                       "on Lower Saxony register labels); any shared copy stays CC BY-SA 4.0 with this attribution"}


class Refused(RuntimeError):
    """An input lies outside ADAPT-MATCH-0's eligible data or binding."""


def run_name(arm: str, seed: int) -> str:
    if arm not in ARMS or seed not in SEEDS:
        raise SystemExit(f"arm {arm!r} / seed {seed} outside ADAPT-MATCH-0")
    return f"{arm}_{seed}"


def a_name(seed: int) -> str:
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} outside ADAPT-MATCH-0")
    return A_NAME.format(seed=seed)


def session_arm(index: int) -> str:
    if index not in SESSIONS:
        raise SystemExit(f"session index {index} outside 1-2")
    return SESSIONS[index]


def start_path(arm: str) -> Path:
    return ROOT / STARTS[arm]["path"]


def free_gb(path: Path | str = ROOT) -> float:
    return N.free_gb(path)
