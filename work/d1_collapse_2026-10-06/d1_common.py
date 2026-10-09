"""D1-COLLAPSE-1 (PACKET.md): shared paths, design constants and helpers. No side effects at import.

Reuses ADAPT-MATCH-0 (frozen 351b5961) read-only by import: match_common (STARTS, LS_DATA, DEV_DATA, DEV_LABELS,
FRESH-1 references, nds_common as N), match_schedule (the all-local plan), match_train (rows, plan, start loading).
Private files live under outputs/d1_collapse_2026-10-06/; code and coordinate-free receipts under
work/d1_collapse_2026-10-06/.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # work/d1_collapse_2026-10-06
ROOT = HERE.parents[1]
MATCH = ROOT / "work/adapt_match_2026-10-05"                   # ADAPT-MATCH-0 (frozen): import only
if str(MATCH) not in sys.path:
    sys.path.insert(0, str(MATCH))
import match_common as MC  # noqa: E402  (also puts ADAPT-NDS-1 on sys.path)

N = MC.N
NDS = MC.NDS
PRIV = ROOT / "outputs/d1_collapse_2026-10-06"                 # private: never committed
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
VM_ROOT = "/content/DB/outputs/d1_collapse_2026-10-06/vm"

# ---------------------------------------------------------------- data and references (ADAPT-MATCH-0, by hash)
STARTS = MC.STARTS
ARMS = MC.ARMS                                                  # ("AO_L", "BARROW_L")
LS_DATA, DEV_DATA, DEV_LABELS = MC.LS_DATA, MC.DEV_DATA, MC.DEV_LABELS
N_UPDATES = MC.N_UPDATES
WEIGHT_BYTES = MC.WEIGHT_BYTES

# ---------------------------------------------------------------- design (PACKET.md)
DEV_SEEDS = (2026092802, 2026092803, 2026092804)              # development: ADAPT-MATCH-0's seeds (diagnosis, selection)
CONFIRM_SEEDS = tuple(range(2026092805, 2026092815))          # confirmation: 10 fresh seeds per arm, never used before
SEEDS = DEV_SEEDS + CONFIRM_SEEDS
RECIPES = ("OLD", "F1", "F2")                                  # OLD = ADAPT-MATCH-0 (probe P0 only)
PROBE = ("OLD", "BARROW_L", 2026092802)
WARMUP_UPDATES, F2_LR_MAX = 256, 5e-5
DETECT_WINDOW, DETECT_DICE, DETECT_MASS = 256, 0.05, 1.0       # online detector (trailing updates; ON iff both)
REFERENCE = {"AO_L": 0.655, "BARROW_L": 0.5555555555555556}    # old healthy AO_L mean (0.672, 0.638); ADAF barrow 0-shot
NOISE = 0.037                                                  # HR1 same-seed C rerun difference
COLLAPSE_FRACTION = 0.5                                        # (b): primary < 0.5 x reference

# sessions: index -> (phase, recipe, [(arm, seed, infer)], probe). Development (DEV_SEEDS): 1 = F1 (+ probe P0);
# 2 = F2, only if F1 is not selected. Confirmation (CONFIRM_SEEDS) with the selected recipe only
# (RECIPE_SELECTION.json, select_recipe.py): 3/4 for F1, 13/14 for F2.
SESSIONS = {
    1: ("dev", "F1", [("BARROW_L", s, True) for s in DEV_SEEDS], True),
    2: ("dev", "F2", [("BARROW_L", s, True) for s in DEV_SEEDS], False),
    3: ("confirm", "F1", [("BARROW_L", s, True) for s in CONFIRM_SEEDS], False),
    4: ("confirm", "F1", [("AO_L", s, True) for s in CONFIRM_SEEDS], False),
    13: ("confirm", "F2", [("BARROW_L", s, True) for s in CONFIRM_SEEDS], False),
    14: ("confirm", "F2", [("AO_L", s, True) for s in CONFIRM_SEEDS], False),
}
PREVIOUS = {2: 1, 4: 3, 14: 13}                                # session -> the session that must be verified first
SELECTION = HERE / "RECIPE_SELECTION.json"                     # written after development by select_recipe.py

CU_CEILING = {"dev": 55.0, "confirm": 120.0}
LOCAL_RESERVE_GB, PAGEFILE_HEADROOM_GB, WEIGHT_DEFER_GB = 20.0, 2.0, 22.0

ATTRIBUTION = {
    **MC.ATTRIBUTION,
    "adapted_weights": "D1 AO_L / BARROW_L runs are adaptations of the ADAF weights above (modified by continued "
                       "training on Lower Saxony register labels); any shared copy stays CC BY-SA 4.0 with this "
                       "attribution"}


def run_name(recipe: str, arm: str, seed: int) -> str:
    if recipe not in RECIPES or arm not in ARMS or seed not in SEEDS:
        raise ValueError(f"run outside the D1 design: {recipe} {arm} {seed}")
    if recipe == "OLD" and (recipe, arm, seed) != PROBE:
        raise ValueError("the old recipe runs only as probe P0")
    return f"D1{recipe}_{arm}_{seed}"


def session_runs(index: int) -> list[dict]:
    """Every training of a session in order: the probe (session 1) first, then the candidate runs."""
    if index not in SESSIONS:
        raise ValueError(f"no D1 session {index}")
    _, recipe, runs, probe = SESSIONS[index]
    out = []
    if probe:
        r, a, s = PROBE
        out.append({"name": run_name(r, a, s), "recipe": r, "arm": a, "seed": s, "infer": False})
    out += [{"name": run_name(recipe, a, s), "recipe": recipe, "arm": a, "seed": s, "infer": inf} for a, s, inf in runs]
    return out


def collapse_threshold(arm: str) -> float:
    return COLLAPSE_FRACTION * REFERENCE[arm]


def free_gb(path: Path | str = ROOT) -> float:
    return MC.free_gb(path)
