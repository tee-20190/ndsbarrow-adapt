"""HOLDOUT-REGION-3 (Ireland) Step 2 chain: shared paths, seeds, geometry and small helpers.

No side effects at import. Values marked "freeze" are bound by the Step 2 freeze record (FREEZE.json).
Design: STEP2_SPEC.md (HR2 arms/recipe/seeds), AMENDMENT_1 (2 km squares, 60 m IGNORE discs), AMENDMENT_2 (clipped
1 km background cells). Reused unchanged: HR1 region_score.py / hier_uncertainty.py (scorer), the AMZ-ADAPT recipe
(adapt_train.py), the frozen English inference path (tta_run / holdout_eval) and the Step 1 adapter (vis_irl.py).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent            # work/hr3_ireland_2026-10-02/step2/chain
STEP2 = HERE.parent
WORK = STEP2.parent
ROOT = WORK.parents[1]
VIS = WORK / "visibility"
HR1 = ROOT / "work/holdout_region_2026-10"
HR1_CHAIN = HR1 / "chain"
HR1_STEP2 = HR1 / "step2"
PRIV = ROOT / "outputs/hr3_ireland_2026-10-02"
STEP0 = PRIV / "step0_v2"
OUT = PRIV / "step2"                              # private build outputs (coordinates): never committed
NODATA_PRIVATE = PRIV / "visibility/NODATA_PARTS_PRIVATE.json"
DEV_SAMPLE = PRIV / "DEV_SAMPLE_PRIVATE.json"
FEAS_PRIVATE = PRIV / "step2_prep/FEASIBILITY_PRIVATE.json"
VALIDITY = PRIV / "step2_prep/VALIDITY_MASKS_PRIVATE.npz"
TILE_LEDGER = PRIV / "step2_prep/TILE_LEDGER.jsonl"

DEV_MANIFEST = OUT / "DEV_MANIFEST_PRIVATE.json"
FIT_TABLES = OUT / "FIT_TABLES_PRIVATE.json"
STAGING_PLAN = OUT / "STAGING_PLAN_PRIVATE.json"
SMR_SUPPLEMENT = OUT / "SMR_SUPPLEMENT_PRIVATE.json"
REPORT = STEP2 / "MANIFEST_REPORT.json"           # coordinate-free aggregate (committed)

# ---------------------------------------------------------------- blocks (Step 0; AMENDMENT_1)
DEV_BLOCK, FIT_BLOCK, TEST_BLOCK = 3, 4, 1        # TEST block 1 (Galway/Mayo) is sealed: never fetched
ITM_EPSG = 2157

# ---------------------------------------------------------------- seeds (freeze)
TRAIN_SEEDS = (2026092800, 2026092801, 2026092802, 2026092803, 2026092804)   # HR1 plans; pairing with reused C
RERUN_SEED = TRAIN_SEEDS[0]          # HR1's C rerun weights (inference only)
JITTER_SEED = 20261006               # DEV positive core jitter: default_rng([JITTER_SEED, 0]), part-id order
DRAW_OFFSET = 20                     # Irish draw stream = training seed + 20 (HR1 Welsh offset)
SUBSET_TAG = 30                      # A_I30 subset for training seed s: default_rng([s, 30])
K = 30
BOOT_SEED, BOOT_DRAWS = 20261001, 2000

ARMS = ("A_I0", "A_I30")             # trained here; C is HR1's frozen C (inference only)
STEP0_SEED = 2026092800
STEP0_REF = 1.0099719719146378       # HR1 common.STEP0_REF: C's first update, seed 2026092800
STEP0_TOL = 1e-6

# ---------------------------------------------------------------- geometry (metres)
CORE_M = 1000
HALO_M = 512
PAD_M = 16
JITTER_MAX_M = 249
FIT_WIN_M = 423                      # 846 px at 0.5 m (English raw-window shape)
CELL_M = 1000                        # AMENDMENT_2 background cells (ITM grid)
SQ2_M = 2000                         # AMENDMENT_1 super-blocks: 2 km ITM squares (SB2_<x//2000>_<y//2000>)
BOX_M = 10                           # AMENDMENT_2 10 m accounting grid
ERODE_M = 20                         # AMENDMENT_2 NODATA margin (square erosion: stricter than Euclidean)
IGNORE_R = 60.0                      # AMENDMENT_1 item 2 IGNORE disc radius around every SMR record
BAND_M = 5.0                         # IGNORE band outside each positive disc (HR1 / r2_masks)
REACH_M = 15.0                       # scorer reach (region_score / score_holdout)
SMR_MARGIN_M = 1000.0                # supplement: SMR records within this margin outside each block bbox
FIT_VALID_MIN = 0.99                 # AMENDMENT_2 FIT background windows

# ---------------------------------------------------------------- floors and ceilings
DEV_PARTS_FLOOR, FIT_FLOOR, BG_FLOOR_KM2 = 120, 30, 100.0
CU_CEILING = 200.0
VM_RESERVE_GB = 20.0


def torch_flags() -> dict:
    """TF32 off at the PyTorch level (NVIDIA_TF32_OVERRIDE=0 in vm_job_irl.sh), as HR1 common.torch_flags."""
    import os
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = os.environ.get("CUDNN_ALLOW_TF32", "0") == "1"
    return {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any, indent: int | None = 1) -> Path:
    """LF-only UTF-8 with a trailing newline, atomic replace (Windows text mode would write CRLF)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((json.dumps(obj, indent=indent) + "\n").encode("utf-8"))
    tmp.replace(path)
    return path


def create_only(path: Path, obj: Any, indent: int | None = 1) -> Path:
    if Path(path).exists():
        raise SystemExit(f"{path} exists; create-only")
    return write_json(path, obj, indent)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def round_half_up(v: float) -> int:
    return math.floor(v + 0.5)


def sb2(x: float, y: float) -> str:
    """AMENDMENT_1 2 km ITM square label, as DEV_SAMPLE_PRIVATE.json's sb2 (e.g. SB2_288_414)."""
    return f"SB2_{math.floor(x / SQ2_M)}_{math.floor(y / SQ2_M)}"


def boxes_overlap(a: Iterable[float], b: Iterable[float]) -> bool:
    """Half-open boxes share interior (touching edges do not)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)


def in_box(x: float, y: float, b: Iterable[float]) -> bool:
    x0, y0, x1, y1 = b
    return x0 <= x < x1 and y0 <= y < y1


def expand(b: Iterable[float], d: float) -> list:
    x0, y0, x1, y1 = b
    return [x0 - d, y0 - d, x1 + d, y1 + d]


def fit_window_bounds(cx: float, cy: float) -> list[int]:
    """HR1 build_manifest.fit_window_bounds: integer 423 m window centred within 0.5 m of the point."""
    x0, y0 = round_half_up(cx - FIT_WIN_M / 2), round_half_up(cy - FIT_WIN_M / 2)
    return [x0, y0, x0 + FIT_WIN_M, y0 + FIT_WIN_M]


def model_names(seed: int) -> list[str]:
    return [f"C_{seed}", f"A_I0_{seed}", f"A_I30_{seed}"]
