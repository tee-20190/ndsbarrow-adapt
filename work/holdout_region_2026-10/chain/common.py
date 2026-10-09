"""HOLDOUT-REGION-1 Stage A chain: shared paths, candidate seeds, declared geometry and small helpers.

No side effects at import. Every constant marked "candidate" is recorded for the freeze and is not final.
Geometry stays inside the Step 1 declared footprints (step1/SPLIT.json geometry_m and footprint_half_width_m):
DEV/TEST half-width 1,278 m = core 500 + jitter 250 + halo 512 + pad 16; FIT envelope 256 m.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REGION = HERE.parent
STEP1 = REGION / "step1"
STEP2 = REGION / "step2"
SOURCES = ROOT / "outputs/holdout_region_2026-10/sources"
CADW = SOURCES / "cadw_sam_p00.geojson"
COFLEIN = SOURCES / "coflein_terrestrial_p00.geojson"
CATALOGUE = SOURCES / "lidar_catalogue_2020_2023_p00.geojson"
CANDIDATES = HERE / "candidates"

# ---------------------------------------------------------------- candidate seeds (freeze fixes the final values)
TRAIN_SEEDS = (2026092800, 2026092801, 2026092802, 2026092803, 2026092804)  # pre-built English plans (plans.py)
RERUN_SEED = TRAIN_SEEDS[0]          # the one same-seed rerun of C (D-noise diagnostic)
MANIFEST_SEED = 20261004             # DEV positive jitter [seed, 0] and DEV background draws [seed, 1]
FIT_BG_SEED = 20261005               # FIT background windows [seed, 1]
WELSH_DRAW_OFFSET = 20               # Welsh draw stream = training seed + 20 (AMZ-ADAPT: plan seed + 10)
W30_SUBSET_TAG = 30                  # A_W30 subset for training seed s: default_rng([s, 30])

ARMS = ("C", "A_W0", "A_W30", "A_WALL")
WELSH_ARMS = ("A_W0", "A_W30", "A_WALL")
W30_K = 30

# ---------------------------------------------------------------- step-0 gate (packet "Steps" 2; STEP0_REFERENCE.json)
STEP0_REF = 1.0099719719146378       # outputs/amazon_adapt_2026-09-28/pilot_chain.log:7, C, seed 2026092800, step 0
STEP0_SEED = 2026092800
STEP0_TOL = 1e-6

# ---------------------------------------------------------------- declared geometry (metres)
CORE_M = 1000                        # 1 km core tile
HALO_M = 512                         # AMZ-ADAPT windowed halo: 1,024 px at 0.5 m
PAD_M = 16                           # Step 1 pad (only a footprint margin; windows are core + halo)
JITTER_MAX_M = 249                   # integer jitter of the core centre from the rounded part centroid: <= 249.5 m
FIT_WIN_M = 423                      # English raw training window (423 x 423 at 1 m -> 846 px at 0.5 m)
FIT_ENVELOPE_HALF_M = 256.0          # Step 1 FIT effective-footprint half-width
DEV_FOOTPRINT_HALF_M = 1278.0        # Step 1 DEV part-square half-width
BG_CLEAR_M = 100.0                   # whole-core exclusion distance (scheduled polygon / listed Coflein point)
DISC_M = 30.0                        # Coflein IGNORE disc radius (training masks)
SAME_SITE_M = 50.0                   # Coflein point within 50 m of a Cadw polygon = same site, never a disc
BAND_M = 5.0                         # r2_masks.BAND_M: IGNORE band outside each positive
SUPERBLOCK_M, SQUARE_M, CELL_M = 20000, 10000, 1000
N_DEV_BG = 300                       # 300 km2 of background core
N_FIT_BG = 741                       # candidate: the English fitting-negative pool size (plans.load_windows)
MAX_DRAWS = 2_000_000

# ---------------------------------------------------------------- ceilings (packet "Whole-chain ceilings")
CU_CEILING = 200.0
G4_HOURS_CEILING = 20.0
VM_DISK_CEILING_GB = 70.0
VM_RESERVE_GB = 20.0
LOCAL_RETAINED_CEILING_GB = 1.0


def torch_flags() -> dict:
    """GPU entry points (step 0, training, smoke, retention, DEV inference): TF32 off at the PyTorch level as well as
    by NVIDIA_TF32_OVERRIDE=0 (vm_job.sh), matching parity.py. CUDNN_ALLOW_TF32=1 restores the PyTorch default (True)
    that the frozen recipe ran with; the smoke uses it only for its diagnostic step-0 comparison."""
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def digest(obj: Any) -> str:
    return sha256_bytes(canonical(obj))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any, indent: int | None = 1) -> Path:
    """LF-only, UTF-8, trailing newline; atomic replace (Windows write_text would emit CRLF in text mode)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((json.dumps(obj, indent=indent) + "\n").encode("utf-8"))
    tmp.replace(path)
    return path


def superblock_id(x: float, y: float) -> str:
    """SPLIT.json super-block id (lower-left corner in km on the 20 km grid), e.g. SB_180_200."""
    return f"SB_{20 * math.floor(x / SUPERBLOCK_M)}_{20 * math.floor(y / SUPERBLOCK_M)}"


def square_index(x: float, y: float) -> tuple[int, int]:
    return math.floor(x / SQUARE_M), math.floor(y / SQUARE_M)


def boxes_overlap(a: Iterable[float], b: Iterable[float]) -> bool:
    """Half-open boxes share a point (touching edges do not)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)


def in_box(x: float, y: float, b: Iterable[float]) -> bool:
    x0, y0, x1, y1 = b
    return x0 <= x < x1 and y0 <= y < y1


def expand(b: Iterable[float], d: float) -> list[float]:
    x0, y0, x1, y1 = b
    return [x0 - d, y0 - d, x1 + d, y1 + d]


def cells_under(b: Iterable[float]) -> list[tuple[int, int]]:
    """1 km catalogue cells whose interior meets the interior of box b (a window edge on a cell edge adds nothing)."""
    x0, y0, x1, y1 = b
    kx0, ky0 = math.floor(x0 / CELL_M), math.floor(y0 / CELL_M)
    kx1, ky1 = math.ceil(x1 / CELL_M), math.ceil(y1 / CELL_M)
    return [(kx, ky) for kx in range(kx0, kx1) for ky in range(ky0, ky1)]
