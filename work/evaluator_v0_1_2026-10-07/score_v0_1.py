"""Interface adapter: evaluator v0.1 for the STAGE2 prespecified sensitivity hook (Verden).

Added 2026-10-07 after the v0.1 freeze (0f3443ea) and the STAGE2 freeze (bd144db0), before any STAGE2 control existed.
The STAGE2 hook (work/stage2_d1_controls_2026-10-07/score_s2.py, v01_sensitivity) imports `score_v0_1.py` and calls
`score_systems(objs)`; the frozen evaluator exposes `eval_v01.score_barrow(objs, setup)`. This file only connects the
two. It changes no frozen file and no scoring rule:

- the region is Verden (the STAGE2 scoring region), from the same manifest as run_checks.py;
- the manifest's sha256 must equal the frozen void table's (VOIDS.json), else it refuses;
- voids come from the frozen VOIDS.json through eval_v01.validate_voids (Verden: none);
- every system is scored by eval_v01.score_barrow unchanged.

Output: {system: score_barrow(...)} plus "_entities_N" (reachable targets under v0.1).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the STAGE2 hook loads this file by path
import eval_v01 as V  # noqa: E402  frozen evaluator v0.1 (0f3443ea)

ROOT = V.ROOT
REGION = "verden"
MANIFEST = ROOT / "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json"
VOIDS = HERE / "VOIDS.json"


def _setup() -> V.Setup:
    table = json.loads(VOIDS.read_text(encoding="utf-8"))["regions"][REGION]
    got = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if got != table["manifest_sha256"]:
        raise SystemExit(f"v0.1 adapter refused: Verden manifest sha256 {got} != frozen void table {table['manifest_sha256']}")
    return V.setup(V.load_json(MANIFEST), V.validate_voids(table))


def score_systems(objs: dict) -> dict:
    st = _setup()
    out = {name: V.score_barrow(o, st) for name, o in objs.items()}
    out["_entities_N"] = st.n
    return out
