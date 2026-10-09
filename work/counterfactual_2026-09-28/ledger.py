"""Experiment 1 v2 GPU ledger and freeze verification (PROTOCOL.md sections 11-12).

Ceilings: pilot 6.0 GPU-h (premise check, smoke, pilot arms, scoring, diagnostic); full test 50 GPU-h.
FREEZE.json holds one section per stage; every GPU stage re-verifies the hashes of its own section.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs/counterfactual_2026-09-28"
LEDGER = OUT / "GPU_TIME.json"
FREEZE = HERE / "FREEZE.json"
CEILINGS_S = {"pilot": 6.0 * 3600, "full": 50.0 * 3600}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"runs": []}


def charged_s(stage: str) -> float:
    return sum(r["wall_s"] for r in read()["runs"] if r["stage"] == stage)


def remaining_s(stage: str) -> float:
    return CEILINGS_S[stage] - charged_s(stage)


def charge(stage: str, kind: str, wall_s: float, note: dict | None = None) -> None:
    if stage not in CEILINGS_S:
        raise ValueError(f"unknown stage {stage}")
    led = read()
    led["runs"].append({"stage": stage, "kind": kind, "wall_s": wall_s,
                        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(note or {})})
    led["charged_s"] = {s: sum(r["wall_s"] for r in led["runs"] if r["stage"] == s) for s in CEILINGS_S}
    led["ceilings_s"] = CEILINGS_S
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, indent=1), encoding="utf-8")
    tmp.replace(LEDGER)


def verify_frozen(stage: str) -> dict:
    """Every file in FREEZE.json's section for `stage` still has its frozen hash."""
    if not FREEZE.exists():
        raise RuntimeError("FREEZE.json is missing; no GPU stage may run")
    sec = json.loads(FREEZE.read_text(encoding="utf-8")).get(f"stage_{stage}")
    if sec is None:
        raise RuntimeError(f"FREEZE.json has no stage_{stage} section")
    bad = [p for p, h in sec["sha256"].items() if sha(ROOT / p) != h]
    if bad:
        raise RuntimeError(f"frozen inputs changed since the {stage} freeze: {bad[:5]}")
    return sec
