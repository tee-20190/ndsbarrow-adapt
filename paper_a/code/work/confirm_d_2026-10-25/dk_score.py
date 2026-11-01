"""CONFIRM-D scoring glue: cr1_score.score UNCHANGED on Danish inputs (FREEZE §4), and the §5 reach decision.

Inputs: PREP_<TAG>.json (cores with B), DK_RECORDS_PRIVATE.json (records; id = systemnr), the S-v1 OSM snapshot per core
(dk_osm.py), and dt1_decode.py's assembled PROPOSALS_<model>_PRIVATE.json. Reach is cr1_score.REACH_M; §5 evaluates the
module constant at 10, 15 and 25 m (it is read at call time; the file is not edited).

  python -B dk_score.py reach      # DEV: D1_05 + PROX at 15 / 25 m -> REACH_DECISION.json (counts only)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIV = ROOT / "outputs/confirm_d_2026-10-25_PRIVATE"
sys.path.insert(0, str(ROOT / "work/context_rescue1_2026-10-23"))
import cr1_score as S  # noqa: E402

CORE_M, RATIO_SWITCH = 5000, 0.85


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inputs(tag: str) -> tuple[list[dict], list[dict], dict]:
    prep = json.loads((PRIV / f"{tag.lower()}/PREP_{tag}.json").read_text(encoding="utf-8"))["clusters"]
    cores = [{"i": c["core"][0] // CORE_M, "j": c["core"][1] // CORE_M, "B": c["B"], "land_km2": c["land_km2"]}
             for c in prep]
    keys = {(c["i"], c["j"]) for c in cores}
    recs = [{"id": r["systemnr"], "x": r["x"], "y": r["y"], "core": (int(r["x"] // CORE_M), int(r["y"] // CORE_M)),
             "area_m2": r["area_m2"], "group200": r["group200"]}
            for r in json.loads((PRIV / "DK_RECORDS_PRIVATE.json").read_text(encoding="utf-8"))["records"]]
    recs = [r for r in recs if r["core"] in keys]
    masks = {}
    for k in keys:
        snap = json.loads((PRIV / f"osm/core_{k[0]}_{k[1]}.json").read_text(encoding="utf-8"))
        masks[k] = S.mask_rule(S.check_reply(snap))
    return cores, recs, masks


def props(tag: str, model: str) -> list[dict]:
    p = PRIV / f"{tag.lower()}/decode/phase0/PROPOSALS_{model}_PRIVATE.json"
    return json.loads(p.read_text(encoding="utf-8"))["objects"]


def recall_at(objs: list[dict], cores: list[dict], recs: list[dict], masks: dict, variant: str, reach: float) -> int:
    old = S.REACH_M
    S.REACH_M = reach
    try:
        return len(S.score(objs, cores, recs, masks, variant, S.scot())["matched_ids"])
    finally:
        S.REACH_M = old


def reach() -> dict:
    cores, recs, masks = inputs("DEV")
    out = {"packet": "CONFIRM-D §5 reach decision (DK-FUNEN-DEV)", "cores": len(cores), "records": len(recs),
           "B_total": sum(c["B"] for c in cores), "recall": {}}
    for model, variant in (("D1_05", "prox"), ("D1_05", "plain"), ("ADAF_barrow", "plain")):
        objs = props("DEV", model)
        out["recall"][f"{model}|{variant}"] = {f"R{int(r)}": recall_at(objs, cores, recs, masks, variant, r)
                                               for r in (10.0, 15.0, 25.0)}
    d = out["recall"]["D1_05|prox"]
    ratio = d["R15"] / d["R25"] if d["R25"] else None
    out["R15_over_R25"] = ratio
    out["primary_reach_m"] = 25.0 if ratio is not None and ratio < RATIO_SWITCH else 15.0
    out["rule"] = "if R15/R25 < 0.85 for D1_05 + PROX on DEV, the primary reach becomes 25 m; else 15 m (FREEZE §5)"
    out["code_sha256"] = sha(Path(__file__))
    out["cr1_score_sha256"] = sha(ROOT / "work/context_rescue1_2026-10-23/cr1_score.py")
    (HERE / "REACH_DECISION.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return out


if __name__ == "__main__":
    print(json.dumps(reach() if sys.argv[1:2] == ["reach"] else {}, indent=1))
