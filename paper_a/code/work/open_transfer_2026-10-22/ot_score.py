"""OPEN-TRANSFER-1 scoring (FREEZE.md section 6). Zero-shot landscape recall of RCE grafheuvels by each arm's ranked
proposals over the EVAL provinces, with a 2 km RD block bootstrap stratified by region (province).

Reuses the SCOT-TRANSFER-1 scorer's functions unchanged (`work/scot_transfer_2026-10-19/scot_score.py`, sha pinned
below): ranking, 15 m one-to-one maximum matching, budgets, matched workload, stratified unit bootstrap, Bonferroni
family and verdict. Only the module constants below differ (bootstrap unit 2 km, seed 20261022).

Inputs (outputs/open_transfer_2026-10-22_PRIVATE/): regions/<R>/SITES.json, regions/<R>/PREP.json,
infer/<R>/PROPOSALS_<arm>_PRIVATE.json. Run once (WSL adaf-h2h-venv):
  python -I work/open_transfer_2026-10-22/ot_score.py <repo-root>   # scores once
  python -I work/open_transfer_2026-10-22/ot_score.py --selftest
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCOT = ROOT / "work/scot_transfer_2026-10-19/scot_score.py"
SCOT_SHA = "00f1fa3098a6d70bda5ab8a2ab340320589f4ebf668556dca8342183a6756dc7"
REGIONS = ("DR", "NH", "OV", "UT")
ARMS = ("D1", "CONT", "ADAF_barrow", "ADAF_AO")
PRIMARY = ("D1", "CONT")
REACH_SENS_M = (10.0, 25.0)


def scorer(raw: bool = False):
    if hashlib.sha256(SCOT.read_bytes()).hexdigest() != SCOT_SHA:
        raise SystemExit("scot_score.py differs from its pin")
    spec = importlib.util.spec_from_file_location("scot_score", SCOT)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    if raw:
        return S
    S.CELL_M = 2000.0          # cluster bootstrap unit: 2 km RD block holding >= 1 eligible record
    S.SEED = 20261022
    S.N_BOOT = 10_000
    S.REACH_M = 15.0
    S.BUDGETS = (1.0, 2.0, 5.0, 10.0)
    S.PRIMARY_B = 5.0
    S.COMPARATORS = ("ADAF_barrow", "ADAF_AO")
    return S


def load(root: Path) -> tuple[dict, dict]:
    priv = root / "outputs/open_transfer_2026-10-22_PRIVATE"
    regions, hashes = {}, {}
    for r in REGIONS:
        sp, pp = priv / f"regions/{r}/SITES.json", priv / f"regions/{r}/PREP.json"
        props = {}
        for a in ARMS:
            f = priv / f"infer/{r}/PROPOSALS_{a}_PRIVATE.json"
            props[a] = json.loads(f.read_text(encoding="utf-8"))["objects"]
            hashes[f"{r}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (sp, pp):
            hashes[f"{r}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
        regions[r] = {"sites": json.loads(sp.read_text(encoding="utf-8")),
                      "area": json.loads(pp.read_text(encoding="utf-8"))["eval_area_km2"], "props": props}
    return regions, hashes


def groups_found(regions: dict, matched: dict) -> dict:
    """Descriptive: share of 60 m record groups with >= 1 matched record at the primary budget."""
    out = {}
    for a, per in matched.items():
        g_all = {(r, s["group60"]) for r, g in regions.items() for s in g["sites"]}
        g_hit = {(r, regions[r]["sites"][j]["group60"]) for r, js in per.items() for j in js}
        out[a] = {"groups_hit": len(g_hit), "groups": len(g_all)}
    return out


def size_split(regions: dict, matched: dict) -> dict:
    """Descriptive: recall at the primary budget by polygon area (< 200 m2, 200-500 m2, >= 500 m2)."""
    bins = (("lt200", 0, 200), ("200_500", 200, 500), ("ge500", 500, 1e12))
    out = {}
    for a, per in matched.items():
        out[a] = {}
        for name, lo, hi in bins:
            tot = sum(lo <= s["area_m2"] < hi for g in regions.values() for s in g["sites"])
            hit = sum(lo <= regions[r]["sites"][j]["area_m2"] < hi for r, js in per.items() for j in js)
            out[a][name] = {"matched": hit, "of": tot}
    return out


def score(root: Path) -> dict:
    S = scorer()
    regions, hashes = load(root)
    keep = ("primary_family", "verdicts", "pooled_recall")
    main = S.analyse(regions, list(ARMS), list(PRIMARY), S.REACH_M)
    sens = {str(rch): {k: v for k, v in S.analyse(regions, list(ARMS), list(PRIMARY), rch).items() if k in keep}
            for rch in REACH_SENS_M}
    equal_w = S.region_equal_weight(main, regions, {})
    matched = main["_matched_primary"]
    out = {"packet": "OPEN-TRANSFER-1", "reach_m": S.REACH_M, "primary_budget_per_km2": S.PRIMARY_B,
           "budgets": S.BUDGETS, "n_boot": S.N_BOOT, "seed": S.SEED, "cell_m": S.CELL_M,
           "eval_area_km2": {r: g["area"] for r, g in regions.items()},
           "sites_by_region": {r: len(g["sites"]) for r, g in regions.items()},
           **{k: v for k, v in main.items() if not k.startswith("_")},
           "sensitivity_reach": sens, "region_equal_weight": equal_w,
           "groups_primary_budget": groups_found(regions, matched),
           "size_primary_budget": size_split(regions, matched),
           "matched_site_ids_primary_budget": {a: {r: [regions[r]["sites"][j]["id"] for j in js]
                                                   for r, js in per.items()} for a, per in matched.items()},
           "inputs_sha256": hashes, "scot_score_sha256": SCOT_SHA,
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dest = root / "outputs/open_transfer_2026-10-22_PRIVATE/RESULT_SCORE.json"
    if dest.exists():
        raise SystemExit("RESULT_SCORE.json exists: OPEN-TRANSFER-1 scores once")
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    return out


def selftest() -> None:
    scorer(raw=True).selftest()     # the pinned scorer's own selftest, at its own constants
    S = scorer()
    assert S.CELL_M == 2000.0 and S.SEED == 20261022
    regs = {"R": {"sites": [{"id": "a", "e": 100.0, "n": 100.0, "group60": 0, "area_m2": 150.0},
                            {"id": "b", "e": 130.0, "n": 100.0, "group60": 0, "area_m2": 600.0}]}}
    assert groups_found(regs, {"X": {"R": [1]}}) == {"X": {"groups_hit": 1, "groups": 1}}
    assert size_split(regs, {"X": {"R": [1]}})["X"]["ge500"] == {"matched": 1, "of": 1}
    print("ot_score selftest PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        r = score(Path(sys.argv[1]).resolve())
        print(json.dumps({k: r[k] for k in ("verdicts", "primary_family", "pooled_recall", "sites_by_region")},
                         indent=1))
