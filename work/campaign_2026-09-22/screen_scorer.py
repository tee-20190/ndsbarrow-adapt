"""Amendment 2 C.3: one parameterised SCREEN scorer for ADAF-pipeline configurations.

Reuses the frozen pieces rather than re-deriving them:
  * labels (101 full / 94 pilot-excluded common parts), review marks and recorded ranked pools come from
    verify/v3/v3_frozen_mixture.py's hash-checked loader (MIXTURE_INPUTS.json);
  * pools use the historical score_staged.global_nms (score desc, id asc; inclusive 60 m) over the cohort's
    blocks, then the frozen ADAF_MEAN re-ranking of that fixed pool (score, mean_probability, area_m2 desc,
    id asc) — the rule behind the recorded ADAF_MEAN pools, which the scorer must rebuild exactly first;
  * matching is the frozen greedy nearest-unused 60 m assignment (match_greedy); criterion 3 is the frozen
    30 m catalogue-mark proximity at B = 311 on pilot-excluded SCREEN.

  python screen_scorer.py reproduce            # rebuild recorded pools + frozen controls (must pass first)
  python screen_scorer.py score CONFIG [...]   # score outputs/campaign_2026-09-22/screen/CONFIG/proposals
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from verify.v3 import v3_e0099_landscape as land  # noqa: E402
from verify.v3 import v3_frozen_mixture as mix  # noqa: E402

MIXTURE_MANIFEST = ROOT / "targets/lidar-archaeology/v3_pro/luna_master_2026-09-22/MIXTURE_INPUTS.json"
RECORDED_AO = ROOT / "work/adaf_followup_2026-09-19/english/objects/proposals/ADAF"
RUN5_NATIVE = ROOT / "work/screen_step1_1b_2026-09-20/run5/objects/proposals/ADAF_NATIVE"
E0099_CANDIDATES = ROOT / "targets/lidar-archaeology/v3_pro/e0099_landscape_2026-09-22/production/candidates"
SCREEN_OUT = ROOT / "outputs/campaign_2026-09-22/screen"
PILOT = "SCREEN_113_53"
BUDGETS = {"full_26": {"0.5": 323, "1.0": 647, "2.0": 1295, "K1294": 1294},
           "pilot_excluded_25": {"0.5": 311, "1.0": 622, "headline": 1229}}
FROZEN_CONTROLS = {("full_26", "K1294"): (46, 46), ("full_26", "0.5"): (32, 34), ("full_26", "1.0"): (40, 40),
                   ("pilot_excluded_25", "0.5"): (30, 31), ("pilot_excluded_25", "1.0"): (39, 36),
                   ("pilot_excluded_25", "headline"): (44, 41)}
FROZEN_PROXIMITY_B311 = {"ADAF_MEAN": 47, "E0099": 40}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_rank_key(p: dict) -> tuple:
    return (-float(p["score"]), -float(p["mean_probability"]), -float(p["area_m2"]), str(p["id"]))


def frozen_inputs() -> tuple[dict, list[dict], object]:
    _, paths = mix._manifest(MIXTURE_MANIFEST)
    data, summary, review = mix._load_checked(paths)
    blocks = read(paths["catalogue_manifest"])["blocks"]
    stage = land.historical_scorer()
    return data, blocks, stage


def cohort_blocks(blocks: list[dict]) -> dict[str, list[str]]:
    ids = [b["id"] for b in blocks]
    return {"full_26": ids, "pilot_excluded_25": [b for b in ids if b != PILOT]}


def build_pool(stage, rows_by_block: dict[str, list[dict]], block_ids: list[str], mean_rank: bool) -> list[dict]:
    pool = stage.global_nms([p for bid in block_ids for p in rows_by_block[bid]])
    return sorted(pool, key=mean_rank_key) if mean_rank else pool


def as_mixture_rows(pool: list[dict], arm: str) -> list[dict]:
    return [{"arm": arm, "original_id": p["id"], "easting": float(p["easting"]), "northing": float(p["northing"]),
             "source_block": p["source_block"], "original_rank": i + 1} for i, p in enumerate(pool)]


def score_pool(pool: list[dict], arm: str, data: dict, cohort: str) -> dict:
    rows = as_mixture_rows(pool, arm)
    out = {"supply": len(rows)}
    for budget_id, budget in BUDGETS[cohort].items():
        k = min(budget, len(rows))
        selected = [dict(r, fused_rank=i + 1) for i, r in enumerate(rows[:k])]
        hits = mix.match_greedy(selected, data["labels"][cohort])
        entry = {"requested": budget, "K": k, "H": len(hits), "supply_limited": k < budget}
        if cohort == "pilot_excluded_25" and budget_id == "0.5":
            prox = mix.proximity(selected, data["marks"][cohort])
            entry["proximity_30m_catalogue"] = prox["catalogue"]["within_30m"]
            entry["proximity_denominator"] = prox["catalogue"]["denominator"]
            entry["proximity_15m_catalogue"] = prox["catalogue"]["within_15m"]
            entry["proximity_60m_catalogue"] = prox["catalogue"]["within_60m"]
        out[budget_id] = entry
    return out


def recorded_ao_rows() -> dict[str, list[dict]]:
    """Recorded AO proposals joined to run5's native mean_probability (same ids, coordinates, scores)."""
    out = {}
    for path in sorted(RECORDED_AO.glob("SCREEN_*.json")):
        bid = path.stem
        rows = read(path)["objects"]
        native = {o["id"]: o for o in read(RUN5_NATIVE / f"{bid}.json")["objects"]}
        joined = []
        for r in rows:
            n = native[r["id"]]
            if (abs(float(n["easting"]) - float(r["easting"])) > 1e-6 or
                    abs(float(n["northing"]) - float(r["northing"])) > 1e-6 or float(n["score"]) != float(r["score"])):
                raise RuntimeError(f"native join mismatch {r['id']}")
            joined.append(dict(r, source_block=bid, mean_probability=float(n["mean_probability"]),
                               area_m2=float(r["area_m2"])))
        out[bid] = joined
    return out


def e0099_rows() -> dict[str, list[dict]]:
    return {p.stem: [dict(o, source_block=p.stem) for o in read(p)["objects"]]
            for p in sorted(E0099_CANDIDATES.glob("SCREEN_*.json"))}


def config_rows(config: str) -> dict[str, list[dict]]:
    out = {}
    for p in sorted((SCREEN_OUT / config / "proposals").glob("SCREEN_*.json")):
        rec = read(p)
        out[p.stem] = [dict(o, source_block=p.stem) for o in rec["objects"]]
    if len(out) != 26:
        raise RuntimeError(f"{config}: expected 26 block files, found {len(out)}")
    return out


def reproduce() -> dict:
    data, blocks, stage = frozen_inputs()
    cohorts = cohort_blocks(blocks)
    ao, e99 = recorded_ao_rows(), e0099_rows()
    checks, scores = {}, {}
    for cohort, ids in cohorts.items():
        rebuilt = {"ADAF_MEAN": build_pool(stage, ao, ids, True), "E0099": build_pool(stage, e99, ids, False)}
        for arm in ("ADAF_MEAN", "E0099"):
            recorded = [r["original_id"] for r in data["pools"][cohort][arm]]
            checks[f"{cohort}/{arm}/pool_identical"] = [p["id"] for p in rebuilt[arm]] == recorded
            scores[f"{cohort}/{arm}"] = score_pool(rebuilt[arm], arm, data, cohort)
    for (cohort, budget_id), expected in FROZEN_CONTROLS.items():
        actual = tuple(scores[f"{cohort}/{arm}"][budget_id]["H"] for arm in ("ADAF_MEAN", "E0099"))
        checks[f"control/{cohort}/{budget_id}"] = actual == expected
    for arm, expected in FROZEN_PROXIMITY_B311.items():
        checks[f"proximity_B311/{arm}"] = (
            scores[f"pilot_excluded_25/{arm}"]["0.5"]["proximity_30m_catalogue"] == expected)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "scores": scores,
              "rules": {"pool": "score_staged.global_nms (score desc, id asc, inclusive 60 m) over cohort blocks",
                        "rank": "ADAF-pipeline pools re-ranked by score, mean_probability, area_m2 desc, id asc",
                        "match": "v3_frozen_mixture.match_greedy (nearest unused <= 60 m, label id tie)",
                        "criterion3": "v3_frozen_mixture.proximity catalogue within_30m at B=311, pilot-excluded"},
              "scorer_sha256": sha(Path(__file__)), "mixture_manifest_sha256": sha(MIXTURE_MANIFEST)}
    (SCREEN_OUT / "SCORER_REPRODUCTION.json").parent.mkdir(parents=True, exist_ok=True)
    (SCREEN_OUT / "SCORER_REPRODUCTION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def score(config: str) -> dict:
    repro = read(SCREEN_OUT / "SCORER_REPRODUCTION.json")
    if repro["status"] != "PASS" or repro["scorer_sha256"] != sha(Path(__file__)):
        raise RuntimeError("run `reproduce` with this scorer version first")
    data, blocks, stage = frozen_inputs()
    rows = config_rows(config)
    result = {"config": config, "scorer_sha256": sha(Path(__file__)),
              "proposal_files_sha256": {b: sha(SCREEN_OUT / config / "proposals" / f"{b}.json") for b in rows}}
    for cohort, ids in cohort_blocks(blocks).items():
        result[cohort] = score_pool(build_pool(stage, rows, ids, True), config, data, cohort)
    (SCREEN_OUT / config / "SCORE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def holdout_diagnostic(config: str, edge_m: float = 32.0, reach_m: float = 15.0) -> dict:
    """Amendment 2 F (diagnostic, not a gate), rules fixed before any R2 output: on holdout-square TRAIN windows,
    a part counts as recalled when any vectorised proposal centroid lies inside the part polygon expanded by
    15 m; parts count once, only if their centroid is >= 32 m inside some holdout window. Also reports proposal
    density on the inner area of holdout negative windows (proxy for false alarms)."""
    sys.path.insert(0, str(HERE))
    import r2_masks
    parts = {q["id"]: q for q in r2_masks.rebuild_parts()[0]}
    manifest = read(HERE / "R2_MASKS_MANIFEST_PRIVATE.json")
    objects = read(ROOT / "outputs/campaign_2026-09-22/holdout_diag" / config / "OBJECTS_PRIVATE.json")["objects"]
    found: dict[str, dict[str, bool]] = {"ignore_unlabelled_family": {}, "positive": {}}
    neg_count, neg_area = 0, 0.0
    for w in manifest["windows"]:
        if not w["holdout"]:
            continue
        a, _, x0, _, e, y1 = w["transform"]
        x1, y0 = x0 + a * w["shape"][1], y1 + e * w["shape"][0]
        inner = (x0 + edge_m, y0 + edge_m, x1 - edge_m, y1 - edge_m)
        props = [(o["easting"], o["northing"]) for o in objects[w["id"]]]
        if w["role"] == "negative":
            neg_count += sum(inner[0] <= x < inner[2] and inner[1] <= y < inner[3] for x, y in props)
            neg_area += (inner[2] - inner[0]) * (inner[3] - inner[1]) / 1e6
        ids = [p["id"] for p in w["positive_parts"]] + [p["id"] for p in w["ignore_parts"]]
        for pid in ids:
            q = parts[pid]
            if q["role"] not in found or not (inner[0] <= q["cx"] < inner[2] and inner[1] <= q["cy"] < inner[3]):
                continue
            zone = q["geom"].buffer(reach_m)
            from shapely.geometry import Point
            hit = any(zone.contains(Point(x, y)) for x, y in props)
            found[q["role"]][pid] = found[q["role"]].get(pid, False) or hit
    result = {"config": config, "rules": {"edge_m": edge_m, "reach_m": reach_m},
              **{f"{role}_parts": len(v) for role, v in found.items()},
              **{f"{role}_recalled": sum(v.values()) for role, v in found.items()},
              "negative_inner_km2": neg_area, "negative_inner_proposals": neg_count,
              "negative_proposals_per_km2": neg_count / neg_area if neg_area else None}
    (ROOT / "outputs/campaign_2026-09-22/holdout_diag" / config / "DIAGNOSTIC.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    if sys.argv[1] == "holdout":
        print(json.dumps(holdout_diagnostic(sys.argv[2])))
        sys.exit(0)
    if sys.argv[1] == "reproduce":
        r = reproduce()
        print(json.dumps({"status": r["status"], "failed": [k for k, v in r["checks"].items() if not v]}))
    elif sys.argv[1] == "score":
        for c in sys.argv[2:]:
            s = score(c)
            print(json.dumps({"config": c, **{k: s[k] for k in ("full_26", "pilot_excluded_25")}}))
    else:
        raise SystemExit(__doc__)
