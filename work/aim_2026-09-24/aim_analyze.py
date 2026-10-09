"""AIM_PREREG.md Use A (13 plausible detections) and Use B (SCREEN known-earthwork yield), as declared.

  python -B work/aim_2026-09-24/aim_analyze.py screen   # Use B: R2, AO_regression (+ E39 when its 26 blocks exist)
  python -B work/aim_2026-09-24/aim_analyze.py premise  # Use A -> USE_A_PRIVATE.json (candidate coordinates)
(WSL adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
FAMILY = re.compile(r"\b(?:barrows?\b|tumul|ring ditch|cairn|mound|enclosure|henge)", re.I)
BARROW_FAMILY = re.compile(r"\b(?:barrows?\b|tumul|ring ditch|cairn|mound)", re.I)
ENCLOSURE_FAMILY = re.compile(r"\b(?:enclosure|henge)", re.I)
RADII = (10.0, 20.0, 40.0)
PRIMARY_R = 20.0
BUDGETS = (323, 1294)
BOOT_N, BOOT_SEED = 2000, 20260924
ARM_DIRS = {"R2": ROOT / "outputs/campaign_2026-09-22/screen/R2",
            "AO": ROOT / "outputs/campaign_2026-09-22/screen/AO_regression",
            "E39": ROOT / "outputs/tta_ensemble_2026-09-24/screen/E39"}


def read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def layer(bid: str, n: int, name: str) -> tuple[list[dict], object]:
    d = read(DATA / f"{bid}_L{n}_{name}.geojson")
    return [f for f in d["features"] if f.get("geometry")], shape(d["query_geometry"])


def extant(props: dict) -> bool:
    toks = set()
    for k in ("EVIDENCE_1", "EVIDENCE_2"):
        toks |= {t.strip() for t in str(props.get(k) or "").split("\\")}
    return "EARTHWORK" in toks


def lidar_completed(props: dict) -> bool:
    return str(props.get("Status")) == "Completed" and "lidar" in str(props.get("Sources") or "").lower()


def feature_sets(feats: list[dict]) -> dict[str, list[tuple]]:
    out = {"ANY": [], "T": [], "T_EXTANT": [], "T_EXTANT_BARROW": [], "T_EXTANT_ENCLOSURE": []}
    for f in feats:
        p, g = f["properties"], shape(f["geometry"])
        uid = p.get("HE_UID") or f"OBJ{p.get('OBJECTID')}"
        mt = str(p.get("MONUMENT_TYPE") or "")
        out["ANY"].append((g, uid))
        if FAMILY.search(mt):
            out["T"].append((g, uid))
            if extant(p):
                out["T_EXTANT"].append((g, uid))
                if BARROW_FAMILY.search(mt):
                    out["T_EXTANT_BARROW"].append((g, uid))
                if ENCLOSURE_FAMILY.search(mt):
                    out["T_EXTANT_ENCLOSURE"].append((g, uid))
    return out


def index(items: list[tuple]) -> tuple:
    return (STRtree([g for g, _ in items]) if items else None, [u for _, u in items])


def near(idx: tuple, pt: Point, r: float) -> set:
    tree, uids = idx
    if tree is None:
        return set()
    return {uids[i] for i in tree.query(pt, predicate="dwithin", distance=r)}


def screen() -> dict:
    import screen_scorer as ss
    import adaf_configs as ac
    data, blocks, stage = ss.frozen_inputs()
    ids = ss.cohort_blocks(blocks)["full_26"]
    interiors = {b["id"]: box(*b["interior"]) for b in ac.screen_blocks()}
    cov, sets, denom, trees = {}, {}, {}, {}
    for bid in ids:
        feats, qg = layer(bid, 0, "Detailed_Mapping")
        areas, _ = layer(bid, 2, "Project_Area")
        lid = [shape(a["geometry"]) for a in areas if lidar_completed(a["properties"])]
        cov[bid] = unary_union(lid).intersection(qg).intersection(interiors[bid]) if lid else None
        sets[bid] = feature_sets(feats)
        trees[bid] = {k: index(v) for k, v in sets[bid].items()}
        c = cov[bid]
        denom[bid] = {k: len({u for g, u in v if c is not None and g.intersects(c)}) for k, v in sets[bid].items()}
    arms = [a for a, d in ARM_DIRS.items() if len(list((d / "proposals").glob("SCREEN_*.json"))) == 26]
    out = {"arms": arms, "coverage_km2": {b: round(c.area / 1e6, 2) if c is not None else 0.0 for b, c in cov.items()},
           "covered_blocks": sum(c is not None and c.area > 0 for c in cov.values()),
           "denominators_distinct_uid": {k: sum(d[k] for d in denom.values()) for k in sets[ids[0]]},
           "results": {}}
    per_block = {}
    for arm in arms:
        rows = {p.stem: [dict(o, source_block=p.stem) for o in read(p)["objects"]]
                for p in sorted((ARM_DIRS[arm] / "proposals").glob("SCREEN_*.json"))}
        pool = ss.build_pool(stage, rows, ids, True)
        res = out["results"][arm] = {"pool": len(pool)}
        for k in BUDGETS:
            top = pool[:k]
            ent = {"covered": 0}
            blk = Counter()
            uids = {r: {s: set() for s in sets[ids[0]]} for r in RADII}
            counts = {r: Counter() for r in RADII}
            for p in top:
                b = p["source_block"]
                pt = Point(float(p["easting"]), float(p["northing"]))
                if cov[b] is None or not cov[b].contains(pt):
                    continue
                ent["covered"] += 1
                for r in RADII:
                    for s, idx_s in trees[b].items():
                        hit = near(idx_s, pt, r)
                        if hit:
                            counts[r][s] += 1
                            uids[r][s] |= hit
                            if r == PRIMARY_R and s == "T_EXTANT":
                                blk[b] += 1
            for r in RADII:
                ent[f"r{int(r)}"] = {"proposals_near": dict(counts[r]),
                                     "distinct_monuments": {s: len(v) for s, v in uids[r].items()}}
            ent["Y_T_by_block"] = [blk[b] for b in ids]
            res[f"K{k}"] = ent
        per_block[arm] = {k: np.array(res[f"K{k}"]["Y_T_by_block"]) for k in BUDGETS}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(ids), size=(BOOT_N, len(ids)))
    out["paired_block_bootstrap_Y_T_diff_90"] = {}
    for a, b in (("R2", "AO"), ("E39", "R2")):
        if a in per_block and b in per_block:
            out["paired_block_bootstrap_Y_T_diff_90"][f"{a}-{b}"] = {
                f"K{k}": [float(q) for q in np.quantile(per_block[a][k][idx].sum(1) - per_block[b][k][idx].sum(1), [.05, .95])]
                for k in BUDGETS}
    return out


def premise() -> dict:
    members = read(HERE / "TILE_MEMBERS_PRIVATE.json")
    key = {k["code"]: k for k in read(ROOT / "outputs/r3_pilot_2026-09-23/curation/KEY_PRIVATE.json")["key"]}
    notes = {json.loads(l)["code"]: json.loads(l)["note"] for l in
             (ROOT / "outputs/r3_pilot_2026-09-23/curation/premise/stage1_verdicts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    nhle = [shape(f["geometry"]) for f in read(ROOT / "work/astra_v2/access/nhle_scheduled_monuments.geojson")["features"]]
    nhle_tree = STRtree(nhle)
    out = []
    for tid, codes in sorted(members.items()):
        dm, _ = layer(tid, 0, "Detailed_Mapping")
        me, _ = layer(tid, 1, "Monument_Extents")
        pa, _ = layer(tid, 2, "Project_Area")
        for code in codes:
            pt = Point(key[code]["easting"], key[code]["northing"])
            cands = [("Detailed_Mapping", f, shape(f["geometry"]).distance(pt)) for f in dm] + \
                    [("Monument_Extents", f, shape(f["geometry"]).distance(pt)) for f in me]
            cands.sort(key=lambda t: t[2])
            within100 = [(l, f["properties"], round(d, 1)) for l, f, d in cands if d <= 100]
            projects = [a["properties"] for a in pa if shape(a["geometry"]).contains(pt)]
            done = [p for p in projects if str(p.get("Status")) == "Completed"]
            nearest_d = cands[0][2] if cands else None
            if nearest_d is not None and nearest_d <= 30:
                cls = "RECORDED_AIM"
            elif any(lidar_completed(p) for p in done):
                cls = "NOT_IN_AIM_LIDAR_AREA"
            elif done:
                cls = "NOT_IN_AIM_AREA_NO_LIDAR"
            else:
                cls = "OUTSIDE_AIM_COVERAGE"
            nd = min(nhle[i].distance(pt) for i in nhle_tree.query(pt.buffer(20000))) if len(nhle_tree.query(pt.buffer(20000))) else None
            keep = ("MONUMENT_TYPE", "PERIOD", "EVIDENCE_1", "HE_UID", "HER_No", "HER_NO", "project")
            out.append({"code": code, "tile": tid, "class": cls, "note": notes[code],
                        "nearest_aim_m": None if nearest_d is None else round(nearest_d, 1),
                        "nearest_aim": None if not cands else {"layer": cands[0][0], **{k: cands[0][1]["properties"].get(k)
                                                                                         for k in keep if k in cands[0][1]["properties"]}},
                        "aim_within_100m": [{"layer": l, "d_m": d, **{k: p.get(k) for k in keep if k in p}} for l, p, d in within100[:8]],
                        "projects": [{k: p.get(k) for k in ("PROJECT_NA", "Status", "Sources", "Year")} for p in projects],
                        "nearest_nhle_m": None if nd is None else round(nd, 1)})
    return {"detections": out, "class_counts": dict(Counter(r["class"] for r in out))}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "screen":
        r = screen()
        (HERE / "USE_B_SCREEN_YIELD.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in r.items() if k != "coverage_km2"}, indent=1))
    elif cmd == "premise":
        r = premise()
        (HERE / "USE_A_PRIVATE.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps(r["class_counts"]))
    else:
        raise SystemExit(__doc__)
