"""FUSION-1: source-preserving, spatially deduplicated unions of two detectors at the same total review budget.

Rules (fixed 2026-10-03 before any fused score was computed; no averaging of uncalibrated scores):
  INTERLEAVE  alternate the two models' own ranked pools (A1, B1, A2, B2, ...)
  A_FIRST     model A's whole ranked pool, then model B's
  RRF         reciprocal rank fusion, k = 60: own 1/(k + rank) plus the best 1/(k + rank) of the other model's
              proposals within 60 m
Each model's ranked pool is region_score's (in-core objects, 60 m NMS, rank_key). The fused order becomes a synthetic
score; region_score then re-applies its 60 m NMS (cross-model duplicates keep the earlier one) and the frozen budget
rule, so background is counted once at the same total budget. ORACLE (hit by either model at budget d each, i.e.
twice the background) is a diagnostic ceiling, not an achievable detector.

  python fuse_eval.py --region NAME (--manifest M | --english-holdout) --a NAME=PATH[#key] --b NAME=PATH[#key] --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/holdout_region_2026-10/step2"))
import region_score as rs  # noqa: E402  frozen HR1 scorer
import score_holdout as shd  # noqa: E402
import r3_curation as cur  # noqa: E402

DENS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
PRIMARY = (0.5, 1.0, 2.0)
RRF_K, DEDUP_M = 60.0, 60.0


def load_objects(spec: str, s: dict) -> tuple[str, dict]:
    name, path = spec.split("=", 1)
    path, _, key = path.partition("#")
    d = json.loads(Path(path).read_text())
    if "objects" in d and isinstance(d["objects"], dict):
        d = d["objects"]
    if key:
        d = d[key]
    ids = {t["id"] for t in s["wins"]}
    if not set(d) <= ids:
        raise SystemExit(f"{path}: object keys outside the manifest")
    return name, {t: d.get(t, []) for t in ids}


def own_pool(objs: dict, s: dict, tag: str) -> list[dict]:
    pool = rs.ranked_pool(objs, s)
    for i, r in enumerate(pool):
        r["src"], r["src_rank"], r["id"] = tag, i, f"{tag}:{r['id']}"
    return pool


def order(rule: str, pa: list[dict], pb: list[dict]) -> list[dict]:
    if rule == "A_FIRST":
        return pa + pb
    if rule == "INTERLEAVE":
        out = []
        for i in range(max(len(pa), len(pb))):
            out += [p[i] for p in (pa, pb) if i < len(p)]
        return out
    if rule == "RRF":
        def best_other(r, other_grid):
            gx, gy = int(r["easting"] // DEDUP_M), int(r["northing"] // DEDUP_M)
            best = math.inf
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for q in other_grid.get((gx + dx, gy + dy), []):
                        if math.hypot(q["easting"] - r["easting"], q["northing"] - r["northing"]) <= DEDUP_M:
                            best = min(best, q["src_rank"])
            return 0.0 if best == math.inf else 1.0 / (RRF_K + best + 1)
        grids = []
        for p in (pa, pb):
            g = {}
            for q in p:
                g.setdefault((int(q["easting"] // DEDUP_M), int(q["northing"] // DEDUP_M)), []).append(q)
            grids.append(g)
        scored = [(1.0 / (RRF_K + r["src_rank"] + 1) + best_other(r, grids[1]), r) for r in pa] + \
                 [(1.0 / (RRF_K + r["src_rank"] + 1) + best_other(r, grids[0]), r) for r in pb]
        return [r for _, r in sorted(scored, key=lambda t: (-t[0], t[1]["src"], t[1]["src_rank"]))]
    raise SystemExit(rule)


def fused_objects(seq: list[dict], s: dict) -> dict:
    n = len(seq)
    objs = {t["id"]: [] for t in s["wins"]}
    for i, r in enumerate(seq):
        o = {k: r[k] for k in ("easting", "northing", "mean_probability", "area_m2") if k in r}
        o.update(id=r["id"], score=1.0 - i / (n + 1), src=r["src"])
        objs[r["window"]].append(o)
    return objs


def curve(pool: list[dict], s: dict) -> dict:
    out = {str(d): shd.recall_at(pool, s, d)[0] for d in DENS}
    out["primary"] = sum(out[str(d)] for d in PRIMARY) / len(PRIMARY)
    area = sum(s["neg_area"].values())
    out["neg_per_km2_pool"] = sum(r["neg"] for r in pool) / area
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path)
    src.add_argument("--english-holdout", action="store_true")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", action="append", required=True, help="one or more B models (seeds are averaged by caller)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    m = rs.english_manifest() if a.english_holdout else json.loads(a.manifest.read_text())
    s = rs.setup(m)
    an, ao = load_objects(a.a, s)
    res = {"region": a.region, "a": an, "b": [], "models": {}, "fused": {}, "oracle": {}}
    pa = own_pool(ao, s, an)
    res["models"][an] = curve(rs.ranked_pool(ao, s), s)
    for spec in a.b:
        bn, bo = load_objects(spec, s)
        res["b"].append(bn)
        res["models"][bn] = curve(rs.ranked_pool(bo, s), s)
        pb = own_pool(bo, s, bn)
        for rule in ("INTERLEAVE", "A_FIRST", "RRF"):
            fo = fused_objects(order(rule, [dict(r) for r in pa], [dict(r) for r in pb]), s)
            pool = rs.ranked_pool(fo, s)
            c = curve(pool, s)
            for d in PRIMARY:
                hit = shd.recall_at(pool, s, d)[1]
                c[f"src_share_d{d}"] = {src: sum(1 for r in pool if r.get("src") == src and set(r["hits"]) & hit)
                                         for src in (an, bn)}
            res["fused"][f"{rule}|{bn}"] = c
        poa, pob = rs.ranked_pool(ao, s), rs.ranked_pool(bo, s)
        res["oracle"][bn] = {str(d): len(shd.recall_at(poa, s, d)[1] | shd.recall_at(pob, s, d)[1]) / len(s["eligible"])
                             for d in DENS}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: round(v["primary"], 4) for k, v in {**res["models"], **res["fused"]}.items()}))


if __name__ == "__main__":
    main(sys.argv[1:])
