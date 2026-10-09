"""Per-part failure decomposition for one region manifest scored with the frozen HR1 region_score (read-only reuse).

For each model and each eligible part: hit at 0.5/1/2 (and 4) background proposals per km2, hit anywhere in the
post-NMS pool, and any pre-NMS proposal within reach. A miss at density d is then
  MISSING  no proposal (pre-NMS) within reach of the part,
  NMS      a pre-NMS proposal existed but NMS removed every one within reach,
  RANK     a post-NMS proposal hits the part but ranks below the density threshold.
Outputs aggregate counts only to OUT_JSON; per-part rows go to a *_PRIVATE.json path supplied by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/holdout_region_2026-10/step2"))
import region_score as rs  # noqa: E402  frozen HR1 scorer
import score_holdout as shd  # noqa: E402  frozen recall_at (imported by region_score)
import r3_curation as cur  # noqa: E402  frozen nms, rank_key

DENS = (0.5, 1.0, 2.0, 4.0)
CURVE = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, None)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def part_flags(objs: dict, s: dict) -> dict:
    """Per part: pre-NMS presence, post-NMS presence and hit per density."""
    pre = set()
    for t in s["wins"]:
        for o in objs.get(t["id"], []):
            if rs.in_core(o, t):
                pre.update(s["eligible"][i] for i in s["tree"].query(Point(o["easting"], o["northing"]),
                                                                    predicate="intersects"))
    pool = rs.ranked_pool(objs, s)
    post = {p for r in pool for p in r["hits"]}
    hits = {d: shd.recall_at(pool, s, d)[1] for d in DENS}
    curve = {("all" if d is None else d): shd.recall_at(pool, s, d)[0] for d in CURVE}
    area = sum(s["neg_area"].values())
    return {"pre": pre, "post": post, "hits": hits, "curve": curve,
            "neg_per_km2": sum(r["neg"] for r in pool) / area, "pool": len(pool)}


def load_objects(path: Path, key: str, s: dict) -> dict:
    """{tile id: [objects]} from a PROPOSALS file ({"objects": ...}), a keyed holdout file or a plain mapping."""
    d = json.loads(path.read_text())
    if "objects" in d and isinstance(d["objects"], dict):
        d = d["objects"]
    if key:
        d = d[key]
    ids = {t["id"] for t in s["wins"]}
    if not set(d) <= ids:
        raise SystemExit(f"{path}: {len(set(d) - ids)} object keys are not tiles of this manifest")
    return {t: d.get(t, []) for t in ids}


def miss_kind(p: str, f: dict, d: float) -> str:
    if p in f["hits"][d]:
        return "HIT"
    if p in f["post"]:
        return "RANK"
    return "NMS" if p in f["pre"] else "MISSING"


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path)
    src.add_argument("--english-holdout", action="store_true", help="frozen English TRAIN holdout (rs.english_manifest)")
    ap.add_argument("--proposals", nargs="+", required=True, help="NAME=path[#key] pairs")
    ap.add_argument("--attrs", type=Path, help="private JSON {part_id: {attr: value}} joined into groups")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--private-out", type=Path, required=True)
    a = ap.parse_args(argv)
    m = rs.english_manifest() if a.english_holdout else json.loads(a.manifest.read_text())
    s = rs.setup(m)
    attrs = json.loads(a.attrs.read_text()) if a.attrs else {}
    flags, hashes = {}, {}
    for spec in a.proposals:
        name, path = str(spec).split("=", 1)
        path, _, key = path.partition("#")
        path = Path(path)
        hashes[name] = sha(path) + (f"#{key}" if key else "")
        flags[name] = part_flags(load_objects(path, key, s), s)
    rows = {}
    for p in s["eligible"]:
        q = next(x for x in m["parts"] if x["id"] == p)
        rows[p] = {"class": s["klass"][p], "monument_class": q.get("monument_class"), **attrs.get(p, {}),
                   "kinds": {n: {str(d): miss_kind(p, f, d) for d in DENS} for n, f in flags.items()}}
    out = {"manifest_sha256": "english_manifest()" if a.english_holdout else sha(a.manifest), "proposals_sha256": hashes, "eligible_parts": len(s["eligible"]),
           "background_km2": float(sum(s["neg_area"].values())),
           "models": {n: {"neg_per_km2": f["neg_per_km2"], "pool": f["pool"], "curve": f["curve"]}
                      for n, f in flags.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.private_out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    a.private_out.write_text(json.dumps(rows, indent=None))
    print(json.dumps({"parts": len(rows), "models": list(flags)}, indent=None))


if __name__ == "__main__":
    main(sys.argv[1:])
