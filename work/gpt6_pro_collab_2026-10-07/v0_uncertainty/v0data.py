"""Frozen v0 inputs for the uncertainty engine: manifests, scored area, every system's complete ranked pool (NMS60 and
RAW) and its proposal-reference edges. Pools come from detector_week1/bench_v0.py functions imported unchanged
(read-only), so all-ones block weights must reproduce BENCH_V0.json exactly. Nothing here writes to the project.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WEEK1 = HERE.parent / "detector_week1"
sys.path.insert(0, str(WEEK1))
import bench_v0 as bv  # noqa: E402  read-only: REGIONS, load, raw_pool, deploy_objs, rs (frozen region_score)

rs = bv.rs
J_SEEDS_VERDEN = ("2026092802", "2026092803", "2026092804", "2026092805", "2026092806")
SSTAR = "J_2026092803"
FRESH_FREEZE = ROOT / "work/fresh_region_2026-10-03/step2/freeze/FREEZE.json"
SUPPLEMENT = {"verden": "outputs/fresh_region_2026-10-03/step0/EDGE_SUPPLEMENT_PRIVATE.json",
              "oldenburg": "outputs/adapt_nds_2026-10-05/register/supplement_Oldenburg_20261005T010102Z.json"}
REGISTER = {"verden": "outputs/fresh_region_2026-10-03/labels/denkmale_Verden_20261003T084547Z.geojson",
            "oldenburg": "outputs/adapt_nds_2026-10-05/register/denkmale_Oldenburg_20261005T010102Z.geojson"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass
class Pool:
    x: np.ndarray
    y: np.ndarray
    edges: list[np.ndarray]
    n_labels: int

    def subset(self, keep: np.ndarray) -> "Pool":
        idx = np.flatnonzero(keep)
        return Pool(self.x[idx], self.y[idx], [self.edges[i] for i in idx], self.n_labels)


def pool_of(rows: list[dict], s: dict) -> Pool:
    x = np.array([r["easting"] for r in rows], dtype=float)
    y = np.array([r["northing"] for r in rows], dtype=float)
    pts = shapely.points(x, y)
    edges = [np.sort(np.asarray(s["tree"].query(p, predicate="intersects"), dtype=np.int64)) for p in pts]
    return Pool(x, y, edges, len(s["eligible"]))


def sstar_macro() -> dict[str, float]:
    """FRESH-1 S* rule inputs: region-macro DEV primary per JOINT-1 confirmation seed (FREEZE.json sstar)."""
    f = json.loads(FRESH_FREEZE.read_text(encoding="utf-8"))
    return {f"J_{k}": float(v) for k, v in f["sstar"]["resolution"]["macro"].items()}


def systems_for(name: str) -> list[str]:
    spec = bv.REGIONS[name]
    if name == "verden":
        return ["ADAF_AO", "ADAF_barrow", "CONT"] + [f"J_{s}" for s in J_SEEDS_VERDEN]
    return list(spec["models"])


def load_region(name: str) -> dict:
    spec = bv.REGIONS[name]
    ip = ROOT / spec["individual"]
    mi = json.loads(ip.read_text(encoding="utf-8"))
    si = rs.setup(mi)
    boxes = [tuple(float(v) for v in b) for t in mi["tiles"] for b in t["core"]]
    area = bv.unary_union([bv.box(*b) for t in mi["tiles"] for b in t["core"]]).area / 1e6  # bench_v0.py:111
    parts = {q["id"]: q for q in mi["parts"]}
    footprints = [shape(parts[pid]["geometry"]) for pid in si["eligible"]]
    unit = ROOT / spec["dir"] / "individual"
    objs = {m: bv.load(unit, m) for m in systems_for(name)}
    inputs = {str(ip.relative_to(ROOT)): sha(ip)}
    inputs |= {str((unit / f"PROPOSALS_{m}_PRIVATE.json").relative_to(ROOT)):
               sha(unit / f"PROPOSALS_{m}_PRIVATE.json") for m in objs}
    if name == "verden":
        for s in J_SEEDS_VERDEN:
            objs[f"DEPLOY_J_{s}"] = bv.deploy_objs(objs["ADAF_AO"], objs[f"J_{s}"], si)
    pools = {m: {"NMS60": pool_of(rs.ranked_pool(o, si), si), "RAW": pool_of(bv.raw_pool(o, si), si)}
             for m, o in objs.items()}
    return {"name": name, "si": si, "manifest": mi, "area_km2": area, "boxes": boxes, "footprints": footprints,
            "N": len(si["eligible"]), "record_ids": [int(parts[pid]["record_id"]) for pid in si["eligible"]],
            "pools": pools, "inputs": inputs}


def bench_v0() -> dict:
    return json.loads((WEEK1 / "BENCH_V0.json").read_text(encoding="utf-8"))
