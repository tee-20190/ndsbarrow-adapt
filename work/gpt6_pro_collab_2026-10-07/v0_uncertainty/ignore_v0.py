"""Task 4: IGNORE under v0 (ROUND3_REPLY.md:646-679). Three interpretations, scored with the boot_v0 engine on the
primary 2 km grid, the same block weights (same stream and J) and the same training draws:

(i)   historical: the frozen implementation. No IGNORE test inside positive cores; background boxes already exclude
      IGNORE by construction (confirm_build.py:244-247; region_score.py:157-169). This is plain v0 (BENCH_V0).
(ii)  literal registered wording (FRESH1_PREPIXEL.md:30, rule 6; adapt_nds PACKET.md:26-28): IGNORE = every register
      record that is not an eligible entity, plus the edge-supplement records, buffered 30 m (step0.py:661-665).
      Proposals there are "neither hits nor false alarms": removed before the budget. The IGNORE area is removed from
      the scored area (consistent treatment), K = floor(b x A'). N stays the registered eligible count; entities
      whose whole 15 m reach lies inside IGNORE are unreachable and reported.
(iii) explicit precedence: as (ii), but Grabhuegelfeld (barrow cemetery) and Grabungsschutzgebiet (excavation
      protection area) polygons are not IGNORE, and every eligible entity's 15 m reach zone takes precedence over any
      remaining IGNORE zone. A fixed geographical mask; no proposal-label outcome is used.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/gpt6_pro_collab_2026-10-07/v0_uncertainty/ignore_v0.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import boot_v0 as BT  # noqa: E402
import engine as E  # noqa: E402
import v0data as D  # noqa: E402

CONTAINERS = ("Grabhügelfeld", "Grabungsschutzgebiet")
IGNORE_M = 30.0
KEY = {"oldenburg": [("A", "C"), ("A", "ADAF_barrow")], "verden": [("DEPLOY", "ADAF_AO"), ("DEPLOY", "ADAF_barrow")]}


def ignore_records(name: str, eligible_ids: set[int]) -> list[tuple[str, object]]:
    """(objekttyp, geometry) of every IGNORE record: register non-entities and the edge supplement."""
    reg = json.loads((D.ROOT / D.REGISTER[name]).read_text(encoding="utf-8"))
    sup = json.loads((D.ROOT / D.SUPPLEMENT[name]).read_text(encoding="utf-8"))
    out = [(f["properties"].get("objekttyp"), shapely.make_valid(shape(f["geometry"])))
           for f in reg["features"] if f.get("geometry") and int(f["id"]) not in eligible_ids]
    out += [((r.get("properties") or {}).get("objekttyp"), shapely.make_valid(shape(r["geometry"])))
            for r in sup["records"] if r.get("geometry")]
    return out


def masks(reg: dict) -> dict:
    recs = ignore_records(reg["name"], set(reg["record_ids"]))
    lit = shapely.union_all([g.buffer(IGNORE_M) for _, g in recs])
    reach = shapely.union_all([fp.buffer(D.rs.REACH_M) for fp in reg["footprints"]])
    prec = shapely.union_all([g.buffer(IGNORE_M) for t, g in recs if t not in CONTAINERS]).difference(reach)
    counts = {"ignore_records": len(recs), "container_records": sum(t in CONTAINERS for t, _ in recs),
              "by_container_type": {t: sum(x == t for x, _ in recs) for t in CONTAINERS}}
    return {"literal": lit, "precedence": prec, "counts": counts}


def unreachable(reg: dict, mask) -> int:
    return sum(fp.buffer(D.rs.REACH_M).difference(mask).area < 1e-6 for fp in reg["footprints"])


def run_region(name: str, ss: dict) -> dict:
    reg = D.load_region(name)
    size, off = BT.GRIDS[BT.PRIMARY_GRID]
    blk = E.build_blocks(reg["boxes"], reg["footprints"], D.rs.REACH_M, size, off)
    w = E.multinomial_weights(np.random.default_rng(ss[f"geo_{name}"]), len(blk.area_km2), BT.REPS)  # = boot_v0
    dsg = BT.design(name)
    idx_train = dsg["draw"](np.random.default_rng(ss[f"train_{name}"]), BT.REPS)
    mk = masks(reg)
    out = {"mask_counts": mk["counts"], "variants": {}}
    for vname, mask in (("historical", None), ("literal", mk["literal"]), ("precedence", mk["precedence"])):
        area_j = blk.masked_area_km2(mask)
        v = {"scored_km2": float(area_j.sum()), "K": {str(b): int(E.budget(b, area_j.sum())) for b in BT.B},
             "N": reg["N"], "unreachable_entities": 0 if mask is None else unreachable(reg, mask),
             "decoders": {}}
        if mask is not None:
            shapely.prepare(mask)
        for dec in BT.DECODERS:
            geo, point, removed = {}, {}, {}
            for sysname, decs in reg["pools"].items():
                pool = decs[dec]
                keep = np.ones(len(pool.x), dtype=bool) if mask is None else ~shapely.contains_xy(mask, pool.x,
                                                                                                    pool.y)
                sub = pool.subset(keep)
                removed[sysname] = int((~keep).sum())
                bo = blk.block_of_points(sub.x, sub.y)
                delta = E.increments(sub.edges, sub.n_labels)
                nj = blk.n_labels.astype(float)
                geo[sysname] = BT.recall_reps(bo, delta, w, area_j, nj)[0]
                point[sysname] = BT.recall_reps(bo, delta, np.ones((1, len(area_j)), dtype=np.int64), area_j, nj)[0]
            dsg_k = dsg | {"contrasts": KEY[name]}
            v["decoders"][dec] = {"proposals_removed": removed,
                                  "systems_point": {s: BT.pt4(p[0]) for s, p in point.items()},
                                  "contrasts": BT.summaries(dsg_k, geo, point, idx_train, BT.REPS)}
            if v["unreachable_entities"]:
                n_r = reg["N"] - v["unreachable_entities"]
                v["decoders"][dec]["systems_point_over_reachable_N"] = {
                    s: BT.pt4(p[0] * reg["N"] / n_r) for s, p in point.items()}
        out["variants"][vname] = v
        print(name, vname, v["scored_km2"], v["K"], v["unreachable_entities"], flush=True)
    return out


def main() -> None:
    ss = BT.streams()
    res = {"schema": "v0_uncertainty_ignore_v1", "master_seed": BT.MASTER_SEED, "reps": BT.REPS,
           "grid": BT.PRIMARY_GRID, "containers": list(CONTAINERS), "ignore_buffer_m": IGNORE_M,
           "regions": {n: run_region(n, ss) for n in ("oldenburg", "verden")}}
    (HERE / "IGNORE_V0.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
