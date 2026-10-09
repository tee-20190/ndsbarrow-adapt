"""D1 BARROW_L vs unchanged ADAF barrow under the three IGNORE readings (manuscript gap 3.2). CPU only, saved proposals
only: no inference, no new proposals, no frozen file modified (all project modules are imported read-only).

Readings exactly as implemented in work/gpt6_pro_collab_2026-10-07/v0_uncertainty/ignore_v0.py (masks(), unreachable()
and the per-variant loop of run_region()):
  (i)   historical: no IGNORE mask (= Stage 1 / Oldenburg primary scorer; = evaluator v0.1 on both districts);
  (ii)  literal: every non-entity register record plus edge-supplement record, buffered 30 m; proposals inside removed
        before K, mask area removed from the scored area, N unchanged (unreachable entities reported);
  (iii) explicit precedence: as (ii) without Grabhuegelfeld / Grabungsschutzgebiet, minus every eligible 15 m reach.
Proposals and pools: Verden from desc_verden.load() (Stage 1 provenance, sha256-checked); Oldenburg from the verified
transfer session via score_oldenburg's own loaders and pins. Bootstrap: frozen v0 engine, 2 km primary grid, geography
stream geo_<district> (same weights as BOOT_V0 / IGNORE_V0), training draws = desc_verden.train_draws() (shared,
frozen); contrast = desc_verden.contrast (D1 ten-run mean - ADAF barrow; score each checkpoint, then average).
Self-checks: reading (i) must reproduce the published D1 results exactly (Stage 1 / D1-VERDEN-DESC / OLDENBURG RESULT),
and ADAF barrow under (ii)/(iii) must reproduce IGNORE_V0.json. Aggregates only (no ids, no coordinates) are written.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/manuscript_2026-10/ignore_rescore/rescore_d1_ignore.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in ("work/stage1_d1_v0_2026-10-07", "work/gpt6_pro_collab_2026-10-07/v0_uncertainty",
          "work/evaluator_v0_1_2026-10-07", "work/d1_verden_descriptive_2026-10-07",
          "work/oldenburg_transfer_2026-10-07"):
    sys.path.insert(0, str(ROOT / p))
import boot_v0 as BT  # noqa: E402
import engine as E  # noqa: E402
import v0data as D  # noqa: E402
import desc_verden as DV  # noqa: E402
import ignore_v0 as IG  # noqa: E402
import score_v0 as S1  # noqa: E402
import score_oldenburg as SO  # noqa: E402

SESSION = ROOT / "outputs/oldenburg_transfer_2026-10-07/chain_runs/run_s1_20261008T203551Z"
OT_RESULT = ROOT / "work/oldenburg_transfer_2026-10-07/RESULT.json"
VD_RESULT = ROOT / "work/d1_verden_descriptive_2026-10-07/RESULT.json"
IG_RESULT = ROOT / "work/gpt6_pro_collab_2026-10-07/v0_uncertainty/IGNORE_V0.json"
ADAF, D1 = DV.ADAF, DV.D1
TOL = 1e-12


def rj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def load_verden() -> dict:
    reg = DV.load()
    pools, _ = DV.pools_and_deltas(reg)
    parts = {q["id"]: q for q in reg["mi"]["parts"]}
    return {"name": "verden", "si": reg["si"], "boxes": reg["boxes"], "area": reg["area"],
            "footprints": reg["footprints"], "N": reg["N"], "pools": pools,
            "record_ids": [int(parts[pid]["record_id"]) for pid in reg["si"]["eligible"]],
            "published": {"exact": {n: {d: {b: reg["s1"]["systems"][n][d][b]["recall"] for b in ("0.5", "1.0", "2.0")}
                                        for d in BT.DECODERS} for n in pools}}}


def load_oldenburg() -> dict:
    ind = rj(SO.IND)
    if SO.sha(SO.IND) != SO.PIN["individual"]:
        raise SystemExit("Oldenburg individual manifest differs from pin")
    si, tiles = S1.rs.setup(ind), {t["id"] for t in ind["tiles"]}
    area = unary_union([box(*b) for t in ind["tiles"] for b in t["core"]]).area / 1e6
    bad: list[str] = []
    objs = {ADAF: S1.load_checked(SO.ADAF_PROPOSALS, ADAF, SO.PIN["adaf_proposals"], SO.PIN["adaf_checkpoint"],
                                  SO.PIN["inference"], tiles, bad)}
    objs |= SO.load_d1(SESSION, tiles, bad)
    if bad:
        raise SystemExit(f"refused: {bad}")
    pools, _ = SO.pools_of(objs, si)
    parts = {q["id"]: q for q in ind["parts"]}
    ot = rj(OT_RESULT)["exact_scorer"]
    return {"name": "oldenburg", "si": si, "area": area, "N": len(si["eligible"]), "pools": pools,
            "boxes": [tuple(float(v) for v in b) for t in ind["tiles"] for b in t["core"]],
            "footprints": [shape(parts[pid]["geometry"]) for pid in si["eligible"]],
            "record_ids": [int(parts[pid]["record_id"]) for pid in si["eligible"]],
            "published": {"exact": {n: {d: {b: ot[n][d][b]["recall"] for b in ("0.5", "1.0", "2.0")}
                                        for d in BT.DECODERS} for n in pools}}}


def published_contrast(name: str) -> dict:
    """Primary 2 km D1 - ADAF barrow contrast as published (reading (i))."""
    if name == "verden":
        return rj(VD_RESULT)["grids"]["2km"]["decoders"]
    return {d: v for d, v in rj(OT_RESULT)["bootstrap"]["2km"]["decoders"].items()}


def run_region(reg: dict, ss: dict, idx: np.ndarray) -> dict:
    size, off = BT.GRIDS[BT.PRIMARY_GRID]
    blk = E.build_blocks(reg["boxes"], reg["footprints"], D.rs.REACH_M, size, off)
    if abs(blk.area_km2.sum() - reg["area"]) > 1e-6 or int(blk.n_labels.sum()) != reg["N"]:
        raise SystemExit("blocks do not partition the scored area / references")
    w = E.multinomial_weights(np.random.default_rng(ss[f"geo_{reg['name']}"]), len(blk.area_km2), BT.REPS)
    nj = blk.n_labels.astype(float)
    mk = IG.masks(reg)
    ig_pub = rj(IG_RESULT)["regions"][reg["name"]]["variants"]
    out = {"N": reg["N"], "mask_counts": mk["counts"], "readings": {}, "checks": {}}
    for vname, mask in (("historical", None), ("literal", mk["literal"]), ("precedence", mk["precedence"])):
        area_j = blk.masked_area_km2(mask)
        unr = 0 if mask is None else int(IG.unreachable(reg, mask))
        v = {"scored_km2": float(area_j.sum()), "K": {str(b): int(E.budget(b, area_j.sum())) for b in BT.B},
             "unreachable_entities": unr, "decoders": {}}
        if mask is not None:
            shapely.prepare(mask)
        for dec in BT.DECODERS:
            geo, point, removed, supply, cmax = {}, {}, {}, {}, {}
            ones = np.ones((1, len(area_j)), dtype=np.int64)
            for sysname, decs in reg["pools"].items():
                pool = decs[dec]
                keep = np.ones(len(pool.x), dtype=bool) if mask is None else ~shapely.contains_xy(mask, pool.x, pool.y)
                sub = pool.subset(keep)
                bo = blk.block_of_points(sub.x, sub.y)
                delta = E.increments(sub.edges, sub.n_labels)
                removed[sysname], supply[sysname], cmax[sysname] = int((~keep).sum()), int(keep.sum()), int(delta.sum())
                geo[sysname] = BT.recall_reps(bo, delta, w, area_j, nj)[0]
                point[sysname] = BT.recall_reps(bo, delta, ones, area_j, nj)[0]
            c = DV.contrast(geo, point, idx)
            d1_mean = np.mean([point[n][0] for n in D1], axis=0)
            v["decoders"][dec] = {
                "adaf_recall": BT.pt4(point[ADAF][0]), "d1_mean_recall": BT.pt4(d1_mean),
                "diff_point": c["point"], "diff_joint_ci95": c["ci95"]["joint"],
                "diff_geography_ci95": c["ci95"]["geography_only"], "diff_training_ci95": c["ci95"]["training_only"],
                "positive_runs": c["positive_runs"],
                "per_run_diff_range": {b: [min(c["per_run"][n][b] for n in D1), max(c["per_run"][n][b] for n in D1)]
                                       for b in ("0.5", "1.0", "2.0")},
                "proposals_removed": {"ADAF_barrow": removed[ADAF], "D1_min": min(removed[n] for n in D1),
                                      "D1_max": max(removed[n] for n in D1)},
                "supply": {"ADAF_barrow": supply[ADAF], "D1_min": min(supply[n] for n in D1),
                           "D1_max": max(supply[n] for n in D1)},
                "C": {"ADAF_barrow": cmax[ADAF], "D1_min": min(cmax[n] for n in D1), "D1_max": max(cmax[n] for n in D1)}}
            if unr:
                f = reg["N"] / (reg["N"] - unr)
                v["decoders"][dec]["over_reachable_N"] = {"adaf_recall": BT.pt4(point[ADAF][0] * f),
                                                          "d1_mean_recall": BT.pt4(d1_mean * f),
                                                          "diff_point": BT.pt4((d1_mean - point[ADAF][0]) * f)}
            # self-checks
            if mask is None:
                pub = reg["published"]["exact"]
                ok = all(abs(point[n][0][i] - pub[n][dec][k]) < TOL for n in point
                         for i, k in enumerate(("0.5", "1.0", "2.0")))
                pc = published_contrast(reg["name"])[dec]
                pc_pub = pc["contrast"] if "contrast" in pc else pc
                ok_ci = c["ci95"]["joint"] == pc_pub["ci95"]["joint"] and c["point"] == pc_pub["point"]
                out["checks"][f"historical_{dec}_reproduces_published_recall"] = bool(ok)
                out["checks"][f"historical_{dec}_reproduces_published_joint_ci"] = bool(ok_ci)
            ip = ig_pub[vname]["decoders"][dec]["systems_point"].get(ADAF)
            if ip is not None:
                out["checks"][f"{vname}_{dec}_adaf_matches_IGNORE_V0"] = bool(
                    all(abs(ip[k] - v["decoders"][dec]["adaf_recall"][k]) < TOL for k in ip))
        if abs(v["scored_km2"] - ig_pub[vname]["scored_km2"]) > 1e-9 or unr != ig_pub[vname]["unreachable_entities"]:
            raise SystemExit(f"{reg['name']} {vname}: mask differs from IGNORE_V0.json")
        out["readings"][vname] = v
        print(reg["name"], vname, round(v["scored_km2"], 2), v["K"], unr,
              {d: round(100 * v["decoders"][d]["diff_point"]["0.5"], 2) for d in BT.DECODERS}, flush=True)
    return out


def main() -> None:
    t0 = time.time()
    ss = BT.streams()
    idx = DV.train_draws()
    res = {"schema": "manuscript_d1_ignore_rescore_v1", "reps": BT.REPS, "grid": BT.PRIMARY_GRID,
           "master_seed": BT.MASTER_SEED, "training_draws": "desc_verden.train_draws (shared, frozen)",
           "readings": {"historical": "(i)", "literal": "(ii)", "precedence": "(iii)"},
           "oldenburg_session": str(SESSION.relative_to(ROOT)),
           "regions": {"verden": run_region(load_verden(), ss, idx),
                       "oldenburg": run_region(load_oldenburg(), ss, idx)}}
    res["all_checks_pass"] = all(all(r["checks"].values()) for r in res["regions"].values())
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / "D1_IGNORE_RESCORE.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("all_checks_pass", res["all_checks_pass"], res["seconds"], "s")


if __name__ == "__main__":
    main()
