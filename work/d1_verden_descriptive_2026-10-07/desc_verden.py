"""D1-VERDEN-DESC: descriptive v0 uncertainty and pool terms for the ten D1 BARROW_L runs vs unchanged ADAF barrow on
Verden (GPT-6 Pro Round 4, R4-4 and R4-5; ROUND4_REVIEW.md "Adopted" item 7). DESCRIPTIVE: Stage 1's gate was
point-only and is unchanged (work/stage1_d1_v0_2026-10-07/FREEZE.md).

Inputs: the Stage 1 proposals and manifest (work/stage1_d1_v0_2026-10-07/RESULT.json provenance, sha256 checked).
Engine: the frozen v0 bootstrap (work/gpt6_pro_collab_2026-10-07/v0_uncertainty/engine.py, boot_v0.py), imported
read-only and unchanged:
- blocks: 2 km grid at (0, 0) primary; 4 km and 2 km offset (1000, 1000) sensitivities; reach-merged anchor cells;
- geography: Multinomial(J; 1/J) block weights from the v0 stream geo_verden (identical weights to BOOT_V0.json);
- training: the ten D1 runs drawn with replacement (10 draws) from a new named stream train_d1 =
  SeedSequence(20261007).spawn(5)[4]; ADAF barrow is one fixed public checkpoint. The same train_d1 draws are frozen
  for the Oldenburg transfer check, so a joint two-district summary shares the checkpoint index;
- joint: geography and training together; 5,000 replicates; percentile 95 % intervals, pointwise; score each
  checkpoint first, then average.
Pool terms per run s: C_s = whole-pool maximum matching, G_ranking,s(b) = min(K_b, C_s) - M_s(b), G_pool,s = N - C_s.
v0.1 sensitivity (work/evaluator_v0_1_2026-10-07, frozen 0f3443ea): barrow endpoint identical (no voids); member-
supported complex endpoint, point values per run.
Aggregates only (no ids, no coordinates) go to RESULT.json.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/d1_verden_descriptive_2026-10-07/desc_verden.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V0U = ROOT / "work/gpt6_pro_collab_2026-10-07/v0_uncertainty"
EV = ROOT / "work/evaluator_v0_1_2026-10-07"
sys.path.insert(0, str(V0U))
sys.path.insert(0, str(EV))
import boot_v0 as BT  # noqa: E402  frozen v0 bootstrap, read-only
import engine as E  # noqa: E402
import v0data as D  # noqa: E402
import eval_v01 as V  # noqa: E402  frozen evaluator v0.1, read-only

STAGE1 = ROOT / "work/stage1_d1_v0_2026-10-07/RESULT.json"
BOOT = V0U / "BOOT_V0.json"
IND = ROOT / "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json"
REGISTER = ROOT / "outputs/fresh_region_2026-10-03/labels/denkmale_Verden_20261003T084547Z.geojson"
SUPPLEMENT = ROOT / "outputs/fresh_region_2026-10-03/step0/EDGE_SUPPLEMENT_PRIVATE.json"
SEEDS = tuple(range(2026092805, 2026092815))
D1 = tuple(f"D1F1_BARROW_L_{s}" for s in SEEDS)
ADAF = "ADAF_barrow"
DEPLOYABLE = "D1F1_BARROW_L_2026092805"
TRAIN_STREAM_INDEX = 4  # SeedSequence(MASTER_SEED).spawn(5)[4]; v0 used children 0-3


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def train_d1_stream() -> np.random.SeedSequence:
    return np.random.SeedSequence(BT.MASTER_SEED).spawn(TRAIN_STREAM_INDEX + 1)[TRAIN_STREAM_INDEX]


def train_draws(reps: int = BT.REPS) -> np.ndarray:
    """(reps, 10) run indices drawn with replacement; frozen for Verden and Oldenburg alike."""
    return np.random.default_rng(train_d1_stream()).integers(0, len(D1), size=(reps, len(D1)))


def load() -> dict:
    s1 = rj(STAGE1)
    if sha(IND) != s1["individual_manifest_sha256"]:
        raise SystemExit("individual manifest differs from Stage 1")
    mi = rj(IND)
    si = D.rs.setup(mi)
    objs, prov = {}, {}
    for name in (ADAF,) + D1:
        p = ROOT / s1["provenance"][name]["path"]
        if sha(p) != s1["provenance"][name]["sha256"]:
            raise SystemExit(f"{name}: proposals sha256 differs from Stage 1 provenance")
        objs[name] = rj(p)["objects"]
        prov[name] = s1["provenance"][name]
    parts = {q["id"]: q for q in mi["parts"]}
    return {"s1": s1, "mi": mi, "si": si, "objs": objs, "prov": prov,
            "boxes": [tuple(float(v) for v in b) for t in mi["tiles"] for b in t["core"]],
            "area": unary_union([box(*b) for t in mi["tiles"] for b in t["core"]]).area / 1e6,
            "footprints": [shape(parts[pid]["geometry"]) for pid in si["eligible"]], "N": len(si["eligible"])}


def pools_and_deltas(reg: dict) -> tuple[dict, dict]:
    pools, deltas = {}, {}
    for name, o in reg["objs"].items():
        pools[name] = {"NMS60": D.pool_of(D.rs.ranked_pool(o, reg["si"]), reg["si"]),
                       "RAW": D.pool_of(D.bv.raw_pool(o, reg["si"]), reg["si"])}
        deltas[name] = {d: E.increments(p.edges, p.n_labels) for d, p in pools[name].items()}
    return pools, deltas


def pool_terms(reg: dict, pools: dict, deltas: dict) -> dict:
    """C_s, G_pool,s, and per b: K, M, G_ranking,s(b); checked against scipy and the Stage 1 match counts."""
    out = {}
    for name in pools:
        out[name] = {}
        for dec in BT.DECODERS:
            delta, c = deltas[name][dec], int(deltas[name][dec].sum())
            if c != E.scipy_matching(pools[name][dec].edges, reg["N"]):
                raise SystemExit(f"{name} {dec}: incremental C != scipy C")
            row = {"supply": len(delta), "C": c, "G_pool": reg["N"] - c, "budgets": {}}
            for b in BT.B:
                k = int(E.budget(b, reg["area"]))
                m = int(delta[:k].sum())
                if m != reg["s1"]["systems"][name][dec][str(b)]["matches"]:
                    raise SystemExit(f"{name} {dec} {b}: matches differ from Stage 1")
                row["budgets"][str(b)] = {"K": k, "M": m, "G_ranking": min(k, c) - m, "min_K_C": min(k, c)}
            out[name][dec] = row
    return out


def contrast(geo: dict, point: dict, idx: np.ndarray) -> dict:
    """D1 (mean over drawn runs) - ADAF barrow; geography-only uses the ten observed runs."""
    obs = np.tile(np.arange(len(D1)), (idx.shape[0], 1))
    g = np.stack([geo[n] for n in D1])      # (10, R, 3)
    p = np.stack([point[n][0] for n in D1])  # (10, 3)
    rows = np.arange(idx.shape[0])[:, None]
    diff = {"geography_only": g[obs, rows, :].mean(axis=1) - geo[ADAF],
            "training_only": p[idx].mean(axis=1) - point[ADAF][0],
            "joint": g[idx, rows, :].mean(axis=1) - geo[ADAF]}
    pt = p.mean(axis=0) - point[ADAF][0]
    return {"point": BT.pt4(pt), "ci95": {k: BT.ci4(v) for k, v in diff.items()},
            "per_run": {n: BT.pt4(point[n][0] - point[ADAF][0]) for n in D1},
            "per_run_geography_ci95": {n: BT.ci4(geo[n] - geo[ADAF]) for n in D1},
            "positive_runs": {str(b): int(sum(point[n][0][i] > point[ADAF][0][i] for n in D1))
                              for i, b in enumerate(BT.B)}}


def run_grid(reg: dict, pools: dict, deltas: dict, gname: str, idx: np.ndarray) -> dict:
    size, off = BT.GRIDS[gname]
    blk = E.build_blocks(reg["boxes"], reg["footprints"], D.rs.REACH_M, size, off)
    if abs(blk.area_km2.sum() - reg["area"]) > 1e-6 or int(blk.n_labels.sum()) != reg["N"]:
        raise SystemExit("blocks do not partition the scored area / references")
    w = E.multinomial_weights(np.random.default_rng(BT.streams()["geo_verden"]), len(blk.area_km2), BT.REPS)
    ones = np.ones((1, len(blk.area_km2)), dtype=np.int64)
    nj = blk.n_labels.astype(float)
    out = {"blocks": blk.diag, "decoders": {}}
    for dec in BT.DECODERS:
        geo, point, bad = {}, {}, 0
        for name, pp in pools.items():
            bo = blk.block_of_points(pp[dec].x, pp[dec].y)
            bad += E.verify_edges(blk, bo, (pp[dec].x, pp[dec].y), pp[dec].edges)["edges_outside_one_block"]
            geo[name] = BT.recall_reps(bo, deltas[name][dec], w, blk.area_km2, nj)[0]
            point[name] = BT.recall_reps(bo, deltas[name][dec], ones, blk.area_km2, nj)[0]
        if bad:
            raise SystemExit(f"{gname} {dec}: {bad} edges cross blocks")
        repro = all(point[n][0][i] == reg["s1"]["systems"][n][dec][str(b)]["recall"]
                    for n in point for i, b in enumerate(BT.B))
        out["decoders"][dec] = {"all_ones_equals_stage1": repro, "contrast": contrast(geo, point, idx),
                                "systems_geo_ci95": {n: BT.ci4(geo[n]) for n in geo}}
        if not repro:
            raise SystemExit(f"{gname} {dec}: all-ones weights do not reproduce Stage 1")
    return out


def complex_sensitivity(reg: dict) -> dict:
    """v0.1 member-supported complex endpoint (frozen 0f3443ea), point values; barrow endpoint unchanged (no voids)."""
    st = V.setup(reg["mi"], [])
    recs = [{"id": f["id"], "geometry": f.get("geometry"), "properties": f.get("properties") or {}}
            for f in rj(REGISTER)["features"] + rj(SUPPLEMENT)["records"]]
    cx = V.build_complexes(recs, reg["mi"], st)
    res = {n: V.score_complex(o, st, cx) for n, o in reg["objs"].items()}
    out = {"inventory": cx.diag, "systems": res, "diff_vs_adaf": {}}
    for dec in BT.DECODERS:
        for b in BT.B:
            ref = res[ADAF][dec][str(b)]["recall"]
            d = [res[n][dec][str(b)]["recall"] - ref for n in D1]
            out["diff_vs_adaf"][f"{dec}_{b}"] = {"adaf": ref, "mean_d1": ref + float(np.mean(d)),
                                                 "mean_diff": float(np.mean(d)), "min": min(d), "max": max(d),
                                                 "positive": int(sum(x > 0 for x in d))}
    barrow = {n: V.score_barrow(o, st) for n, o in reg["objs"].items()}
    out["barrow_endpoint_identical_to_stage1"] = all(
        barrow[n][dec][str(b)]["matches"] == reg["s1"]["systems"][n][dec][str(b)]["matches"]
        for n in barrow for dec in BT.DECODERS for b in BT.B)
    return out


def boot_crosscheck(grid: dict) -> dict:
    """ADAF barrow's 2 km geography interval must equal BOOT_V0.json (same blocks, same weights, same pool)."""
    ref = rj(BOOT)["regions"]["verden"]["grids"]["2km"]["decoders"]
    return {dec: grid["decoders"][dec]["systems_geo_ci95"][ADAF] == ref[dec]["systems_geo_ci95"][ADAF]
            for dec in BT.DECODERS}


def main() -> None:
    t0 = time.time()
    reg = load()
    pools, deltas = pools_and_deltas(reg)
    idx = train_draws()
    out = {"schema": "d1_verden_descriptive_v1", "label": "DESCRIPTIVE",
           "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (STAGE1, BOOT, IND, REGISTER, SUPPLEMENT)},
           "engine_sha256": {f: sha(V0U / f) for f in ("engine.py", "boot_v0.py", "v0data.py")},
           "evaluator_v0_1_sha256": sha(EV / "eval_v01.py"),
           "proposals_sha256": {n: p["sha256"] for n, p in reg["prov"].items()},
           "N": reg["N"], "total_scored_km2": reg["area"], "reps": BT.REPS, "master_seed": BT.MASTER_SEED,
           "streams": {"geography": "boot_v0.streams()['geo_verden'] (spawn_key (2,))",
                       "training": f"SeedSequence({BT.MASTER_SEED}).spawn(5)[{TRAIN_STREAM_INDEX}] (spawn_key (4,))"},
           "training_design": "ten D1 runs drawn with replacement (10 draws); ADAF barrow fixed",
           "training_draw_run_counts": np.bincount(idx.ravel(), minlength=len(D1)).tolist(),
           "pool_terms": pool_terms(reg, pools, deltas), "grids": {}}
    for gname in BT.GRIDS:
        out["grids"][gname] = run_grid(reg, pools, deltas, gname, idx)
        print(gname, f"{time.time() - t0:.0f}s", flush=True)
    out["boot_v0_crosscheck_adaf_geo_ci_identical"] = boot_crosscheck(out["grids"]["2km"])
    out["v0_1_sensitivity"] = complex_sensitivity(reg)
    out["deployable_checkpoint"] = DEPLOYABLE
    (HERE / "RESULT.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("done", f"{time.time() - t0:.0f}s", out["boot_v0_crosscheck_adaf_geo_ci_identical"], flush=True)


if __name__ == "__main__":
    main()
