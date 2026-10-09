"""OLDENBURG-TRANSFER-1 scorer (PACKET.md; frozen in FREEZE.json). Runs ONCE, after the GPU step, on the verified
session's Oldenburg proposals of the ten D1 BARROW_L checkpoints. CPU only. Aggregates only go to RESULT.json.

Primary: the exact Stage 1 scorer (work/stage1_d1_v0_2026-10-07/score_v0.py, imported unchanged: v0, raw_pool,
match_count) on the frozen Oldenburg individual manifest (3531675c...), every in-core proposal charged, K = floor(b x
451.117 km2), b = 0.5 / 1 / 2, RAW primary decoder, NMS60 secondary. Comparator: the unchanged ADAF barrow checkpoint's
existing frozen Oldenburg proposals (21421a9c...; same infer_irl.py, inference manifest, pinned DGM1 bytes and
environment as the D1 pass; PACKET.md "Comparator").

Gate (b = 0.5, RAW, unrounded; 1e-12 float guard as Stage 1): mean over the ten runs of (run recall - ADAF barrow
recall) >= +0.030 AND >= 8 of 10 differences > 0. Clear transfer improvement: gate PASS AND the primary 2 km joint
95 % lower bound (RAW, b = 0.5) > 0. Secondary: every run, b = 1, 2, NMS60, supply, C_s / G terms, the frozen v0
bootstrap (2 km primary, 4 km, offset; geography-only, training-only, joint; training draws shared with Verden),
the joint two-district summary, the deployable checkpoint's own result, evaluator v0.1 (sensitivity) and the same-pass
ADAF barrow parity check.

  run:      MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
              /home/tee-/adaf-h2h-venv/bin/python -B work/oldenburg_transfer_2026-10-07/score_oldenburg.py \
              outputs/oldenburg_transfer_2026-10-07/chain_runs/<run>
  fixture:  ... score_oldenburg.py --fixture     (software check: ten synthetic re-rankings of ADAF barrow's
            proposals stand in for the D1 runs; writes DRYRUN_FIXTURE.json; no D1 output is read)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("work/stage1_d1_v0_2026-10-07", "work/gpt6_pro_collab_2026-10-07/v0_uncertainty",
          "work/evaluator_v0_1_2026-10-07", "work/d1_verden_descriptive_2026-10-07"):
    sys.path.insert(0, str(ROOT / p))
import score_v0 as S1  # noqa: E402  exact Stage 1 scorer, read-only
import boot_v0 as BT  # noqa: E402  frozen v0 bootstrap
import engine as E  # noqa: E402
import v0data as D  # noqa: E402
import eval_v01 as V  # noqa: E402  frozen evaluator v0.1 (0f3443ea)
import desc_verden as DV  # noqa: E402  shared training draws, contrast, Verden pools

OLD = ROOT / "outputs/adapt_nds_2026-10-05/confirm/run"
IND = OLD / "INDIVIDUAL_MANIFEST_PRIVATE.json"
INF = OLD / "INFER_MANIFEST_PRIVATE.json"
ADAF_PROPOSALS = ROOT / "outputs/adapt_nds_2026-10-05/score/confirm/individual/PROPOSALS_ADAF_barrow_PRIVATE.json"
REGISTER = ROOT / "outputs/adapt_nds_2026-10-05/register/denkmale_Oldenburg_20261005T010102Z.geojson"
SUPPLEMENT = ROOT / "outputs/adapt_nds_2026-10-05/register/supplement_Oldenburg_20261005T010102Z.json"
D1_SESSION = ROOT / "outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/SESSION_VERIFIED.json"
PIN = {"individual": "3531675cf1323801f1674ee0448c9ff3099da880ed348e1807bc6b3037b92863",
       "inference": "8fce717eea156a9a00ec1bcdea4a936b4b89454c166b1bf4b878f702dc31f411",
       "adaf_proposals": "21421a9c7e99cf337c64467349878e7062571768147a459164b6ddd92b3afef6",
       "adaf_checkpoint": "51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216"}
GATE = {"b": "0.5", "decoder": "RAW", "mean_diff_min": 0.030, "positive_min": 8, "float_guard": 1e-12}
ADAF, D1, DEPLOYABLE = DV.ADAF, DV.D1, DV.DEPLOYABLE


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def preconditions(fixture: bool) -> tuple[dict, dict]:
    if rj(HERE / "LEGITIMACY.json")["legitimacy"] != "PASS":
        raise SystemExit("legitimacy check is not PASS: nothing is scored")
    if fixture and not (HERE / "FREEZE.json").exists():
        frz = {"files_sha256": {}, "inputs_sha256": {}}  # pre-freeze software check only
    else:
        frz = rj(HERE / "FREEZE.json")
    bad = [f for f, h in frz["files_sha256"].items() if sha(HERE / f) != h]
    bad += [f for f, h in frz["inputs_sha256"].items() if sha(ROOT / f) != h]
    if bad:
        raise SystemExit(f"frozen files changed: {bad}")
    ind, inf = rj(IND), rj(INF)
    if sha(IND) != PIN["individual"] or sha(INF) != PIN["inference"] or S1.tile_keys(ind) != S1.tile_keys(inf):
        raise SystemExit("Oldenburg manifests differ from the frozen pins")
    return ind, frz


def load_d1(session: Path, tiles: set, bad: list) -> dict:
    sv = rj(session / "SESSION_VERIFIED.json")
    if sv.get("status") != "PASS":
        bad.append("session not verified PASS")
    ckpt = {n: h for n, h in rj(D1_SESSION)["trained"].items() if n in D1}
    objs = {}
    for name in D1:
        p = session / "v1/results/confirm" / f"PROPOSALS_{name}_PRIVATE.json"
        side = p.with_name(p.name + ".sha256")
        if not side.exists() or side.read_text().strip() != sha(p):
            bad.append(f"{name}: proposals sha256 differs from its sidecar")
        pin = (sv.get("proposals") or {}).get(f"confirm/{p.name}")
        if pin is not None and pin != sha(p):
            bad.append(f"{name}: proposals sha256 differs from the session pin")
        objs[name] = S1.load_checked(p, name, None, ckpt[name], PIN["inference"], tiles, bad)
    return objs


def fixture_runs(adaf: dict) -> dict:
    """Ten synthetic re-rankings of ADAF barrow's proposals (score x (1 + 0.5 u), u ~ U(0, 1), seeded per run)."""
    out = {}
    for i, name in enumerate(D1):
        rng = np.random.default_rng(1000 + i)
        out[name] = {t: [dict(o, score=o["score"] * (1 + 0.5 * rng.random())) for o in rows]
                     for t, rows in sorted(adaf.items())}
    return out


def gate(res: dict) -> dict:
    ref = res[ADAF][GATE["decoder"]][GATE["b"]]
    diffs = {n: res[n][GATE["decoder"]][GATE["b"]] - ref for n in D1}
    mean = float(np.mean(list(diffs.values())))
    pos = sum(d > 0 for d in diffs.values())
    ok = mean >= GATE["mean_diff_min"] - GATE["float_guard"] and pos >= GATE["positive_min"]
    return {**GATE, "adaf_barrow_recall": ref, "per_run_diff": diffs, "mean_diff": mean, "positive_diffs": pos,
            "verdict": "PASS" if ok else "FAIL"}


def reading(g: dict, joint_lb: float) -> dict:
    """Pro's interpretation sentences (ROUND4_REPLY.md:213-222), fixed precedence."""
    m, pos = g["mean_diff"], g["positive_diffs"]
    if g["verdict"] == "PASS":
        main = "The prespecified transfer gate passed on Oldenburg."
        unc = ("Clear transfer improvement: the practical gate passed and the primary joint 95 % lower bound is "
               "positive." if joint_lb > 0 else "Point pass with interval spanning zero: the practical gate passed; "
               "the uncertainty criterion (positive primary joint 95 % lower bound) was not met.")
        return {"main": main, "uncertainty": unc, "clear_transfer_improvement": joint_lb > 0}
    if m <= 0:
        main = "The Verden benefit did not extend to Oldenburg under this comparison."
    elif m >= GATE["mean_diff_min"] - GATE["float_guard"]:
        main = "The transfer result did not meet the replication criterion."
    else:
        main = "A smaller positive transfer effect was estimated; the practical gate was not met."
        if pos < GATE["positive_min"]:
            main += " The replication criterion was also not met."
    return {"main": main, "uncertainty": None, "clear_transfer_improvement": False}


def region_arrays(boxes: list, footprints: list, area: float, n: int, pools: dict, deltas: dict, geo_stream: str,
                  gname: str) -> dict:
    size, off = BT.GRIDS[gname]
    blk = E.build_blocks(boxes, footprints, D.rs.REACH_M, size, off)
    if abs(blk.area_km2.sum() - area) > 1e-6 or int(blk.n_labels.sum()) != n:
        raise SystemExit("blocks do not partition the scored area / references")
    w = E.multinomial_weights(np.random.default_rng(BT.streams()[geo_stream]), len(blk.area_km2), BT.REPS)
    ones = np.ones((1, len(blk.area_km2)), dtype=np.int64)
    geo, point = {}, {}
    for dec in BT.DECODERS:
        geo[dec], point[dec] = {}, {}
        for name, pp in pools.items():
            bo = blk.block_of_points(pp[dec].x, pp[dec].y)
            if E.verify_edges(blk, bo, (pp[dec].x, pp[dec].y), pp[dec].edges)["edges_outside_one_block"]:
                raise SystemExit(f"{gname} {dec} {name}: edges cross blocks")
            geo[dec][name] = BT.recall_reps(bo, deltas[name][dec], w, blk.area_km2, blk.n_labels.astype(float))[0]
            point[dec][name] = BT.recall_reps(bo, deltas[name][dec], ones, blk.area_km2,
                                              blk.n_labels.astype(float))[0]
    return {"geo": geo, "point": point, "blocks": blk.diag}


def pools_of(objs: dict, si: dict) -> tuple[dict, dict]:
    pools = {n: {"NMS60": D.pool_of(D.rs.ranked_pool(o, si), si), "RAW": D.pool_of(D.bv.raw_pool(o, si), si)}
             for n, o in objs.items()}
    deltas = {n: {d: E.increments(p.edges, p.n_labels) for d, p in pp.items()} for n, pp in pools.items()}
    return pools, deltas


def pool_terms(deltas: dict, n: int, area: float) -> dict:
    out = {}
    for name, dd in deltas.items():
        out[name] = {}
        for dec, delta in dd.items():
            c = int(delta.sum())
            row = {"supply": len(delta), "C": c, "G_pool": n - c, "budgets": {}}
            for b in BT.B:
                k = int(E.budget(b, area))
                m = int(delta[:k].sum())
                row["budgets"][str(b)] = {"K": k, "M": m, "G_ranking": min(k, c) - m}
            out[name][dec] = row
    return out


def joint_two_district(old: dict, ver: dict, idx: np.ndarray) -> dict:
    """Equal-weight mean of the two districts' D1 - ADAF barrow differences; the checkpoint index is shared across
    regions in the training resampling (same idx), geography is resampled separately per region (2 km grid)."""
    out = {}
    rows = np.arange(idx.shape[0])[:, None]
    obs = np.tile(np.arange(len(D1)), (idx.shape[0], 1))
    for dec in BT.DECODERS:
        parts = {}
        for tag, r in (("oldenburg", old), ("verden", ver)):
            g = np.stack([r["geo"][dec][n] for n in D1])
            p = np.stack([r["point"][dec][n][0] for n in D1])
            parts[tag] = {"geography_only": g[obs, rows, :].mean(axis=1) - r["geo"][dec][ADAF],
                          "training_only": p[idx].mean(axis=1) - r["point"][dec][ADAF][0],
                          "joint": g[idx, rows, :].mean(axis=1) - r["geo"][dec][ADAF],
                          "point": p.mean(axis=0) - r["point"][dec][ADAF][0]}
        comb = {k: 0.5 * (parts["oldenburg"][k] + parts["verden"][k]) for k in parts["verden"]}
        out[dec] = {"point": BT.pt4(comb.pop("point")), "ci95": {k: BT.ci4(v) for k, v in comb.items()}}
    return out


def parity(session: Path | None, frozen: dict, si: dict, area: float) -> dict:
    """Same-pass ADAF barrow re-inference vs the frozen comparator (technical; never replaces the primary)."""
    p = None if session is None else session / "v1/results/confirm/PROPOSALS_ADAF_barrow_PRIVATE.json"
    if p is None or not p.exists():
        return {"status": "NOT_AVAILABLE"}
    new = rj(p)["objects"]
    pos = all(len(new.get(t, [])) == len(rows) and all(
        abs(a["easting"] - b["easting"]) < 1e-6 and abs(a["northing"] - b["northing"]) < 1e-6
        and abs(a["score"] - b["score"]) <= 2e-6 for a, b in zip(new[t], rows)) for t, rows in frozen.items())
    v_new, v_old = S1.v0(new, si, area), S1.v0(frozen, si, area)
    same = all(v_new[d][str(b)]["matches"] == v_old[d][str(b)]["matches"] for d in BT.DECODERS for b in BT.B)
    return {"status": "PASS" if pos and same else "DEVIATION", "proposals_within_eq1_tolerance": pos,
            "v0_match_counts_identical": same, "sha256": sha(p)}


def v01_sensitivity(ind: dict, objs: dict) -> dict:
    st = V.setup(ind, [])
    recs = [{"id": f["id"], "geometry": f.get("geometry"), "properties": f.get("properties") or {}}
            for f in rj(REGISTER)["features"] + rj(SUPPLEMENT)["records"]]
    cx = V.build_complexes(recs, ind, st)
    out = {"barrow": {}, "complex": {}, "complex_inventory": cx.diag}
    for n, o in objs.items():
        out["barrow"][n] = {d: {str(b): r[str(b)]["recall"] for b in BT.B} for d, r in V.score_barrow(o, st).items()}
        out["complex"][n] = V.score_complex(o, st, cx)
    return out


def main() -> None:
    fixture = "--fixture" in sys.argv
    session = None if fixture else (ROOT / sys.argv[1]).resolve()
    ind, frz = preconditions(fixture)
    si, tiles = S1.rs.setup(ind), {t["id"] for t in ind["tiles"]}
    area = unary_union([box(*b) for t in ind["tiles"] for b in t["core"]]).area / 1e6
    bad: list[str] = []
    adaf = S1.load_checked(ADAF_PROPOSALS, ADAF, PIN["adaf_proposals"], PIN["adaf_checkpoint"], PIN["inference"],
                           tiles, bad)
    objs = {ADAF: adaf} | (fixture_runs(adaf) if fixture else load_d1(session, tiles, bad))
    if bad:
        raise SystemExit(f"refused: {bad}")
    exact = {n: S1.v0(o, si, area) for n, o in objs.items()}
    rec_tab = {n: {d: {str(b): r[d][str(b)]["recall"] for b in BT.B} for d in BT.DECODERS} for n, r in exact.items()}
    g = gate(rec_tab)
    idx = DV.train_draws()
    parts = {q["id"]: q for q in ind["parts"]}
    fps = [shape(parts[pid]["geometry"]) for pid in si["eligible"]]
    boxes = [tuple(float(v) for v in b) for t in ind["tiles"] for b in t["core"]]
    pools, deltas = pools_of(objs, si)
    boot = {}
    for gname in BT.GRIDS:
        arr = region_arrays(boxes, fps, area, len(si["eligible"]), pools, deltas, "geo_oldenburg", gname)
        boot[gname] = {"blocks": arr["blocks"], "decoders": {
            d: DV.contrast(arr["geo"][d], arr["point"][d], idx) for d in BT.DECODERS}}
        if gname == "2km":
            old2 = arr
    vr = DV.load()
    vpools, vdeltas = DV.pools_and_deltas(vr)
    ver2 = region_arrays(vr["boxes"], vr["footprints"], vr["area"], vr["N"], vpools, vdeltas, "geo_verden", "2km")
    joint_lb = boot["2km"]["decoders"]["RAW"]["ci95"]["joint"]["0.5"][0]
    out = {"schema": "oldenburg_transfer_result_v1", "mode": "FIXTURE" if fixture else "RUN",
           "name": "a prespecified transfer check on a previously inspected, training-disjoint district that was "
                   "not used to select this recipe",
           "freeze_sha256": sha(HERE / "FREEZE.json") if (HERE / "FREEZE.json").exists() else None,
           "N": len(si["eligible"]), "total_scored_km2": area,
           "K": {str(b): math.floor(b * area + 1e-9) for b in BT.B}, "exact_scorer": exact, "gate": g,
           "reading": reading(g, joint_lb), "primary_joint_95_lower_bound": joint_lb,
           "deployable_checkpoint": {"name": DEPLOYABLE, "diff": boot["2km"]["decoders"]["RAW"]["per_run"][DEPLOYABLE],
                                     "geography_ci95": boot["2km"]["decoders"]["RAW"]["per_run_geography_ci95"][
                                         DEPLOYABLE]},
           "pool_terms": pool_terms(deltas, len(si["eligible"]), area), "bootstrap": boot,
           "joint_two_district": joint_two_district(old2, ver2, idx), "v0_1_sensitivity": v01_sensitivity(ind, objs),
           "adaf_parity": parity(session, adaf, si, area),
           "training_draw_run_counts": np.bincount(idx.ravel(), minlength=len(D1)).tolist()}
    name = "DRYRUN_FIXTURE.json" if fixture else "RESULT.json"
    (HERE / name).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"gate": g, "reading": out["reading"], "joint_lb": joint_lb}, indent=1))


if __name__ == "__main__":
    main()
