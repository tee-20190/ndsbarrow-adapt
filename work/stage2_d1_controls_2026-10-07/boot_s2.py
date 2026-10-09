"""STAGE2-D1-CONTROLS paired v0 block bootstrap, frozen with the packet before any control exists (ROUND4_REPLY.md
R4-2 "Freeze the bootstrap before any control is scored"; ROUND4_REVIEW.md "Adopted" item 4).

Read-only reuse: the uncertainty engine work/gpt6_pro_collab_2026-10-07/v0_uncertainty/engine.py (build_blocks,
increments, m_star, multinomial_weights, budget; its tests) and the exact Stage 1 scorer
work/stage1_d1_v0_2026-10-07/score_v0.py (raw_pool, region_score.ranked_pool, the 15 m footprint tree). Nothing in
either directory is edited.

Frozen analysis configuration (ANALYSIS below):
- Scorer: score_v0 (benchmark v0, legacy IGNORE). All-ones block weights must reproduce score_v0.v0 recall of every
  system exactly, else the bootstrap refuses.
- Pairs: the ten prespecified (D1F1_BARROW_L_s, S2CTL_BARROW_s), s = 2026092805 ... 14, in that order. Exactly ten
  valid pairs are required; with fewer the result is NOT_EVALUATED (no smaller denominator).
- Geography: Verden scored area cut into 2 km cells (origin 0, 0), merged by the R3-2 anchor/15 m-reach rule
  (engine.build_blocks); J blocks; replicate weights (w_1..w_J) ~ Multinomial(J; 1/J ...), shared by every system and
  both arms within a replicate; K*_b = floor(b x A*) on the replicate area; recall = M*/N*.
- Primary (design "paired"): each replicate draws ten pair indices with replacement; the statistic is the mean over the
  drawn pairs of recall(D1_i) - recall(CTL_i), both evaluated on the replicate's shared geography. RAW decoder,
  b = 0.5, 5,000 replicates, numpy SeedSequence(MASTER_SEED) streams, percentile 2.5 / 97.5 (numpy linear
  interpolation). Clear advantage iff the unrounded lower bound > 0 (strict).
- Sensitivity, prespecified and never substituted for the primary: design "independent" (ten D1 runs and ten controls
  drawn independently, shared geography), and the 4 km and 2 km offset (1,000 m, 1,000 m) grids under both designs.
  Every other decoder / budget is descriptive.
- Replicates with N* = 0 (no reference in the drawn blocks) are undefined; they are dropped and counted (expected 0).

  python -B boot_s2.py pseudo OUT.json [--reps R]   # end-to-end proof on existing data: the ten D1 runs against
                                                    # unchanged ADAF barrow as a pseudo-control (no control exists)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UNC = ROOT / "work/gpt6_pro_collab_2026-10-07/v0_uncertainty"
STAGE1 = ROOT / "work/stage1_d1_v0_2026-10-07"
for p in (str(UNC), str(STAGE1)):
    if p not in sys.path:
        sys.path.insert(0, p)
import engine as E  # noqa: E402  read-only (v0_uncertainty, tested there)
import score_v0 as V0  # noqa: E402  read-only (the exact Stage 1 scorer)

MASTER_SEED = 2026100702
REPS = 5000
CHUNK = 500
B = (0.5, 1.0, 2.0)
DECODERS = ("RAW", "NMS60")
GRIDS = {"2km": (2000.0, (0.0, 0.0)), "4km": (4000.0, (0.0, 0.0)), "2km_offset1000": (2000.0, (1000.0, 1000.0))}
DESIGNS = ("paired", "independent")
STREAMS = ("geo", "train_paired", "train_independent_d1", "train_independent_ctl")
N_PAIRS = 10
PRIMARY = {"grid": "2km", "design": "paired", "decoder": "RAW", "b": "0.5"}
ANALYSIS = {"scorer": "work/stage1_d1_v0_2026-10-07/score_v0.py (benchmark v0, legacy IGNORE)",
            "engine": "work/gpt6_pro_collab_2026-10-07/v0_uncertainty/engine.py",
            "master_seed": MASTER_SEED, "streams": list(STREAMS), "reps": REPS, "budgets_per_km2": list(B),
            "grids": {k: {"size_m": v[0], "offset_m": list(v[1])} for k, v in GRIDS.items()},
            "designs": {"paired": "ten prespecified (D1_s, CTL_s) pairs resampled with replacement (one index draws "
                                  "both arms); shared geography",
                        "independent": "ten D1 runs and ten controls resampled independently; shared geography"},
            "primary": PRIMARY, "interval": "percentile 2.5 / 97.5, numpy linear interpolation, unrounded",
            "clear_rule": "primary lower95 > 0 (strict, unrounded)",
            "n_star_zero": "replicate undefined: dropped and counted (expected 0)",
            "incomplete": "fewer than ten valid pairs -> NOT_EVALUATED; no smaller denominator"}


def streams() -> dict[str, np.random.SeedSequence]:
    return dict(zip(STREAMS, np.random.SeedSequence(MASTER_SEED).spawn(len(STREAMS))))


# ---------------------------------------------------------------- pools (score_v0's exact orders and edges)
@dataclass
class Pool:
    x: np.ndarray
    y: np.ndarray
    edges: list[np.ndarray]
    n_labels: int


def pool_of(rows: list[dict], s: dict) -> Pool:
    x = np.array([r["easting"] for r in rows], dtype=float)
    y = np.array([r["northing"] for r in rows], dtype=float)
    edges = [np.sort(np.asarray(s["tree"].query(p, predicate="intersects"), dtype=np.int64))
             for p in shapely.points(x, y)]
    return Pool(x, y, edges, len(s["eligible"]))


def pools(objs: dict, s: dict) -> dict:
    return {m: {"RAW": pool_of(V0.raw_pool(o, s), s), "NMS60": pool_of(V0.rs.ranked_pool(o, s), s)}
            for m, o in objs.items()}


def region(ind: dict, s: dict) -> dict:
    boxes = [tuple(float(v) for v in b) for t in ind["tiles"] for b in t["core"]]
    area = unary_union([box(*b) for b in boxes]).area / 1e6                      # = score_v0 / bench_v0 area
    parts = {q["id"]: q for q in ind["parts"]}
    return {"boxes": boxes, "area_km2": area, "N": len(s["eligible"]),
            "footprints": [shape(parts[pid]["geometry"]) for pid in s["eligible"]]}


# ---------------------------------------------------------------- replicate recall
def recall_reps(block_of: np.ndarray, delta: np.ndarray, w: np.ndarray, area_j: np.ndarray,
                n_j: np.ndarray) -> np.ndarray:
    """(R, 3) recall M*/N* for block weights w (R, J); NaN where N* = 0."""
    a_star, n_star = w @ area_j, w @ n_j
    rec = np.empty((w.shape[0], len(B)))
    for s0 in range(0, w.shape[0], CHUNK):
        ws = w[s0:s0 + CHUNK]
        for i, b in enumerate(B):
            m, _ = E.m_star(block_of, delta, ws, E.budget(b, a_star[s0:s0 + CHUNK]))
            with np.errstate(divide="ignore", invalid="ignore"):
                rec[s0:s0 + CHUNK, i] = np.where(n_star[s0:s0 + CHUNK] > 0, m / n_star[s0:s0 + CHUNK], np.nan)
    return rec


def ci(x: np.ndarray) -> dict:
    ok = np.isfinite(x)
    lo, hi = np.percentile(x[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
    return {"lower95": float(lo), "upper95": float(hi), "dropped_undefined": int((~ok).sum()),
            "replicates": int(ok.sum())}


# ---------------------------------------------------------------- the paired statistic (pure numpy; tested)
def check_pairs(pairs: list[tuple[str, str]], available: set) -> None:
    if len(pairs) != N_PAIRS or len({p[0] for p in pairs}) != N_PAIRS:
        raise ValueError(f"the bootstrap needs exactly {N_PAIRS} distinct prespecified pairs (got {len(pairs)})")
    missing = [n for p in pairs for n in p if n not in available]
    if missing:
        raise ValueError(f"pairs reference systems without a valid score: {missing[:3]}")


def train_draws(ss: dict, reps: int) -> dict[str, np.ndarray]:
    return {"paired": np.random.default_rng(ss["train_paired"]).integers(0, N_PAIRS, size=(reps, N_PAIRS)),
            "independent_d1": np.random.default_rng(ss["train_independent_d1"]).integers(0, N_PAIRS,
                                                                                        size=(reps, N_PAIRS)),
            "independent_ctl": np.random.default_rng(ss["train_independent_ctl"]).integers(0, N_PAIRS,
                                                                                          size=(reps, N_PAIRS))}


def paired_stats(geo: dict[str, np.ndarray], pairs: list[tuple[str, str]], draws: dict[str, np.ndarray]) -> dict:
    """geo[name]: (R, 3) recall on the replicate geography. Returns (R, 3) differences per design."""
    d1 = np.stack([geo[a] for a, _ in pairs])                                   # (10, R, 3)
    ct = np.stack([geo[c] for _, c in pairs])
    rows = np.arange(d1.shape[1])[:, None]
    ip, idd, idc = draws["paired"], draws["independent_d1"], draws["independent_ctl"]
    return {"paired": (d1[ip, rows, :] - ct[ip, rows, :]).mean(axis=1),
            "independent": d1[idd, rows, :].mean(axis=1) - ct[idc, rows, :].mean(axis=1)}


# ---------------------------------------------------------------- whole analysis
def run(objs: dict, s: dict, ind: dict, pairs: list[tuple[str, str]], v0res: dict, reps: int = REPS,
        grids: tuple[str, ...] = tuple(GRIDS)) -> dict:
    """The frozen analysis. objs: proposals of every paired system; v0res: score_v0.v0 results (point values)."""
    t0 = time.time()
    check_pairs(pairs, set(objs) & set(v0res))
    names = sorted({n for p in pairs for n in p})
    reg = region(ind, s)
    pl = pools({n: objs[n] for n in names}, s)
    deltas = {n: {d: E.increments(p.edges, p.n_labels) for d, p in pl[n].items()} for n in names}
    ss = streams()
    draws = train_draws(ss, reps)
    out = {"analysis": ANALYSIS | {"reps": reps}, "pairs": [list(p) for p in pairs], "N": reg["N"],
           "total_scored_km2": reg["area_km2"], "grids": {}}
    for gname in grids:
        size, off = GRIDS[gname]
        blk = E.build_blocks(reg["boxes"], reg["footprints"], V0.rs.REACH_M, size, off)
        if abs(blk.area_km2.sum() - reg["area_km2"]) > 1e-6 or int(blk.n_labels.sum()) != reg["N"]:
            raise SystemExit("blocks do not partition the scored area / references")
        ones = np.ones((1, len(blk.area_km2)), dtype=np.int64)
        w = E.multinomial_weights(np.random.default_rng(ss["geo"]), len(blk.area_km2), reps)
        g = {"blocks": blk.diag, "decoders": {}}
        for dec in DECODERS:
            geo, bad = {}, []
            for n in names:
                bo = blk.block_of_points(pl[n][dec].x, pl[n][dec].y)
                if E.verify_edges(blk, bo, (pl[n][dec].x, pl[n][dec].y), pl[n][dec].edges)["edges_outside_one_block"]:
                    raise SystemExit(f"{n} {dec}: an edge crosses blocks")
                one = recall_reps(bo, deltas[n][dec], ones, blk.area_km2, blk.n_labels.astype(float))[0]
                want = np.array([v0res[n][dec][str(b)]["recall"] for b in B])
                if not np.array_equal(one, want):
                    bad.append([n, dec, one.tolist(), want.tolist()])
                geo[n] = recall_reps(bo, deltas[n][dec], w, blk.area_km2, blk.n_labels.astype(float))
            if bad:
                raise SystemExit(f"all-ones weights do not reproduce score_v0 exactly: {bad[:2]}")
            diffs = paired_stats(geo, pairs, draws)
            pt = np.mean([[v0res[a][dec][str(b)]["recall"] - v0res[c][dec][str(b)]["recall"] for b in B]
                          for a, c in pairs], axis=0)
            g["decoders"][dec] = {str(b): {"point": float(pt[i]), **{des: ci(diffs[des][:, i]) for des in DESIGNS}}
                                  for i, b in enumerate(B)}
        out["grids"][gname] = g
        print(json.dumps({"grid": gname, "J": blk.diag["J"], "s": round(time.time() - t0, 1)}), flush=True)
    if "2km" in out["grids"]:
        p = out["grids"]["2km"]["decoders"]["RAW"]["0.5"]
        out["primary"] = {**PRIMARY, "point": p["point"], **p["paired"],
                          "clear": bool(math.isfinite(p["paired"]["lower95"]) and p["paired"]["lower95"] > 0)}
        out["sensitivity_raw_0.5"] = {f"{gn}/{des}": out["grids"][gn]["decoders"]["RAW"]["0.5"][des]
                                      for gn in out["grids"] for des in DESIGNS}
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- end-to-end proof on existing data
def pseudo(reps: int = REPS, grids: tuple[str, ...] = tuple(GRIDS)) -> dict:
    """The ten D1 runs against unchanged ADAF barrow as a pseudo-control (each pair (D1_s, ADAF_barrow)), scored with
    score_v0 exactly as Stage 1. The point estimate must equal Stage 1's gate mean_diff."""
    ind, inf = V0.rj(V0.IND), V0.rj(V0.INF)
    if V0.tile_keys(ind) != V0.tile_keys(inf):
        raise SystemExit("individual and inference manifests differ in tiles")
    s = V0.rs.setup(ind)
    area = unary_union([box(*b) for t in ind["tiles"] for b in t["core"]]).area / 1e6
    bad: list[str] = []
    objs, _ = V0.systems(ind, inf, bad)
    if bad:
        raise SystemExit(f"refused: {bad}")
    res = {m: V0.v0(o, s, area) for m, o in objs.items()}
    pairs = [(f"D1F1_BARROW_L_{x}", "ADAF_barrow") for x in V0.SEEDS]
    return {"kind": "pseudo_control_ADAF_barrow", "result": run(objs, s, ind, pairs, res, reps, grids),
            "stage1_gate_mean_diff": V0.rj(STAGE1 / "RESULT.json")["gate"]["mean_diff"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pseudo",))
    ap.add_argument("out", type=Path)
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    r = pseudo(a.reps)
    rec = {"schema": "s2_boot_pseudo_v1", "note": "end-to-end proof only: ADAF barrow stands in for every control; "
           "no control exists", **r, "code_sha256": {"boot_s2.py": V0.sha(Path(__file__)),
                                                     "engine.py": V0.sha(Path(E.__file__)),
                                                     "score_v0.py": V0.sha(Path(V0.__file__))}}
    a.out.write_bytes((json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({"primary": r["result"]["primary"], "stage1": r["stage1_gate_mean_diff"]}, indent=1))


if __name__ == "__main__":
    main()
