"""Tasks 1-3 and 6: v0 block bootstrap (R3-2), budget/oracle table (R3-1) and adaptation-vs-start (R3-8 item 3).

CPU only; reads frozen manifests and saved proposals; writes BOOT_V0.json beside this file (aggregates only: no ids,
no coordinates).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/gpt6_pro_collab_2026-10-07/v0_uncertainty/boot_v0.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine as E  # noqa: E402
import v0data as D  # noqa: E402

MASTER_SEED = 20261007
REPS = 5000
CHUNK = 500
B = (0.5, 1.0, 2.0)
DECODERS = ("NMS60", "RAW")
GRIDS = {"2km": (2000.0, (0.0, 0.0)), "4km": (4000.0, (0.0, 0.0)), "2km_offset1000": (2000.0, (1000.0, 1000.0))}
PRIMARY_GRID = "2km"
SEEDS3 = ("2026092802", "2026092803", "2026092804")
STREAM_NAMES = ("geo_oldenburg", "train_oldenburg", "geo_verden", "train_verden")


def streams() -> dict[str, np.random.SeedSequence]:
    """Separate reproducible streams (ROUND3_REPLY.md:253): SeedSequence(MASTER_SEED).spawn, fixed name order."""
    return dict(zip(STREAM_NAMES, np.random.SeedSequence(MASTER_SEED).spawn(len(STREAM_NAMES))))


# ---------------------------------------------------------------- training-run designs
def design(region: str) -> dict:
    """Oldenburg: A_s and C_s both start from J_s's final weights with torch.manual_seed(s) and identical non-Lower-
    Saxony steps (adapt_nds PACKET.md:42-48; nds_train.py:1-13,176-188): (J_s, A_s, C_s) is one family, resampled
    together. ADAF and CONT are fixed public/single checkpoints.
    Verden: the registered S* is J_2026092803, selected by FRESH-1's rule (highest region-macro DEV primary among the
    five JOINT-1 confirmation seeds, ties to the lower seed; FRESH1_PREPIXEL.md:75, FREEZE.json sstar). The
    selection-aware training draw resamples the five seeds and re-applies that rule; DEPLOY = INTERLEAVE(ADAF AO, S*)."""
    if region == "oldenburg":
        mem = {"A": [f"A_NDS_{s}" for s in SEEDS3], "C": [f"C_NDS_{s}" for s in SEEDS3],
               "J": [f"J_{s}" for s in SEEDS3]}
        mem |= {m: [m] for m in ("ADAF_barrow", "ADAF_AO", "CONT")}
        return {"members": mem, "obs_idx": np.array([[0, 1, 2]]),
                "draw": lambda rng, r: rng.integers(0, 3, size=(r, 3)),
                "contrasts": [("A", "C"), ("A", "ADAF_barrow"), ("A", "J"), ("C", "J")],
                "rule": "families (J_s, A_s, C_s) resampled together, 3 draws with replacement; ADAF, CONT fixed"}
    macro = D.sstar_macro()
    seeds = [f"J_{s}" for s in D.J_SEEDS_VERDEN]
    mvals = np.array([macro[s] for s in seeds])

    def draw(rng: np.random.Generator, r: int) -> np.ndarray:
        idx = rng.integers(0, len(seeds), size=(r, len(seeds)))
        # argmax of macro over the drawn seeds; ties to the lower seed (np.argmax on a seed-ordered key)
        key = mvals[idx] - idx * 1e-15
        return idx[np.arange(r), np.argmax(key, axis=1)][:, None]

    mem = {"DEPLOY": [f"DEPLOY_{s}" for s in seeds], "S*": seeds}
    mem |= {m: [m] for m in ("ADAF_barrow", "ADAF_AO", "CONT")}
    return {"members": mem, "obs_idx": np.array([[seeds.index(D.SSTAR)]]), "draw": draw,
            "contrasts": [("DEPLOY", "ADAF_AO"), ("DEPLOY", "ADAF_barrow")],
            "rule": "selection-aware: 5 JOINT-1 confirmation seeds drawn with replacement, S* = highest region-macro "
                    "DEV primary among those drawn (FRESH-1 rule); ADAF, CONT fixed"}


def value(dsg: dict, system: str, scores: dict, idx: np.ndarray) -> np.ndarray:
    """(R, 3) per-budget recall of a system: mean over its drawn training runs (score first, then average)."""
    names = dsg["members"][system]
    r = idx.shape[0]
    if len(names) == 1:
        return np.broadcast_to(scores[names[0]], (max(r, scores[names[0]].shape[0]), 3))
    s = np.stack([scores[n] for n in names])  # (M, R', 3)
    rr = np.arange(r)[:, None] if s.shape[1] == r else np.zeros((r, 1), dtype=int)
    return s[idx, rr, :].mean(axis=1)


def ci4(x: np.ndarray) -> dict:
    """Percentile 95 % intervals per budget and for the three-budget mean (pointwise)."""
    out = {str(b): E.percentile_ci(x[:, i]) for i, b in enumerate(B)}
    out["mean3"] = E.percentile_ci(x.mean(axis=1))
    return out


def pt4(x: np.ndarray) -> dict:
    out = {str(b): float(x[i]) for i, b in enumerate(B)}
    out["mean3"] = float(np.mean(x))
    return out


# ---------------------------------------------------------------- scoring
def recall_reps(block_of: np.ndarray, delta: np.ndarray, w: np.ndarray, area_j: np.ndarray,
                n_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, 3) recall M*/N* and (R, 3) underfilled flags for weights w (R, J)."""
    a_star, n_star = w @ area_j, w @ n_j
    rec = np.empty((w.shape[0], len(B)))
    under = np.zeros((w.shape[0], len(B)), dtype=bool)
    for s0 in range(0, w.shape[0], CHUNK):
        ws = w[s0:s0 + CHUNK]
        for i, b in enumerate(B):
            k = E.budget(b, a_star[s0:s0 + CHUNK])
            m, used = E.m_star(block_of, delta, ws, k)
            with np.errstate(divide="ignore", invalid="ignore"):
                rec[s0:s0 + CHUNK, i] = m / n_star[s0:s0 + CHUNK]
            under[s0:s0 + CHUNK, i] = used < k
    return rec, under


def oracle_table(reg: dict, deltas: dict) -> dict:
    """R3-1 (ROUND3_REPLY.md:19-30, 81-87): N, K, M, R, c, matches per 100 slots, supply, C and R_oracle."""
    n, area = reg["N"], reg["area_km2"]
    out = {}
    for sysname, decs in reg["pools"].items():
        out[sysname] = {}
        for dec, pool in decs.items():
            delta = deltas[sysname][dec]
            c_full = int(delta.sum())
            if c_full != E.scipy_matching(pool.edges, pool.n_labels):
                raise SystemExit(f"{sysname} {dec}: incremental C != scipy C")
            row = {"supply": len(pool.edges), "C": c_full, "budgets": {}}
            for b in B:
                k = int(E.budget(b, area))
                m = int(delta[:k].sum())
                used = min(k, len(pool.edges))
                row["budgets"][str(b)] = {"N": n, "K": k, "M": m, "R": m / n, "c": min(1.0, k / n),
                                          "matches_per_100_slots": 100 * m / k, "slots_used": used,
                                          "unused_slots": k - used, "submitted_unmatched": used - m,
                                          "R_oracle": min(k, c_full) / n}
            out[sysname][dec] = row
    return out


def reproduce_bench(reg: dict, deltas: dict, blk: E.Blocks, block_of: dict) -> dict:
    """All-ones block weights must reproduce every BENCH_V0 v0 score exactly (and its K)."""
    bench = D.bench_v0()[reg["name"]]
    ones = np.ones((1, len(blk.area_km2)), dtype=np.int64)
    alias = {"J_2026092803": "J_2026092803", "DEPLOY": f"DEPLOY_{D.SSTAR}"}
    checked, bad = 0, []
    for name, rec in bench["systems"].items():
        sysname = alias.get(name, name)
        for dec in DECODERS:
            r, _ = recall_reps(block_of[sysname][dec], deltas[sysname][dec], ones, blk.area_km2,
                               blk.n_labels.astype(float))
            for i, b in enumerate(B):
                checked += 1
                if r[0, i] != rec["v0_individual"][dec][str(b)]:
                    bad.append([name, dec, b, float(r[0, i]), rec["v0_individual"][dec][str(b)]])
    ks = {str(b): int(E.budget(b, blk.area_km2.sum())) for b in B}
    if bad or ks != bench["K"]:
        raise SystemExit(f"all-ones reproduction failed: {bad[:5]} K {ks} vs {bench['K']}")
    return {"scores_checked": checked, "exact_matches": checked, "K": ks}


def formula_check(blk: E.Blocks, block_of: dict, deltas: dict, rng: np.random.Generator, reps: int = 20) -> int:
    """Vectorised shortcut == literal R3-2 formula (F_j, t_j, q, d) on real pools for random multinomial weights."""
    w = E.multinomial_weights(rng, len(blk.area_km2), reps)
    n = 0
    for sysname, decs in block_of.items():
        for dec, bo in decs.items():
            for b in B:
                k = E.budget(b, w @ blk.area_km2)
                m, used = E.m_star(bo, deltas[sysname][dec], w, k)
                for r in range(reps):
                    if (int(m[r]), int(used[r])) != E.m_star_formula(bo, deltas[sysname][dec], w[r], int(k[r])):
                        raise SystemExit(f"formula mismatch {sysname} {dec} {b} rep {r}")
                    n += 1
    return n


# ---------------------------------------------------------------- per region
def grid_block(reg: dict, size: float, off: tuple) -> tuple[E.Blocks, dict, dict]:
    blk = E.build_blocks(reg["boxes"], reg["footprints"], D.rs.REACH_M, size, off)
    if abs(blk.area_km2.sum() - reg["area_km2"]) > 1e-6 or int(blk.n_labels.sum()) != reg["N"]:
        raise SystemExit("blocks do not partition the scored area / references")
    block_of, tot = {}, {"edges": 0, "edges_outside_one_block": 0, "edges_crossing_initial_cell": 0, "pools": 0}
    for sysname, decs in reg["pools"].items():
        block_of[sysname] = {}
        for dec, pool in decs.items():
            bo = blk.block_of_points(pool.x, pool.y)
            chk = E.verify_edges(blk, bo, (pool.x, pool.y), pool.edges)
            for k2 in chk:
                tot[k2] += chk[k2]
            tot["pools"] += 1
            block_of[sysname][dec] = bo
    if tot["edges_outside_one_block"]:
        raise SystemExit(f"{reg['name']}: {tot['edges_outside_one_block']} edges cross blocks")
    return blk, block_of, tot


def summaries(dsg: dict, geo: dict, point: dict, idx_train: np.ndarray, reps: int) -> dict:
    """Geography-only, training-only and joint paired differences for every contrast (ROUND3_REPLY.md:251-271)."""
    obs = np.broadcast_to(dsg["obs_idx"], (reps, dsg["obs_idx"].shape[1]))
    out = {}
    for x, y in dsg["contrasts"]:
        diff = {
            "geography_only": value(dsg, x, geo, obs) - value(dsg, y, geo, obs),
            "training_only": value(dsg, x, point, idx_train) - value(dsg, y, point, idx_train),
            "joint": value(dsg, x, geo, idx_train) - value(dsg, y, geo, idx_train)}
        p = value(dsg, x, point, dsg["obs_idx"])[0] - value(dsg, y, point, dsg["obs_idx"])[0]
        rec = {"point": pt4(p), "ci95": {k: ci4(np.asarray(v)) for k, v in diff.items()},
               "nan_replicates": {k: int(np.isnan(np.asarray(v)).any(axis=1).sum()) for k, v in diff.items()}}
        if len(dsg["members"][x]) > 1:
            xs, ys = dsg["members"][x], dsg["members"][y]
            rec["per_run"] = {xs[i]: pt4(point[xs[i]][0] - point[ys[i] if len(ys) > 1 else ys[0]][0])
                              for i in range(len(xs))}
        out[f"{x}-{y}"] = rec
    return out


def run_region(name: str, ss: dict) -> dict:
    t0 = time.time()
    reg = D.load_region(name)
    deltas = {s: {d: E.increments(p.edges, p.n_labels) for d, p in decs.items()} for s, decs in reg["pools"].items()}
    dsg = design(name)
    idx_train = dsg["draw"](np.random.default_rng(ss[f"train_{name}"]), REPS)
    out = {"inputs_sha256": reg["inputs"], "N": reg["N"], "total_scored_km2": reg["area_km2"],
           "training_design": dsg["rule"], "oracle_table": oracle_table(reg, deltas), "grids": {}}
    if name == "verden":
        out["training_draw_selected_counts"] = {dsg["members"]["S*"][i]: int(c) for i, c in
                                                enumerate(np.bincount(idx_train[:, 0], minlength=5))}
    for gname, (size, off) in GRIDS.items():
        blk, block_of, edge_tot = grid_block(reg, size, off)
        g = {"size_m": size, "offset_m": list(off), "blocks": blk.diag, "edge_check": edge_tot}
        if gname == PRIMARY_GRID:
            g["all_ones_reproduction"] = reproduce_bench(reg, deltas, blk, block_of)
            g["formula_vs_shortcut_checks"] = formula_check(blk, block_of, deltas, np.random.default_rng(7))
        w = E.multinomial_weights(np.random.default_rng(ss[f"geo_{name}"]), len(blk.area_km2), REPS)
        n_star = w @ blk.n_labels
        g["replicates"] = {"reps": REPS, "N_star_zero": int((n_star == 0).sum()),
                           "A_star_km2_range": [float((w @ blk.area_km2).min()), float((w @ blk.area_km2).max())],
                           "N_star_range": [int(n_star.min()), int(n_star.max())]}
        g["decoders"] = {}
        for dec in DECODERS:
            geo, point, under = {}, {}, {}
            for sysname in reg["pools"]:
                geo[sysname], u = recall_reps(block_of[sysname][dec], deltas[sysname][dec], w, blk.area_km2,
                                              blk.n_labels.astype(float))
                point[sysname] = recall_reps(block_of[sysname][dec], deltas[sysname][dec],
                                             np.ones((1, len(blk.area_km2)), dtype=np.int64), blk.area_km2,
                                             blk.n_labels.astype(float))[0]
                under[sysname] = {str(b): float(u[:, i].mean()) for i, b in enumerate(B)}
            g["decoders"][dec] = {
                "contrasts": summaries(dsg, geo, point, idx_train, REPS),
                "systems_point": {s: pt4(point[s][0]) for s in point},
                "systems_geo_ci95": {s: ci4(geo[s]) for s in geo},
                "underfilled_replicate_share": {s: v for s, v in under.items() if any(x > 0 for x in v.values())}}
        out["grids"][gname] = g
        print(name, gname, blk.diag["J"], f"{time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ss = streams()
    res = {"schema": "v0_uncertainty_boot_v1", "master_seed": MASTER_SEED, "reps": REPS,
           "streams": {k: {"entropy": MASTER_SEED, "spawn_key": list(v.spawn_key)} for k, v in ss.items()},
           "budgets_per_km2": list(B), "grids": {k: {"size_m": v[0], "offset_m": list(v[1])} for k, v in GRIDS.items()},
           "anchor_rule": "cell holding the footprint centroid (floor((x - ox) / size), floor((y - oy) / size))",
           "merge_rule": "anchor cell + every cell whose closed square intersects the 15 m reach buffer "
                         "(region_score REACH_M); union-find transitive closure; all scored cells kept",
           "interval": "percentile 2.5 / 97.5, pointwise", "regions": {}}
    for name in ("oldenburg", "verden"):
        res["regions"][name] = run_region(name, ss)
    (HERE / "BOOT_V0.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
