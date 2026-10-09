"""ADAPT-MATCH-0 step plan (pure, CPU; no torch): every one of the 5,116 updates is a Lower Saxony step.

Structure (PACKET.md "Arms"): A's run shape (ADAPT-NDS-1 nds_schedule: two epochs numbered 4 and 5 of 2,558 steps;
pair p = positive step 2p, background step 2p + 1; update index t = (epoch - 4) x 2,558 + step) with every pair a Lower
Saxony pair, so the run has 2,558 Lower Saxony pairs against A's 852.

Draws: the n-th pair of the run, n = (epoch - 4) x 1,279 + p, uses A's own Lower Saxony streams (nds_schedule.ls_plan,
same expressions): positive window pool[perm[idx]] with (pass, idx) = divmod(n, len(pool)) and
perm = default_rng([seed + 30, 0, pass]).permutation(len(pool)); background default_rng([seed + 30, 1, pass])
.permutation(bg)[idx] with (pass, idx) = divmod(n, len(bg)); crop offsets and dihedral
r2_train.draw(default_rng([seed + 30, 2, epoch, step])) (nds_schedule.draw, checked equal to the frozen function).
Each pass is a fresh permutation seeded per seed and per pass. Pass 0 is the order in which A meets the same windows
(A's 852 Lower Saxony pairs are n = 0..851). With 852 positive and 852 background windows, 2,558 = 3 x 852 + 2: passes
0-2 are complete and pass 3 holds two pairs, so 850 windows of each kind are drawn 3 times and 2 are drawn 4 times.

Learning rate: nds_schedule.lr_at (J's cosine 1e-4 -> 1e-6 with no warm-up, compressed to 5,116 updates); the update
index is A's (nds_schedule.t_index), so every update has A's learning rate at that index.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import match_common as M  # noqa: F401  (puts ADAPT-NDS-1 on sys.path)
import nds_schedule as SCH  # noqa: E402  frozen (ADAPT-NDS-1)

STEPS_PER_EPOCH = SCH.STEPS_PER_EPOCH          # 2,558
N_PAIRS = SCH.N_PAIRS                          # 1,279 per epoch
EPOCHS = SCH.EPOCHS                            # (4, 5)
N_UPDATES = SCH.N_UPDATES                      # 5,116
DRAW_OFFSET = SCH.LS_DRAW_OFFSET               # seed + 30 (A's Lower Saxony streams)
ROLES = ("positive", "background")


def total_pairs() -> int:
    return N_PAIRS * len(EPOCHS)


def pair_index(epoch: int, p: int) -> int:
    SCH.check_epoch(epoch)
    if not 0 <= p < N_PAIRS:
        raise ValueError("pair outside the epoch")
    return (epoch - EPOCHS[0]) * N_PAIRS + p


def _perm(seed: int, kind: int, cycle: int, pool: list[int], cache: dict) -> np.ndarray:
    """A's permutation for one kind and pass, cached per plan (the same expressions as nds_schedule.ls_plan)."""
    key = (kind, cycle)
    if key not in cache:
        rng = np.random.default_rng([seed + DRAW_OFFSET, kind, cycle])
        cache[key] = rng.permutation(len(pool)) if kind == 0 else rng.permutation(pool)
    return cache[key]


def pair_windows(seed: int, n: int, pool: list[int], bg: list[int], cache: dict | None = None) -> tuple[int, int]:
    """Row indices (positive, background) of the n-th Lower Saxony pair of a run (A's streams)."""
    cache = {} if cache is None else cache
    cycle, idx = divmod(n, len(pool))
    pos = pool[int(_perm(seed, 0, cycle, pool, cache)[idx])]
    cycle, idx = divmod(n, len(bg))
    neg = int(_perm(seed, 1, cycle, bg, cache)[idx])
    return int(pos), neg


def local_plan(seed: int, epoch: int, pool: list[int], bg: list[int]) -> dict[int, tuple[int, int, int, int]]:
    """Step -> (row index, dy, dx, k) for every step of the epoch (positive at 2p, background at 2p + 1)."""
    if not pool or not bg:
        raise ValueError("Lower Saxony pools are empty")
    SCH.check_epoch(epoch)
    d, cache, out = seed + DRAW_OFFSET, {}, {}
    for p in range(N_PAIRS):
        wp, wb = pair_windows(seed, pair_index(epoch, p), pool, bg, cache)
        out[2 * p] = (wp, *SCH.draw(np.random.default_rng([d, 2, epoch, 2 * p])))
        out[2 * p + 1] = (wb, *SCH.draw(np.random.default_rng([d, 2, epoch, 2 * p + 1])))
    return out


def run_plan(seed: int, pool: list[int], bg: list[int]) -> dict[int, dict]:
    return {e: local_plan(seed, e, pool, bg) for e in EPOCHS}


def a_order(seed: int, pool: list[int], bg: list[int]) -> list[tuple[int, int]]:
    """A's Lower Saxony pairs in run order, from the frozen nds_schedule.ls_plan (epoch 4 then 5, by step)."""
    out = []
    for e in EPOCHS:
        m = SCH.ls_plan(seed, e, pool, bg)
        out += [(m[2 * p][0], m[2 * p + 1][0]) for p in SCH.ls_pairs(e)]
    return out


def exposure(plan: dict[int, dict], pool: list[int], bg: list[int]) -> dict:
    """Draws per window over the run, by role; histogram {draws: windows}."""
    cnt = {"positive": Counter({w: 0 for w in pool}), "background": Counter({w: 0 for w in bg})}
    for e in EPOCHS:
        for s, v in plan[e].items():
            cnt[ROLES[s % 2]][v[0]] += 1
    out = {}
    for role, c in cnt.items():
        vals = list(c.values())
        out[role] = {"windows": len(vals), "draws": sum(vals), "min": min(vals), "max": max(vals),
                     "mean": sum(vals) / len(vals), "histogram": {str(k): v for k, v in sorted(Counter(vals).items())}}
    return out


def expected_exposure(n_pool: int, n_bg: int) -> dict:
    """Closed form: pairs = q x size + r -> r windows drawn q + 1 times, size - r windows q times."""
    out = {}
    for role, size in (("positive", n_pool), ("background", n_bg)):
        q, r = divmod(total_pairs(), size)
        out[role] = {str(k): v for k, v in sorted({q: size - r, q + 1: r}.items()) if v}
    return out


def pass_complete(plan: dict[int, dict], pool: list[int], bg: list[int]) -> list[str]:
    """Violations: within every complete pass each window of each role is drawn exactly once."""
    seq = {"positive": [], "background": []}
    for e in EPOCHS:
        for s in range(STEPS_PER_EPOCH):
            seq[ROLES[s % 2]].append(plan[e][s][0])
    bad = []
    for role, size, want in (("positive", len(pool), set(pool)), ("background", len(bg), set(bg))):
        full = len(seq[role]) // size
        for k in range(full):
            chunk = seq[role][k * size:(k + 1) * size]
            if len(set(chunk)) != size or set(chunk) != want:
                bad.append(f"{role} pass {k} is not a permutation of the pool")
    return bad


def lr_trace() -> dict:
    """First, middle and last learning rates of the compressed schedule (A's index -> A's rate)."""
    return {"t0": SCH.lr_at(0), "t_mid": SCH.lr_at(N_UPDATES // 2), "t_last": SCH.lr_at(N_UPDATES - 1),
            "n_updates": N_UPDATES, "warmup_updates": 0}
