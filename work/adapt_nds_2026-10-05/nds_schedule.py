"""ADAPT-NDS-1 step schedule (pure, CPU; no torch): which region each of the 5,116 updates draws from, in A and in C.

Continuation of JOINT-1's J (PACKET.md "Arms"): two J epochs, numbered 4 and 5 so that every J draw stream continues
where J's four epochs (0-3) stopped: the English plan plans.real_plan(seed, epoch) (the generator of J's PLAN files),
HR1 A_W30's Welsh pairs on (p - e) % 4 == 0 and HR3 A_I30's Irish streams on (p - e) % 4 == 2 (JOINT-1 joint_train),
English otherwise. Pair p = (positive step 2p, background step 2p + 1); 1,279 pairs per epoch.

  C (control)  J's structure unchanged: every step is J's English, Welsh or Irish draw for epochs 4-5.
  A (adapt)    C's schedule with every pair whose (p - e) % 3 == 0 replaced by a Lower Saxony (positive, background)
               pair. All other steps keep C's draw exactly (same window, crop offsets, dihedral, update index, LR).
Shares: A gives Lower Saxony 852 of 2,558 pairs (1,704 of 5,116 updates; one third rounded down, since 2,558 is not a
multiple of 3). Because 3 and 4 are coprime, every complete 12-pair cycle holds, in A, 4 Lower Saxony, 4 English,
2 Welsh, 2 Irish pairs, and C replaces the 4 Lower Saxony pairs by 2 English, 1 Welsh and 1 Irish pair, i.e. J's
proportions (1/2, 1/4, 1/4); C as a whole is J's structure. Exact counts over the run (pairs; updates = 2 x pairs):
A: Lower Saxony 852, English 853, Welsh 426, Irish 427; C: English 1,279, Welsh 640, Irish 639 (J's epochs 4-5);
C's replacements of the 852 Lower Saxony pairs: English 426, Welsh 214, Irish 212 (106 complete 4-pair cycles per
epoch plus one English and one Welsh pair). Functions: ls_pair_count(), kind_counts(), replacement_counts().

Lower Saxony draws mirror HR1 A_W30's Welsh pair draws (welsh_train.welsh_plan): the n-th Lower Saxony pair of the run
(n counts from epoch 4) takes positive window pool[perm_cycle[idx]] with (cycle, idx) = divmod(n, len(pool)) and
perm_cycle = default_rng([seed + 30, 0, cycle]).permutation(len(pool)), and background window
default_rng([seed + 30, 1, cycle]).permutation(bg)[idx]; crop offsets and dihedral r2_train.draw(default_rng(
[seed + 30, 2, epoch, step])). With pool and background sizes equal to the run's 852 pairs, every Lower Saxony window
is drawn exactly once per run.

The learning rate is J's schedule compressed: plans.lr_at is a cosine from 1e-4 to 1e-6 over 10,232 updates with no
warm-up phase (warm-up length 0); compressed, lr(t) = 1e-6 + 0.5 (1e-4 - 1e-6) (1 + cos(pi t / 5,115)), t = 0..5,115,
t = (epoch - 4) * 2,558 + step.
"""
from __future__ import annotations

import math

import numpy as np

STEPS_PER_EPOCH = 2558               # plans.STEPS_PER_EPOCH (2 x 1,279 English FIT positives); checked by the trainer
N_PAIRS = STEPS_PER_EPOCH // 2
EPOCHS = (4, 5)
N_UPDATES = STEPS_PER_EPOCH * len(EPOCHS)
LR_MAX, LR_MIN = 1e-4, 1e-6
WELSH_RESIDUE, IRISH_RESIDUE, J_MOD = 0, 2, 4      # JOINT-1 joint_common: (p - e) % 4
LS_MOD, LS_RESIDUE = 3, 0                          # ADAPT-NDS-1: (p - e) % 3 == 0 -> Lower Saxony in A
LS_DRAW_OFFSET = 30
OFFSET_PX = 160                                    # r2_train.OFFSET_PX
KINDS = ("english", "welsh", "irish", "lower_saxony")


def check_epoch(epoch: int) -> None:
    if epoch not in EPOCHS:
        raise ValueError(f"epoch {epoch} outside the continuation epochs {EPOCHS}")


def j_kind(p: int, epoch: int) -> str:
    r = (p - epoch) % J_MOD
    return "welsh" if r == WELSH_RESIDUE else "irish" if r == IRISH_RESIDUE else "english"


def is_ls_pair(p: int, epoch: int) -> bool:
    return (p - epoch) % LS_MOD == LS_RESIDUE


def pair_kind(arm: str, p: int, epoch: int) -> str:
    if arm not in ("A", "C"):
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "A" and is_ls_pair(p, epoch):
        return "lower_saxony"
    return j_kind(p, epoch)


def ls_pairs(epoch: int) -> list[int]:
    check_epoch(epoch)
    return [p for p in range(N_PAIRS) if is_ls_pair(p, epoch)]


def ls_steps(epoch: int) -> set[int]:
    return {s for p in ls_pairs(epoch) for s in (2 * p, 2 * p + 1)}


def ls_pair_count() -> int:
    return sum(len(ls_pairs(e)) for e in EPOCHS)


def ls_offset(epoch: int) -> int:
    """Lower Saxony pairs drawn before this epoch in the run (the n counter starts at 0 in epoch 4)."""
    check_epoch(epoch)
    return sum(len(ls_pairs(e)) for e in EPOCHS if e < epoch)


def kind_counts(arm: str) -> dict:
    """Pairs and updates per region over the whole run (two epochs)."""
    pairs = {k: 0 for k in KINDS}
    for e in EPOCHS:
        for p in range(N_PAIRS):
            pairs[pair_kind(arm, p, e)] += 1
    return {"pairs": pairs, "updates": {k: 2 * v for k, v in pairs.items()},
            "pair_share": {k: v / (N_PAIRS * len(EPOCHS)) for k, v in pairs.items()}}


def replacement_counts() -> dict:
    """J's kinds of the pairs that A gives to Lower Saxony (what C trains on in their place)."""
    out = {"english": 0, "welsh": 0, "irish": 0}
    for e in EPOCHS:
        for p in ls_pairs(e):
            out[j_kind(p, e)] += 1
    return out


def t_index(epoch: int, step: int) -> int:
    check_epoch(epoch)
    if not 0 <= step < STEPS_PER_EPOCH:
        raise ValueError("step outside the epoch")
    return (epoch - EPOCHS[0]) * STEPS_PER_EPOCH + step


def lr_at(t: int) -> float:
    """J's cosine (plans.lr_at: 1e-4 -> 1e-6, no warm-up) compressed from 10,232 to 5,116 updates."""
    if not 0 <= t < N_UPDATES:
        raise ValueError("update index outside the compressed schedule")
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t / (N_UPDATES - 1)))


def draw(rng: np.random.Generator) -> tuple[int, int, int]:
    """r2_train.draw (work/campaign_2026-09-22/r2_train.py), copied verbatim as a pure helper: offsets +-160 px and
    one of 8 dihedral transforms (tests compare it with the frozen function)."""
    dy, dx = (int(v) for v in rng.integers(-OFFSET_PX, OFFSET_PX + 1, size=2))
    return dy, dx, int(rng.integers(0, 8))


def ls_plan(seed: int, epoch: int, pool: list[int], bg: list[int]) -> dict[int, tuple[int, int, int, int]]:
    """Step -> (row index, dy, dx, k) for this epoch's Lower Saxony pairs (A only). pool = positive row indices (one
    window per drawn group), bg = background row indices."""
    if not pool or not bg:
        raise ValueError("Lower Saxony pools are empty")
    d = seed + LS_DRAW_OFFSET
    off = ls_offset(epoch)
    out = {}
    for m, p in enumerate(ls_pairs(epoch)):
        n = off + m
        cycle, idx = divmod(n, len(pool))
        wi = pool[int(np.random.default_rng([d, 0, cycle]).permutation(len(pool))[idx])]
        out[2 * p] = (int(wi), *draw(np.random.default_rng([d, 2, epoch, 2 * p])))
        cycle, idx = divmod(n, len(bg))
        wi = int(np.random.default_rng([d, 1, cycle]).permutation(bg)[idx])
        out[2 * p + 1] = (wi, *draw(np.random.default_rng([d, 2, epoch, 2 * p + 1])))
    return out


def compose(arm: str, epoch: int, j_map: dict, ls_map: dict) -> dict:
    """The arm's regional step map for one epoch. j_map = J's Welsh + Irish map for the epoch (JOINT-1 draws); ls_map =
    ls_plan's map with row indices already offset into the joint row table. C = j_map; A = j_map without the Lower
    Saxony pairs' steps, plus ls_map. Raises if the inputs do not have J's and the Lower Saxony structure."""
    check_epoch(epoch)
    want_j = {s for p in range(N_PAIRS) if j_kind(p, epoch) != "english" for s in (2 * p, 2 * p + 1)}
    if set(j_map) != want_j:
        raise RuntimeError(f"epoch {epoch}: J's regional steps differ from the pairs with (p - e) % 4 in (0, 2)")
    if arm == "C":
        return dict(j_map)
    if arm != "A":
        raise ValueError(f"unknown arm {arm!r}")
    ls = ls_steps(epoch)
    if set(ls_map) != ls:
        raise RuntimeError(f"epoch {epoch}: Lower Saxony steps differ from the pairs with (p - e) % 3 == 0")
    out = {s: v for s, v in j_map.items() if s not in ls}
    out.update(ls_map)
    return out


def step_sources(arm: str, epoch: int, real: list, wmap: dict, rows: list[dict]) -> list[tuple]:
    """Per step: (source, window uid or English window index, dy, dx, k) exactly as AdaptSteps reads it (a step in
    wmap reads the regional row, otherwise the English plan entry)."""
    out = []
    for s in range(STEPS_PER_EPOCH):
        if s in wmap:
            wi, dy, dx, k = wmap[s]
            out.append((rows[wi]["region"], rows[wi]["uid"], int(dy), int(dx), int(k)))
        else:
            wi, dy, dx, k = real[s]
            out.append(("english", int(wi), int(dy), int(dx), int(k)))
    return out
