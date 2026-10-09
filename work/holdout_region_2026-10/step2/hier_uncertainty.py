"""HOLDOUT-REGION-1 Step 2a (b): hierarchical uncertainty for the paired DEV endpoint (packet "Endpoints").

Each of 2,000 draws: (1) resample DEV super-blocks with replacement (positives and background together); (2) recompute
every model's thresholds from the resampled background (allowance floor(d x resampled background km2), proposals
counted with their super-block multiplicity, exactly score_holdout.recall_at with wmult); (3) resample the five seed
pairs with replacement; (4) take the mean paired delta. The 95 % interval is the 2.5/97.5 percentile of the draws.

Vectorised form of recall_at. A model is a ModelArrays: ranking keys (higher = earlier in the pool); a part counts as
hit under weights w when one of its hitting proposals lies in a cluster with w > 0 and precedes the first background
proposal whose cumulative weight exceeds the allowance. test_hier_uncertainty.py checks it equals recall_at.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DENSITIES = (0.5, 1.0, 2.0)
DRAWS, SEED = 2000, 20261001
NEG_INF = -np.inf


@dataclass
class ModelArrays:
    neg_key: np.ndarray       # (N,) background proposal keys, sorted descending
    neg_cluster: np.ndarray   # (N,) cluster index of each background proposal
    hit_key: np.ndarray       # (P, H) keys of proposals hitting each part, -inf padded
    hit_cluster: np.ndarray   # (P, H) cluster of each hitting proposal (pad 0)
    part_cluster: np.ndarray  # (P,) cluster of each part's home tile
    part_class: np.ndarray    # (P,) 0 barrow, 1 enclosure

    def __post_init__(self) -> None:
        if self.neg_key.size and np.any(np.diff(self.neg_key) > 0):
            raise ValueError("neg_key must be sorted descending")


def from_pool(pool: list[dict], s: dict, cluster_index: dict[str, int] | None = None) -> ModelArrays:
    """ModelArrays from a region_score / score_holdout pool (key = -position in rank order).
    cluster_index maps s['cluster'] labels to integers; without it each tile is its own cluster."""
    ntiles = len(s["wins"])
    tc = (np.arange(ntiles) if cluster_index is None
          else np.array([cluster_index[c] for c in s["cluster"]]))
    neg = [(-i, tc[r["wi"]]) for i, r in enumerate(pool) if r["neg"]]
    hits: dict[str, list] = {p: [] for p in s["eligible"]}
    for i, r in enumerate(pool):
        for p in r["hits"]:
            hits[p].append((-i, tc[r["wi"]]))
    h = max([1] + [len(v) for v in hits.values()])
    hk = np.full((len(s["eligible"]), h), NEG_INF)
    hc = np.zeros((len(s["eligible"]), h), dtype=int)
    for j, p in enumerate(s["eligible"]):
        for m, (k, c) in enumerate(hits[p]):
            hk[j, m], hc[j, m] = k, c
    return ModelArrays(np.array([k for k, _ in neg], dtype=float), np.array([c for _, c in neg], dtype=int), hk, hc,
                       np.array([tc[s["home"][p]] for p in s["eligible"]], dtype=int),
                       np.array([1 if s["klass"][p] == "enclosure" else 0 for p in s["eligible"]], dtype=int))


def break_keys(m: ModelArrays, w: np.ndarray, bg_area: np.ndarray, densities=DENSITIES) -> np.ndarray:
    """(len(densities), D) key of the first background proposal whose cumulative weight exceeds
    floor(d x weighted area); -inf when none does (recall_at's break)."""
    area = w @ bg_area
    if m.neg_key.size == 0:
        return np.full((len(densities), w.shape[0]), NEG_INF)
    cum = np.cumsum(w[:, m.neg_cluster], axis=1)
    out = []
    for d in densities:
        over = cum > np.floor(d * area + 1e-9)[:, None]
        out.append(np.where(over.any(axis=1), m.neg_key[np.argmax(over, axis=1)], NEG_INF))
    return np.stack(out)


def detected(m: ModelArrays, w: np.ndarray, bg_area: np.ndarray, densities=DENSITIES) -> np.ndarray:
    """(len(densities), D, P) boolean detection under cluster weights w (D, C)."""
    valid = w[:, m.hit_cluster] > 0                               # (D, P, H)
    best = np.where(valid, m.hit_key[None], NEG_INF).max(axis=2)   # (D, P)
    return best[None] > break_keys(m, w, bg_area, densities)[:, :, None]


def recalls(m: ModelArrays, w: np.ndarray, bg_area: np.ndarray, densities=DENSITIES) -> dict:
    """Per-density recall (len(densities), D) overall and per class, weighted by home-cluster multiplicity."""
    det = detected(m, w, bg_area, densities)
    pw = w[:, m.part_cluster]                                      # (D, P)
    out = {"all": (det * pw[None]).sum(axis=2) / pw.sum(axis=1)[None]}
    for k, name in ((0, "barrow"), (1, "enclosure")):
        sel = m.part_class == k
        den = pw[:, sel].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = (det[:, :, sel] * pw[None, :, sel]).sum(axis=2) / den[None]
    return out


def primary(m: ModelArrays, w: np.ndarray, bg_area: np.ndarray, key: str = "all") -> np.ndarray:
    return recalls(m, w, bg_area)[key].mean(axis=0)


def cluster_weights(rng: np.random.Generator, n_clusters: int, draws: int) -> np.ndarray:
    idx = rng.integers(0, n_clusters, size=(draws, n_clusters))
    return np.stack([np.bincount(r, minlength=n_clusters) for r in idx]).astype(float)


def hierarchical_ci(a0: list[ModelArrays], ak: list[ModelArrays], bg_area: np.ndarray, draws: int = DRAWS,
                    seed: int = SEED, chunk: int = 500) -> dict:
    """Paired seeds (a0[p], ak[p]); returns the percentile 95 % interval of the mean paired delta (primary)."""
    if len(a0) != len(ak) or not a0:
        raise ValueError("need equal, non-empty paired model lists")
    n_c, n_p = len(bg_area), len(a0)
    rng = np.random.default_rng(seed)
    w_all = cluster_weights(rng, n_c, draws)
    pairs = rng.integers(0, n_p, size=(draws, n_p))
    deltas = np.empty((draws, n_p))
    for s0 in range(0, draws, chunk):
        w = w_all[s0:s0 + chunk]
        for p in range(n_p):
            deltas[s0:s0 + chunk, p] = primary(ak[p], w, bg_area) - primary(a0[p], w, bg_area)
    if not np.all(np.isfinite(deltas)):
        raise ValueError("non-finite delta in a draw (no weighted parts); fail closed")
    means = np.take_along_axis(deltas, pairs, axis=1).mean(axis=1)
    point_w = np.ones((1, n_c))
    point = float(np.mean([primary(ak[p], point_w, bg_area)[0] - primary(a0[p], point_w, bg_area)[0]
                           for p in range(n_p)]))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"lo95": float(lo), "hi95": float(hi), "median": float(np.median(means)), "point_mean_delta": point,
            "draws": draws, "n_clusters": n_c, "n_pairs": n_p, "seed": seed}


def point_recall_exact_allowance(d: float, area: float) -> int:
    """Allowance used by score_holdout.recall_at (documented helper for tests)."""
    return math.floor(d * area + 1e-9)
