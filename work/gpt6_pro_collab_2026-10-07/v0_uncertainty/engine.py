"""v0 uncertainty engine (GPT-6 Pro Round 3, R3-2; ROUND3_REPLY.md:110-224).

Pure numpy / scipy / shapely; no project imports. Pieces:

- build_blocks: grid cells aligned to (offset) in the metric CRS; every eligible reference is anchored to the cell
  holding its footprint centroid (fixed anchor rule); its anchor cell is merged with every cell whose closed square
  intersects the reference's 15 m matching buffer; transitive closure by union-find. Every cell holding scored area is
  a block member, so background-only blocks are included and the scored area is assigned exactly once.
- increments: delta[i] = F(i+1) - F(i) for the prefix maximum one-to-one matching over proposals in frozen rank
  order (Kuhn augmenting paths). With edges contained in blocks, F decomposes over blocks, so the same delta gives
  every block's F_j(t) curve.
- m_star: the exact shortcut M*_b = sum_j w_j F_j(t_j) + d [F_q(t_q + 1) - F_q(t_q)] (ROUND3_REPLY.md:184-214),
  computed as a weighted prefix sum of delta, vectorised over replicates.
- m_star_formula: the same quantity written literally with F_j, t_j, q and d (cross-check).
- copied_graph_m: literal copied-graph matching (copies with separate namespaces, order = original rank then copy id,
  first K* copies, scipy maximum matching) for tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from shapely.geometry import box
from shapely.ops import unary_union

K_EPS = 1e-9  # bench_v0.py:95 uses floor(b * area + 1e-9); kept identical so all-ones weights reproduce BENCH_V0


def budget(b: float, area_km2: np.ndarray | float) -> np.ndarray:
    """K*_b = floor(b * A*) with A* in km2 (same tolerance as bench_v0)."""
    return np.floor(b * np.asarray(area_km2, dtype=float) + K_EPS).astype(np.int64)


# ---------------------------------------------------------------- matching increments
def _augment(root: int, edges: list[np.ndarray], match_label: np.ndarray) -> bool:
    """Iterative Kuhn DFS from left vertex root; flips the path and returns True when a free label is reached."""
    visited: set[int] = set()
    stack = [[root, 0]]
    path: list[tuple[int, int]] = []
    while stack:
        u, k = stack[-1]
        e = edges[u]
        if k >= len(e):
            stack.pop()
            if path:
                path.pop()
            continue
        stack[-1][1] = k + 1
        v = int(e[k])
        if v in visited:
            continue
        visited.add(v)
        if match_label[v] < 0:
            for pu, pv in path + [(u, v)]:
                match_label[pv] = pu
            return True
        path.append((u, v))
        stack.append([int(match_label[v]), 0])
    return False


def increments(edges: list[np.ndarray], n_labels: int) -> np.ndarray:
    """delta[i] in {0, 1}: increase of the maximum matching when proposal i joins the prefix 0..i-1. Kuhn's invariant
    (the matching after processing a prefix is maximum for that prefix) makes this exact."""
    match_label = np.full(n_labels, -1, dtype=np.int64)
    delta = np.zeros(len(edges), dtype=np.int64)
    for i, e in enumerate(edges):
        if len(e) and _augment(i, edges, match_label):
            delta[i] = 1
    return delta


def scipy_matching(edges: list[np.ndarray], n_labels: int) -> int:
    """Independent maximum-cardinality matching (scipy), as bench_v0.match_count computes it."""
    r = [i for i, e in enumerate(edges) for _ in e]
    c = [int(v) for e in edges for v in e]
    if not r:
        return 0
    g = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(edges), n_labels))
    return int((maximum_bipartite_matching(g, perm_type="column") >= 0).sum())


# ---------------------------------------------------------------- the shortcut
def m_star(block_of: np.ndarray, delta: np.ndarray, w: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact copied-graph M* for each replicate.

    block_of (P,): block of each proposal, in frozen rank order; delta (P,): increments(); w (R, J) block weights;
    k (R,) copied budgets. Returns (M* (R,), copies supplied within the budget (R,)); supplied < k means underfilled.
    """
    w = np.atleast_2d(w)
    r_n = w.shape[0]
    k = np.broadcast_to(np.asarray(k, dtype=np.int64), (r_n,))
    if len(block_of) == 0:
        return np.zeros(r_n, dtype=np.int64), np.zeros(r_n, dtype=np.int64)
    mult = w[:, block_of]
    cum = np.cumsum(mult, axis=1)
    cumd = np.cumsum(mult * delta[None, :], axis=1)
    total = cum[:, -1]
    i_cut = (cum < k[:, None]).sum(axis=1)  # first index with cum >= k (P when the pool is exhausted)
    full = i_cut >= len(block_of)
    ic = np.minimum(i_cut, len(block_of) - 1)
    rows = np.arange(r_n)
    before_cum = np.where(ic > 0, cum[rows, np.maximum(ic - 1, 0)], 0)
    before_m = np.where(ic > 0, cumd[rows, np.maximum(ic - 1, 0)], 0)
    d = k - before_cum
    m = np.where(full, cumd[:, -1], before_m + d * delta[ic])
    supplied = np.where(full, total, k)
    return m.astype(np.int64), supplied.astype(np.int64)


def m_star_formula(block_of: np.ndarray, delta: np.ndarray, w: np.ndarray, k: int) -> tuple[int, int]:
    """The R3-2 formula written out: F_j prefix curves, t_j before the cutoff, cutoff block q, d copies of the cutoff
    proposal. Returns (M*, copies supplied)."""
    n_blocks = len(w)
    local = [[] for _ in range(n_blocks)]
    for i, j in enumerate(block_of):
        local[j].append(int(delta[i]))
    f = [np.concatenate([[0], np.cumsum(x)]).astype(int) for x in local]
    t = np.zeros(n_blocks, dtype=int)
    cum = 0
    for i, j in enumerate(block_of):
        if w[j] == 0:
            t[j] += 1
            continue
        if cum + w[j] >= k:
            d = k - cum
            m = sum(int(w[b]) * f[b][t[b]] for b in range(n_blocks)) + d * (f[j][t[j] + 1] - f[j][t[j]])
            return int(m), int(k)
        cum += int(w[j])
        t[j] += 1
    return int(sum(int(w[b]) * f[b][-1] for b in range(n_blocks))), int(cum)


def copied_graph_m(edges: list[np.ndarray], block_of: np.ndarray, label_block: np.ndarray, w: np.ndarray,
                   k: int) -> int:
    """Literal procedure (ROUND3_REPLY.md:167-176): w_j copies of every block's labels and proposals, separate copy
    namespaces, order = original rank then copy id, first k copies, one-to-one maximum matching."""
    lab_index: dict[tuple[int, int], int] = {}
    for lab, j in enumerate(label_block):
        for c in range(int(w[j])):
            lab_index[(lab, c)] = len(lab_index)
    rows: list[list[int]] = []
    for i, j in enumerate(block_of):
        for c in range(int(w[j])):
            if len(rows) == k:
                break
            rows.append([lab_index[(int(v), c)] for v in edges[i]])
        if len(rows) == k:
            break
    return scipy_matching([np.asarray(r, dtype=np.int64) for r in rows], len(lab_index))


# ---------------------------------------------------------------- geographical blocks
def cell_of(x: np.ndarray | float, y: np.ndarray | float, size: float, offset: tuple[float, float]) -> tuple:
    return (np.floor((np.asarray(x) - offset[0]) / size).astype(np.int64),
            np.floor((np.asarray(y) - offset[1]) / size).astype(np.int64))


def cell_box(c: tuple[int, int], size: float, offset: tuple[float, float]):
    x0, y0 = offset[0] + c[0] * size, offset[1] + c[1] * size
    return box(x0, y0, x0 + size, y0 + size)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


@dataclass
class Blocks:
    size: float
    offset: tuple[float, float]
    cell_block: dict
    cell_poly: dict  # cell -> scored-area polygon inside it (m2)
    area_km2: np.ndarray
    n_labels: np.ndarray
    label_block: np.ndarray
    label_anchor: list
    merges: int
    diag: dict = field(default_factory=dict)

    def block_of_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cx, cy = cell_of(x, y, self.size, self.offset)
        try:
            return np.array([self.cell_block[(int(a), int(b))] for a, b in zip(cx, cy)], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"a proposal lies in a cell without scored area or reference: {exc}") from None

    def masked_area_km2(self, mask) -> np.ndarray:
        """Per-block scored area after removing a geographical exclusion mask (None: unchanged)."""
        if mask is None:
            return self.area_km2.copy()
        out = np.zeros(len(self.area_km2))
        for c, poly in self.cell_poly.items():
            out[self.cell_block[c]] += poly.difference(mask).area / 1e6
        return out


def _cell_polys(scored_boxes: list[tuple[float, float, float, float]], size: float,
                offset: tuple[float, float]) -> dict:
    clipped: dict = {}
    for bx in scored_boxes:
        x0, y0, x1, y1 = bx
        (i0, j0), (i1, j1) = cell_of(x0, y0, size, offset), cell_of(x1, y1, size, offset)
        for i in range(int(i0), int(i1) + 1):
            for j in range(int(j0), int(j1) + 1):
                g = box(x0, y0, x1, y1).intersection(cell_box((i, j), size, offset))
                if g.area > 0:
                    clipped.setdefault((i, j), []).append(g)
    return {c: unary_union(gs) for c, gs in clipped.items()}


def build_blocks(scored_boxes: list[tuple[float, float, float, float]], footprints: list, reach_m: float,
                 size: float, offset: tuple[float, float] = (0.0, 0.0)) -> Blocks:
    """R3-2 step 2 (ROUND3_REPLY.md:121-136). footprints: shapely polygons of the eligible references, in the
    order of the scorer's eligible list."""
    polys = _cell_polys(scored_boxes, size, offset)
    uf = _UnionFind()
    for c in polys:
        uf.find(c)
    anchors, merges = [], 0
    for fp in footprints:
        cen = fp.centroid
        a = tuple(int(v) for v in cell_of(cen.x, cen.y, size, offset))
        anchors.append(a)
        uf.find(a)
        buf = fp.buffer(reach_m)
        x0, y0, x1, y1 = buf.bounds
        (i0, j0), (i1, j1) = cell_of(x0, y0, size, offset), cell_of(x1, y1, size, offset)
        for i in range(int(i0), int(i1) + 1):
            for j in range(int(j0), int(j1) + 1):
                if (i, j) != a and cell_box((i, j), size, offset).intersects(buf):
                    merges += uf.union(a, (i, j))
    roots = sorted({uf.find(c) for c in uf.parent})
    rid = {r: n for n, r in enumerate(roots)}
    cell_block = {c: rid[uf.find(c)] for c in uf.parent}
    area = np.zeros(len(roots))
    for c, p in polys.items():
        area[cell_block[c]] += p.area / 1e6
    label_block = np.array([cell_block[a] for a in anchors], dtype=np.int64)
    n_lab = np.bincount(label_block, minlength=len(roots)).astype(np.int64)
    blk = Blocks(size, offset, cell_block, polys, area, n_lab, label_block, anchors, merges)
    blk.diag = concentration(area, n_lab) | {"cells_with_scored_area": len(polys),
                                              "cells_total": len(cell_block), "cell_merges": merges}
    return blk


def concentration(area: np.ndarray, n_lab: np.ndarray) -> dict:
    a, n = float(area.sum()), float(n_lab.sum())
    return {"J": int(len(area)), "blocks_with_labels": int((n_lab > 0).sum()),
            "background_only_blocks": int((n_lab == 0).sum()), "zero_area_blocks": int((area <= 0).sum()),
            "total_area_km2": a, "total_labels": int(n),
            "largest_area_share": float(area.max() / a), "largest_label_share": float(n_lab.max() / n) if n else None,
            "J_area": a * a / float((area ** 2).sum()), "J_labels": n * n / float((n_lab.astype(float) ** 2).sum())
            if n else None}


def verify_edges(blk: Blocks, prop_block: np.ndarray, prop_xy: tuple[np.ndarray, np.ndarray],
                 edges: list[np.ndarray]) -> dict:
    """Every proposal-reference edge inside one block; counts edges whose proposal cell differs from the reference's
    anchor cell (edges that cross an initial grid boundary and were rescued by the merge)."""
    cx, cy = cell_of(prop_xy[0], prop_xy[1], blk.size, blk.offset)
    n_edges = bad = crossing = 0
    for i, e in enumerate(edges):
        for v in e:
            n_edges += 1
            bad += int(blk.label_block[v] != prop_block[i])
            crossing += int(blk.label_anchor[v] != (int(cx[i]), int(cy[i])))
    return {"edges": n_edges, "edges_outside_one_block": bad, "edges_crossing_initial_cell": crossing}


def multinomial_weights(rng: np.random.Generator, n_blocks: int, reps: int) -> np.ndarray:
    """(w_1..w_J) ~ Multinomial(J; 1/J, ..., 1/J), one row per replicate."""
    return rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=reps).astype(np.int64)


def percentile_ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]

