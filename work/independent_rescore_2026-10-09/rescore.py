"""Independent re-scorer for the German benchmark-v0 headline numbers (2026-10-09).

Written from protocol text only (evaluator v0.1 SPEC.md sections 4 and 6, Stage 1 FREEZE.md,
Stage 2 PACKET.md "Primary scorer", Oldenburg PACKET.md items 3-4, manuscript DRAFT_v1 sections 4.1/4.4/4.5).
No project scorer or evaluator module was read or imported.

Rule as implemented (reading (i), historical: no IGNORE inside positive cores, every in-core proposal counts):
  - entities: every part of the frozen individual manifest (N = 728 Verden, 959 Oldenburg);
  - scored area A: union of the manifest tile cores (boxes); K = floor(b * A_km2) [+1e-9 guard];
  - proposals: every saved proposal whose centroid lies in its own tile's core (half-open box test);
  - RAW order: score desc, mean_probability desc, area_m2 desc, id asc; top K kept;
  - edge: proposal point within 15 m of the entity footprint polygon (distance 0 inside), inclusive;
  - matches: maximum-cardinality bipartite matching (scipy Hopcroft-Karp), cross-checked with networkx.
Sensitivity variants: strict < 15 m edge; closed core test; no in-core filter; tie-boundary audit.

Output is aggregate only (no coordinates). Run in WSL v3venv, CPU only.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "RESCORE_RAW.json"
BUDGETS = (0.5, 1.0, 2.0)
REACH = 15.0
SEEDS = [f"20260928{s:02d}" for s in range(5, 15)]

VERDEN = dict(
    manifest="outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json",
    adaf="outputs/fresh_region_2026-10-03/step2/score/individual/PROPOSALS_ADAF_barrow_PRIVATE.json",
    d1="outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/v1/results/dev/PROPOSALS_D1F1_BARROW_L_{s}_PRIVATE.json",
    ctl="outputs/stage2_d1_controls_2026-10-07/chain_runs/run_s1_20261008T011425Z/v1/results/dev/PROPOSALS_S2CTL_BARROW_{s}_PRIVATE.json",
)
OLDENBURG = dict(
    manifest="outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json",
    adaf="outputs/adapt_nds_2026-10-05/score/confirm/individual/PROPOSALS_ADAF_barrow_PRIVATE.json",
    adaf_reinfer="outputs/oldenburg_transfer_2026-10-07/chain_runs/run_s1_20261008T203551Z/v1/results/confirm/PROPOSALS_ADAF_barrow_PRIVATE.json",
    d1="outputs/oldenburg_transfer_2026-10-07/chain_runs/run_s1_20261008T203551Z/v1/results/confirm/PROPOSALS_D1F1_BARROW_L_{s}_PRIVATE.json",
)


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


class Region:
    def __init__(self, manifest_path):
        m = load(manifest_path)
        self.manifest_sha = sha(manifest_path)
        self.cores = {}
        boxes = []
        for t in m["tiles"]:
            bs = [tuple(float(v) for v in c) for c in t["core"]]
            self.cores[t["id"]] = bs
            boxes += [box(*b) for b in bs]
        union = unary_union(boxes)
        self.area_km2 = union.area / 1e6
        self.area_sum_km2 = sum(b.area for b in boxes) / 1e6
        self.n_tiles = len(m["tiles"])
        self.polys = [shape(p["geometry"]) for p in m["parts"]]
        self.N = len(self.polys)
        cents = np.array([[p.centroid.x, p.centroid.y] for p in self.polys])
        # radius bound: farthest bounding-box corner from the centroid (works for any geometry type)
        self.rad = np.array([max(math.dist((c[0], c[1]), xy) for xy in
                                 ((p.bounds[0], p.bounds[1]), (p.bounds[0], p.bounds[3]),
                                  (p.bounds[2], p.bounds[1]), (p.bounds[2], p.bounds[3])))
                             for p, c in zip(self.polys, cents)])
        self.cents = cents
        self.tree = cKDTree(cents)
        self.maxrad = float(self.rad.max())

    def K(self, b, guard=True):
        return int(math.floor(b * self.area_km2 + (1e-9 if guard else 0.0)))


def in_core(x, y, bs, mode):
    for x0, y0, x1, y1 in bs:
        if mode == "halfopen" and x0 <= x < x1 and y0 <= y < y1:
            return True
        if mode == "closed" and x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def pool(region, objs, core_mode="halfopen"):
    out, outside, unknown = [], 0, 0
    for tid, lst in objs.items():
        bs = region.cores.get(tid)
        if bs is None:
            unknown += len(lst)
            continue
        for o in lst:
            x, y = float(o["easting"]), float(o["northing"])
            if core_mode != "none" and not in_core(x, y, bs, core_mode):
                outside += 1
                continue
            out.append(o)
    out.sort(key=lambda o: (-float(o["score"]), -float(o["mean_probability"]), -float(o["area_m2"]), str(o["id"])))
    return out, outside, unknown


def edges(region, props, strict=False):
    """adjacency list: proposal index -> list of entity indices within reach."""
    adj = []
    for o in props:
        x, y = float(o["easting"]), float(o["northing"])
        cand = region.tree.query_ball_point((x, y), REACH + region.maxrad + 1.0)
        pt = Point(x, y)
        hits = []
        for j in cand:
            d = region.polys[j].distance(pt)
            if (d < REACH) if strict else (d <= REACH):
                hits.append(j)
        adj.append(hits)
    return adj


def max_match(adj, k, N):
    rows, cols = [], []
    for i in range(min(k, len(adj))):
        for j in adj[i]:
            rows.append(i)
            cols.append(j)
    n = min(k, len(adj))
    if n == 0 or not rows:
        return 0
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, N))
    m = maximum_bipartite_matching(g, perm_type="column")
    return int((m >= 0).sum())


def nx_match(adj, k):
    G = nx.Graph()
    top = []
    for i in range(min(k, len(adj))):
        G.add_node(("p", i))
        top.append(("p", i))
        for j in adj[i]:
            G.add_edge(("p", i), ("e", j))
    if G.number_of_edges() == 0:
        return 0
    m = nx.bipartite.hopcroft_karp_matching(G, top_nodes=[n for n in top if G.degree(n) > 0])
    return sum(1 for a in m if a[0] == "p")


def tie_audit(props, k):
    """count proposals whose full sort key except id equals the K-th one (tie straddling the cut)."""
    if k <= 0 or k >= len(props):
        return 0
    key = lambda o: (float(o["score"]), float(o["mean_probability"]), float(o["area_m2"]))
    kk = key(props[k - 1])
    return sum(1 for o in props if key(o) == kk)


def score_system(region, path, xcheck=False):
    d = load(path)
    objs = d["objects"]
    res = {"file": path, "sha256": sha(path), "model": d.get("model")}
    props, outside, unknown = pool(region, objs, "halfopen")
    res["supply_saved"] = sum(len(v) for v in objs.values())
    res["supply_incore"] = len(props)
    res["outside_core_halfopen"] = outside
    res["unknown_tile"] = unknown
    adj = edges(region, props)
    res["C_pool"] = max_match(adj, len(adj), region.N)
    per = {}
    for b in BUDGETS:
        K = region.K(b)
        M = max_match(adj, K, region.N)
        per[str(b)] = {"K": K, "matches": M, "recall": M / region.N,
                       "ceiling": min(1.0, K / region.N),
                       "matches_per_100_slots": 100.0 * M / K,
                       "G_ranking": min(K, res["C_pool"]) - M,
                       "tie_group_at_cut": tie_audit(props, K)}
        if xcheck:
            per[str(b)]["networkx_matches"] = nx_match(adj, K)
    res["RAW"] = per
    # sensitivities
    adj_s = edges(region, props, strict=True)
    props_c, _, _ = pool(region, objs, "closed")
    props_n, _, _ = pool(region, objs, "none")
    adj_c = adj if len(props_c) == len(props) else edges(region, props_c)
    adj_n = adj if len(props_n) == len(props) else edges(region, props_n)
    sens = {}
    for b in BUDGETS:
        K = region.K(b)
        Kng = region.K(b, guard=False)
        sens[str(b)] = {
            "strict_lt15": max_match(adj_s, K, region.N),
            "closed_core": max_match(adj_c, K, region.N),
            "no_core_filter": max_match(adj_n, K, region.N),
            "K_without_guard": Kng,
            "K_sum_of_core_areas": int(math.floor(b * region.area_sum_km2 + 1e-9)),
        }
    res["sensitivity_matches"] = sens
    return res


def summarise(base, systems):
    out = {}
    for b in BUDGETS:
        bs = str(b)
        a = base["RAW"][bs]["recall"]
        diffs = [s["RAW"][bs]["recall"] - a for s in systems]
        out[bs] = {"base_recall": a,
                   "mean_recall": float(np.mean([s["RAW"][bs]["recall"] for s in systems])),
                   "mean_diff_points": 100 * float(np.mean(diffs)),
                   "diffs_points": [100 * x for x in diffs],
                   "n_positive": int(sum(1 for x in diffs if x > 0))}
    return out


def main():
    report = {"schema": "independent_rescore_v1", "rule": __doc__.strip().splitlines()[0]}
    # Verden
    V = Region(VERDEN["manifest"])
    report["verden_region"] = {"manifest_sha256": V.manifest_sha, "tiles": V.n_tiles, "N": V.N,
                               "area_km2_union": V.area_km2, "area_km2_sum": V.area_sum_km2,
                               "K": {str(b): V.K(b) for b in BUDGETS}}
    adaf_v = score_system(V, VERDEN["adaf"], xcheck=True)
    d1_v = [score_system(V, VERDEN["d1"].format(s=s), xcheck=(s == SEEDS[0])) for s in SEEDS]
    ctl_v = [score_system(V, VERDEN["ctl"].format(s=s)) for s in SEEDS]
    report["verden"] = {"ADAF_barrow": adaf_v, "D1": d1_v, "CTL": ctl_v}
    report["stage1"] = summarise(adaf_v, d1_v)
    st2 = {}
    for b in BUDGETS:
        bs = str(b)
        d = [100 * (x["RAW"][bs]["recall"] - c["RAW"][bs]["recall"]) for x, c in zip(d1_v, ctl_v)]
        dm = [x["RAW"][bs]["matches"] - c["RAW"][bs]["matches"] for x, c in zip(d1_v, ctl_v)]
        cv = [100 * (c["RAW"][bs]["recall"] - adaf_v["RAW"][bs]["recall"]) for c in ctl_v]
        st2[bs] = {"d1_minus_ctl_points": d, "d1_minus_ctl_matches": dm, "sum_d_matches": sum(dm),
                   "mean_points": float(np.mean(d)), "n_positive": sum(1 for x in dm if x > 0),
                   "ctl_minus_adaf_points": cv, "ctl_minus_adaf_mean_points": float(np.mean(cv))}
    report["stage2"] = st2
    # Oldenburg
    O = Region(OLDENBURG["manifest"])
    report["oldenburg_region"] = {"manifest_sha256": O.manifest_sha, "tiles": O.n_tiles, "N": O.N,
                                  "area_km2_union": O.area_km2, "area_km2_sum": O.area_sum_km2,
                                  "K": {str(b): O.K(b) for b in BUDGETS}}
    adaf_o = score_system(O, OLDENBURG["adaf"], xcheck=True)
    adaf_o_re = score_system(O, OLDENBURG["adaf_reinfer"])
    d1_o = [score_system(O, OLDENBURG["d1"].format(s=s), xcheck=(s == SEEDS[0])) for s in SEEDS]
    report["oldenburg"] = {"ADAF_barrow": adaf_o, "ADAF_barrow_reinferred_parity": adaf_o_re, "D1": d1_o}
    report["oldenburg_summary"] = summarise(adaf_o, d1_o)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    # brief console summary
    s1 = report["stage1"]["0.5"]
    print("Verden A=%.5f N=%d K=%s" % (V.area_km2, V.N, report["verden_region"]["K"]))
    for b in BUDGETS:
        s = report["stage1"][str(b)]
        print("S1 b=%s ADAF %.5f D1mean %.5f diff %+.3f pos %d/10" % (b, s["base_recall"], s["mean_recall"], s["mean_diff_points"], s["n_positive"]))
    print("S2 b=0.5 mean %+.3f pos %d/10 sum %d ctl-adaf %+.3f" % (st2["0.5"]["mean_points"], st2["0.5"]["n_positive"], st2["0.5"]["sum_d_matches"], st2["0.5"]["ctl_minus_adaf_mean_points"]))
    print("Oldenburg A=%.5f N=%d K=%s" % (O.area_km2, O.N, report["oldenburg_region"]["K"]))
    for b in BUDGETS:
        s = report["oldenburg_summary"][str(b)]
        print("OL b=%s ADAF %.5f (%d) D1mean %.5f diff %+.3f pos %d/10 run05 %+.3f" % (
            b, s["base_recall"], adaf_o["RAW"][str(b)]["matches"], s["mean_recall"], s["mean_diff_points"], s["n_positive"], s["diffs_points"][0]))


if __name__ == "__main__":
    sys.exit(main())
