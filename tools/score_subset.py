"""Score proposal files on a subset of tiles (or on the whole region) with the released independent scorer.

`work/independent_rescore_2026-10-09/rescore.py` implements the benchmark-v0 matching rule, but its main() reads the
original run's proposal files at fixed repository paths, and those files are not released. This tool imports its
functions (Region, in_core, edges, max_match, nx_match, tie_audit) unchanged and applies them to proposal files you
name, for example proposals regenerated with infer_irl.py.

Rule (rescore.py, reading (i)): entities = every part of the individual manifest; a proposal counts when its centroid
lies in its own tile's core (half-open box test); RAW order = score desc, mean_probability desc, area_m2 desc, id asc;
edge = proposal within 15 m of the entity footprint (inclusive); matches = maximum-cardinality bipartite matching of
the top K, K = floor(b * A_km2 + 1e-9). With --tiles, only proposals on those tiles are pooled and A is the union of
their core boxes; matching is still against every entity of the manifest. Without --tiles, every manifest tile is
used and the result equals rescore.py's RAW row for that file.

  python -B tools/score_subset.py --manifest outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json \
      --proposals regen=DEST_DIR/PROPOSALS_<NAME>_PRIVATE.json --tiles VER-P-000 VER-P-007 --out SCORE.json

Output (stdout and --out): aggregates only (counts, K, matches, matches per tile id); no coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from shapely.geometry import box
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "work/independent_rescore_2026-10-09"))
import rescore as R  # noqa: E402  released independent scorer; functions only, main() is not called


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_tiles(tiles: list[str], tiles_file: str | None) -> list[str]:
    out = list(tiles or [])
    if tiles_file:
        out += [ln.strip() for ln in Path(tiles_file).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    return list(dict.fromkeys(out))


def tagged_pool(region: R.Region, objs: dict, tiles: set[str]) -> tuple[list[tuple[str, dict]], int]:
    """rescore.pool (half-open core test, RAW order) restricted to `tiles`, keeping each proposal's tile id."""
    out, outside = [], 0
    for tid, lst in objs.items():
        if tid not in tiles:
            continue
        bs = region.cores[tid]
        for o in lst:
            if R.in_core(float(o["easting"]), float(o["northing"]), bs, "halfopen"):
                out.append((tid, o))
            else:
                outside += 1
    out.sort(key=lambda r: (-float(r[1]["score"]), -float(r[1]["mean_probability"]),
                            -float(r[1]["area_m2"]), str(r[1]["id"])))
    return out, outside


def match_top(region: R.Region, tagged: list[tuple[str, dict]], adj: list[list[int]], k: int) -> tuple[int, dict]:
    """maximum matching of the top-k proposals; per-tile counts of matched proposals (cross-checked with rescore)."""
    n = min(k, len(adj))
    rows = [i for i in range(n) for _ in adj[i]]
    cols = [j for i in range(n) for j in adj[i]]
    per: dict[str, int] = {}
    if n == 0 or not rows:
        return 0, per
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, region.N))
    m = maximum_bipartite_matching(g, perm_type="column")
    for i in range(n):
        if m[i] >= 0:
            per[tagged[i][0]] = per.get(tagged[i][0], 0) + 1
    total = int((m >= 0).sum())
    if total != R.max_match(adj, k, region.N):
        raise SystemExit("matching cross-check against rescore.max_match failed")
    return total, per


def score_file(region: R.Region, path: Path, tiles: list[str], budgets: list[float], area_km2: float,
               xcheck: bool) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    objs = d["objects"]
    absent = [t for t in tiles if t not in objs]
    if absent:
        raise SystemExit(f"{path.name}: {len(absent)} selected tile(s) have no entry in 'objects', e.g. {absent[:3]}")
    tagged, outside = tagged_pool(region, objs, set(tiles))
    adj = R.edges(region, [o for _, o in tagged])
    res = {"sha256": sha256(path), "model": d.get("model"), "checkpoint_sha256": d.get("checkpoint_sha256"),
           "supply_selected_tiles": sum(len(objs[t]) for t in tiles), "supply_incore": len(tagged),
           "outside_core_halfopen": outside}
    pool_total, pool_per = match_top(region, tagged, adj, len(adj))
    res["pool_matches"] = pool_total
    res["pool_matches_per_tile"] = {t: pool_per.get(t, 0) for t in tiles}
    props = [o for _, o in tagged]
    res["budgets"] = {}
    for b in budgets:
        k = int(math.floor(b * area_km2 + 1e-9))
        tot, per = match_top(region, tagged, adj, k)
        row = {"K": k, "matches": tot, "per_tile": {t: per.get(t, 0) for t in tiles},
               "tie_group_at_cut": R.tie_audit(props, k)}
        if xcheck:
            row["networkx_matches"] = R.nx_match(adj, k)
        res["budgets"][str(b)] = row
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="individual (entity) manifest, e.g. step0/INDIVIDUAL_MANIFEST_PRIVATE.json")
    ap.add_argument("--proposals", required=True, nargs="+", metavar="LABEL=PATH",
                    help="one or more proposal files ({'objects': {tile id: [...]}}), each with a label")
    ap.add_argument("--tiles", nargs="*", default=[], help="tile ids to score (default: every manifest tile)")
    ap.add_argument("--tiles-file", help="text file with one tile id per line")
    ap.add_argument("--budgets", nargs="+", type=float, default=list(R.BUDGETS), help="budgets b (proposals per km2)")
    ap.add_argument("--xcheck", action="store_true", help="also cross-check every budget with networkx Hopcroft-Karp")
    ap.add_argument("--out", help="write the aggregate result as JSON here")
    a = ap.parse_args()

    manifest = Path(a.manifest).resolve()
    region = R.Region(str(manifest))
    tiles = read_tiles(a.tiles, a.tiles_file) or sorted(region.cores)
    unknown = [t for t in tiles if t not in region.cores]
    if unknown:
        raise SystemExit(f"{len(unknown)} tile id(s) not in the manifest, e.g. {unknown[:3]}")
    union = unary_union([box(*b) for t in tiles for b in region.cores[t]])
    area_km2 = union.area / 1e6
    report = {"schema": "score_subset_v1", "scorer": "work/independent_rescore_2026-10-09/rescore.py",
              "scorer_sha256": sha256(Path(R.__file__)), "manifest_sha256": region.manifest_sha,
              "entities_N": region.N, "tiles": tiles, "n_tiles": len(tiles),
              "whole_region": len(tiles) == region.n_tiles, "area_km2_union": area_km2,
              "budgets": a.budgets, "files": {}}
    for spec in a.proposals:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--proposals expects LABEL=PATH, got {spec!r}")
        report["files"][label] = score_file(region, Path(path), tiles, a.budgets, area_km2, a.xcheck)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    summary = {lab: {"pool_matches": r["pool_matches"],
                     **{f"b={b}": (row["K"], row["matches"]) for b, row in r["budgets"].items()}}
               for lab, r in report["files"].items()}
    print(json.dumps({"n_tiles": len(tiles), "area_km2_union": round(area_km2, 6), "files": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
