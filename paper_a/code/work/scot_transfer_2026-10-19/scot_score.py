"""SCOT-TRANSFER-1 scoring (FREEZE.md). Landscape recall of HES Canmore small-upland-earthwork records by each arm's
ranked proposals, pooled over the frozen evaluation regions, with a 1 km cell bootstrap stratified by region.

Inputs (private dir outputs/scot_transfer_2026-10-19_PRIVATE/):
  regions/<R>/SITES.json        frozen eligible records: [{"id", "cls", "e", "n"}]            (written before inference)
  regions/<R>/PREP.json         {"eval_area_km2": float, ...}  valid DTM area of the cluster cores
  infer/<R>/PROPOSALS_<arm>_PRIVATE.json   {"objects": [{"id","easting","northing","score","mean_probability","area_m2"}]}
Run once after inference (WSL adaf-h2h-venv or any python with numpy):
  python -I scot_score.py <repo_root> A|B      # stage A or B; each stage scores once
  python -I scot_score.py --selftest
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGES = ("A", "B", "B_NOSTAB")    # SCORE_CONFIG_<stage>.json, frozen with FREEZE.md; B_NOSTAB applies only if
                                   # SMALL-FEAT-3 reports NO_STABLE_RECIPE (ARRAN_STAB_ENS3 dropped)
ORDER_PINS = ("work/scot_transfer_2026-10-19/scot_train.py", "work/scot_transfer_2026-10-19/TRAIN_CONFIG.json")
REACH_M = 15.0                     # primary: proposal centroid within 15 m of the Canmore point
REACH_SENS_M = (10.0, 25.0)        # sensitivity only
BUDGETS = (1.0, 2.0, 5.0, 10.0)    # proposals per km2 of evaluation area, per region
PRIMARY_B = 5.0
CELL_M = 1000.0                    # bootstrap unit: 1 km OS grid cell holding >= 1 eligible record
N_BOOT, SEED = 10_000, 20261019
COMPARATORS = ("ADAF_barrow", "ADAF_AO")


def sim_q(n_primary: int) -> tuple[float, float]:
    """Bonferroni over primary arms x {ADAF barrow, ADAF AO} x {matched workload, 5/km2}."""
    n = 2 * len(COMPARATORS) * n_primary
    return 100 * 0.025 / n, 100 * (1 - 0.025 / n)


def rank_key(p: dict) -> tuple:   # r3_curation.rank_key (frozen project order)
    return (-p["score"], -p["mean_probability"], -p["area_m2"], p["id"])


def edges(props: list[dict], sites: list[dict], reach: float) -> list[list[int]]:
    """For each proposal (in the given order), the indices of sites within reach (grid hash, 100 m cells)."""
    grid: dict[tuple, list[int]] = {}
    for j, s in enumerate(sites):
        for gx in range(int((s["e"] - reach) // 100), int((s["e"] + reach) // 100) + 1):
            for gy in range(int((s["n"] - reach) // 100), int((s["n"] + reach) // 100) + 1):
                grid.setdefault((gx, gy), []).append(j)
    out = []
    for p in props:
        cand = grid.get((int(p["easting"] // 100), int(p["northing"] // 100)), [])
        out.append(sorted({j for j in cand
                           if math.hypot(p["easting"] - sites[j]["e"], p["northing"] - sites[j]["n"]) <= reach}))
    return out


def max_matching(adj: list[list[int]], n_sites: int) -> set[int]:
    """Maximum-cardinality bipartite matching (Kuhn, iterative DFS); returns the matched site indices."""
    match_site = [-1] * n_sites
    for u in range(len(adj)):
        if not adj[u]:
            continue
        seen: set[int] = set()
        stack = [(u, iter(adj[u]))]
        path: list[tuple[int, int]] = []
        found = False
        while stack and not found:
            v, it = stack[-1]
            for s in it:
                if s in seen:
                    continue
                seen.add(s)
                path.append((v, s))
                if match_site[s] == -1:
                    found = True
                    break
                stack.append((match_site[s], iter(adj[match_site[s]])))
                break
            else:
                stack.pop()
                if path:
                    path.pop()
        if found:
            for v, s in path:
                match_site[s] = v
    return {s for s, v in enumerate(match_site) if v != -1}


def region_arm(props: list[dict], sites: list[dict], area: float, reach: float) -> dict:
    ranked = sorted(props, key=rank_key)
    adj = edges(ranked, sites, reach)
    S = len(sites)
    res: dict = {"pool": len(ranked), "pool_per_km2": len(ranked) / area, "budgets": {}, "matched": {}, "_adj": adj}
    for b in BUDGETS:
        n = int(round(b * area))
        take = min(n, len(ranked))
        m = max_matching(adj[:take], S)
        res["budgets"][str(b)] = {"N_budget": n, "N_taken": take, "matched": len(m), "recall": len(m) / S if S else None,
                                  "precision_lb": len(m) / take if take else None,     # one-to-one matched / taken
                                  "share_within_reach": sum(bool(a) for a in adj[:take]) / take if take else None}
        res["matched"][str(b)] = sorted(m)
    m = max_matching(adj, S)
    res["whole_pool"] = {"matched": len(m), "recall": len(m) / S if S else None,
                         "precision_lb": len(m) / len(ranked) if ranked else None}
    res["matched"]["pool"] = sorted(m)
    return res


def cell_units(sites: list[dict], offset: int) -> tuple[np.ndarray, int]:
    keys = sorted({(int(s["e"] // CELL_M), int(s["n"] // CELL_M)) for s in sites})
    idx = {k: i for i, k in enumerate(keys)}
    return np.array([offset + idx[(int(s["e"] // CELL_M), int(s["n"] // CELL_M))] for s in sites], dtype=int), len(keys)


def boot_weights(strata: list[tuple[int, int]], n_total: int) -> np.ndarray:
    """Stratified (by region) unit bootstrap: W[b, u] = times unit u is drawn in replicate b."""
    rng = np.random.default_rng(SEED)
    W = np.zeros((N_BOOT, n_total))
    for off, n in strata:
        draws = rng.integers(0, n, size=(N_BOOT, n))
        for b in range(N_BOOT):
            W[b, off:off + n] = np.bincount(draws[b], minlength=n)
    return W


def ci(d: np.ndarray, q: tuple) -> list[float]:
    return [float(np.percentile(d, q[0])), float(np.percentile(d, q[1]))]


def contrast(mx: set[int], my: set[int], unit: np.ndarray, W: np.ndarray, q: tuple) -> dict:
    n_units, S = W.shape[1], len(unit)
    s_t = np.bincount(unit, minlength=n_units).astype(float)
    x_t = np.bincount(unit[sorted(mx)], minlength=n_units).astype(float) if mx else np.zeros(n_units)
    y_t = np.bincount(unit[sorted(my)], minlength=n_units).astype(float) if my else np.zeros(n_units)
    d = (W @ x_t - W @ y_t) / (W @ s_t)
    return {"diff_pts": 100 * (len(mx) - len(my)) / S, "x": len(mx), "y": len(my),
            "ci95_pts": [100 * v for v in ci(d, (2.5, 97.5))], "ci_sim_pts": [100 * v for v in ci(d, q)],
            "x_only": len(mx - my), "y_only": len(my - mx), "both": len(mx & my)}


def pooled_sets(per_region: dict, arm: str, key: str, offsets: dict) -> set[int]:
    return {offsets[r] + j for r, R in per_region.items() for j in R[arm]["matched"][key]}


def matched_workload_sets(per_region: dict, x: str, y: str, offsets: dict) -> tuple[set, set, dict]:
    """Per region N_r = min(pool_x, pool_y); both arms take their top N_r; pooled over regions."""
    mx, my, ns = set(), set(), {}
    for r, R in per_region.items():
        n = min(R[x]["pool"], R[y]["pool"])
        S = R["_S"]
        mx |= {offsets[r] + j for j in max_matching(R[x]["_adj"][:n], S)}
        my |= {offsets[r] + j for j in max_matching(R[y]["_adj"][:n], S)}
        ns[r] = n
    return mx, my, ns


def verdict(rows: list[dict]) -> str:
    if all(r["ci_sim_pts"][0] > 0 for r in rows):
        return "BEATS_PUBLIC_ADAF"
    if any(r["ci_sim_pts"][1] < 0 for r in rows):
        return "LOSES_TO_PUBLIC_ADAF"
    return "NOT_DISTINGUISHED"


def analyse(regions: dict, arms: list[str], primary: list[str], reach: float) -> dict:
    """regions: {R: {"sites": [...], "area": km2, "props": {arm: [...]}}}. primary: the pre-declared transfer arms."""
    per_region, offsets, strata, units, off_s, off_u = {}, {}, [], [], 0, 0
    for r in sorted(regions):
        g = regions[r]
        per_region[r] = {a: region_arm(g["props"][a], g["sites"], g["area"], reach) for a in arms}
        per_region[r]["_S"] = len(g["sites"])
        offsets[r] = off_s
        u, n = cell_units(g["sites"], off_u)
        units.append(u)
        strata.append((off_u, n))
        off_s += len(g["sites"])
        off_u += n
    unit = np.concatenate(units)
    W = boot_weights(strata, off_u)
    q = sim_q(len(primary))
    pb = str(PRIMARY_B)
    family, pairs, verdicts = {}, {}, {}
    for x in primary:
        rows = []
        for y in COMPARATORS:
            mx, my, ns = matched_workload_sets(per_region, x, y, offsets)
            family[f"{x}-{y}@matched_workload"] = dict(contrast(mx, my, unit, W, q), N_per_region=ns)
            family[f"{x}-{y}@{pb}/km2"] = contrast(pooled_sets(per_region, x, pb, offsets),
                                                    pooled_sets(per_region, y, pb, offsets), unit, W, q)
            rows += [family[f"{x}-{y}@matched_workload"], family[f"{x}-{y}@{pb}/km2"]]
        verdicts[x] = verdict(rows)
    for x, y in itertools.combinations(arms, 2):
        if f"{x}-{y}@matched_workload" in family:
            continue
        pairs[f"{x}-{y}@{pb}/km2"] = contrast(pooled_sets(per_region, x, pb, offsets),
                                               pooled_sets(per_region, y, pb, offsets), unit, W, q)
        mx, my, ns = matched_workload_sets(per_region, x, y, offsets)
        pairs[f"{x}-{y}@matched_workload"] = dict(contrast(mx, my, unit, W, q), N_per_region=ns)
    region_rows = {}
    if len(regions) > 1:
        for r in per_region:
            sub = analyse({r: regions[r]}, arms, primary, reach)
            region_rows[r] = {"primary_family": sub["primary_family"], "verdicts": sub["verdicts"]}
    S_tot = len(unit)
    return {"S": S_tot, "n_units": off_u, "sim_quantiles": q, "primary_family": family, "verdicts": verdicts,
            "secondary_pairs": pairs,
            "arms": {r: {a: {k: v for k, v in R[a].items() if k not in ("_adj", "matched")} for a in arms}
                     | {"S": R["_S"]} for r, R in per_region.items()},
            "pooled_recall": {a: {b: sum(len(per_region[r][a]["matched"][b]) for r in per_region) / S_tot
                                  for b in [str(x) for x in BUDGETS] + ["pool"]} for a in arms},
            "per_region": region_rows,
            "_matched_primary": {a: {r: per_region[r][a]["matched"][pb] for r in per_region} for a in arms}}


def load(root: Path, cfg: dict) -> tuple[dict, dict]:
    priv = root / "outputs/scot_transfer_2026-10-19_PRIVATE"
    regions, hashes = {}, {}
    for r in cfg["regions"]:
        sp, pp = priv / f"regions/{r}/SITES.json", priv / f"regions/{r}/PREP.json"
        props = {}
        for a in cfg["arms"]:
            f = priv / f"infer/{r}/PROPOSALS_{a}_PRIVATE.json"
            props[a] = json.loads(f.read_text(encoding="utf-8"))["objects"]
            hashes[f"{r}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (sp, pp):
            hashes[f"{r}/{f.name}"] = hashlib.sha256(f.read_bytes()).hexdigest()
        regions[r] = {"sites": json.loads(sp.read_text(encoding="utf-8")),
                      "area": json.loads(pp.read_text(encoding="utf-8"))["eval_area_km2"], "props": props}
    return regions, hashes


def order_guard(root: Path) -> dict:
    """FREEZE.md section 6 order of operations: score only when the stage-B training code equals its section-11 pin."""
    fm = HERE / "FREEZE.md"
    text = fm.read_text(encoding="utf-8") if fm.exists() else ""
    out = {}
    for rel in ORDER_PINS:
        pin = next((ln.split("`")[3] for ln in text.splitlines() if ln.startswith(f"| `{rel}` |")), None)
        have = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if pin is None or pin != have:
            raise SystemExit(f"{rel}: sha {have} is not its FREEZE.md section-11 pin {pin}")
        out[rel] = have
    return out


def region_equal_weight(main: dict, regions: dict, cfg: dict) -> dict:
    """Descriptive: mean over regions of each primary contrast's per-region difference (each region weight 1/k)."""
    if not main["per_region"]:
        return {}
    out = {}
    for key in main["primary_family"]:
        vals = [main["per_region"][r]["primary_family"][key]["diff_pts"] for r in sorted(regions)]
        out[key] = {"mean_diff_pts": sum(vals) / len(vals), "per_region_diff_pts": dict(zip(sorted(regions), vals)),
                    "regions_positive": sum(v > 0 for v in vals), "regions_negative": sum(v < 0 for v in vals)}
    return out


def score(root: Path, stage: str) -> dict:
    if stage not in STAGES:
        raise SystemExit(f"stage must be one of {STAGES}")
    config = HERE / f"SCORE_CONFIG_{stage}.json"
    cfg = json.loads(config.read_text(encoding="utf-8"))
    order = order_guard(root)
    regions, hashes = load(root, cfg)
    if not all(isinstance(s.get("possible"), bool) for g in regions.values() for s in g["sites"]):
        raise SystemExit("every eligible record must carry a boolean 'possible'")
    keep = ("primary_family", "verdicts", "pooled_recall")
    main = analyse(regions, cfg["arms"], cfg["primary_arms"], REACH_M)
    sens = {str(rch): {k: v for k, v in analyse(regions, cfg["arms"], cfg["primary_arms"], rch).items()
                       if k in keep} for rch in REACH_SENS_M}
    definite = {r: dict(g, sites=[x for x in g["sites"] if not x["possible"]],
                        props=g["props"]) for r, g in regions.items()}
    sens_definite = {k: v for k, v in analyse(definite, cfg["arms"], cfg["primary_arms"], REACH_M).items() if k in keep}
    sens_definite["S"] = sum(len(g["sites"]) for g in definite.values())
    equal_w = region_equal_weight(main, regions, cfg)
    far = ("E2",)                     # >= 10 km from every training/validation tile (FREEZE.md section 6)
    split = {} if not stage.startswith("B") else {
        name: {k: v for k, v in analyse({r: g for r, g in regions.items() if (r in far) == want}, cfg["arms"],
                                        cfg["primary_arms"], REACH_M).items() if k in keep}
        for name, want in (("far_E2", True), ("near_E1_E3_E4", False))}
    classes = sorted({s["cls"] for g in regions.values() for s in g["sites"]})
    per_class = {}
    for a in cfg["arms"]:
        per_class[a] = {}
        for c in classes:
            tot = sum(1 for g in regions.values() for s in g["sites"] if s["cls"] == c)
            hit = sum(1 for r, g in regions.items() for j in main["_matched_primary"][a][r] if g["sites"][j]["cls"] == c)
            per_class[a][c] = {"matched": hit, "of": tot}
    matched_ids = {a: {r: [regions[r]["sites"][j]["id"] for j in main["_matched_primary"][a][r]] for r in regions}
                   for a in cfg["arms"]}
    out = {"stage": stage, "config": cfg, "reach_m": REACH_M, "primary_budget_per_km2": PRIMARY_B, "budgets": BUDGETS,
           "n_boot": N_BOOT, "seed": SEED, "cell_m": CELL_M,
           "eval_area_km2": {r: g["area"] for r, g in regions.items()},
           "sites_by_region": {r: len(g["sites"]) for r, g in regions.items()},
           **{k: v for k, v in main.items() if not k.startswith("_")},
           "per_class_primary_budget": per_class, "sensitivity_reach": sens,
           "sensitivity_without_possible": sens_definite, "region_equal_weight": equal_w, "order_guard": order,
           "stage_b_distance_split": split,
           "matched_site_ids_primary_budget": matched_ids, "inputs_sha256": hashes,
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
    dest = root / f"outputs/scot_transfer_2026-10-19_PRIVATE/RESULT_SCORE_{stage}.json"
    if dest.exists():
        raise SystemExit(f"{dest.name} exists: each frozen stage scores once")
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    return out


def selftest() -> None:
    sites = [{"id": f"s{i}", "cls": "c", "e": 1000.0 * i + 500, "n": 500.0} for i in range(4)]
    P = lambda i, e, n, sc: {"id": f"p{i}", "easting": e, "northing": n, "score": sc,  # noqa: E731
                             "mean_probability": sc, "area_m2": 40.0}
    props = [P(0, 505, 505, .9), P(1, 510, 490, .8), P(2, 1514, 500, .7), P(3, 9000, 9000, .95), P(4, 2500, 516, .6)]
    adj = edges(sorted(props, key=rank_key), sites, REACH_M)
    assert adj == [[], [0], [0], [1], []], adj        # p3 background; p2 14 m from s1; p4 16 m from s2 (out of reach)
    assert len(max_matching(adj, 4)) == 2
    assert len(max_matching([[0, 1], [0]], 2)) == 2   # augmenting path
    rng = np.random.default_rng(1)
    for _ in range(300):                              # Kuhn vs brute force
        nu, ns = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        g = [sorted(set(rng.integers(0, ns, size=int(rng.integers(0, 3))).tolist())) for _ in range(nu)]
        best = 0
        for choice in itertools.product(*[[-1] + a for a in g]):
            used = [c for c in choice if c >= 0]
            if len(used) == len(set(used)):
                best = max(best, len(used))
        assert len(max_matching(g, ns)) == best, (g, best)
    m = region_arm(props, sites, 1.0, REACH_M)        # budgets 1,2,5,10 per km2 on 1 km2 -> N = 1, 2, 5, 10
    assert m["budgets"]["1.0"]["matched"] == 0 and m["budgets"]["2.0"]["matched"] == 1
    assert m["budgets"]["5.0"]["matched"] == 2 and m["whole_pool"]["matched"] == 2
    # two regions, arm A finds everything, arm B nothing -> diff +100 pts, CI degenerate at +100
    regs = {}
    for r in ("R1", "R2"):
        st = [{"id": f"{r}s{i}", "cls": "c", "e": 1000.0 * i + 500, "n": 500.0} for i in range(3)]
        good = [P(i, s["e"], s["n"], 1 - i / 10) for i, s in enumerate(st)]
        bad = [P(i, 50_000 + i, 50_000, .5) for i in range(3)]
        regs[r] = {"sites": st, "area": 1.0, "props": {"A": good, "B": bad, "ADAF_barrow": bad, "ADAF_AO": bad}}
    global N_BOOT
    keep, N_BOOT = N_BOOT, 200
    try:
        res = analyse(regs, ["A", "B", "ADAF_barrow", "ADAF_AO"], ["A", "B"], REACH_M)
    finally:
        N_BOOT = keep
    fam = res["primary_family"]
    assert res["verdicts"] == {"A": "BEATS_PUBLIC_ADAF", "B": "NOT_DISTINGUISHED"}, res["verdicts"]
    assert all(abs(v["diff_pts"] - 100) < 1e-9 for k, v in fam.items() if k.startswith("A-"))
    assert res["sim_quantiles"] == (0.3125, 99.6875)
    assert fam["A-ADAF_barrow@matched_workload"]["N_per_region"] == {"R1": 3, "R2": 3}
    assert res["pooled_recall"]["A"]["5.0"] == 1.0 and res["S"] == 6 and res["n_units"] == 6
    # matched workload truncates to the smaller pool
    regs["R1"]["props"]["ADAF_barrow"] = bad[:1]
    mx, my, ns = matched_workload_sets({r: {a: region_arm(regs[r]["props"][a], regs[r]["sites"], 1.0, REACH_M)
                                            for a in ("A", "ADAF_barrow")} | {"_S": 3} for r in regs},
                                       "A", "ADAF_barrow", {"R1": 0, "R2": 3})
    assert ns == {"R1": 1, "R2": 3} and len(mx) == 4 and not my, (ns, mx, my)
    print("selftest PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        r = score(Path(sys.argv[1]).resolve(), sys.argv[2])
        print(json.dumps({k: r[k] for k in ("verdicts", "primary_family", "pooled_recall", "sites_by_region")}, indent=1))
