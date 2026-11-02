"""CONTEXT-RESCUE-1 shared evaluation definition, part 3: strata, contrasts and decision rules (FREEZE.md §5-§7).

Strata (truth records of EVAL_TRUTH_PRIVATE.json; H from step E0, T = T_FAINT):
  grouped          member of a census complex (record "complex" is not None)
  faint_grouped    grouped and T <= H < 2T                    (the primary population)
  strong_grouped   grouped and H >= 2T
  isolated         statewide 200 m group of fewer than 3 records
  all              every truth record
Contrast X - Y on a population P, with K paired seeds (X_k, Y_k):
  delta_k = (hits of X_k in P - hits of Y_k in P) / N;  delta = mean_k delta_k
  D_j = mean_k (hits of X_k - hits of Y_k) in core j;  J = cores with n_j > 0
  se2_core = J / (J - 1) * sum_j (D_j - n_j * delta)^2 / N^2           (ratio estimator, as power.py)
  var_seed = max(var(delta_k, ddof = 1) / K, SEED_FLOOR)              (floor = power.py's seed term)
  95 % interval: delta +- 1.96 * sqrt(se2_core + var_seed)
Primary passes when the lower bound > 0; a guard passes when the lower bound > -0.02.

  python -B cr1_analysis.py selftest
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable

T_FAINT = 0.3290754481743007       # CONTEXT-RESCUE-0 FAINT_B: 95th percentile of control dome height (m)
SEED_FLOOR = (1.3 / 100) ** 2 * 2 / 3
Z = 1.96
GUARD_MARGIN = -0.02
SELECT_TIE = 0.005
PARITY_TOL = 0.02
DEV_FLOOR = 0.10


def strata(records: list[dict], H: dict[str, float | None]) -> dict[str, list[dict]]:
    T = T_FAINT
    g = [r for r in records if r["complex"] is not None]
    h = lambda r: H.get(r["id"])  # noqa: E731
    return {"grouped": g,
            "faint_grouped": [r for r in g if h(r) is not None and T <= h(r) < 2 * T],
            "strong_grouped": [r for r in g if h(r) is not None and h(r) >= 2 * T],
            "isolated": [r for r in records if r["group_size_statewide"] < 3],
            "all": list(records)}


def contrast(x_hits: list[Iterable[str]], y_hits: list[Iterable[str]], pop: list[dict]) -> dict:
    """x_hits[k], y_hits[k]: matched record ids of seed pair k (paired by seed)."""
    K = len(x_hits)
    if K != len(y_hits) or K < 1:
        raise ValueError("seed pairs must match")
    cores: dict[tuple, int] = {}
    for r in pop:
        cores[tuple(r["core"])] = cores.get(tuple(r["core"]), 0) + 1
    N, J = len(pop), len(cores)
    D = {c: 0.0 for c in cores}
    deltas = []
    for xs, ys in zip(x_hits, y_hits):
        xs, ys = set(xs), set(ys)
        d = 0
        for r in pop:
            v = (r["id"] in xs) - (r["id"] in ys)
            D[tuple(r["core"])] += v / K
            d += v
        deltas.append(d / N)
    delta = sum(deltas) / K
    se2 = J / (J - 1) * sum((D[c] - n * delta) ** 2 for c, n in cores.items()) / N ** 2 if J > 1 else float("inf")
    obs_seed = (sum((d - delta) ** 2 for d in deltas) / (K - 1) / K) if K > 1 else 0.0
    var_seed = max(obs_seed, SEED_FLOOR)
    hw = Z * math.sqrt(se2 + var_seed)
    return {"n": N, "cores": J, "seeds": K, "delta": delta, "delta_by_seed": deltas, "se2_core": se2,
            "var_seed_observed": obs_seed, "var_seed_used": var_seed, "lower": delta - hw, "upper": delta + hw}


def primary_pass(c: dict) -> bool:
    return c["lower"] > 0


def guard_pass(c: dict) -> bool:
    return c["lower"] > GUARD_MARGIN


def select_context_arm(recall_b: list[float], recall_c: list[float]) -> str:
    """Development selection: higher mean grouped recall over the seeds; within 0.5 points, B (the simpler arm)."""
    mb, mc = sum(recall_b) / len(recall_b), sum(recall_c) / len(recall_c)
    return "C" if mc - mb >= SELECT_TIE else "B"


def parity_ok(recall_a05: float, recall_d1run05: float) -> bool:
    """Trainer parity: arm A seed 05 within 2 points of D1 run 05 on the development cores."""
    return abs(recall_a05 - recall_d1run05) <= PARITY_TOL


def dev_gate_ok(recall_arm: float, recall_a: float, collapsed: bool) -> bool:
    """After seed 05: a context arm continues unless collapsed (F1 definition) or > 10 points below A."""
    return not collapsed and recall_arm >= recall_a - DEV_FLOOR


def selftest() -> dict:
    recs = [{"id": f"r{k}", "core": [k % 4, 0], "complex": 0 if k < 30 else None,
             "group_size_statewide": 5 if k < 30 else 1} for k in range(40)]
    H = {f"r{k}": (0.4 if k < 20 else 0.9 if k < 30 else 0.1) for k in range(40)}
    st = strata(recs, H)
    checks = {"strata_sizes": [len(st[k]) for k in ("grouped", "faint_grouped", "strong_grouped", "isolated", "all")]
              == [30, 20, 10, 10, 40]}
    same = [{f"r{k}" for k in range(0, 20, 2)}] * 3
    c0 = contrast(same, same, st["faint_grouped"])
    checks["identical_arms_zero"] = c0["delta"] == 0 and c0["var_seed_used"] == SEED_FLOOR and not primary_pass(c0)
    x = [{f"r{k}" for k in range(20)}] * 3              # X hits all 20 faint records, Y hits 10
    c1 = contrast(x, same, st["faint_grouped"])
    # cores: r0..r19 -> core k % 4: 5 records each; Y hits even k: cores 0 and 2 have 5 hits, 1 and 3 none
    D = {0: 0, 1: 5, 2: 0, 3: 5}
    se2 = 4 / 3 * sum((D[c] - 5 * 0.5) ** 2 for c in D) / 400
    checks["contrast_by_hand"] = abs(c1["delta"] - 0.5) < 1e-12 and abs(c1["se2_core"] - se2) < 1e-12
    checks["seed_floor_applies"] = c1["var_seed_used"] == SEED_FLOOR
    xs = [{f"r{k}" for k in range(20)}, {f"r{k}" for k in range(16)}, {f"r{k}" for k in range(12)}]
    c2 = contrast(xs, same, st["faint_grouped"])
    checks["observed_seed_var_used"] = c2["var_seed_used"] == c2["var_seed_observed"] > SEED_FLOOR
    checks["rules"] = (select_context_arm([0.50, 0.52], [0.504, 0.51]) == "B"
                       and select_context_arm([0.50, 0.52], [0.52, 0.53]) == "C"
                       and parity_ok(0.70, 0.715) and not parity_ok(0.70, 0.73)
                       and dev_gate_ok(0.61, 0.70, False) and not dev_gate_ok(0.59, 0.70, False)
                       and not dev_gate_ok(0.80, 0.70, True))
    checks["guard_margin"] = guard_pass({"lower": -0.019}) and not guard_pass({"lower": -0.021})
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        print(json.dumps(selftest(), indent=1))
