"""CONFIRM-D statistics (DRAFT_FREEZE §6-§7): thin wrappers around the UNCHANGED cr1_analysis ratio estimator.

  fixed_contrast(x, y, pop)            A1/A2: frozen systems, one hit set each; se2_core only, no seed term
  recipe_contrast(xs, ys, pop, floor)  B: K seed pairs; floor "scaled" = max(obs, 2 * (1.3 pts)^2 / K)
                                       (POWER.md §2.3) or "constant" = cr1_analysis.SEED_FLOOR (the DT1 rule)
  block_contrast(x, y, pop, block_m)   descriptive sensitivity: 10 km block clusters instead of 5 km cores
  guard_label / a1_label               §7 labels
At K = 3 the scaled floor equals the constant floor (2 * 0.013^2 / 3 == SEED_FLOOR), so DT1 numbers are reproduced.

  python -B cd_analysis.py selftest        # synthetic checks
  python -B cd_analysis.py selftest dt1    # + reproduce DT1 D-R0 contrasts from its private hits (WSL venv)
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "work/context_rescue1_2026-10-23"))
import cr1_analysis as AN  # noqa: E402

Z = AN.Z
GUARD = -0.02
SEED_SD_PTS = 1.3


def _interval(c: dict, var_seed: float, note: str) -> dict:
    hw = Z * math.sqrt(c["se2_core"] + var_seed)
    return dict(c, var_seed_used=var_seed, seed_term=note, lower=c["delta"] - hw, upper=c["delta"] + hw,
                se=math.sqrt(c["se2_core"] + var_seed), delta_points=100 * c["delta"],
                lower_points=100 * (c["delta"] - hw), upper_points=100 * (c["delta"] + hw))


def fixed_contrast(x: set, y: set, pop: list[dict]) -> dict:
    """Frozen weights: geography is the only random factor, so no seed term (bypasses only SEED_FLOOR)."""
    return _interval(AN.contrast([x], [y], pop), 0.0, "none (fixed systems)")


def recipe_contrast(xs: list[set], ys: list[set], pop: list[dict], floor: str = "scaled") -> dict:
    c = AN.contrast(xs, ys, pop)
    K = c["seeds"]
    if floor == "constant":
        return _interval(c, max(c["var_seed_observed"], AN.SEED_FLOOR), "constant SEED_FLOOR (DT1 rule)")
    if floor != "scaled":
        raise ValueError(floor)
    return _interval(c, max(c["var_seed_observed"], 2 * (SEED_SD_PTS / 100) ** 2 / K), f"scaled floor, K={K}")


def block_contrast(x: set, y: set, pop: list[dict], block_m: int = 10_000) -> dict:
    """Same estimator with records re-clustered into block_m squares (needs record x, y in metres)."""
    re = [dict(r, core=(int(r["x"] // block_m), int(r["y"] // block_m))) for r in pop]
    return dict(fixed_contrast(x, y, re), cluster=f"{block_m // 1000} km blocks")


def guard_label(c: dict) -> str:
    return "PASS" if c["lower"] > GUARD else "HARM" if c["upper"] < GUARD else "UNRESOLVED"


def a1_label(primary: dict, guards: dict[str, dict], implementation_ok: bool = True) -> str:
    if not implementation_ok:
        return "INCONCLUSIVE_IMPLEMENTATION"
    if primary["upper"] < 0:
        return "REVERSED"
    if primary["lower"] <= 0:
        return "NOT_DISTINGUISHED"
    labels = [guard_label(g) for g in guards.values()]
    if "HARM" in labels:
        return "PASS_GUARD_HARM"
    return "CONFIRMED" if all(lb == "PASS" for lb in labels) else "CONFIRMED_GUARD_UNRESOLVED"


def _synthetic() -> dict:
    pop = [{"id": f"r{i}", "core": (i % 7, 0), "x": (i % 7) * 5000 + 10, "y": 10} for i in range(140)]
    x = {r["id"] for r in pop[:80]}
    y = {r["id"] for r in pop[:70]}
    f = fixed_contrast(x, y, pop)
    raw = AN.contrast([x], [y], pop)
    k3 = recipe_contrast([x, x, x], [y, y, y], pop, "scaled")
    k9 = recipe_contrast([x] * 9, [y] * 9, pop, "scaled")
    con = recipe_contrast([x] * 9, [y] * 9, pop, "constant")
    blk = block_contrast(x, y, pop, 10_000)
    return {
        "fixed_no_seed_term": f["var_seed_used"] == 0 and f["lower"] > raw["lower"] and abs(f["delta"] - 10 / 140) < 1e-12,
        "fixed_uses_unchanged_se2": f["se2_core"] == raw["se2_core"],
        "scaled_equals_constant_at_K3": abs(k3["var_seed_used"] - AN.SEED_FLOOR) < 1e-18,
        "scaled_shrinks_with_K": abs(k9["var_seed_used"] - AN.SEED_FLOOR / 3) < 1e-18 and k9["upper"] < con["upper"],
        "block_reclusters": blk["cores"] == 4 and blk["delta"] == f["delta"],
        "labels": (a1_label({"lower": 0.01, "upper": 0.03}, {"g": {"lower": -0.01, "upper": 0.02}}) == "CONFIRMED"
                   and a1_label({"lower": 0.01, "upper": 0.03}, {"g": {"lower": -0.03, "upper": 0.0}})
                   == "CONFIRMED_GUARD_UNRESOLVED"
                   and a1_label({"lower": -0.01, "upper": 0.03}, {}) == "NOT_DISTINGUISHED"
                   and a1_label({"lower": -0.03, "upper": -0.01}, {}) == "REVERSED"),
    }


def _dt1() -> dict:
    """Recompute DT1's recorded D+PROX - R0+PROX and D - R0|plain (all records) from its private hit sets."""
    sys.path.insert(0, str(ROOT / "work/context_rescue1_2026-10-23/eval_score"))
    import cr1_eval_score as CE
    hp = ROOT / "outputs/dense_terrain1_2026-10-25_PRIVATE/score/FULL_SCORES_PRIVATE.json"
    res = json.loads((ROOT / "work/dense_terrain1_2026-10-25/RESULT_FULL.json").read_text(encoding="utf-8"))
    ok_sha = hashlib.sha256(hp.read_bytes()).hexdigest() == res["private_hits_sha256"]
    hits = {k: set(v) for k, v in json.loads(hp.read_text(encoding="utf-8"))["hits"].items()}
    pop = CE.load_eval_base()["records"]
    out = {"hits_sha_matches_result": ok_sha}
    for key, v in (("D+PROX-R0+PROX", "prox"), ("D-R0|plain", "plain")):
        c = recipe_contrast([hits[f"D_{s}|{v}"] for s in (101, 202, 303)],
                            [hits[f"D1_{s}|{v}"] for s in ("05", "06", "07")], pop, "constant")
        want = res["contrasts"][key]["all"]
        out[key] = all(abs(c[f] - want[f]) < 1e-12 for f in ("delta", "lower", "upper", "se2_core", "var_seed_used"))
    return out


def selftest(dt1: bool = False) -> dict:
    checks = _synthetic()
    if dt1:
        checks.update(_dt1())
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        r = selftest("dt1" in sys.argv[2:])
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["status"] == "PASS" else 1)
