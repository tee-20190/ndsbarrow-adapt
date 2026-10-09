"""Stage H scorer (PREREG.md): recall of holdout positive parts at fixed negative-window proposal density.

  python -B score_holdout.py            # reads outputs/tta_ensemble_2026-09-24/HOLDOUT_OBJECTS_PRIVATE.json
  python -B score_holdout.py recorded   # self-check: recorded R2 holdout objects scored as arm E38
(WSL adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
sys.path.insert(0, str(ROOT / "work/next_experiment_2026-09-23"))
import r2_masks  # noqa: E402
import r3_curation as cur  # noqa: E402

OUT = ROOT / "outputs/tta_ensemble_2026-09-24"
REACH_M = 15.0
DENSITIES = (0.5, 1.0, 2.0)
DESCRIPTIVE_DENSITIES = (4.0,)
GATE = {"min_gain": 0.010, "max_loss_any_density": 0.010}
CANDIDATES = ("E39", "T4", "C2", "C2T4")
PASSES = {"E38": 1, "E39": 1, "C2": 2, "T4": 4, "C2T4": 8}
BOOT_N, BOOT_SEED = 2000, 20260924


def read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def setup() -> dict:
    manifest = read(ROOT / "work/campaign_2026-09-22/R2_MASKS_MANIFEST_PRIVATE.json")
    wins = [w for w in manifest["windows"] if w["holdout"]]
    parts = {q["id"]: q for q in r2_masks.rebuild_parts()[0]}
    home, neg_area = {}, {}
    for wi, w in enumerate(wins):
        x0, y0, x1, y1 = cur.window_bounds(w)
        inner = (x0 + cur.EDGE_M, y0 + cur.EDGE_M, x1 - cur.EDGE_M, y1 - cur.EDGE_M)
        if w["role"] == "negative":
            neg_area[wi] = (inner[2] - inner[0]) * (inner[3] - inner[1]) / 1e6
        for p in w["positive_parts"]:
            q = parts[p["id"]]
            if q["role"] == "positive" and inner[0] <= q["cx"] < inner[2] and inner[1] <= q["cy"] < inner[3]:
                home.setdefault(q["id"], wi)
    eligible = sorted(home)
    klass = {pid: ("enclosure" if {"enclosure", "hillfort"} & set(parts[pid]["classes"]) else "barrow")
             for pid in eligible}
    tree = STRtree([parts[pid]["geom"].buffer(REACH_M) for pid in eligible])
    return {"wins": wins, "index": {w["id"]: i for i, w in enumerate(wins)}, "eligible": eligible, "home": home,
            "klass": klass, "tree": tree, "neg_area": neg_area}


def ranked_pool(objects: dict[str, list[dict]], s: dict) -> list[dict]:
    rows = []
    for w in s["wins"]:
        rows += [dict(o, window=w["id"]) for o in objects[w["id"]] if cur.inner(o, w)]
    pool = sorted(cur.nms(rows), key=cur.rank_key)
    for r in pool:
        wi = s["index"][r["window"]]
        r["wi"] = wi
        r["neg"] = wi in s["neg_area"]
        r["hits"] = [s["eligible"][i] for i in s["tree"].query(Point(r["easting"], r["northing"]), predicate="intersects")]
    return pool


def recall_at(pool: list[dict], s: dict, d: float | None, wmult: np.ndarray | None = None) -> tuple[float, set]:
    """Recall (weighted by home-window multiplicity) above the first negative proposal beyond floor(d x A)."""
    w = np.ones(len(s["wins"])) if wmult is None else wmult
    area = sum(w[i] * a for i, a in s["neg_area"].items())
    allow = math.inf if d is None else math.floor(d * area + 1e-9)
    neg, hit = 0.0, set()
    for r in pool:
        m = w[r["wi"]]
        if m == 0:
            continue
        if r["neg"]:
            if neg + m > allow:
                break
            neg += m
        hit.update(r["hits"])
    den = sum(w[s["home"][p]] for p in s["eligible"])
    return sum(w[s["home"][p]] for p in hit) / den, hit


def arm_summary(pool: list[dict], s: dict) -> dict:
    out = {"pool": len(pool), "negative_proposals": sum(r["neg"] for r in pool),
           "negative_per_km2": sum(r["neg"] for r in pool) / sum(s["neg_area"].values())}
    for d in DENSITIES + DESCRIPTIVE_DENSITIES + (None,):
        rec, hit = recall_at(pool, s, d)
        key = "all" if d is None else f"d{d}"
        out[key] = {"recall": rec, "parts": len(hit),
                    "by_class": {k: sum(s["klass"][p] == k for p in hit) for k in ("barrow", "enclosure")}}
    out["primary"] = float(np.mean([out[f"d{d}"]["recall"] for d in DENSITIES]))
    return out


def bootstrap(pools: dict[str, list[dict]], s: dict) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    pos = [i for i in range(len(s["wins"])) if i not in s["neg_area"]]
    neg = sorted(s["neg_area"])
    diffs = {a: [] for a in pools if a != "E38"}
    for _ in range(BOOT_N):
        w = np.zeros(len(s["wins"]))
        np.add.at(w, rng.choice(pos, len(pos)), 1)
        np.add.at(w, rng.choice(neg, len(neg)), 1)
        prim = {a: np.mean([recall_at(p, s, d, w)[0] for d in DENSITIES]) for a, p in pools.items()}
        for a in diffs:
            diffs[a].append(prim[a] - prim["E38"])
    return {a: {"median": float(np.median(v)), "q05": float(np.quantile(v, .05)), "q95": float(np.quantile(v, .95)),
                "share_gt_0": float(np.mean(np.array(v) > 0))} for a, v in diffs.items()}


def decide(summ: dict) -> dict:
    base = summ["E38"]
    ok = []
    for a in CANDIDATES:
        gain = summ[a]["primary"] - base["primary"]
        worst = min(summ[a][f"d{d}"]["recall"] - base[f"d{d}"]["recall"] for d in DENSITIES)
        if gain >= GATE["min_gain"] - 1e-12 and worst >= -GATE["max_loss_any_density"] - 1e-12:
            ok.append((summ[a]["primary"], -PASSES[a], a))
    best = max(((summ[a]["primary"], -PASSES[a], a) for a in CANDIDATES))[2]
    sel = max(ok)[2] if ok else None
    return {"best_candidate": best, "qualifying": [a for _, _, a in sorted(ok, reverse=True)], "selected": sel,
            "outcome": f"PROCEED_SCREEN_{sel}" if sel else "STOP_NO_ARM", "gate": GATE}


def main() -> None:
    s = setup()
    if len(sys.argv) > 1 and sys.argv[1] == "recorded":
        objs = read(ROOT / "outputs/campaign_2026-09-22/holdout_diag/R2_holdout/OBJECTS_PRIVATE.json")["objects"]
        print(json.dumps({"eligible_parts": len(s["eligible"]), "E38_recorded": arm_summary(ranked_pool(objs, s), s)}))
        return
    rec = read(OUT / "HOLDOUT_OBJECTS_PRIVATE.json")
    if rec["receipt"]["e38_reproduction_mismatch_windows"]:
        raise SystemExit("E38 reproduction failed; stop (PREREG gate)")
    pools = {a: ranked_pool(rec["objects"][a], s) for a in rec["objects"]}
    summ = {a: arm_summary(p, s) for a, p in pools.items()}
    result = {"eligible_parts": len(s["eligible"]), "negative_area_km2": sum(s["neg_area"].values()),
              "arms": summ, "decision": decide(summ), "bootstrap_primary_minus_E38": bootstrap(pools, s),
              "receipt": rec["receipt"]}
    (OUT / "HOLDOUT_RESULT.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({a: {"primary": round(v["primary"], 4), **{f"d{d}": round(v[f"d{d}"]["recall"], 4)
                                                                  for d in DENSITIES + DESCRIPTIVE_DENSITIES},
                          "all": round(v["all"]["recall"], 4), "neg_per_km2": round(v["negative_per_km2"], 2)}
                      for a, v in summ.items()}, indent=1))
    print(json.dumps(result["decision"]), json.dumps(result["bootstrap_primary_minus_E38"]))


if __name__ == "__main__":
    main()
