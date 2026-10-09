"""Stage S scorer (PREREG.md): selected arm vs frozen R2 on SCREEN with the frozen Amendment 2 scorer pieces.

  python -B score_screen.py ARM      # outputs/tta_ensemble_2026-09-24/screen/ARM vs outputs/campaign_2026-09-22/screen/R2
(WSL adaf-h2h-venv, from the repo root). E38 (this run's reproduction arm) must score exactly as frozen R2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
import screen_scorer as ss  # noqa: E402

OUT = ROOT / "outputs/tta_ensemble_2026-09-24"
FROZEN_R2 = ROOT / "outputs/campaign_2026-09-22/screen/R2"
VIS = ROOT / "work/fable_investigation_2026-09-23/diag/screen_parts_table.json"
BUDGETS = {"full_26": (323, 647, 1294), "pilot_excluded_25": (311, 1229)}
FROZEN_EXPECT = {("full_26", 323): 39, ("full_26", 1294): 49}
BOOT_N, BOOT_SEED = 2000, 20260924


def read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def rows_for(arm_dir: Path) -> dict[str, list[dict]]:
    out = {p.stem: [dict(o, source_block=p.stem) for o in read(p)["objects"]]
           for p in sorted((arm_dir / "proposals").glob("SCREEN_*.json"))}
    if len(out) != 26:
        raise RuntimeError(f"{arm_dir}: expected 26 blocks, found {len(out)}")
    return out


def hits_for(pool: list[dict], arm: str, k: int, labels: list[dict]) -> list[dict]:
    selected = [dict(r, fused_rank=i + 1) for i, r in enumerate(ss.as_mixture_rows(pool, arm)[:k])]
    return ss.mix.match_greedy(selected, labels)


def main(arm: str) -> None:
    data, blocks, stage = ss.frozen_inputs()
    cohorts = ss.cohort_blocks(blocks)
    classes = {l["id"]: set(l["classes"]) for b in blocks for l in b["labels"]}
    verdict = {r["part"]: r["verdict"] for r in read(VIS)}
    dirs = {"R2": FROZEN_R2, "E38": OUT / "screen/E38", arm: OUT / "screen" / arm}
    rows = {a: rows_for(d) for a, d in dirs.items()}
    res: dict = {"arm": arm, "cohorts": {}}
    for cohort, budgets in BUDGETS.items():
        ids, labels = cohorts[cohort], data["labels"][cohort]
        pools = {a: ss.build_pool(stage, r, ids, True) for a, r in rows.items()}
        c = res["cohorts"][cohort] = {"pool_size": {a: len(p) for a, p in pools.items()}}
        for k in budgets:
            per = {}
            for a, pool in pools.items():
                h = hits_for(pool, a, k, labels)
                got = sorted(x["label_id"] for x in h)
                per[a] = {"H": len(got), "ids": got, "by_block": [sum(x["label_owner_block"] == b for x in h) for b in ids],
                          "barrow": sum("barrow" in classes[i] for i in got),
                          "enclosure": sum("enclosure" in classes[i] for i in got),
                          "by_visibility": {v: sum(verdict.get(i) == v for i in got)
                                            for v in ("clearly_visible", "possibly_visible", "not_visible")}}
            if (cohort, k) in FROZEN_EXPECT and (per["R2"]["H"], per["E38"]["H"]) != (FROZEN_EXPECT[(cohort, k)],) * 2:
                raise SystemExit(f"reproduction failed at {cohort} K={k}: R2 {per['R2']['H']}, E38 {per['E38']['H']}")
            if per["E38"]["ids"] != per["R2"]["ids"]:
                raise SystemExit(f"E38 hit set differs from frozen R2 at {cohort} K={k}")
            n = np.array([sum(l["owner_block"] == b for l in labels) for b in ids])
            r2, new = np.array(per["R2"]["by_block"]), np.array(per[arm]["by_block"])
            idx = np.random.default_rng(BOOT_SEED).integers(0, len(ids), size=(BOOT_N, len(ids)))
            den = n[idx].sum(1)
            keep = den > 0
            delta = (new[idx].sum(1) - r2[idx].sum(1))[keep] / den[keep]
            c[f"K{k}"] = {"R2": {kk: v for kk, v in per["R2"].items() if kk != "by_block"},
                          arm: {kk: v for kk, v in per[arm].items() if kk != "by_block"},
                          "delta": per[arm]["H"] - per["R2"]["H"],
                          "gained": sorted(set(per[arm]["ids"]) - set(per["R2"]["ids"])),
                          "lost": sorted(set(per["R2"]["ids"]) - set(per[arm]["ids"])),
                          "paired_block_bootstrap_90_points": [100 * float(q) for q in np.quantile(delta, [.05, .95])],
                          "blocks_better_worse_tied": [int((new > r2).sum()), int((new < r2).sum()), int((new == r2).sum())]}
    d323, d1294 = (res["cohorts"]["full_26"][f"K{k}"]["delta"] for k in (323, 1294))
    res["outcome"] = ("GAIN" if d323 >= 0 and d1294 >= 0 and d323 + d1294 >= 2 else
                      "HARM" if min(d323, d1294) <= -2 else "NULL")
    res["rule"] = "GAIN: d323>=0, d1294>=0, sum>=2; HARM: any d<=-2; else NULL (PREREG.md Stage S)"
    (OUT / f"SCREEN_RESULT_{arm}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    full = res["cohorts"]["full_26"]
    print(json.dumps({"outcome": res["outcome"], **{f"K{k}": {"R2": full[f"K{k}"]["R2"]["H"], arm: full[f"K{k}"][arm]["H"],
                                                              "boot90": full[f"K{k}"]["paired_block_bootstrap_90_points"]}
                                                    for k in BUDGETS["full_26"]}}))


if __name__ == "__main__":
    main(sys.argv[1])
