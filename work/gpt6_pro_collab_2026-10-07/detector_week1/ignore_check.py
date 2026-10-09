"""Task 2 side check: how many legacy barrow-group hits rest only on proposals that lie inside an IGNORE zone
(FRESH1_PREPIXEL.md:30, rule 6: proposals in IGNORE are "neither hits nor false alarms"). region_score has no IGNORE
test inside positive cores, so such proposals still count as hits. IGNORE is approximated from the local register
snapshot: every feature whose register id is not an eligible manifest member, buffered 30 m (the edge supplement and
NODATA-failed members are covered because they are not eligible members). Aggregate counts only.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/gpt6_pro_collab_2026-10-07/detector_week1/ignore_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "work/holdout_region_2026-10/step2"))
sys.path.insert(0, str(ROOT / "work/fusion_2026-10-03"))
import region_score as rs  # noqa: E402
import fuse_eval as fe  # noqa: E402
shd = rs.shd

SETS = {
    "verden": ("outputs/fresh_region_2026-10-03/labels/denkmale_Verden_20261003T084547Z.geojson",
               "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json",
               "outputs/fresh_region_2026-10-03/step0/GROUP_MANIFEST_PRIVATE.json",
               "outputs/fresh_region_2026-10-03/step2/score/group",
               ["ADAF_AO", "ADAF_barrow", "CONT", "J_2026092803"], ("ADAF_AO", "J_2026092803")),
    "oldenburg": ("outputs/adapt_nds_2026-10-05/register/denkmale_Oldenburg_20261005T010102Z.geojson",
                  "outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json",
                  "outputs/adapt_nds_2026-10-05/confirm/run/GROUP_MANIFEST_PRIVATE.json",
                  "outputs/adapt_nds_2026-10-05/score/confirm/group",
                  ["ADAF_AO", "ADAF_barrow", "CONT"] + [f"{a}_NDS_{s}" for a in "AC"
                                                         for s in ("2026092802", "2026092803", "2026092804")], None),
}


def rd(p: str) -> dict:
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def run(reg: str, ind: str, grp: str, pdir: str, models: list[str], deploy, skip_types=()) -> dict:
    elig = {int(p["record_id"]) for p in rd(ind)["parts"]}
    feats = rd(reg)["features"]
    ign = prep(unary_union([shape(f["geometry"]).buffer(30.0) for f in feats
                            if f.get("geometry") and int(f["id"]) not in elig
                            and f["properties"].get("objekttyp") not in skip_types]))
    s = rs.setup(rd(grp))
    objs = {m: rd(f"{pdir}/PROPOSALS_{m}_PRIVATE.json")["objects"] for m in models}
    if deploy:
        a, b = deploy
        objs["DEPLOY"] = fe.fused_objects(fe.order("INTERLEAVE", fe.own_pool(objs[a], s, "A"),
                                                   fe.own_pool(objs[b], s, "S")), s)
    out = {"ignore_types_excluded": list(skip_types), "eligible_register_ids": len(elig), "register_features": len(feats), "systems": {}}
    for m, o in objs.items():
        pool = rs.ranked_pool(o, s)
        row = {}
        for d in (0.5, 1.0, 2.0):
            rec, hit = shd.recall_at(pool, s, d)
            allow = int(np.floor(d * sum(s["neg_area"].values()) + 1e-9))
            neg, clean = 0, set()
            for r in pool:
                if r["neg"]:
                    if neg + 1 > allow:
                        break
                    neg += 1
                if not ign.contains(Point(r["easting"], r["northing"])):
                    clean.update(r["hits"])
            row[str(d)] = {"hits": len(hit), "hits_only_via_ignore": len(hit - clean),
                           "recall": rec, "recall_ignore_excluded": len(hit & clean) / len(s["eligible"])}
        row["primary"] = float(np.mean([row[str(d)]["recall"] for d in (0.5, 1.0, 2.0)]))
        row["primary_ignore_excluded"] = float(np.mean([row[str(d)]["recall_ignore_excluded"] for d in (0.5, 1.0, 2.0)]))
        out["systems"][m] = row
        print(m, round(row["primary"], 4), round(row["primary_ignore_excluded"], 4),
              [row[str(d)]["hits_only_via_ignore"] for d in (0.5, 1.0, 2.0)], flush=True)
    return out


def main() -> None:
    res = {}
    for n, args in SETS.items():
        print("==", n, flush=True)
        res[n] = run(*args)
        print("== without Grabhügelfeld", n, flush=True)
        res[n + "_no_grabhuegelfeld"] = run(*args, skip_types=("Grabhügelfeld",))
        print("== without container areas", n, flush=True)
        res[n + "_no_containers"] = run(*args, skip_types=("Grabhügelfeld", "Grabungsschutzgebiet"))
    (HERE / "IGNORE_CHECK.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
