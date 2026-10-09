"""Which register types make up the IGNORE zones that contain group hits (aggregate type counts only)."""
import collections, json, sys
from pathlib import Path
import numpy as np
from shapely.geometry import Point, shape
from shapely.strtree import STRtree
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "work/holdout_region_2026-10/step2"))
import region_score as rs
rd = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))
for reg, ind, grp, prop in (
    ("outputs/adapt_nds_2026-10-05/register/denkmale_Oldenburg_20261005T010102Z.geojson",
     "outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json",
     "outputs/adapt_nds_2026-10-05/confirm/run/GROUP_MANIFEST_PRIVATE.json",
     "outputs/adapt_nds_2026-10-05/score/confirm/group/PROPOSALS_A_NDS_2026092802_PRIVATE.json"),
    ("outputs/fresh_region_2026-10-03/labels/denkmale_Verden_20261003T084547Z.geojson",
     "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json",
     "outputs/fresh_region_2026-10-03/step0/GROUP_MANIFEST_PRIVATE.json",
     "outputs/fresh_region_2026-10-03/step2/score/group/PROPOSALS_J_2026092803_PRIVATE.json")):
    elig = {int(p["record_id"]) for p in rd(ind)["parts"]}
    fs = [f for f in rd(reg)["features"] if f.get("geometry") and int(f["id"]) not in elig]
    geoms = [shape(f["geometry"]).buffer(30.0) for f in fs]; tree = STRtree(geoms)
    s = rs.setup(rd(grp)); pool = rs.ranked_pool(rd(prop)["objects"], s)
    allow = int(np.floor(2.0 * sum(s["neg_area"].values()) + 1e-9)); neg = 0; c = collections.Counter()
    for r in pool:
        if r["neg"]:
            if neg + 1 > allow: break
            neg += 1
        if r["hits"]:
            for i in tree.query(Point(r["easting"], r["northing"]), predicate="intersects"):
                c[fs[i]["properties"].get("objekttyp")] += 1
    print(Path(reg).name[:18], "IGNORE-type memberships of hitting proposals at d=2:", c.most_common(8))
