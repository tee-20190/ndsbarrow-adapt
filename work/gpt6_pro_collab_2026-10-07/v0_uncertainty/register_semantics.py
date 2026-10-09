"""Task 5 (R3-8 item 2, ROUND3_REPLY.md:699-703): what the register's Grabhuegelfeld (barrow cemetery) and
Grabungsschutzgebiet (excavation protection area) polygons are, and how they relate to the eligible individual
barrows. Aggregate counts only: no names, ids or coordinates are written (site names are parsed in memory only to
test whether a cemetery and its member barrows share a register site number).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/gpt6_pro_collab_2026-10-07/v0_uncertainty/register_semantics.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REG = {"verden": ("outputs/fresh_region_2026-10-03/labels/denkmale_Verden_20261003T084547Z.geojson",
                  "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json"),
       "oldenburg": ("outputs/adapt_nds_2026-10-05/register/denkmale_Oldenburg_20261005T010102Z.geojson",
                     "outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json")}
TYPES = ("Grabhügelfeld", "Grabungsschutzgebiet")
ATTRS = ("fachbereich", "ausweisung", "ausweisung_code", "siteProtectionClassification",
         "siteDesignation.designation.title", "legalFoundationDocument.title")
FST = re.compile(r"Gmkg\.\s*(.+?)\s+FStNr\.\s*(\d+)")
NUMWORDS = "ein|eine|einen|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf"
COUNT_RE = re.compile(rf"\b(\d{{1,3}}|{NUMWORDS})\s+(?:\w+\s+){{0,2}}(?:Grabh\u00fcgel|H\u00fcgel)n?\b", re.IGNORECASE)
PROTECT_RE = re.compile(r"§\s*16|Grabungsschutzgebiet", re.IGNORECASE)


def centroids(geoms: list) -> np.ndarray:
    return shapely.points(np.array([[g.centroid.x, g.centroid.y] for g in geoms]))


def site_key(props: dict) -> tuple | None:
    m = FST.search(str(props.get("siteName") or ""))
    return (m.group(1), int(m.group(2))) if m else None


WORDVAL = {w: i for i, ws in enumerate(["", "ein|eine|einen", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
                                          "acht", "neun", "zehn", "elf", "zwölf"]) for w in ws.split("|") if w}


def stated_count(desc: str) -> int | None:
    """First 'N (adjective) Grabhuegel/Huegel' count in a description (digits or German number words)."""
    m = COUNT_RE.search(desc)
    if not m:
        return None
    tok = m.group(1).lower()
    return int(tok) if tok.isdigit() else WORDVAL.get(tok)


def agreement(desc: list[str], inside: list[int]) -> dict:
    pairs = [(stated_count(d), n) for d, n in zip(desc, inside)]
    pairs = [(s, n) for s, n in pairs if s is not None]
    return {"with_stated_count": len(pairs), "stated_equals_records_inside": sum(s == n for s, n in pairs),
            "stated_more_than_inside": sum(s > n for s, n in pairs), "stated_fewer_than_inside": sum(s < n for s, n in pairs),
            "sum_stated": int(sum(s for s, _ in pairs)), "sum_grabhuegel_records_inside": int(sum(n for _, n in pairs))}


def bins(counts: list[int]) -> dict:
    c = np.asarray(counts)
    return {"0": int((c == 0).sum()), "1": int((c == 1).sum()), "2-4": int(((c >= 2) & (c <= 4)).sum()),
            "5-9": int(((c >= 5) & (c <= 9)).sum()), ">=10": int((c >= 10).sum()), "max": int(c.max()) if len(c) else 0}


def region(name: str) -> dict:
    reg_p, man_p = REG[name]
    feats = json.loads((ROOT / reg_p).read_text(encoding="utf-8"))["features"]
    man = json.loads((ROOT / man_p).read_text(encoding="utf-8"))
    elig = {int(p["record_id"]) for p in man["parts"]}
    scored = shapely.union_all([shapely.box(*b) for t in man["tiles"] for b in t["core"]])
    by_id = {int(f["id"]): f for f in feats}
    barrows = [f for f in feats if f["properties"].get("objekttyp") == "Grabhügel" and f.get("geometry")]
    el_pts = centroids([shape(by_id[i]["geometry"]) for i in sorted(elig)])
    all_pts = centroids([shape(f["geometry"]) for f in barrows])
    el_keys = [site_key(by_id[i]["properties"]) for i in sorted(elig)]
    out = {"register_records": len(feats), "grabhuegel_records": len(barrows), "eligible_entities": len(elig),
           "types": {}}
    for t in TYPES:
        fs = [f for f in feats if f["properties"].get("objekttyp") == t and f.get("geometry")]
        rec: dict = {"records": len(fs)}
        if not fs:
            out["types"][t] = rec
            continue
        geoms = [shapely.make_valid(shape(f["geometry"])) for f in fs]
        area_ha = np.array([g.area / 1e4 for g in geoms])
        el_in = [int(shapely.contains_xy(g, shapely.get_x(el_pts), shapely.get_y(el_pts)).sum()) for g in geoms]
        all_in = [int(shapely.contains_xy(g, shapely.get_x(all_pts), shapely.get_y(all_pts)).sum()) for g in geoms]
        union = shapely.union_all(geoms)
        inside_any = shapely.contains_xy(union, shapely.get_x(el_pts), shapely.get_y(el_pts))
        near_any = shapely.contains_xy(union.buffer(30.0), shapely.get_x(el_pts), shapely.get_y(el_pts))
        desc = [str(f["properties"].get("beschreibung") or "") for f in fs]
        keyed, linked = 0, 0
        for f, g in zip(fs, geoms):
            k = site_key(f["properties"])
            if k is None:
                continue
            keyed += 1
            members = [ek for ek, inside in zip(el_keys, shapely.contains_xy(g, shapely.get_x(el_pts),
                                                                               shapely.get_y(el_pts))) if inside]
            linked += sum(ek == k for ek in members)
        rec |= {"attributes": {a: dict(Counter(str(f["properties"].get(a)) for f in fs)) for a in ATTRS},
                "geometry_types": dict(Counter(f["geometry"]["type"] for f in fs)),
                "area_ha": {"median": float(np.median(area_ha)), "p90": float(np.percentile(area_ha, 90)),
                            "max": float(area_ha.max()), "sum": float(area_ha.sum())},
                "intersecting_scored_area": int(sum(g.intersects(scored) for g in geoms)),
                "eligible_barrow_centroids_per_polygon": bins(el_in),
                "all_grabhuegel_centroids_per_polygon": bins(all_in),
                "eligible_inside_any_polygon": int(inside_any.sum()),
                "eligible_inside_any_polygon_plus_30m": int(near_any.sum()),
                "descriptions_present": int(sum(bool(d) for d in desc)),
                "descriptions_stating_a_barrow_count": int(sum(bool(COUNT_RE.search(d)) for d in desc)),
                "descriptions_citing_protection_law": int(sum(bool(PROTECT_RE.search(d)) for d in desc)),
                "stated_count_vs_individual_records": agreement(desc, all_in),
                "area_ha_of_polygons_without_any_grabhuegel_record": {
                    "n": int(sum(n == 0 for n in all_in)),
                    "median": float(np.median([a for a, n in zip(area_ha, all_in) if n == 0])) if 0 in all_in else None,
                    "max": float(max([a for a, n in zip(area_ha, all_in) if n == 0], default=0.0))},
                "polygons_with_parsable_site_number": keyed,
                "eligible_inside_sharing_the_polygon_site_number": linked}
        out["types"][t] = rec
    gf = [shapely.make_valid(shape(f["geometry"])) for f in feats
          if f["properties"].get("objekttyp") == TYPES[0] and f.get("geometry")]
    gs = [shapely.make_valid(shape(f["geometry"])) for f in feats
          if f["properties"].get("objekttyp") == TYPES[1] and f.get("geometry")]
    if gf and gs:
        u = shapely.union_all(gs)
        out["cemeteries_inside_protection_area"] = int(sum(g.representative_point().within(u) for g in gf))
        out["eligible_inside_protection_but_no_cemetery"] = int(
            (shapely.contains_xy(u, shapely.get_x(el_pts), shapely.get_y(el_pts))
             & ~shapely.contains_xy(shapely.union_all(gf), shapely.get_x(el_pts), shapely.get_y(el_pts))).sum())
    return out


def main() -> None:
    res = {"schema": "v0_register_semantics_v1", "regions": {n: region(n) for n in REG}}
    (HERE / "REGISTER_SEMANTICS.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8",
                                                  newline="\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
