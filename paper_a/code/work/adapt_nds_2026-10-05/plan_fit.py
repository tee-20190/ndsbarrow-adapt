"""ADAPT-NDS-1 Step 0: Lower Saxony FIT window plan (vectors only; no pixel is read).

FIT districts Uelzen, Harburg and Heidekreis (packet). Windows follow the HR1/HR3 FIT format (423 m integer windows;
masks on HR1's frozen grid, rebuilt on the VM by HR1 masks.py and verified against label_sha256):
  labels     FRESH-1 rules 1-4 per district (prepixel_stats.eligible_barrows); rule 5 groups (60 m linkage) over the
             union of the three districts' eligible barrows; a group belongs to the FIT union when its member-centroid
             mean lies inside it
  positive   N_POS groups drawn by default_rng(20261005).choice(sorted eligible group ids, N_POS, replace=False); one
             423 m window centred (<= 0.5 m) on the group's member-centroid mean; mask: every eligible barrow polygon
             = positive (HR1 masks adds its 5 m IGNORE band), every other register record (FIT snapshots and their
             1 km edge supplements, any type) buffered 30 m = IGNORE (FRESH-1 rule 6); positives overwrite IGNORE
  background N_BG cells of the 423 m grid (origin 0, 0) drawn by default_rng(20261006).choice(sorted eligible cells,
             N_BG, replace=False): wholly inside the FIT union shrunk by 100 m and >= 251 m from every register record;
             all-zero masks (voids become IGNORE through the trainer's void policy V)
  halo       every window lies >= 2 km from Verden and from the CONFIRM district (packet: >= 2 km inside on any side
             that touches Verden or CONFIRM)
  terrain    pixel centres outside the FIT union are NoData (FRESH-1 STEP0_DECISIONS item 1, with the union as the
             district); tiles = LGLN DGM1 index records meeting a window with >= 1 pixel centre inside the union; pins:
             HEAD Content-Length and ETag (verified on the VM; sha256 recorded at fetch)
N_POS = N_BG = the number of Lower Saxony pairs of one ADAPT run (852): every window is drawn once per run.
Outputs (private; deterministic, an existing file must be byte-identical): outputs/adapt_nds_2026-10-05/fit/
FIT_TABLE_PRIVATE.json, FIT_TILE_INDEX_PRIVATE.json, FIT_STAGING_PLAN_PRIVATE.json, FIT_INSIDE_MASKS_PRIVATE.npz;
coordinate-free work/adapt_nds_2026-10-05/FIT_PLAN_REPORT.json.

  python -B work/adapt_nds_2026-10-05/plan_fit.py build
  python -B work/adapt_nds_2026-10-05/plan_fit.py pin      # after head_check.py: FIT_STAGING_PINNED_PRIVATE.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nds_common as N  # noqa: E402
import nds_schedule as SCH  # noqa: E402

sys.path.insert(0, str(N.FRESH))
import prepixel_stats as ps  # noqa: E402  locked FRESH-1 label rules (read-only)
import step0 as S0  # noqa: E402  FRESH-1 Step 0 functions (read-only)

sys.path.insert(0, str(N.HR1_CHAIN))
import masks as HM  # noqa: E402  HR1 FIT mask grid and label hash (frozen, read-only)

import shapely  # noqa: E402
from shapely.geometry import box, mapping, shape  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

TABLE = N.FIT_DIR / "FIT_TABLE_PRIVATE.json"
TILE_INDEX = N.FIT_DIR / "FIT_TILE_INDEX_PRIVATE.json"
STAGING = N.FIT_DIR / "FIT_STAGING_PLAN_PRIVATE.json"
PINNED = N.FIT_DIR / "FIT_STAGING_PINNED_PRIVATE.json"           # staging plan + HEAD pins + refused regions (VM)
HEAD_LEDGER = N.PRIV / "HEAD_LEDGER_PRIVATE.json"
INSIDE = N.FIT_DIR / "FIT_INSIDE_MASKS_PRIVATE.npz"
REPORT = HERE / "FIT_PLAN_REPORT.json"
W = N.FIT_WINDOW_M
HALF = W / 2.0


def snapshots() -> dict[str, dict]:
    out = {}
    for d in N.FIT_DISTRICTS:
        snaps = sorted(N.REGISTER.glob(f"SNAPSHOT_district_{N.district_tag(d)}_*.json"))
        if not snaps:
            raise SystemExit(f"no register snapshot for {d}")
        s = N.read_json(snaps[-1]) | {"snapshot_file": N.rel(snaps[-1])}
        for k in ("records_file", "supplement_file"):
            if N.sha256_file(N.ROOT / s[k]["path"]) != s[k]["sha256"]:
                raise SystemExit(f"{s[k]['path']} differs from its snapshot record")
        out[d] = s
    return out


def polygons(snaps: dict) -> dict:
    sel = N.read_json(N.CONFIRM_SELECTION)
    confirm = sel["confirm"]["district"]
    b = N.read_json(N.ROOT / next(iter(snaps.values()))["boundary_snapshot"])
    polys: dict[str, list] = {}
    for p, g in ps.gml_polygons(N.ROOT / b["file"]["path"]):
        polys.setdefault(p.get("landkreis", ""), []).append(g)
    need = list(N.FIT_DISTRICTS) + [N.DEV_DISTRICT, confirm]
    missing = [n for n in need if n not in polys]
    if missing:
        raise SystemExit(f"district polygons missing: {missing}")
    out = {n: unary_union(polys[n]) for n in need}
    out["_confirm_name"] = confirm
    out["_boundary_file"] = b["file"]
    return out


def labels(snaps: dict) -> dict:
    feats, sup, rows = {}, {}, []
    excl = {}
    for d, s in snaps.items():
        fs = json.loads((N.ROOT / s["records_file"]["path"]).read_text(encoding="utf-8"))["features"]
        for f in fs:
            feats[f["id"]] = f
        kept, ex = ps.eligible_barrows(fs)
        excl[d] = ex
        rows += [r | {"district": d} for r in kept]
        for r in N.read_json(N.ROOT / s["supplement_file"]["path"])["records"]:
            sup.setdefault(r["id"], r)
    sup = {k: v for k, v in sup.items() if k not in feats}          # FIT records come from their own snapshots
    rows.sort(key=lambda r: r["id"])
    return {"feats": feats, "supplement": sup, "rows": rows, "exclusions": excl}


def fit_groups(L: dict, union) -> list[dict]:
    groups = ps.describe_groups(L["rows"], ps.link_groups(L["rows"]), union)
    return [g for g in groups if g["inside_district"]]


def window_at(cx: float, cy: float) -> list[int]:
    """423 m integer window whose centre lies within 0.5 m of (cx, cy) in each axis."""
    x0, y0 = int(round(cx - HALF)), int(round(cy - HALF))
    return [x0, y0, x0 + W, y0 + W]


def halo_ok(b: list[int], verden, confirm) -> bool:
    g = box(*b)
    return g.distance(verden) >= N.FIT_HALO_M and g.distance(confirm) >= N.FIT_HALO_M


def ignore_geoms(L: dict) -> list[tuple[str, object]]:
    """Every register record that is not an eligible FIT barrow (FIT snapshots + edge supplements), made valid."""
    elig = {r["id"] for r in L["rows"]}
    out = []
    for fid, f in sorted(L["feats"].items()):
        if fid not in elig and f.get("geometry"):
            out.append((fid, S0.valid_geom(shape(f["geometry"]))))
    for fid, r in sorted(L["supplement"].items()):
        out.append((fid, S0.valid_geom(shape(r["geometry"]))))
    return out


def all_record_geoms(L: dict) -> list:
    return [S0.valid_geom(shape(f["geometry"])) for f in L["feats"].values() if f.get("geometry")] + \
        [S0.valid_geom(shape(r["geometry"])) for r in L["supplement"].values()]


def positive_shapes(b: list[int], pos_tree: STRtree, pos_geoms: list, pos_ids: list, ign_tree: STRtree,
                    ign_geoms: list, ign_ids: list) -> tuple[dict, list, list]:
    wb = box(*b)
    pos_hit = sorted(int(i) for i in pos_tree.query(wb.buffer(HM_BAND + 1.0), predicate="intersects"))
    ign_hit = sorted(int(i) for i in ign_tree.query(wb, predicate="dwithin", distance=N.FIT_IGNORE_M + 1.0))
    clip = wb.buffer(2.0, join_style="mitre")
    pos = [mapping(pos_geoms[i]) for i in pos_hit]
    ign = []
    for i in ign_hit:
        g = ign_geoms[i].buffer(N.FIT_IGNORE_M).intersection(clip)
        if not g.is_empty:
            ign.append(mapping(g))
    return {"positive": pos, "ignore": ign}, [pos_ids[i] for i in pos_hit], [ign_ids[i] for i in ign_hit]


HM_BAND = 5.0


def background_cells(frame, verden, confirm, rec_tree: STRtree) -> list[list[int]]:
    x0, y0, x1, y1 = frame.bounds
    cells = []
    for i in range(math.floor(x0 / W), math.ceil(x1 / W)):
        for j in range(math.floor(y0 / W), math.ceil(y1 / W)):
            cells.append([i * W, j * W, (i + 1) * W, (j + 1) * W])
    boxes = shapely.box(*np.array(cells, dtype=float).T)
    ok = shapely.covered_by(boxes, frame)
    idx = np.flatnonzero(ok)
    hits = rec_tree.query(boxes[idx], predicate="dwithin", distance=N.FIT_SEP_M)
    near = set(idx[np.unique(hits[0])].tolist()) if len(hits[0]) else set()
    out = []
    for k in idx.tolist():
        if k in near:
            continue
        if halo_ok(cells[k], verden, confirm):
            out.append(cells[k])
    return out


def separation_check(cells: list[list[int]], recs: list) -> dict:
    """Second method (no STRtree): exact minimum distance from each drawn background window to every record whose
    envelope lies within 260 m of it."""
    env = np.array([g.bounds for g in recs])
    worst, bad = float("inf"), 0
    for c in cells:
        m = ((env[:, 0] <= c[2] + 260) & (env[:, 2] >= c[0] - 260) & (env[:, 1] <= c[3] + 260)
             & (env[:, 3] >= c[1] - 260))
        b = box(*c)
        d = min((b.distance(recs[i]) for i in np.flatnonzero(m)), default=float("inf"))
        worst = min(worst, d)
        bad += d < N.FIT_SEP_M
    return {"windows": len(cells), "below_251m": int(bad), "min_distance_m": round(worst, 1)}


def tile_plan(windows: list[dict], union) -> tuple[dict, dict, dict]:
    env = unary_union([box(*w["bounds"]) for w in windows]).bounds
    feats, count = S0.index_records(env)
    rows = [S0.tile_row(f["attributes"]) for f in feats]
    need = {}
    for r in rows:
        if any(N.boxes_overlap(r["box"], w["bounds"]) for w in windows):
            need[r["name"]] = r
    masks, modes = {}, {}
    for name, r in sorted(need.items()):
        m = S0.inside_mask(r["box"], union)
        r["inside_px"] = int(m.sum())
        if r["inside_px"] == 0:
            modes[name] = "none"
        elif m.all():
            modes[name] = "all"
        else:
            modes[name] = "mask"
            masks[name] = np.packbits(m, axis=None)
    index = {"utc": N.utc(), "service": N.INDEX, "envelope": list(env), "index_count": count,
             "index_records": len(rows), "selected": [need[n] for n in sorted(need) if modes[n] != "none"],
             "skipped_no_inside_pixel": sorted(n for n in need if modes[n] == "none")}
    return index, masks, modes


def build() -> dict:
    N.check_disk("before the FIT plan")
    snaps = snapshots()
    P = polygons(snaps)
    union = unary_union([P[d] for d in N.FIT_DISTRICTS])
    frame = union.buffer(-N.FIT_FRAME_M)
    shapely.prepare(union)
    shapely.prepare(frame)
    verden, confirm = P[N.DEV_DISTRICT], P[P["_confirm_name"]]
    L = labels(snaps)
    groups = fit_groups(L, union)
    n_pairs = SCH.ls_pair_count()
    gwin = {g["id"]: window_at(*g["centre"]) for g in groups}
    elig_groups = sorted(g["id"] for g in groups if halo_ok(gwin[g["id"]], verden, confirm))
    pos_ids = sorted(np.random.default_rng(N.SEED_FIT_POS).choice(elig_groups, n_pairs, replace=False).tolist())
    gby = {g["id"]: g for g in groups}
    pos_geoms = [r["geom"] for r in L["rows"]]
    pos_rec_ids = [r["id"] for r in L["rows"]]
    ign = ignore_geoms(L)
    ign_tree = STRtree([g for _, g in ign])
    pos_tree = STRtree(pos_geoms)
    recs = all_record_geoms(L)
    rec_tree = STRtree(recs)
    pwins = []
    for gid in pos_ids:
        b = gwin[gid]
        shapes, inc, ign_hit = positive_shapes(b, pos_tree, pos_geoms, pos_rec_ids, ign_tree, [g for _, g in ign],
                                               [i for i, _ in ign])
        lab = HM.label_array(b, shapes)
        pwins.append({"id": f"NDS-P-{gid}", "role": "positive", "sam": gid, "monument": gid, "class": "barrow",
                      "district": Counter(next(r["district"] for r in L["rows"] if r["id"] == m)
                                          for m in gby[gid]["members"]).most_common(1)[0][0],
                      "bounds": b, "members": gby[gid]["members"], "shapes": shapes,
                      "incidental_positives": sorted(set(inc) - set(gby[gid]["members"])),
                      "ignore_records": len(ign_hit), "label_sha256": HM.label_sha256(lab),
                      "positive_px": int((lab == HM.POS_V).sum()), "ignore_px": int((lab == HM.IGNORE_V).sum())})
    cells = background_cells(frame, verden, confirm, rec_tree)
    bg_pick = sorted(np.random.default_rng(N.SEED_FIT_BG).choice(len(cells), n_pairs, replace=False).tolist())
    bwins = []
    for k, ci in enumerate(bg_pick):
        b = cells[ci]
        lab = HM.label_array(b, {})
        bwins.append({"id": f"NDS-B-{k:04d}", "role": "background", "bounds": b, "shapes": {"positive": [],
                                                                                           "ignore": []},
                      "ignore_records": 0, "label_sha256": HM.label_sha256(lab), "positive_px": 0, "ignore_px": 0})
    windows = pwins + bwins
    for w in windows:
        if not halo_ok(w["bounds"], verden, confirm):
            raise SystemExit(f"{w['id']}: inside the 2 km halo")
    index, masks, modes = tile_plan(windows, union)
    by = {t["name"]: t for t in index["selected"]}
    for w in windows:
        w["tiles"] = sorted(n for n, t in by.items() if N.boxes_overlap(t["box"], w["bounds"]))
        if not w["tiles"]:
            raise SystemExit(f"{w['id']}: no DGM1 tile with a pixel inside the FIT union")
    inside_bytes = npz_bytes(masks)
    table = {"schema": "adapt_nds_fit_table_v1", "region": "Lower Saxony FIT (Uelzen, Harburg, Heidekreis)",
             "role": "FIT", "crs": N.CRS, "monuments": {w["sam"]: [w["id"]] for w in pwins},
             "positive_windows": pwins, "background_windows": bwins,
             "rules": {"positive_window": f"{W} m integer window centred (<= 0.5 m) on each drawn group's member-"
                                          "centroid mean", "draw": f"default_rng({N.SEED_FIT_POS}).choice(sorted "
                                          f"eligible groups, {n_pairs}) and default_rng({N.SEED_FIT_BG}).choice("
                                          f"eligible 423 m cells, {n_pairs})",
                       "masks": "HR1 masks.py label_array (0.5 m grid, pixel-centre rule): eligible barrow polygons "
                                "positive + 5 m IGNORE band; every other register record + 30 m IGNORE; positives "
                                "overwrite IGNORE; voids IGNORE via void policy V",
                       "terrain": "pixel centres outside the FIT union are NoData; >= 2 km from Verden and CONFIRM"},
             "attribution": N.ATTRIBUTION}
    staging = {"schema": "adapt_nds_fit_staging_v1", "windows": [{"id": w["id"], "role": w["role"],
                                                                  "bounds": w["bounds"], "tiles": w["tiles"]}
                                                                 for w in windows],
               "tiles": {n: {"url": t["url"], "box": t["box"], "square_km": t["square_km"], "year": t["year"],
                             "delivery": t["delivery"], "inside_px": t["inside_px"], "inside": modes[n]}
                         for n, t in sorted(by.items())},
               "inside_masks_sha256": N.sha256_bytes(inside_bytes),
               "union_envelope": [float(v) for v in union.bounds],
               "rule": "each window reads the LGLN DGM1 records on the squares it meets (official url); pixel centres "
                       "outside the FIT union are NoData; tiles pasted in name order, first valid value wins (older "
                       "delivery on two-delivery squares); pins: HEAD Content-Length and ETag (head_check.py)"}
    N.FIT_DIR.mkdir(parents=True, exist_ok=True)
    N.write_same(TABLE, table, indent=None)
    N.write_same(TILE_INDEX, index, indent=None)
    write_same_bytes(INSIDE, inside_bytes)
    N.write_same(STAGING, staging, indent=None)
    sep = separation_check([w["bounds"] for w in bwins], recs)
    if sep["below_251m"]:
        raise SystemExit(f"background separation failed by the second method: {sep}")
    rep = report(L, groups, elig_groups, pwins, bwins, cells, index, modes, union, frame, P, snaps) | {
        "background_separation_check": sep}
    N.write_json(REPORT, rep)
    return rep


def report(L, groups, elig_groups, pwins, bwins, cells, index, modes, union, frame, P, snaps) -> dict:
    fresh_verden = unary_union([g for _, g in ps.gml_polygons(
        N.FRESH_PRIV / "labels/landkreis_Verden_20261003T084547Z.gml")])
    sym = fresh_verden.symmetric_difference(P[N.DEV_DISTRICT]).area
    return {"schema": "adapt_nds_fit_plan_report_v1", "utc": N.utc(), "coordinate_free": True,
            "districts": list(N.FIT_DISTRICTS), "union_km2": round(union.area / 1e6, 1),
            "frame_km2": round(frame.area / 1e6, 1),
            "records": len(L["feats"]), "supplement_records": len(L["supplement"]),
            "eligible_barrows": len(L["rows"]), "exclusions": L["exclusions"],
            "eligible_by_district": dict(Counter(r["district"] for r in L["rows"])),
            "groups_in_union": len(groups), "groups_outside_halo": len(elig_groups),
            "groups_in_halo_excluded": len(groups) - len(elig_groups),
            "positive_windows": len(pwins), "background_windows": len(bwins),
            "positive_windows_by_district": dict(Counter(w["district"] for w in pwins)),
            "barrows_in_drawn_groups": sum(len(w["members"]) for w in pwins),
            "incidental_positive_barrows": sum(len(w["incidental_positives"]) for w in pwins),
            "positive_px_mean": round(float(np.mean([w["positive_px"] for w in pwins])), 1),
            "ignore_px_mean_positive_windows": round(float(np.mean([w["ignore_px"] for w in pwins])), 1),
            "background_candidate_cells": len(cells),
            "window_km2_sum": round(len(pwins + bwins) * W * W / 1e6, 2),
            "window_union_km2": round(unary_union([box(*w["bounds"]) for w in pwins + bwins]).area / 1e6, 2),
            "tiles": len(index["selected"]), "tiles_skipped_no_inside_pixel": len(index["skipped_no_inside_pixel"]),
            "tile_modes": dict(Counter(modes[t["name"]] for t in index["selected"])),
            "tile_years": dict(Counter(t["year"] for t in index["selected"])),
            "squares_with_multiple_tiles": sum(v > 1 for v in Counter(tuple(t["square_km"])
                                                                      for t in index["selected"]).values()),
            "min_window_distance_to_verden_m": round(min(box(*w["bounds"]).distance(P[N.DEV_DISTRICT])
                                                         for w in pwins + bwins), 1),
            "min_window_distance_to_confirm_m": round(min(box(*w["bounds"]).distance(P[P["_confirm_name"]])
                                                          for w in pwins + bwins), 1),
            "verden_polygon_vs_fresh1_symdiff_m2": round(sym, 3),
            "files": {N.rel(p): N.sha256_file(p) for p in (TABLE, TILE_INDEX, STAGING, INSIDE)},
            "snapshots": {d: s["snapshot_file"] for d, s in snaps.items()},
            "statement": "vectors and tile metadata only: no DTM pixel was read", "attribution": N.ATTRIBUTION}


def npz_bytes(arrays: dict) -> bytes:
    sys.path.insert(0, str(N.FRESH_STEP2))
    import plan_fresh  # noqa: E402  deterministic npz writer (read-only)
    return plan_fresh.npz_bytes(arrays)


def write_same_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path.name} exists with different content; remove it deliberately to rebuild")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def pin() -> dict:
    """The VM's FIT staging plan: FIT_STAGING_PLAN + each tile's HEAD pins (Content-Length, ETag = content md5) + the
    Verden and CONFIRM polygons (LGLN boundaries, CC BY 4.0) that no FIT window may approach within 2 km."""
    staging, head = N.read_json(STAGING), N.read_json(HEAD_LEDGER)
    rows = {r["name"]: r for r in head["rows"]}
    if not head.get("etag_probe") or not all(p["md5_equals_etag"] and p["bytes_equal_head"] for p in head["etag_probe"]):
        raise SystemExit("the HEAD ledger does not establish ETag = content md5")
    snaps = snapshots()
    P = polygons(snaps)
    tiles = {}
    for n, t in staging["tiles"].items():
        r = rows.get(n)
        if r is None or r["status"] != "ok" or r["url"] != t["url"] or not re.fullmatch(r"[0-9a-f]{32}", r["etag"]):
            raise SystemExit(f"{n}: no usable HEAD pin")
        tiles[n] = t | {"bytes": r["bytes"], "md5": r["etag"], "last_modified": r["last_modified"]}
    rec = {"schema": "adapt_nds_fit_staging_pinned_v1", "kind": "fit", "windows": staging["windows"],
           "tiles": tiles, "inside_masks_sha256": staging["inside_masks_sha256"],
           "envelope": staging["union_envelope"], "confirm_district": P["_confirm_name"],
           "refused_regions": [{"name": "Verden", "geometry": mapping(P[N.DEV_DISTRICT])},
                               {"name": P["_confirm_name"], "geometry": mapping(P[P["_confirm_name"]])}],
           "staging_plan_sha256": N.sha256_file(STAGING), "head_ledger_sha256": N.sha256_file(HEAD_LEDGER),
           "rule": staging["rule"] + "; pins verified on the VM: Content-Length and md5 == ETag; sha256 logged"}
    N.write_same(PINNED, rec, indent=None)
    return {"tiles": len(tiles), "windows": len(rec["windows"]), "path": N.rel(PINNED),
            "sha256": N.sha256_file(PINNED)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "pin"))
    a = ap.parse_args()
    out = build() if a.cmd == "build" else pin()
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, ensure_ascii=False, indent=1)[:5000])
