"""HOLDOUT-REGION-3 (Ireland) visibility audit (PACKET_PREPIXEL.md section 3 Step 1, AMENDMENT_1 items 3-5,
STEP1_PROCEDURE.md): WSL side. Adapted from the HR2 audit's vis_scot.py.

CPU only: no GPU, no detector output on Irish data, no TEST tile. Controls and smoke points are drawn from metadata
before any pixel is read. fetch reads each GSI zip in memory and keeps only the item windows; nodata computes validity
only; no DEV window or panel is viewed before the lock (the builder views the smoke panels S1-S3 only).

  python -B work/hr3_ireland_2026-10-02/visibility/vis_irl.py <command>     (WSL adaf-h2h-venv, repo root)
    points     POINTS_PRIVATE.json (create-only): 20 DEV controls + 3 FIT smoke points (metadata only)
    fetch      raw_items/<id>.tif for the 214 sampled parts, the controls and the smoke points, FETCH_LEDGER.jsonl,
               FETCH_RECEIPT_PRIVATE.json (run inside storage_guard)
    controls-a2  POINTS_A2_PRIVATE.json (create-only): Amendment 2, the first 20 draw-order control candidates that
               pass V2, with windows for new candidates (FETCH_RECEIPT_A2_PRIVATE.json; inside storage_guard)
    nodata     NODATA_PARTS_PRIVATE.json + NODATA_RESULT.json (HR2 V1/V2; floor N >= 120)
    sample     SAMPLE_PRIVATE.json (create-only): audit parts by stratum + the 20 controls
    freeze     AUDIT_FREEZE.json (create-only)
    items      ITEMS_PRIVATE.json: the 80 audit items and 3 smoke items with their raw windows
    render     SEALED_MAPPING.json (secrets shuffle), panels, RENDER_RECEIPT.json, RENDER_INPUTS.json
"""
from __future__ import annotations

import io
import json
import math
import re
import secrets
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vis_common as V  # noqa: E402

STEP0 = V.PREP_OUT / "step0_v2"
DEV_SAMPLE = V.PREP_OUT / "DEV_SAMPLE_PRIVATE.json"
SPLIT = {"DEV": [3], "TEST": [1], "FIT": [4]}
BLOCK_TILES = {3: 91, 1: 117, 4: 44}
POS_HALF, CORE, SMR_CLEAR_M, MAX_DRAWS = 750, 1000, 100.0, 20000
REACH_M, SQ_HALF, V1_MIN, V2_MIN = 15.0, 128, 0.95, 0.90
WIN = 2 * V.HALF_WIN
ENGLISH_NODATA = float(np.finfo(np.float32).min)        # the English raw-window NODATA (welsh_fetch.py)
ITM = "EPSG:2157"
SRC_NODATA = -9999.0
DTM_MEMBER = re.compile(r"(?:_dtm\.asc|(?:^|/)dtm/[^/]+\.tif)$", re.I)
USER_AGENT = "hr3-visibility-audit/1.0"


def seeded(tag: str, ident: str) -> str:
    import hashlib
    return hashlib.sha256(f"{V.SEED}:{tag}:{ident}".encode()).hexdigest()


def _intersects(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def window_bounds(cx: float, cy: float) -> list[int]:
    x, y = math.floor(cx + 0.5), math.floor(cy + 0.5)
    return [x - V.HALF_WIN, y - V.HALF_WIN, x + V.HALF_WIN, y + V.HALF_WIN]


def tile_box(t: dict) -> list[int]:
    b = [t["xmin"], t["ymin"], t["xmax"], t["ymax"]]
    if any(float(v) != int(v) for v in b):
        raise SystemExit(f"{t['name']}: non-integer tile bounds {b}")
    return [int(v) for v in b]


def check_procedure_lock() -> dict:
    lk = V.read(V.PROCEDURE_LOCK)
    if V.sha(V.PROCEDURE) != lk["sha256"]:
        raise SystemExit("STEP1_PROCEDURE.md changed after its lock")
    bad = [p for p, h in lk["inputs_sha256"].items() if V.sha(V.ROOT / p) != h]
    if bad:
        raise SystemExit(f"locked inputs changed: {bad}")
    for doc, lock in ((V.PROCEDURE_A1, V.PROCEDURE_A1_LOCK), (V.PROCEDURE_A2, V.PROCEDURE_A2_LOCK)):
        if lock.exists() and V.sha(doc) != V.read(lock)["sha256"]:
            raise SystemExit(f"{doc.name} changed after its lock")
    return lk


# ---------------------------------------------------------------- metadata: blocks, records, sampled parts
def block_tiles() -> dict[int, list[dict]]:
    """Step 0's blocks (connected eligible tiles), reproduced with its own function; DEV/TEST/FIT sizes checked."""
    tiles = [t for t in V.read(STEP0 / "TILES.json")["tiles"] if t["licence_ok"]]
    comps = V.load_module("hr3_step0", V.WORK / "step0_metadata.py").blocks(tiles)
    out = {b: sorted((tiles[i] for i in comp), key=lambda t: t["name"]) for b, comp in enumerate(comps)}
    if {b: len(out[b]) for b in BLOCK_TILES} != BLOCK_TILES or V.read(STEP0 / "PARTS.json")["split"] != SPLIT:
        raise SystemExit("blocks or split differ from Step 0")
    return out


def smr_xy() -> np.ndarray:
    recs = V.read(STEP0 / "SMR_SNAPSHOT.json")["records"]
    return np.array([[r["ITM_E"], r["ITM_N"]] for r in recs if r.get("ITM_E") is not None
                     and r.get("ITM_N") is not None], dtype=float)


def sampled_parts() -> list[dict]:
    d = V.read(DEV_SAMPLE)
    # "role" in these rows is Step 0's superseded 5 km selection (DEV / DEV_UNSAMPLED); membership is the list itself
    if d["n_parts"] != len(d["parts"]) or any(p["block"] != 3 or p["flagged"] for p in d["parts"]):
        raise SystemExit("DEV_SAMPLE_PRIVATE.json is inconsistent")
    return sorted(d["parts"], key=lambda p: p["id"])


# ---------------------------------------------------------------- points (controls, smoke): metadata only
def draw_cores(rng, tiles: list[dict], pos_boxes: list, xy: np.ndarray, n: int, prefix: str) -> tuple[list, dict]:
    """STEP1_PROCEDURE section 3: integer 1 km cores uniform over the tiles' bounding box; inside the tile union, off
    every positive box, centre >= 100 m from every record, no overlap with earlier cores; point = core centre."""
    import shapely
    from shapely import STRtree
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    region = unary_union([box(*tile_box(t)) for t in tiles])
    shapely.prepare(region)
    x0, y0, x1, y1 = region.bounds
    pos_tree = STRtree([box(*b) for b in pos_boxes]) if pos_boxes else None
    pt_tree = STRtree(shapely.points(xy))
    out, rej, draws = [], Counter(), 0
    while len(out) < n:
        draws += 1
        if draws > MAX_DRAWS:
            raise SystemExit(f"{prefix}: {n} cores not found in {MAX_DRAWS} draws")
        x = int(rng.integers(math.floor(x0), math.ceil(x1) - CORE + 1))
        y = int(rng.integers(math.floor(y0), math.ceil(y1) - CORE + 1))
        core = [x, y, x + CORE, y + CORE]
        g, c = box(*core), Point(x + CORE / 2, y + CORE / 2)
        if not region.covers(g):
            rej["outside_tiles"] += 1
        elif pos_tree is not None and len(pos_tree.query(g, predicate="intersects")):
            rej["positive_box"] += 1
        elif len(pt_tree.query(c, predicate="dwithin", distance=SMR_CLEAR_M)):
            rej["record_within_100m"] += 1
        elif any(_intersects(core, o["core"]) for o in out):
            rej["overlaps_earlier_core"] += 1
        else:
            _, d = pt_tree.query_nearest(c, return_distance=True)
            out.append({"id": f"{prefix}{len(out) + 1:02d}" if prefix == "C" else f"{prefix}{len(out) + 1}",
                        "core": core, "cx": x + CORE / 2, "cy": y + CORE / 2, "draw": draws,
                        "smr_nearest_m": round(float(d[0]), 1)})
    return out, {"draws": draws, "rejections": dict(rej)}


def pos_boxes(parts: list[dict]) -> list[list[float]]:
    return [[p["e"] - POS_HALF, p["n"] - POS_HALF, p["e"] + POS_HALF, p["n"] + POS_HALF] for p in parts]


def build_points() -> dict:
    bt, xy, parts = block_tiles(), smr_xy(), sampled_parts()
    controls, cstats = draw_cores(np.random.default_rng([V.SEED, 2]), bt[3], pos_boxes(parts), xy, V.N_CONTROLS, "C")
    smoke, sstats = draw_cores(np.random.default_rng([V.SEED, 3]), bt[4], [], xy, 3, "S")
    for c in controls:
        c["nearest_sampled_part_m"] = round(min(math.hypot(c["cx"] - p["e"], c["cy"] - p["n"]) for p in parts), 1)
    return {"controls": controls, "control_stats": cstats, "smoke": smoke, "smoke_stats": sstats}


def inputs_sha256() -> dict:
    files = [STEP0 / f for f in ("TILES.json", "PARTS.json", "SMR_SNAPSHOT.json", "BLOCKS.json")]
    files += [DEV_SAMPLE, V.PROCEDURE, V.PROCEDURE_LOCK, V.WORK / "step0_metadata.py", V.WORK / "PACKET_PREPIXEL.md",
              V.WORK / "PRE_PIXEL_LOCK.json", V.WORK / "AMENDMENT_1.md", V.WORK / "AMENDMENT_1_LOCK.json"]
    files += [p for p in (V.PROCEDURE_A1, V.PROCEDURE_A1_LOCK, V.PROCEDURE_A2, V.PROCEDURE_A2_LOCK) if p.exists()]
    return {V.rel(p): V.sha(p) for p in files}


def points() -> dict:
    check_procedure_lock()
    if V.POINTS_PRIVATE.exists():
        raise SystemExit("POINTS_PRIVATE.json exists; points are drawn once")
    p = build_points()
    V.create_only(V.POINTS_PRIVATE, {"version": V.VERSION, "utc": V.utc(), **p, "seed_streams": {
        "controls": [V.SEED, 2], "smoke": [V.SEED, 3]}, "inputs_sha256": inputs_sha256(),
        "note": "metadata only (tile boxes, SMR points, sampled part points); no pixel read"})
    return {"controls": len(p["controls"]), "control_stats": p["control_stats"], "smoke_stats": p["smoke_stats"]}


# ---------------------------------------------------------------- fetch: tiles in memory, item windows kept
def fetch_items() -> list[dict]:
    pts = V.read(V.POINTS_PRIVATE)
    items = [{"id": p["id"], "kind": "part", "block": 3, "cx": float(p["e"]), "cy": float(p["n"])}
             for p in sampled_parts()]
    items += [{"id": c["id"], "kind": "control", "block": 3, "cx": c["cx"], "cy": c["cy"]} for c in pts["controls"]]
    items += [{"id": s["id"], "kind": "smoke", "block": 4, "cx": s["cx"], "cy": s["cy"]} for s in pts["smoke"]]
    for it in items:
        it["raw_bounds"] = window_bounds(it["cx"], it["cy"])
    return items


def read_tile(data: bytes, t: dict) -> tuple[np.ndarray, dict]:
    """STEP1_PROCEDURE Amendment 1: the zip's one DTM member (*_dtm.asc or DTM/*.tif) read in memory; the source
    pixels over the coverage box (+2 px) resampled bilinearly (GDAL, nodata-aware) onto the box's integer 1 m ITM grid,
    an identity copy when the grids align; only the box is returned; invalid -> NaN."""
    from rasterio.io import ZipMemoryFile
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window, from_bounds
    names = [m for m in zipfile.ZipFile(io.BytesIO(data)).namelist() if DTM_MEMBER.search(m)]
    if len(names) != 1:
        raise SystemExit(f"{t['name']}: expected one DTM member, found {names}")
    x0, y0, x1, y1 = tile_box(t)
    with ZipMemoryFile(data) as z, z.open(names[0]) as ds:
        epsg = ds.crs.to_epsg() if ds.crs else None
        tr = ds.transform
        if ((ds.crs is not None and epsg != 2157) or abs(ds.res[0] - 1) > 1e-3 or abs(ds.res[1] - 1) > 1e-3
                or tr.b != 0 or tr.d != 0 or tr.e >= 0):
            raise SystemExit(f"{t['name']}: unexpected grid {ds.crs} {tr}")
        w = from_bounds(x0 - 2, y0 - 2, x1 + 2, y1 + 2, tr)
        c0, r0 = math.floor(w.col_off), math.floor(w.row_off)
        win = Window(c0, r0, math.ceil(w.col_off + w.width) - c0, math.ceil(w.row_off + w.height) - r0)
        nodata = ds.nodata
        src = ds.read(1, window=win, boundless=True,
                      fill_value=nodata if nodata is not None else np.nan).astype("float32")
        src_tr = ds.window_transform(win)
        grid = {"origin": [tr.c, tr.f], "pixel": [ds.res[0], ds.res[1]], "size": [ds.width, ds.height],
                "crs_epsg": epsg, "driver": ds.driver}
    bad = ~np.isfinite(src)
    if nodata is not None:
        bad |= src == np.float32(nodata)
    src[bad] = SRC_NODATA
    dst = np.full((y1 - y0, x1 - x0), np.nan, dtype="float32")
    reproject(src, dst, src_transform=src_tr, src_crs=ITM, src_nodata=SRC_NODATA,
              dst_transform=Affine(1.0, 0, x0, 0, -1.0, y1), dst_crs=ITM, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    dst[~np.isfinite(dst) | (dst == SRC_NODATA)] = np.nan
    return dst, {"member": names[0], "nodata": nodata, "source_grid": grid,
                 "valid_share": round(float(np.isfinite(dst).mean()), 6)}


def paste(win: np.ndarray, wb: list[int], a: np.ndarray, tb: list[int]) -> int:
    """Copy tile pixels into the window where the window is still invalid (first valid value wins); returns the number
    of pixels valid in both (overlap)."""
    ox0, ox1, oy0, oy1 = max(wb[0], tb[0]), min(wb[2], tb[2]), max(wb[1], tb[1]), min(wb[3], tb[3])
    if ox0 >= ox1 or oy0 >= oy1:
        return 0
    w = win[wb[3] - oy1:wb[3] - oy0, ox0 - wb[0]:ox1 - wb[0]]
    s = a[tb[3] - oy1:tb[3] - oy0, ox0 - tb[0]:ox1 - tb[0]]
    both = np.isfinite(w) & np.isfinite(s)
    take = ~np.isfinite(w) & np.isfinite(s)
    w[take] = s[take]
    return int(both.sum())


def write_raw(arr: np.ndarray, x0: float, y1: float, out: Path) -> None:
    """English raw-window format (welsh_fetch.write_english_raw) with the Irish CRS."""
    import rasterio
    from rasterio.transform import Affine
    prof = {"driver": "GTiff", "dtype": "float32", "nodata": ENGLISH_NODATA, "width": arr.shape[1],
            "height": arr.shape[0], "count": 1, "crs": ITM, "transform": Affine(1.0, 0, x0, 0, -1.0, y1),
            "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(np.where(np.isnan(arr), ENGLISH_NODATA, arr).astype("float32"), 1)


def read_raw(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1, masked=True).filled(np.nan).astype("float32")


def download(url: str, tries: int = 3) -> bytes:
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:
                expect = int(r.headers.get("Content-Length") or -1)
                data = r.read()
            if expect >= 0 and len(data) != expect:
                raise OSError(f"short read {len(data)} of {expect}")
            return data
        except OSError as e:
            err = e
            time.sleep(5 * (k + 1))
    raise SystemExit(f"{url}: {err}")


def _fetch_windows(items: list[dict], receipt: Path, new_dir: bool) -> dict:
    """Read every needed tile once (in memory), paste the item windows, write them to raw_items/ (create-only per
    file), and write a create-only receipt; TEST tiles are refused."""
    import hashlib
    if receipt.exists() or (new_dir and V.RAW_ITEMS.exists()):
        raise SystemExit(f"{receipt.name} or raw_items exists; this fetch writes once")
    bt = block_tiles()
    test_names = {t["name"] for t in bt[1]}
    need, by_tile = {}, {}
    for it in items:
        it["raw_bounds"] = window_bounds(it["cx"], it["cy"])
        it["tiles"] = sorted(t["name"] for t in bt[it["block"]] if _intersects(tile_box(t), it["raw_bounds"]))
        for name in it["tiles"]:
            need[name] = next(t for t in bt[it["block"]] if t["name"] == name)
            by_tile.setdefault(name, []).append(it)
    if test_names & set(need):
        raise SystemExit("a TEST tile would be read")
    wins = {it["id"]: np.full((WIN, WIN), np.nan, dtype="float32") for it in items}
    overlap, t0, got = Counter(), time.time(), 0
    for name in sorted(need):
        t = need[name]
        data = download(t["url"])
        a, info = read_tile(data, t)
        for it in by_tile[name]:
            overlap[it["id"]] += paste(wins[it["id"]], it["raw_bounds"], a, tile_box(t))
        got += len(data)
        V.append_jsonl(V.FETCH_LEDGER, {"name": name, "url": t["url"], "bytes": len(data),
                                        "sha256": hashlib.sha256(data).hexdigest(), "utc": V.utc(), **info,
                                        "items": len(by_tile[name]), "receipt": receipt.name})
        del a, data
    V.RAW_ITEMS.mkdir(parents=True, exist_ok=not new_dir)
    rows = {}
    for it in items:
        path = V.RAW_ITEMS / f"{it['id']}.tif"
        if path.exists():
            raise SystemExit(f"{path.name} exists; windows are written once")
        write_raw(wins[it["id"]], it["raw_bounds"][0], it["raw_bounds"][3], path)
        if not np.array_equal(np.isfinite(read_raw(path)), np.isfinite(wins[it["id"]])):
            raise SystemExit(f"{it['id']}: the written window does not read back")
        rows[it["id"]] = {**{k: it[k] for k in ("kind", "block", "cx", "cy", "raw_bounds", "tiles")},
                          "raw": V.rel(path), "raw_sha256": V.sha(path), "overlap_pixels": overlap[it["id"]],
                          "valid_share_window": round(float(np.isfinite(wins[it["id"]]).mean()), 6)}
    V.create_only(receipt, {"version": V.VERSION, "utc": V.utc(), "seconds": round(time.time() - t0, 1),
                            "tiles": len(need), "bytes_downloaded": got, "points_sha256": V.sha(V.POINTS_PRIVATE),
                            "items": rows, "procedure_lock_sha256": V.sha(V.PROCEDURE_LOCK)})
    return rows


def fetch() -> dict:
    check_procedure_lock()
    items = fetch_items()
    t0 = time.time()
    rows = _fetch_windows(items, V.FETCH_RECEIPT, new_dir=True)
    return {"items": len(rows), "seconds": round(time.time() - t0, 1),
            "min_valid_share": min(r["valid_share_window"] for r in rows.values()),
            "items_with_overlap": sum(1 for r in rows.values() if r["overlap_pixels"])}


def fetch_rows() -> dict:
    """Windows from the main fetch and, when present, Amendment 2's control fetch (disjoint ids)."""
    rows = dict(V.read(V.FETCH_RECEIPT)["items"])
    if V.FETCH_RECEIPT_A2.exists():
        extra = V.read(V.FETCH_RECEIPT_A2)["items"]
        if set(extra) & set(rows):
            raise SystemExit("Amendment 2 windows reuse an id")
        rows.update(extra)
    return rows


def control_v2(row: dict) -> float:
    path = V.ROOT / row["raw"]
    if V.sha(path) != row["raw_sha256"]:
        raise SystemExit(f"{path.name}: window changed since its fetch")
    return nodata_fracs(read_raw(path), row["raw_bounds"], row["cx"], row["cy"], 0.0)[1]


def controls_a2() -> dict:
    """Amendment 2: candidates are the locked metadata draw in order (the first 20 are POINTS_PRIVATE's controls);
    the controls are the first 20 candidates passing V2 (>= 90 % valid in the 256 m square at the point)."""
    check_procedure_lock()
    if not V.PROCEDURE_A2_LOCK.exists():
        raise SystemExit("Amendment 2 is not locked")
    if V.POINTS_A2_PRIVATE.exists():
        raise SystemExit("POINTS_A2_PRIVATE.json exists")
    bt, xy, parts = block_tiles(), smr_xy(), sampled_parts()
    first = V.read(V.POINTS_PRIVATE)["controls"]
    rows = V.read(V.FETCH_RECEIPT)["items"]
    cands = [{**c, "v2": round(control_v2(rows[c["id"]]), 6)} for c in first]
    needed = V.N_CONTROLS - sum(c["v2"] >= V2_MIN for c in cands)
    stats, fetched = None, {}
    if needed:
        m = len(first) + 3 * needed + 5          # one batch, about 3x the expected need at the observed failure rate
        drawn, stats = draw_cores(np.random.default_rng([V.SEED, 2]), bt[3], pos_boxes(parts), xy, m, "C")
        if [(d["id"], d["core"]) for d in drawn[:len(first)]] != [(c["id"], c["core"]) for c in first]:
            raise SystemExit("the extended draw does not reproduce POINTS_PRIVATE.json")
        fresh = drawn[len(first):]
        for d in fresh:
            d["nearest_sampled_part_m"] = round(min(math.hypot(d["cx"] - p["e"], d["cy"] - p["n"]) for p in parts), 1)
        todo = [{"id": d["id"], "kind": "control", "block": 3, "cx": d["cx"], "cy": d["cy"]} for d in fresh]
        fetched = _fetch_windows(todo, V.FETCH_RECEIPT_A2, new_dir=False)
        cands += [{**d, "v2": round(control_v2(fetched[d["id"]]), 6)} for d in fresh]
    if sum(c["v2"] >= V2_MIN for c in cands) < V.N_CONTROLS:
        raise SystemExit("too few candidates passed V2 in the batch; extending needs a dated amendment")
    chosen = [c for c in cands if c["v2"] >= V2_MIN][:V.N_CONTROLS]
    rec = {"version": V.VERSION, "utc": V.utc(), "rule": "STEP1_PROCEDURE_AMENDMENT_2: first 20 draw-order candidates "
           "with V2 >= 0.90", "controls": chosen,
           "candidates": [{k: c[k] for k in ("id", "core", "cx", "cy", "draw", "v2")} for c in cands],
           "replaced": [c["id"] for c in cands[:len(first)] if c["v2"] < V2_MIN],
           "draw_stats_final": stats, "amendment_lock_sha256": V.sha(V.PROCEDURE_A2_LOCK),
           "points_sha256": V.sha(V.POINTS_PRIVATE), "fetch_receipt_sha256": V.sha(V.FETCH_RECEIPT),
           "fetch_receipt_a2_sha256": V.sha(V.FETCH_RECEIPT_A2) if V.FETCH_RECEIPT_A2.exists() else None}
    V.create_only(V.POINTS_A2_PRIVATE, rec)
    return {"replaced": rec["replaced"], "candidates": len(cands), "chosen": [c["id"] for c in chosen],
            "new_windows": len(fetched)}


# ---------------------------------------------------------------- NODATA (HR2 V1/V2)
def nodata_fracs(arr: np.ndarray, wb: list[int], e: float, n: float, r: float) -> tuple[float, float, int, int]:
    """V1: valid share of pixel centres strictly inside the disc of radius r + 15 m; V2: of those strictly inside
    the 256 m square centred on the point (HR2 uses shapely contains, which excludes the boundary)."""
    xs = wb[0] + 0.5 + np.arange(arr.shape[1])
    ys = wb[3] - 0.5 - np.arange(arr.shape[0])
    dx, dy = xs[None, :] - e, ys[:, None] - n
    disc = dx ** 2 + dy ** 2 < (r + REACH_M) ** 2
    sq = (np.abs(dx) < SQ_HALF) & (np.abs(dy) < SQ_HALF)
    valid = np.isfinite(arr)
    return float(valid[disc].mean()), float(valid[sq].mean()), int(disc.sum()), int(sq.sum())


def nodata() -> dict:
    check_procedure_lock()
    if V.NODATA_PRIVATE.exists():
        raise SystemExit("NODATA_PARTS_PRIVATE.json exists")
    fr = V.read(V.FETCH_RECEIPT)["items"]
    rows = []
    for p in sampled_parts():
        f = fr[p["id"]]
        path = V.ROOT / f["raw"]
        if V.sha(path) != f["raw_sha256"]:
            raise SystemExit(f"{p['id']}: raw window changed since the fetch")
        v1, v2, n1, n2 = nodata_fracs(read_raw(path), f["raw_bounds"], p["e"], p["n"], p["radius_m"])
        rows.append({"id": p["id"], "class": p["class"], "monument_class": p["monument_class"], "sb2": p["sb2"],
                     "valid_r15": round(v1, 6), "valid_sq256": round(v2, 6), "pixels_r15": n1, "pixels_sq256": n2,
                     "eligible": v1 >= V1_MIN and v2 >= V2_MIN})
    el = [r for r in rows if r["eligible"]]
    n_pop = {h: sum(1 for r in el if r["class"] == h) for h in V.STRATA}
    res = {"schema": "hr3_nodata_v1", "version": V.VERSION, "utc": V.utc(),
           "rule": {"V1_r15_min": V1_MIN, "V2_sq256_min": V2_MIN, "reach_m": REACH_M,
                    "valid": "finite and != the grid NODATA_value; no eligible tile of the block = invalid"},
           "sampled": len(rows), "eligible": len(el), "N_POP": n_pop, "floor": V.DEV_FLOOR,
           "floor_met": len(el) >= V.DEV_FLOOR,
           "excluded_by_test": {"V1_only": sum(1 for r in rows if r["valid_r15"] < V1_MIN and r["valid_sq256"] >= V2_MIN),
                                "V2_only": sum(1 for r in rows if r["valid_sq256"] < V2_MIN and r["valid_r15"] >= V1_MIN),
                                "both": sum(1 for r in rows if r["valid_sq256"] < V2_MIN and r["valid_r15"] < V1_MIN)},
           "excluded_by_class": dict(Counter(r["class"] for r in rows if not r["eligible"])),
           "eligible_by_monument_class": dict(Counter(r["monument_class"] for r in el)),
           "eligible_superblocks": len({r["sb2"] for r in el}),
           "fetch_receipt_sha256": V.sha(V.FETCH_RECEIPT), "procedure_lock_sha256": V.sha(V.PROCEDURE_LOCK)}
    V.create_only(V.NODATA_PRIVATE, {**res, "rows": rows})
    V.create_only(V.NODATA_RESULT, {**res, "nodata_private_sha256": V.sha(V.NODATA_PRIVATE)})
    return {k: res[k] for k in ("sampled", "eligible", "N_POP", "floor_met", "excluded_by_test", "excluded_by_class")}


# ---------------------------------------------------------------- audit sample
def eligible_dev() -> list[dict]:
    rows = {r["id"]: r for r in V.read(V.NODATA_PRIVATE)["rows"]}
    dev = [dict(p, stratum=p["class"]) for p in sampled_parts() if rows[p["id"]]["eligible"]]
    if dict(Counter(p["stratum"] for p in dev)) != {h: v for h, v in V.N_POP.items() if v}:
        raise SystemExit(f"eligible strata {Counter(p['stratum'] for p in dev)} differ from vis_common.N_POP")
    return dev


def select_parts(dev: list[dict]) -> list[dict]:
    """HR2's rule: census when n = N, else seeded choice without replacement over ids sorted within the stratum."""
    rng = np.random.default_rng([V.SEED, 1])
    chosen = []
    for h in V.STRATA:
        members = [p for p in dev if p["stratum"] == h]
        n, big = V.N_SAMPLE[h], len(members)
        idx = range(big) if n == big else sorted(int(i) for i in rng.choice(big, size=n, replace=False))
        for i in idx:
            p = members[i]
            chosen.append({"id": p["id"], "class": p["class"], "monument_class": p["monument_class"], "stratum": h,
                           "inclusion_probability": f"{n}/{big}", "pi": n / big, "cx": float(p["e"]),
                           "cy": float(p["n"]), "radius_m": p["radius_m"], "sb2": p["sb2"]})
    return chosen


def build_sample() -> dict:
    return {"parts": select_parts(eligible_dev()), "controls": V.read(V.POINTS_A2_PRIVATE)["controls"]}


def sample() -> dict:
    check_procedure_lock()
    if V.SAMPLE_PRIVATE.exists():
        raise SystemExit("SAMPLE_PRIVATE.json exists; the sample is drawn once")
    s = build_sample()
    V.create_only(V.SAMPLE_PRIVATE, {"schema": "hr3_visibility_sample_v1", "version": V.VERSION, "utc": V.utc(),
                                     "seed": V.SEED, "seed_streams": {"parts": [V.SEED, 1], "controls": [V.SEED, 2]},
                                     **s, "nodata_private_sha256": V.sha(V.NODATA_PRIVATE),
                                     "points_sha256": V.sha(V.POINTS_PRIVATE),
                                     "points_a2_sha256": V.sha(V.POINTS_A2_PRIVATE), "code_sha256": V.code_hashes(),
                                     "note": "drawn from the eligible list and metadata; no pixel value used"})
    return {"parts": dict(Counter(p["stratum"] for p in s["parts"])), "controls": len(s["controls"])}


# ---------------------------------------------------------------- freeze
def freeze_inputs() -> dict:
    out = inputs_sha256()
    for p in (V.POINTS_PRIVATE, V.FETCH_RECEIPT, V.FETCH_LEDGER, V.NODATA_PRIVATE, V.NODATA_RESULT,
              V.POINTS_A2_PRIVATE, V.FETCH_RECEIPT_A2):
        out[V.rel(p)] = V.sha(p)
    return out


def freeze() -> dict:
    if V.FREEZE.exists():
        raise SystemExit("AUDIT_FREEZE.json exists; the freeze is written once")
    if any((V.OUT / n).exists() for n in ("panels", "raw", "smoke_panels", "SEALED_MAPPING.json")):
        raise SystemExit("panels or a mapping already exist; the freeze must come first")
    check_procedure_lock()
    s = V.read(V.SAMPLE_PRIVATE)
    again = build_sample()
    if json.loads(json.dumps(again, default=str)) != {k: s[k] for k in again}:
        raise SystemExit("SAMPLE_PRIVATE.json does not reproduce from its seed and inputs")
    tests = V.TEST_OUTPUT.read_text(encoding="utf-8") if V.TEST_OUTPUT.exists() else ""
    if " passed" not in tests or "failed" in tests or "error" in tests.lower():
        raise SystemExit("TEST_OUTPUT.txt does not show a clean pass")
    nd = V.read(V.NODATA_RESULT)
    if nd["N_POP"] != V.N_POP or not nd["floor_met"]:
        raise SystemExit(f"N_POP {V.N_POP} differs from NODATA_RESULT {nd['N_POP']} or the floor failed")
    rec = {
        "schema": "hr3_visibility_audit_freeze_v1", "version": V.VERSION, "frozen_utc": V.utc(),
        "packet": "work/hr3_ireland_2026-10-02/PACKET_PREPIXEL.md section 3 Step 1; AMENDMENT_1 items 3-5; "
                  "STEP1_PROCEDURE.md",
        "author": "Claude Opus 5.5 (fresh coordinator; builder/operator, not the reviewer), 2026-10-02",
        "note": "Written after the window fetch and NODATA (validity only) and before any panel, neutral folder or "
                "review session. No DEV window has been viewed.",
        "seed": V.SEED, "seed_streams": s["seed_streams"],
        "population": {"N_DEV": V.N_DEV, "by_stratum": V.N_POP,
                       "definition": "sampled DEV parts (AMENDMENT_1 2 km squares) passing HR2 V1/V2 (NODATA_RESULT)"},
        "sample": {"n_by_stratum": V.N_SAMPLE,
                   "rule": "barrows: census if N <= 10, else 10; enclosure classes: 50; numpy default_rng([SEED, 1])"
                           ".choice without replacement over ids sorted within stratum, strata in order " +
                           ", ".join(V.STRATA),
                   "parts": [{k: p[k] for k in ("id", "class", "monument_class", "stratum", "inclusion_probability")}
                             for p in s["parts"]],
                   "controls": {"n": V.N_CONTROLS, "ids": [c["id"] for c in s["controls"]],
                                "rule": "STEP1_PROCEDURE section 3 (metadata draw before any pixel read) and "
                                        "Amendment 2 (first 20 draw-order candidates passing V2)",
                                "replaced_by_amendment_2": V.read(V.POINTS_A2_PRIVATE)["replaced"],
                                "not_applied": "window valid share is reported, never used to select or replace"},
                   "sample_private_sha256": V.sha(V.SAMPLE_PRIVATE)},
        "panel_spec": {"raw_window": f"integer square X +- {V.HALF_WIN} m, X = floor(c + 0.5), centre = SMR point or "
                                     "control point",
                       "adapter": "vis_irl.py read_tile/paste/write_raw: GSI *_dtm.asc read in memory, grid checked "
                                  "against the coverage box, English raw-window format with EPSG:2157",
                       "renderer": "work/enclosure_line_2026-09-24/enc_step1.py read_window + render_panel, unchanged "
                                   "(V0-V4 170 m, V5 340 m, 1 m pixels, edge ticks, 1566 x 1160 px)",
                       "qa": "panel size 1566 x 1160 and grey SD > 5; builder views smoke panels S1-S3 only",
                       "codes": "Q01-Q80 from secrets.SystemRandom().shuffle; SEALED_MAPPING.json create-only, read "
                                "only by join after LOCK.json",
                       "smoke_panels": "S1-S3: FIT-block points by the control rule (default_rng([SEED, 3]))"},
        "question": V.QUESTION, "answers": list(V.ANSWERS), "forms": list(V.FORMS),
        "reviewer": {"route": "work/discovery_cont_2026-09-25/review/ops/run_review_v2.sh unchanged (claude-opus-5, "
                              "--safe-mode --restricted, Read only, confined to the session folder)",
                     "prompt": "work/holdout_region_2026-10/step2b/REVIEWER_PROMPT.md (byte-identical)",
                     "schema": "work/holdout_region_2026-10/step2b/SCHEMA.json (byte-identical)",
                     "sessions": "one panel per session in a fresh folder C:/dev/" + V.NEUTRAL_PREFIX +
                                 "_<code>_a<n> holding MANIFEST.json, SCHEMA.json, REVIEWER_PROMPT.md, panels/<code>.png",
                     "limits": {"max_attempts_per_code": V.MAX_ATTEMPTS,
                                "stop_after_consecutive_failures": V.MAX_CONSECUTIVE_FAILURES,
                                "spend_cap_usd": V.SPEND_CAP_USD, "session_budget_usd": V.SESSION_BUDGET_USD},
                     "acceptance": "unchanged review_ops.audit_session accepts the record with no problem; final "
                                   "result without error; model claude-opus-5; tools [Read]; no MCP; folder unchanged",
                     "smoke": "canary smoke first (HR2 smoke_vis.sh with HR3 paths)",
                     "reviewer_kind": "model", "human_confirmed": False},
        "lock": "create-only LOCK.json of verdicts, audit and every transcript before SEALED_MAPPING.json is read",
        "estimator": {"v_hat": f"stratified Horvitz-Thompson DEV-weighted VISIBLE rate over N = {V.N_DEV}: "
                               "sum_h N_h k_h / n_h / N, strata barrow and enclosure classes",
                      "se": "sqrt(sum_h (N_h/N)^2 (1 - n_h/N_h) s_h^2 / n_h), s_h^2 = p_h (1 - p_h) n_h / (n_h - 1); "
                            "census strata add 0",
                      "unjudged": "a code with no accepted record after 2 attempts is UNJUDGED and counts as not "
                                  "VISIBLE",
                      "reported": "overall, per stratum, each answer; controls separately (k/20, Wilson 95 %)"},
        "decision_table": list(V.DECISION_TABLE), "decision_basis": V.DECISION_BASIS,
        "decision_reading": "the row is read from the exact overall point estimate v_hat; the audit never changes "
                            "the eligible set, the denominators or the Step 2 rule",
        "forecasts": {k: {"event": e, "p": p} for k, (e, p) in V.FORECASTS.items()},
        "declared_deviations": [
            "zips are read in memory and not kept (STEP1_PROCEDURE section 1); item windows were written at fetch "
            "time, before this freeze, and only their validity was computed",
            "background controls follow AMENDMENT_1 item 4 (record clearance at the point, not the HR2 core rule) "
            "and STEP1_PROCEDURE Amendment 2 (V2 at the point; written after control validity was known)",
            "the 256 m and disc tests use strict inequalities, matching HR2's shapely contains",
            "spend cap USD 15 (PACKET section 5); retries within the cap"],
        "tests": {"output_sha256": V.sha(V.TEST_OUTPUT), "summary": tests.strip().splitlines()[-1]},
        "code_sha256": V.code_hashes(), "reused_sha256": V.verify_reused(), "inputs_sha256": freeze_inputs(),
        "derived_from_sha256": {p: V.sha(V.ROOT / p) for p in V.DERIVED_FROM},
    }
    V.create_only(V.FREEZE, rec)
    return {"frozen_utc": rec["frozen_utc"], "freeze_sha256": V.sha(V.FREEZE)}


def check_freeze() -> dict:
    fr = V.read(V.FREEZE)
    bad = [f for f, h in fr["code_sha256"].items() if V.sha(V.WORK / f) != h]
    bad += [f for f, h in fr["reused_sha256"].items() if V.sha(V.ROOT / f) != h]
    bad += [f for f, h in fr["inputs_sha256"].items() if V.sha(V.ROOT / f) != h]
    if V.sha(V.SAMPLE_PRIVATE) != fr["sample"]["sample_private_sha256"]:
        bad.append("SAMPLE_PRIVATE.json")
    if bad:
        raise SystemExit(f"frozen files changed: {bad}")
    return fr


# ---------------------------------------------------------------- items
def items() -> dict:
    check_freeze()
    fr = fetch_rows()
    s, pts = V.read(V.SAMPLE_PRIVATE), V.read(V.POINTS_PRIVATE)

    def item(ident: str, kind: str, extra: dict) -> dict:
        f = fr[ident]
        return {"id": ident, "kind": kind, "cx": f["cx"], "cy": f["cy"], "raw_bounds": f["raw_bounds"],
                "raw": f["raw"], "raw_sha256": f["raw_sha256"], "valid_share_window": f["valid_share_window"],
                "tiles": f["tiles"], **extra}
    out = [item(p["id"], "part", {k: p[k] for k in ("class", "monument_class", "stratum", "inclusion_probability",
                                                     "radius_m", "sb2")}) for p in s["parts"]]
    out += [item(c["id"], "control", {"class": None, "monument_class": None, "stratum": "control", "core": c["core"],
                                      "smr_nearest_m": c["smr_nearest_m"],
                                      "nearest_sampled_part_m": c["nearest_sampled_part_m"]}) for c in s["controls"]]
    smoke = [item(x["id"], "smoke", {"core": x["core"]}) for x in pts["smoke"]]
    if len(out) != V.N_ITEMS or len(smoke) != 3:
        raise SystemExit("items incomplete")
    rec = {"version": V.VERSION, "freeze_sha256": V.sha(V.FREEZE), "items": out, "smoke": smoke}
    if V.ITEMS_PRIVATE.exists():
        if {k: v for k, v in V.read(V.ITEMS_PRIVATE).items() if k != "utc"} != json.loads(json.dumps(rec)):
            raise SystemExit("ITEMS_PRIVATE.json exists with different content")
    else:
        V.create_only(V.ITEMS_PRIVATE, {**rec, "utc": V.utc()})
    return {"items": len(out), "smoke": len(smoke),
            "min_valid_share": min(i["valid_share_window"] for i in out + smoke)}


# ---------------------------------------------------------------- render
def read_square(enc, raw_path: Path, cx: float, cy: float) -> tuple:
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(raw_path) as src:
        if abs(src.res[0] - 1.0) > 1e-9 or abs(src.res[1] - 1.0) > 1e-9:
            raise SystemExit(f"{raw_path}: expected a 1 m raw window")
        col0, row0, inside = enc.read_window(src, cx, cy)
        a = src.read(1, window=Window(col0, row0, enc.READ_M, enc.READ_M), boundless=True, masked=True)
        z = a.filled(np.nan).astype("float64")
    return z, inside, col0, row0


def panel_ok(png: Path) -> bool:
    from PIL import Image
    with Image.open(png) as im:
        size = im.size
        a = np.asarray(im.convert("L"), dtype=float)
    return size == (1566, 1160) and float(a.std()) > 5.0


_R2: dict = {}


def v1_detector_r(enc, raw_path: Path, z, col0: int, row0: int):
    """Provenance diagnostic only (never shown to the reviewer), as HR2: Pearson r between V1 and the detector-input
    relief (r2_train.relief_window, preprocessing only, no model) averaged 2 x 2 back to the 1 m grid."""
    try:
        if "m" not in _R2:
            _R2["m"] = V.load_module("r2_train_hr3vis", V.ROOT / "work/campaign_2026-09-22/r2_train.py")
        det = _R2["m"].relief_window(raw_path).astype("float64") - 0.5
        h, w = det.shape[0] // 2, det.shape[1] // 2
        det1 = det.reshape(h, 2, w, 2).mean(axis=(1, 3))[row0:row0 + enc.READ_M, col0:col0 + enc.READ_M]
        v1 = np.clip(enc.relief_views(z)["lrm10"], -0.5, 0.5)
        a, b, fin = enc.close(v1), enc.close(det1), enc.close(np.isfinite(z) & np.isfinite(v1))
        if fin.sum() < 100 or np.std(a[fin]) == 0 or np.std(b[fin]) == 0:
            return None
        return round(float(np.corrcoef(a[fin], b[fin])[0, 1]), 4)
    except Exception as e:  # diagnostic only; recorded, never blocks the render
        return f"unavailable: {type(e).__name__}: {str(e)[:120]}"


def _render_set(enc, coded: dict, out_dir: Path) -> dict:
    """Render in a seeded order, copy in code order (file times carry no order); per-code stats only."""
    out_dir.mkdir(parents=True, exist_ok=False)
    stats = {}
    with tempfile.TemporaryDirectory(dir=str(V.OUT)) as tmp:
        for code, it in sorted(coded.items(), key=lambda kv: seeded("render", kv[1]["id"])):
            raw_path = V.ROOT / it["raw"]
            if V.sha(raw_path) != it["raw_sha256"]:
                raise SystemExit(f"{code}: raw window changed since the fetch")
            z, inside, col0, row0 = read_square(enc, raw_path, it["cx"], it["cy"])
            if not inside:
                raise SystemExit(f"{code}: 340 m square outside its raw window")
            png = Path(tmp) / f"{code}.png"
            s = enc.render_panel(code, z, png)
            if not panel_ok(png):
                raise SystemExit(f"{code}: panel failed the size/contrast check")
            stats[code] = {**s, "square_inside_raw": inside, "raw_sha256": it["raw_sha256"],
                           "valid_share_window": it["valid_share_window"],
                           "v1_detector_r": v1_detector_r(enc, raw_path, z, col0, row0)}
        for code in sorted(stats):
            shutil.copyfile(Path(tmp) / f"{code}.png", out_dir / f"{code}.png")
            stats[code]["png_sha256"] = V.sha(out_dir / f"{code}.png")
    return stats


def render() -> dict:
    check_freeze()
    if V.RENDER_RECEIPT.exists() or V.MAPPING.exists():
        raise SystemExit("already rendered (or a mapping exists); a re-render needs a dated amendment")
    it = V.read(V.ITEMS_PRIVATE)
    V.verify_reused()
    enc = V.load_module("enc_step1_hr3vis", V.ROOT / "work/enclosure_line_2026-09-24/enc_step1.py")
    shuffled = list(it["items"])
    secrets.SystemRandom().shuffle(shuffled)
    coded = {f"{V.PREFIX}{i + 1:02d}": x for i, x in enumerate(shuffled)}
    V.create_only(V.MAPPING, {"version": V.VERSION, "utc": V.utc(), "items_sha256": V.sha(V.ITEMS_PRIVATE),
                              "drawn_with": "secrets.SystemRandom().shuffle",
                              "warning": "sealed code-to-identity mapping; never in a neutral folder; read only by "
                                         "join after LOCK.json exists", "codes": coded})
    stats = _render_set(enc, coded, V.PANELS)
    sstats = _render_set(enc, {x["id"]: x for x in it["smoke"]}, V.SMOKE_PANELS)
    rs = [s["v1_detector_r"] for s in stats.values() if isinstance(s["v1_detector_r"], float)]
    rec = {"version": V.VERSION, "utc": V.utc(), "mapping_sha256": V.sha(V.MAPPING),
           "items_sha256": V.sha(V.ITEMS_PRIVATE), "freeze_sha256": V.sha(V.FREEZE),
           "renderer": {"module": "work/enclosure_line_2026-09-24/enc_step1.py", "functions": ["read_window",
                        "render_panel"], "read_m": enc.READ_M, "close_m": enc.CLOSE_M},
           "adapter": {"module": "work/hr3_ireland_2026-10-02/visibility/vis_irl.py",
                       "sha256": V.sha(HERE / "vis_irl.py")},
           "view_titles": enc.VIEW_TITLES, "panels": stats, "smoke_panels": sstats,
           "void_flags_gt_5pct_close": sorted(c for c, s in stats.items() if s["void_share_close"] > 0.05),
           "v1_detector_r": {"n": len(rs), "min": min(rs) if rs else None,
                             "median": sorted(rs)[len(rs) // 2] if rs else None,
                             "unavailable": sum(1 for s in stats.values() if isinstance(s["v1_detector_r"], str))}}
    V.create_only(V.RENDER_RECEIPT, rec)
    files = [V.ITEMS_PRIVATE, V.MAPPING, V.RENDER_RECEIPT]
    V.create_only(V.RENDER_INPUTS, {"utc": V.utc(), "freeze_sha256": V.sha(V.FREEZE),
                                    "files": {V.rel(p): V.sha(p) for p in files},
                                    "note": "after render, before the smoke and any review session; the mapping is "
                                            "hashed, not read"})
    return {"panels": len(stats), "smoke_panels": len(sstats), "void_flags": rec["void_flags_gt_5pct_close"],
            "v1_detector_r": rec["v1_detector_r"],
            "min_valid_share_window": min(s["valid_share_window"] for s in stats.values())}


def main(argv: list) -> object:
    cmd = argv[1] if len(argv) > 1 else ""
    fn = {"points": points, "fetch": fetch, "controls-a2": controls_a2, "nodata": nodata, "sample": sample, "freeze": freeze, "items": items,
          "render": render, "check-freeze": lambda: check_freeze() and "ok"}.get(cmd)
    if fn is None or len(argv) != 2:
        raise SystemExit(__doc__)
    return fn()


if __name__ == "__main__":
    print(json.dumps(main(sys.argv), indent=1, default=str))
