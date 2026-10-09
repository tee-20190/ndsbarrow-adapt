"""FRESH-1 Step 2: the LGLN DGM1 reader (the only new data code of the chain).

Reads the staging plan (plan_fresh.py). Each needed DGM1 tile is downloaded once into memory from the official URL
that Step 0 recorded (TILE_INDEX_PRIVATE.json); its bytes and sha256 must equal Step 0's read ledger
(TILE_LEDGER_PRIVATE.jsonl), and its grid must be Step 0's contract (EPSG:25832, one float32 band, 1000 x 1000, 1 m,
origin at the tile corner). Valid = finite and not NoData (-9999 and the file value). Pixels whose centres lie outside
the Verden polygon become NoData before any use (STEP0_DECISIONS item 1; the masks are step0.inside_mask, precomputed
by plan_fresh.py and pinned by sha256). Tiles are pasted into each window in name order with vis_irl.paste (first
valid value wins), so on the 19 two-delivery squares the older delivery wins (STEP0_DECISIONS item 2). Windows are
written in the English raw-window format (float32 GeoTIFF, 1 m, nodata float32 min, 256 px blocks, no compression)
with EPSG:25832, so the frozen render (adapter.source_view bilinear 0.5 m, SLRM 20 cells, +-0.5 m) applies unchanged.
Tile bytes never reach the disk.

Refusals before any request (exit 3): a tile outside Step 0's list (name, url, box, ledger pins), an off-host url, a
window outside the district envelope or not in the inference manifest, a plan made for another manifest.

  python -B lgln_reader.py check-roles PLAN TILE_INDEX TILE_LEDGER --manifest main=M --manifest enclosure=E [--only ..]
  python -B lgln_reader.py stage PLAN TILE_INDEX TILE_LEDGER MASKS WIN_DIR LOG.jsonl --manifest ... [--workers 8]
                                [--only ID ...]
  python -B lgln_reader.py verify PLAN TILE_INDEX TILE_LEDGER WIN_DIR LOG.jsonl RECEIPT.json --manifest ... [--only ..]
  python -B lgln_reader.py void MANIFEST WIN_DIR OUT.json [--only ID ...]   (descriptive tiles: share only)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_common as F  # noqa: E402

sys.path.insert(0, str(F.HR3_VIS))
import vis_irl as I  # noqa: E402  frozen HR3 adapter: paste (first valid value wins), ENGLISH_NODATA

ROLE_EXIT, FETCH_EXIT, VERIFY_EXIT = 3, 4, 5
USER_AGENT = "fresh1-step2-reader/1.0"
ENGLISH_NODATA = I.ENGLISH_NODATA                           # float32 min (welsh_fetch / vis_irl English format)


class RoleError(RuntimeError):
    """A tile or window lies outside the Step 0 terrain of Landkreis Verden: nothing is fetched."""


# ---------------------------------------------------------------- Step 0 records and role checks
def ledger_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["name"]] = r
    return rows


def select(plan: dict, only: list[str] | None) -> list[dict]:
    wins = plan["windows"]
    if only:
        missing = sorted(set(only) - {w["id"] for w in wins})
        if missing:
            raise RoleError(f"unknown window ids: {missing[:5]}")
        wins = [w for w in wins if w["id"] in set(only)]
    return wins


def tile_violations(name: str, t: dict | None, index: dict, ledger: dict) -> list[str]:
    """Every plan tile must be a Step 0 record with Step 0's url, box and read pins (status ok)."""
    if t is None:
        return [f"{name}: missing from the plan's tile table"]
    rec, lg, bad = index.get(name), ledger.get(name), []
    if rec is None:
        return [f"{name}: not in Step 0's tile list (TILE_INDEX_PRIVATE.json)"]
    try:
        F.check_url(name, t["url"])
        if F.tile_name_box(name) != t["box"]:
            bad.append(f"{name}: box differs from the tile name")
    except F.RefusedTerrain as e:
        bad.append(str(e))
    if t["url"] != rec["url"] or t["box"] != rec["box"]:
        bad.append(f"{name}: url or box differs from Step 0's record")
    if lg is None or lg.get("status") != "ok" or lg.get("url") != rec["url"]:
        bad.append(f"{name}: no successful Step 0 read")
    elif t.get("bytes") != lg["bytes"] or t.get("sha256") != lg["sha256"] or t.get("inside_px") != lg["inside_px"]:
        bad.append(f"{name}: pins differ from Step 0's read ledger")
    if t.get("inside") not in ("all", "mask") or (t["inside"] == "all") != (t.get("inside_px") == F.TILE_PX ** 2):
        bad.append(f"{name}: inside mode inconsistent with its inside pixel count")
    return bad


def manifest_windows(plan: dict, manifests: dict[str, dict]) -> dict[str, tuple[str, list]]:
    """{window id: (manifest tag, bounds)}; the plan must bind exactly these manifests (canonical digests)."""
    if set(plan.get("manifests", {})) != set(manifests):
        raise RoleError(f"the plan binds manifests {sorted(plan.get('manifests', {}))}, given {sorted(manifests)}")
    out = {}
    for tag, m in manifests.items():
        F.check_region(m)
        if plan["manifests"][tag] != F.digest(m):
            raise RoleError(f"the staging plan was made for another {tag} manifest")
        for t in m["tiles"]:
            if len(t["windows"]) != 1 or t["windows"][0]["core_index"] != "all":
                raise RoleError(f"{t['id']}: one window owning all rectangles is required (infer_irl contract)")
            out[t["windows"][0]["id"]] = (tag, t["windows"][0]["bounds"])
    return out


def role_check(plan: dict, manifests: dict[str, dict], index_doc: dict, ledger: dict,
               only: list[str] | None = None) -> dict:
    mwins = manifest_windows(plan, manifests)
    index = {r["name"]: r for r in index_doc["selected"]}
    env = plan["district_envelope"]
    lim = [env[0] - F.DISTRICT_MARGIN_M, env[1] - F.DISTRICT_MARGIN_M, env[2] + F.DISTRICT_MARGIN_M,
           env[3] + F.DISTRICT_MARGIN_M]
    bad, names = [], set()
    for w in select(plan, only):
        if mwins.get(w["id"]) != (w.get("manifest"), w["bounds"]):
            bad.append(f"{w['id']}: not a window of its manifest (or other bounds)")
        b = w["bounds"]
        if not (lim[0] <= b[0] and lim[1] <= b[1] and b[2] <= lim[2] and b[3] <= lim[3]):
            bad.append(f"{w['id']}: outside the district envelope")
        for name in w["tiles"]:
            t = plan["tiles"].get(name)
            bad += tile_violations(name, t, index, ledger)
            if t is not None and not F.boxes_overlap(t["box"], b):
                bad.append(f"{w['id']}: tile {name} does not meet the window")
            names.add(name)
        if w["tiles"] != sorted(w["tiles"]):
            bad.append(f"{w['id']}: tiles not in name order")
    if bad:
        raise RoleError(f"{len(bad)} role violation(s); nothing fetched: " + "; ".join(bad[:6]))
    return {"windows": len(select(plan, only)), "tiles": len(names), "status": "PASS"}


# ---------------------------------------------------------------- tiles: download, decode, district mask
def download(url: str, tries: int = 3, timeout: int = 300) -> bytes:
    err: Exception | None = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                expect = int(r.headers.get("Content-Length") or -1)
                data = r.read()
            if expect >= 0 and len(data) != expect:
                raise OSError(f"short read {len(data)} of {expect}")
            return data
        except (OSError, ValueError) as e:
            err = e
            time.sleep(5 * (k + 1))
    raise SystemExit(f"{url.rsplit('/', 1)[-1]}: download failed ({type(err).__name__})")


def decode_tile(data: bytes, box: list[int]) -> tuple[np.ndarray, dict]:
    """Step 0's grid contract (step0.tile_validity) and validity rule; invalid -> NaN; values unchanged (no
    resampling: the tile grid is the window grid)."""
    from rasterio.io import MemoryFile
    with MemoryFile(data) as mf, mf.open() as ds:
        tr = ds.transform
        epsg = ds.crs.to_epsg() if ds.crs else None
        ok = (epsg == F.EPSG and ds.count == 1 and ds.width == F.TILE_PX and ds.height == F.TILE_PX
              and ds.dtypes[0] == "float32" and (tr.a, tr.b, tr.d, tr.e) == (1.0, 0.0, 0.0, -1.0)
              and tr.c == box[0] and tr.f == box[3])
        meta = {"crs_epsg": epsg, "dtype": ds.dtypes[0], "nodata": ds.nodata, "grid_ok": bool(ok)}
        if not ok:
            raise SystemExit(f"tile at {box[0]}/{box[1]}: grid differs from Step 0's contract: {meta}")
        a = ds.read(1)
    bad = ~np.isfinite(a) | (a == np.float32(F.SRC_NODATA))
    if meta["nodata"] is not None and math.isfinite(meta["nodata"]):
        bad |= a == np.float32(meta["nodata"])
    a = np.array(a, dtype="float32", copy=True)
    a[bad] = np.nan
    return a, meta


def inside_mask(masks, name: str, t: dict) -> np.ndarray:
    if t["inside"] == "all":
        return np.ones((F.TILE_PX, F.TILE_PX), dtype=bool)
    m = np.unpackbits(masks[name], count=F.TILE_PX * F.TILE_PX).reshape(F.TILE_PX, F.TILE_PX).astype(bool)
    if int(m.sum()) != t["inside_px"]:
        raise SystemExit(f"{name}: district mask has {int(m.sum())} inside pixels, Step 0 recorded {t['inside_px']}")
    return m


def fetch_tile(name: str, t: dict, masks) -> tuple[np.ndarray, dict]:
    data = download(t["url"])
    h = F.sha256_bytes(data)
    if len(data) != t["bytes"] or h != t["sha256"]:
        raise SystemExit(f"{name}: payload differs from Step 0's read ledger ({len(data)} B, {h[:12]}); stop")
    a, meta = decode_tile(data, t["box"])
    del data
    inside = inside_mask(masks, name, t)
    a[~inside] = np.nan                                       # other districts' values are never used
    return a, {"name": name, "bytes": t["bytes"], "sha256": h, "utc": F.utc(), "inside_px": int(inside.sum()),
               "valid_px": int(np.isfinite(a).sum()), **meta}


# ---------------------------------------------------------------- windows
def write_raw(arr: np.ndarray, x0: float, y1: float, out: Path) -> None:
    """English raw-window format (vis_irl.write_raw / welsh_fetch.write_english_raw) with EPSG:25832."""
    import rasterio
    from rasterio.transform import Affine
    prof = {"driver": "GTiff", "dtype": "float32", "nodata": ENGLISH_NODATA, "width": arr.shape[1],
            "height": arr.shape[0], "count": 1, "crs": F.CRS, "transform": Affine(1.0, 0, x0, 0, -1.0, y1),
            "tiled": True, "blockxsize": 256, "blockysize": 256}
    tmp = out.with_name(out.name + ".part")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(np.where(np.isnan(arr), ENGLISH_NODATA, arr).astype("float32"), 1)
    tmp.replace(out)


def read_raw(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(1, masked=True).filled(np.nan).astype("float32")


def mosaic(bounds: list[int], names: list[str], tiles: dict[str, np.ndarray], boxes: dict[str, list[int]]) -> tuple:
    x0, y0, x1, y1 = bounds
    arr = np.full((y1 - y0, x1 - x0), np.nan, dtype="float32")
    overlap = 0
    for name in sorted(names):                                # name order: older delivery first on a shared square
        overlap += I.paste(arr, bounds, tiles[name], boxes[name])
    return arr, overlap


def render_window(w: dict, tiles: dict, boxes: dict, out: Path) -> dict:
    arr, overlap = mosaic(w["bounds"], w["tiles"], tiles, boxes)
    write_raw(arr, w["bounds"][0], w["bounds"][3], out)
    back = read_raw(out)
    fin = np.isfinite(arr)
    if not np.array_equal(np.isfinite(back), fin) or not np.array_equal(back[fin], arr[fin]):
        raise SystemExit(f"{w['id']}: the written window does not read back")
    return {"id": w["id"], "sha256": F.sha256_file(out), "valid_fraction": float(fin.mean()),
            "overlap_pixels": overlap}


def load_masks(path: Path, plan: dict):
    if F.sha256_file(path) != plan["inside_masks_sha256"]:
        raise SystemExit("district masks differ from the plan's pinned sha256")
    return np.load(path)


def stage(plan: dict, manifests: dict, index_doc: dict, ledger: dict, masks_path: Path, win_dir: Path, log: Path,
          workers: int = 8, only: list[str] | None = None) -> dict:
    role_check(plan, manifests, index_doc, ledger, only)
    wins = select(plan, only)
    names = sorted({n for w in wins for n in w["tiles"]})
    masks = load_masks(masks_path, plan)
    win_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    tiles, boxes = {}, {n: plan["tiles"][n]["box"] for n in names}
    with ThreadPoolExecutor(max_workers=workers) as pool, log.open("a", encoding="utf-8", newline="\n") as f:
        for name, (a, info) in zip(names, pool.map(lambda n: fetch_tile(n, plan["tiles"][n], masks), names)):
            tiles[name] = a
            f.write(json.dumps(info) + "\n")
            f.flush()
    t_fetch = time.monotonic() - t0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda w: render_window(w, tiles, boxes, win_dir / f"{w['id']}.tif"), wins))
    F.write_json(win_dir / "RENDER_LOG.json", {"utc": F.utc(), "rows": rows})
    return {"tiles": len(names), "bytes": int(sum(plan["tiles"][n]["bytes"] for n in names)), "windows": len(rows),
            "fetch_decode_s": round(t_fetch, 1), "wall_s": round(time.monotonic() - t0, 1),
            "min_valid_fraction": min((r["valid_fraction"] for r in rows), default=None)}


def verify(plan: dict, manifests: dict, index_doc: dict, ledger: dict, win_dir: Path, log: Path,
           only: list[str] | None) -> dict:
    """Receipt; exit 5 unless every selected window exists with its rendered sha256, every needed tile was logged
    with its Step 0 pins, and no tile outside the Step 0 list (or the plan) was logged."""
    role_check(plan, manifests, index_doc, ledger, only)
    wins, bad = select(plan, only), []
    rendered = {r["id"]: r for r in F.read_json(win_dir / "RENDER_LOG.json")["rows"]} \
        if (win_dir / "RENDER_LOG.json").exists() else {}
    for w in wins:
        p = win_dir / f"{w['id']}.tif"
        if not p.exists() or w["id"] not in rendered or F.sha256_file(p) != rendered[w["id"]]["sha256"]:
            bad.append(f"{w['id']}: raw window missing or changed since the render")
    logged = ledger_rows(log) if log.exists() else {}
    for name in {n for w in wins for n in w["tiles"]}:
        t, r = plan["tiles"][name], logged.get(name)
        if r is None or r["sha256"] != t["sha256"] or r["bytes"] != t["bytes"]:
            bad.append(f"{name}: not logged with its Step 0 pins")
    extra = sorted(set(logged) - set(plan["tiles"]))
    if extra:
        bad.append(f"{len(extra)} logged tile(s) outside the plan")
    return {"utc": F.utc(), "windows": len(wins), "tiles_logged": len(logged), "status": "FAIL" if bad else "PASS",
            "violations": bad[:20]}


def box_pixels(bounds: list[int], b: list[float]) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1): window pixels whose centres (x0 + c + 0.5, y1 - r - 0.5) lie in the half-open box
    x0 <= e < x1, y0 <= n < y1 (region_score.in_core)."""
    c0, c1 = math.ceil(b[0] - bounds[0] - 0.5), math.ceil(b[2] - bounds[0] - 0.5)
    r0, r1 = math.floor(bounds[3] - b[3] - 0.5) + 1, math.floor(bounds[3] - b[1] - 0.5) + 1
    return r0, r1, c0, c1


def owned_share(arr: np.ndarray, bounds: list[int], boxes: list[list[float]]) -> tuple[int, int]:
    """(pixels, finite pixels) of the 1 m raw window whose centres lie in the tile's half-open boxes."""
    n = fin = 0
    for b in boxes:
        r0, r1, c0, c1 = box_pixels(bounds, b)
        sub = arr[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
        n += sub.size
        fin += int(np.isfinite(sub).sum())
    return n, fin


def void(manifest: dict, win_dir: Path, only: list[str] | None) -> dict:
    """Descriptive void share of every tile's scored boxes, plus the Step 0 recheck: every background rectangle was
    accepted only where 1 m validity (district-masked) was complete after a 20 m erosion, so it must be wholly finite
    in the rendered window; a mismatch means the reader and Step 0 disagree (exit 5)."""
    F.check_region(manifest)
    rows, bad = [], []
    for t in manifest["tiles"]:
        w = t["windows"][0]
        p = win_dir / f"{w['id']}.tif"
        if (only and w["id"] not in only) or not p.exists():
            continue
        n, fin = owned_share(read_raw(p), w["bounds"], t["core"])
        rows.append({"window": w["id"], "tile": t["id"], "role": t["role"], "void_share": 1 - fin / n if n else None})
        if t["role"] == "background" and fin != n:
            bad.append(t["id"])
    out = {"rule": "void = non-finite share of the 1 m raw window over pixel centres in the tile's scored boxes",
           "windows": rows, "background_rectangles_not_wholly_valid": bad}
    for role in ("background", "positive", "descriptive"):
        v = [r["void_share"] for r in rows if r["role"] == role and r["void_share"] is not None]
        out[role] = {"windows": len(v), "mean_void_share": sum(v) / len(v) if v else None,
                     "max_void_share": max(v) if v else None}
    return out


def parse_manifests(specs: list[str]) -> dict[str, dict]:
    out = {}
    for s in specs:
        tag, _, path = s.partition("=")
        if not tag or not path or tag in out:
            raise SystemExit(f"--manifest {s!r}: use TAG=PATH once per tag")
        F.check_path(path)
        out[tag] = F.read_json(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check-roles", "stage", "verify", "void"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--manifest", action="append", default=[], help="TAG=PATH (every manifest the plan binds)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    x = a.args
    try:
        if a.cmd == "void":
            out = void(F.read_json(x[0]), Path(x[1]), a.only)
            F.write_json(Path(x[2]), out)
            print(json.dumps({k: out[k] for k in ("background", "positive", "descriptive")}))
            if out["background_rectangles_not_wholly_valid"]:
                raise SystemExit(VERIFY_EXIT)
            return
        plan, manifests = F.read_json(x[0]), parse_manifests(a.manifest)
        index_doc, ledger = F.read_json(x[1]), ledger_rows(Path(x[2]))
        if a.cmd == "check-roles":
            out = role_check(plan, manifests, index_doc, ledger, a.only)
        elif a.cmd == "stage":
            out = stage(plan, manifests, index_doc, ledger, Path(x[3]), Path(x[4]), Path(x[5]), a.workers, a.only)
        else:
            out = verify(plan, manifests, index_doc, ledger, Path(x[3]), Path(x[4]), a.only)
            F.write_json(Path(x[5]), out)
            if out["status"] != "PASS":
                print(json.dumps(out))
                raise SystemExit(VERIFY_EXIT)
    except (RoleError, F.RefusedTerrain) as e:
        print(f"ROLE CHECK FAILED: {e}", file=sys.stderr)
        raise SystemExit(ROLE_EXIT)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
