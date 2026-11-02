"""Addendum E: the Lower Saxony DGM1 core-mosaic builder (CONTEXT-RESCUE-1 FREEZE.md §2.4) and its parity test.

Mosaic (§2.4): one per 5 km core (EPSG:25832; i = floor(x / 5000), j = floor(y / 5000)) covering the core plus a 256 m
halo on the 1 m grid: 5,512 x 5,512 cells, upper-left corner (5000 i - 256, 5000 (j + 1) + 256). Each 1 km square takes
the tile `inventory_b_faint.tile_url` selects (the index entry with the latest Aktualitaet), read by
`inventory_b_faint.load_tile`'s rules (masked -> NaN, < -1000 or > 1e30 -> NaN, 1 m, 1000 x 1000, corner check). Cells
are copied without resampling or gap fill; a square without a tile stays NaN. Output: float32 GeoTIFF, NaN nodata,
deflate, 256 px blocks (stage A's mosaic convention), written atomically, with a receipt (tile urls and sha256,
values_sha256 over the C-order float32 cells, valid km²). `build` also writes the region's PREP.json for sb_infer.py.

Index: one envelope query per core (same FeatureServer, fields and host check as tile_url) instead of one point query
per square. A square's candidates are the features whose file name names that square; the latest Aktualitaet wins, as in
tile_url. A square whose best Aktualitaet is tied falls back to tile_url's own point query. The parity test proves the
envelope index equal to tile_url on every square of the two parity cores.

Reads: every public request (index and tile hosts alike) starts >= GAP_S after the previous start, from one pacer; up to
WORKERS downloads overlap; HTTP 429 stops the run; 3 attempts as inventory_b_faint.get. Tile bytes are cached by file
name with a sha256 sidecar and pinned in the receipts.

Parity (Addendum E): two Verden development cores (DEV_CORES_PRIVATE.json), the two with the largest overlap with the
six FRESH-1 Verden-parity windows (ties: lower (i, j)). For each: (1) the envelope index equals tile_url on every
square; (2) N_CELLS seeded random cells of the written GeoTIFF equal the source cell read by the frozen
`inventory_b_faint.load_tile` on the cached bytes, bit for bit (NaN where the source is NaN): max |delta| = 0;
(3) N_REFETCH seeded tiles re-downloaded through inventory_b_faint.get match the cache by sha256. Descriptive only: the
cells shared with the FRESH-1 windows (which may hold an older delivery on two-delivery squares).

  python -B ls_mosaic.py selftest
  python -B ls_mosaic.py parity                      # ADDENDUM_E_PARITY.json (aggregates) + private receipts
  python -B ls_mosaic.py build CORES.json REGION_DIR [--workers N] [--prune]
      # CORES.json {"cores": [{"i", "j", ...}]} -> REGION_DIR/mosaics/LS-<i>_<j>.tif (+ .json), REGION_DIR/PREP.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IBF_PATH = ROOT / "work/context_rescue0_2026-10-23/inventory_b_faint.py"
IBF_SHA = "296960596a638a762682cbbb680e0137e590cf625d665e9b7ebc8cf78883f5e9"   # CONTEXT-RESCUE-1 FREEZE_PINS.json
sys.path.insert(0, str(IBF_PATH.parent))
import inventory_b_faint as IBF  # noqa: E402

CORE_M, HALO_M, TILE_M = 5000, 256, 1000
N = CORE_M + 2 * HALO_M
PRIV = ROOT / "outputs/subtle1_2026-10-23_PRIVATE"
CACHE = Path(os.environ.get("LS_TILE_CACHE", PRIV / "dgm1_tiles"))
DEV_FILE = ROOT / "outputs/context_rescue1_2026-10-23_PRIVATE/DEV_CORES_PRIVATE.json"
DEV_SHA = "07e58c31b8ae6cfa8846883c7dc4a0cafed9582ac0845cd3953ceaa79bcd5ce6"
WINDOWS = ROOT / "work/release_2026-10/repro/runnable/outputs/d1_collapse_2026-10-06/vm/dev_windows"
PARITY_PRIV = PRIV / "addendum_e"
PARITY_PUB = HERE / "ADDENDUM_E_PARITY.json"
SEED, N_CELLS, N_REFETCH = 20261023, 2000, 4
GAP_S, WORKERS = 1.0, 6
ENVELOPE = IBF.INDEX.replace("geometry={e},{n}&geometryType=esriGeometryPoint",
                             "geometry={x0},{y0},{x1},{y1}&geometryType=esriGeometryEnvelope")
REGION = "LS"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    return sha_bytes(Path(p).read_bytes())


CODE_SHA = sha_file(Path(__file__))


def check_pins() -> None:
    if sha_file(IBF_PATH) != IBF_SHA:
        raise SystemExit("inventory_b_faint.py differs from its CONTEXT-RESCUE-1 pin")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------- geometry
def extent(i: int, j: int) -> tuple[int, int, int, int]:
    return CORE_M * i - HALO_M, CORE_M * j - HALO_M, CORE_M * (i + 1) + HALO_M, CORE_M * (j + 1) + HALO_M


def squares(i: int, j: int) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = extent(i, j)
    return [(a, b) for b in range(y0 // TILE_M, (y1 - 1) // TILE_M + 1)
            for a in range(x0 // TILE_M, (x1 - 1) // TILE_M + 1)]


def name_square(url: str) -> tuple[int, int]:
    e, n = (int(v) for v in url.rsplit("/", 1)[-1].split("_")[2:4])
    return e, n


# ---------------------------------------------------------------- paced public reads
class Pacer:
    """Request starts >= GAP_S apart across all threads; HTTP 429 stops the whole run."""

    def __init__(self, gap_s: float = GAP_S) -> None:
        self.gap, self.lock, self.last, self.n = gap_s, threading.Lock(), 0.0, 0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.last + self.gap:
                time.sleep(self.last + self.gap - now)
            self.last = time.monotonic()
            self.n += 1

    def get(self, url: str) -> bytes:
        last: Exception | None = None
        for attempt in range(3):
            self.wait()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=IBF.UA), timeout=120) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise SystemExit("HTTP 429: stopping (FREEZE.md limits)")
                last = e
            except urllib.error.URLError as e:
                last = e
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"GET failed: {url}: {last}")


PACER = Pacer()


# ---------------------------------------------------------------- index
def pick(feats: list[dict]) -> tuple[str | None, bool]:
    """tile_url's choice among one square's features (max Aktualitaet); also whether the maximum is tied."""
    if not feats:
        return None, False
    keyf = [f["attributes"].get("Aktualitaet") or 0 for f in feats]
    best = max(range(len(feats)), key=lambda k: keyf[k])
    return feats[best]["attributes"]["dgm1"], keyf.count(keyf[best]) > 1


def envelope_index(i: int, j: int, index: dict, pacer: Pacer = PACER) -> dict:
    """Fill `index` (tile_url's cache format, key 'e_n' -> url or None) for every square of core (i, j)."""
    todo = [sq for sq in squares(i, j) if f"{sq[0]}_{sq[1]}" not in index]
    if not todo:
        return {"queried": False, "ties": []}
    x0, y0, x1, y1 = extent(i, j)
    reply = json.loads(pacer.get(ENVELOPE.format(x0=x0, y0=y0, x1=x1, y1=y1)))
    if reply.get("exceededTransferLimit"):
        raise SystemExit(f"index envelope for core {i}_{j} exceeded the transfer limit")
    by_sq: dict[tuple[int, int], list[dict]] = {}
    for f in reply.get("features", []):
        url = f["attributes"]["dgm1"]
        if url and not url.startswith(IBF.HOST):
            raise SystemExit(f"tile url off the LGLN host: {url}")
        if url:
            by_sq.setdefault(name_square(url), []).append(f)
    ties = []
    for sq in todo:
        url, tied = pick(by_sq.get(sq, []))
        if tied:
            ties.append(f"{sq[0]}_{sq[1]}")
            pacer.wait()
            url = IBF.tile_url(sq, index)
        index[f"{sq[0]}_{sq[1]}"] = url
    return {"queried": True, "ties": ties}


# ---------------------------------------------------------------- tiles
def cache_paths(url: str) -> tuple[Path, Path]:
    name = url.rsplit("/", 1)[-1]
    return CACHE / name, CACHE / (name + ".sha256")


def cached(url: str) -> bytes | None:
    p, s = cache_paths(url)
    if p.exists() and s.exists():
        data = p.read_bytes()
        if sha_bytes(data) == s.read_text(encoding="utf-8").strip():
            return data
    return None


def fetch(url: str, pacer: Pacer = PACER) -> bytes:
    """Tile bytes, from the cache when the cached file matches its sha256 sidecar, else from the LGLN host."""
    data = cached(url)
    if data is not None:
        return data
    data = pacer.get(url)
    p, s = cache_paths(url)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{threading.get_ident()}.part")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    s.write_text(sha_bytes(data) + "\n", encoding="utf-8")
    return data


def prefetch(urls: list[str], workers: int) -> None:
    todo = [u for u in dict.fromkeys(urls) if cached(u) is None]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(fetch, todo))


def parse(url: str, data: bytes) -> np.ndarray:
    """inventory_b_faint.load_tile's rules, applied to given bytes."""
    from rasterio.io import MemoryFile
    e, n = name_square(url)
    with MemoryFile(data) as m, m.open() as ds:
        a = ds.read(1, masked=True).filled(np.nan).astype("float32")
        assert ds.res == (1.0, 1.0) and a.shape == (1000, 1000), (ds.res, a.shape)
        assert (ds.bounds.left, ds.bounds.top) == (e * 1000, (n + 1) * 1000), (url, ds.bounds)
    a[(a < -1000) | (a > 1e30)] = np.nan
    return a


def frozen_tile(url: str, data: bytes) -> np.ndarray:
    """inventory_b_faint.load_tile itself, fed the given bytes (its get() is swapped for this one call)."""
    real = IBF.get

    def given(u: str) -> bytes:
        if u != url:
            raise RuntimeError(f"unexpected read {u}")
        return data
    IBF.get = given
    try:
        return IBF.load_tile(url)
    finally:
        IBF.get = real


# ---------------------------------------------------------------- mosaic
def assemble(i: int, j: int, index: dict, getter=fetch) -> tuple[np.ndarray, dict]:
    x0, y0, x1, y1 = extent(i, j)
    arr = np.full((N, N), np.nan, dtype="float32")
    tiles: dict[str, dict | None] = {}
    for sq in squares(i, j):
        url = index[f"{sq[0]}_{sq[1]}"]
        if url is None:
            tiles[f"{sq[0]}_{sq[1]}"] = None
            continue
        if name_square(url) != sq:
            raise SystemExit(f"index returned {url} for square {sq}")
        data = getter(url)
        t = parse(url, data)
        e, n = sq
        ix0, ix1 = max(x0, e * TILE_M), min(x1, (e + 1) * TILE_M)
        iy0, iy1 = max(y0, n * TILE_M), min(y1, (n + 1) * TILE_M)
        arr[y1 - iy1:y1 - iy0, ix0 - x0:ix1 - x0] = t[(n + 1) * TILE_M - iy1:(n + 1) * TILE_M - iy0,
                                                      ix0 - e * TILE_M:ix1 - e * TILE_M]
        tiles[f"{e}_{n}"] = {"url": url, "sha256": sha_bytes(data), "bytes": len(data)}
    return arr, tiles


def write_mosaic(arr: np.ndarray, i: int, j: int, out: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    x0, _, _, y1 = extent(i, j)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with rasterio.open(tmp, "w", driver="GTiff", width=N, height=N, count=1, dtype="float32", crs="EPSG:25832",
                       transform=from_origin(x0, y1, 1.0, 1.0), nodata=float("nan"), compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256) as ds:
        ds.write(arr, 1)
    os.replace(tmp, out)


def values_sha(arr: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def build_core(i: int, j: int, out_dir: Path, index: dict, getter=fetch) -> dict:
    arr, tiles = assemble(i, j, index, getter)
    out = out_dir / f"{REGION}-{i}_{j}.tif"
    write_mosaic(arr, i, j, out)
    core = arr[HALO_M:HALO_M + CORE_M, HALO_M:HALO_M + CORE_M]
    rec = {"id": f"{REGION}-{i}_{j}", "i": i, "j": j,
           "core": [CORE_M * i, CORE_M * j, CORE_M * (i + 1), CORE_M * (j + 1)],
           "bounds": list(extent(i, j)), "shape": [N, N], "halo_m": HALO_M, "res_m": 1.0, "path": rel(out),
           "sha256": sha_file(out), "values_sha256": values_sha(arr),
           "valid_km2": round(float(np.isfinite(arr).sum()) / 1e6, 6),
           "valid_core_km2": round(float(np.isfinite(core).sum()) / 1e6, 6),
           "tiles_missing": sorted(k for k, v in tiles.items() if v is None), "tiles": tiles,
           "code_sha256": {"ls_mosaic.py": CODE_SHA, "inventory_b_faint.py": IBF_SHA}, "utc": utc()}
    write_json(out.with_suffix(".json"), rec)
    return rec


def prune(index: dict, keep: set[str]) -> None:
    for url in {u for u in index.values() if u} - keep:
        for p in cache_paths(url):
            p.unlink(missing_ok=True)


def build(cores_file: Path, region_dir: Path, workers: int = WORKERS, prune_tiles: bool = False) -> dict:
    """Mosaics in core order, then PREP.json. Resumable: a core whose receipt matches its file is kept."""
    check_pins()
    cores = json.loads(Path(cores_file).read_text(encoding="utf-8"))["cores"]
    index_file, mos = region_dir / "TILE_INDEX.json", region_dir / "mosaics"
    index = json.loads(index_file.read_text(encoding="utf-8")) if index_file.exists() else {}
    recs, t0, ties = [], time.monotonic(), []
    for k, c in enumerate(cores):
        i, j = c["i"], c["j"]
        rp = mos / f"{REGION}-{i}_{j}.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8"))
            if (ROOT / r["path"]).exists() and sha_file(ROOT / r["path"]) == r["sha256"] \
                    and r["code_sha256"]["ls_mosaic.py"] == CODE_SHA:
                recs.append(r)
                continue
        ties += envelope_index(i, j, index)["ties"]
        write_json(index_file, index)
        prefetch([index[f"{a}_{b}"] for a, b in squares(i, j) if index[f"{a}_{b}"]], workers)
        recs.append(build_core(i, j, mos, index))
        if prune_tiles:
            later = {index.get(f"{a}_{b}") for c2 in cores[k + 1:] for a, b in squares(c2["i"], c2["j"])}
            prune(index, {u for u in later if u})
        print(f"{utc()} core {k + 1}/{len(cores)} {recs[-1]['id']} valid_core {recs[-1]['valid_core_km2']} km2 "
              f"requests {PACER.n} elapsed {time.monotonic() - t0:.0f} s", flush=True)
    prep = {"region": REGION, "code_sha256": CODE_SHA, "ibf_sha256": IBF_SHA, "utc": utc(),
            "cores_file_sha256": sha_file(Path(cores_file)), "index_ties": ties,
            "eval_area_km2": round(sum(r["valid_core_km2"] for r in recs), 6),
            "clusters": [{"id": r["id"], "core": r["core"], "path": r["path"], "sha256": r["sha256"],
                          "values_sha256": r["values_sha256"], "valid_core_km2": r["valid_core_km2"]}
                         for r in recs]}
    write_json(region_dir / "PREP.json", prep)
    return prep


# ---------------------------------------------------------------- parity (Addendum E)
def window_bounds() -> list[tuple[str, tuple[float, float, float, float]]]:
    import rasterio
    out = []
    for p in sorted(WINDOWS.glob("VER-P-*-w0.tif")):
        with rasterio.open(p) as ds:
            out.append((p.name, tuple(ds.bounds)))
    return out


def overlap(a: tuple, b: tuple) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def parity_cores() -> list[tuple[int, int]]:
    if sha_file(DEV_FILE) != DEV_SHA:
        raise SystemExit("DEV_CORES_PRIVATE.json differs from its pin")
    wins = window_bounds()
    cores = json.loads(DEV_FILE.read_text(encoding="utf-8"))["cores"]
    score = {(c["i"], c["j"]): sum(overlap(extent(c["i"], c["j"]), w) for _, w in wins) for c in cores}
    return sorted(sorted(score), key=lambda k: (-score[k], k))[:2]


def cell_check(path: Path, index: dict, rng: random.Random, i: int, j: int) -> dict:
    """N_CELLS random cells of the written GeoTIFF against frozen load_tile on the cached bytes."""
    import rasterio
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    x0, _, _, y1 = extent(i, j)
    tiles, worst, nan_mismatch, n_nan = {}, 0.0, 0, 0
    for _ in range(N_CELLS):
        r, c = rng.randrange(N), rng.randrange(N)
        x, y = x0 + c, y1 - r - 1
        url = index[f"{x // TILE_M}_{y // TILE_M}"]
        if url is None:
            src = np.float32(np.nan)
        else:
            if url not in tiles:
                tiles[url] = frozen_tile(url, cached(url))
            src = tiles[url][TILE_M - 1 - (y % TILE_M), x % TILE_M]
        got = arr[r, c]
        if np.isnan(src) or np.isnan(got):
            nan_mismatch += int(np.isnan(src) != np.isnan(got))
            n_nan += int(np.isnan(src))
        else:
            worst = max(worst, abs(float(got) - float(src)))
            if np.float32(got).tobytes() != np.float32(src).tobytes():
                worst = max(worst, 1e-30)
    return {"cells": N_CELLS, "max_abs_delta": worst, "nan_mismatch": nan_mismatch, "nan_cells": n_nan,
            "PASS": worst == 0.0 and nan_mismatch == 0}


def index_check(i: int, j: int, index: dict) -> dict:
    point: dict = {}
    for sq in squares(i, j):
        PACER.wait()
        IBF.tile_url(sq, point)
    diff = sorted(k for k in point if point[k] != index.get(k))
    return {"squares": len(point), "differ": diff, "PASS": not diff}


def refetch_check(index: dict, rng: random.Random, i: int, j: int) -> dict:
    urls = sorted({index[f"{a}_{b}"] for a, b in squares(i, j) if index[f"{a}_{b}"]})
    rows = []
    for url in rng.sample(urls, min(N_REFETCH, len(urls))):
        PACER.wait()
        rows.append({"tile": url.rsplit("/", 1)[-1], "equal": sha_bytes(IBF.get(url)) == sha_bytes(cached(url))})
    return {"tiles": len(rows), "equal": sum(r["equal"] for r in rows), "rows": rows,
            "PASS": all(r["equal"] for r in rows)}


def window_check(path: Path, i: int, j: int) -> dict:
    """Descriptive: cells shared with the FRESH-1 Verden windows (mosaic vs window values)."""
    import rasterio
    from rasterio.windows import from_bounds
    out = {}
    with rasterio.open(path) as m:
        for name, b in window_bounds():
            ov = (max(b[0], m.bounds.left), max(b[1], m.bounds.bottom), min(b[2], m.bounds.right),
                  min(b[3], m.bounds.top))
            if ov[0] >= ov[2] or ov[1] >= ov[3]:
                continue
            with rasterio.open(WINDOWS / name) as w:
                if w.res != (1.0, 1.0):
                    out[name] = {"skipped": f"window res {w.res}"}
                    continue
                a = m.read(1, window=from_bounds(*ov, transform=m.transform))
                b2 = w.read(1, window=from_bounds(*ov, transform=w.transform), masked=True).filled(np.nan)
            ok = np.isfinite(a) & np.isfinite(b2)
            out[name] = {"cells": int(ok.sum()),
                         "equal_share": round(float((a[ok] == b2[ok]).mean()), 6) if ok.any() else None,
                         "max_abs_delta": float(np.abs(a[ok] - b2[ok]).max()) if ok.any() else None}
    return out


def parity() -> dict:
    check_pins()
    rng = random.Random(SEED)
    index, rows = {}, {}
    for i, j in parity_cores():
        envelope_index(i, j, index)
        prefetch([index[f"{a}_{b}"] for a, b in squares(i, j) if index[f"{a}_{b}"]], WORKERS)
        rec = build_core(i, j, PARITY_PRIV / "mosaics", index)
        path = ROOT / rec["path"]
        rows[rec["id"]] = {"index": index_check(i, j, index), "cells": cell_check(path, index, rng, i, j),
                           "refetch": refetch_check(index, rng, i, j), "windows": window_check(path, i, j),
                           "valid_core_km2": rec["valid_core_km2"], "tiles_missing": len(rec["tiles_missing"]),
                           "values_sha256": rec["values_sha256"]}
    passed = all(r[k]["PASS"] for r in rows.values() for k in ("index", "cells", "refetch"))
    res = {"step": "Addendum E parity (CONTEXT-RESCUE-1 FREEZE.md §2.4)", "utc": utc(),
           "code_sha256": {"ls_mosaic.py": CODE_SHA, "inventory_b_faint.py": IBF_SHA}, "seed": SEED,
           "cores": rows, "rule": "index equal on every square; cells max |delta| = 0 at N_CELLS each; refetch equal",
           "PASS": passed}
    write_json(PARITY_PUB, res)
    return res


# ---------------------------------------------------------------- selftest (no network)
def synthetic_tile(e: int, n: int, fill: float) -> bytes:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    a = (np.arange(1_000_000, dtype="float32").reshape(1000, 1000) * 1e-3 + fill).astype("float32")
    a[0, 0] = -9999.0
    with MemoryFile() as m:
        with m.open(driver="GTiff", width=1000, height=1000, count=1, dtype="float32", crs="EPSG:25832",
                    transform=from_origin(e * 1000, (n + 1) * 1000, 1.0, 1.0), nodata=-9999.0) as ds:
            ds.write(a, 1)
        return m.read()


def selftest() -> dict:
    global CACHE
    check_pins()
    checks = {}
    i, j = 100, 1173
    sqs = squares(i, j)
    checks["squares_7x7"] = len(sqs) == 49 and sqs[0] == (499, 5864) and sqs[-1] == (505, 5870)
    url = lambda e, n, y: f"{IBF.HOST}L/Vollkacheln/dgm1_32_{e}_{n}_1_ni_{y}.tif"  # noqa: E731
    feats = [{"attributes": {"dgm1": url(500, 5868, 2015), "Aktualitaet": 1}},
             {"attributes": {"dgm1": url(500, 5868, 2021), "Aktualitaet": 5}}]
    checks["pick_latest"] = pick(feats) == (url(500, 5868, 2021), False)
    checks["pick_tie"] = pick(feats[:1] * 2)[1] is True and pick([]) == (None, False)
    index = {f"{a}_{b}": (None if (a, b) == (505, 5870) else url(a, b, 2020)) for a, b in sqs}
    blobs = {index[k]: synthetic_tile(*map(int, k.split("_")), fill=float(sum(map(int, k.split("_"))) % 7))
             for k in index if index[k]}
    arr, tiles = assemble(i, j, index, getter=lambda u: blobs[u])
    x0, _, _, y1 = extent(i, j)
    x, y = 500_123, 5_868_456                                  # inside square (500, 5868)
    want = frozen_tile(index["500_5868"], blobs[index["500_5868"]])[TILE_M - 1 - (y % TILE_M), x % TILE_M]
    checks["cell_placement"] = arr[y1 - y - 1, x - x0] == want
    checks["missing_square_nan"] = bool(np.isnan(arr[0:HALO_M, -HALO_M:]).all()) and tiles["505_5870"] is None
    corner = (500 * TILE_M, 5869 * TILE_M - 1)                  # tile (500, 5868) cell [0, 0] = nodata
    checks["nodata_nan"] = bool(np.isnan(arr[y1 - corner[1] - 1, corner[0] - x0]))
    checks["parse_equals_frozen"] = all(np.array_equal(parse(u, b), frozen_tile(u, b), equal_nan=True)
                                        for u, b in list(blobs.items())[:3])
    with tempfile.TemporaryDirectory() as td:
        CACHE = Path(td) / "cache"
        out = Path(td) / "m.tif"
        write_mosaic(arr, i, j, out)
        import rasterio
        with rasterio.open(out) as ds:
            back = ds.read(1)
            checks["write_roundtrip"] = np.array_equal(back, arr, equal_nan=True) and ds.res == (1.0, 1.0) \
                and (ds.bounds.left, ds.bounds.top) == (x0, y1)
        u0 = next(iter(blobs))
        p, s = cache_paths(u0)
        p.parent.mkdir(parents=True)
        p.write_bytes(blobs[u0])
        s.write_text(sha_bytes(blobs[u0]) + "\n", encoding="utf-8")
        checks["cache_hit"] = cached(u0) == blobs[u0]
        p.write_bytes(b"corrupt")
        checks["cache_reject_corrupt"] = cached(u0) is None
    pacer, t = Pacer(0.2), time.monotonic()
    for _ in range(4):
        pacer.wait()
    checks["pacer_gap"] = time.monotonic() - t >= 0.6
    checks["envelope_url"] = "esriGeometryEnvelope" in ENVELOPE and "{x0}" in ENVELOPE
    checks = {k: bool(v) for k, v in checks.items()}
    res = {"selftest": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "code_sha256": CODE_SHA}
    print(json.dumps(res, indent=1))
    return res


def main() -> None:
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        sys.exit(0 if selftest()["selftest"] == "PASS" else 1)
    if a[:1] == ["parity"]:
        r = parity()
        print(json.dumps({k: r[k] for k in ("utc", "PASS")} | {"cores": {c: {k: v.get("PASS") for k, v in d.items()
              if isinstance(v, dict) and "PASS" in v} for c, d in r["cores"].items()}}, indent=1))
        sys.exit(0 if r["PASS"] else 1)
    if a[:1] == ["build"] and len(a) >= 3:
        workers = int(a[a.index("--workers") + 1]) if "--workers" in a else WORKERS
        prep = build(Path(a[1]), Path(a[2]), workers, "--prune" in a)
        print(json.dumps({k: v for k, v in prep.items() if k != "clusters"} | {"n": len(prep["clusters"])}))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
