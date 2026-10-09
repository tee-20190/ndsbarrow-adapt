"""ADAPT-NDS-1 LGLN DGM1 reader (VM side): Lower Saxony FIT training windows and CONFIRM inference windows.

Derived from FRESH-1 step2/lgln_reader.py (imported read-only for its pure helpers: decode_tile, inside_mask,
fetch_tile, render_window, read_raw, owned_share, ledger_rows); only the role checks and the FIT pins are new:
  fit      plan = FIT_STAGING_PINNED_PRIVATE.json (plan_fit.py pin): every tile is fetched once from its official
           URL; Content-Length must equal the HEAD pin and md5(bytes) the HEAD ETag (head_check.py established that the
           ETag is the content md5); sha256 is logged. Pixel centres outside the FIT union are NoData. Each window must
           lie >= 2 km from Verden and from the CONFIRM district (both polygons carried by the plan) and inside the FIT
           union's envelope.
  confirm  plan = CONFIRM_STAGING_PLAN_PRIVATE.json (confirm_build.py, after the freeze): pins = the frozen run's
           validity-pass ledger (bytes, sha256, inside pixels); pixel centres outside the CONFIRM district are NoData;
           the plan binds the CONFIRM inference manifest (no parts: the VM never holds CONFIRM labels).
Both: grid contract EPSG:25832, float32, 1000 x 1000, 1 m, origin at the tile corner; valid = finite and not NoData;
tiles pasted in name order, first valid value wins (older delivery on two-delivery squares; FRESH-1 STEP0_DECISIONS
item 2); windows written in the English raw-window format (lgln_reader.write_raw). Tile bytes never reach the disk.
Staging streams: windows in spatial order, 64 at a time, each tile fetched once and released after its last window.
Refusals before any request (exit 3): a tile not in the plan or off the LGLN host, a box that differs from the tile
name, a window outside its envelope, a FIT window within 2 km of Verden or CONFIRM, a plan made for another manifest.

  python -B nds_reader.py check-roles fit|confirm PLAN [--manifest main=M] [--only ID ...]
  python -B nds_reader.py stage fit|confirm PLAN MASKS WIN_DIR LOG.jsonl [--manifest main=M] [--workers 8] [--only ..]
  python -B nds_reader.py verify fit|confirm PLAN WIN_DIR LOG.jsonl RECEIPT.json [--manifest main=M] [--only ..]
  python -B nds_reader.py void MANIFEST WIN_DIR OUT.json [--only ID ...]       (CONFIRM: background wholly valid)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nds_common as N  # noqa: E402

sys.path.insert(0, str(N.FRESH_STEP2))
import lgln_reader as LR  # noqa: E402  FRESH-1 reader (frozen): pure helpers only

ROLE_EXIT, VERIFY_EXIT = 3, 5
MARGIN_M = 2000.0                     # windows lie inside the region envelope + this margin (FRESH-1: 1,500 m)
CHUNK = 64                            # windows rendered per chunk; tiles evicted after their last window
KINDS = ("fit", "confirm")
MD5 = re.compile(r"^[0-9a-f]{32}$")
SHA = re.compile(r"^[0-9a-f]{64}$")


class RoleError(RuntimeError):
    """A tile or window lies outside the planned ADAPT-NDS-1 terrain: nothing is fetched."""


def select(plan: dict, only: list[str] | None) -> list[dict]:
    wins = plan["windows"]
    if only:
        missing = sorted(set(only) - {w["id"] for w in wins})
        if missing:
            raise RoleError(f"unknown window ids: {missing[:5]}")
        wins = [w for w in wins if w["id"] in set(only)]
    return wins


def tile_violations(kind: str, name: str, t: dict | None) -> list[str]:
    if t is None:
        return [f"{name}: missing from the plan's tile table"]
    bad = []
    try:
        N.check_url(name, t["url"])
        if N.tile_name_box(name) != t["box"]:
            bad.append(f"{name}: box differs from the tile name")
    except N.RefusedTerrain as e:
        bad.append(str(e))
    if not isinstance(t.get("bytes"), int) or t["bytes"] <= 0:
        bad.append(f"{name}: no byte pin")
    if kind == "fit" and not MD5.match(str(t.get("md5", ""))):
        bad.append(f"{name}: no md5 (ETag) pin")
    if kind == "confirm" and not SHA.match(str(t.get("sha256", ""))):
        bad.append(f"{name}: no sha256 pin")
    if t.get("inside") not in ("all", "mask") or (t["inside"] == "all") != (t.get("inside_px") == N.TILE_PX ** 2):
        bad.append(f"{name}: inside mode inconsistent with its inside pixel count")
    return bad


def manifest_windows(plan: dict, manifests: dict[str, dict]) -> dict[str, tuple[str, list]]:
    """CONFIRM: {window id: (tag, bounds)}; the plan must bind exactly these manifests (canonical digests), each the
    CONFIRM district's EVAL manifest in EPSG:25832 with no parts."""
    if set(plan.get("manifests", {})) != set(manifests):
        raise RoleError(f"the plan binds manifests {sorted(plan.get('manifests', {}))}, given {sorted(manifests)}")
    out = {}
    for tag, m in manifests.items():
        if m.get("crs") != N.CRS or m.get("role") != "CONFIRM" or plan.get("district") not in str(m.get("region")):
            raise RoleError(f"{tag}: not the CONFIRM district's manifest in {N.CRS}")
        if m.get("parts"):
            raise RoleError(f"{tag}: a VM manifest must not carry labels")
        if plan["manifests"][tag] != N.digest(m):
            raise RoleError(f"the staging plan was made for another {tag} manifest")
        for t in m["tiles"]:
            if len(t["windows"]) != 1 or t["windows"][0]["core_index"] != "all":
                raise RoleError(f"{t['id']}: one window owning all rectangles is required (infer_irl contract)")
            out[t["windows"][0]["id"]] = (tag, t["windows"][0]["bounds"])
    return out


def refused_regions(plan: dict) -> list:
    from shapely.geometry import shape
    return [(r["name"], shape(r["geometry"])) for r in plan.get("refused_regions", [])]


def role_check(kind: str, plan: dict, manifests: dict[str, dict] | None = None, only: list[str] | None = None) -> dict:
    if kind not in KINDS or plan.get("kind") != kind:
        raise RoleError(f"plan kind {plan.get('kind')!r} is not {kind!r}")
    from shapely.geometry import box
    env = plan["envelope"]
    lim = [env[0] - MARGIN_M, env[1] - MARGIN_M, env[2] + MARGIN_M, env[3] + MARGIN_M]
    mwins = manifest_windows(plan, manifests or {}) if kind == "confirm" else {}
    refused = refused_regions(plan) if kind == "fit" else []
    if kind == "fit" and {n for n, _ in refused} != {"Verden", plan.get("confirm_district")}:
        raise RoleError("the FIT plan does not carry Verden and the CONFIRM district as refused regions")
    bad, names = [], set()
    for w in select(plan, only):
        b = w["bounds"]
        if kind == "confirm" and mwins.get(w["id"]) != (w.get("manifest"), b):
            bad.append(f"{w['id']}: not a window of its manifest (or other bounds)")
        if not (lim[0] <= b[0] and lim[1] <= b[1] and b[2] <= lim[2] and b[3] <= lim[3]):
            bad.append(f"{w['id']}: outside the region envelope")
        for rname, g in refused:
            if box(*b).distance(g) < N.FIT_HALO_M:
                bad.append(f"{w['id']}: within {N.FIT_HALO_M:.0f} m of {rname}")
        for name in w["tiles"]:
            t = plan["tiles"].get(name)
            bad += tile_violations(kind, name, t)
            if t is not None and not N.boxes_overlap(t["box"], b):
                bad.append(f"{w['id']}: tile {name} does not meet the window")
            names.add(name)
        if w["tiles"] != sorted(w["tiles"]):
            bad.append(f"{w['id']}: tiles not in name order")
    if bad:
        raise RoleError(f"{len(bad)} role violation(s); nothing fetched: " + "; ".join(bad[:6]))
    return {"kind": kind, "windows": len(select(plan, only)), "tiles": len(names), "status": "PASS"}


# ---------------------------------------------------------------- fetch
def download(url: str) -> bytes:
    return LR.download(url)


def fetch_fit_tile(name: str, t: dict, masks) -> tuple[np.ndarray, dict]:
    data = download(t["url"])
    md5, sha = hashlib.md5(data).hexdigest(), N.sha256_bytes(data)
    if len(data) != t["bytes"] or md5 != t["md5"]:
        raise SystemExit(f"{name}: payload differs from its HEAD pins ({len(data)} B, md5 {md5[:12]}); stop")
    a, meta = LR.decode_tile(data, t["box"])
    del data
    inside = LR.inside_mask(masks, name, t)
    a[~inside] = np.nan                                       # outside the FIT union: never used
    return a, {"name": name, "bytes": t["bytes"], "md5": md5, "sha256": sha, "utc": N.utc(),
               "inside_px": int(inside.sum()), "valid_px": int(np.isfinite(a).sum()), **meta}


def fetch_any(kind: str, name: str, t: dict, masks) -> tuple[np.ndarray, dict]:
    if kind == "fit":
        return fetch_fit_tile(name, t, masks)
    a, info = LR.fetch_tile(name, t, masks)                   # bytes + sha256 against the validity-pass ledger
    return a, info


def load_masks(path: Path, plan: dict):
    if N.sha256_file(path) != plan["inside_masks_sha256"]:
        raise SystemExit("region masks differ from the plan's pinned sha256")
    return np.load(path)


def stage_order(wins: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Windows in south-to-north, west-to-east order and each tile's last use (index into that order)."""
    order = sorted(wins, key=lambda w: (w["bounds"][1], w["bounds"][0], w["id"]))
    last = {}
    for i, w in enumerate(order):
        for n in w["tiles"]:
            last[n] = i
    return order, last


def stage(kind: str, plan: dict, manifests: dict, masks_path: Path, win_dir: Path, log: Path, workers: int = 8,
          only: list[str] | None = None, chunk: int = CHUNK) -> dict:
    """Fetch each needed tile exactly once and render the windows in spatial order, CHUNK windows at a time; a tile
    leaves memory after the last window that needs it (FIT: 2,806 tiles would need ~11 GB if held together)."""
    role_check(kind, plan, manifests, only)
    order, last = stage_order(select(plan, only))
    masks = load_masks(masks_path, plan)
    win_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    tiles, boxes, rows, fetched, peak = {}, {n: plan["tiles"][n]["box"] for n in last}, [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool, log.open("a", encoding="utf-8", newline="\n") as f:
        for c0 in range(0, len(order), chunk):
            part = order[c0:c0 + chunk]
            need = sorted({n for w in part for n in w["tiles"]} - set(tiles))
            if set(need) & set(fetched):
                raise SystemExit("stage: a tile would be fetched twice")
            for name, (a, info) in zip(need, pool.map(lambda n: fetch_any(kind, n, plan["tiles"][n], masks), need)):
                tiles[name] = a
                f.write(json.dumps(info) + "\n")
                f.flush()
            fetched += need
            peak = max(peak, len(tiles))
            rows += list(pool.map(lambda w: LR.render_window(w, tiles, boxes, win_dir / f"{w['id']}.tif"), part))
            for n in [n for n in tiles if last[n] < c0 + len(part)]:
                del tiles[n]
    if sorted(fetched) != sorted(last):
        raise SystemExit("stage: fetched tiles differ from the windows' tiles")
    rank = {w["id"]: i for i, w in enumerate(plan["windows"])}
    rows.sort(key=lambda r: rank[r["id"]])
    N.write_json(win_dir / "RENDER_LOG.json", {"utc": N.utc(), "kind": kind, "rows": rows})
    return {"kind": kind, "tiles": len(fetched), "bytes": int(sum(plan["tiles"][n]["bytes"] for n in fetched)),
            "windows": len(rows), "peak_tiles_in_memory": peak, "wall_s": round(time.monotonic() - t0, 1),
            "min_valid_fraction": min((r["valid_fraction"] for r in rows), default=None)}


def verify(kind: str, plan: dict, manifests: dict, win_dir: Path, log: Path, only: list[str] | None) -> dict:
    """Receipt; exit 5 unless every selected window exists with its rendered sha256, every needed tile was logged
    with its pins, and no tile outside the plan was logged."""
    role_check(kind, plan, manifests, only)
    wins, bad = select(plan, only), []
    rendered = {r["id"]: r for r in N.read_json(win_dir / "RENDER_LOG.json")["rows"]} \
        if (win_dir / "RENDER_LOG.json").exists() else {}
    for w in wins:
        p = win_dir / f"{w['id']}.tif"
        if not p.exists() or w["id"] not in rendered or N.sha256_file(p) != rendered[w["id"]]["sha256"]:
            bad.append(f"{w['id']}: raw window missing or changed since the render")
    logged = LR.ledger_rows(log) if log.exists() else {}
    for name in {n for w in wins for n in w["tiles"]}:
        t, r = plan["tiles"][name], logged.get(name)
        ok = r is not None and r["bytes"] == t["bytes"] and (
            r.get("md5") == t.get("md5") if kind == "fit" else r.get("sha256") == t.get("sha256"))
        if not ok:
            bad.append(f"{name}: not logged with its pins")
    extra = sorted(set(logged) - set(plan["tiles"]))
    if extra:
        bad.append(f"{len(extra)} logged tile(s) outside the plan")
    return {"utc": N.utc(), "kind": kind, "windows": len(wins), "tiles_logged": len(logged),
            "attribution": N.ATTRIBUTION["terrain"],
            "status": "FAIL" if bad else "PASS", "violations": bad[:20]}


def void(manifest: dict, win_dir: Path, only: list[str] | None) -> dict:
    """lgln_reader.void for the CONFIRM manifest: descriptive void shares; every background rectangle was accepted
    only where 1 m validity was complete after a 20 m erosion, so it must be wholly finite in the rendered window."""
    if manifest.get("crs") != N.CRS or manifest.get("role") != "CONFIRM":
        raise N.RefusedTerrain("void: not a CONFIRM manifest")
    rows, bad = [], []
    for t in manifest["tiles"]:
        w = t["windows"][0]
        p = win_dir / f"{w['id']}.tif"
        if (only and w["id"] not in only) or not p.exists():
            continue
        n, fin = LR.owned_share(LR.read_raw(p), w["bounds"], t["core"])
        rows.append({"window": w["id"], "tile": t["id"], "role": t["role"], "void_share": 1 - fin / n if n else None})
        if t["role"] == "background" and fin != n:
            bad.append(t["id"])
    out = {"rule": "void = non-finite share of the 1 m raw window over pixel centres in the tile's scored boxes",
           "windows": rows, "background_rectangles_not_wholly_valid": bad}
    for role in ("background", "positive"):
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
        N.check_path(path)
        out[tag] = N.read_json(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check-roles", "stage", "verify", "void"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    x = a.args
    try:
        if a.cmd == "void":
            out = void(N.read_json(x[0]), Path(x[1]), a.only)
            N.write_json(Path(x[2]), out)
            print(json.dumps({k: out[k] for k in ("background", "positive")}))
            if out["background_rectangles_not_wholly_valid"]:
                raise SystemExit(VERIFY_EXIT)
            return
        kind, plan, manifests = x[0], N.read_json(x[1]), parse_manifests(a.manifest)
        if a.cmd == "check-roles":
            out = role_check(kind, plan, manifests, a.only)
        elif a.cmd == "stage":
            out = stage(kind, plan, manifests, Path(x[2]), Path(x[3]), Path(x[4]), a.workers, a.only)
        else:
            out = verify(kind, plan, manifests, Path(x[2]), Path(x[3]), a.only)
            N.write_json(Path(x[4]), out)
            if out["status"] != "PASS":
                print(json.dumps(out))
                raise SystemExit(VERIFY_EXIT)
    except (RoleError, N.RefusedTerrain) as e:
        print(f"ROLE CHECK FAILED: {e}", file=sys.stderr)
        raise SystemExit(ROLE_EXIT)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
