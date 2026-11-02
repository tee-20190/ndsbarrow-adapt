"""DENSE-TERRAIN-1 evaluation-core decoder (FREEZE.md Section 7), VM side.

One pass per (core, phase) over every listed model. The tile loop is tta_run.tile_probabilities_multi's, unchanged in
arithmetic (1,024 px tiles, 64 px halo, SLRM radius 20, normalise +-0.5, NaN -> 0, centre crop [8:-8], pixelwise max,
tiling phase); a 3-channel model gets torch.tensor(np.repeat(norm[None], 3, 0)) exactly as tta_run.predictor, a
4-channel model gets [norm, asinh(r2/.1), asinh(r32/1), v] where the three added channels are rendered once per core
from the 1 m mosaic (dt1_channels) and sliced at the tile's 1,152 px view (0 outside the mosaic). Each map is then
treated as tta_run.objects_multi does (validity zeroing, adaf_configs.vectorise, map sha256) and an object is kept iff its
centroid lies in its core (ot_common.in_core). Per core the decoder also writes 5 m cell counts over the core (FREEZE
Section 9): pixels with finite |SLRM| >= 0.5 (legacy clipping) and pixels with v = 0.

  python -B dt1_decode.py run MODELS.json PREP.json OUT_DIR --phase 0|31 [--cores LIST.json] [--workers W] [--shard k/n]
  python -B dt1_decode.py parity MODELS.json PREP.json OUT.json --model NAME --cores ID [ID ...]
        (the frozen tta_run.objects_multi and this loop on the same 3-channel model and cores: map sha256 and objects)
  python -B dt1_decode.py assemble MODELS.json PREP.json OUT_DIR --phase P [--cores LIST.json]
MODELS.json: {"models": {name: {"path": repo-relative, "sha256": ..., "channels": 3|4}}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("work/tta_ensemble_2026-09-24", "work/campaign_2026-09-22", "work/adaf_head_to_head_2026-09-19",
          "work/subtle1_2026-10-23/pipeline"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(HERE))
import dt1_channels as CH  # noqa: E402

CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HALO, RADIUS, TILE, CELL_PX = 64, 20, 1024, 10          # CELL_PX: 5 m cells on the 0.5 m view


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rd(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_atomic(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    tmp.replace(p)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def decoder_sha() -> dict:
    return {r: sha(ROOT / r) for r in ("work/tta_ensemble_2026-09-24/tta_run.py", "work/campaign_2026-09-22/adaf_configs.py",
                                       "work/adaf_head_to_head_2026-09-19/adapter.py",
                                       "work/dense_terrain1_2026-10-25/dt1_channels.py",
                                       "work/dense_terrain1_2026-10-25/dt1_model.py")}


def load_models(specs: dict, device: str) -> dict:
    import torch
    import dt1_model as DM
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = {}
    for name, s in specs.items():
        p = ROOT / s["path"]
        if sha(p) != s["sha256"]:
            raise SystemExit(f"{name}: weights differ from their pin")
        if s["channels"] == 3:
            import adaf_configs as ac
            (m,), _ = ac.load_models([p], device=device)
        else:
            m = DM.load_checkpoint(p, 4, device)
        out[name] = (m, s["channels"])
    return out


def core_channels(src_1m, view_shape: tuple[int, int]) -> np.ndarray | None:
    """(3, H, W) added channels on the 0.5 m view of the whole mosaic."""
    z = src_1m.read(1, masked=True).filled(np.nan).astype("float32")
    ch = CH.view_channels(CH.native_products(z))
    if ch.shape[1:] != view_shape:
        raise SystemExit(f"rendered channels {ch.shape[1:]} != view {view_shape}")
    return ch


def tile_maps(src, models: dict, extra: np.ndarray | None, phase_m: float, device: str):
    """tta_run.tile_probabilities_multi for 3- and 4-channel models, plus the clip mask (each pixel once)."""
    import torch
    from rasterio.windows import Window
    import adaf_configs as ac
    height, width = src.height, src.width
    results = {k: np.zeros((height, width), dtype="float32") for k in models}
    clip = np.zeros((height, width), dtype=bool)
    phase = int(round(phase_m / .5))
    start = phase - TILE if phase else 0
    patches = 0
    for col in range(start, width, TILE):
        for row in range(start, height, TILE):
            x0, y0, size = ac._tile_origin(col, row, HALO, RADIUS, TILE)
            data = np.full((size, size), np.nan, dtype="float32")
            xa, ya = max(x0, 0), max(y0, 0)
            xb, yb = min(x0 + size, width), min(y0 + size, height)
            if xa >= xb or ya >= yb:
                continue
            part = src.read(1, window=Window(xa, ya, xb - xa, yb - ya), masked=True).filled(np.nan).astype("float32")
            data[ya - y0:yb - y0, xa - x0:xb - x0] = part
            if not np.isfinite(data[RADIUS:-RADIUS, RADIUS:-RADIUS]).any():
                continue
            slrm = ac.adapter.rvt.vis.slrm(data, radius_cell=RADIUS)
            norm = ac.adapter.normalize_image("slrm", slrm, -.5, .5, "value")[RADIUS:-RADIUS, RADIUS:-RADIUS]
            norm = np.nan_to_num(norm, nan=0.).astype("float32")
            core = slrm[RADIUS + HALO:RADIUS + HALO + TILE, RADIUS + HALO:RADIUS + HALO + TILE]
            cy, cx = row, col
            ya2, xa2 = max(cy, 0), max(cx, 0)
            yb2, xb2 = min(cy + TILE, height), min(cx + TILE, width)
            if ya2 < yb2 and xa2 < xb2:
                with np.errstate(invalid="ignore"):
                    c = np.isfinite(core) & (np.abs(core) >= 0.5)
                clip[ya2:yb2, xa2:xb2] = c[ya2 - cy:yb2 - cy, xa2 - cx:xb2 - cx]
            n = norm.shape[0]
            maps = {}
            with torch.inference_mode():
                x3 = None
                for name, (m, ch) in models.items():
                    if ch == 3:
                        if x3 is None:
                            x3 = torch.tensor(np.repeat(norm[None, :, :], 3, axis=0), dtype=torch.float32,
                                              device=device)[None]
                        maps[name] = torch.sigmoid(m(x3)["out"])[0, 1].cpu().numpy()
                    else:
                        e = np.zeros((3, n, n), dtype="float32")
                        vy0, vx0 = y0 + RADIUS, x0 + RADIUS
                        sya, sxa = max(vy0, 0), max(vx0, 0)
                        syb, sxb = min(vy0 + n, height), min(vx0 + n, width)
                        if sya < syb and sxa < sxb:
                            e[:, sya - vy0:syb - vy0, sxa - vx0:sxb - vx0] = extra[:, sya:syb, sxa:sxb]
                        x4 = torch.from_numpy(np.concatenate([norm[None], e])).to(device)[None]
                        maps[name] = torch.sigmoid(m(x4)["out"])[0, 1].cpu().numpy()
            x, y = col - HALO + 8, row - HALO + 8
            for k in models:
                probs = maps[k][8:-8, 8:-8]
                xa, ya = max(x, 0), max(y, 0)
                xb, yb = min(x + probs.shape[1], width), min(y + probs.shape[0], height)
                if xa < xb and ya < yb:
                    np.maximum(results[k][ya:yb, xa:xb], probs[ya - y:yb - y, xa - x:xb - x],
                               out=results[k][ya:yb, xa:xb])
            patches += 1
    return results, patches, clip


def cell_counts(mask: np.ndarray, src, core: list[float]) -> np.ndarray:
    """Counts of True pixels per 5 m cell over the core (1,000 x 1,000 uint8 for a 5 km core, row 0 = north)."""
    col0, row0 = (int(round(v)) for v in ~src.transform * (core[0], core[3]))
    n = int(round((core[2] - core[0]) / 5))
    sub = mask[row0:row0 + n * CELL_PX, col0:col0 + n * CELL_PX]
    if sub.shape != (n * CELL_PX, n * CELL_PX):
        raise SystemExit(f"core window {sub.shape} outside the view")
    return sub.reshape(n, CELL_PX, n, CELL_PX).sum(axis=(1, 3)).astype(np.uint8)


def decode_core(c: dict, models: dict, phase_m: float, device: str) -> dict:
    import rasterio
    import adaf_configs as ac
    import ot_common as OC
    path = ROOT / c["path"]
    with rasterio.open(path) as original:
        src = ac.adapter.source_view(original)
        need4 = any(ch == 4 for _, ch in models.values())
        t0 = time.monotonic()
        extra = core_channels(original, (src.height, src.width)) if need4 else None
        t_ch = time.monotonic() - t0
        results, patches, clip = tile_maps(src, models, extra, phase_m, device)
        valid = src.read(1, masked=True)
        valid = np.isfinite(valid.filled(np.nan)) & ~np.ma.getmaskarray(valid)
        out = {"patches": patches, "valid_km2": float(valid.sum()) * .25 / 1e6, "channels_s": round(t_ch, 1),
               "arms": {}}
        for k in models:
            res = results[k]
            res[~valid] = 0
            objs = ac.vectorise(res, src.transform, path.stem, path.relative_to(ROOT).as_posix())
            out["arms"][k] = {"objects": [dict(o, cluster=c["id"]) for o in objs
                                          if OC.in_core(o["easting"], o["northing"], c["core"])],
                              "n_raw": len(objs), "map_sha256": hashlib.sha256(res.tobytes()).hexdigest()}
        if phase_m == 0:
            strata = {"clip": cell_counts(clip, src, c["core"])}
            if extra is not None:
                strata["v0"] = cell_counts(extra[2] == 0, src, c["core"])
            out["strata"] = strata
        if src is not original:
            src.close()
    return out


def part_path(out: Path, model: str, cid: str, phase: float) -> Path:
    return out / f"phase{int(phase)}" / "parts" / model / f"{cid}.json"


def run(specs: dict, prep: dict, out: Path, phase: float, cores: set | None, shard: tuple[int, int], device: str) -> dict:
    clusters = [c for c in prep["clusters"] if cores is None or c["id"] in cores]
    clusters = clusters[shard[0]::shard[1]]
    todo = [c for c in clusters if any(not part_path(out, m, c["id"], phase).exists() for m in specs)]
    if not todo:
        return {"done": len(clusters), "decoded": 0}
    models = load_models(specs, device)
    dec = decoder_sha()
    log = []
    for c in todo:
        if sha(ROOT / c["path"]) != c["sha256"]:
            raise SystemExit(f"{c['id']}: mosaic bytes differ from PREP.json")
        t1 = time.monotonic()
        rec = decode_core(c, models, phase, device)
        wall = round(time.monotonic() - t1, 1)
        if "strata" in rec:
            sp = out / "strata" / f"{c['id']}.npz"
            sp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sp, **rec["strata"])
        for m in specs:
            a = rec["arms"][m]
            write_atomic(part_path(out, m, c["id"], phase), {
                "model": m, "cluster": c["id"], "core": c["core"], "phase_m": phase, "mosaic_sha256": c["sha256"],
                "checkpoint_sha256": specs[m]["sha256"], "channels": specs[m]["channels"], "decoder_sha256": dec,
                "code_sha256": CODE_SHA, "patches": rec["patches"], "valid_km2": rec["valid_km2"],
                "map_sha256": a["map_sha256"], "n_raw": a["n_raw"], "n_core": len(a["objects"]), "wall_s_core": wall,
                "channels_s": rec["channels_s"], "n_models_in_pass": len(specs), "utc": utc(), "objects": a["objects"]})
        log.append({"cluster": c["id"], "wall_s": wall, "channels_s": rec["channels_s"]})
        print(json.dumps(log[-1]), flush=True)
    return {"done": len(clusters), "decoded": len(log), "wall_s": sum(r["wall_s"] for r in log)}


def assemble(specs: dict, prep: dict, out: Path, phase: float, cores: set | None) -> dict:
    clusters = [c for c in prep["clusters"] if cores is None or c["id"] in cores]
    summ = {"utc": utc(), "phase_m": phase, "cores": len(clusters), "code_sha256": CODE_SHA, "models": {}}
    for m, s in specs.items():
        objs, walls, maps = [], [], {}
        for c in clusters:
            part = rd(part_path(out, m, c["id"], phase))
            if part["checkpoint_sha256"] != s["sha256"] or part["mosaic_sha256"] != c["sha256"]:
                raise SystemExit(f"{m} {c['id']}: part made from other inputs")
            objs += part["objects"]
            walls.append(part["wall_s_core"])
            maps[c["id"]] = part["map_sha256"]
        ids = [o["id"] for o in objs]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{m}: duplicate object ids")
        p = out / f"phase{int(phase)}" / f"PROPOSALS_{m}_PRIVATE.json"
        write_atomic(p, {"model": m, "phase_m": phase, "checkpoint_sha256": s["sha256"], "channels": s["channels"],
                         "rule": "decoder objects whose centroid lies in the cluster core (half-open)", "n": len(objs),
                         "prep_sha256": prep.get("_sha256"), "objects": objs})
        summ["models"][m] = {"n": len(objs), "sha256": sha(p), "map_sha256": maps}
    write_atomic(out / f"phase{int(phase)}" / "DECODE_SUMMARY.json", summ)
    return {m: v["n"] for m, v in summ["models"].items()}


def parity(specs: dict, prep: dict, out_p: Path, model: str, core_ids: list[str], device: str) -> dict:
    """Frozen tta_run.objects_multi vs this loop on one 3-channel model: map sha256 and objects must be identical."""
    import tta_run as tr
    import torch
    sp = {model: specs[model]}
    if sp[model]["channels"] != 3:
        raise SystemExit("parity needs a 3-channel model")
    models = load_models(sp, device)
    m = models[model][0]

    def predict(a: np.ndarray) -> dict:
        x = torch.tensor(np.repeat(a[None, :, :], 3, axis=0), dtype=torch.float32, device=device)[None]
        with torch.inference_mode():
            return {model: torch.sigmoid(m(x)["out"])[0, 1].cpu().numpy()}
    rows = []
    for cid in core_ids:
        c = next(c for c in prep["clusters"] if c["id"] == cid)
        fz = tr.objects_multi(ROOT / c["path"], predict, [model])
        mine = decode_core(c, models, 0.0, device)
        fo = [o for o in fz["arms"][model]["objects"]]
        mo_all = mine["arms"][model]
        import ot_common as OC
        fo_core = [o for o in fo if OC.in_core(o["easting"], o["northing"], c["core"])]
        rows.append({"cluster": cid, "frozen_map_sha256": fz["arms"][model]["map_sha256"],
                     "dt1_map_sha256": mo_all["map_sha256"],
                     "map_equal": fz["arms"][model]["map_sha256"] == mo_all["map_sha256"],
                     "objects_equal": [dict(o, cluster=cid) for o in fo_core] == mo_all["objects"],
                     "n_core": len(mo_all["objects"])})
    res = {"check": "DT1 decoder loop vs frozen tta_run.objects_multi", "model": model, "utc": utc(), "rows": rows,
           "PASS": all(r["map_equal"] and r["objects_equal"] for r in rows), "code_sha256": CODE_SHA}
    write_atomic(out_p, res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "parity", "assemble"))
    ap.add_argument("models")
    ap.add_argument("prep")
    ap.add_argument("out")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--cores")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--model")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    specs = rd(Path(a.models))["models"]
    prep = rd(Path(a.prep))
    prep["_sha256"] = sha(Path(a.prep))
    cores = set(rd(Path(a.cores))["cores"]) if a.cores else None
    if a.cmd == "run":
        k, n = (int(v) for v in a.shard.split("/"))
        res = run(specs, prep, Path(a.out), a.phase, cores, (k, n), a.device)
    elif a.cmd == "assemble":
        res = assemble(specs, prep, Path(a.out), a.phase, cores)
    else:
        res = parity(specs, prep, Path(a.out), a.model, a.ids, a.device)
        if not res["PASS"]:
            print(json.dumps(res))
            raise SystemExit(1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
