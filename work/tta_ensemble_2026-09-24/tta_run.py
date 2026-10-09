"""Inference-variance arms for frozen R2 (no training): flip test-time augmentation and a two-checkpoint mean.

Arms (sigmoid channel-1 probability maps, averaged before the frozen 0.5 threshold and vectorising):
  E38   best.pt (epoch 38), identity: must reproduce the recorded R2 objects exactly (reproduction gate)
  E39   last.pt (epoch 39), identity
  T4    E38 mean over identity, horizontal, vertical and both flips
  C2    mean of E38 and E39 (identity)
  C2T4  mean over both checkpoints x 4 flips

The tile loop mirrors adaf_configs.tile_probabilities line for line (1,024 px tiles, 64 px halo, SLRM radius 20,
8 px trim, pixelwise max over overlaps); only the per-tile prediction returns one map per arm. Vectorising,
validity masking, interior masking and SCREEN half-open filtering are the frozen functions.

  python -B tta_run.py smoke N                # holdout: first N windows, timing + E38 reproduction only
  python -B tta_run.py holdout                # 804 holdout-square windows, all arms
  python -B tta_run.py screen ARM [ARM ...]   # 26 SCREEN blocks, requested arms (+ E38 reproduction)
(WSL adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
import adaf_configs as ac  # noqa: E402

OUT = ROOT / "outputs/tta_ensemble_2026-09-24"
TRAINING = ROOT / "outputs/campaign_2026-09-22/training/R2_adaf_hrnet_slrm"
CKPTS = {"e38": TRAINING / "best.pt", "e39": TRAINING / "last.pt"}
CKPT_SHA256 = {"e38": "fbd1cbacdf10c2d2806ec6dc7662196ebeed37416b908bad8f076df3113e7392"}
FLIPS = {"id": (), "h": (-1,), "v": (-2,), "hv": (-2, -1)}
ARMS = {"E38": [("e38", "id")], "E39": [("e39", "id")],
        "T4": [("e38", f) for f in FLIPS], "C2": [("e38", "id"), ("e39", "id")],
        "C2T4": [(c, f) for c in ("e38", "e39") for f in FLIPS]}
RECORDED_HOLDOUT = ROOT / "outputs/campaign_2026-09-22/holdout_diag/R2_holdout/OBJECTS_PRIVATE.json"
RECORDED_SCREEN = ROOT / "outputs/campaign_2026-09-22/screen/R2/proposals"


def load(needed: set[str]) -> dict[str, torch.nn.Module]:
    ac.check_frozen()
    names = sorted(needed)
    models, _ = ac.load_models([CKPTS[n] for n in names])
    return dict(zip(names, models))


def predictor(models: dict[str, torch.nn.Module], arms: list[str]):
    passes = sorted({p for a in arms for p in ARMS[a]})

    def predict(a: np.ndarray) -> dict[str, np.ndarray]:
        tensor = torch.tensor(np.repeat(a[None, :, :], 3, axis=0), dtype=torch.float32, device="cuda")[None]
        maps = {}
        with torch.inference_mode():
            for ck, fl in passes:
                dims = FLIPS[fl]
                x = torch.flip(tensor, dims) if dims else tensor
                y = torch.sigmoid(models[ck](x)["out"])[0, 1]
                maps[(ck, fl)] = torch.flip(y, dims) if dims else y
            out = {}
            for arm in arms:
                parts = ARMS[arm]
                if len(parts) == 1:
                    out[arm] = maps[parts[0]].cpu().numpy()  # identical op sequence to adaf_configs.predict_max
                else:
                    out[arm] = torch.stack([maps[p] for p in parts]).mean(0).cpu().numpy()
        return out
    return predict


def tile_probabilities_multi(src, predict, arms: list[str], phase_m: float = 0.,
                             tile: int = 1024) -> tuple[dict[str, np.ndarray], int]:
    height, width = src.height, src.width
    results = {k: np.zeros((height, width), dtype="float32") for k in arms}
    phase = int(round(phase_m / .5))
    halo, radius = 64, 20
    start = phase - tile if phase else 0
    patches = 0
    for col in range(start, width, tile):
        for row in range(start, height, tile):
            x0, y0, size = ac._tile_origin(col, row, halo, radius, tile)
            data = np.full((size, size), np.nan, dtype="float32")
            xa, ya = max(x0, 0), max(y0, 0)
            xb, yb = min(x0 + size, width), min(y0 + size, height)
            if xa >= xb or ya >= yb:
                continue
            part = src.read(1, window=Window(xa, ya, xb - xa, yb - ya), masked=True).filled(np.nan).astype("float32")
            data[ya - y0:yb - y0, xa - x0:xb - x0] = part
            if not np.isfinite(data[radius:-radius, radius:-radius]).any():
                continue
            slrm = ac.adapter.rvt.vis.slrm(data, radius_cell=radius)
            norm = ac.adapter.normalize_image("slrm", slrm, -.5, .5, "value")[radius:-radius, radius:-radius]
            norm = np.nan_to_num(norm, nan=0.).astype("float32")
            maps = predict(norm)
            x, y = col - halo + 8, row - halo + 8
            for k in arms:
                probs = maps[k][8:-8, 8:-8]
                xa, ya = max(x, 0), max(y, 0)
                xb, yb = min(x + probs.shape[1], width), min(y + probs.shape[0], height)
                if xa < xb and ya < yb:
                    np.maximum(results[k][ya:yb, xa:xb], probs[ya - y:yb - y, xa - x:xb - x],
                               out=results[k][ya:yb, xa:xb])
            patches += 1
    return results, patches


def objects_multi(path: Path, predict, arms: list[str], interior: list[float] | None = None) -> dict:
    """adaf_configs.probability_raster for several arms (no raster written; a hash of each map is kept)."""
    with rasterio.open(path) as original:
        src = ac.adapter.source_view(original)
        results, patches = tile_probabilities_multi(src, predict, arms)
        valid = src.read(1, masked=True)
        valid = np.isfinite(valid.filled(np.nan)) & ~np.ma.getmaskarray(valid)
        if interior:
            valid &= geometry_mask([mapping(box(*interior))], out_shape=valid.shape, transform=src.transform,
                                   invert=True)
        out = {"patches": patches, "valid_km2": float(valid.sum()) * .25 / 1e6, "arms": {}}
        for k in arms:
            res = results[k]
            res[~valid] = 0
            out["arms"][k] = {"objects": ac.vectorise(res, src.transform, path.stem, path.relative_to(ROOT).as_posix()),
                              "map_sha256": hashlib.sha256(res.tobytes()).hexdigest()}
        if src is not original:
            src.close()
    return out


def same_objects(a: list[dict], b: list[dict]) -> bool:
    return len(a) == len(b) and all(abs(float(p["easting"]) - float(q["easting"])) <= .01 and
                                    abs(float(p["northing"]) - float(q["northing"])) <= .01 and
                                    abs(float(p["score"]) - float(q["score"])) <= 1e-4 for p, q in zip(a, b))


def holdout_windows() -> list[dict]:
    m = json.loads((ROOT / "work/campaign_2026-09-22/R2_MASKS_MANIFEST_PRIVATE.json").read_text(encoding="utf-8"))
    return [w for w in m["windows"] if w["holdout"]]


def run_holdout(limit: int | None, arms: list[str]) -> dict:
    if ac.digest(CKPTS["e38"]) != CKPT_SHA256["e38"]:
        raise RuntimeError("best.pt changed")
    recorded = json.loads(RECORDED_HOLDOUT.read_text(encoding="utf-8"))["objects"]
    wins = holdout_windows()[:limit] if limit else holdout_windows()
    models = load({c for a in arms for c, _ in ARMS[a]})
    predict = predictor(models, arms)
    torch.set_num_threads(4)
    t0, objs, mism = time.monotonic(), {a: {} for a in arms}, []
    for w in wins:
        rec = objects_multi(ROOT / "outputs/campaign_2026-09-22/train_raw" / f"{w['id']}.tif", predict, arms)
        for a in arms:
            objs[a][w["id"]] = rec["arms"][a]["objects"]
        if "E38" in arms and not same_objects(rec["arms"]["E38"]["objects"], recorded[w["id"]]):
            mism.append(w["id"])
    wall = time.monotonic() - t0
    receipt = {"windows": len(wins), "arms": arms, "wall_s": wall, "s_per_window": wall / max(len(wins), 1),
               "e38_reproduction_mismatch_windows": mism, "checkpoint_sha256": {k: ac.digest(p) for k, p in CKPTS.items()},
               "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    if limit is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "HOLDOUT_OBJECTS_PRIVATE.json").write_text(json.dumps({"receipt": receipt, "objects": objs}),
                                                          encoding="utf-8")
    return receipt


def run_screen(arms: list[str]) -> dict:
    if "E38" not in arms:
        arms = ["E38"] + arms
    models = load({c for a in arms for c, _ in ARMS[a]})
    predict = predictor(models, arms)
    torch.set_num_threads(4)
    t0, per_block = time.monotonic(), {}
    for block in ac.screen_blocks():
        targets = {a: OUT / "screen" / a / "proposals" / f"{block['id']}.json" for a in arms}
        if all(t.exists() for t in targets.values()):  # resumable; one writer per output directory
            continue
        start = time.monotonic()
        source = ac.source_for(block["id"])
        rec = objects_multi(source, predict, arms, interior=block["interior"])
        recorded = json.loads((RECORDED_SCREEN / f"{block['id']}.json").read_text(encoding="utf-8"))["objects"]
        rows = {a: [dict(p, source_block=block["id"]) for p in rec["arms"][a]["objects"]
                    if ac.half_open(p, block["interior"])] for a in arms}
        if not same_objects(rows["E38"], recorded):
            raise RuntimeError(f"{block['id']}: E38 does not reproduce the recorded R2 objects; stop")
        for a in arms:
            targets[a].parent.mkdir(parents=True, exist_ok=True)
            targets[a].write_text(json.dumps({"status": "INFERENCE_COMPLETE", "config": a, "block_id": block["id"],
                                              "proposal_count": len(rows[a]), "objects": rows[a],
                                              "valid_km2": rec["valid_km2"], "patches": rec["patches"],
                                              "map_sha256": rec["arms"][a]["map_sha256"],
                                              "wall_seconds": time.monotonic() - start}, indent=1), encoding="utf-8")
        per_block[block["id"]] = {a: len(rows[a]) for a in arms}
        print(json.dumps({"block": block["id"], "counts": per_block[block["id"]],
                          "wall_s": round(time.monotonic() - start, 1)}), flush=True)
    return {"arms": arms, "blocks_run": len(per_block), "wall_s": time.monotonic() - t0,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "smoke":
        print(json.dumps(run_holdout(int(sys.argv[2]), list(ARMS))))
    elif cmd == "holdout":
        print(json.dumps(run_holdout(None, list(ARMS))))
    elif cmd == "screen":
        print(json.dumps(run_screen(sys.argv[2:])))
    else:
        raise SystemExit(__doc__)
