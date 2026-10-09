"""Amendment 2 C: run public-ADAF configurations (any checkpoint list) through adapter.py's exact pipeline.

The frozen adapter.py / run_staged.py assert the AO checkpoint and must not be edited, and adapter.qualify()
overwrites a hash-checked SCREEN input, so this module re-implements adapter.probability_raster for a list of
checkpoints (pixelwise max of sigmoid(out)[:, 1] before the 0.5 threshold) and repeats the synthetic-tile
qualification per checkpoint with receipts under work/campaign_2026-09-22/ only.

  python adaf_configs.py qualify NAME            # NAME in CHECKPOINTS, or a path to a .tar/.pt state dict
  python adaf_configs.py screen CONFIG NAME...   # run CONFIG = max over NAMEs on the 26 SCREEN blocks
  python adaf_configs.py regress CONFIG          # compare CONFIG proposals with the recorded AO proposals
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
from affine import Affine
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / "work/adaf_head_to_head_2026-09-19"
ENGLISH = ROOT / "work/adaf_followup_2026-09-19/english"
sys.path.insert(0, str(PILOT))
import adapter  # noqa: E402  frozen module; adapter.qualify() is never called

ADAPTER_SHA256 = "85621f71c93a1aead31ec7eb3d4b78e46ff20fb7f3af24473bb72a58e9547d27"
HRNET_SOURCE_SHA256 = "4fd8706232e8e73ea8f193f057e1401eddc6e0bcf2c810198dcc8a5c8fc175ce"
_SUFFIX = "_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar"
CHECKPOINTS = {  # name -> (path, pinned md5 from ADAF_ASSET_SOURCE.json)
    "AO": (PILOT / f"weights/AO{_SUFFIX}", "fba3667a886b7b55d6e745f836951919"),
    "barrow": (HERE / f"adaf_weights/barrow{_SUFFIX}", "8c2cf34634c1bc45874a324a92d2096b"),
    "enclosure": (HERE / f"adaf_weights/enclosure{_SUFFIX}", "53498746632f045a48c06bbc0e6b736b"),
}
QUAL_DIR = HERE / "adaf_qualification"
SOURCE_TEXT = Path(__file__).read_bytes()  # the code actually loaded; records never re-hash the file later
MODULE_SHA256 = hashlib.sha256(SOURCE_TEXT).hexdigest()
SCREEN_OUT = ROOT / "outputs/campaign_2026-09-22/screen"
RECORDED_AO = ENGLISH / "objects/proposals/ADAF"


def digest(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(name: str) -> tuple[str, Path, str | None]:
    """Named pinned checkpoint (md5-checked) or an explicit path (e.g. fine-tuned R2 weights)."""
    if name in CHECKPOINTS:
        path, md5 = CHECKPOINTS[name]
        if digest(path, "md5") != md5:
            raise RuntimeError(f"{name}: md5 mismatch against ADAF_ASSET_SOURCE.json")
        return name, path, md5
    path = Path(name).resolve()
    return path.stem, path, None


def check_frozen() -> None:
    if digest(PILOT / "adapter.py") != ADAPTER_SHA256:
        raise RuntimeError("adapter.py changed")


def load_models(paths: list[Path], device: str = "cuda") -> tuple[list[torch.nn.Module], dict]:
    ns, methods, source_sha = adapter.official_nodes()
    if source_sha != HRNET_SOURCE_SHA256:
        raise RuntimeError("official HRNet source changed")
    models = []
    for p in paths:
        model = ns["HRNetModule"](ns["HRNetSegHead"](2), pretrained=False)
        state = torch.load(p, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"], strict=True)
        del state
        models.append(model.eval().to(device))
    return models, methods


def predict_max(models: list[torch.nn.Module], a: np.ndarray, device: str = "cuda") -> np.ndarray:
    """adapter.predict for one model; pixelwise max of sigmoid(out)[0, 1] for several."""
    tensor = torch.tensor(np.repeat(a[None, :, :], 3, axis=0), dtype=torch.float32, device=device)[None]
    with torch.inference_mode():
        if len(models) == 1:
            return torch.sigmoid(models[0](tensor)["out"])[0, 1].cpu().numpy()
        probs = [torch.sigmoid(m(tensor)["out"])[0, 1] for m in models]
        return torch.stack(probs).amax(0).cpu().numpy()


def qualify(name: str) -> dict:
    """Repeat adapter.qualify()'s synthetic-tile equality check for one checkpoint; receipts stay here."""
    check_frozen()
    label, path, md5 = resolve(name)
    out = QUAL_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    (model,), methods = load_models([path])
    a = np.random.default_rng(20260919).uniform(0, 1, (64, 64)).astype("float32")
    inp = out / "synthetic_input.tif"
    with rasterio.open(inp, "w", driver="GTiff", height=64, width=64, count=1, dtype="float32",
                       crs="EPSG:27700", transform=Affine(0.5, 0, 0, 0, -0.5, 32)) as ds:
        ds.write(a, 1)

    class View:
        def __init__(self) -> None:
            self.model, self.device = model, "cuda"
            self.config = type("Config", (), {"threshold": .5})()

        def __call__(self, x: torch.Tensor) -> dict:
            return self.model(x)

    View.get_predicted = methods["get_predicted"]
    View.predict_masks_tiff_probs_binary = methods["predict_masks_tiff_probs_binary"]
    with torch.inference_mode():
        View().predict_masks_tiff_probs_binary(
            image_path=str(inp), label=label,
            data_transforms=lambda x: torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32),
            predictions_dir=str(out))
    with rasterio.open(out / f"synthetic_input_{label}_segmentation_mask_probs.tif") as ds:
        official = ds.read(1)
    candidate = predict_max([model], a)
    err = float(np.max(np.abs(official - candidate)))
    receipt = {"status": "PASS" if err <= 1e-4 else "FAIL", "checkpoint": label,
               "checkpoint_path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
               "checkpoint_md5_pinned": md5, "checkpoint_sha256": digest(path), "strict_state_dict": True,
               "official_method_adapter_max_abs_error": err, "tolerance": 1e-4,
               "hrnet_source_sha256": HRNET_SOURCE_SHA256, "adapter_sha256": ADAPTER_SHA256,
               "adaf_configs_sha256": MODULE_SHA256, "torch": torch.__version__,
               "cuda": torch.cuda.get_device_name()}
    (out / "qualification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def _tile_origin(col: int, row: int, halo: int, radius: int, tile: int = 1024) -> tuple[int, int, int]:
    return col - halo - radius, row - halo - radius, tile + 2 * halo + 2 * radius


def tile_probabilities(src, models: list[torch.nn.Module], phase_m: float,
                       tile: int = 1024) -> tuple[np.ndarray, int]:
    """adapter.probability_raster's tiling/SLRM/normalisation loop with predict -> predict_max.
    tile=1024 is the released ADAF/adapter tiling; tile=512 is the paper's stated size (diagnostic only)."""
    height, width = src.height, src.width
    result = np.zeros((height, width), dtype="float32")
    phase = int(round(phase_m / .5))
    halo, radius = 64, 20
    start = phase - tile if phase else 0
    patches = 0
    for col in range(start, width, tile):
        for row in range(start, height, tile):
            x0, y0, size = _tile_origin(col, row, halo, radius, tile)
            data = np.full((size, size), np.nan, dtype="float32")
            xa, ya = max(x0, 0), max(y0, 0)
            xb, yb = min(x0 + size, width), min(y0 + size, height)
            if xa >= xb or ya >= yb:
                continue
            part = src.read(1, window=Window(xa, ya, xb - xa, yb - ya), masked=True).filled(np.nan).astype("float32")
            data[ya - y0:yb - y0, xa - x0:xb - x0] = part
            if not np.isfinite(data[radius:-radius, radius:-radius]).any():
                continue
            slrm = adapter.rvt.vis.slrm(data, radius_cell=radius)
            norm = adapter.normalize_image("slrm", slrm, -.5, .5, "value")[radius:-radius, radius:-radius]
            norm = np.nan_to_num(norm, nan=0.).astype("float32")
            probs = predict_max(models, norm)[8:-8, 8:-8]
            x, y = col - halo + 8, row - halo + 8
            xa, ya = max(x, 0), max(y, 0)
            xb, yb = min(x + probs.shape[1], width), min(y + probs.shape[0], height)
            if xa < xb and ya < yb:
                np.maximum(result[ya:yb, xa:xb], probs[ya - y:yb - y, xa - x:xb - x], out=result[ya:yb, xa:xb])
            patches += 1
    return result, patches


def vectorise(result: np.ndarray, transform, stem: str, source_rel: str) -> list[dict]:
    """adapter's 0.5 threshold, 4-connected polygons, area > 30 m2 and roundness > 0.4, plus mean probability."""
    mask = (result >= .5).astype("uint8")
    objects = []
    for g, _ in shapes(mask, mask=mask.astype(bool), transform=transform):
        poly = shape(g)
        area = poly.area
        roundness = round(4 * np.pi * area / (poly.convex_hull.length ** 2), 3) if poly.length else 0
        if area <= 30 or roundness <= .4:
            continue
        win = rasterio.windows.from_bounds(*poly.bounds, transform).round_offsets().round_lengths()
        rr, cc, hh, ww = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
        covered = geometry_mask([mapping(poly)], out_shape=(hh, ww),
                                transform=rasterio.windows.transform(win, transform), invert=True)
        values = result[rr:rr + hh, cc:cc + ww][covered]
        centroid = poly.centroid
        objects.append({"id": f"{stem}-{len(objects):05d}", "easting": centroid.x, "northing": centroid.y,
                        "score": float(values.max()), "mean_probability": float(values.mean()),
                        "area_m2": area, "roundness": roundness, "source": source_rel})
    return objects


def probability_raster(path: Path, out: Path, models: list[torch.nn.Module], phase_m: float = 0.,
                       interior: list[float] | None = None, tile: int = 1024) -> dict:
    started = time.monotonic()
    with rasterio.open(path) as original:
        src = adapter.source_view(original)
        result, patches = tile_probabilities(src, models, phase_m, tile)
        valid = src.read(1, masked=True)
        valid = np.isfinite(valid.filled(np.nan)) & ~np.ma.getmaskarray(valid)
        result[~valid] = 0
        if interior:
            valid &= geometry_mask([mapping(box(*interior))], out_shape=result.shape,
                                   transform=src.transform, invert=True)
            result[~valid] = 0
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="float32", nodata=None, compress="deflate",
                       predictor=2, tiled=True, blockxsize=256, blockysize=256)
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(result, 1)
        objects = vectorise(result, src.transform, path.stem, path.relative_to(ROOT).as_posix())
        receipt = {"source": path.relative_to(ROOT).as_posix(), "source_sha256": digest(path), "phase_m": phase_m,
                   "valid_km2": float(valid.sum()) * .25 / 1e6, "patches": patches, "objects": objects,
                   "wall_s": time.monotonic() - started, "output_sha256": digest(out)}
        if src is not original:
            src.close()
    return receipt


def screen_blocks() -> list[dict]:
    contract = json.loads((ENGLISH / "CONTRACT.json").read_text(encoding="utf-8"))
    blocks = contract["stage1"]["blocks"]
    if len(blocks) != 26:
        raise RuntimeError("expected the 26 frozen SCREEN blocks")
    return blocks


def source_for(block_id: str) -> Path:
    rec = json.loads((ENGLISH / "receipts/source_records" / f"{block_id}.json").read_text(encoding="utf-8"))
    src = ROOT / rec["source_relative_path"]
    if digest(src) != rec["source_sha256"]:
        raise RuntimeError(f"{block_id}: SCREEN source hash changed")
    return src


def half_open(p: dict, interior: list[float]) -> bool:
    return interior[0] <= float(p["easting"]) < interior[2] and interior[1] <= float(p["northing"]) < interior[3]


def run_screen(config: str, names: list[str], tile: int = 1024) -> dict:
    check_frozen()
    resolved = [resolve(n) for n in names]
    out_dir = SCREEN_OUT / config
    (out_dir / "proposals").mkdir(parents=True, exist_ok=True)
    (out_dir / f"adaf_configs_{MODULE_SHA256[:12]}.py.txt").write_bytes(SOURCE_TEXT)
    torch.set_num_threads(4)
    t0 = time.monotonic()
    models, _ = load_models([p for _, p, _ in resolved])
    load_s = time.monotonic() - t0
    torch.cuda.reset_peak_memory_stats()
    per_block = {}
    for block in screen_blocks():
        target = out_dir / "proposals" / f"{block['id']}.json"
        if target.exists():  # resumable; one writer per output directory
            per_block[block["id"]] = json.loads(target.read_text(encoding="utf-8"))["proposal_count"]
            continue
        source = source_for(block["id"])
        temp = out_dir / "temp" / f"{block['id']}.tif"
        start = time.monotonic()
        res = probability_raster(source, temp, models, phase_m=0.0, interior=block["interior"], tile=tile)
        rows = [dict(p, source_block=block["id"]) for p in res["objects"] if half_open(p, block["interior"])]
        rec = {"status": "INFERENCE_COMPLETE", "config": config, "checkpoints": [
                   {"name": n, "path": str(p.relative_to(ROOT).as_posix()) if p.is_relative_to(ROOT) else str(p),
                    "md5_pinned": m, "sha256": digest(p)} for n, p, m in resolved],
               "block_id": block["id"], "source_sha256": res["source_sha256"], "proposal_count": len(rows),
               "objects": rows, "phase_m": 0.0, "tile_px": tile, "valid_km2": res["valid_km2"],
               "patches": res["patches"],
               "wall_seconds": time.monotonic() - start, "probability_output_sha256": res["output_sha256"],
               "adapter_sha256": ADAPTER_SHA256, "adaf_configs_sha256": MODULE_SHA256}
        target.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        temp.unlink()  # float raster removed after its hash and objects are retained (as run_staged.py)
        per_block[block["id"]] = len(rows)
        print(json.dumps({"block": block["id"], "proposals": len(rows), "wall_s": round(rec["wall_seconds"], 1)}),
              flush=True)
    summary = {"config": config, "checkpoints": names, "tile_px": tile, "blocks": len(per_block),
               "proposals_raw": sum(per_block.values()), "per_block": per_block, "model_load_s": load_s,
               "wall_s_total": time.monotonic() - t0,
               "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    (out_dir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_holdout(config: str, names: list[str]) -> dict:
    """Amendment 2 F diagnostic inputs: the same pipeline on every holdout-square TRAIN window (no interior)."""
    check_frozen()
    manifest = json.loads((HERE / "R2_MASKS_MANIFEST_PRIVATE.json").read_text(encoding="utf-8"))
    windows = [w for w in manifest["windows"] if w["holdout"]]
    resolved = [resolve(n) for n in names]
    models, _ = load_models([p for _, p, _ in resolved])
    out_dir = ROOT / "outputs/campaign_2026-09-22/holdout_diag" / config
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, objects = time.monotonic(), {}
    for w in windows:
        raw = ROOT / "outputs/campaign_2026-09-22/train_raw" / f"{w['id']}.tif"
        res = probability_raster(raw, out_dir / "temp.tif", models, phase_m=0.0)
        objects[w["id"]] = res["objects"]
    (out_dir / "temp.tif").unlink()
    rec = {"config": config, "checkpoints": [{"name": n, "sha256": digest(p)} for n, p, _ in resolved],
           "windows": len(windows), "objects": objects, "wall_s": time.monotonic() - t0,
           "adaf_configs_sha256": MODULE_SHA256}
    (out_dir / "OBJECTS_PRIVATE.json").write_text(json.dumps(rec), encoding="utf-8")
    return {k: v for k, v in rec.items() if k != "objects"}


def regress(config: str) -> dict:
    """Recorded AO proposals vs CONFIG: per-block count, centroid within 0.01 m, score within 1e-4."""
    report, worst_xy, worst_score, failures = {}, 0.0, 0.0, []
    for block in screen_blocks():
        rec = json.loads((RECORDED_AO / f"{block['id']}.json").read_text(encoding="utf-8"))
        new = json.loads((SCREEN_OUT / config / "proposals" / f"{block['id']}.json").read_text(encoding="utf-8"))
        a, b = rec["objects"], new["objects"]
        ok = len(a) == len(b)
        if ok:
            for p, q in zip(a, b):
                dxy = float(np.hypot(float(p["easting"]) - q["easting"], float(p["northing"]) - q["northing"]))
                ds = abs(float(p["score"]) - q["score"])
                worst_xy, worst_score = max(worst_xy, dxy), max(worst_score, ds)
                ok &= dxy <= 0.01 and ds <= 1e-4
        if not ok:
            failures.append(block["id"])
        report[block["id"]] = {"recorded": len(a), "new": len(b), "pass": ok}
    result = {"status": "PASS" if not failures else "FAIL", "config": config, "failures": failures,
              "max_centroid_m": worst_xy, "max_score_abs": worst_score,
              "recorded_total": sum(v["recorded"] for v in report.values()),
              "new_total": sum(v["new"] for v in report.values()), "blocks": report,
              "tolerances": {"centroid_m": 0.01, "score": 1e-4}}
    (HERE / f"ADAF_REGRESSION_{config}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "qualify":
        print(json.dumps(qualify(sys.argv[2])))
    elif cmd == "screen":
        print(json.dumps(run_screen(sys.argv[2], sys.argv[3:])))
    elif cmd == "screen512":  # diagnostic: the ADAF paper's stated 512 px tiles (+ same 32 m buffer)
        print(json.dumps(run_screen(sys.argv[2], sys.argv[3:], tile=512)))
    elif cmd == "regress":
        print(json.dumps(regress(sys.argv[2])))
    elif cmd == "holdout":
        print(json.dumps(run_holdout(sys.argv[2], sys.argv[3:])))
    else:
        raise SystemExit(__doc__)
