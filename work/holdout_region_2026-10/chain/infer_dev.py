"""HOLDOUT-REGION-1 Stage A item 4: DEV inference driver (zero-shot CONT e41 and ADAF AO; every trained model).

Every model and every window go through the frozen English inference path, B0 (tta_run.objects_multi:
adapter.source_view bilinear 0.5 m, 1,024 px tiles + 64 px halo, SLRM radius 20 cells, normalise +-0.5 m, NaN -> 0,
sigmoid channel 1, validity masking, adaf_configs.vectorise); several models share one pass (holdout_eval.predictor,
one map per model). Geometry is identical for positive and background tiles: each manifest window is its core box +
512 m halo (2,024 m square, 16 tiles of 1,024 px). Window j of a tile keeps the objects whose centroid lies in its
owned core (core j and no earlier core of the same tile, half-open), so a merged tile scores each location once.

Output per model and window: OUT/<model>/<window>.json (atomic; resumable: a window is skipped when every requested
model already has it for the same checkpoint and manifest). `assemble` joins windows into the region_score.py input
{"objects": {tile id: [...]}} with every tile present (fails closed on a missing window or mixed checkpoints).

  python -B infer_dev.py run MANIFEST WIN_DIR OUT --model CONT --model ADAF_AO [--model NAME=PATH@SHA256] [--shard 0/1]
  python -B infer_dev.py assemble MANIFEST OUT NAME DEST_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

ROOT = C.ROOT
for _p in ("work/campaign_2026-09-22", "work/tta_ensemble_2026-09-24", "work/aim_train_audit_2026-09-24"):
    sys.path.insert(0, str(ROOT / _p))
CONT = ROOT / "outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt"
CONT_SHA256 = "91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383"
AO_SHA256 = "a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb"   # CURRENT_DETECTOR.json control


def model_spec(text: str) -> tuple[str, Path, str | None]:
    """CONT | ADAF_AO | NAME=PATH[@SHA256]."""
    if text == "CONT":
        return "CONT", CONT, CONT_SHA256
    if text == "ADAF_AO":
        import tta_run as tr
        return "ADAF_AO", tr.ac.resolve("AO")[1], AO_SHA256
    name, _, rest = text.partition("=")
    path, _, sha = rest.partition("@")
    if not name or not path:
        raise SystemExit(f"model spec {text!r}: use CONT, ADAF_AO or NAME=PATH[@SHA256]")
    return name, Path(path), sha or None


def windows(manifest: dict) -> list[tuple[dict, int, dict]]:
    return sorted(((t, j, w) for t in manifest["tiles"] for j, w in enumerate(t["windows"])), key=lambda r: r[2]["id"])


def owned(o: dict, tile: dict, j: int) -> bool:
    e, n = float(o["easting"]), float(o["northing"])
    return C.in_box(e, n, tile["core"][j]) and not any(C.in_box(e, n, tile["core"][i]) for i in range(j))


def out_path(out: Path, name: str, wid: str) -> Path:
    return out / name / f"{wid}.json"


def done(out: Path, name: str, wid: str, ck: str, msha: str) -> bool:
    p = out_path(out, name, wid)
    if not p.exists():
        return False
    r = C.read_json(p)
    return r["checkpoint_sha256"] == ck and r["manifest_sha256"] == msha


def run(manifest_path: Path, win_dir: Path, out: Path, specs: list[str], shard: str = "0/1") -> dict:
    import torch
    import holdout_eval as he
    import tta_run as tr
    flags = C.torch_flags()                  # TF32 off at the PyTorch level too, as in the parity gate (parity.py)
    i, n = (int(v) for v in shard.split("/"))
    m, msha = C.read_json(manifest_path), C.sha256_file(manifest_path)
    if not win_dir.resolve().is_relative_to(ROOT):
        raise SystemExit(f"{win_dir}: raw windows must sit under the repo root (vectorise records relative paths)")
    models = [model_spec(s) for s in specs]
    shas = {}
    for name, path, want in models:
        shas[name] = C.sha256_file(path)
        if want and shas[name] != want:
            raise SystemExit(f"{name}: checkpoint sha256 {shas[name]} != declared {want}")
    todo = [r for k, r in enumerate(windows(m)) if k % n == i
            and not all(done(out, name, r[2]["id"], shas[name], msha) for name, _, _ in models)]
    if not todo:
        return {"shard": shard, "windows": 0, "models": [name for name, _, _ in models], "status": "ALREADY_DONE"}
    tr.ac.check_frozen()
    nets, _ = tr.ac.load_models([p for _, p, _ in models])
    names = [name for name, _, _ in models]
    predict = he.predictor(dict(zip(names, nets)))
    torch.set_num_threads(4)
    t0 = time.monotonic()
    for tile, j, w in todo:
        path = win_dir / f"{w['id']}.tif"
        s0 = time.monotonic()
        rec = tr.objects_multi(path, predict, names)
        raw_sha = C.sha256_file(path)
        for name in names:
            objs = [o for o in rec["arms"][name]["objects"] if owned(o, tile, j)]
            C.write_json(out_path(out, name, w["id"]), {
                "model": name, "window": w["id"], "tile": tile["id"], "role": tile["role"], "core_index": j,
                "checkpoint_sha256": shas[name], "manifest_sha256": msha, "raw_sha256": raw_sha,
                "map_sha256": rec["arms"][name]["map_sha256"], "valid_km2": rec["valid_km2"],
                "patches": rec["patches"], "wall_s": round(time.monotonic() - s0, 2), "objects": objs}, indent=None)
    return {"shard": shard, "windows": len(todo), "models": names, "wall_s": round(time.monotonic() - t0, 1),
            "s_per_window": round((time.monotonic() - t0) / max(len(todo), 1), 2),
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "torch_flags": flags}


def assemble(manifest_path: Path, out: Path, name: str, dest: Path) -> Path:
    """PROPOSALS_<name>_PRIVATE.json in the region_score.py input form; fails closed on gaps."""
    m, msha = C.read_json(manifest_path), C.sha256_file(manifest_path)
    objects, shas, ids = {t["id"]: [] for t in m["tiles"]}, set(), set()
    for tile, j, w in windows(m):
        p = out_path(out, name, w["id"])
        if not p.exists():
            raise SystemExit(f"{name}: window {w['id']} has no proposals")
        r = C.read_json(p)
        if r["manifest_sha256"] != msha or r["tile"] != tile["id"] or r["core_index"] != j:
            raise SystemExit(f"{name}: {w['id']} was produced for another manifest or tile")
        if any(not owned(o, tile, j) for o in r["objects"]):
            raise SystemExit(f"{name}: {w['id']} holds an object outside its owned core")
        shas.add(r["checkpoint_sha256"])
        for o in r["objects"]:
            if o["id"] in ids:
                raise SystemExit(f"{name}: duplicate object id {o['id']}")
            ids.add(o["id"])
        objects[tile["id"]] += r["objects"]
    if len(shas) != 1:
        raise SystemExit(f"{name}: windows come from {len(shas)} checkpoints")
    target = dest / f"PROPOSALS_{name}_PRIVATE.json"
    C.write_json(target, {"model": name, "checkpoint_sha256": shas.pop(), "manifest_sha256": msha,
                          "objects": objects}, indent=None)
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "assemble"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.cmd == "run":
        print(json.dumps(run(Path(a.args[0]), Path(a.args[1]), Path(a.args[2]), a.model, a.shard)))
    else:
        print(assemble(Path(a.args[0]), Path(a.args[1]), a.args[2], Path(a.args[3])))
