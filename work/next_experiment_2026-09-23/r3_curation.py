"""R3 pilot, hard-negative curation on FITTING squares only (never holdout, SCREEN, VALIDATION or TEST terrain).

Frozen R2 is run on the 741 local fitting negative windows; its detections are reviewed source-blind with three
outcomes, and only `non_archaeological` becomes negative supervision. `archaeological_or_plausible` and `unsure`
are ignored in the loss; `archaeological_or_plausible` detections form the fixed retention set and enter the
private lead queue. This curation review is separate from the proposal-quality evaluation review (r3_evaluate.py).

  python r3_curation.py mine [N]     # frozen R2 on the fitting negatives (first N windows only: timing smoke)
  python r3_curation.py select       # inner-area filter, 60 m NMS, review-set rule -> REVIEW_SET_PRIVATE.json
  python r3_curation.py panels       # source-blind panels, MANIFEST.json, sealed KEY_PRIVATE.json
  python r3_curation.py apply        # after LOCK.json: HN supervision, edited masks, retention set, lead queue
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize, shapes
from shapely.geometry import mapping, shape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / "work/campaign_2026-09-22"
sys.path.insert(0, str(CAMPAIGN))
sys.path.insert(0, str(HERE))
OUT = ROOT / "outputs/r3_pilot_2026-09-23"
CUR = OUT / "curation"
REVIEW = CUR / "review"
MASKS = CAMPAIGN / "R2_MASKS_MANIFEST_PRIVATE.json"
RAW = ROOT / "outputs/campaign_2026-09-22/train_raw"
R2_BEST = ROOT / "outputs/campaign_2026-09-22/training/R2_adaf_hrnet_slrm/best.pt"
R2_SHA256 = "fbd1cbacdf10c2d2806ec6dc7662196ebeed37416b908bad8f076df3113e7392"
EDGE_M, NMS_M, PER_WINDOW_CAP, REVIEW_CAP = 32.0, 60.0, 2, 400
BUFFER_M = 10.0
MIN_HARD_NEGATIVES, MIN_HN_WINDOWS = 100, 40
SEED = 20260924
OUTCOMES = ("non_archaeological", "archaeological_or_plausible", "unsure")
FORMS = ("discrete", "linear_or_field_system")
QUESTION = ("What is inside the cyan outline? Answer exactly one of: non_archaeological (a recognisable "
            "non-archaeological feature: modern track or road, quarry or modern pit, drain or ditch of modern form, "
            "field boundary of modern form, modern building platform or structure, natural landform such as a "
            "hummock, tree-throw, stream bank or slip, or a data artefact); archaeological_or_plausible (an earthwork "
            "that is or plausibly is archaeological, of any period or class, recorded or not, for example a mound, "
            "ring ditch, enclosure bank or ditch, ridge-and-furrow, lynchet, hollow-way or platform); unsure (cannot "
            "tell). For archaeological_or_plausible also give its form: discrete (mound, ring ditch, enclosure, "
            "platform, pit or other self-contained earthwork) or linear_or_field_system (ridge-and-furrow, lynchet, "
            "hollow-way, or a bank or ditch that belongs to a field system or route). Add one line saying what you "
            "see. Judge the outlined location only.")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fitting_negatives() -> list[dict]:
    m = json.loads(MASKS.read_text(encoding="utf-8"))
    return sorted((w for w in m["windows"] if w["role"] == "negative" and not w["holdout"]), key=lambda w: w["id"])


def window_bounds(w: dict) -> tuple[float, float, float, float]:
    a, _, x0, _, e, y1 = w["transform"]
    return x0, y1 + e * w["shape"][0], x0 + a * w["shape"][1], y1


# ---------------------------------------------------------------- vectorising with geometry (== adaf_configs)
def vectorise_geom(result: np.ndarray, transform, stem: str, threshold: float = .5,
                   with_geometry: bool = True) -> list[dict]:
    """adaf_configs.vectorise plus the component polygon; identical selection, ids and fields at threshold 0.5.
    Lower thresholds serve only the count-matched retention endpoint (G1)."""
    from rasterio.features import geometry_mask
    mask = (result >= threshold).astype("uint8")
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
        c = poly.centroid
        obj = {"id": f"{stem}-{len(objects):05d}", "easting": c.x, "northing": c.y,
               "score": float(values.max()), "mean_probability": float(values.mean()),
               "area_m2": area, "roundness": roundness}
        if with_geometry:
            obj["geometry"] = mapping(poly)
        objects.append(obj)
    return objects


def infer_windows(ckpt: Path, windows: list[dict], label: str, thresholds: tuple[float, ...] = ()) -> dict:
    """adaf_configs.probability_raster per window (1,024 px tiling from the window origin, no interior), then the
    same vectorising with geometry; each window's objects are checked field-by-field against the pipeline's own.
    `thresholds` adds geometry-free vectorisations at lower pixel thresholds (G1 count matching only)."""
    import torch
    import adaf_configs
    adaf_configs.check_frozen()
    torch.set_num_threads(4)
    (model,), _ = adaf_configs.load_models([ckpt])
    tmp = OUT / "tmp" / f"{label}.tif"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    t0, objects, mismatches = time.monotonic(), {}, 0
    by_threshold: dict[str, dict[str, list[dict]]] = {str(t): {} for t in thresholds}
    torch.cuda.reset_peak_memory_stats()
    for w in windows:
        raw = RAW / f"{w['id']}.tif"
        receipt = adaf_configs.probability_raster(raw, tmp, [model], phase_m=0.0)
        with rasterio.open(tmp) as ds:
            field = ds.read(1)
            geo = vectorise_geom(field, ds.transform, raw.stem)
            for t in thresholds:
                by_threshold[str(t)][w["id"]] = vectorise_geom(field, ds.transform, raw.stem, t, False)
        ref = receipt["objects"]
        same = len(ref) == len(geo) and all(
            abs(a["easting"] - b["easting"]) < 1e-6 and abs(a["northing"] - b["northing"]) < 1e-6 and
            a["score"] == b["score"] for a, b in zip(ref, geo))
        mismatches += not same
        objects[w["id"]] = geo
    tmp.unlink(missing_ok=True)
    return {"checkpoint_sha256": sha(ckpt), "windows": len(windows), "objects": objects,
            "objects_by_threshold": by_threshold,
            "vectorise_mismatch_windows": mismatches, "wall_s": time.monotonic() - t0,
            "s_per_window": (time.monotonic() - t0) / max(len(windows), 1),
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


def mine(limit: int | None) -> dict:
    if sha(R2_BEST) != R2_SHA256:
        raise RuntimeError("R2 best.pt hash changed")
    windows = fitting_negatives()
    if len(windows) != 741:
        raise RuntimeError(f"expected 741 local fitting negatives, found {len(windows)}")
    missing = [w["id"] for w in windows if not (RAW / f"{w['id']}.tif").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} fitting negative windows are not local")
    sel = windows[:limit] if limit else windows
    rec = infer_windows(R2_BEST, sel, "mine")
    rec.update({"status": "SMOKE" if limit else "COMPLETE", "model": "frozen R2 best.pt (epoch 38)",
                "terrain": "fitting-square negative windows (R2 cohort), no holdout/SCREEN/VALIDATION/TEST",
                "r3_curation_sha256": sha(Path(__file__))})
    CUR.mkdir(parents=True, exist_ok=True)
    name = "MINING_SMOKE_PRIVATE.json" if limit else "MINING_PRIVATE.json"
    (CUR / name).write_text(json.dumps(rec), encoding="utf-8")
    n = sum(len(v) for v in rec["objects"].values())
    return {k: v for k, v in rec.items() if k != "objects"} | {"detections": n}


# ---------------------------------------------------------------- review set
def rank_key(p: dict) -> tuple:
    return (-p["score"], -p["mean_probability"], -p["area_m2"], p["id"])


def nms(rows: list[dict], reach: float = NMS_M) -> list[dict]:
    """Greedy global NMS, score desc then id asc, inclusive reach (as score_staged.global_nms)."""
    kept: list[dict] = []
    for p in sorted(rows, key=lambda r: (-r["score"], r["id"])):
        if all(math.hypot(p["easting"] - q["easting"], p["northing"] - q["northing"]) > reach for q in kept):
            kept.append(p)
    return kept


def inner(p: dict, w: dict) -> bool:
    x0, y0, x1, y1 = window_bounds(w)
    return x0 + EDGE_M <= p["easting"] < x1 - EDGE_M and y0 + EDGE_M <= p["northing"] < y1 - EDGE_M


def select() -> dict:
    rec = json.loads((CUR / "MINING_PRIVATE.json").read_text(encoding="utf-8"))
    if rec["status"] != "COMPLETE" or rec["vectorise_mismatch_windows"]:
        raise RuntimeError("mining incomplete or vectorising mismatch")
    wins = {w["id"]: w for w in fitting_negatives()}
    rows = [dict(p, window=wid) for wid, objs in rec["objects"].items() for p in objs if inner(p, wins[wid])]
    kept = sorted(nms(rows), key=rank_key)
    if len(kept) <= REVIEW_CAP:
        chosen, rule = kept, "all"
    else:
        per, capped = Counter(), []
        for p in kept:
            if per[p["window"]] < PER_WINDOW_CAP:
                capped.append(p)
                per[p["window"]] += 1
        chosen, rule = capped[:REVIEW_CAP], f"top {REVIEW_CAP} by rank with <= {PER_WINDOW_CAP} per window"
    out = {"rule": rule, "raw_detections": sum(len(v) for v in rec["objects"].values()),
           "inner_detections": len(rows), "after_nms": len(kept), "review_set": len(chosen),
           "windows_with_review_items": len({p["window"] for p in chosen}), "mining_sha256": sha(CUR / "MINING_PRIVATE.json"),
           "items": chosen}
    (CUR / "REVIEW_SET_PRIVATE.json").write_text(json.dumps(out), encoding="utf-8")
    return {k: v for k, v in out.items() if k != "items"}


# ---------------------------------------------------------------- panels (source-blind)
def hillshade(z: np.ndarray, az: float = 315., alt: float = 45.) -> np.ndarray:
    gy, gx = np.gradient(z)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azr, altr = np.radians(360. - az + 90.), np.radians(alt)
    return np.clip(np.sin(altr) * np.cos(slope) + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect), 0, 1)


def render_panel(raw: Path, marker, code: str, out_png: Path, half_m: int = 150) -> dict:
    """Relief-inspection layout: hillshade + outline | LRM 10 m radius +-0.5 m, unmarked | LRM +-0.15 m + outline.
    `marker` is a shapely geometry in EPSG:27700 (detection polygon or a circle)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.features import geometry_mask
    from rasterio.transform import rowcol
    from scipy import ndimage as ndi
    with rasterio.open(raw) as ds:
        a = ds.read(1, masked=True).filled(np.nan).astype("float64")
        t = ds.transform
    fin = np.isfinite(a)
    af = np.where(fin, a, np.nanmean(a))
    k = 10
    num = ndi.uniform_filter(af, 2 * k + 1, mode="nearest")
    den = ndi.uniform_filter(fin.astype(float), 2 * k + 1, mode="nearest")
    lrm = af - num / np.maximum(den, 1e-6)
    outline = geometry_mask([mapping(marker)], out_shape=a.shape, transform=t, invert=True)
    r, c = rowcol(t, marker.centroid.x, marker.centroid.y)
    half = int(min(half_m, min(a.shape) // 2 - 1))
    r0 = int(np.clip(r - half, 0, a.shape[0] - 2 * half))
    c0 = int(np.clip(c - half, 0, a.shape[1] - 2 * half))
    sl = (slice(r0, r0 + 2 * half), slice(c0, c0 + 2 * half))
    hs, l1, m = hillshade(af)[sl], lrm[sl], outline[sl].astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.3))
    ax[0].imshow(hs, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title(f"{code}  hillshade 1 m, outline", fontsize=10)
    ax[1].imshow(l1, cmap="gray", vmin=-0.5, vmax=0.5)
    ax[1].set_title("LRM 10 m radius, +-0.5 m, unmarked", fontsize=10)
    ax[2].imshow(l1, cmap="gray", vmin=-0.15, vmax=0.15)
    ax[2].set_title("LRM +-0.15 m (high gain), outline", fontsize=10)
    for i in (0, 2):
        ax[i].contour(m, levels=[0.5], colors="cyan", linewidths=0.9)
    for x in ax:
        x.plot([10, 60], [2 * half - 10, 2 * half - 10], color="yellow", lw=2)
        x.text(12, 2 * half - 16, "50 m", color="yellow", fontsize=8)
        x.set_xticks([])
        x.set_yticks([])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=75)
    plt.close(fig)
    return {"core_m": 2 * half, "marker_inside_view": bool(m.any())}


def panels() -> dict:
    rs = json.loads((CUR / "REVIEW_SET_PRIVATE.json").read_text(encoding="utf-8"))
    items = rs["items"]
    order = np.random.default_rng(SEED).permutation(len(items))
    key, manifest = [], []
    for n, i in enumerate(order, 1):
        p, code = items[int(i)], f"M{n:03d}"
        info = render_panel(RAW / f"{p['window']}.tif", shape(p["geometry"]), code, REVIEW / "panels" / f"{code}.png")
        if not info["marker_inside_view"]:
            raise RuntimeError(f"{code}: outline not inside the rendered view")
        key.append({"code": code, "window": p["window"], "detection_id": p["id"], "easting": p["easting"],
                    "northing": p["northing"], "score": p["score"], "geometry": p["geometry"], **info})
        manifest.append({"code": code})
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "MANIFEST.json").write_text(json.dumps({"n": len(manifest), "question": QUESTION, "outcomes": OUTCOMES,
                                                      "panels": manifest}, indent=1), encoding="utf-8")
    (CUR / "KEY_PRIVATE.json").write_text(json.dumps({"seed": SEED, "review_set_sha256":
                                                      sha(CUR / "REVIEW_SET_PRIVATE.json"), "key": key}),
                                          encoding="utf-8")
    return {"panels": len(manifest), "manifest_sha256": sha(REVIEW / "MANIFEST.json")}


# ---------------------------------------------------------------- after the locked review
def locked_verdicts(review_dir: Path, codes: list[str]) -> tuple[dict, str]:
    lock = json.loads((review_dir / "LOCK.json").read_text(encoding="utf-8"))
    vpath = review_dir / "verdicts.jsonl"
    digest = sha(vpath)
    if lock["verdicts_sha256"] != digest:
        raise RuntimeError("verdicts.jsonl does not match LOCK.json")
    issues = review_dir / "RENDER_ISSUES.jsonl"
    rows = [json.loads(line) for line in vpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {r["code"]: r for r in rows}
    if len(by) != len(rows) or set(by) != set(codes) or any(r["verdict"] not in OUTCOMES for r in rows):
        raise RuntimeError("verdicts incomplete, duplicated, outside the manifest or with an invalid outcome")
    if any(r["verdict"] == "archaeological_or_plausible" and r.get("form") not in FORMS for r in rows):
        raise RuntimeError("archaeological_or_plausible verdict without a valid form")
    if issues.exists():
        pending = {json.loads(line)["code"] for line in issues.read_text(encoding="utf-8").splitlines() if line.strip()}
        if pending - set(by):
            raise RuntimeError(f"render issues still pending: {sorted(pending - set(by))}")
    return by, digest


def apply() -> dict:
    key = json.loads((CUR / "KEY_PRIVATE.json").read_text(encoding="utf-8"))["key"]
    verdicts, digest = locked_verdicts(REVIEW, [k["code"] for k in key])
    wins = {w["id"]: w for w in fitting_negatives()}
    groups = {o: [k for k in key if verdicts[k["code"]]["verdict"] == o] for o in OUTCOMES}
    hard, ignore = [], {}
    for k in groups["non_archaeological"]:
        w = wins[k["window"]]
        t = Affine(*w["transform"])
        col, row = ~t * (k["easting"], k["northing"])
        hard.append({"code": k["code"], "window": k["window"], "detection_id": k["detection_id"],
                     "polygon": k["geometry"], "col_px": col, "row_px": row, "score": k["score"]})
    for o in ("archaeological_or_plausible", "unsure"):
        for k in groups[o]:
            ignore.setdefault(k["window"], []).append(k)
    edited = []
    for wid, ks in sorted(ignore.items()):
        w = wins[wid]
        lab = np.load(ROOT / w["mask_path"])["label"].copy()
        region = rasterize([(shape(k["geometry"]).buffer(BUFFER_M), 1) for k in ks], out_shape=lab.shape,
                           transform=Affine(*w["transform"]), fill=0, dtype="uint8") == 1
        lab[region & (lab != 1)] = 255
        target = OUT / "hn_masks" / f"{wid}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, label=lab)
        rel = target.relative_to(ROOT).as_posix() if target.is_relative_to(ROOT) else target.as_posix()
        edited.append({"window": wid, "mask_path": rel, "mask_sha256": sha(target),
                       "ignored_detections": [k["code"] for k in ks], "ignored_pixels_added": int(region.sum())})
    n_windows = len({h["window"] for h in hard})
    status = "READY" if len(hard) >= MIN_HARD_NEGATIVES and n_windows >= MIN_HN_WINDOWS else "INSUFFICIENT_HARD_NEGATIVES"
    sup = {"status": status, "verdicts_sha256": digest, "key_sha256": sha(CUR / "KEY_PRIVATE.json"),
           "reviewer_ids": sorted({str(v.get("reviewer_id")) for v in verdicts.values()}),
           "counts": {o: len(v) for o, v in groups.items()}, "hard_negative_windows": n_windows,
           "minimums": {"hard_negatives": MIN_HARD_NEGATIVES, "windows": MIN_HN_WINDOWS}, "buffer_m": BUFFER_M,
           "hard_negatives": hard, "edited_masks": edited}
    (CUR / "HN_SUPERVISION.json").write_text(json.dumps(sup, indent=1), encoding="utf-8")
    retention = [{"code": k["code"], "window": k["window"], "easting": k["easting"], "northing": k["northing"],
                  "geometry": k["geometry"]} for k in groups["archaeological_or_plausible"]]
    (CUR / "RETENTION_SET.json").write_text(json.dumps({"verdicts_sha256": digest, "n": len(retention),
                                                        "items": retention}, indent=1), encoding="utf-8")
    with (CUR / "LEADS_QUEUE_PRIVATE.jsonl").open("w", encoding="utf-8") as f:
        for k in groups["archaeological_or_plausible"]:
            f.write(json.dumps({"lead_id": f"R3M-{k['code']}", "source": "R3 mining review (fitting squares)",
                                "window": k["window"], "easting": k["easting"], "northing": k["northing"],
                                "crs": "EPSG:27700", "form": verdicts[k["code"]].get("form"),
                                "review_note": verdicts[k["code"]].get("note"),
                                "reviewer_kind": verdicts[k["code"]].get("reviewer_kind"),
                                "reviewer_id": verdicts[k["code"]].get("reviewer_id"), "human_confirmed": False,
                                "status": "unvetted: not checked against HER, NMP or literature; no claim"}) + "\n")
    return {k: v for k, v in sup.items() if k not in ("hard_negatives", "edited_masks")} | {
        "retention_set": len(retention), "edited_windows": len(edited)}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "mine":
        print(json.dumps(mine(int(sys.argv[2]) if len(sys.argv) > 2 else None), indent=1))
    elif cmd == "select":
        print(json.dumps(select(), indent=1))
    elif cmd == "panels":
        print(json.dumps(panels(), indent=1))
    elif cmd == "apply":
        print(json.dumps(apply(), indent=1))
    else:
        raise SystemExit(__doc__)
