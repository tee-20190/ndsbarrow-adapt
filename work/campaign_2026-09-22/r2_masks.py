"""Amendment 2 B.3-B.4: rebuild NHLE scheduled parts and rasterise 0.5 m catalogue masks for every R2 window.

Label raster per window, on adapter.source_view's grid (raw 1 m transform * scale(0.5), 846 x 846):
  1   positive  part labelled by verify/astra_v2_labels.py's rule whose classes include barrow or enclosure, except
                barrow-only parts with equivalent radius > 40 m
  255 ignore    5 m band outside each positive; labelled parts of other classes only; barrow-only groups > 40 m;
                unlabelled parts named like the barrow family
  0   background, everything else. Positive pixels overwrite ignore pixels.

Parts are rebuilt from the pinned NHLE geojson with astra_v2_labels.py's numbering, which the chip inventory uses.
labels_v3_parts_fullcohort_v1 numbers some parts of 10 multi-part monuments differently; that is reported, not used.

  python r2_masks.py build      # parts rebuild + checks + masks + manifests (resumable, create-only masks)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely import STRtree
from shapely.geometry import box, shape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "verify"))
from astra_v2_classes import CLASS_MAP  # noqa: E402
from astra_v2_labels import EXCLUDE, POS, TEST, square  # noqa: E402

NHLE = ROOT / "work/astra_v2/access/nhle_scheduled_monuments.geojson"
NHLE_SHA256 = "106cf74eca30b3a836f51cd36e69381f15085ca4406bb0e76a8f7565e706672d"
LABELS = ROOT / "targets/lidar-archaeology/v3_pro/results/labels_v3_parts_fullcohort_v1.geojson"
LABELS_SHA256 = "a46044a6bdfa8bc1de6555dba155f1e9ca6e05f11631878af205e140d4cc7dd4"
INVENTORY = ROOT / "targets/lidar-archaeology/v3_pro/results/chip_inventory_fullcohort_v1.json"
COHORT = HERE / "R2_COHORT_PRIVATE.json"
RAW = ROOT / "outputs/campaign_2026-09-22/train_raw"
OUT = ROOT / "outputs/campaign_2026-09-22/r2_masks"
MANIFEST = HERE / "R2_MASKS_MANIFEST_PRIVATE.json"
FAMILY = re.compile(r"\b(?:barrows?\b|tumul|ring ditch|cairn|mound|enclosure|henge)", re.I)
COMMON = {"barrow", "enclosure"}
BAND_M, GROUP_RADIUS_M, MIN_POSITIVE_PARTS = 5.0, 40.0, 500
POS_V, IGNORE_V = 1, 255


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rule_labelled(name: str, scale: float, centroid, test_boxes: list) -> bool:
    """verify/astra_v2_labels.py acceptance, reproduced term for term."""
    if not re.search(POS, name, re.I) or re.search(EXCLUDE, name, re.I) or not 15 <= scale <= 250:
        return False
    if square(centroid.x, centroid.y) in TEST:
        return True
    return min(b.distance(centroid) for b in test_boxes) >= 5040


def part_role(labelled: bool, classes: set[str], scale: float, name: str) -> str:
    if labelled:
        if classes & COMMON:
            return "ignore_barrow_group" if classes == {"barrow"} and scale > GROUP_RADIUS_M else "positive"
        return "ignore_other_class"
    return "ignore_unlabelled_family" if FAMILY.search(name) else "none"


def labels_file_consistency(parts: list[dict]) -> dict:
    """Informational: the labels file vs the id-based rebuild (numbering and geometry twins)."""
    by_entry: dict[str, list[dict]] = {}
    by_id = {q["id"]: q for q in parts}
    for q in parts:
        if q["labelled"]:
            by_entry.setdefault(q["id"].split("-")[1], []).append(q)
    same, renumbered, no_twin = 0, [], []
    for f in json.loads(LABELS.read_text(encoding="utf-8-sig"))["features"]:
        pid, g = f["properties"]["id"], shape(f["geometry"])
        c = g.centroid
        q = by_id.get(pid)
        if q is not None and math.hypot(q["cx"] - c.x, q["cy"] - c.y) <= 0.5:
            same += 1
            continue
        twins = [t for t in by_entry.get(pid.split("-")[1], [])
                 if math.hypot(t["cx"] - c.x, t["cy"] - c.y) <= 0.5 and abs(t["geom"].area - g.area) <= 0.01 * g.area]
        (renumbered if len(twins) == 1 else no_twin).append((pid, twins[0]["id"] if len(twins) == 1 else None))
    return {"labels_parts": same + len(renumbered) + len(no_twin), "same_id_same_geometry": same,
            "renumbered_same_geometry": renumbered, "no_geometry_twin": no_twin,
            "rule_labelled_rebuild_parts": sum(q["labelled"] for q in parts)}


def rebuild_parts() -> tuple[list[dict], dict]:
    """verify/astra_v2_labels.py part rebuild (id NHLE-{ListEntry}-p{j} over all parts), plus ignore parts."""
    if sha(NHLE) != NHLE_SHA256 or sha(LABELS) != LABELS_SHA256:
        raise RuntimeError("NHLE source or labels hash changed; stop")
    test_boxes = [box(x, y, x + 100000, y + 100000) for x, y in TEST.values()]
    parts = []
    for f in json.loads(NHLE.read_text(encoding="utf-8"))["features"]:
        p = f["properties"]
        geom = shape(f["geometry"])
        pieces = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for j, g in enumerate(pieces):
            pid, scale, c = f"NHLE-{p['ListEntry']}-p{j}", math.sqrt(g.area / math.pi), g.centroid
            labelled = rule_labelled(p["Name"], scale, c, test_boxes)
            classes = {k for k, r in CLASS_MAP.items() if re.search(r, p["Name"], re.I)} if labelled else set()
            role = part_role(labelled, classes, scale, p["Name"])
            if role != "none":
                parts.append({"id": pid, "geom": g, "scale": scale, "cx": c.x, "cy": c.y,
                              "classes": sorted(classes), "role": role, "labelled": labelled})
    return parts, labels_file_consistency(parts)


def check_centroids(parts: list[dict], windows: list[dict]) -> dict:
    """Every positive window's own part: rebuilt centroid within 0.5 m of the manifest centroid."""
    inv = {r["id"]: r for r in json.loads(INVENTORY.read_text(encoding="utf-8-sig"))["records"]}
    by_id = {q["id"]: q for q in parts}
    worst, failures, own_roles = 0.0, [], Counter()
    for w in windows:
        if w["role"] != "positive":
            continue
        q, r = by_id.get(w["id"]), inv[w["id"]]
        if q is None:
            failures.append((w["id"], "not rebuilt"))
            continue
        d = math.hypot(q["cx"] - float(r["centroid_easting"]), q["cy"] - float(r["centroid_northing"]))
        worst = max(worst, d)
        own_roles[q["role"]] += 1
        if d > 0.5:
            failures.append((w["id"], round(d, 3)))
    if failures:
        raise RuntimeError(f"centroid check failed for {len(failures)} positives: {failures[:5]}")
    return {"checked": sum(own_roles.values()), "max_centroid_offset_m": worst, "own_part_roles": dict(own_roles)}


def window_grid(raw_path: Path) -> tuple[Affine, tuple[int, int], tuple[float, ...]]:
    with rasterio.open(raw_path) as ds:
        if not np.allclose(ds.res, (1.0, 1.0)):
            raise RuntimeError(f"{raw_path.name}: expected 1 m")
        return ds.transform * Affine.scale(.5, .5), (ds.height * 2, ds.width * 2), tuple(ds.bounds)


_PARTS: list[dict] = []
_TREE: STRtree | None = None


def _init(parts: list[dict]) -> None:
    global _PARTS, _TREE
    _PARTS = parts
    _TREE = STRtree([q["geom"] for q in parts])


def geometry_check(m: np.ndarray, t: Affine, g) -> dict:
    """Rasterised area within 2 % and pixel-centre centroid within 0.5 m of the stored polygon."""
    rows, cols = np.nonzero(m)
    area_px = rows.size * 0.25
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    c = g.centroid
    d = float(math.hypot(x.mean() - c.x, y.mean() - c.y)) if rows.size else float("inf")
    rel = abs(area_px - g.area) / g.area
    return {"area_rel_err": rel, "centroid_err_m": d, "pass": rel <= 0.02 and d <= 0.5}


def build_window(w: dict) -> dict:
    t, shp, bounds = window_grid(RAW / f"{w['id']}.tif")
    wbox = box(*bounds)
    reach = wbox.buffer(BAND_M)
    near = [_PARTS[i] for i in _TREE.query(reach) if _PARTS[i]["geom"].intersects(reach)]
    pos = [q for q in near if q["role"] == "positive"]
    ign = [q for q in near if q["role"] != "positive"]
    shapes_ignore = [(q["geom"].buffer(BAND_M), IGNORE_V) for q in pos] + [(q["geom"], IGNORE_V) for q in ign]
    label = (rasterize(shapes_ignore, out_shape=shp, transform=t, fill=0, dtype="uint8")
             if shapes_ignore else np.zeros(shp, np.uint8))
    pos_parts = []
    for q in pos:
        m = rasterize([(q["geom"], 1)], out_shape=shp, transform=t, fill=0, dtype="uint8")
        label[m == 1] = POS_V
        entry = {"id": q["id"], "classes": q["classes"], "pixels": int(m.sum()),
                 "inside": bool(wbox.contains(q["geom"]))}
        if entry["inside"] and q["geom"].area > 100:
            entry["geometry_check"] = geometry_check(m, t, q["geom"])
        pos_parts.append(entry)
    target = OUT / f"{w['id']}.npz"
    np.savez_compressed(target, label=label)
    return {"id": w["id"], "role": w["role"], "square": w["square"], "holdout": w["holdout"],
            "mask_path": target.relative_to(ROOT).as_posix(), "mask_sha256": sha(target),
            "shape": list(shp), "transform": list(t)[:6],
            "positive_pixels": int((label == POS_V).sum()), "ignore_pixels": int((label == IGNORE_V).sum()),
            "positive_parts": pos_parts,
            "ignore_parts": [{"id": q["id"], "role": q["role"], "classes": q["classes"], "scale_m": q["scale"],
                              "cx": q["cx"], "cy": q["cy"], "inside": bool(wbox.contains(q["geom"]))} for q in ign]}


def summarise(records: list[dict], parts: list[dict], centroid_check: dict) -> dict:
    pos_seen = {p["id"]: p["classes"] for r in records for p in r["positive_parts"] if p["pixels"] > 0}
    ign_seen = {p["id"]: p["role"] for r in records for p in r["ignore_parts"]}
    checks = [p["geometry_check"] for r in records for p in r["positive_parts"] if "geometry_check" in p]
    total = sum(r["shape"][0] * r["shape"][1] for r in records)
    return {
        "windows": len(records), "positive_windows": sum(r["role"] == "positive" for r in records),
        "negative_windows": sum(r["role"] == "negative" for r in records),
        "positive_parts_with_masks": len(pos_seen),
        "positive_parts_by_class": {c: sum(c in v for v in pos_seen.values()) for c in sorted(COMMON)},
        "ignored_parts_by_role": dict(Counter(ign_seen.values())),
        "positive_pixel_share": sum(r["positive_pixels"] for r in records) / total,
        "ignore_pixel_share": sum(r["ignore_pixels"] for r in records) / total,
        "positive_windows_without_positive_pixels": sum(r["role"] == "positive" and r["positive_pixels"] == 0
                                                        for r in records),
        "negative_windows_with_positive_pixels": sum(r["role"] == "negative" and r["positive_pixels"] > 0
                                                     for r in records),
        "geometry_checks": {"n": len(checks), "pass": sum(c["pass"] for c in checks),
                            "max_area_rel_err": max((c["area_rel_err"] for c in checks), default=None),
                            "max_centroid_err_m": max((c["centroid_err_m"] for c in checks), default=None)},
        "centroid_check": centroid_check,
        "rebuilt_parts_by_role": dict(Counter(q["role"] for q in parts)),
        "minimum_positive_parts": MIN_POSITIVE_PARTS,
        "gate": "PASS" if len(pos_seen) >= MIN_POSITIVE_PARTS else "STOP_FEWER_THAN_500_POSITIVE_PARTS",
    }


def build() -> dict:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    windows = cohort["windows"]
    missing = [w["id"] for w in windows if not (RAW / f"{w['id']}.tif").is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} raw windows missing; finish the fetch first")
    parts, join = rebuild_parts()
    centroid_check = check_centroids(parts, windows)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4, initializer=_init, initargs=(parts,)) as pool:
        records = list(pool.map(build_window, windows, chunksize=16))
    summary = summarise(records, parts, centroid_check)
    summary["nhle_labels_geometry_join"] = join
    manifest = {"schema": "r2_masks_manifest_private_v1", "cohort_sha256": sha(COHORT), "nhle_sha256": NHLE_SHA256,
                "labels_sha256": LABELS_SHA256, "builder_sha256": sha(Path(__file__)),
                "rules": {"positive": "rule-labelled part (astra_v2_labels.py) with barrow/enclosure, "
                                      "except barrow-only > 40 m",
                          "ignore": "5 m outer band of positives; other-class labelled; barrow-only > 40 m; "
                                    "unlabelled part with a barrow-family name",
                          "family_regex": FAMILY.pattern, "grid": "raw 1 m transform * scale(0.5)",
                          "values": {"background": 0, "positive": POS_V, "ignore": IGNORE_V}},
                "summary": summary, "windows": records}
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "SUMMARY.json").write_text(json.dumps({"builder_sha256": manifest["builder_sha256"],
                                                   "manifest_sha256": sha(MANIFEST), **summary}, indent=2),
                                      encoding="utf-8")
    return summary


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(json.dumps(build(), indent=1))
    else:
        raise SystemExit(__doc__)
