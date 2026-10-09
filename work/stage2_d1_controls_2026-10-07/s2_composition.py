"""STAGE2-D1-CONTROLS composition of the two training treatments (ROUND4_REPLY.md R4-1, "The exposure mismatch need not
be fixed for this question", items 1-5; ROUND4_REVIEW.md "Adopted" item 2). Descriptive only: these numbers describe
the intervention and never change the replay (no tuning knob).

Per arm (D1 = the all-local Lower Saxony plan; control = ADAPT-NDS-1 arm C's step map) and per seed (2026092805-14),
from the actual step tables (s2_schedule.control_steps / d1_steps: the frozen draw functions, the real (dy, dx, k) of
every update) and the masks built locally from their tables (CHECKLIST.md step 4):
  (a) draws per distinct source window (region x role);
  (b) exposure per underlying monument (positive-role draws per primary monument; identity links: English NHLE list
      entry from the window id, Welsh / Irish / Lower Saxony 'sam' of the table row), plus the share of positive draws
      whose window holds further (incidental) monuments;
  (c) the fraction of actual augmented crops (rt.crop at (dy, dx), then rt.dihedral(k)) whose target has no positive
      pixel, separately for positive-role and background-role steps (F1 applies Dice only where sum_g > 0);
  (d) the foreground-pixel-fraction distribution of those crops (positive pixels / valid pixels, zeros included);
  (e) class composition of positive-role steps (barrow / enclosure / other / mixed) overall, by region and by update
      decile (t // 512).
Target definition exactly as d1_train.terms: valid = label != IGNORE (255); g = label == 1; an all-IGNORE crop is a
skipped update. Mask-level for every region: the trainer's void policy V (adapt_train.apply_v: non-finite relief ->
IGNORE) needs the terrain, which is local only for the English windows (outputs/campaign_2026-09-22/train_raw); for
those the exact V-applied target is also computed and the difference reported (english_void_v_check).
Output: aggregates only (no window ids, monument ids or coordinates).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import s2_common as S
import s2_schedule as SS

N = S.N
rt = SS.rt
QUANTILES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
DECILE = 512
ENGLISH_POS = re.compile(r"^NHLE-(\d+)-p\d+$")


def _q(x: list[float] | np.ndarray) -> dict:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {}
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(a, q)) for q in QUANTILES}


def _class_key(classes) -> str:
    c = sorted({str(x) for x in (classes if isinstance(classes, (list, tuple)) else [classes]) if x})
    if not c:
        return "unknown"
    if len(c) > 1:
        return "mixed:" + "+".join(c)
    return c[0] if c[0] in ("barrow", "enclosure") else "other:" + c[0]


# ---------------------------------------------------------------- window metadata (monument, class, incidental)
def english_meta(windows: list[dict]) -> dict:
    out = {}
    for w in windows:
        uid = f"england:{w['id']}"
        m = ENGLISH_POS.match(w["id"])
        if w["role"] == "positive":
            if not m:
                raise SystemExit("English positive window without an NHLE identity link")
            entry = f"NHLE-{m.group(1)}-"
            own = [p for p in w["positive_parts"] if p["id"] == w["id"]] or \
                [p for p in w["positive_parts"] if p["id"].startswith(entry)]
            out[uid] = {"monument": f"england:NHLE-{m.group(1)}",
                        "class": _class_key([c for p in own for c in p["classes"]]),   # "unknown": no own part listed
                        "incidental": sum(1 for p in w["positive_parts"] if not p["id"].startswith(entry))}
        else:
            out[uid] = {"monument": None, "class": "background", "incidental": len(w["positive_parts"])}
    return out


def table_meta(region: str, table: dict, rows: list[dict]) -> dict:
    """Monument = the table's own link (table["monuments"]: monument -> its positive window ids)."""
    cls = {w["id"]: w.get("class") for w in table["positive_windows"]}
    mon = {wid: m for m, wids in table["monuments"].items() for wid in wids}
    out = {}
    for r in rows:
        if r["region"] != region:
            continue
        if r["role"] == "positive":
            if r["id"] not in mon:
                raise SystemExit(f"{region} positive row without a monument link")
            out[r["uid"]] = {"monument": f"{region}:{mon[r['id']]}", "class": _class_key(cls.get(r["id"])),
                             "incidental": int(r.get("incidental") or 0)}
        else:
            out[r["uid"]] = {"monument": None, "class": "background", "incidental": int(r.get("incidental") or 0)}
    return out


# ---------------------------------------------------------------- crop statistics (grouped by mask: one load each)
def _crop_counts(lab: np.ndarray, dy: int, dx: int, k: int) -> tuple[int, int]:
    c = rt.dihedral(rt.crop(lab, dy, dx), k)
    return int((c != rt.IGNORE).sum()), int((c == 1).sum())


def crop_tables(runs: dict, root: Path = S.ROOT, override: dict | None = None) -> dict:
    """{run key: {index in its steps: (n_valid, n_pos)}} for several runs at once, each mask loaded once; override
    maps mask path -> V-applied label (English check; only those masks are replaced)."""
    by_mask: dict[str, list[tuple]] = defaultdict(list)
    for key, steps in runs.items():
        for i, r in enumerate(steps):
            by_mask[r["mask"]].append((key, i))
    out: dict = {key: {} for key in runs}
    for m, refs in by_mask.items():
        p = Path(m) if Path(m).is_absolute() else root / m
        lab = override[m] if override and m in override else np.load(p)["label"]
        for key, i in refs:
            r = runs[key][i]
            out[key][i] = _crop_counts(lab, r["dy"], r["dx"], r["k"])
    return out


def crop_table(steps: list[dict], root: Path = S.ROOT, override: dict | None = None) -> dict[int, tuple[int, int]]:
    """{index in steps: (n_valid, n_pos)} of one run."""
    return crop_tables({0: steps}, root, override)[0]


def _void_v_label(args: tuple[str, str]) -> tuple[str, np.ndarray | None]:
    """English window: the V-applied label when the relief has non-finite pixels, else None (label unchanged)."""
    wid, mask = args
    from nds_train import at
    norm = at.relief_v(rt.RAW / f"{wid}.tif")
    lab = np.load(S.ROOT / mask)["label"]
    if norm.shape != lab.shape:
        raise SystemExit(f"English input/mask shape mismatch ({norm.shape} vs {lab.shape})")
    bad = ~np.isfinite(norm)
    if not bad.any():
        return mask, None
    out = lab.copy()
    out[bad] = rt.IGNORE
    return mask, out


def english_void_v(steps: list[dict], workers: int = 8) -> dict:
    """mask path -> V-applied label for every English window the steps read whose relief has voids."""
    need = sorted({(r["uid"].split(":", 1)[1], r["mask"]) for r in steps if r["kind"] == "english"})
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for mask, lab in ex.map(_void_v_label, need, chunksize=16):
            if lab is not None:
                out[mask] = lab
    return out


# ---------------------------------------------------------------- the five quantities for one run
def region_of(r: dict) -> str:
    return r["kind"] if r["kind"] != "lower_saxony" else "lower_saxony"


def run_composition(steps: list[dict], meta: dict, crops: dict[int, tuple[int, int]]) -> dict:
    # (a) draws per distinct source window
    a = {}
    for key in sorted({f"{region_of(r)}/{r['role']}" for r in steps}):
        c = Counter(r["uid"] for r in steps if f"{region_of(r)}/{r['role']}" == key)
        a[key] = {"draws": sum(c.values()), "distinct_windows": len(c), "max_draws": max(c.values()),
                  "histogram": {str(k): v for k, v in sorted(Counter(c.values()).items())}}
    # (b) exposure per monument (positive-role draws by primary monument)
    pos = [r for r in steps if r["role"] == "positive"]
    b: dict = {}
    for reg in sorted({region_of(r) for r in pos}) + ["all"]:
        sel = [r for r in pos if reg in ("all", region_of(r))]
        c = Counter(meta[r["uid"]]["monument"] for r in sel)
        if None in c:
            raise SystemExit("positive draw without a monument")
        b[reg] = {"positive_draws": len(sel), "monuments": len(c), "max_draws_per_monument": max(c.values()),
                  "draws_per_monument": _q(list(c.values())),
                  "histogram": {str(k): v for k, v in sorted(Counter(c.values()).items())},
                  "share_of_draws_with_incidental_monuments":
                      sum(1 for r in sel if meta[r["uid"]]["incidental"] > 0) / len(sel)}
    # (c) + (d) empty targets and foreground fraction, per role (and per region x role)
    cd: dict = {}
    for scope in ["role"] + ["region_role"]:
        keys = sorted({r["role"] if scope == "role" else f"{region_of(r)}/{r['role']}" for r in steps})
        for key in keys:
            idx = [i for i, r in enumerate(steps) if (r["role"] if scope == "role" else
                                                       f"{region_of(r)}/{r['role']}") == key]
            nv = np.array([crops[i][0] for i in idx])
            npos = np.array([crops[i][1] for i in idx])
            ok = nv > 0
            frac = npos[ok] / nv[ok]
            cd[key] = {"steps": len(idx), "all_ignore_skipped": int((~ok).sum()),
                       "empty_target": int((npos[ok] == 0).sum()),
                       "empty_target_share": float((npos[ok] == 0).mean()) if ok.any() else None,
                       "nonempty_target_share": float((npos[ok] > 0).mean()) if ok.any() else None,
                       "fg_fraction_mean": float(frac.mean()) if ok.any() else None,
                       "fg_fraction_zero_share": float((frac == 0).mean()) if ok.any() else None,
                       "fg_fraction_quantiles": _q(frac),
                       "fg_fraction_quantiles_nonzero": _q(frac[frac > 0]),
                       "valid_pixel_share_mean": float((nv / (rt.CROP * rt.CROP)).mean())}
    # (e) class composition of positive-role steps
    e: dict = {"positive_steps": dict(sorted(Counter(meta[r["uid"]]["class"] for r in pos).items())), "by_region": {},
               "barrow_share_by_decile": []}
    for reg in sorted({region_of(r) for r in pos}):
        e["by_region"][reg] = dict(sorted(Counter(meta[r["uid"]]["class"] for r in pos if region_of(r) == reg).items()))
    for d in range(-(-S.N_UPDATES // DECILE)):
        sel = [r for r in pos if r["t"] // DECILE == d]
        e["barrow_share_by_decile"].append(sum(1 for r in sel if meta[r["uid"]]["class"] == "barrow") / len(sel))
    e["barrow_share"] = e["positive_steps"].get("barrow", 0) / len(pos)
    e["enclosure_share"] = e["positive_steps"].get("enclosure", 0) / len(pos)
    return {"a_draws_per_window": a, "b_monument_exposure": b, "cd_crops": cd, "e_class_by_step": e}


def english_check(steps: list[dict], crops_mask: dict, crops_v: dict) -> dict:
    idx = [i for i, r in enumerate(steps) if r["kind"] == "english"]
    changed = [i for i in idx if crops_mask[i] != crops_v[i]]
    emp = [i for i in idx if (crops_mask[i][1] == 0) != (crops_v[i][1] == 0)]
    skip = [i for i in idx if (crops_v[i][0] == 0) and crops_mask[i][0] > 0]
    return {"english_steps": len(idx), "crops_with_target_changed_by_void_v": len(changed),
            "empty_status_changed_by_void_v": len(emp), "all_ignore_created_by_void_v": len(skip),
            "max_valid_pixels_lost": max((crops_mask[i][0] - crops_v[i][0] for i in idx), default=0)}


# ---------------------------------------------------------------- summary across seeds
def _walk(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out |= _walk(v, p)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p] = float(v)
    return out


def ranges(per_seed: dict[str, dict]) -> dict:
    """min / max over seeds of every numeric leaf (histogram leaves excluded)."""
    flat = {s: {k: v for k, v in _walk(r).items() if ".histogram." not in k and "barrow_share_by_decile" not in k}
            for s, r in per_seed.items()}
    keys = sorted(set().union(*[set(f) for f in flat.values()]))
    return {k: [min(f[k] for f in flat.values() if k in f), max(f[k] for f in flat.values() if k in f)] for k in keys}
