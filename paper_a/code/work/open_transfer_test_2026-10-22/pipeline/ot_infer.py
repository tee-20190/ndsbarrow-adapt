"""OPEN-TRANSFER-1 landscape inference: configurable arms over a region's cluster mosaics through the project's
frozen raw decoder, unchanged and as work/ext_benchmark_2026-10-18/ext_infer.py calls it: tta_run.objects_multi
(adapter.source_view 0.5 m, 1,024 px tiles + 64 px halo, SLRM radius 20, normalise +-0.5, NaN -> 0, sigmoid channel 1
of a 3-channel repeat, pixelwise max over tile overlaps, 0.5 threshold, adaf_configs.vectorise: area > 30 m2,
roundness > 0.4; no NMS). TF32 off, cudnn.benchmark off. Decoder sources are pinned (DECODER).

Arms (ARMS.json, `--arms-file`): {"arms": {"CONT": {"path", "sha256", "bytes"},
                                          "X_ENS3": {"members": [{"path", "sha256", "bytes"}, ...]}}}
  single arm   = sigmoid channel-1 map of one checkpoint (op sequence identical to ext_infer.backend.predict);
  ensemble arm = arithmetic mean of its members' sigmoid channel-1 maps per decoder tile (torch.stack(...).mean(0), as
                 tta_run's C2 arm), then decoded once by the same tile-max/threshold/vectorise path.
Core rule: a proposal is kept iff its centroid lies in its cluster's core (ot_common.in_core, half-open); cores
tile the region, so every location is decided once. Object ids are the decoder's '<cluster id>-NNNNN' (mosaic stem =
cluster id), unique within a region.

  python -I ot_infer.py check  R               # pins: arms, decoder sources, region mosaics; no model loaded
  python -I ot_infer.py freezecheck [--pin F]  # WINDOWS python: FREEZE_TEST.md committed + unchanged; pins code files
  python -I ot_infer.py equiv  [--device cuda|cpu] [--out DIR]   # Verden parity (CONT, ADAF_barrow, ADAF_AO)
  python -I ot_infer.py run    R [--device cuda] [--group N] [--arms A,B]   # refuses unless gates pass
Outputs: infer/<R>/PROPOSALS_<arm>_PRIVATE.json, infer/<R>/INFER_SUMMARY.json, per-cluster parts
infer/<R>/clusters/<arm>/<cluster>.json (atomic; a finished part is reused on resume when all its hashes match).
Gates for `run`: infer/EQUIV_RESULT.json verdict PASS, produced on THIS host and device with the current decoder;
infer/FREEZE_CHECK.json present, FREEZE_TEST.md bytes equal to the committed blob, and every pipeline file pinned there
unchanged; regions/<R>/SITES.json present (written before inference) unless --no-sites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ot_common as C  # noqa: E402

ROOT = C.ROOT
for _p in (ROOT / "work/campaign_2026-09-22", ROOT / "work/tta_ensemble_2026-09-24",
           ROOT / "work/next_experiment_2026-09-23", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ARMS_FILE = HERE / "ARMS.json"
DECODER = {  # files whose bytes define the decoder (ext_infer.DECODER, unchanged)
    "work/tta_ensemble_2026-09-24/tta_run.py": "07e9566eccf436e30762b82cd567fb77d018d0e5f09fa521486cfacc6bcf8dd2",
    "work/campaign_2026-09-22/adaf_configs.py": "9eb322234a9dd28a36154849290fa40442a5f0054ad0a6a93bd43576ca63de36",
    "work/adaf_head_to_head_2026-09-19/adapter.py": "85621f71c93a1aead31ec7eb3d4b78e46ff20fb7f3af24473bb72a58e9547d27",
}
EQ_PINS = {  # ext_infer.WEIGHTS for the parity arms (sha256 from DETECTOR_REGISTRY.json)
    "CONT": "91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383",
    "ADAF_barrow": "51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216",
    "ADAF_AO": "a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb",
}
EQ_WINDOWS = ROOT / "work/release_2026-10/repro/runnable/outputs/d1_collapse_2026-10-06/vm/dev_windows"
_EQ_RUN = ROOT / "outputs/fresh_region_2026-10-03/step2/chain_runs/run_s1_20261004T174340Z/v1/results"
EQ_RENDER_LOG = _EQ_RUN / "RENDER_LOG.json"
EQ_MANIFEST = ROOT / "outputs/fresh_region_2026-10-03/step2/INFER_MANIFEST_PRIVATE.json"
EQ_MANIFEST_SHA = "c1e1ba018bf1f510baec56331dd65d0c0fa80b6059258f0c65c4cd09c374d305"
EQ_ARMS = ("CONT", "ADAF_barrow", "ADAF_AO")
EQ_TILES = ("VER-P-000", "VER-P-004", "VER-P-007", "VER-P-019", "VER-P-044", "VER-P-045")
EQ_PROPOSALS = {a: _EQ_RUN / f"main/PROPOSALS_{a}_PRIVATE.json" for a in EQ_ARMS}
EQ_TOL = 1e-3
GPU_BUSY_MIB = 3000.0
FREEZE_REL = "work/open_transfer_test_2026-10-22/FREEZE_TEST.md"
PINNED_CODE = ("ot_common.py", "ot_fetch.py", "ot_infer.py", "ARMS.json")
CODE_SHA = C.sha(Path(__file__))
_SHA_CACHE: dict[str, str] = {}


# ------------------------------------------------------------------ arms and pins
def load_arms(path: Path = ARMS_FILE, only: list[str] | None = None) -> dict:
    """Normalise ARMS.json: every arm -> {"kind": single|ensemble, "members": [{path, sha256, bytes}]}."""
    raw = C.rd(path)["arms"]
    out = {}
    for name, spec in raw.items():
        if only and name not in only:
            continue
        members = spec["members"] if "members" in spec else [spec]
        kind = "ensemble" if "members" in spec else "single"
        if kind == "ensemble" and len(members) < 2:
            raise SystemExit(f"{name}: an ensemble needs >= 2 members")
        out[name] = {"kind": kind, "members": [{k: m[k] for k in ("path", "sha256", "bytes")} for m in members]}
    missing = sorted(set(only or ()) - set(out))
    if missing:
        raise SystemExit(f"arms not in {path.name}: {missing}")
    return out


def arm_identity(spec: dict) -> dict:
    if spec["kind"] == "single":
        return {"checkpoint_sha256": spec["members"][0]["sha256"]}
    return {"ensemble": "mean of members' sigmoid channel-1 maps", "members_sha256": [m["sha256"] for m in spec["members"]],
            "checkpoint_sha256": None}


def check_weights(specs: dict) -> dict:
    out = {}
    for name, s in specs.items():
        for m in s["members"]:
            p = ROOT / m["path"]
            if m["path"] not in _SHA_CACHE:
                if not p.exists() or p.stat().st_size != m["bytes"]:
                    raise SystemExit(f"{name}: {m['path']} missing or wrong size")
                _SHA_CACHE[m["path"]] = C.sha(p)
            if _SHA_CACHE[m["path"]] != m["sha256"]:
                raise SystemExit(f"{name}: {m['path']} differs from its sha256 pin")
        out[name] = {"kind": s["kind"], **arm_identity(s), "members": s["members"]}
    return out


def decoder_sha() -> dict:
    dec = {k: C.sha(ROOT / k) for k in DECODER}
    for k, v in DECODER.items():
        if dec[k] != v:
            raise SystemExit(f"{k}: decoder source changed")
    return dec


def check_prep(region: str) -> dict:
    prep_p = C.REGIONS / region / "PREP.json"
    prep = C.rd(prep_p)
    for c in prep["clusters"]:
        if C.sha(ROOT / c["path"]) != c["sha256"]:
            raise SystemExit(f"{c['id']}: mosaic bytes differ from PREP.json")
        if Path(c["path"]).stem != c["id"]:
            raise SystemExit(f"{c['id']}: mosaic stem must equal the cluster id (object ids)")
    return {"prep_sha256": C.sha(prep_p), "n_clusters": len(prep["clusters"]), "eval_area_km2": prep.get("eval_area_km2")}


def check(region: str, arms_file: Path) -> dict:
    res = {"utc": C.utc(), "weights": check_weights(load_arms(arms_file)), "decoder_sha256": decoder_sha(),
           **check_prep(region), "code_sha256": CODE_SHA}
    C.write_atomic(C.INFER / region / "INPUT_CHECK.json", res)
    return res


# ------------------------------------------------------------------ backend (ext_infer.backend + ensembles)
def gpu_used_mib() -> float | None:
    for exe in ("/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi"):
        try:
            r = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return float(r.stdout.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            continue
    return None


def backend(specs: dict, device: str):
    """Load every distinct checkpoint once; predict(a) -> {arm: probability map} for single and ensemble arms."""
    import numpy as np
    import torch
    import tta_run as tr
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tr.ac.check_frozen()
    if device == "cuda":
        import gc
        gc.collect()                  # release this process's own models from an earlier arm/group, so the busy
        if torch.cuda.is_initialized():   # check below sees other jobs, not our caching allocator (VM 2026-10-19)
            torch.cuda.empty_cache()
        used = gpu_used_mib()
        if used is None or used > GPU_BUSY_MIB:
            raise SystemExit(f"GPU not confirmed free ({used} MiB): one GPU job at a time")
    ckpts = list(dict.fromkeys(m["path"] for s in specs.values() for m in s["members"]))
    models, _ = tr.ac.load_models([ROOT / p for p in ckpts], device=device)
    models = dict(zip(ckpts, models))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    plan = {a: (s["kind"], [m["path"] for m in s["members"]]) for a, s in specs.items()}

    def predict(a: np.ndarray) -> dict:   # validation_run.predictor with a device argument (ext_infer)
        x = torch.tensor(np.repeat(a[None, :, :], 3, axis=0), dtype=torch.float32, device=device)[None]
        with torch.inference_mode():
            maps = {p: torch.sigmoid(m(x)["out"])[0, 1] for p, m in models.items()}
            return {arm: maps[ps[0]].cpu().numpy() if kind == "single" else
                    torch.stack([maps[p] for p in ps]).mean(0).cpu().numpy() for arm, (kind, ps) in plan.items()}
    return tr, predict, {"torch": torch.__version__, "device": device, "host": C.host(),
                         "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}


# ------------------------------------------------------------------ Verden parity gate (ext_infer.equiv, ported)
def equiv_arm(arm: str, spec: dict, device: str, tiles: dict, logged: dict) -> tuple[dict, dict]:
    import r3_curation as cur
    saved = C.rd(EQ_PROPOSALS[arm])
    if saved["checkpoint_sha256"] != EQ_PINS[arm] or saved["manifest_sha256"] != EQ_MANIFEST_SHA:
        raise SystemExit(f"saved {arm} proposals are not this checkpoint on the Verden manifest")
    tr, predict, flags = backend({arm: spec}, device)
    mine_all, saved_all, rows = [], [], {}
    for tid in EQ_TILES:
        wid = tiles[tid]["windows"][0]["id"]
        path = EQ_WINDOWS / f"{wid}.tif"
        if C.sha(path) != logged[wid]:
            raise SystemExit(f"{wid}: window differs from the FRESH-1 render log")
        rec = tr.objects_multi(path, predict, [arm])
        mine = [o for o in rec["arms"][arm]["objects"] if any(
            c[0] <= float(o["easting"]) < c[2] and c[1] <= float(o["northing"]) < c[3] for c in tiles[tid]["core"])]
        a, b = {o["id"]: o for o in mine}, {o["id"]: o for o in saved["objects"][tid]}
        common = sorted(set(a) & set(b))
        rows[tid] = {"n_mine": len(a), "n_saved": len(b), "missing": len(set(b) - set(a)),
                     "extra": len(set(a) - set(b)),
                     "geometry_differs": sum(any(a[k][f] != b[k][f] for f in ("easting", "northing", "area_m2",
                                                                                  "roundness")) for k in common),
                     "max_abs_dscore": max((abs(a[k]["score"] - b[k]["score"]) for k in common), default=0.0),
                     "max_abs_dmean": max((abs(a[k]["mean_probability"] - b[k]["mean_probability"])
                                           for k in common), default=0.0)}
        mine_all += mine
        saved_all += saved["objects"][tid]
    order = [o["id"] for o in sorted(mine_all, key=cur.rank_key)] == \
            [o["id"] for o in sorted(saved_all, key=cur.rank_key)]
    exact = order and all(r["missing"] == r["extra"] == r["geometry_differs"] == 0 for r in rows.values())
    worst = max(max(r["max_abs_dscore"], r["max_abs_dmean"]) for r in rows.values())
    return {"verdict": "PASS" if exact and worst <= EQ_TOL else "FAIL", "exact_ids_geometry_order": exact,
            "max_abs_delta": worst, "tiles": rows}, flags


def equiv(device: str, out: Path, arms_file: Path) -> dict:
    if C.sha(EQ_MANIFEST) != EQ_MANIFEST_SHA:
        raise SystemExit("Verden manifest differs")
    specs = load_arms(arms_file, list(EQ_ARMS))
    for a in EQ_ARMS:
        if specs[a]["kind"] != "single" or specs[a]["members"][0]["sha256"] != EQ_PINS[a]:
            raise SystemExit(f"{a}: {arms_file.name} entry is not the pinned parity checkpoint")
    check_weights(specs)
    tiles = {t["id"]: t for t in C.rd(EQ_MANIFEST)["tiles"]}
    logged = {r["id"]: r["sha256"] for r in C.rd(EQ_RENDER_LOG)["rows"]}
    t0, per, flags = time.monotonic(), {}, {}
    for arm in EQ_ARMS:
        per[arm], flags = equiv_arm(arm, specs[arm], device, tiles, logged)
    res = {"verdict": "PASS" if all(v["verdict"] == "PASS" for v in per.values()) else "FAIL", "utc": C.utc(),
           "rule": f"per arm: exact ids, geometry, rank_key order; |dscore|, |dmean_probability| <= {EQ_TOL}",
           "arms": per, "flags": flags, "decoder_sha256": decoder_sha(), "wall_s": round(time.monotonic() - t0, 1),
           "code_sha256": CODE_SHA}
    C.write_atomic(out / "EQUIV_RESULT.json", res)
    return res


# ------------------------------------------------------------------ freeze gate
def freezecheck(pins: list[str], addendum: bool = False) -> dict:
    """Windows-side (git never from WSL): FREEZE_TEST.md committed and unchanged; pins the pipeline files shipped to the VM,
    each of which must equal its blob at the FREEZE_TEST.md commit. `pins` = extra repo-relative files to pin (another arms
    file, region PREP/SITES). Without --addendum it refuses once any region proposals exist locally (a freeze must not
    post-date outputs). With --addendum (stage B) the existing proposals are allowed but their sha256 are recorded, the
    previous FREEZE_CHECK.json must exist, and FREEZE_TEST.md must have been re-committed after it."""
    r = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H %cI", "--", FREEZE_REL],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit("FREEZE_TEST.md is not committed: no region inference")
    commit, when = r.stdout.split()
    blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{FREEZE_REL}"], capture_output=True).stdout
    import hashlib
    res = {"utc": C.utc(), "commit": commit, "commit_time": when, "blob_sha256": hashlib.sha256(blob).hexdigest(),
           "worktree_sha256": C.sha(ROOT / FREEZE_REL),
           "files_sha256": {**{f"work/open_transfer_test_2026-10-22/pipeline/{f}": C.sha(HERE / f) for f in PINNED_CODE},
                            **{p: C.sha(ROOT / p) for p in pins}}}
    res["equal"] = res["blob_sha256"] == res["worktree_sha256"]
    if not res["equal"]:
        raise SystemExit("FREEZE_TEST.md differs from its committed version: no region inference")
    uncommitted = []
    for rel, have in res["files_sha256"].items():
        b = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"], capture_output=True)
        if rel.startswith("work/") and (b.returncode != 0 or hashlib.sha256(b.stdout).hexdigest() != have):
            uncommitted.append(rel)
    if uncommitted:
        raise SystemExit(f"pinned code not committed unchanged with FREEZE_TEST.md at {commit[:9]}: {uncommitted}")
    existing = sorted(p.relative_to(C.INFER).as_posix() for p in C.INFER.glob("*/PROPOSALS_*"))
    if not addendum:
        if existing or any(C.INFER.glob("*/INFER_SUMMARY*.json")):
            raise SystemExit("region proposals already exist: freeze would post-date outputs")
    else:
        prev_p = C.INFER / "FREEZE_CHECK.json"
        if not prev_p.exists():
            raise SystemExit("--addendum needs the stage-A FREEZE_CHECK.json")
        prev = C.rd(prev_p)
        if prev["commit"] == commit:
            raise SystemExit("--addendum needs FREEZE_TEST.md re-committed with the addendum text")
        res["addendum_of"] = {k: prev[k] for k in ("commit", "commit_time", "blob_sha256", "files_sha256")}
        res["proposals_at_addendum_sha256"] = {p: C.sha(C.INFER / p) for p in existing}
        C.write_atomic(C.INFER / f"FREEZE_CHECK_A_{prev['commit'][:9]}.json", prev)
    C.write_atomic(C.INFER / "FREEZE_CHECK.json", res)
    return res


def gates(sites_required: bool, region: str, arms_file: Path, device: str = "cuda") -> dict:
    eq_p, fc_p = C.INFER / "EQUIV_RESULT.json", C.INFER / "FREEZE_CHECK.json"
    if not eq_p.exists() or C.rd(eq_p)["verdict"] != "PASS":
        raise SystemExit("Verden parity has not passed: no region inference")
    eq = C.rd(eq_p)
    if eq["flags"].get("host") != C.host() or eq["decoder_sha256"] != decoder_sha():
        raise SystemExit("EQUIV_RESULT.json was not produced on this host with this decoder: rerun equiv here")
    if eq.get("code_sha256") != CODE_SHA or eq["flags"].get("device") != device:
        raise SystemExit("EQUIV_RESULT.json was produced by another ot_infer.py or device: rerun equiv here")
    if not fc_p.exists():
        raise SystemExit("run `ot_infer.py freezecheck` (Windows) and ship FREEZE_CHECK.json: no region inference")
    fc = C.rd(fc_p)
    if not fc["equal"] or C.sha(ROOT / FREEZE_REL) != fc["blob_sha256"]:
        raise SystemExit("FREEZE_TEST.md differs from the committed version recorded in FREEZE_CHECK.json")
    bad = [k for k, v in fc["files_sha256"].items() if C.sha(ROOT / k) != v]
    if bad:
        raise SystemExit(f"pipeline files changed since freezecheck: {bad}")
    if C.rel(arms_file) not in fc["files_sha256"]:
        raise SystemExit(f"{C.rel(arms_file)} is not pinned in FREEZE_CHECK.json (freezecheck --pin it)")
    import ot_fetch
    prep = C.rd(C.REGIONS / region / "PREP.json")
    if prep.get("code_sha256") != ot_fetch.CODE_SHA:
        raise SystemExit(f"{region}/PREP.json was built by another ot_fetch.py: refetch with the pinned code")
    sp = C.REGIONS / region / "SITES.json"
    if sites_required and not sp.exists():
        raise SystemExit(f"{sp.name} absent: eligible sites must be frozen before inference (or pass --no-sites)")
    return {"freeze_commit": f"{fc['commit']} {fc['commit_time']}", "equiv_utc": eq["utc"],
            "equiv_device": eq["flags"], "sites_sha256": C.sha(sp) if sp.exists() else None}


# ------------------------------------------------------------------ region run (resumable per cluster x arm)
def part_path(out: Path, arm: str, cid: str) -> Path:
    return out / "clusters" / arm / f"{cid}.json"


def part_ok(out: Path, arm: str, c: dict, ident: dict, dec: dict) -> bool:
    p = part_path(out, arm, c["id"])
    if not p.exists():
        return False
    r = C.rd(p)
    return (r["mosaic_sha256"] == c["sha256"] and r["arm_identity"] == ident and r["decoder_sha256"] == dec
            and r["code_sha256"] == CODE_SHA and r["core"] == c["core"])


def infer_clusters(clusters: list[dict], specs: dict, out: Path, device: str, group: int, dec: dict) -> list[dict]:
    """Every (cluster, arm) part not already done; returns the per-cluster log rows written this call."""
    idents = {a: arm_identity(s) for a, s in specs.items()}
    need = [a for a in specs if any(not part_ok(out, a, c, idents[a], dec) for c in clusters)]
    log = []
    for g in [need[i:i + group] for i in range(0, len(need), group)]:
        tr, predict, flags = backend({a: specs[a] for a in g}, device)
        for c in clusters:
            todo = [a for a in g if not part_ok(out, a, c, idents[a], dec)]
            if not todo:
                continue
            if C.sha(ROOT / c["path"]) != c["sha256"]:
                raise SystemExit(f"{c['id']}: mosaic bytes differ from PREP.json")
            t1 = time.monotonic()
            rec = tr.objects_multi(ROOT / c["path"], predict, todo)
            wall = round(time.monotonic() - t1, 1)
            for a in todo:
                objs = rec["arms"][a]["objects"]
                kept = [dict(o, cluster=c["id"]) for o in objs if C.in_core(o["easting"], o["northing"], c["core"])]
                C.write_atomic(part_path(out, a, c["id"]), {
                    "arm": a, "cluster": c["id"], "core": c["core"], "mosaic_sha256": c["sha256"],
                    "arm_identity": idents[a], "decoder_sha256": dec, "code_sha256": CODE_SHA, "device": flags,
                    "patches": rec["patches"], "valid_km2": rec["valid_km2"], "map_sha256": rec["arms"][a]["map_sha256"],
                    "n_raw": len(objs), "n_core": len(kept), "wall_s": wall, "utc": C.utc(), "objects": kept})
            log.append({"group": g, "cluster": c["id"], "arms": todo, "patches": rec["patches"],
                        "n_core": {a: sum(C.in_core(o["easting"], o["northing"], c["core"])
                                          for o in rec["arms"][a]["objects"]) for a in todo}, "wall_s": wall})
            print(json.dumps(log[-1]), flush=True)
        del tr, predict
    return log


def assemble(region: str, prep: dict, specs: dict, out: Path, extra: dict) -> dict:
    """PROPOSALS_<arm>_PRIVATE.json (cluster order, decoder order) and INFER_SUMMARY.json from the finished parts."""
    n, shas, parts_log = {}, {}, []
    for a, s in specs.items():
        objs = []
        for c in prep["clusters"]:
            part = C.rd(part_path(out, a, c["id"]))
            objs += part["objects"]
            parts_log.append({k: part[k] for k in ("arm", "cluster", "patches", "n_raw", "n_core", "map_sha256",
                                                    "wall_s")})
        ids = [o["id"] for o in objs]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{a}: duplicate object ids in region {region}")
        C.write_atomic(out / f"PROPOSALS_{a}_PRIVATE.json", {
            "arm": a, "region": region, **arm_identity(s), "kind": s["kind"],
            "rule": "decoder objects whose centroid lies in the cluster core (half-open)", "n": len(objs),
            "prep_sha256": extra["prep_sha256"], "decoder_sha256": extra["decoder_sha256"], "objects": objs})
        n[a], shas[a] = len(objs), C.sha(out / f"PROPOSALS_{a}_PRIVATE.json")
    summary = {"status": "COMPLETE", "region": region, "utc": C.utc(), "n_proposals": n, "proposals_sha256": shas,
               "mosaics_sha256": {c["id"]: c["sha256"] for c in prep["clusters"]}, "parts": parts_log, **extra}
    tag = hashlib.sha256(",".join(sorted(specs)).encode()).hexdigest()[:8]
    summary["arms_tag"] = tag
    C.write_atomic(out / f"INFER_SUMMARY_{tag}.json", summary)
    return {k: v for k, v in summary.items() if k not in ("parts", "mosaics_sha256", "weights")}


def run_region(region: str, prep: dict, specs: dict, out: Path, device: str, group: int, extra: dict) -> dict:
    t0 = time.monotonic()
    dec = decoder_sha()
    log = infer_clusters(prep["clusters"], specs, out, device, group, dec)
    extra = dict(extra, decoder_sha256=dec, weights=check_weights(specs), group=group, run_log=log,
                 wall_s_this_call=round(time.monotonic() - t0, 1), code_sha256=CODE_SHA)
    return assemble(region, prep, specs, out, extra)


def run(region: str, device: str, group: int, arms: list[str] | None, arms_file: Path, no_sites: bool) -> dict:
    g = gates(not no_sites, region, arms_file, device)
    inputs = check_prep(region)
    specs = load_arms(arms_file, arms)
    check_weights(specs)
    prep = C.rd(C.REGIONS / region / "PREP.json")
    return run_region(region, prep, specs, C.INFER / region, device, group,
                      {**g, **inputs, "arms_file_sha256": C.sha(arms_file)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "freezecheck", "equiv", "run"))
    ap.add_argument("region", nargs="?")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--arms", help="comma-separated subset of the arms file (default: all)")
    ap.add_argument("--arms-file", type=Path, default=ARMS_FILE)
    ap.add_argument("--out", type=Path, default=C.INFER)
    ap.add_argument("--no-sites", action="store_true")
    ap.add_argument("--pin", action="append", default=[], help="freezecheck: extra repo-relative file to pin")
    ap.add_argument("--addendum", action="store_true", help="freezecheck: stage-B addendum after stage-A proposals")
    a = ap.parse_args()
    if a.mode in ("check", "run") and not a.region:
        raise SystemExit("region required")
    if a.mode == "check":
        res = check(a.region, a.arms_file)
    elif a.mode == "freezecheck":
        res = freezecheck(a.pin, a.addendum)
    elif a.mode == "equiv":
        res = equiv(a.device, a.out, a.arms_file)
    else:
        res = run(a.region, a.device, a.group, a.arms.split(",") if a.arms else None, a.arms_file, a.no_sites)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
