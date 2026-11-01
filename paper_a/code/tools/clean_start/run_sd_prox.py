"""Run the S_D + PROX operating system on one clean-start example (tools/clean_start/README.md, step 4).

  <env>/bin/python -B tools/clean_start/run_sd_prox.py tools/clean_start/examples/<example>.json [--out DIR]

Run from the repository root, with the environment from setup_env.sh, after restore.py. Stages, each checked before the
next one starts:
  1 mosaic  ls_mosaic.build_core over the manifest's pinned LGLN DGM1 tiles (no index query; tiles cached and paced by
            ls_mosaic, HTTP 429 stops); each tile's sha256 and the mosaic's values_sha256 must equal the manifest
  2 decode  dt1_decode.py run + assemble, phase 0, both members, on this interpreter (MODELS.json and PREP.json are
            written next to the outputs)
  3 select  merge1.merged + cr1_score.select(..., "prox") at the example's B under its S-v1 snapshot mask: the
            computation of work/ls_sd_pool_2026-10-26/pool.py stage 1
  4 check   counts and digests against the manifest's `expected` (exact checks; positions and scores are compared
            with tolerances only by the evaluator, work/clean_start_2026-10-30/compare.py)
Outputs stay private under outputs/: DIR/regions/LS/mosaics/, DIR/decode/, DIR/KEPT_PRIVATE.json and DIR/RUN_SUMMARY.json
(counts, digests, hashes and timings only). Default DIR: outputs/clean_start_PRIVATE/<example>/run. Exit 1 on a check
failure.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DECODER = ROOT / "work/dense_terrain1_2026-10-25/dt1_decode.py"
sys.path[:0] = [str(ROOT / "work/subtle1_2026-10-23"), str(ROOT / "work/merge1_2026-10-25")]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def digest(items: list) -> str:
    return hashlib.sha256(json.dumps(sorted(items), separators=(",", ":")).encode()).hexdigest()


def rd(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def wr(p: Path, obj, indent: int | None = 1) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(p)


def check_inputs(man: dict) -> None:
    bad = [c for c, s in man["code_sha256"].items() if sha(ROOT / c) != s]
    bad += [v["path"] for v in (*man["models"].values(), man["osm_snapshot"]) if sha(ROOT / v["path"]) != v["sha256"]]
    if bad:
        raise SystemExit(f"differ from the manifest (run restore.py, or the checkout is not the documented one): {bad}")


def mosaic(man: dict, out: Path) -> dict:
    import ls_mosaic as LM
    LM.check_pins()
    index = {k: v["url"] for k, v in man["terrain"]["tiles"].items()}
    LM.prefetch(list(index.values()), LM.WORKERS)
    rec = LM.build_core(man["i"], man["j"], out / "regions/LS/mosaics", index)
    ok = rec["tiles"] == man["terrain"]["tiles"] and rec["values_sha256"] == man["terrain"]["values_sha256"] \
        and rec["valid_core_km2"] == man["terrain"]["valid_core_km2"] and not rec["tiles_missing"]
    if not ok:
        raise SystemExit("mosaic differs from the manifest (LGLN delivery changed?); see its receipt")
    prep = {"region": "LS", "clusters": [{k: rec[k] for k in ("id", "core", "path", "sha256", "values_sha256",
                                                                 "valid_core_km2")}]}
    wr(out / "PREP.json", prep)
    return rec


def decoder_env() -> dict:
    """The reference decodes' process environment (dt1_ei_job.py, lssd_job.py): TF32 off in the CUDA libraries too,
    and none of the notebook variables a Colab kernel exports (its MPLBACKEND breaks matplotlib's import)."""
    env = dict(os.environ, NVIDIA_TF32_OVERRIDE="0", PYTHONDONTWRITEBYTECODE="1")
    for k in ("MPLBACKEND", "PYTHONPATH", "PYTHONSTARTUP"):
        env.pop(k, None)
    return env


def decode(man: dict, out: Path, device: str) -> None:
    models = {"models": {m: {"path": v["path"], "sha256": v["sha256"], "channels": v["channels"]}
                         for m, v in man["models"].items()}}
    wr(out / "MODELS.json", models)
    for cmd in ("run", "assemble"):
        args = [sys.executable, "-B", str(DECODER), cmd, str(out / "MODELS.json"), str(out / "PREP.json"),
                str(out / "decode"), "--phase", "0"] + (["--device", device] if cmd == "run" else [])
        subprocess.run(args, check=True, cwd=ROOT, env=decoder_env())


def select(man: dict, out: Path) -> list[dict]:
    """pool.py stage 1 for one core: MERGE-1 interleave under the S-v1 mask, then PROX at B."""
    import merge1 as M1
    S, SC = M1.S, M1.SC
    k, B = (man["i"], man["j"]), man["B"]
    masks = {k: S.mask_rule(S.check_reply(rd(ROOT / man["osm_snapshot"]["path"])))}
    P = {m: rd(out / f"decode/phase0/PROPOSALS_{m}_PRIVATE.json")["objects"] for m in man["models"]}
    src = {(round(p["easting"], 3), round(p["northing"], 3)): (m, p) for m in P for p in P[m]}
    kept = S.select(M1.merged(P["D_101"], P["D_202"], masks, {k: B}), {k: B}, masks, "prox", SC)
    rows = []
    for core, lst in kept.items():
        for rank, p in enumerate(lst):
            m, o = src[(round(p["easting"], 3), round(p["northing"], 3))]
            rows.append({"core": list(core), "rank_in_core": rank, "B": B, "model": m, "easting": p["easting"],
                         "northing": p["northing"], "mean_probability": o.get("mean_probability"),
                         "area_m2": o.get("area_m2"), "roundness": o.get("roundness"), "src_id": o.get("id")})
    wr(out / "KEPT_PRIVATE.json", {"example": man["example"], "B": B, "rows": rows}, None)
    return rows


def check(man: dict, out: Path, rows: list[dict]) -> dict:
    exp = man["expected"]
    props = {m: [o for o in rd(out / f"decode/phase0/PROPOSALS_{m}_PRIVATE.json")["objects"]
                 if o["cluster"] == man["example"]] for m in man["models"]}
    strata = np.load(out / f"decode/strata/{man['example']}.npz")
    got = {"proposals": {m: len(v) for m, v in props.items()},
           "proposal_ids_sha256": {m: digest([o["id"] for o in v]) for m, v in props.items()},
           "kept": len(rows), "kept_by_model": {m: sum(r["model"] == m for r in rows) for m in props},
           "kept_set_sha256": digest([[r["model"], r["src_id"]] for r in rows]),
           "strata_sha256": {k: hashlib.sha256(np.ascontiguousarray(strata[k]).tobytes()).hexdigest()
                             for k in strata.files}}
    return {k: {"got": v, "expected": exp[k], "PASS": v == exp[k]} for k, v in got.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    man = rd(Path(a.manifest))
    out = Path(a.out).resolve() if a.out else ROOT / f"outputs/clean_start_PRIVATE/{man['example']}/run"
    check_inputs(man)
    t, times = time.monotonic(), {}
    rec = mosaic(man, out)
    times["mosaic_s"], t = round(time.monotonic() - t, 1), time.monotonic()
    decode(man, out, a.device)
    times["decode_s"], t = round(time.monotonic() - t, 1), time.monotonic()
    rows = select(man, out)
    times["select_s"] = round(time.monotonic() - t, 1)
    checks = check(man, out, rows)
    summ = {"example": man["example"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifest_sha256": sha(Path(a.manifest)), "python": sys.executable,
            "mosaic": {"values_sha256": rec["values_sha256"], "container_sha256": rec["sha256"],
                       "valid_core_km2": rec["valid_core_km2"]},
            "decode_summary_sha256": sha(out / "decode/phase0/DECODE_SUMMARY.json"), "seconds": times,
            "checks": checks, "PASS": all(c["PASS"] for c in checks.values())}
    wr(out / "RUN_SUMMARY.json", summ)
    print(json.dumps({"PASS": summ["PASS"], "seconds": times, **{k: c["PASS"] for k, c in checks.items()}}))
    return 0 if summ["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
