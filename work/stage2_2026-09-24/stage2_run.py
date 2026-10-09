"""Stage-2 landscape confirmation (PREREG.md): streamed fetch, four-arm inference, frozen-rule scoring.

  python -B work/stage2_2026-09-24/stage2_run.py prefetch N   # fetch up to N more blocks now (disk floor 6 GiB)
  python -B work/stage2_2026-09-24/stage2_run.py infer R2,AO        # pass 1 (primary); source deleted after each block
  python -B work/stage2_2026-09-24/stage2_run.py infer CONT,AIMX     # pass 2 (secondary); sources re-fetched and compared
  python -B work/stage2_2026-09-24/stage2_run.py score        # STAGE2_RESULT.json
(WSL adaf-h2h-venv, from the repo root; resumable, one writer per output directory)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/stage2_2026-09-24"
SRC = OUT / "sources"
PROPS = OUT / "proposals"
STAGES = ROOT / "work/adaf_followup_2026-09-19/english/STAGES_26_PLUS_73.json"
STAGES_SHA256 = "a0447a2d3815309d30f6a88da8da61908512cfc7455075c639225285f1661804"
AIM_OUT = ROOT / "outputs/aim_train_audit_2026-09-24"
R2 = ROOT / "outputs/campaign_2026-09-22/training/R2_adaf_hrnet_slrm/best.pt"
R2_SHA256 = "fbd1cbacdf10c2d2806ec6dc7662196ebeed37416b908bad8f076df3113e7392"
ARMS = ("R2", "AO", "CONT", "AIMX")
DENSITIES = (0.5, 1.0, 2.0)
DISK_FLOOR = 6 * 2**30
CEILING_S = 3.5 * 3600
LEDGER = OUT / "GPU_TIME.json"
BOOT_N, BOOT_SEED = 2000, 20260924
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
sys.path.insert(0, str(ROOT / "work/tta_ensemble_2026-09-24"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blocks() -> list[dict]:
    if sha(STAGES) != STAGES_SHA256:
        raise RuntimeError("stage list changed")
    return json.loads(STAGES.read_text(encoding="utf-8"))["stage2_conditional"]


def done(b: dict, arms: tuple | list = ARMS) -> bool:
    return all((PROPS / a / f"{b['id']}.json").exists() for a in arms)


def fetched(b: dict) -> bool:
    rec = SRC / f"{b['id']}.json"
    return rec.exists() and (SRC / f"{b['id']}.tif").exists() and (SRC / f"{b['id']}_valid.tif").exists()


def fetch(b: dict) -> None:
    import verify.v3.v3_fetch_england as fetcher
    if shutil.disk_usage(ROOT).free < DISK_FLOOR:
        raise SystemExit(f"disk floor reached before fetching {b['id']}")
    fetcher.ACCESS_LOG = OUT / "english_source_access.jsonl"
    x0, y0, x1, y1 = b["fetch"]
    fetcher.assert_not_test(x0, y0, x1, y1)
    t0 = time.monotonic()
    m = fetcher.fetch_area(b["id"], x0, y0, x1, y1, SRC, allow_test=False, keep_pieces=False)
    rec = json.loads(m.receipt.read_text(encoding="utf-8"))
    rec["download_wall_seconds"] = time.monotonic() - t0
    m.receipt.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({"fetched": b["id"], "s": round(rec["download_wall_seconds"], 1),
                      "valid_km2": round(rec["valid_km2"], 2),
                      "free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2)}), flush=True)


def prefetch(n: int) -> None:
    todo = [b for b in blocks() if not done(b) and not fetched(b)][:n]
    for b in todo:
        fetch(b)


# ---------------------------------------------------------------- inference
def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"runs": []}


def charge(wall_s: float, note: dict) -> None:
    led = ledger()
    led["runs"].append({"wall_s": wall_s, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **note})
    led["total_wall_s"] = sum(r["wall_s"] for r in led["runs"])
    led["ceiling_s"] = CEILING_S
    LEDGER.write_text(json.dumps(led, indent=2), encoding="utf-8")


def checkpoints(arms: list[str]) -> dict[str, Path]:
    import tta_run as tr
    if sha(R2) != R2_SHA256:
        raise RuntimeError("frozen best.pt changed")
    ck = {"R2": R2, "AO": tr.ac.resolve("AO")[1]}
    for arm in [a for a in ("CONT", "AIMX") if a in arms]:
        sel = json.loads((AIM_OUT / f"HOLDOUT_SELECTION_{arm}.json").read_text(encoding="utf-8"))
        p = AIM_OUT / "training" / arm / f"epoch{sel['selected'][1:]}.pt"
        if sha(p) != sel["selected_checkpoint"]:
            raise RuntimeError(f"{arm}: selected checkpoint hash mismatch")
        ck[arm] = p
    return {a: ck[a] for a in arms}


def infer(arms: list[str]) -> dict:
    import torch
    import tta_run as tr
    sys.path.insert(0, str(ROOT / "work/aim_train_audit_2026-09-24"))
    import holdout_eval as he
    ck = checkpoints(arms)
    ck_sha = {a: sha(path) for a, path in ck.items()}
    tr.ac.check_frozen()
    models, _ = tr.ac.load_models([ck[a] for a in arms])
    predict = he.predictor(dict(zip(arms, models)))
    torch.set_num_threads(4)
    spent = sum(r["wall_s"] for r in ledger()["runs"])
    started, n_done = time.monotonic(), 0
    todo = [b for b in blocks() if not done(b, arms)]
    nxt: dict = {}
    check_path = OUT / "PASS2_SOURCE_CHECK.json"
    src_check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {}

    def bg_fetch(b: dict) -> None:
        try:
            if not fetched(b):
                fetch(b)
        except BaseException as e:  # reported when the block is reached
            nxt["error"] = repr(e)

    try:
        for i, b in enumerate(todo):
            if spent + time.monotonic() - started >= CEILING_S:
                raise SystemExit("stage-2 GPU ceiling reached; comparison INCOMPLETE")
            if "thread" in nxt:
                nxt.pop("thread").join()
            if not fetched(b):
                fetch(b)
            if i + 1 < len(todo):
                nxt["thread"] = threading.Thread(target=bg_fetch, args=(todo[i + 1],))
                nxt["thread"].start()
            t0 = time.monotonic()
            src = SRC / f"{b['id']}.tif"
            src_sha = json.loads((SRC / f"{b['id']}.json").read_text(encoding="utf-8"))["dtm_sha256"]
            first = PROPS / "R2" / f"{b['id']}.json"
            if "R2" not in arms and first.exists():
                src_check[b["id"]] = {"pass1_source_sha256": json.loads(first.read_text(encoding="utf-8"))["source_sha256"],
                                      "pass2_source_sha256": src_sha}
                check_path.write_text(json.dumps(src_check, indent=1), encoding="utf-8")
            rec = tr.objects_multi(src, predict, list(arms), interior=b["interior"])
            for a in arms:
                rows = [dict(p, source_block=b["id"]) for p in rec["arms"][a]["objects"] if tr.ac.half_open(p, b["interior"])]
                target = PROPS / a / f"{b['id']}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps({"status": "INFERENCE_COMPLETE", "config": a, "block_id": b["id"],
                                              "checkpoint_sha256": ck_sha[a], "source_sha256": src_sha,
                                              "proposal_count": len(rows), "objects": rows,
                                              "valid_km2": rec["valid_km2"], "patches": rec["patches"],
                                              "map_sha256": rec["arms"][a]["map_sha256"],
                                              "wall_seconds": time.monotonic() - t0}, indent=1), encoding="utf-8")
            for p in (SRC / f"{b['id']}.tif", SRC / f"{b['id']}_valid.tif"):
                p.unlink()
            n_done += 1
            print(json.dumps({"block": b["id"], "counts": {a: len(rec["arms"][a]["objects"]) for a in arms},
                              "wall_s": round(time.monotonic() - t0, 1), "done": n_done}), flush=True)
    finally:
        if "thread" in nxt:
            nxt["thread"].join()
        charge(time.monotonic() - started, {"blocks_run": n_done, "arms": arms})
    return {"arms": arms, "blocks_run": n_done, "remaining": sum(not done(b, arms) for b in blocks()),
            "pass2_source_mismatches": sum(v["pass1_source_sha256"] != v["pass2_source_sha256"]
                                           for v in src_check.values())}


# ---------------------------------------------------------------- scoring
def outcome(d05: int, d2: int) -> str:
    return "GAIN" if d05 >= 0 and d2 >= 0 and d05 + d2 >= 2 else "HARM" if min(d05, d2) <= -2 else "NULL"


def score() -> dict:
    import screen_scorer as ss
    bl = blocks()
    if not all(done(b, ("R2", "AO")) for b in bl):
        raise SystemExit("primary inference incomplete")
    arms = [a for a in ARMS if all(done(b, (a,)) for b in bl)]
    incomplete = [a for a in ARMS if a not in arms]
    _, _, stage = ss.frozen_inputs()
    ids = [b["id"] for b in bl]
    area = sum((b["interior"][2] - b["interior"][0]) * (b["interior"][3] - b["interior"][1]) for b in bl) / 1e6
    common = [dict(l, owner_block=b["id"]) for b in bl for l in b["labels"] if {"barrow", "enclosure"} & set(l["classes"])]
    allc = [dict(l, owner_block=b["id"]) for b in bl for l in b["labels"]]
    klass = {l["id"]: set(l["classes"]) for l in allc}
    if len(common) != 88:
        raise RuntimeError("expected 88 common-class parts")
    pools = {}
    for a in arms:
        rows = {b["id"]: [dict(o, source_block=b["id"]) for o in json.loads((PROPS / a / f"{b['id']}.json")
                                                                            .read_text(encoding="utf-8"))["objects"]] for b in bl}
        pools[a] = ss.build_pool(stage, rows, ids, True)
    n = np.array([sum(l["owner_block"] == i for l in common) for i in ids])
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(ids), size=(BOOT_N, len(ids)))
    res = {"area_km2": area, "blocks": len(ids), "common_parts": len(common), "all_parts": len(allc),
           "arms_complete": arms, "arms_incomplete": incomplete,
           "pool_size": {a: len(p) for a, p in pools.items()}, "budgets": {}}
    per_budget = {}
    for d in DENSITIES:
        k = round(d * area)
        per = {}
        for a, pool in pools.items():
            sel = [dict(r, fused_rank=i + 1) for i, r in enumerate(ss.as_mixture_rows(pool, a)[:k])]
            hits = ss.mix.match_greedy(sel, common)
            got = sorted(h["label_id"] for h in hits)
            hits_all = ss.mix.match_greedy(sel, allc)
            per[a] = {"K": k, "supply_limited": len(pool) < k, "H": len(got), "ids": got,
                      "by_block": [sum(h["label_owner_block"] == i for h in hits) for i in ids],
                      "barrow": sum("barrow" in klass[i] for i in got), "enclosure": sum("enclosure" in klass[i] for i in got),
                      "all_class_H": len(hits_all)}
        per_budget[d] = per
        entry = {a: {kk: v for kk, v in per[a].items() if kk != "by_block"} for a in arms}
        for x, y in [pr for pr in (("R2", "AO"), ("AIMX", "CONT"), ("AIMX", "R2"), ("CONT", "R2"))
                     if pr[0] in arms and pr[1] in arms]:
            xa, ya = np.array(per[x]["by_block"]), np.array(per[y]["by_block"])
            den = n[idx].sum(1)
            keep = den > 0
            delta = (xa[idx].sum(1) - ya[idx].sum(1))[keep]
            entry[f"{x}-{y}"] = {"delta": per[x]["H"] - per[y]["H"],
                                 "gained": sorted(set(per[x]["ids"]) - set(per[y]["ids"])),
                                 "lost": sorted(set(per[y]["ids"]) - set(per[x]["ids"])),
                                 "paired_block_bootstrap_90_parts": [float(q) for q in np.quantile(delta, [.05, .95])]}
        res["budgets"][f"d{d}"] = entry
    p = res["budgets"]["d0.5"]["R2-AO"]
    lo, hi = p["paired_block_bootstrap_90_parts"]
    res["primary"] = {"R2_minus_AO_d0.5": p["delta"], "interval_90": [lo, hi],
                      "verdict": ("CONFIRMED" if p["delta"] > 0 and lo > 0 else "SUPPORTED" if p["delta"] > 0
                                  else "NOT_REPLICATED")}
    res["secondary_rule_outcomes"] = {f"{x}-{y}": (outcome(res["budgets"]["d0.5"][f"{x}-{y}"]["delta"],
                                                           res["budgets"]["d2.0"][f"{x}-{y}"]["delta"])
                                                   if x in arms and y in arms else "INCOMPLETE")
                                      for x, y in (("AIMX", "CONT"), ("AIMX", "R2"), ("CONT", "R2"))}
    res["discovery_candidates_private"] = discovery(pools["R2"][:round(0.5 * area)])
    (OUT / "STAGE2_RESULT_PRIVATE.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    public = {k: v for k, v in res.items() if k != "discovery_candidates_private"}
    public["discovery_candidates_count"] = len(res["discovery_candidates_private"])
    (OUT / "STAGE2_RESULT.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    return public


def discovery(top: list[dict]) -> list[dict]:
    """R2 proposals within the d = 0.5 budget lying > 100 m from every NHLE scheduled-monument polygon."""
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree
    nhle = [shape(f["geometry"]) for f in json.loads((ROOT / "work/astra_v2/access/nhle_scheduled_monuments.geojson")
                                                      .read_text(encoding="utf-8"))["features"]]
    tree = STRtree(nhle)
    out = []
    for rank, p in enumerate(top, 1):
        pt = Point(float(p["easting"]), float(p["northing"]))
        if len(tree.query(pt, predicate="dwithin", distance=100.0)) == 0:
            out.append({"rank": rank, "id": p["id"], "block": p["source_block"], "easting": p["easting"],
                        "northing": p["northing"], "score": p["score"], "mean_probability": p["mean_probability"],
                        "area_m2": p["area_m2"]})
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prefetch":
        prefetch(int(sys.argv[2]))
    elif cmd == "infer":
        print(json.dumps(infer(sys.argv[2].split(","))))
    elif cmd == "score":
        print(json.dumps(score(), indent=1))
    else:
        raise SystemExit(__doc__)
