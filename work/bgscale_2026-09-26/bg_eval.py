"""PACKET.md evaluation: SCREEN (26 cached blocks), stage-2 (73 cached blocks) and TRAIN-holdout inference, then the
frozen pooled rule and the declared secondaries.

Inference is tta_run's copy of the frozen tiling / validity masking / vectorising (identity pass), the half-open
interior filter and one writer per arm directory, exactly as anneal_screen.py and stage2_run.py. Guards:
  SCREEN   ANNEAL e45 runs in every SCREEN pass and must reproduce its recorded objects in every block (tr.same_objects)
  stage-2  every cached mosaic must match its cache receipt hash; CONT e41 is re-inferred where the cached bytes differ
           from its recorded source and on the first matching block, where it must reproduce its recorded objects;
           elsewhere its recorded objects are reused (so NEG - CONT is also on identical bytes)
  holdout  ANNEAL e45 must reproduce its recorded holdout objects in every window
Scoring: frozen NHLE SCREEN scorer pieces (screen_scorer.build_pool / as_mixture_rows / match_greedy, full_26 cohort),
the stage-2 scorer's own recipe (STAGES_26_PLUS_73.json, 88 common parts, K = round(d x 1,817.65 km2)), bg_rule.decide.

  python -B work/bgscale_2026-09-26/bg_eval.py screen    # GPU
  python -B work/bgscale_2026-09-26/bg_eval.py stage2    # GPU, after bg_stage2_cache.py fetch
  python -B work/bgscale_2026-09-26/bg_eval.py holdout   # GPU
  python -B work/bgscale_2026-09-26/bg_eval.py score     # CPU -> RESULT.json (+ RESULT_PRIVATE.json with part ids)
(WSL /home/tee-/adaf-h2h-venv, from the repo root)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in ("work/campaign_2026-09-22", "work/tta_ensemble_2026-09-24", "work/aim_train_audit_2026-09-24",
          "work/anneal_2026-09-25", "work/stage2_2026-09-24", "work/aim_2026-09-24"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(HERE))
import bg_rule as br  # noqa: E402
import bg_train as bt  # noqa: E402
import stage2_run as s2  # noqa: E402

OUT = bt.OUT
SCREEN_DIR, STAGE2_DIR = OUT / "screen", OUT / "stage2"
CACHE = OUT / "stage2_cache"
CACHE_RECEIPT = OUT / "STAGE2_CACHE_RECEIPT.json"
HOLDOUT_OBJECTS = OUT / "HOLDOUT_OBJECTS_PRIVATE.json"
RECORDED_CONT_SCREEN = ROOT / "outputs/aim_train_audit_2026-09-24/screen/CONT"
RECORDED_ANNEAL_SCREEN = ROOT / "outputs/anneal_2026-09-25/screen/ANNEAL"
RECORDED_STAGE2 = ROOT / "outputs/stage2_2026-09-24/proposals"
RECORDED_ANNEAL_HOLDOUT = ROOT / "outputs/anneal_2026-09-25/HOLDOUT_OBJECTS_PRIVATE.json"
TRAIN_RAW = ROOT / "outputs/campaign_2026-09-22/train_raw"
RECOUNT = ROOT / "work/stage2_2026-09-24/independent_recount.py"
EXPECT_SCREEN = {"CONT": (39, 45, 52), "ANNEAL": (37, 46, 49)}
EXPECT_STAGE2 = {"CONT": (50, 55, 59), "R2": (45, 49, 52), "AO": (41, 45, 48)}
DENS = {"d0.5": 0.5, "d1.0": 1.0, "d2.0": 2.0}
HOLDOUT_DROP = 0.30  # forecast F2: NEG e45 negative-window proposal density <= 70 % of the control's


def read(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def epochs_of(arm: str) -> dict[int, Path]:
    rows = [json.loads(line) for line in (OUT / "training" / arm / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if [r["epoch"] for r in rows] != list(range(bt.FIRST_EPOCH, bt.FINAL_EPOCH + 1)):
        raise SystemExit(f"{arm}: training incomplete")
    for r in rows:
        if bt.sha(ROOT / r["checkpoint"]) != r["checkpoint_sha256"]:
            raise RuntimeError(f"{r['checkpoint']} changed since training")
    return {r["epoch"]: ROOT / r["checkpoint"] for r in rows}


def final_checkpoint(arm: str) -> Path:
    if arm == "ANNEAL":
        if bt.sha(bt.ANNEAL_E45) != bt.ANNEAL_E45_SHA256:
            raise RuntimeError("ANNEAL e45 changed")
        return bt.ANNEAL_E45
    if arm == "CONT":
        return bt.START
    return epochs_of(arm)[bt.FINAL_EPOCH]


def over_budget(started: float) -> bool:
    return time.monotonic() - started >= bt.remaining_s()


def load_models(ck: dict[str, Path]) -> dict:
    import tta_run as tr
    tr.ac.check_frozen()
    names = sorted(ck)
    models, _ = tr.ac.load_models([ck[n] for n in names])
    return dict(zip(names, models))


def write_props(path: Path, arm: str, block_id: str, ck_sha: str, rows: list[dict], rec: dict | None, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"status": "INFERENCE_COMPLETE" if rec is not None else "REUSED_RECORDED", "config": arm,
            "block_id": block_id, "checkpoint_sha256": ck_sha, "proposal_count": len(rows), "objects": rows, **extra}
    if rec is not None:
        body.update({"valid_km2": rec["valid_km2"], "patches": rec["patches"], "map_sha256": rec["arms"][arm]["map_sha256"]})
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- inference (GPU, Phase B)
def infer_screen() -> dict:
    import holdout_eval as he
    import torch
    import tta_run as tr
    ctrl = bt.control_arm()
    ck = {"NEG": final_checkpoint("NEG"), "ANNEAL": final_checkpoint("ANNEAL")}
    if ctrl == "CTRL":
        ck["CTRL"] = final_checkpoint("CTRL")
    if bt.remaining_s() <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    ck_sha = {a: bt.sha(p) for a, p in ck.items()}
    started, done = time.monotonic(), 0
    models = load_models(ck)
    predict = he.predictor(models)
    torch.set_num_threads(4)
    try:
        for block in tr.ac.screen_blocks():
            bid = block["id"]
            targets = {a: SCREEN_DIR / a / "proposals" / f"{bid}.json" for a in models}
            if all(t.exists() for t in targets.values()):
                continue
            if over_budget(started):
                raise SystemExit("GPU ceiling reached mid-SCREEN; comparison INCOMPLETE")
            t0 = time.monotonic()
            rec = tr.objects_multi(tr.ac.source_for(bid), predict, sorted(models), interior=block["interior"])
            rows = {a: [dict(p, source_block=bid) for p in rec["arms"][a]["objects"] if tr.ac.half_open(p, block["interior"])]
                    for a in models}
            recorded = read(RECORDED_ANNEAL_SCREEN / "proposals" / f"{bid}.json")
            if not tr.same_objects(rows["ANNEAL"], recorded["objects"]):
                raise RuntimeError(f"{bid}: ANNEAL e45 does not reproduce its recorded SCREEN objects; stop")
            for a in models:
                write_props(targets[a], a, bid, ck_sha[a], rows[a], rec,
                            {"wall_seconds": time.monotonic() - t0,
                             "map_equal_recorded": rec["arms"][a]["map_sha256"] == recorded["map_sha256"] if a == "ANNEAL" else None})
            done += 1
            print(json.dumps({"block": bid, "counts": {a: len(rows[a]) for a in models},
                              "wall_s": round(time.monotonic() - t0, 1)}), flush=True)
    finally:
        bt.charge("screen", time.monotonic() - started, {"blocks_run": done, "arms": sorted(models)})
    return {"arms": sorted(models), "blocks_run": done}


def infer_stage2() -> dict:
    import holdout_eval as he
    import torch
    import tta_run as tr
    cache = read(CACHE_RECEIPT)
    if cache["status"] != "COMPLETE" or bt.sha(CACHE_RECEIPT) != bt.stage_b()["sha256"][CACHE_RECEIPT.relative_to(ROOT).as_posix()]:
        raise RuntimeError("stage-2 cache is not the COMPLETE cache frozen at stage B")
    ctrl = bt.control_arm()
    ck = {"NEG": final_checkpoint("NEG"), ctrl: final_checkpoint(ctrl), "CONT": final_checkpoint("CONT")}
    if bt.remaining_s() <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    ck_sha = {a: bt.sha(p) for a, p in ck.items()}
    per = cache["per_block"]
    blocks = s2.blocks()
    repro = next((b["id"] for b in blocks if per[b["id"]]["matches_cont_recorded"]), None)
    started, done = time.monotonic(), 0
    models = load_models(ck)
    main = sorted(a for a in models if a != "CONT")
    pred = {False: he.predictor({a: models[a] for a in main}), True: he.predictor(models)}
    torch.set_num_threads(4)
    try:
        for b in blocks:
            bid = b["id"]
            targets = {a: STAGE2_DIR / a / "proposals" / f"{bid}.json" for a in models}
            if all(t.exists() for t in targets.values()):
                continue
            if over_budget(started):
                raise SystemExit("GPU ceiling reached mid-stage-2; comparison INCOMPLETE")
            src = CACHE / f"{bid}.tif"
            if bt.sha(src) != per[bid]["dtm_sha256"]:
                raise RuntimeError(f"{bid}: cached mosaic changed")
            cont_run = (not per[bid]["matches_cont_recorded"]) or bid == repro
            names = sorted(models) if cont_run else main
            t0 = time.monotonic()
            rec = tr.objects_multi(src, pred[cont_run], names, interior=b["interior"])
            rows = {a: [dict(p, source_block=bid) for p in rec["arms"][a]["objects"] if tr.ac.half_open(p, b["interior"])]
                    for a in names}
            recorded = read(RECORDED_STAGE2 / "CONT" / f"{bid}.json")
            if bid == repro and not tr.same_objects(rows["CONT"], recorded["objects"]):
                raise RuntimeError(f"{bid}: CONT e41 does not reproduce its recorded stage-2 objects on identical bytes")
            extra = {"source_sha256": per[bid]["dtm_sha256"], "wall_seconds": time.monotonic() - t0}
            for a in names:
                write_props(targets[a], a, bid, ck_sha[a], rows[a], rec, extra)
            if not cont_run:
                write_props(targets["CONT"], "CONT", bid, ck_sha["CONT"], recorded["objects"], None,
                            {"source_sha256": recorded["source_sha256"],
                             "recorded_path": (RECORDED_STAGE2 / "CONT" / f"{bid}.json").relative_to(ROOT).as_posix(),
                             "recorded_sha256": bt.sha(RECORDED_STAGE2 / "CONT" / f"{bid}.json")})
            done += 1
            print(json.dumps({"block": bid, "cont_run": cont_run, "counts": {a: len(rows[a]) for a in names},
                              "wall_s": round(time.monotonic() - t0, 1)}), flush=True)
    finally:
        bt.charge("stage2", time.monotonic() - started, {"blocks_run": done, "arms": sorted(models)})
    return {"arms": sorted(models), "blocks_run": done, "cont_reproduction_block": repro}


def infer_holdout() -> dict:
    import holdout_eval as he
    import torch
    import tta_run as tr
    ctrl = bt.control_arm()
    ck = {f"NEG_e{e}": p for e, p in epochs_of("NEG").items()}
    ck["ANNEAL_e45"] = final_checkpoint("ANNEAL")
    if ctrl == "CTRL":
        ck["CTRL_e45"] = final_checkpoint("CTRL")
    if bt.remaining_s() <= 0:
        raise SystemExit("GPU ceiling reached; comparison INCOMPLETE")
    recorded = read(RECORDED_ANNEAL_HOLDOUT)["objects"]["ANNEAL_e45"]
    started = time.monotonic()
    models = load_models(ck)
    predict = he.predictor(models)
    torch.set_num_threads(4)
    objs: dict[str, dict] = {n: {} for n in models}
    wins = tr.holdout_windows()
    try:
        for w in wins:
            if over_budget(started):
                raise SystemExit("GPU ceiling reached mid-holdout; secondary INCOMPLETE")
            rec = tr.objects_multi(TRAIN_RAW / f"{w['id']}.tif", predict, sorted(models))
            if not tr.same_objects(rec["arms"]["ANNEAL_e45"]["objects"], recorded[w["id"]]):
                raise RuntimeError(f"{w['id']}: ANNEAL e45 does not reproduce its recorded holdout objects; stop")
            for n in models:
                objs[n][w["id"]] = rec["arms"][n]["objects"]
    finally:
        bt.charge("holdout", time.monotonic() - started, {"windows": len(objs["ANNEAL_e45"]), "checkpoints": sorted(models)})
    receipt = {"windows": len(wins), "checkpoints": {n: bt.sha(p) for n, p in ck.items()},
               "wall_s": time.monotonic() - started, "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    HOLDOUT_OBJECTS.write_text(json.dumps({"receipt": receipt, "objects": objs}), encoding="utf-8")
    return receipt


# ---------------------------------------------------------------- scoring (CPU)
def screen_hits(dirs: dict[str, Path]) -> dict:
    """Frozen SCREEN scorer, full_26 cohort: per arm and density, hits, part ids, per-block hits and classes."""
    import score_screen as sc
    ss = sc.ss
    data, blocks, stage = ss.frozen_inputs()
    ids = ss.cohort_blocks(blocks)["full_26"]
    labels = data["labels"]["full_26"]
    classes = {lab["id"]: set(lab["classes"]) for b in blocks for lab in b["labels"]}
    out: dict = {"_blocks": ids}
    for arm, d in dirs.items():
        pool = ss.build_pool(stage, sc.rows_for(d), ids, True)
        out[arm] = {"pool": len(pool)}
        for dens, k in br.SCREEN_K.items():
            h = sc.hits_for(pool, arm, k, labels)
            got = sorted(x["label_id"] for x in h)
            out[arm][dens] = {"K": k, "H": len(got), "ids": got, "supply_limited": len(pool) < k,
                              "by_block": [sum(x["label_owner_block"] == b for x in h) for b in ids],
                              "barrow": sum("barrow" in classes[i] for i in got),
                              "enclosure": sum("enclosure" in classes[i] for i in got)}
    return out


def stage2_rows(arm_dir: Path, recorded_layout: bool = False) -> dict[str, list[dict]]:
    f = (lambda b: arm_dir / f"{b}.json") if recorded_layout else (lambda b: arm_dir / "proposals" / f"{b}.json")
    return {b["id"]: [dict(o, source_block=b["id"]) for o in read(f(b["id"]))["objects"]] for b in s2.blocks()}


def stage2_hits(rows_by_arm: dict[str, dict]) -> dict:
    """stage2_run.score's recipe for any arm: global NMS + mean-rank pool, K = round(d x area), greedy 60 m match."""
    import screen_scorer as ss
    bl = s2.blocks()
    ids = [b["id"] for b in bl]
    area = sum((b["interior"][2] - b["interior"][0]) * (b["interior"][3] - b["interior"][1]) for b in bl) / 1e6
    common = [dict(lab, owner_block=b["id"]) for b in bl for lab in b["labels"] if {"barrow", "enclosure"} & set(lab["classes"])]
    if len(common) != 88:
        raise RuntimeError("expected 88 common-class parts")
    klass = {lab["id"]: set(lab["classes"]) for lab in common}
    _, _, stage = ss.frozen_inputs()
    out: dict = {"_blocks": ids, "_area_km2": area}
    for arm, rows in rows_by_arm.items():
        pool = ss.build_pool(stage, rows, ids, True)
        out[arm] = {"pool": len(pool)}
        for dens, d in DENS.items():
            k = round(d * area)
            if k != br.STAGE2_K[dens]:
                raise RuntimeError("stage-2 budget differs from the frozen K")
            sel = [dict(r, fused_rank=i + 1) for i, r in enumerate(ss.as_mixture_rows(pool, arm)[:k])]
            hits = ss.mix.match_greedy(sel, common)
            got = sorted(h["label_id"] for h in hits)
            out[arm][dens] = {"K": k, "H": len(got), "ids": got, "supply_limited": len(pool) < k,
                              "by_block": [sum(h["label_owner_block"] == i for h in hits) for i in ids],
                              "barrow": sum("barrow" in klass[i] for i in got),
                              "enclosure": sum("enclosure" in klass[i] for i in got)}
    return out


def pooled_delta(sh: dict, th: dict, a: str, b: str) -> dict[str, np.ndarray]:
    return {d: np.array(sh[a][d]["by_block"] + th[a][d]["by_block"]) - np.array(sh[b][d]["by_block"] + th[b][d]["by_block"])
            for d in DENS}


def pooled_table(sh: dict, th: dict, arms: list[str]) -> dict:
    return {a: {d: {"H": sh[a][d]["H"] + th[a][d]["H"], "screen": sh[a][d]["H"], "stage2": th[a][d]["H"],
                    "barrow": sh[a][d]["barrow"] + th[a][d]["barrow"],
                    "enclosure": sh[a][d]["enclosure"] + th[a][d]["enclosure"]} for d in DENS} for a in arms}


def holdout_scores(ctrl: str) -> dict:
    import anneal_holdout as ah
    import score_holdout as sh
    rec = read(HOLDOUT_OBJECTS)
    s = sh.setup()
    pools = {n: sh.ranked_pool(o, s) for n, o in rec["objects"].items()}
    summ = {n: sh.arm_summary(p, s) for n, p in pools.items()}
    table = {n: {"primary": round(v["primary"], 4), **{f"d{d}": round(v[f"d{d}"]["recall"], 4) for d in sh.DENSITIES},
                 "enclosure_d0.5_1_2": [v[f"d{d}"]["by_class"]["enclosure"] for d in sh.DENSITIES],
                 "neg_per_km2": round(v["negative_per_km2"], 3)} for n, v in sorted(summ.items())}
    a, b = "NEG_e45", f"{ctrl}_e45"
    ratio = summ[a]["negative_per_km2"] / summ[b]["negative_per_km2"]
    return {"table": table, "pair": [a, b], "primary_delta": round(summ[a]["primary"] - summ[b]["primary"], 4),
            "bootstrap_90": ah.paired_bootstrap(pools[a], pools[b], s), "neg_density_ratio": round(ratio, 4),
            "F2_neg_density_fell_30pct": ratio <= 1 - HOLDOUT_DROP,
            "ANNEAL_e45_primary_recorded_0.7767": round(summ["ANNEAL_e45"]["primary"], 4) == 0.7767}


def aim_yield(dirs: dict[str, Path], a: str, b: str) -> dict:
    import aim_analyze as aa
    aa.ARM_DIRS = dict(dirs)
    res = aa.screen()
    rng = np.random.default_rng(br.BOOT_SEED)
    idx = rng.integers(0, br.N_SCREEN, size=(br.BOOT_N, br.N_SCREEN))
    out = {"Y_T_r20": {arm: {f"K{k}": sum(res["results"][arm][f"K{k}"]["Y_T_by_block"]) for k in aa.BUDGETS}
                       for arm in res["arms"]}, "coverage_km2_total": round(sum(res["coverage_km2"].values()), 1)}
    for k in aa.BUDGETS:
        x, y = np.array(res["results"][a][f"K{k}"]["Y_T_by_block"]), np.array(res["results"][b][f"K{k}"]["Y_T_by_block"])
        out[f"{a}-{b}_K{k}_boot90"] = [float(q) for q in np.quantile(x[idx].sum(1) - y[idx].sum(1), [.05, .95])]
    return out


def independent_recount_module():
    """work/stage2_2026-09-24/independent_recount.py loaded by path (the campaign has a different module of that name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage2_independent_recount", RECOUNT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recount(dirs_screen: dict[str, Path], dirs_stage2: dict[str, Path]) -> dict:
    """The stage-2 independent recount's functions (written from the rule text; no shared scorer code)."""
    ir = independent_recount_module()
    out = {}
    for stage, dirs, ks in (("stage1", dirs_screen, (323, 1294)), ("stage2", dirs_stage2, (909, 3635))):
        ids, lab = ir.labels(stage)
        for arm, d in dirs.items():
            ranked = ir.pool(d, ids)
            out[f"{stage}_{arm}"] = {f"K{k}": ir.hits(ranked, lab, k) for k in ks}
    return out


def score() -> dict:
    ctrl = bt.control_arm()
    level = br.selected_level()
    sdirs = {"NEG": SCREEN_DIR / "NEG", ctrl: SCREEN_DIR / ctrl, "CONT": RECORDED_CONT_SCREEN}
    if ctrl == "CTRL":
        sdirs["ANNEAL"] = SCREEN_DIR / "ANNEAL"
    sh = screen_hits(sdirs)
    if tuple(sh["CONT"][d]["H"] for d in DENS) != EXPECT_SCREEN["CONT"]:
        raise SystemExit("recorded CONT e41 SCREEN proposals no longer score 39/45/52")
    tdirs = {"NEG": STAGE2_DIR / "NEG", ctrl: STAGE2_DIR / ctrl, "CONT": STAGE2_DIR / "CONT"}
    th = stage2_hits({**{a: stage2_rows(d) for a, d in tdirs.items()},
                      "CONT_recorded": stage2_rows(RECORDED_STAGE2 / "CONT", recorded_layout=True)})
    if tuple(th["CONT_recorded"][d]["H"] for d in DENS) != EXPECT_STAGE2["CONT"]:
        raise SystemExit("recorded CONT e41 stage-2 proposals no longer score 50/55/59")
    primary = br.decide(pooled_delta(sh, th, "NEG", ctrl), level)
    vs_cont = br.decide(pooled_delta(sh, th, "NEG", "CONT"), level)
    adopt = primary["outcome"] == "GAIN" and vs_cont["d0.5"]["D"] >= 0 and vs_cont["d2.0"]["D"] >= -br.MAX_D2_DROP
    res = {"control": ctrl, "level": level, "rule": br.rule_text(level),
           "pooled": pooled_table(sh, th, ["NEG", ctrl, "CONT"]),
           "primary_NEG_minus_control": primary, "NEG_minus_CONT_e41": vs_cont,
           "adoption": {"adopt_NEG_e45_as_reference": adopt,
                        "rule": "primary GAIN, and NEG - CONT e41 pooled D(0.5) >= 0 and D(2) >= -2"},
           "screen": {a: {d: {k: v for k, v in sh[a][d].items() if k != "by_block"} for d in DENS} | {"pool": sh[a]["pool"]}
                      for a in sdirs},
           "stage2": {a: {d: {k: v for k, v in th[a][d].items() if k != "by_block"} for d in DENS} | {"pool": th[a]["pool"]}
                      for a in (*tdirs, "CONT_recorded")},
           "recount": recount({a: sdirs[a] for a in ("NEG", ctrl)}, {a: tdirs[a] for a in ("NEG", ctrl)})}
    for a in ("NEG", ctrl):
        if [res["recount"][f"stage1_{a}"][f"K{k}"] for k in (323, 1294)] != [sh[a]["d0.5"]["H"], sh[a]["d2.0"]["H"]] or \
                [res["recount"][f"stage2_{a}"][f"K{k}"] for k in (909, 3635)] != [th[a]["d0.5"]["H"], th[a]["d2.0"]["H"]]:
            res["recount_disagrees"] = True
    if ctrl == "CTRL":
        res["replicate_ANNEAL_vs_CTRL_screen"] = {d: sh["ANNEAL"][d]["H"] - sh["CTRL"][d]["H"] for d in DENS}
    res["holdout"] = holdout_scores(ctrl) if HOLDOUT_OBJECTS.exists() else "INCOMPLETE"
    res["aim_screen"] = aim_yield({"NEG": sdirs["NEG"], ctrl: sdirs[ctrl], "CONT": RECORDED_CONT_SCREEN}, "NEG", ctrl)
    res["forecasts"] = {"F1_P0.25_GAIN": primary["outcome"] == "GAIN",
                        "F2_P0.60_holdout_neg_density_fell_30pct": res["holdout"]["F2_neg_density_fell_30pct"]
                        if isinstance(res["holdout"], dict) else None,
                        "F3_P0.30_pooled_2km2_drop_ge_3": primary["d2.0"]["D"] <= -3}
    res["gpu_ledger"] = bt.ledger()
    (OUT / "RESULT_PRIVATE.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    public = json.loads(json.dumps(res))
    for part in ("screen", "stage2"):
        for a in public[part]:
            for d in DENS:
                public[part][a][d].pop("ids", None)
    (OUT / "RESULT.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    return public


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"screen": infer_screen, "stage2": infer_stage2, "holdout": infer_holdout, "score": score}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    r = fn()
    print(json.dumps({k: r[k] for k in ("control", "primary_NEG_minus_control", "adoption", "forecasts")}
                     if cmd == "score" else r, indent=1))
