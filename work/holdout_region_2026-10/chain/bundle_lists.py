"""Bundle lists (repo-relative files) for the Colab chain, and the Step 2c smoke plan.

  python -B bundle_lists.py lists MODE OUT_DIR [--extra PATH ...]   # MODE smoke|run -> OUT_DIR/{code,english}.txt
  python -B bundle_lists.py smoke-plan                              # candidates/SMOKE_{PLAN,MANIFEST}_CANDIDATE.json

code:    chain code, the Step 2 helpers, the frozen modules the chain imports (transitive, checked by
         tests/test_bundle.py with an import audit), the COLAB-ASSESS-1 environment files, CONT e41, the pinned public
         ADAF AO weights, one EQ-1 block with its recorded local CONT/AO outputs, the English plans of the candidate
         seeds, the NHLE sources that score_holdout.setup hash-checks, the candidate manifests, and --extra files
         (the freeze pins and the Step 2b lock, which is hashed on the VM and never read).
english: run = every R2 window (1,279 fitting positives, 741 fitting negatives, 804 holdout) raw 1 m tif + mask;
         smoke = the windows of the first 50 steps of plan 2026092800 plus the first HOLDOUT_SMOKE holdout windows.
The Step 2c smoke plan picks, by a fixed rule, the FIT positive window, FIT background window and DEV background tile
whose windows need the fewest catalogue tiles (ties: id order).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

REL = HERE.relative_to(C.ROOT).as_posix()
EQ1_BLOCK = "SCREEN_117_54"
SMOKE_UPDATES, HOLDOUT_SMOKE = 50, 4
FROZEN = [
    "verify/v3/storage_guard.py",
    "work/amazon_adapt_2026-09-28/adapt_train.py",
    "work/campaign_2026-09-22/r2_train.py", "work/campaign_2026-09-22/adaf_configs.py",
    "work/campaign_2026-09-22/r2_masks.py", "work/campaign_2026-09-22/R2_MASKS_MANIFEST_PRIVATE.json",
    "work/counterfactual_2026-09-28/plans.py", "work/counterfactual_2026-09-28/cf2.py",
    "work/counterfactual_2026-09-28/paired_train.py", "work/counterfactual_2026-09-28/ledger.py",
    "work/counterfactual_2026-09-28/cf_v1/counterfactual.py",
    "work/aim_train_audit_2026-09-24/cont_train.py", "work/aim_train_audit_2026-09-24/holdout_eval.py",
    "work/tta_ensemble_2026-09-24/tta_run.py", "work/tta_ensemble_2026-09-24/score_holdout.py",
    "work/next_experiment_2026-09-23/r3_curation.py",
    "work/adaf_head_to_head_2026-09-19/adapter.py",
    "work/adaf_head_to_head_2026-09-19/vendor/adaf/installation/aitlas-0.0.1-py3-none-any.whl",
    "work/adaf_followup_2026-09-19/english/STAGES_26_PLUS_73.json",
    "work/colab_assessment_2026-09-29/compare_eq.py", "work/colab_assessment_2026-09-29/make_bundle.py",
    "work/colab_assessment_2026-09-29/env/setup_env.sh", "work/colab_assessment_2026-09-29/env/freeze.txt",
    "verify/astra_v2_access.py", "verify/astra_v2_classes.py", "verify/astra_v2_labels.py",
    "work/astra_v2/access/nhle_scheduled_monuments.geojson",
    "targets/lidar-archaeology/v3_pro/results/labels_v3_parts_fullcohort_v1.geojson",
    "CURRENT_DETECTOR.json",
    "work/holdout_region_2026-10/step2/region_score.py", "work/holdout_region_2026-10/step2/run_chain.sh",
    "work/holdout_region_2026-10/step2/welsh_fetch.py",
    "work/holdout_region_2026-10/step2/hier_uncertainty.py", "work/holdout_region_2026-10/step2/SENSITIVITY.json",
    "work/holdout_region_2026-10/step1/SPLIT.json", "work/holdout_region_2026-10/step1/COVERAGE.json",
]
WEIGHTS = [
    "outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt",
    "work/adaf_head_to_head_2026-09-19/weights/AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
]
EQ1 = [f"outputs/bgscale_2026-09-26/stage2_cache/{EQ1_BLOCK}.tif", f"outputs/bgscale_2026-09-26/stage2_cache/{EQ1_BLOCK}.json",
       f"outputs/stage2_2026-09-24/proposals/AO/{EQ1_BLOCK}.json", f"outputs/stage2_2026-09-24/proposals/CONT/{EQ1_BLOCK}.json"]
CHAIN_CODE = ["common.py", "build_manifest.py", "masks.py", "stage_welsh.py", "welsh_train.py", "infer_dev.py",
              "parity.py", "gates.py", "cat_parts.py", "chain_lib.sh", "chain_run.sh", "chain_smoke.sh", "shards.sh",
              "vm_job.sh", "resume_merge.py", "vm_late_sink.sh",
              # driver and Step 5 side (not run on the VM): shipped so the freeze pins, checked on every VM, cover them
              "assign.py", "local_driver.sh", "drive_mount.sh", "drive_check.py", "bundle_lists.py", "bundle_stream.py",
              "projections.py", "freeze_binding.py", "step5_score.py", "sequential_sessions.py"]
CANDIDATES = {"run": ["STAGING_PLAN_CANDIDATE.json", "FIT_TABLES_CANDIDATE.json", "DEV_MANIFEST_CANDIDATE.json"],
              "smoke": ["STAGING_PLAN_CANDIDATE.json", "FIT_TABLES_CANDIDATE.json", "SMOKE_MANIFEST_CANDIDATE.json",
                        "SMOKE_PLAN_CANDIDATE.json",
                        # job.env MANIFEST is the DEV manifest in both modes; the smoke's void gate reads it
                        # (smoke 2, 2026-09-30: FileNotFoundError before this file was shipped)
                        "DEV_MANIFEST_CANDIDATE.json"]}
# Modules the chain imports inside functions (the import audit in tests/test_bundle.py loads them explicitly).
LAZY_MODULES = ("tta_run", "holdout_eval", "score_holdout", "region_score", "welsh_fetch", "compare_eq")


def plan_files(seeds: tuple[int, ...]) -> list[str]:
    return [f"outputs/counterfactual_2026-09-28/plans/PLAN_{s}_PRIVATE.json" for s in seeds]


def code_list(mode: str, extra: list[str]) -> list[str]:
    seeds = (C.STEP0_SEED,) if mode == "smoke" else tuple(sorted(set(C.TRAIN_SEEDS) | {C.STEP0_SEED}))
    out = [f"{REL}/{f}" for f in CHAIN_CODE] + FROZEN + WEIGHTS + EQ1 + plan_files(seeds)
    out += [f"{REL}/candidates/{f}" for f in CANDIDATES[mode]] + list(extra)
    return list(dict.fromkeys(out))


def r2_windows() -> list[dict]:
    return sorted(C.read_json(C.ROOT / "work/campaign_2026-09-22/R2_MASKS_MANIFEST_PRIVATE.json")["windows"],
                  key=lambda w: w["id"])


def english_list(mode: str) -> list[str]:
    wins = r2_windows()
    if mode == "smoke":
        plan = C.read_json(C.ROOT / plan_files((C.STEP0_SEED,))[0])
        idx = sorted({int(s[0]) for s in plan["real"][0][:SMOKE_UPDATES]})
        hold = [w for w in C.read_json(C.ROOT / "work/campaign_2026-09-22/R2_MASKS_MANIFEST_PRIVATE.json")["windows"]
                if w["holdout"]][:HOLDOUT_SMOKE]            # tta_run.holdout_windows() order
        wins = [wins[i] for i in idx] + hold
    rows = []
    for w in wins:
        rows += [f"outputs/campaign_2026-09-22/train_raw/{w['id']}.tif", w["mask_path"]]
    return list(dict.fromkeys(rows))


def lists(mode: str, out_dir: Path, extra: list[str]) -> dict:
    import bundle_stream as bs
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, rels in (("code", code_list(mode, extra)), ("english", english_list(mode))):
        bs.check_list(rels, tuple(extra))
        (out_dir / f"{name}.txt").write_bytes(("\n".join(rels) + "\n").encode())
        res[name] = {"files": len(rels), "bytes": sum((C.ROOT / r).stat().st_size for r in rels)}
    return res


def smoke_plan() -> dict:
    """Fixed rule: fewest catalogue tiles, then id order; writes the one-tile smoke manifest (no parts)."""
    plan = C.read_json(C.CANDIDATES / "STAGING_PLAN_CANDIDATE.json")
    man = C.read_json(C.CANDIDATES / "DEV_MANIFEST_CANDIDATE.json")

    def pick(role: str, kind: str) -> dict:
        return min((w for w in plan["windows"] if w["role"] == role and w["kind"] == kind),
                   key=lambda w: (len(w["cells"]), w["id"]))
    fit_p, fit_b, dev_b = pick("FIT", "positive"), pick("FIT", "background"), pick("DEV", "background")
    tile = next(t for t in man["tiles"] if any(w["id"] == dev_b["id"] for w in t["windows"]))
    smoke_man = {k: v for k, v in man.items() if k not in ("tiles", "parts")}
    smoke_man.update(tiles=[tile], parts=[], status="SMOKE (one background tile; format check only)",
                     parent_manifest_sha256=C.sha256_file(C.CANDIDATES / "DEV_MANIFEST_CANDIDATE.json"))
    rec = {"schema": "holdout_region_smoke_plan_v1", "rule": "fewest catalogue tiles per window, then id",
           "windows": [fit_p["id"], fit_b["id"], dev_b["id"]],
           "tiles": sorted({c for w in (fit_p, fit_b, dev_b) for c in w["cells"]}),
           "updates": SMOKE_UPDATES, "eq1_block": EQ1_BLOCK, "holdout_windows": HOLDOUT_SMOKE}
    C.write_json(C.CANDIDATES / "SMOKE_MANIFEST_CANDIDATE.json", smoke_man, indent=None)
    C.write_json(C.CANDIDATES / "SMOKE_PLAN_CANDIDATE.json", rec)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("lists", "smoke-plan"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "lists":
        print(json.dumps(lists(a.args[0], Path(a.args[1]), a.extra)))
    else:
        print(json.dumps(smoke_plan()))
