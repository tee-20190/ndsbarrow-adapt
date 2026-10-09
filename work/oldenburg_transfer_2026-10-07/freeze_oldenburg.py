"""Write FREEZE.json for OLDENBURG-TRANSFER-1: sha256 of this packet's files and of every input the scorer, the
legitimacy check and the inference plan rely on; the deployable checkpoint; the bootstrap streams; tests and gates.

  python -B work/oldenburg_transfer_2026-10-07/freeze_oldenburg.py TESTS_LOG ARCHIVE_GATE_JSON
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ("PACKET.md", "INFERENCE_PLAN.md", "legit.py", "LEGITIMACY.json", "score_oldenburg.py", "cu_projection.py",
         "CU_PROJECTION.json", "STORAGE_CHECK.json", "tests/test_gate_reading.py", "TESTS.log", "freeze_oldenburg.py")
INPUTS = (
    "outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json",
    "outputs/adapt_nds_2026-10-05/confirm/run/INFER_MANIFEST_PRIVATE.json",
    "outputs/adapt_nds_2026-10-05/confirm/run/STAGING_PLAN_PRIVATE.json",
    "outputs/adapt_nds_2026-10-05/confirm/run/INSIDE_MASKS_PRIVATE.npz",
    "outputs/adapt_nds_2026-10-05/register/denkmale_Oldenburg_20261005T010102Z.geojson",
    "outputs/adapt_nds_2026-10-05/register/supplement_Oldenburg_20261005T010102Z.json",
    "outputs/adapt_nds_2026-10-05/score/confirm/individual/PROPOSALS_ADAF_barrow_PRIVATE.json",
    "outputs/adapt_nds_2026-10-05/chain_runs/run_s1_20261005T043612Z/v1/results/confirm/PROPOSALS_ADAF_barrow_PRIVATE.json",
    "outputs/adapt_nds_2026-10-05/fit/FIT_TABLE_PRIVATE.json",
    "outputs/adapt_nds_2026-10-05/fit/FIT_STAGING_PINNED_PRIVATE.json",
    "outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/SESSION_VERIFIED.json",
    "work/stage1_d1_v0_2026-10-07/score_v0.py",
    "work/stage1_d1_v0_2026-10-07/RESULT.json",
    "work/holdout_region_2026-10/step2/region_score.py",
    "work/next_experiment_2026-09-23/r3_curation.py",
    "work/gpt6_pro_collab_2026-10-07/v0_uncertainty/engine.py",
    "work/gpt6_pro_collab_2026-10-07/v0_uncertainty/boot_v0.py",
    "work/gpt6_pro_collab_2026-10-07/v0_uncertainty/v0data.py",
    "work/gpt6_pro_collab_2026-10-07/detector_week1/bench_v0.py",
    "work/gpt6_pro_collab_2026-10-07/detector_week1/BENCH_V0.json",
    "work/evaluator_v0_1_2026-10-07/eval_v01.py",
    "work/evaluator_v0_1_2026-10-07/FREEZE.json",
    "work/d1_verden_descriptive_2026-10-07/desc_verden.py",
    "work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py",
    "work/adapt_nds_2026-10-05/nds_reader.py",
    "work/fresh_region_2026-10-03/step2/fresh_gates.py",
    "work/colab_assessment_2026-09-29/env/freeze.txt",
    "work/campaign_2026-09-22/adaf_weights/barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    log, gate = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    sv = json.loads((ROOT / INPUTS[10]).read_text(encoding="utf-8"))
    d1 = {f"D1F1_BARROW_L_{s}": sv["trained"][f"D1F1_BARROW_L_{s}"] for s in range(2026092805, 2026092815)}
    rec = {
        "schema": "oldenburg_transfer_freeze_v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": "a prespecified transfer check on a previously inspected, training-disjoint district that was not "
                "used to select this recipe",
        "files_sha256": {f: sha(HERE / f) for f in FILES},
        "inputs_sha256": {f: sha(ROOT / f) for f in INPUTS},
        "d1_checkpoints_sha256": d1,
        "deployable_checkpoint": {"name": "D1F1_BARROW_L_2026092805", "sha256": d1["D1F1_BARROW_L_2026092805"],
                                  "rule": "earliest run in immutable run-ID order; frozen before any Oldenburg D1 "
                                          "score"},
        "comparator": {"checkpoint_sha256": "51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216",
                       "proposals_sha256": "21421a9c7e99cf337c64467349878e7062571768147a459164b6ddd92b3afef6"},
        "gate": {"b": 0.5, "decoder": "RAW", "mean_diff_min": 0.030, "positive_min": 8, "float_guard": 1e-12},
        "clear_transfer_improvement": "gate PASS and 2 km joint 95 % lower bound (RAW, b = 0.5) > 0",
        "bootstrap": {"engine": "v0_uncertainty engine.py + boot_v0.py", "reps": 5000, "master_seed": 20261007,
                      "geography_stream": "geo_oldenburg (spawn_key (0,))",
                      "training_stream": "train_d1 = SeedSequence(20261007).spawn(5)[4], shared with Verden",
                      "grids": ["2km", "4km", "2km_offset1000"], "primary_grid": "2km"},
        "legitimacy": json.loads((HERE / "LEGITIMACY.json").read_text(encoding="utf-8"))["legitimacy"],
        "tests": log.read_text(encoding="utf-8").strip().splitlines()[-1],
        "archive_gate": gate,
    }
    if len(d1) != 10:
        raise SystemExit(f"expected ten D1 BARROW_L confirmation checkpoints, found {len(d1)}")
    (HERE / "FREEZE.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rec[k] for k in ("utc", "deployable_checkpoint", "legitimacy", "tests")}, indent=1))


if __name__ == "__main__":
    main()
