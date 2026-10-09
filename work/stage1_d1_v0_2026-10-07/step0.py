"""STAGE1-D1-V0 Step 0 (FREEZE.md): legitimacy check. Did the ten D1 BARROW_L confirmation runs (seeds 05-14) train
on Verden terrain or labels? CPU, read-only. Writes aggregates only (no coordinates, no record ids) to STEP0.json.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/stage1_d1_v0_2026-10-07/step0.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shapely.geometry import box, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIT = ROOT / "outputs/adapt_nds_2026-10-05/fit"
FIT_TABLE = FIT / "FIT_TABLE_PRIVATE.json"
FIT_PINNED = FIT / "FIT_STAGING_PINNED_PRIVATE.json"
VER = ROOT / "outputs/fresh_region_2026-10-03"
IND = VER / "step0/INDIVIDUAL_MANIFEST_PRIVATE.json"
INF = VER / "step2/INFER_MANIFEST_PRIVATE.json"
S3 = ROOT / "outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z"
SEEDS = range(2026092805, 2026092815)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def squares(names: list[str]) -> set[tuple[int, int]]:
    """1 km DGM1 squares from file names dgm1_32_<E>_<N>_1_ni_<year>.tif."""
    return {(int(n.split("_")[2]), int(n.split("_")[3])) for n in names}


def run_bindings() -> dict:
    """Per confirmation run: training table hash, update count and Lower Saxony share, from the session receipts."""
    sv = rj(S3 / "SESSION_VERIFIED.json")
    out = {}
    for s in SEEDS:
        name = f"D1F1_BARROW_L_{s}"
        tr = rj(S3 / "v1/results/runs" / name / "TRAIN_RESULT.json")
        out[name] = {"ls_table_sha256": tr["tables_sha256"]["ls_table"], "updates": tr["updates"],
                     "lower_saxony_updates": tr["lower_saxony_updates"], "start_sha256": tr["start"]["sha256"],
                     "checkpoint_sha256": tr["final_sha256"], "session_trained": sv["trained"].get(name)}
    return {"session_status": sv["status"], "runs": out}


def main() -> None:
    t, pin, ind, inf = rj(FIT_TABLE), rj(FIT_PINNED), rj(IND), rj(INF)
    wins = t["positive_windows"] + t["background_windows"]
    fit_union = unary_union([box(*w["bounds"]) for w in wins])
    districts = sorted({w.get("district") for w in t["positive_windows"]})
    verden_poly = shape(next(r["geometry"] for r in pin["refused_regions"] if r["name"] == "Verden"))
    cores = unary_union([box(*b) for tl in ind["tiles"] for b in tl["core"]])
    reads = unary_union([box(*w["bounds"]) for tl in inf["tiles"] for w in tl["windows"]])
    parts = unary_union([shape(p["geometry"]) for p in ind["parts"]])
    fit_ids = {m for w in t["positive_windows"] for m in w["members"]}
    ver_ids = {p["record_id"] for p in ind["parts"]}
    fit_sq = squares([n for w in pin["windows"] for n in w["tiles"]])
    ls_log = [json.loads(x)["name"] for x in (S3 / "v1/results/LS_FETCH_LOG.jsonl").read_text().splitlines() if x]
    dev_log = [json.loads(x)["name"] for x in (S3 / "v1/results/DEV_FETCH_LOG.jsonl").read_text().splitlines() if x]
    rb = run_bindings()
    table_sha = sha(FIT_TABLE)
    rec = {
        "schema": "stage1_d1_v0_step0_v1",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (FIT_TABLE, FIT_PINNED, IND, INF)},
        "training": {
            "table_region": t["region"], "table_role": t["role"],
            "positive_windows": len(t["positive_windows"]), "background_windows": len(t["background_windows"]),
            "positive_window_districts": districts,
            "window_union_km2": fit_union.area / 1e6,
            "terrain_rule": t["rules"]["terrain"],
            "every_run_trained_on_this_table": all(r["ls_table_sha256"] == table_sha for r in rb["runs"].values()),
            "every_run_all_updates_lower_saxony_fit": all(r["updates"] == r["lower_saxony_updates"] == 5116
                                                          for r in rb["runs"].values()),
            "runs": rb,
        },
        "separation": {
            "min_m_fit_windows_to_verden_district": fit_union.distance(verden_poly),
            "fit_windows_intersecting_verden_district": sum(box(*w["bounds"]).intersects(verden_poly) for w in wins),
            "min_m_fit_windows_to_verden_scored_cores": fit_union.distance(cores),
            "min_m_fit_windows_to_verden_inference_read_windows": fit_union.distance(reads),
            "min_m_fit_windows_to_verden_entity_footprints": fit_union.distance(parts),
            "shared_register_record_ids_fit_vs_verden_entities": len(fit_ids & ver_ids),
            "shared_1km_squares_fit_pins_vs_verden_dev_fetch": len(fit_sq & squares(dev_log)),
            "shared_1km_squares_training_fetch_log_vs_verden_dev_fetch": len(squares(ls_log) & squares(dev_log)),
            "training_fetch_log_files": len(ls_log), "verden_dev_fetch_files": len(dev_log),
        },
        "verden_eval_set": {"tiles": len(ind["tiles"]), "entities": len(ind["parts"]),
                            "scored_km2": cores.area / 1e6},
    }
    sep = rec["separation"]
    clean = (rec["training"]["every_run_trained_on_this_table"] and rec["training"]["every_run_all_updates_lower_saxony_fit"]
             and sep["fit_windows_intersecting_verden_district"] == 0
             and sep["shared_register_record_ids_fit_vs_verden_entities"] == 0
             and sep["shared_1km_squares_training_fetch_log_vs_verden_dev_fetch"] == 0
             and sep["min_m_fit_windows_to_verden_inference_read_windows"] > 0)
    rec["eligible_subset"] = {
        "rule": "Verden tiles/entities outside every run's training windows and their context buffer. Training pixels "
                "are confined to the 423 m FIT windows (NoData outside the FIT union); the buffer is any distance up to "
                "the measured minimum separation, so no Verden tile or entity is excluded and no split is drawn.",
        "tiles": len(ind["tiles"]) if clean else 0, "entities": len(ind["parts"]) if clean else 0,
        "scored_km2": cores.area / 1e6 if clean else 0.0,
    }
    rec["step0"] = "PASS" if clean else "NOT_EVALUABLE"
    (HERE / "STEP0.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("step0", "separation", "eligible_subset")}, indent=1))


if __name__ == "__main__":
    main()
