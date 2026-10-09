"""ADAPT-MATCH-0 bundle list (repo-relative files streamed to the Colab VM; derived from ADAPT-NDS-1 bundle_adapt.py).

  python -B bundle_match.py lists INDEX OUT_DIR [--extra REL ...]     # -> OUT_DIR/code.txt (one bundle per session)

code: every ADAPT-MATCH-0 file (so the pins checked on the VM cover the local tools too); the ADAPT-NDS-1 modules the
      chain imports or runs read-only (nds_common, nds_schedule, nds_train, nds_reader, smoke_local, adapt_gates,
      sessions_adapt, package_results.sh, colab_local.sh); JOINT-1's trainer modules and freeze (imported by nds_train);
      the HR3 chain (inference) and HR1 chain (masks, gates, parity, trainer helpers) with HR1's frozen module list
      (bundle_lists.FROZEN), CONT e41 and the pinned ADAF AO weights, the EQ-1 parity block; the step-0 environment
      fixture (JOINT-1's English plan of seed 2026092800 and the one R2 window + mask its first update reads: the
      declared step-0 reference, not training data); FRESH-1's DEV reader (fresh_common, lgln_reader, fresh_gates)
      with its label-free inference/enclosure manifests, staging plan, district masks, tile index and read ledger;
      ADAPT-NDS-1's Lower Saxony FIT table (labels for training), pinned staging plan and union masks; the session's
      start weights (session 1: ADAF AO, already in HR1's list; session 2: ADAF barrow). --extra: the session pins.
No English bundle: no English, Welsh or Irish window trains or is evaluated (PACKET.md "Eligible data").
Refused: TEST, VALIDATION, reserved, French, EBA and Amazon data; label-bearing or reference files that must never reach
a VM (NEVER: ADAPT-NDS-1 CONFIRM, register, score and chain_runs; FRESH-1 group/individual manifests, labels and
enclosure records; this packet's score and plan checks).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import match_common as M  # noqa: E402

N = M.N
sys.path.insert(0, str(N.HR1_CHAIN))
import bundle_lists as HB  # noqa: E402  HR1 (frozen): FROZEN, WEIGHTS, EQ1, plan_files, r2_windows
sys.path.insert(0, str(N.J1_CHAIN))
import joint_common as J  # noqa: E402  JOINT-1 (frozen): STEP0_SEED

REL = M.HERE.relative_to(M.ROOT).as_posix()
MATCH_CODE = ["match_common.py", "match_schedule.py", "match_train.py", "score_match.py", "cu_projection_match.py",
              "dryrun_match.py", "start_check.py", "smoke_match.py",
              "chain/match_gates.py", "chain/sessions_match.py", "chain/bundle_match.py", "chain/freeze_match.py",
              "chain/chain_lib_match.sh", "chain/chain_match_run.sh", "chain/chain_match_smoke.sh",
              "chain/vm_job_match.sh", "chain/driver_match.sh"]
MATCH_RECORDS = ["PACKET.md", "CU_PROJECTION.json"]
NDS_REL = M.NDS.relative_to(M.ROOT).as_posix()
NDS_FILES = [f"{NDS_REL}/{f}" for f in ("nds_common.py", "nds_schedule.py", "nds_train.py", "nds_reader.py",
                                         "smoke_local.py", "chain/adapt_gates.py", "chain/sessions_adapt.py",
                                         "chain/package_results.sh", "chain/colab_local.sh")]
J1_REL = "work/joint_campaign_2026-10-03"
JOINT1_FILES = [f"{J1_REL}/chain/joint_common.py", f"{J1_REL}/chain/joint_train.py", f"{J1_REL}/chain/joint_gates.py",
                f"{J1_REL}/freeze/FREEZE.json", f"{J1_REL}/freeze/FREEZE_LOCK.json"]
HR3_FILES = [f"work/hr3_ireland_2026-10-02/step2/chain/{f}" for f in
             ("irl_common.py", "irl_manifest.py", "irl_masks.py", "stage_irl.py", "infer_irl.py", "irl_gates.py")]
HR3_VIS = ["work/hr3_ireland_2026-10-02/visibility/vis_irl.py", "work/hr3_ireland_2026-10-02/visibility/vis_common.py"]
HR1_CHAIN_FILES = [f"work/holdout_region_2026-10/chain/{f}" for f in
                   ("common.py", "masks.py", "welsh_train.py", "infer_dev.py", "gates.py", "parity.py", "cat_parts.py",
                    "bundle_lists.py", "bundle_stream.py", "stage_welsh.py", "shards.sh")]
FRESH_CODE = [f"work/fresh_region_2026-10-03/step2/{f}" for f in ("fresh_common.py", "lgln_reader.py", "fresh_gates.py")]
NDS_PRIV_REL = M.NDS_PRIV.relative_to(M.ROOT).as_posix()
PRIV_REL = M.PRIV.relative_to(M.ROOT).as_posix()
NEVER_PREFIXES = (f"{NDS_PRIV_REL}/register/", f"{NDS_PRIV_REL}/confirm/", f"{NDS_PRIV_REL}/score/",
                  f"{NDS_PRIV_REL}/chain_runs/", f"{NDS_PRIV_REL}/weights/", "outputs/fresh_region_2026-10-03/labels/",
                  "outputs/fresh_region_2026-10-03/step2/chain_runs/", f"{PRIV_REL}/score/", f"{PRIV_REL}/plans/",
                  f"{PRIV_REL}/chain_runs/")
NEVER_FILES = tuple(M.DEV_LABELS.values()) + (
    f"{M.FRESH_S0}/NODATA_MEMBERS_PRIVATE.json", f"{M.FRESH_S0}/BUILD_RECORD_PRIVATE.json",
    f"{M.FRESH_S0}/VALIDITY_MASKS_PRIVATE.npz", f"{M.FRESH_OUT}/ENCLOSURE_RECORDS_PRIVATE.json",
    f"{NDS_REL}/CONFIRM_BUILD_LOCK.json", f"{NDS_REL}/CONFIRM_SELECTION.json")
REFUSED_DATA = ("test", "validation", "reserved", "bertranges", "french", "france", "eba_", "amazon_sensor",
                "fr_disc", "amazon")
ALLOWED_CODE = ("work/amazon_adapt_2026-09-28/adapt_train.py",)


def refused_data(rels: list[str]) -> list[str]:
    return [r for r in rels if r not in ALLOWED_CODE and any(t in r.lower() for t in REFUSED_DATA)]


def never_violations(rels: list[str]) -> list[str]:
    bad = [r for r in rels if r in NEVER_FILES]
    bad += [r for r in rels if r.startswith(NEVER_PREFIXES)]
    return sorted(set(bad))


def step0_fixture() -> list[str]:
    """The R2 window and mask that the step-0 gate's single update reads (plan 2026092800, epoch 0, step 0)."""
    plan = N.read_json(M.ROOT / HB.plan_files((J.STEP0_SEED,))[0])
    w = HB.r2_windows()[int(plan["real"][0][0][0])]
    return [f"outputs/campaign_2026-09-22/train_raw/{w['id']}.tif", w["mask_path"]]


def weights(index: int) -> list[str]:
    """The session's start tar (ADAF AO is also in HR1's WEIGHTS list: parity and the step-0 module set use it)."""
    return [M.STARTS[M.session_arm(index)]["path"]]


def code_list(index: int, extra: list[str], pre_freeze: bool = False) -> list[str]:
    M.session_arm(index)
    out = [f"{REL}/{f}" for f in MATCH_CODE + MATCH_RECORDS] + NDS_FILES + JOINT1_FILES + HR3_FILES + HR3_VIS
    out += HR1_CHAIN_FILES + list(HB.FROZEN) + HB.WEIGHTS + HB.EQ1 + FRESH_CODE
    out += HB.plan_files((J.STEP0_SEED,)) + step0_fixture()
    out += list(M.DEV_DATA.values()) + list(M.LS_DATA.values()) + weights(index) + list(extra)
    out = list(dict.fromkeys(out))
    if pre_freeze:
        out = [r for r in out if (M.ROOT / r).is_file() or r.startswith(f"{REL}/")]
    leaked = never_violations([r for r in out if r not in extra])     # the session pins file lives in its TOP
    if leaked:
        raise SystemExit(f"bundle refused: label-bearing or reference files would travel to the VM: {leaked[:5]}")
    bad = refused_data(out)
    if bad:
        raise SystemExit(f"bundle refused: refused data (PACKET Eligible data): {bad[:3]}")
    return out


def lists(index: int, out_dir: Path, extra: list[str]) -> dict:
    import bundle_stream as bs
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = code_list(index, extra)
    bs.check_list(rels, tuple(extra))
    (out_dir / "code.txt").write_bytes(("\n".join(rels) + "\n").encode())
    return {"code": {"files": len(rels), "bytes": sum((M.ROOT / r).stat().st_size for r in rels)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("lists",))
    ap.add_argument("args", nargs=2)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    print(json.dumps(lists(int(a.args[0]), Path(a.args[1]), a.extra)))
