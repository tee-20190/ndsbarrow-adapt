"""ADAPT-MATCH-0 freeze and the run-time binding checks the driver calls (built, not run: the freeze follows the GPU
smoke and ADAPT-NDS-1's scoring).

  python -B freeze_match.py freeze                                  # create-only: freeze/FREEZE.json, FREEZE_LOCK.json,
                                                                    # PINS_FREEZE.json
  python -B freeze_match.py check-run FREEZE INDEX PREV_TOP OUT.json # before allocation (driver)
  python -B freeze_match.py session-pins FREEZE INDEX OUT.json      # the VM pins file of one session
  python -B freeze_match.py check-lists FREEZE LISTS_DIR PINS_REL   # bundle list = the session pins

The freeze refuses unless: ADAPT-NDS-1 has been scored (its RESULT.json under freeze ba07548b; packet "Starts after");
A's Verden proposals exist for all three seeds and match their ADAPT-NDS-1 SESSION_VERIFIED.json hashes (score_match
refuses while any is missing); FRESH-1's zero-shot proposals match their hashes; the Lower Saxony FIT and Verden files
equal ADAPT-NDS-1's frozen pins (all eligible data by hash, nothing new fetched); both ADAF tars equal their registry
hashes; the three plan-checks, the start check and the dry run passed; the CU projection is within 60 CU; and the
GPU smoke receipt is PASS on byte-identical code. It pins every file the sessions ship and every local file the result
depends on (scorer and its frozen modules, DEV manifests, references, plan-checks, records).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import match_common as M  # noqa: E402
import bundle_match as BM  # noqa: E402
import score_match as SM  # noqa: E402

N = M.N
FREEZE, LOCK, PINS = M.FREEZE_FILE, M.FREEZE_DIR / "FREEZE_LOCK.json", M.FREEZE_DIR / "PINS_FREEZE.json"
W = M.HERE.relative_to(M.ROOT).as_posix()
NDS_W = M.NDS.relative_to(M.ROOT).as_posix()
SCORER = {"score_match": f"{W}/score_match.py", "score_adapt": f"{NDS_W}/score_adapt.py",
          "match_common": f"{W}/match_common.py",
          "region_score": "work/holdout_region_2026-10/step2/region_score.py",
          "hier_uncertainty": "work/holdout_region_2026-10/step2/hier_uncertainty.py",
          "fuse_eval": "work/fusion_2026-10-03/fuse_eval.py",
          "dev_infer_manifest": M.DEV_DATA["dev_infer_manifest"], **M.DEV_LABELS, "fresh1_result": M.FRESH1_RESULT}
DATA = {**M.LS_DATA, **{k: v for k, v in M.DEV_DATA.items() if k != "dev_infer_manifest"}}
RECORDS = [f"{W}/{f}" for f in ("PACKET.md", "CU_PROJECTION.json", "SMOKE_RECEIPT.json", "START_CHECK.json",
                                "DRYRUN_SUMMARY.json", "tests/conftest.py", "tests/toy_match.py",
                                "tests/test_schedule.py", "tests/test_train.py", "tests/test_score.py",
                                "tests/test_chain.py")]
PLAN_CHECK = "outputs/adapt_match_2026-10-05/plans/PLAN_CHECK_{seed}.json"


def sha_map(rels) -> dict[str, str]:
    return {r: N.sha256_file(M.ROOT / r) for r in sorted(set(rels))}


def refuse(msgs: list[str]) -> None:
    raise SystemExit("freeze refused:\n  " + "\n  ".join(msgs))


# ---------------------------------------------------------------- preconditions
def nds_scored(v: list) -> dict:
    if not M.NDS_FREEZE.is_file() or N.sha256_file(M.NDS_FREEZE) != M.NDS_FREEZE_SHA256:
        v.append("ADAPT-NDS-1 FREEZE.json is not ba07548b")
    r = N.read_json(M.NDS_RESULT) if M.NDS_RESULT.is_file() else {}
    if r.get("freeze_sha256") != M.NDS_FREEZE_SHA256:
        v.append("ADAPT-NDS-1 has not been scored under freeze ba07548b (packet 'Starts after')")
        return {}
    return {"result": N.rel(M.NDS_RESULT), "sha256": N.sha256_file(M.NDS_RESULT)}


def by_nds_hash(v: list) -> dict:
    """Every FIT and Verden input equals ADAPT-NDS-1's frozen pin (eligible data read by hash, nothing new fetched)."""
    pins = N.read_json(M.NDS_PINS)["files"]
    out = {}
    for k, rel in {**M.LS_DATA, **M.DEV_DATA, **M.DEV_LABELS}.items():
        h = N.sha256_file(M.ROOT / rel) if (M.ROOT / rel).is_file() else None
        if h is None or pins.get(rel) != h:
            v.append(f"{k}: {rel} differs from ADAPT-NDS-1's frozen pin")
        out[k] = {"path": rel, "sha256": h}
    return out


def starts(v: list) -> dict:
    reg = {c["id"]: c for c in N.read_json(M.ROOT / "DETECTOR_REGISTRY.json")["checkpoints"]}
    out = {}
    for arm, s in M.STARTS.items():
        p = M.start_path(arm)
        ok = p.is_file() and p.stat().st_size == s["bytes"] and N.sha256_file(p) == s["sha256"]
        if not ok or reg.get(s["registry_id"], {}).get("sha256") != s["sha256"]:
            v.append(f"{arm}: start tar differs from DETECTOR_REGISTRY.json {s['registry_id']}")
        out[arm] = dict(s)
    return out


def plan_checks(ls_sha: str | None, v: list) -> dict:
    out = {}
    for seed in M.SEEDS:
        p = M.ROOT / PLAN_CHECK.format(seed=seed)
        r = N.read_json(p) if p.is_file() else {}
        if r.get("status") != "PASS" or r.get("ls_table_sha256") != ls_sha:
            v.append(f"plan-check of seed {seed} missing, failed or made on another FIT table")
            continue
        out[str(seed)] = {"path": N.rel(p), "sha256": N.sha256_file(p), "plan_digest": r["plan_digest"],
                          "epoch_digests": {str(e["epoch"]): e["draw_digest"] for e in r["plan_summary"]["epochs"]},
                          "exposure": r["exposure"]}
    return out


def record_pass(name: str, v: list) -> dict:
    p = M.HERE / name
    r = N.read_json(p) if p.is_file() else {}
    if r.get("status") != "PASS":
        v.append(f"{name} missing or not PASS")
        return {}
    return {"path": N.rel(p), "sha256": N.sha256_file(p), "utc": r.get("utc")}


def smoke_ok(code_pins: dict, v: list) -> dict:
    r = N.read_json(M.SMOKE_RECEIPT) if M.SMOKE_RECEIPT.is_file() else {}
    if r.get("status") != "PASS":
        v.append("no PASS SMOKE_RECEIPT.json: the freeze follows the GPU smoke on the production artifact route")
        return {}
    diff = [k for k, h in r.get("code_sha256", {}).items() if code_pins.get(k) not in (None, h)]
    if diff or not r.get("code_sha256"):
        v.append(f"code changed after the smoke: {diff[:5]}")
    return {"receipt": N.rel(M.SMOKE_RECEIPT), "sha256": N.sha256_file(M.SMOKE_RECEIPT), "utc": r.get("utc")}


# ---------------------------------------------------------------- the freeze
def shipped() -> list[str]:
    out = []
    for i in M.SESSIONS:
        out += BM.code_list(i, [])
    return sorted(set(out))


def local_bound(a_refs: dict, zs: dict, pc: dict) -> list[str]:
    out = list(RECORDS) + list(SCORER.values()) + [x["path"] for x in pc.values()]
    out += [M.NDS_FREEZE.relative_to(M.ROOT).as_posix(), N.rel(M.NDS_RESULT), f"{M.FRESH1_S1}/SESSION_VERIFIED.json"]
    out += [r["path"] for r in zs.values()]
    out += [r["path"] for r in a_refs.values()] + [r["session_verified"] for r in a_refs.values()]
    return out


def preconditions() -> tuple[dict, list[str]]:
    v = []
    pre = {"nds": nds_scored(v), "data": by_nds_hash(v), "starts": starts(v)}
    a_refs, missing = SM.a_references()
    v += missing + [f"{M.a_name(s)}: A's Verden proposals not available" for s in M.SEEDS
                    if M.a_name(s) not in a_refs and not missing]
    zs, bad = SM.zero_shot_references()
    v += bad
    pre |= {"a_references": a_refs, "zero_shot": zs,
            "plan_checks": plan_checks(pre["data"].get("ls_fit", {}).get("sha256"), v),
            "start_check": record_pass("START_CHECK.json", v), "dryrun": record_pass("DRYRUN_SUMMARY.json", v)}
    proj = N.read_json(M.CU_PROJECTION) if M.CU_PROJECTION.is_file() else {}
    if proj.get("projection", {}).get("status") != "PASS":
        v.append("CU projection missing or above the 60 CU ceiling")
    pre["cu_projection"] = proj.get("projection", {})
    smoke = N.read_json(M.SMOKE_RECEIPT) if M.SMOKE_RECEIPT.is_file() else {}
    if smoke.get("status") != "PASS":                      # code hashes are compared in make(), after the pins
        v.append("no PASS SMOKE_RECEIPT.json: the freeze follows the GPU smoke on the production artifact route")
    return pre, v


def make() -> dict:
    if FREEZE.exists() or LOCK.exists():
        raise SystemExit("the freeze exists; it is made once")
    pre, v = preconditions()
    if v:
        refuse(v)
    ship = shipped()
    pins = sha_map(ship + local_bound(pre["a_references"], pre["zero_shot"], pre["plan_checks"]))
    smoke = smoke_ok(pins, v)
    if v:
        refuse(v)
    rec = {"schema": "adapt_match_freeze_v1", "utc": N.utc(), "packet_sha256": N.sha256_file(M.PACKET),
           "pins": {k: {"path": p, "sha256": pins[p]} for k, p in (SCORER | DATA).items()},
           "starts": pre["starts"], "a_references": pre["a_references"], "zero_shot": pre["zero_shot"],
           "adapt_nds": pre["nds"] | {"freeze_sha256": M.NDS_FREEZE_SHA256}, "plan_checks": pre["plan_checks"],
           "start_check": pre["start_check"], "dryrun": pre["dryrun"], "cu_projection": pre["cu_projection"],
           "shipped_files": len(ship), "pinned_files": len(pins), "smoke": smoke,
           "rules": {"main": "A_AHEAD if mean >= +0.03 and lo95 > 0; A_BEHIND if hi95 < 0; else NOT_DISTINGUISHED",
                     "bootstrap": "cluster x seed, 2,000 draws, seed 20261005; better baseline re-selected per draw"},
           "attribution": M.ATTRIBUTION}
    M.FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    N.write_json(PINS, {"kind": "adapt_match_freeze", "utc": rec["utc"], "files": pins})
    rec["pins_freeze_sha256"] = N.sha256_file(PINS)
    N.create_only(FREEZE, rec)
    sha = N.sha256_file(FREEZE)
    N.create_only(LOCK, {"freeze_sha256": sha, "pins_freeze_sha256": rec["pins_freeze_sha256"], "utc": rec["utc"]})
    return {"freeze_sha256": sha, "pinned_files": len(pins)}


# ---------------------------------------------------------------- run-time binding (driver, before allocation)
def load(freeze: Path) -> dict:
    lock = N.read_json(Path(freeze).with_name("FREEZE_LOCK.json"))
    sha = N.sha256_file(freeze)
    if sha != lock["freeze_sha256"]:
        raise SystemExit("FREEZE.json differs from FREEZE_LOCK.json")
    f = N.read_json(freeze)
    if N.sha256_file(Path(freeze).with_name("PINS_FREEZE.json")) != f["pins_freeze_sha256"]:
        raise SystemExit("PINS_FREEZE.json differs from the freeze")
    f["_sha256"] = sha
    return f


def check_run(freeze: Path, index: int, prev_top: str, out: Path) -> dict:
    f = load(freeze)
    M.session_arm(index)
    pins = N.read_json(Path(freeze).with_name("PINS_FREEZE.json"))["files"]
    bad = [r for r, h in pins.items() if not (M.ROOT / r).is_file() or N.sha256_file(M.ROOT / r) != h]
    if bad:
        raise SystemExit(f"{len(bad)} pinned files missing or changed, e.g. {bad[:3]}")
    prev = {}
    if index > 1:
        p = Path(prev_top) / "SESSION_VERIFIED.json" if prev_top else None
        prev = N.read_json(p) if p and p.is_file() else {}
        if prev.get("status") != "PASS" or prev.get("index") != index - 1 or prev.get("freeze_sha256") != f["_sha256"]:
            raise SystemExit(f"session {index} needs session {index - 1} verified under this freeze")
    rec = {"status": "PASS", "freeze_sha256": f["_sha256"], "index": index, "arm": M.session_arm(index),
           "pinned_files": len(pins), "previous": prev.get("top"), "utc": N.utc()}
    N.write_json(out, rec)
    return rec


def session_pins(freeze: Path, index: int, out: Path) -> dict:
    f = load(freeze)
    pins = N.read_json(Path(freeze).with_name("PINS_FREEZE.json"))["files"]
    files, bad = {}, []
    for r in sorted(set(BM.code_list(index, []))):
        h = N.sha256_file(M.ROOT / r) if (M.ROOT / r).is_file() else None
        if h is None or pins.get(r) != h:
            bad.append(r)
        files[r] = h
    if bad:
        raise SystemExit(f"session pins refused: {len(bad)} listed files are not bound by the freeze, e.g. {bad[:3]}")
    N.write_json(out, {"kind": "adapt_match_run", "utc": N.utc(), "freeze_sha256": f["_sha256"], "index": index,
                       "files": files})
    return {"kind": "adapt_match_run", "files": len(files)}


def check_lists(freeze: Path, lists_dir: Path, pins_rel: str) -> dict:
    load(freeze)
    pins = N.read_json(M.ROOT / pins_rel)["files"]
    listed = set((lists_dir / "code.txt").read_text(encoding="utf-8").split()) - {pins_rel}
    if listed != set(pins):
        raise SystemExit(f"bundle list differs from the session pins: +{sorted(listed - set(pins))[:3]} "
                         f"-{sorted(set(pins) - listed)[:3]}")
    return {"status": "PASS", "files": len(listed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("freeze", "check-run", "session-pins", "check-lists"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    x = a.args
    if a.cmd == "freeze":
        print(json.dumps(make()))
    elif a.cmd == "check-run":
        print(json.dumps(check_run(Path(x[0]), int(x[1]), x[2] if len(x) > 3 else "", Path(x[-1]))))
    elif a.cmd == "session-pins":
        print(json.dumps(session_pins(Path(x[0]), int(x[1]), Path(x[2]))))
    else:
        print(json.dumps(check_lists(Path(x[0]), Path(x[1]), x[2])))


if __name__ == "__main__":
    main()
