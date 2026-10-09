"""D1-COLLAPSE-1 freeze and the run-time binding checks the driver calls (a reduced copy of ADAPT-MATCH-0's
freeze_match.py, frozen 351b5961).

  python -B freeze_d1.py plan-checks                               # CPU: d1_train plan-check of all 13 seeds (private)
  python -B freeze_d1.py freeze                                    # create-only: freeze/FREEZE.json, FREEZE_LOCK.json,
                                                                   # PINS_FREEZE.json
  python -B freeze_d1.py check-run FREEZE INDEX PREV_TOP OUT.json  # before allocation (driver)
  python -B freeze_d1.py session-pins FREEZE INDEX OUT.json        # the VM pins file of one session
  python -B freeze_d1.py check-lists FREEZE LISTS_DIR PINS_REL     # bundle list = the session pins

The freeze refuses unless: ADAPT-MATCH-0's freeze is intact (FREEZE_LOCK) and every FIT / Verden input equals its pin;
both start tars equal DETECTOR_REGISTRY.json; all 13 plan-checks PASS on the pinned FIT table (dev seeds equal
ADAPT-MATCH-0's plan digests); TESTS.json is PASS on the current code bytes; the CU projection exists. It pins every
file any session ships and every local file the result depends on (scorer modules, labels, packet, tests, receipts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import d1_common as D  # noqa: E402
import bundle_d1 as BD  # noqa: E402

N = D.N
import freeze_match as FM  # noqa: E402  ADAPT-MATCH-0 (frozen): SCORER pins

FREEZE, LOCK, PINS = D.FREEZE_FILE, D.FREEZE_DIR / "FREEZE_LOCK.json", D.FREEZE_DIR / "PINS_FREEZE.json"
W = D.HERE.relative_to(D.ROOT).as_posix()
MATCH_FREEZE = D.MATCH / "freeze/FREEZE.json"
PLAN_CHECK = "outputs/d1_collapse_2026-10-06/plans/PLAN_CHECK_{seed}.json"
RECORDS = [f"{W}/{x}" for x in ("PACKET.md", "DIAGNOSIS.md", "CU_PROJECTION.json", "TESTS.json", "OLD_RUNS_CHECK.json",
                                "score_d1.py", "tests/conftest.py", "tests/test_d1_train.py", "tests/test_chain.py")]
SCORER = {"score_d1": f"{W}/score_d1.py", **FM.SCORER}


def refuse(msgs: list[str]) -> None:
    raise SystemExit("freeze refused:\n  " + "\n  ".join(msgs))


def match_intact(v: list) -> dict:
    lock = N.read_json(MATCH_FREEZE.with_name("FREEZE_LOCK.json"))
    sha = N.sha256_file(MATCH_FREEZE)
    if sha != lock["freeze_sha256"]:
        v.append("ADAPT-MATCH-0 FREEZE.json differs from its lock")
    pins = N.read_json(MATCH_FREEZE)["pins"]
    out = {}
    for k, rel in {**D.LS_DATA, **D.DEV_DATA, **D.DEV_LABELS}.items():
        h = N.sha256_file(D.ROOT / rel) if (D.ROOT / rel).is_file() else None
        if pins.get(k, {}).get("sha256") not in (None, h) or h is None:
            v.append(f"{k}: {rel} differs from ADAPT-MATCH-0's pin")
        out[k] = {"path": rel, "sha256": h}
    return {"match_freeze_sha256": sha, "pins": out}


def starts(v: list) -> dict:
    reg = {c["id"]: c for c in N.read_json(D.ROOT / "DETECTOR_REGISTRY.json")["checkpoints"]}
    for arm, s in D.STARTS.items():
        p = D.ROOT / s["path"]
        if not (p.is_file() and p.stat().st_size == s["bytes"] and N.sha256_file(p) == s["sha256"]
                and reg.get(s["registry_id"], {}).get("sha256") == s["sha256"]):
            v.append(f"{arm}: start tar differs from DETECTOR_REGISTRY.json {s['registry_id']}")
    return {a: dict(s) for a, s in D.STARTS.items()}


def run_plan_checks() -> dict:
    import d1_train as T
    out = {}
    for seed in D.SEEDS:
        p = D.ROOT / PLAN_CHECK.format(seed=seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        out[seed] = T.plan_check(seed, {"ls_table": D.ROOT / D.LS_DATA["ls_fit"]}, p)["status"]
    return out


def plan_checks(ls_sha: str | None, v: list) -> dict:
    old = N.read_json(MATCH_FREEZE)["plan_checks"]
    out = {}
    for seed in D.SEEDS:
        p = D.ROOT / PLAN_CHECK.format(seed=seed)
        r = N.read_json(p) if p.is_file() else {}
        if r.get("status") != "PASS" or r.get("ls_table_sha256") != ls_sha:
            v.append(f"plan-check of seed {seed} missing, failed or made on another FIT table")
            continue
        if str(seed) in old and old[str(seed)]["plan_digest"] != r["plan_digest"]:
            v.append(f"plan of development seed {seed} differs from ADAPT-MATCH-0's")
        out[str(seed)] = {"path": N.rel(p), "sha256": N.sha256_file(p), "plan_digest": r["plan_digest"],
                          "epoch_digests": {str(e["epoch"]): e["draw_digest"] for e in r["plan_summary"]["epochs"]}}
    return out


def tests_ok(pins: dict, v: list) -> dict:
    p = D.HERE / "TESTS.json"
    r = N.read_json(p) if p.is_file() else {}
    if r.get("status") != "PASS":
        v.append("TESTS.json missing or not PASS")
        return {}
    diff = [k for k, h in r.get("code_sha256", {}).items() if pins.get(k) not in (None, h)]
    if diff or not r.get("code_sha256"):
        v.append(f"code changed after the tests: {diff[:5]}")
    return {"path": N.rel(p), "utc": r.get("utc"), "passed": r.get("passed")}


def shipped() -> list[str]:
    out = []
    for i in D.SESSIONS:
        out += BD.code_list(i, [])
    return sorted(set(out))


def make() -> dict:
    if FREEZE.exists() or LOCK.exists():
        raise SystemExit("the freeze exists; it is made once")
    v = []
    mi = match_intact(v)
    st = starts(v)
    pc = plan_checks(mi["pins"]["ls_fit"]["sha256"], v)
    proj = N.read_json(D.CU_PROJECTION) if D.CU_PROJECTION.is_file() else {}
    if not proj.get("projection", {}).get("sessions"):
        v.append("CU_PROJECTION.json missing")
    if v:
        refuse(v)
    ship = shipped()
    local = RECORDS + list(SCORER.values()) + [x["path"] for x in pc.values()] + [N.rel(MATCH_FREEZE)]
    pins = {r: N.sha256_file(D.ROOT / r) for r in sorted(set(ship + local))}
    tst = tests_ok(pins, v)
    if v:
        refuse(v)
    rec = {"schema": "d1_freeze_v1", "utc": N.utc(), "packet_sha256": N.sha256_file(D.PACKET),
           "pins": {**mi["pins"], **{k: {"path": p, "sha256": pins[p]} for k, p in SCORER.items()}},
           "adapt_match": {"freeze_sha256": mi["match_freeze_sha256"]}, "starts": st, "plan_checks": pc,
           "tests": tst, "cu_projection": proj["projection"], "shipped_files": len(ship), "pinned_files": len(pins),
           "design": {"dev_seeds": list(D.DEV_SEEDS), "confirm_seeds": list(D.CONFIRM_SEEDS),
                      "reference": D.REFERENCE, "noise": D.NOISE, "collapse_fraction": D.COLLAPSE_FRACTION,
                      "detector": [D.DETECT_WINDOW, D.DETECT_DICE, D.DETECT_MASS], "ceilings": D.CU_CEILING},
           "attribution": D.ATTRIBUTION}
    D.FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    N.write_json(PINS, {"kind": "d1_freeze", "utc": rec["utc"], "files": pins})
    rec["pins_freeze_sha256"] = N.sha256_file(PINS)
    N.create_only(FREEZE, rec)
    sha = N.sha256_file(FREEZE)
    N.create_only(LOCK, {"freeze_sha256": sha, "pins_freeze_sha256": rec["pins_freeze_sha256"], "utc": rec["utc"]})
    return {"freeze_sha256": sha, "pinned_files": len(pins), "shipped_files": len(ship)}


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
    import sessions_d1 as S
    f = load(freeze)
    pins = N.read_json(Path(freeze).with_name("PINS_FREEZE.json"))["files"]
    bad = [r for r, h in pins.items() if not (D.ROOT / r).is_file() or N.sha256_file(D.ROOT / r) != h]
    if bad:
        raise SystemExit(f"{len(bad)} pinned files missing or changed, e.g. {bad[:3]}")
    for why in (S.previous_ok(f, index, Path(prev_top) if prev_top else None), S.selection_ok(index)[0]):
        if why:
            raise SystemExit(why)
    rec = {"status": "PASS", "freeze_sha256": f["_sha256"], "index": index, "pinned_files": len(pins),
           "selection_sha256": S.selection_ok(index)[1], "previous": prev_top or None, "utc": N.utc()}
    N.write_json(out, rec)
    return rec


def session_pins(freeze: Path, index: int, out: Path) -> dict:
    f = load(freeze)
    pins = N.read_json(Path(freeze).with_name("PINS_FREEZE.json"))["files"]
    files, bad = {}, []
    for r in sorted(set(BD.code_list(index, []))):
        h = N.sha256_file(D.ROOT / r) if (D.ROOT / r).is_file() else None
        if h is None or pins.get(r) != h:
            bad.append(r)
        files[r] = h
    if bad:
        raise SystemExit(f"session pins refused: {len(bad)} listed files are not bound by the freeze, e.g. {bad[:3]}")
    N.write_json(out, {"kind": "d1_run", "utc": N.utc(), "freeze_sha256": f["_sha256"], "index": index,
                       "files": files})
    return {"kind": "d1_run", "files": len(files)}


def check_lists(freeze: Path, lists_dir: Path, pins_rel: str) -> dict:
    load(freeze)
    pins = N.read_json(D.ROOT / pins_rel)["files"]
    listed = set((lists_dir / "code.txt").read_text(encoding="utf-8").split()) - {pins_rel}
    if listed != set(pins):
        raise SystemExit(f"bundle list differs from the session pins: +{sorted(listed - set(pins))[:3]} "
                         f"-{sorted(set(pins) - listed)[:3]}")
    return {"status": "PASS", "files": len(listed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan-checks", "freeze", "check-run", "session-pins", "check-lists"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    x = a.args
    if a.cmd == "plan-checks":
        print(json.dumps(run_plan_checks()))
    elif a.cmd == "freeze":
        print(json.dumps(make()))
    elif a.cmd == "check-run":
        print(json.dumps(check_run(Path(x[0]), int(x[1]), x[2] if len(x) > 3 else "", Path(x[-1]))))
    elif a.cmd == "session-pins":
        print(json.dumps(session_pins(Path(x[0]), int(x[1]), Path(x[2]))))
    else:
        print(json.dumps(check_lists(Path(x[0]), Path(x[1]), x[2])))


if __name__ == "__main__":
    main()
