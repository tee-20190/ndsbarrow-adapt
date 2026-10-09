"""STAGE2-D1-CONTROLS sequential sessions (PACKET "Compute and storage"; one VM, one heavy job at a time).

  python -B sessions_s2.py projection                                       # -> CU_PROJECTION.json (pre-freeze)
  python -B sessions_s2.py plan FREEZE INDEX [--previous TOP]               # shell assignments for the driver
  python -B sessions_s2.py verify TOP FREEZE INDEX WEIGHTS_DIR [--previous TOP]   # -> TOP/SESSION_VERIFIED.json

Projection: D1-COLLAPSE-1's measured confirmation sessions (run_s3 / run_s4 job.log and CU ledger) for everything the
control shares with D1 (setup, Verden DEV staging, gates, 5,116-update F1 training, the 10-model DEV pass, export),
ADAPT-NDS-1's measured sessions for what differs (English bundle upload, Welsh and Irish staging). The rate is the
higher of the two D1 sessions (8.854 CU/h). Wall cap 1.25 x chain; launch refused unless spent + 1.10 x projection
<= 60 CU (hard ceiling, s2_gates.py cu-room).
Session 0 (smoke) needs nothing; session 1 needs session 0's SESSION_VERIFIED.json (PASS, this freeze).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import s2_common as S  # noqa: E402

N = S.N
sys.path.insert(0, str(S.NDS / "chain"))
import sessions_adapt as SAD  # noqa: E402  ADAPT-NDS-1 (frozen): proposals_ok, consume_ok

REQUIRE_MARGIN, CAP_MARGIN, WALL_MARGIN = 1.10, 1.25, 1.25

# measured inputs (seconds unless stated); sources in EVIDENCE
MEASURED = {
    "cu_per_h": 8.854,               # D1 run_s4: 37.25 CU over 15,146 s (run_s3: 37.09 CU / 15,223 s = 8.771)
    "setup_code_s": 262,             # D1 run_s3: allocation 22:28:27Z -> chain start 22:32:49Z (code bundle 199 s)
    "english_upload_s": 390,         # ADAPT-NDS-1 english bundle 1,248,015,783 B: 287 / 390 / 293 s
    "stage_welsh_s": 1000,           # ADAPT-NDS-1 fetch+render+masks+verify: ~642 / ~993 / ~641 s
    "stage_irish_s": 100,            # ADAPT-NDS-1 stage+masks+verify: 57-77 s
    "stage_dev_s": 220,              # D1 run_s3 stage_dev + verify: 22:43:07Z -> 22:46:27Z (200 s)
    "gates_s": 200,                  # D1 run_s3 void, step0 x2, parity, plan checks: 22:46:27Z -> 22:48:42Z (135 s)
    "train_run_s": 345,              # D1 run_s3/s4: 338.3-339.0 s wall per 5,116-update F1 run + stage overhead
    "dev_base_s": 2.38, "dev_per_model_s": 1.25, "dev_windows": 717,   # D1's model; measured 10,443 / 10,437 s (10 models)
    "export_s": 300,                 # D1 run_s3: chain exit 02:40:14Z -> stop 02:42:10Z (weights harvested in-run)
    "smoke_stage_s": 240, "smoke_train_s": 60, "smoke_export_s": 420,
    # technical checks (s2_barrier.py; ESTIMATES, not yet measured: replaced by session 0's measured barrier time)
    "barrier_s": 300,                # vm-check + start difference + data identity (~4 GB hashed) + fixture + ACK wait
                                     # (D1 run_s3 harvest of a 320 MB final.pt: 12-22 s after WEIGHTS_READY)
    "run_check_s": 75,               # vm-check without start difference / data hashing, per run 2..10
    "smoke_barrier_s": 240}
EVIDENCE = {"d1_s3": f"{S.D1_S3_TOP}/v1/job.log, driver.log; work/d1_collapse_2026-10-06/chain/CU_SPEND_confirm.jsonl",
            "d1_s4": "outputs/d1_collapse_2026-10-06/chain_runs/run_s4_20261007T030004Z/v1/job.log",
            "nds": "outputs/adapt_nds_2026-10-05/chain_runs/run_s{1,2,3}_*/v1/{driver,job}.log"}


# ---------------------------------------------------------------- projection
def session_projection(index: int) -> dict:
    m, runs = MEASURED, S.session_runs(index)
    setup = m["setup_code_s"] + m["english_upload_s"]
    if S.SESSIONS[index]["kind"] == "smoke":
        checks = m["smoke_barrier_s"]
        chain = m["smoke_stage_s"] + m["gates_s"] + m["smoke_train_s"] + checks
        export = m["smoke_export_s"]
        dev = 0
        train = m["smoke_train_s"]
    else:
        train = len(runs) * m["train_run_s"]
        checks = m["barrier_s"] + (len(runs) - 1) * m["run_check_s"]
        dev = round(m["dev_windows"] * (m["dev_base_s"] + len(runs) * m["dev_per_model_s"]))
        chain = m["stage_welsh_s"] + m["stage_irish_s"] + m["stage_dev_s"] + m["gates_s"] + train + checks + dev
        export = m["export_s"]
    wall = setup + chain + export
    cu = round(wall / 3600 * m["cu_per_h"], 2)
    return {"kind": S.SESSIONS[index]["kind"], "runs": [r["name"] for r in runs], "setup_s": setup, "train_s": train,
            "checks_s": checks, "dev_pass_s": dev, "chain_s": chain, "export_s": export, "wall_s": wall, "wall_cap_s": int(WALL_MARGIN * chain) + 600, "cu": cu,
            "cu_require": round(REQUIRE_MARGIN * cu, 2), "cu_cap": round(CAP_MARGIN * cu, 2),
            "local_peak_gb": round(len(runs) * S.WEIGHT_BYTES / 1e9 + 0.5, 2)}


def projection() -> dict:
    sessions = {str(i): session_projection(i) for i in S.SESSIONS}
    total = round(sum(s["cu"] for s in sessions.values()), 2)
    caps = round(sum(s["cu_cap"] for s in sessions.values()), 2)
    rec = {"schema": "s2_cu_projection_v1", "utc": N.utc(), "measured": MEASURED, "evidence": EVIDENCE,
           "projection": {"cu_per_h": MEASURED["cu_per_h"], "per_window_base_s": MEASURED["dev_base_s"],
                          "per_model_s": MEASURED["dev_per_model_s"], "dev_windows": MEASURED["dev_windows"],
                          "sessions": sessions, "total_cu": total, "total_at_session_caps_cu": caps,
                          "ceiling_cu": S.CU_CEILING, "headroom_cu": round(S.CU_CEILING - total, 2),
                          "status": "PASS" if caps <= S.CU_CEILING else "OVER_CEILING_AT_CAPS"},
           "d1_reference": {"session3_cu": 37.09, "session4_cu": 37.25, "per_run_cu": round(37.09 / 10, 3),
                            "note": "per-run = session spend / 10 (training ~0.84 CU + DEV inference ~2.13 CU per "
                                    "model + shared ~7.4 CU per session)"}}
    N.write_json(S.CU_PROJECTION, rec)
    return rec


# ---------------------------------------------------------------- plan
def load_freeze(path: Path) -> dict:
    lock = N.read_json(Path(path).with_name("FREEZE_LOCK.json"))
    sha = N.sha256_file(path)
    if sha != lock["freeze_sha256"]:
        raise SystemExit("FREEZE.json differs from FREEZE_LOCK.json")
    f = N.read_json(path)
    f["_sha256"] = sha
    return f


def session_record(top: Path | str | None) -> dict:
    p = Path(top) / "SESSION_VERIFIED.json" if top else None
    return N.read_json(p) if p and p.is_file() else {}


def previous_ok(f: dict, index: int, previous: Path | None) -> str | None:
    if index not in S.PREVIOUS:
        return None
    prev, want = session_record(previous), S.PREVIOUS[index]
    if prev.get("status") != "PASS" or prev.get("index") != want or prev.get("freeze_sha256") != f["_sha256"]:
        return f"session {index} needs session {want}'s SESSION_VERIFIED.json (PASS, this freeze)"
    return None


def cu_spent() -> float:
    p = S.CU_SPEND
    if not p.is_file():
        return 0.0
    return round(sum(json.loads(x)["spent"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()), 2)


def plan(f: dict, index: int, previous: Path | None) -> dict:
    if index not in S.SESSIONS:
        raise SystemExit(f"no stage-2 session {index}")
    why = previous_ok(f, index, previous)
    if why:
        raise SystemExit(why)
    pr = f["cu_projection"]["sessions"][str(index)]
    spent = cu_spent()
    if spent + pr["cu_require"] > S.CU_CEILING:
        raise SystemExit(f"CU: spent {spent} + 1.10 x projection {pr['cu_require']} exceeds the {S.CU_CEILING} CU ceiling")
    runs = S.session_runs(index)
    # hard stop at 60 CU: the VM-side wall cap (HR1 gates.py budget refuses every stage past it) is sized so that
    # spend so far + setup + chain wall + export stays below the ceiling minus 1 CU kept for the export hold
    rate = f["cu_projection"]["cu_per_h"]
    hard_left = S.CU_CEILING - spent
    hard_wall = int((hard_left - 1.0) / rate * 3600) - pr["setup_s"] - pr["export_s"]
    wall_cap = min(int(pr["wall_cap_s"]), hard_wall)
    if wall_cap < pr["chain_s"]:
        raise SystemExit(f"CU: the wall cap the 60 CU ceiling allows ({wall_cap} s) is below the projected chain "
                         f"({pr['chain_s']} s); nothing is allocated")
    return {"KIND": S.SESSIONS[index]["kind"], "SEEDS": " ".join(str(r["seed"]) for r in runs),
            "RUN_NAMES": " ".join(r["name"] for r in runs),
            "INFER_NAMES": " ".join(r["name"] for r in runs if r["infer"]),
            "PLAN_CU_REQUIRE": pr["cu_require"], "PLAN_SESSION_CU_CAP": min(pr["cu_cap"], round(S.CU_CEILING - spent, 2)),
            "PLAN_HARD_CEILING_LEFT": round(hard_left, 2), "PLAN_WALL_CAP_S": wall_cap,
            "PLAN_LOCAL_PEAK_GB": pr["local_peak_gb"]}


# ---------------------------------------------------------------- verification after the export
def train_ok(tr: dict, run: dict, f: dict, weights_dir: Path, v: list) -> None:
    name, seed = run["name"], run["seed"]
    smoke = run["kind"] == "smoke"
    if smoke:
        if tr.get("status") != "SMOKE" or tr.get("seed") != seed:
            v.append(f"{name}: smoke TRAIN_RESULT missing or not a smoke of seed {seed}")
    else:
        if tr.get("status") != "COMPLETE" or tr.get("seed") != seed or tr.get("recipe") != S.D1_RECIPE or \
                tr.get("updates") != S.N_UPDATES or tr.get("local_labels_read") is not False or \
                tr.get("lower_saxony_updates") != 0:
            v.append(f"{name}: TRAIN_RESULT is not a complete F1 control of seed {seed} without local labels")
        if tr.get("start", {}).get("sha256") != S.START["sha256"]:
            v.append(f"{name}: start weights differ from ADAF barrow")
        tabs = tr.get("tables_sha256", {})
        if tabs.get("w_table") != S.W_TABLE_SHA256 or tabs.get("i_table") != S.I_TABLE_SHA256:
            v.append(f"{name}: Welsh/Irish tables differ from C's")
        want = f["plan_checks"][str(seed)]
        got = {str(e["epoch"]): (e.get("draw_digest"), e.get("steps_digest")) for e in tr.get("plan_summary", {})
               .get("epochs", [])}
        if got != {e: (want["epoch_draw_digests"][e], want["epoch_steps_digests"][e]) for e in want["epoch_draw_digests"]}:
            v.append(f"{name}: trained steps differ from the frozen plan-check of seed {seed}")
        det = tr.get("detector", {})
        if det.get("last_update") != S.N_UPDATES - 1 or "on_at_end" not in det:
            v.append(f"{name}: detector record incomplete")
    vf = weights_dir / (f"SMOKE_{name}" if smoke else name) / "VERIFIED.json"
    if not vf.is_file() or N.read_json(vf).get("sha256") != tr.get("final_sha256"):
        v.append(f"{name}: final weights not verified locally")


def checks_ok(res: Path, top: Path, index: int, v: list) -> dict:
    """Every run's RUN_CHECK.json PASS (technical verification, s2_barrier.py vm-check); the barrier run (the first in
    frozen run order) also BARRIER_ACK.json PASS on the VM and the driver's local readback ACK PASS, with the same
    sha256; the next seed started only after the barrier (chain.log order)."""
    smoke = S.SESSIONS[index]["kind"] == "smoke"
    runs = [(f"SMOKE_{r['name']}" if smoke else r["name"]) for r in S.session_runs(index)]
    out = {"runs": {}}
    for d in runs:
        c = N.read_json(res / "runs" / d / "RUN_CHECK.json") if (res / "runs" / d / "RUN_CHECK.json").is_file() else {}
        if c.get("status") != "PASS":
            v.append(f"{d}: technical run check missing or not PASS")
        out["runs"][d] = {"status": c.get("status"), "updates": c.get("updates", {}).get("update_records"),
                          "optimizer_step_counter": c.get("updates", {}).get("optimizer_step_counter"),
                          "skipped_all_ignore": c.get("updates", {}).get("skipped_all_ignore"),
                          "crop_stats": c.get("targets", {}).get("crop_stats"),
                          "fixture_digest": c.get("checkpoint", {}).get("fixture", {}).get("fixture_digest"),
                          "environment": c.get("environment")}
    first = runs[0]
    vm_ack = N.read_json(res / "runs" / first / "BARRIER_ACK.json") if (res / "runs" / first / "BARRIER_ACK.json")         .is_file() else {}
    loc = top / "v1" / f"BARRIER_ACK_{first}.json"
    loc_ack = N.read_json(loc) if loc.is_file() else {}
    if vm_ack.get("status") != "PASS" or loc_ack.get("status") != "PASS" or             vm_ack.get("ack", {}).get("readback_sha256") != loc_ack.get("readback_sha256"):
        v.append(f"{first}: first-run barrier (durable readback) not PASS")
    log = (res / "chain.log").read_text(encoding="utf-8", errors="replace") if (res / "chain.log").is_file() else ""
    if not smoke and len(runs) > 1:
        i_pass, i_next = log.find(f"BARRIER PASS {first}"), log.find(f"STAGE train_{runs[1]} START")
        if i_pass < 0 or i_next < 0 or i_next < i_pass:
            v.append(f"{runs[1]} did not start after the barrier PASS of {first} (chain.log)")
    out["barrier"] = {"run": first, "status": "PASS" if vm_ack.get("status") == loc_ack.get("status") == "PASS"
                      else "FAIL", "readback_sha256": loc_ack.get("readback_sha256"),
                      "waited_s": vm_ack.get("waited_s"), "vm_check": out["runs"].get(first, {}).get("status")}
    return out


def verify(top: Path, f: dict, index: int, weights_dir: Path, previous: Path | None) -> dict:
    v, res = [], top / "v1" / "results"
    exp = N.read_json(top / "v1" / "EXPORT.json") if (top / "v1" / "EXPORT.json").is_file() else {}
    if exp.get("status") != "VERIFIED":
        v.append("EXPORT.json is not VERIFIED")
    if not (res / f"session_{index}.COMPLETE").is_file():
        v.append(f"session_{index}.COMPLETE missing")
    why = previous_ok(f, index, previous)
    v += [why] if why else []
    trained, detectors, proposals = {}, {}, {}
    smoke = S.SESSIONS[index]["kind"] == "smoke"
    for run in S.session_runs(index):
        d = f"SMOKE_{run['name']}" if smoke else run["name"]
        p = res / "runs" / d / "TRAIN_RESULT.json"
        tr = N.read_json(p) if p.is_file() else {}
        train_ok(tr, run, f, weights_dir, v)
        cur = res / "runs" / d / "curves.jsonl"
        if not cur.is_file() or N.sha256_file(cur) != tr.get("curves_sha256"):
            v.append(f"{run['name']}: curves.jsonl missing or differs from TRAIN_RESULT")
        trained[run["name"]] = tr.get("final_sha256")
        detectors[run["name"]] = tr.get("detector")
    chk = checks_ok(res, top, index, v)
    faults = [p for p in (top / "faults").glob("*") if not p.name.endswith(".lock")] if (top / "faults").is_dir() else []
    if smoke and not faults:
        v.append("smoke: no forced transfer fault recorded (FAULT_INJECT must exercise the hold/retry route)")
    if not smoke:
        inf = S.ROOT / f["pins"]["dev_infer_manifest"]["path"]
        tiles = {t["id"] for t in N.read_json(inf)["tiles"]}
        for run in S.session_runs(index):
            name = run["name"]
            h = SAD.proposals_ok(res / "dev" / f"PROPOSALS_{name}_PRIVATE.json", name, trained.get(name) or "?",
                                 f["pins"]["dev_infer_manifest"]["sha256"], tiles, v)
            SAD.consume_ok(res / "dev" / f"CONSUME_{name}.json", v)
            proposals[f"dev/PROPOSALS_{name}_PRIVATE.json"] = h
    rec = {"index": index, "kind": S.SESSIONS[index]["kind"], "status": "FAIL" if v else "PASS", "violations": v,
           "freeze_sha256": f["_sha256"], "trained": trained, "detectors": detectors, "proposals": proposals,
           "barrier": chk["barrier"], "run_checks": chk["runs"],
           "top": N.rel(top), "previous": session_record(previous).get("top") if index in S.PREVIOUS else None,
           "utc": N.utc(), "attribution": S.ATTRIBUTION}
    N.write_json(top / "SESSION_VERIFIED.json", rec)
    if v:
        raise SystemExit(f"session {index} not verified: {v[:5]}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("projection", "plan", "verify"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--previous", type=Path)
    a = ap.parse_args()
    if a.cmd == "projection":
        print(json.dumps(projection()["projection"], indent=1))
    elif a.cmd == "plan":
        p = plan(load_freeze(Path(a.args[0])), int(a.args[1]), a.previous)
        print("\n".join(f"{k}={shlex.quote(str(v))}" for k, v in p.items()))
    else:
        top, f, idx, wdir = Path(a.args[0]), load_freeze(Path(a.args[1])), int(a.args[2]), Path(a.args[3])
        rec = verify(top, f, idx, wdir, a.previous)
        print(json.dumps({k: rec[k] for k in ("index", "kind", "status", "trained")}))


if __name__ == "__main__":
    main()
