"""D1-COLLAPSE-1 chain gates: a checked copy of ADAPT-MATCH-0's match_gates.py (frozen 351b5961) with D1's local disk
rule. HR1 gates.py is reused unchanged for env, pins, disk, budget, sink, complete; ADAPT-NDS-1 adapt_gates for
head_check and windows.

  lint CHAIN_FILE MODE              mandatory do_* calls present, in order, before the first training/inference call
  head OUT.json KIND=PATH ...       preflight: every pinned tile URL answers HEAD 200 with its pinned size (fit, dev)
  local OUT.json PEAK_GB            C: free >= 20 GB reserve + PEAK_GB + 2 GB page-file headroom (PACKET "Resources")
  weights-room BYTES                exit 0 when C: free - BYTES >= 22.0 GB (a weight download waits otherwise)
  windows MANIFEST                  number of windows of an inference manifest (stage budgets)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import d1_common as D  # noqa: E402

N = D.N
sys.path.insert(0, str(D.NDS / "chain"))
import adapt_gates as AG  # noqa: E402  ADAPT-NDS-1 (frozen): head_check, windows

MANDATORY = {"run": ("do_lint", "do_env", "do_pins", "do_roles", "do_stage_ls", "do_stage_dev", "do_void", "do_step0",
                     "do_parity", "do_plan_check")}
WORK_CALLS = ("do_train", "do_dev")
LOCAL_EXIT, ROOM_EXIT = 73, 75


def lint(chain_file: Path, mode: str) -> dict:
    calls = [ln.strip().split()[0] for ln in Path(chain_file).read_text(encoding="utf-8").splitlines()
             if ln.strip().startswith("do_")]
    seq = MANDATORY[mode]
    first = {g: calls.index(g) for g in seq if g in calls}
    missing = [g for g in seq if g not in first]
    order = [g for g in seq if g in first]
    ordered = all(first[a] < first[b] for a, b in zip(order, order[1:]))
    work = min((i for i, c in enumerate(calls) if c in WORK_CALLS), default=len(calls))
    late = [g for g in order if first[g] > work]
    if missing or late or not ordered or work == len(calls):
        raise SystemExit(f"lint ({mode}): missing {missing}; after the first work call {late}; in order {ordered}; "
                         f"work call present {work < len(calls)}")
    return {"status": "PASS", "mode": mode, "mandatory": list(seq)}


def head(out: Path, specs: list[str]) -> dict:
    kinds = {s.partition("=")[0] for s in specs}
    if not kinds <= {"fit", "dev"}:
        raise SystemExit(f"head: D1 reads only fit and dev tiles, not {sorted(kinds - {'fit', 'dev'})}")
    return AG.head_check(specs, out)


def weights_gb() -> float:
    d = D.WEIGHTS_DIR
    if not d.is_dir():
        return 0.0
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and not p.is_symlink()) / 1e9


def local(out: Path, peak_gb: float) -> dict:
    free = D.free_gb(D.ROOT)
    need = D.LOCAL_RESERVE_GB + peak_gb + D.PAGEFILE_HEADROOM_GB
    rec = {"utc": N.utc(), "c_free_gb": round(free, 3), "peak_gb": peak_gb, "reserve_gb": D.LOCAL_RESERVE_GB,
           "pagefile_headroom_gb": D.PAGEFILE_HEADROOM_GB, "need_gb": round(need, 3),
           "d1_weights_local_gb": round(weights_gb(), 3),
           "rule": "PACKET 'Resources': C: free >= 20 GB reserve + declared peak + 2 GB page-file headroom",
           "pass": free >= need}
    N.write_json(out, rec)
    if not rec["pass"]:
        print(json.dumps(rec), file=sys.stderr)
        raise SystemExit(LOCAL_EXIT)
    return rec


def weights_room(nbytes: int) -> dict:
    free = D.free_gb(D.ROOT)
    rec = {"c_free_gb": round(free, 3), "bytes": nbytes, "after_gb": round(free - nbytes / 1e9, 3),
           "floor_gb": D.WEIGHT_DEFER_GB}
    if free - nbytes / 1e9 < D.WEIGHT_DEFER_GB:
        print(json.dumps(rec), file=sys.stderr)
        raise SystemExit(ROOM_EXIT)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("lint", "head", "local", "weights-room", "windows"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    x = a.args
    if a.cmd == "lint":
        print(json.dumps(lint(Path(x[0]), x[1])))
    elif a.cmd == "head":
        print(json.dumps(head(Path(x[0]), x[1:])))
    elif a.cmd == "local":
        print(json.dumps(local(Path(x[0]), float(x[1]))))
    elif a.cmd == "weights-room":
        print(json.dumps(weights_room(int(x[0]))))
    else:
        print(AG.windows(Path(x[0])))


if __name__ == "__main__":
    main()
