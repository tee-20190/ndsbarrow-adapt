"""ADAPT-NDS-1 chain gates that differ from HR1's (HR1 gates.py is reused unchanged for env, pins, disk, budget, sink,
complete; FRESH-1 fresh_gates.py consume is reused unchanged for the region_score input contract).

  lint CHAIN_FILE MODE                 mandatory do_* calls present, in order, before the first training/inference call
  head OUT.json PLAN_KIND=PATH ...     preflight (driver, before any allocation): every pinned tile URL of the given
                                       staging plans answers HEAD 200 with its pinned Content-Length (FIT: and its
                                       pinned ETag = md5); kinds fit | dev | confirm | gsi (HR3 Irish plan zips)
  local OUT.json PEAK_GB               driver: C: free >= 24.0 GB + PEAK_GB, and the local final weights
                                       (outputs/adapt_nds_2026-10-05/weights) plus this session's two (A and C,
                                       2 x 319,970,119 B) <= 2.0 GB (BUILD_DECISIONS.md 1); the packet footprint
                                       (outputs/ + work/ of this packet) is reported
  weights-room BYTES                   exit 0 when C: free - BYTES >= 24.0 GB (weight downloads are deferred otherwise)
  windows MANIFEST                     number of windows of an inference manifest (stage budgets)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nds_common as N  # noqa: E402

_STAGING = ("do_lint", "do_env", "do_pins", "do_roles", "do_stage_w", "do_stage_i", "do_stage_ls", "do_stage_dev",
            "do_stage_confirm", "do_void", "do_step0", "do_parity", "do_plan_check")
MANDATORY = {"run": _STAGING,
             "smoke": ("do_lint", "do_pins", "do_roles", "do_stage_ls", "do_stage_confirm", "do_void")}
WORK_CALLS = ("do_train", "do_confirm", "do_dev", "do_wi", "do_smoke_train", "do_smoke_infer")
HEAD_EXIT, LOCAL_EXIT, ROOM_EXIT = 4, 73, 75


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


# ---------------------------------------------------------------- HEAD preflight
def head(url: str, tries: int = 3, timeout: int = 60) -> dict:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": N.USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return {"http": r.status, "bytes": int(r.headers.get("Content-Length") or -1),
                        "etag": (r.headers.get("ETag") or "").strip('"')}
        except (OSError, ValueError):
            time.sleep(3 * (k + 1))
    return {"http": -1, "bytes": None, "etag": None}


def pinned_urls(kind: str, plan: dict) -> dict[str, dict]:
    """name -> {url, bytes[, md5]} from a staging plan of the given kind (only the pins the VM checks)."""
    if kind in ("fit", "dev", "confirm"):
        out = {}
        for n, t in plan["tiles"].items():
            N.check_url(n, t["url"])
            out[n] = {"url": t["url"], "bytes": t["bytes"]} | ({"md5": t["md5"]} if kind == "fit" else {})
        return out
    if kind == "gsi":
        return {n: {"url": t["url"], "bytes": t["bytes"]} for n, t in plan["tiles"].items()}
    raise SystemExit(f"head: unknown plan kind {kind!r}")


def head_check(specs: list[str], out: Path, workers: int = 16) -> dict:
    rows, bad = {}, []
    for spec in specs:
        kind, _, path = spec.partition("=")
        urls = pinned_urls(kind, N.read_json(path))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            got = dict(zip(urls, pool.map(lambda n: head(urls[n]["url"]), urls)))
        for n, want in urls.items():
            g = got[n]
            ok = g["http"] == 200 and g["bytes"] == want["bytes"] and ("md5" not in want or g["etag"] == want["md5"])
            if not ok:
                bad.append({"kind": kind, "name": n, "got": g})
        rows[kind] = {"tiles": len(urls), "failed": sum(1 for b in bad if b["kind"] == kind)}
    rec = {"utc": N.utc(), "sets": rows, "failures": bad[:50], "failures_total": len(bad),
           "pass": not bad, "attribution": N.ATTRIBUTION["terrain"]}
    N.write_json(out, rec)
    if bad:
        raise SystemExit(HEAD_EXIT)
    return {k: v for k, v in rec.items() if k != "failures"}


# ---------------------------------------------------------------- local disk rules (packet "Resources")
def footprint_gb() -> float:
    total = 0
    for d in (N.PRIV, N.HERE):
        for p in d.rglob("*"):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
    return total / 1e9


def weights_gb() -> float:
    d = N.WEIGHTS_DIR
    if not d.is_dir():
        return 0.0
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and not p.is_symlink()) / 1e9


def local(out: Path, peak_gb: float) -> dict:
    free, used, wgt = N.free_gb(N.ROOT), footprint_gb(), weights_gb()
    new_w = len(N.ARMS) * N.WEIGHT_BYTES / 1e9                       # this session's two final weights (A and C)
    rec = {"utc": N.utc(), "c_free_gb": round(free, 3), "stop_below_gb": N.LOCAL_STOP_GB, "peak_gb": peak_gb,
           "weights_local_gb": round(wgt, 3), "session_new_weights_gb": round(new_w, 3),
           "weight_budget_gb": N.LOCAL_WEIGHT_GB, "budget_rule": "BUILD_DECISIONS.md decision 1",
           "packet_footprint_gb": round(used, 3),
           "free_ok": free >= N.LOCAL_STOP_GB + peak_gb, "budget_ok": wgt + new_w <= N.LOCAL_WEIGHT_GB}
    rec["pass"] = rec["free_ok"] and rec["budget_ok"]
    N.write_json(out, rec)
    if not rec["pass"]:
        print(json.dumps(rec), file=sys.stderr)
        raise SystemExit(LOCAL_EXIT)
    return rec


def weights_room(nbytes: int) -> dict:
    free = N.free_gb(N.ROOT)
    rec = {"c_free_gb": round(free, 3), "bytes": nbytes, "after_gb": round(free - nbytes / 1e9, 3),
           "floor_gb": N.WEIGHT_DEFER_GB}
    if free - nbytes / 1e9 < N.WEIGHT_DEFER_GB:
        print(json.dumps(rec), file=sys.stderr)
        raise SystemExit(ROOM_EXIT)
    return rec


def windows(manifest: Path) -> int:
    return sum(len(t["windows"]) for t in N.read_json(manifest)["tiles"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("lint", "head", "local", "weights-room", "windows"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    x = a.args
    if a.cmd == "lint":
        print(json.dumps(lint(Path(x[0]), x[1])))
    elif a.cmd == "head":
        print(json.dumps(head_check(x[1:], Path(x[0]))))
    elif a.cmd == "local":
        print(json.dumps(local(Path(x[0]), float(x[1]))))
    elif a.cmd == "weights-room":
        print(json.dumps(weights_room(int(x[0]))))
    else:
        print(windows(Path(x[0])))


if __name__ == "__main__":
    main()
