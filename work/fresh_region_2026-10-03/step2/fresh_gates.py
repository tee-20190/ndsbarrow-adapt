"""FRESH-1 Step 2 chain gates that differ from HR1's (HR1 gates.py is reused unchanged for env, pins, disk, budget and
complete).

  lint CHAIN_FILE MODE                 mandatory do_* calls present, in order, before the first inference call
  consume MANIFEST PROPOSALS OUT.json  region_score's input contract accepts the proposals (every tile present, the
                                       manifest's sha256, a ranked pool); no recall is computed (no labels on the VM)
  names SPEC ...                       model names of NAME=PATH@SHA / CONT / ADAF_AO specs (space separated)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_common as F  # noqa: E402

_STAGING = ("do_lint", "do_env", "do_pins", "do_roles", "do_stage", "do_verify", "do_void", "do_parity")
MANDATORY = {"run": _STAGING, "smoke": _STAGING}
INFER_CALLS = ("do_infer_core", "do_infer_extras", "do_smoke_infer")


def lint(chain_file: Path, mode: str) -> dict:
    calls = [ln.strip().split()[0] for ln in Path(chain_file).read_text(encoding="utf-8").splitlines()
             if ln.strip().startswith("do_")]
    seq = MANDATORY[mode]
    first = {g: calls.index(g) for g in seq if g in calls}
    missing = [g for g in seq if g not in first]
    order = [g for g in seq if g in first]
    ordered = all(first[a] < first[b] for a, b in zip(order, order[1:]))
    infer = min((i for i, c in enumerate(calls) if c in INFER_CALLS), default=len(calls))
    late = [g for g in order if first[g] > infer]
    if missing or late or not ordered or infer == len(calls):
        raise SystemExit(f"lint ({mode}): missing {missing}; after the first inference call {late}; "
                         f"in order {ordered}; inference call present {infer < len(calls)}")
    return {"status": "PASS", "mode": mode, "mandatory": list(seq)}


def consume(manifest: Path, proposals: Path) -> dict:
    sys.path.insert(0, str(F.HR1_STEP2))
    import region_score as rs  # noqa: E402  frozen
    m, p = F.read_json(manifest), F.read_json(proposals)
    if p.get("manifest_sha256") != F.sha256_file(manifest):
        raise SystemExit(f"{Path(proposals).name}: made on another manifest")
    if set(p["objects"]) != {t["id"] for t in m["tiles"]}:
        raise SystemExit(f"{Path(proposals).name}: tiles differ from the manifest")
    s = rs.setup(m)
    pool = rs.ranked_pool(p["objects"], s)
    return {"status": "PASS", "model": p.get("model"), "checkpoint_sha256": p.get("checkpoint_sha256"),
            "tiles": len(m["tiles"]), "objects": sum(len(v) for v in p["objects"].values()), "pool": len(pool),
            "background_proposals": sum(1 for r in pool if r["neg"]), "utc": F.utc()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("lint", "consume", "names"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    x = a.args
    if a.cmd == "lint":
        print(json.dumps(lint(Path(x[0]), x[1])))
    elif a.cmd == "consume":
        rec = consume(Path(x[0]), Path(x[1]))
        F.write_json(Path(x[2]), rec)
        print(json.dumps(rec))
    else:
        print(" ".join(F.model_spec_name(s) for s in x))


if __name__ == "__main__":
    main()
