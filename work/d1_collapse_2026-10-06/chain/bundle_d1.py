"""D1-COLLAPSE-1 bundle list: ADAPT-MATCH-0's session list for the arm (bundle_match.code_list, frozen; it carries the
MATCH trainer, ADAPT-NDS-1 / JOINT-1 / HR1 / HR3 / FRESH-1 modules, the step-0 fixture, the FIT and Verden label-free
inputs and the arm's start tar, with MATCH's refusals of label-bearing and reserved files) plus every D1 code file.
Nothing under outputs/d1_collapse_2026-10-06/ travels (no private D1 file is needed on the VM).

  python -B bundle_d1.py lists INDEX OUT_DIR [--extra REL ...]     # -> OUT_DIR/code.txt
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
sys.path.insert(0, str(D.MATCH / "chain"))
import bundle_match as BM  # noqa: E402  ADAPT-MATCH-0 (frozen)

REL = D.HERE.relative_to(D.ROOT).as_posix()
D1_CODE = ["d1_common.py", "d1_train.py", "PACKET.md", "CU_PROJECTION.json",
           "chain/d1_gates.py", "chain/sessions_d1.py", "chain/bundle_d1.py", "chain/freeze_d1.py",
           "chain/chain_lib_d1.sh", "chain/chain_d1_run.sh", "chain/vm_job_d1.sh", "chain/driver_d1.sh"]
MATCH_SESSION_OF = {"AO_L": 1, "BARROW_L": 2}                  # bundle_match session index carrying the arm's start tar
PRIV_REL = D.PRIV.relative_to(D.ROOT).as_posix()


def arms(index: int) -> list[str]:
    return sorted({r["arm"] for r in D.session_runs(index)})


def code_list(index: int, extra: list[str]) -> list[str]:
    out = []
    for arm in arms(index):
        out += BM.code_list(MATCH_SESSION_OF[arm], [])
    out += [f"{REL}/{f}" for f in D1_CODE] + list(extra)
    out = list(dict.fromkeys(out))
    leaked = [r for r in out if r.startswith(f"{PRIV_REL}/") and r not in extra]
    leaked += BM.never_violations([r for r in out if r not in extra])
    if leaked:
        raise SystemExit(f"bundle refused: private or label-bearing files would travel to the VM: {leaked[:5]}")
    bad = BM.refused_data(out)
    if bad:
        raise SystemExit(f"bundle refused: refused data: {bad[:3]}")
    return out


def lists(index: int, out_dir: Path, extra: list[str]) -> dict:
    import bundle_stream as bs
    out_dir.mkdir(parents=True, exist_ok=True)
    rels = code_list(index, extra)
    bs.check_list(rels, tuple(extra))
    (out_dir / "code.txt").write_bytes(("\n".join(rels) + "\n").encode())
    return {"code": {"files": len(rels), "bytes": sum((D.ROOT / r).stat().st_size for r in rels)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("lists",))
    ap.add_argument("args", nargs=2)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    print(json.dumps(lists(int(a.args[0]), Path(a.args[1]), a.extra)))
