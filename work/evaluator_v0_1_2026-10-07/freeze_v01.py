"""Write FREEZE.json for evaluator v0.1: sha256 of every frozen file, the test summary and the archive-gate status.

  python -B work/evaluator_v0_1_2026-10-07/freeze_v01.py TESTS_LOG ARCHIVE_GATE_JSON
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ("SPEC.md", "eval_v01.py", "VOIDS.json", "run_checks.py", "CHECKS.json", "tests/test_eval_v01.py",
          "freeze_v01.py")
DEPENDS = ("work/holdout_region_2026-10/step2/region_score.py", "work/next_experiment_2026-09-23/r3_curation.py")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    log, gate = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    checks = json.loads((HERE / "CHECKS.json").read_text(encoding="utf-8"))
    ver = checks["regions"]["verden"]
    rec = {
        "schema": "evaluator_v0_1_freeze_v1", "evaluator": "v0.1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files_sha256": {f: sha(HERE / f) for f in FROZEN},
        "depends_sha256": {d: sha(ROOT / d) for d in DEPENDS},
        "tests": log.read_text(encoding="utf-8").strip().splitlines()[-1],
        "checks": {"stage1_reproduction": ver["stage1_reproduction"],
                   "v01_equals_reading_i": {d: ver["readings_adaf_barrow"][d]["v01_equals_reading_i"]
                                            for d in ("RAW", "NMS60")},
                   "oldenburg_adaf_barrow_equals_bench_v0":
                       checks["regions"]["oldenburg"]["adaf_barrow_bench_v0_reproduction"]["identical_to_bench_v0"]},
        "archive_gate": gate,
        "use": "Sensitivity for Stage 2 (work/stage2_d1_controls_2026-10-07) and the Oldenburg transfer check "
               "(work/oldenburg_transfer_2026-10-07); primary for experiments frozen after this file.",
    }
    (HERE / "FREEZE.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
