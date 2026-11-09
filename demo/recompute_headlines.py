"""Aggregate-results verification demo: recompute two published headline numbers from released aggregate files.

It verifies arithmetic and file integrity only. It does not run detection or scoring.

Standard library only; runs on any laptop CPU in well under a second. No terrain, register geometry,
coordinates, weights or network access is needed or used.

  python demo/recompute_headlines.py            (run from the repository root)

Steps:
 1. Check the SHA-256 of the two released result files against paper_a/MANIFEST.sha256.
 2. Dutch sealed test (paper_a, Table 2): rebuild the D1 and ADAF-barrow hit counts from the paired
    both / only-D1 / only-ADAF counts, recompute recall and the difference in points, and check them
    against the values stored in the file. The stored 95 % simultaneous interval comes from a cluster
    bootstrap over 153 spatial units; it needs per-site hits, which are not released, so it is printed
    as stored, not recomputed.
 3. Denmark CONFIRM-D stage A (the later two-checkpoint system S_D + PROX, not the D1 checkpoint of
    step 2): read S_D + PROX and ADAF-barrow plain recall at 2 proposals/km2 and check
    that each equals a whole number of the 5,273 protected barrows within the stored 3-decimal rounding.
Exit status 0 only if every check passes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PA = ROOT / 'paper_a'
DUTCH = 'results/work/open_transfer_test_2026-10-22/RESULT_TEST.json'
DANISH = 'results/work/confirm_d_2026-10-25/RESULT_A1.json'
failures: list[str] = []


def check(cond: bool, what: str) -> None:
    print(f'  [{"PASS" if cond else "FAIL"}] {what}')
    if not cond:
        failures.append(what)


def manifest() -> dict[str, str]:
    rows = {}
    for line in (PA / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, path = line.split('  ', 1)
            rows[path] = digest
    return rows


def main() -> int:
    pins = manifest()
    print('1. Released file integrity (paper_a/MANIFEST.sha256)')
    for rel in (DUTCH, DANISH):
        digest = hashlib.sha256((PA / rel).read_bytes()).hexdigest()
        check(pins.get(rel) == digest, f'{rel} sha256 {digest[:16]}...')

    d = json.loads((PA / DUTCH).read_text(encoding='utf-8'))
    S = d['S']
    area = sum(d['eval_area_km2'].values())
    r = d['primary_family']['D1-ADAF_barrow@matched_workload']
    x, y = r['both'] + r['x_only'], r['both'] + r['y_only']
    diff = 100 * (x - y) / S
    print('\n2. Dutch sealed test: D1 vs public ADAF barrow model, matched workload')
    print(f'  registered barrows S = {S}; regions sum = {sum(d["sites_by_region"].values())}; '
          f'evaluated area = {area:,.1f} km2; spatial units = {d["n_units"]}')
    print(f'  D1 hits = both {r["both"]} + only-D1 {r["x_only"]} = {x}  ({100 * x / S:.1f} %)')
    print(f'  ADAF hits = both {r["both"]} + only-ADAF {r["y_only"]} = {y}  ({100 * y / S:.1f} %)')
    print(f'  difference = {diff:+.2f} points')
    lo, hi = r['ci_sim_pts']
    print(f'  stored (not recomputed) simultaneous 95 % interval (cluster bootstrap, n_boot={d["n_boot"]}): {lo:+.1f} to {hi:+.1f} points')
    check(sum(d['sites_by_region'].values()) == S, 'regional site counts sum to S')
    check((x, y) == (r['x'], r['y']), f'recomputed hit counts equal stored x={r["x"]}, y={r["y"]}')
    check(abs(diff - r['diff_pts']) < 1e-9, f'recomputed difference equals stored {r["diff_pts"]:.4f}')
    check(lo < diff < hi and lo > 0, 'stored interval contains the estimate and excludes zero')
    check(d['verdicts']['D1'] == 'BEATS_PUBLIC_ADAF', f'stored verdict D1 = {d["verdicts"]["D1"]}')

    k = json.loads((PA / DANISH).read_text(encoding='utf-8'))
    N = k['N']['all']
    sd, adaf = k['recall_points']['S_D|prox|2']['all'], k['recall_points']['ADAF_barrow|plain|2']['all']
    print(f'\n3. Denmark CONFIRM-D stage A, later system S_D + PROX vs ADAF ({k["cores"]} unseen cores, {N:,} protected barrows, 2 proposals/km2)')
    for name, v in (('S_D + PROX', sd), ('ADAF barrow plain', adaf)):
        hits = v * N / 100
        print(f'  {name}: {v:.1f} % = {hits:.2f} of {N:,}')
        # stored recall is rounded to 3 decimals, so a whole count must reproduce it within 0.0005 points
        check(abs(100 * round(hits) / N - v) <= 0.0005 + 1e-12,
              f'{name} recall equals a whole number of barrows ({round(hits)}) within the stored rounding')

    print('\nScope: registered-monument recall at fixed review workloads, not discovery precision.')
    print(f'Result: {"ALL CHECKS PASS" if not failures else f"{len(failures)} CHECK(S) FAILED"}')
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
