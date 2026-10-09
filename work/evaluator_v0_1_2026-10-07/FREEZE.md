# Evaluator v0.1: frozen 2026-10-07T22:36Z (Opus)

Hashes, tests and check results: `FREEZE.json`. The rule is `SPEC.md`; the implementation is `eval_v01.py`.

**What is frozen:**

- the role table (target / container / legal / other / void);
- void semantics, applied to area, proposals before NMS and K, and references by reach;
- the void tables for Verden and Oldenburg (empty, with evidence);
- the barrow endpoint (v0, unchanged);
- the member-supported complex endpoint: site-number link, then containment in a polygon of at least 200 m²,
  smallest polygon first, ties to the lower id; one hit per complex; one complex per proposal; legal polygons never
  complexes.

**Checks passed before the freeze:**

- 6/6 unit tests.
- Exact reproduction of the Stage 1 scorer on Verden: 440/440 values for ADAF barrow and the ten D1 runs.
- v0.1 = reading (i) on Verden ADAF barrow at every cell, with v0.1 − (iii) of 0 to −0.41 points.
- Oldenburg ADAF barrow = BENCH_V0 v0.
- `archive_gate.py check` PASS (`ARCHIVE_GATE.json`).

**Use:**

| Where | Role of v0.1 |
|---|---|
| Stage 2 controls | Sensitivity |
| Oldenburg transfer check | Sensitivity |
| New experiments frozen after this commit | Primary |

No D1 or control score under v0.1 had been computed when this file was written, except the barrow-level Verden
reproduction. That reproduction equals the already published Stage 1 numbers by construction.

A change to any frozen file needs a new version (v0.2), not an edit.
