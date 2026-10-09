# Stage B amendment (before B0, before any VALIDATION fetch or reservation)

This amendment applies `COORDINATOR_B0_DECISION.md` (coordinator, 2026-09-26, after `READY.json`). It supersedes the conflicting lines of `PACKET.md`, and everything else in `PACKET.md` stays in force.

The stage-A files are unchanged:
- `FREEZE.json` (`a8eae6f4…`) and `READY.json` are kept as they were.
- Every file changed here is preserved byte-for-byte in `stageA_frozen/`. Its hashes equal the stage-A freeze.
- `FREEZE_B.json` freezes the result of this amendment.

## 1. B0 before the reservations (approved)

- `derive_blocks.py validation` reads the VALIDATION labels for cell membership and class tags only. It writes geometry and counts only.
- It runs after the coordinator's go is recorded in `PHASE_B_GO.json` and before `reserve`.
- B0 stops with no charge, writing `B0_STOP.json`, in either case:
  - a census mismatch against 735 / 393 / 351 / 46 / 45 / 265 / 275;
  - a feature difference between the G015 and E0097 label copies.

## 2. Primary scoring at each arm's own supply

- **Primary.** Each arm is scored at min(K_d, its own supply) (CAMPAIGN_AMENDMENT_2 item C.3.3). Stage 2's code did the same.
- **Descriptive only.** The Phase A equal-K rule, both arms at min(K_d, supply_AO, supply_CONT). Reported under `equal_K_descriptive` at every density.
- **Reported.** Every supply, and whether it limited the arm.
- **Unchanged.** The success rule's Δ, Δpp, bootstrap and thresholds. Its denominator is the common-class parts of the run blocks.

## 3. TEST protection for evaluation fetches

- **Enforced.** No fetched pixel, including the tile halo, lies within 500 m of any TEST square.
  - The footprint is the fetch rectangle grown by 64 m. This counts the 64 px tile halo at the 1 m source resolution, which also covers the 0.5 m grid's 64 px halo plus 20 px SLRM radius (42 m). Tile reads are in fact clipped to the fetched mosaic.
  - The guard runs on every fetch (`vguard.check_fetch`) and on the block list (`load_blocks`).
- **Descriptive.** The 5,040 m distance is computed and reported in `BLOCKS.json`, `descriptive_within_5040m`.
- **Exclusion.**
  - B0 excludes a block that fails the enforced guard before any charge, and reports it with its census counts under `excluded_hard_test_guard`.
  - The census is still checked on all 45 blocks.
  - K = round(d × A) uses the interior area A of the remaining blocks. The denominator is the common-class parts of those blocks.
  - `load_blocks` accepts an exclusion only if the block really fails the guard.
- **Geometry.** Only 3 SE/SP cells can fail the enforced guard: (80, 99), (99, 80) and (99, 59), each touching a TEST corner. If none of them is among the 45, the fixed 45 run unchanged.

## Code and evidence changes

| File | Change |
|---|---|
| `vguard.py` | Enforced footprint guard; `hard_guard_fail_cells`; exclusions checked in `load_blocks` |
| `derive_blocks.py` | Exclusion instead of stop; the 5,040 m list; run counts and K in the receipt; receipts written to `receipts/stageB/` |
| `validation_run.py` | Per-arm-supply primary with an `equal_K_descriptive` block; `FREEZE_B.json` / `READY_B.json`; smoke directory `smoke_stageB`; census check against the run blocks |
| `recount.py` | Per-arm supply |
| `smoke_check.py` | Stage-B schema; receipts written to `receipts/stageB/` |
| `freeze.py` | Stage-B freeze and ready |
| `tests/test_guards.py` | Hard-guard, margin, exclusion and per-arm-supply tests |
| `CONFIG.json` | `stage`, `budget_rule`, `test_guard`, `smoke.dir` |
| `DATA_MANIFEST.json` | Regenerated |

Because the fetch guard is on the inference path, the GPU smoke is re-run with the stage-B code. It runs on the same two development blocks into `outputs/validation_cont_2026-09-26/smoke_stageB/`, against the same recorded proposals. The Phase A smoke outputs stay in `smoke/`.
