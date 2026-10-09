# STAGE2-D1-CONTROLS freeze receipt (2026-10-07)

- **`freeze/FREEZE.json` sha256 `b5fdd0a9b59a62f2151da1adcd16e4187223420e135f012ad9244ac23445cb5a`** (equals `freeze/FREEZE_LOCK.json`).
- `freeze/PINS_FREEZE.json` sha256 `945bc48a49de1fff27bcfd847188ba70c007f2cfcde1a60089d745d7ce02511b`: 5851 pinned files (every file the sessions ship, every local file the result depends on, and every file of this directory).
- `FREEZE.json` `packet_files_sha256`: sha256 of all 47 packet files (code, plans, receipts, tests); private plan pins (`SUBSETS_S2`, ten `ENGLISH_E45`) and the 13 plan-checks are in `pins` / `plan_checks`.
- Freeze UTC: 2026-10-07T23:29:40Z. `PACKET.md` sha256 `5c954c65308f94b4631227d7a93264dfea2d0128ea9e79ff98cd4a05731d295c`.
- Authorisation: the owner's standing execution authority (AGENTS.md, 2026-09-24; reaffirmed 2026-10-01/05). No separate owner review is claimed.

## Checks run for the freeze

- `python -B verify/v3/archive_gate.py check` → `{"utc": "2026-10-07T23:29:50Z", "pending_archives": [], "status": "PASS"}`.
- Tests (`tests/record_tests.py` → `TESTS.json`, 2026-10-07T23:25:03Z): **PASS, 46 passed** (`46 passed, 6 warnings in 433.80s (0:07:13)`); by file: test_s2 27, test_boot_s2 8, test_barrier 11 (the 22 original packet tests, two of them rewritten for the integer gate, plus the new packet, bootstrap and barrier tests). Interpreter `/home/tee-/adaf-h2h-venv/bin/python`, `-B`.
- Plan / mask-level dry run (`DRYRUN_PLAN.json`): PASS, mask_level True, 415.7 s; composition `COMPOSITION.json` sha256 `2cf8c914dd9327c8135b0d9ef1b78538cf8de944e656e083ad3edc4189d6affe`.
- Barrier-route dry run (`DRYRUN_BARRIER.json`): PASS (wait_ack_pass, next_seed_after_barrier, weights_verified, faults_recorded, vm_job_exit_0, driver_subshell_exit_0, corrupt_ack_refused).
- Pseudo-control bootstrap (`BOOT_S2_PSEUDO.json`): point +0.0625, 95 % [+0.0389, +0.0849] (equals Stage 1's +0.0625 point estimate).
- CU projection: 47.66 CU projected, 59.58 CU at session caps, ceiling 60.0 CU (PASS).
- Storage: pre-freeze mask build declared 0.5 GB under `storage_guard.py` (actual 5.6 MB); session launches declare 0.82 GB (smoke) and 3.7 GB (run) and require C: free ≥ 22.8 / 25.7 GB.

## Not done / open at the freeze

- The adversarial critic pass (CHECKLIST step 2) was not run; the coordinator's freeze criteria did not include it.
- The D1 export route (results parts, export hold) was not re-run locally; it is unchanged from D1's verified sessions. The changed route (barrier harvest + readback ACK) was dry-run locally.
- The barrier and run-check times in the CU projection are estimates; session 0 measures the smoke barrier.
- Evaluator v0.1 (`work/evaluator_v0_1_2026-10-07/`) did not exist at the freeze; its sensitivity runs only if a FREEZE receipt exists at scoring time and the entry module follows the interface in `PACKET.md`.

## After the freeze

Do not edit any file of this directory (all are pinned; `check-run` refuses a launch on any change). A defect found by the smoke needs a documented freeze amendment with the same design, data, gate and analysis.

## Not committed

`freeze/PINS_FREEZE.json` stays local (uncommitted): its file list names English R2 mask files by NHLE list-entry
number, and record IDs are not committed. Its sha256 is pinned in `FREEZE.json` (`pins_freeze_sha256`) and checked by
`freeze_s2.py load` at every launch. Private plan pins and masks live under `outputs/` and are never committed.
