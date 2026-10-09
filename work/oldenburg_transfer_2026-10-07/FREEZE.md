# OLDENBURG-TRANSFER-1: frozen 2026-10-07T22:55Z (Opus), before any Oldenburg D1 proposal exists

**Hashes:** `FREEZE.json` holds the sha256 of every packet file and of 28 inputs. The inputs include:

- the Oldenburg manifests, staging plan, masks and register;
- the frozen ADAF barrow comparator proposals;
- the D1 session pins;
- the Stage 1 scorer;
- the v0 bootstrap engine;
- evaluator v0.1;
- `infer_irl.py`, `nds_reader.py` and `fresh_gates.py`;
- the environment freeze;
- the ADAF barrow tar.

`score_oldenburg.py` refuses to score if any of them changed.

**Frozen:**

- **The rule:** Pro R4-3 items 1-9.
  - Gate: RAW, b = 0.5, ten-run mean of (D1 − ADAF barrow) ≥ +0.030, unrounded, and ≥ 8/10 strictly positive.
  - Clear transfer improvement: gate plus a positive 2 km joint 95 % lower bound.
- The reading table, the secondary list and the joint two-district rule (shared checkpoint index).
- Evaluator v0.1 as a sensitivity.
- The deployable checkpoint `D1F1_BARROW_L_2026092805` (`d9ece1fb…`).
- The inference plan and the CU/storage projection.

**Checks before the freeze:**

- Legitimacy **PASS**: about 48 km from the FIT windows; no shared records or DGM1 squares; no D1 session touched
  Oldenburg.
- Comparator reuse verified by hash: `infer_irl.py`, manifest, pinned input bytes and environment.
- 10/10 gate and reading tests.
- `archive_gate.py check` PASS.
- storage_guard PASS at the declared 7.5 GB peak.

**After the freeze:** `DRYRUN_FIXTURE.json` runs the frozen scorer end to end. Ten synthetic re-rankings of ADAF
barrow's proposals stand in for D1, and no D1 output is read. It is a software check, not a result.

**Not done here:**

- The GPU inference. The coordinator launches it per `INFERENCE_PLAN.md`; driver build and smoke come first.
- Any D1 score on Oldenburg.

A changed recipe, decoder, checkpoint set, entity set or scorer is a new experiment (rule item 9).
