# Exploratory VALIDATION comparison: CONT e41 vs public ADAF AO

This packet was written on 2026-09-26, in Phase A, by Claude Opus 5.5 working for the coordinator (Claude Opus 5.5, owner session) under the AGENTS.md standing authority. It was frozen in `FREEZE.json` before any VALIDATION terrain fetch, any VALIDATION label parse and any reservation. The machine-readable parameters are in `CONFIG.json` and `DATA_MANIFEST.json`, and the code sits beside them. Outputs go to `outputs/validation_cont_2026-09-26/`.

## Owner go-ahead (B3-equivalent addendum)

- **The approval.** On 2026-09-26 the owner answered the coordinator's question in chat: "Yes, run it". This approves a one-time VALIDATION check of CONT e41 against public ADAF HRNet (AO), using 2 of the 26 remaining VALIDATION reservations, with no retry.
- **The record.** The coordinator relayed the approval to this executor and recorded it in LOG.md (2026-09-26 03:20Z section, 03:35Z bullet).
- **Scope.** It authorizes this comparison only.
- **Not the campaign's finish line.** The 2026-09-22 campaign closed when R2 failed its SCREEN gate (campaign `PREREG.md` lines 69, 116 and 120). This is a new exploratory packet that reuses the campaign's finish-line margin as its success rule. It revives no arm, and it is not the campaign's B3.

## Outcome it advances

Scoreboard line "detector vs the public ADAF baseline": the first held-out (VALIDATION) landscape evidence for any challenger.

## Hypothesis

CONT e41 recovers more common-class (barrow + enclosure) catalogue parts than public ADAF AO on the fixed 45 VALIDATION blocks.

## Population: the fixed 45 VALIDATION blocks

- **Rule.** Sources: `decisions/2026-09-20-detector-breakthrough-plan.md` line 31, Revision 1 item 5 at line 81, and the stage-1 contract.
  - Labels are the G015 copy of the VALIDATION labels: `targets/lidar-archaeology/v3_pro/results/G015_frozen_e0099_validation_eval/labels_VALIDATION.geojson`, SHA-256 `f10253ca…`. The E0097 copy (`69cd44c9…`) must hold identical features.
  - A part is owned by the half-open 5 km cell (floor(E/5000), floor(N/5000)) of its centroid.
  - A block is a cell in square SE or SP holding at least 3 distinct parts tagged barrow or enclosure.
  - Geometry: a 5 km core. The fetch rectangle is the core ± 210 m, clipped at the VALIDATION square. The interior is the core trimmed by 210 m wherever that clip applied.
  - Blocks are in numeric (i, j) order, with ids `VALIDATION_i_j`.
- **Census (2026-09-20), which B0 must reproduce exactly.**
  - 735 labels in total.
  - 393 common-class parts: 351 barrow, 46 enclosure.
  - 45 blocks holding 265 common-class parts and 275 parts of all classes (the extra 10 are 6 moated and 4 motte). About 89 % of the 265 are barrows (per `NEXT_STEPS_2026-09-26.md`).
- **The block list is not frozen in Phase A.**
  - No repository file lists the 45 blocks; the census recorded only the rule and the counts.
  - The list is a deterministic function of the labels file above and `derive_blocks.py`. Deriving it parses the VALIDATION labels, which Phase A forbids.
  - Phase A freezes the rule, the label bytes' hash and the derivation code. `derive_blocks.py selftest` reproduces both frozen SCREEN stage lists exactly from `labels_SCREEN.geojson`: 26 + 73 blocks, with identical geometry, counts, label ids and interior flags.
  - Phase B step 0 derives the VALIDATION list (see "Phase B sequence").
- **TEST halo.** By geometry alone, 9 SE/SP cells have fetch rectangles within 5,040 m of a TEST square (listed in `DATA_MANIFEST.json`).
  - If any of them is among the 45, B0 stops before any reservation.
  - The guard refuses such rectangles. The frozen SCREEN stage lists include 4 such blocks (`receipts/DERIVE_SELFTEST.json`); they were fetched under the TEST-square check alone.

## Arms (only the weights differ)

| Arm | Weights | SHA-256 | Ledger bucket |
|---|---|---|---|
| ADAF AO (public) | `work/adaf_head_to_head_2026-09-19/weights/AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar` (md5 pinned `fba3667a…`, CC BY-SA 4.0) | `a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb` | baseline |
| CONT e41 | `outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt` (TRAIN-holdout selection receipt `HOLDOUT_SELECTION_CONT.json`) | `91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383` | improvement |

## Inference path (stage 2, weights swapped)

The path is stage 2's (`work/stage2_2026-09-24/stage2_run.py`, SHA-256 `97257c5f…`):

- **Terrain.** EA LiDAR Composite DTM 1 m, fetched through `v3_fetch_england.fetch_area`.
- **Tiling and probability maps** (`tta_run.objects_multi`):
  - 0.5 m bilinear source view, 1,024 px tiles, 64 px halo;
  - RVT SLRM radius 20, normalised to −0.5..0.5;
  - sigmoid of channel 1, pixelwise max over overlaps;
  - validity and interior masking.
- **Objects** (`adaf_configs.vectorise`): threshold 0.5, 4-connected polygons with area > 30 m² and roundness > 0.4. Each object keeps its centroid, max score and mean probability.
- **Interior filter:** half-open.
- **Predictor:** a verbatim copy of `holdout_eval.predictor`, the one stage 2 used; a test compares the text.
- **Loading:** both models are loaded once. Each block is fetched once and its one mosaic feeds both arms in the same pass. Both proposal files record the mosaic SHA-256, and the lock refuses a block whose arms saw different bytes.
- **Kept mosaics.** The 45 mosaics stay in `outputs/validation_cont_2026-09-26/mosaics/`, NTFS-compressed before the first write, with a README saying that any access needs its own charged reservation.

## Scoring (frozen; `validation_run.score_population`)

- **Pool per arm.** The global 60 m NMS of `score_staged.global_nms` (score desc, id asc) over all 45 blocks. The pool is then ranked by (score, mean_probability, area_m2 desc, id asc) (`screen_scorer.build_pool`).
- **Budgets.** A is the sum of block interior areas in km², and K_d = round(d × A) for d = 0.5, 1 and 2, pooled over the 45 blocks as in stage 2.
- **Equal-K rule.** If either arm's pool is smaller than K_d, both arms are scored at K = min(K_d, supply_AO, supply_CONT), and the uncapped count at K_d is reported descriptively.
  - ADAF's supply on SCREEN was exactly 2.0 per km² (1,294 at 647.9 km²).
  - On stage 2 the cap never bound (AO 3,973 ≥ 3,635), so stage 2's results are unchanged by this rule.
- **Matching.** `v3_frozen_mixture.match_greedy`: proposals in rank order claim the nearest unclaimed label within 60 m or less, ties going to the label id. A hit counts for the label's owner block.
- **Denominator.** All common-class parts owned by the 45 blocks (raw labels, 265). There is no visibility or assessability filter. Parts outside a block interior and parts on void source cells are counted and reported descriptively.

## Primary endpoint and success rule

- **Endpoint.** Common-class recovery at 2 proposals per km². Δ = H_CONT − H_ADAF in parts, and Δpp = 100 Δ / 265.
- **Uncertainty.** The paired block bootstrap is stage 2's code verbatim (`stage2_run.py` lines 209-238):
  - 2,000 draws, seed 20260924;
  - the 45 blocks are resampled with replacement, jointly for both arms;
  - the global selection and the matches stay fixed;
  - the 90 % interval is the 5th and 95th percentile of the resampled part-count difference.
- **MET** if, and only if, both hold:
  - Δpp ≥ 5.0, which with 265 parts means Δ ≥ 14;
  - the interval's lower bound (parts) is > 0, strictly.
- **Otherwise NOT_MET.** There is one run, with no retry and no tuning.
- **Points interval.** The same draws divided by each draw's own resampled denominator are reported in points, descriptively. The rule uses the part-count interval, whose sign criterion does not depend on the scale.

## Secondary endpoints (descriptive, no rule)

- Recovery at 0.5 and 1 per km².
- Barrows and enclosures separately.
- Supply (pool size) per arm and whether it limited the budget.
- All-class hits over the 275 parts.
- Per-block hits and the parts each arm gains or loses against the other.
- Labels outside interiors and on void source cells.

## Three repairs (recorded before any VALIDATION read)

- **(a) Reservations.**
  - The `baseline` bucket is exhausted: 10 of 10 used, sequences 1-6 and 9-12.
  - `decisions/2026-09-26-validation-bucket-reallocation.md` moves one unused family slot to baseline. The new allocation is baseline 11, family 3, improvement 16, redteam 4, final 6, so the total stays 40 and final stays 6.
  - Phase B passes this allocation, plus `CONFIG.json` `reservation.reason` with the decision's SHA-256, as `reallocation_reason`. ADAF AO charges `baseline` (expected sequence 15) and CONT e41 charges `improvement` (expected 16).
  - The dry-run on a temporary ledger copy (`receipts/RESERVATION_DRYRUN.json`) returns sequences 15 and 16, shows that the unrepaired call raises "Validation bucket exhausted", and leaves the real ledger unchanged.
- **(b) The E0097 all-class criterion is dropped.**
  - Campaign PREREG line 9 adds "at least equal to E0097 on all-class recovery at the same budget". The source decision dropped that safeguard in Revision 1 item 5 (line 81): the 45 blocks hold only 10 parts outside the common classes.
  - E0097's recorded VALIDATION result is chip-level (R2 0.6925, 509/735). There is no landscape all-class number at 2 per km² to compare against.
  - E0097 is not re-run. All-class hits are reported descriptively for both arms.
- **(c) TEST stays sealed.** MET sets `READY_FOR_TEST` and nothing more. `verify_freeze` (`verify/v3/v3_protocol.py` lines 82-101) additionally requires:
  - a FROZEN detector freeze with a model list;
  - `threshold_source == VALIDATION`;
  - the development gates `baselines_reproduced`, `three_seed_confirmation`, `three_clean_redteams` and `clean_checkout_reproduced` all true;
  - a hashed file inventory.

  After that, the owner must type "unseal TEST" in the session. None of this is produced here.

## Multiplicity disclosure

- At least nine checkpoints have been scored on SCREEN.
- CONT meets campaign gate criterion 1 at exactly its threshold: 52 of 101 at K = 1,294 (`work/next_steps_2026-09-26/GATE_CONT.json`). Its neighbour e40 scored 48.
- On the other two criteria CONT scored 39 vs 32 at 0.5 per km² and 62 ≥ 47 marks within 30 m.
- CONT was chosen by TRAIN-holdout selection, and SCREEN was then scored repeatedly, so the SCREEN pass is weak evidence on its own.
- Development context (descriptive only):
  - On the 73 stage-2 blocks, CONT − AO was +9, +10 and +11 of 88 parts at 0.5, 1 and 2 per km². At 2 per km² that pair's interval was [+5, +17] parts, but it was not a declared stage-2 pair.
  - On SCREEN-26 it was +7, +5 and +6 of 101, with interval [−1, +14] at K = 1,294.
  - The stage-2 gain was mostly enclosures. VALIDATION is about 89 % barrows.

## Forecasts (copied from `decisions/2026-09-26-rank-next-steps-after-storage-recovery.md`; not changed)

| # | Event | P |
|---|---|---|
| F1 | The success rule is met (Δ ≥ 5 points at 2 per km² with the 90 % interval excluding zero) | 0.30 |
| F2 | CONT − ADAF > 0 at 2 per km² | 0.70 |

The same line is in LOG.md before any VALIDATION fetch (marker `VALCMP-2026-09-26 forecast`).

## Guards (code: `vguard.py`; tests: `tests/test_guards.py`)

- **Fetch.** A request is fetched only if all of the following hold:
  - the block id is in the frozen list for the mode;
  - the rectangle equals that block's frozen fetch rectangle, recomputed from (i, j);
  - the block lies in SE or SP (or, for the dev smoke, in NU or TL);
  - the rectangle is at least 5,040 m from every TEST square.

  The existing `assert_not_test` also runs, with no override. `validation_run.py infer` refuses any VALIDATION fetch until both reservations are in the ledger.
- **Labels.**
  - VALIDATION labels are opened only through `load_labels_after_lock`. That needs all 90 proposal files and a `PROPOSALS_LOCK.json` whose hashes match the current bytes. The lock is create-only and refuses mismatched source bytes across arms.
  - The scorer and the recount both refuse before the lock.
  - B0 is the single earlier read. It uses only cell membership and class tags, and it writes geometry and counts only.
- **Code.** Phase B entry points verify every `FREEZE.json` hash. The smoke's GPU-ledger entry records the inference-code hashes, and READY checks that they equal the frozen ones.

## Resources and storage

- **GPU ceiling:** 3.0 h cumulative over Phase A and B, in the durable ledger `outputs/validation_cont_2026-09-26/GPU_TIME.json`. Wall time includes model loading and fetch waits.
- **Smoke:** at most 0.25 h.
- **One GPU job at a time:** a run lock file, and a refusal when the GPU already holds more than 2,500 MiB.
- **Phase B GPU:** the smoke measured 88.5 and 87.5 s per block with both arms. Fetches took about 27 s and overlap inference through the prefetch thread. The smoke used 233 s in total, peak CUDA reserved 996 MiB. Projection: 45 × 88 s plus about 60 s for loading and the first fetch ≈ 1.1 h. The reserve step refuses if 1.25 × 45 × 100 s does not fit the remaining ceiling.
- **Disk:** mosaics are about 45-50 MB each, so about 2.2 GB is kept for 45 blocks. The declared storage-guard peak is 4 GB, which with the 20 GB reserve means 24 GB free is required (C: had 53.2 GB free at the smoke).
- **Wrapper:** `python verify/v3/storage_guard.py run --path /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH/outputs --peak-gb 4 -- python -B work/validation_cont_2026-09-26/validation_run.py infer` (WSL, adaf-h2h-venv).

## Phase B sequence (only after the coordinator's go)

- **B0.**
  - Record the go in `PHASE_B_GO.json` (`{"phase_b_go": true, ...}`), then run `derive_blocks.py validation`. It writes `BLOCKS.json` and `BLOCKS_RECEIPT.json`, create-only.
  - It stops (`B0_STOP.json`, no reservation) on any of: a census mismatch (735 / 393 / 351 / 46 / 45 / 265 / 275), a block inside the TEST halo, or a feature difference between the two label copies.
  - B0 comes before the reservations so that a failed population definition costs no charge. It evaluates no model.
  - This ordering needs the coordinator's explicit agreement, because B0 is a VALIDATION label read.
- **B1.** `validation_run.py reserve` → ledger sequences 15 (ADAF, baseline) and 16 (CONT, improvement), recorded in `RESERVATION_RECEIPT.json`.
  - Preconditions: FREEZE verifies, READY records passing Phase A checks, the LOG forecast line is present, BLOCKS verifies, and the projected GPU time fits.
  - Any other sequence means the ledger changed; stop.
- **B2.** `storage_guard.py run --peak-gb 4 -- validation_run.py infer` → 45 blocks with both arms per mosaic.
  - Proposal files are create-only.
  - An interruption may be resumed on the same frozen code; this is not a retry, and a resumed block must reproduce any file already written.
  - Running out of GPU or disk leaves the comparison INCOMPLETE and charged.
- **B3.** `validation_run.py lock` → `PROPOSALS_LOCK.json`.
- **B4.** `validation_run.py score` → `RESULT.json`, create-only. The labels are read only after the lock verifies.
- **B5.** `recount.py validation` → `RECOUNT.json`, which must PASS.
- **B6.** Report against this packet. Score F1 and F2 in LOG.md and in the forecast decision. The coordinator updates DIRECTION and state.

## Stop rules and interpretation limits

- **MET:** status `READY_FOR_TEST`; stop. TEST needs (c) above.
- **NOT_MET:** report the measured result, name one next hypothesis, stop. A positive sub-threshold margin is reported as such. It does not erase the development trade-off, and it does not support a claim either.
- **Failure after reservation:** the charge stands. Report the cause, with no retry.
- **What MET supports:** only "CONT e41 beats frozen public ADAF AO on this specified 45-block English VALIDATION evaluation".
  - VALIDATION has carried 14 prior charges, so this is a conditional comparison, not an untouched test.
  - ADAF's barrow class includes about 1,939 Historic England scheduled monuments from the same NHLE source as our labels. Their overlap with these blocks is unknown, and it may favour ADAF on barrows.
- **What the numbers are not.** Proposal density is not a measured false-alarm rate. Labels are NHLE part centroids, and 60 m matching does not verify outlines.

## Files

| File | Purpose |
|---|---|
| `CONFIG.json` | All run parameters |
| `DATA_MANIFEST.json` | Population rule, label bytes hash, halo cells, block-list status |
| `vguard.py` | Fetch guard, block-list checks, proposal lock, label gate |
| `derive_blocks.py` | B0 derivation and its SCREEN self-test |
| `validation_run.py` | Smoke, infer, lock, score, reserve |
| `recount.py` | Independent recount |
| `smoke_check.py` | Phase A reproduction checks |
| `freeze.py` | Manifest, FREEZE and READY |
| `tests/test_guards.py` | Guard and fidelity tests |
| `receipts/` | Phase A evidence |
