# Three-seed replication of the CONT e41 recipe, and a three-seed ensemble

Written 2026-09-26 by Claude Opus 5.5 (coordinator session), before any seed was trained. Standing execution authority
(AGENTS.md); the owner chose this line in chat on 2026-09-26 ("Prepare; I'll unseal today") and asked that the GPU
launch wait for Codex's archive receipt of the background run's intermediates and a fresh storage-guard check.

**Project outcome advanced.** Detector quality. (1) The TEST prerequisite `three_seed_confirmation`
(`verify/v3/v3_protocol.verify_freeze`), which asks whether CONT e41's VALIDATION gain over public ADAF survives a
change of training seed. (2) One capability test: does averaging three seeds beat the single selected model?

## Hypotheses

- **H1 (replication).** Continuations of frozen R2 (epoch 38), identical to CONT except for the sampling seed and
  selected by CONT's own holdout rule, also beat public ADAF AO on the fixed 45 VALIDATION blocks.
- **H2 (ensemble).** ENS3 recovers more parts than CONT e41 at a fixed proposal density on the same 45 blocks.
  ENS3 is the pixelwise mean of the three selected seeds' sigmoid channel-1 maps (CONT e41, S2\*, S3\*), identity
  pass, followed by the unchanged frozen threshold and vectorising (tta_run's `C2` construction, 2026-09-24).

## Arms and recipe

| Arm | `r2_train.SEED` | Everything else |
|---|---|---|
| CONT (recorded) | 20260923 | `work/aim_train_audit_2026-09-24/cont_train.py` as run on 2026-09-24 |
| S2 | 20260924 | identical |
| S3 | 20260925 | identical |

- `seed_train.py` imports the unchanged `cont_train.py` (sha `572502ce…`, equal to CONT's recorded
  `script_sha256`) and `r2_train.py` (sha `11ef821b…`, checked by `cont_train.train`); holdout selection imports the
  unchanged `holdout_eval.py` (sha `c817a761…`, as frozen 2026-09-24). It sets `r2_train.SEED` and redirects only `OUT`, `LEDGER`,
  `CEILING_S` and the arm → manifest map (both new arms use CONT's `R2_MASKS_MANIFEST_PRIVATE.json`).
- The seed enters only through the sampling plan: window order, negative cycle, crop offsets and dihedral index
  (`epoch_plan`, `negative_at`, `draw`). Start weights (R2 `best.pt`, sha `fbd1cbac…`), fresh AdamW (lr = wd = 1e-4),
  plan epochs 40-45, loss, BN handling, masks, holdout set and per-epoch saves are unchanged.
- **Reproduction gate (smoke, before freeze).** With `SEED = 20260923`, step 0 of plan epoch 40 must give CONT's
  logged loss 0.30280110239982605 exactly (`outputs/aim_train_audit_2026-09-24/train_CONT.log` line 1). With 20260924
  and 20260925 the step-0 batch must differ from CONT's.

## Selection (declared before training)

Per arm, `holdout_eval.score`'s rule: the epoch with the highest holdout primary recall (mean recall of the 506
holdout parts at 0.5, 1 and 2 negative-window proposals per km², 804 windows, identity pass); tie → later epoch.
No SCREEN, stage-2 or VALIDATION input is used. The selected checkpoints are called S2\* and S3\*. Report every
epoch's holdout row.

## VALIDATION stage (stage B; code frozen separately before it runs, rules fixed here)

- **Population and data.** The fixed 45 VALIDATION blocks of `work/validation_cont_2026-09-26/BLOCKS.json`
  (1,119.75 km², 265 common-class parts), read only from the cached mosaics of that run. Every mosaic must match
  both its fetch receipt's `dtm_sha256` and the `source_sha256` recorded in the locked seq-16 CONT proposal file
  before inference; any mismatch stops the stage. No refetch.
- **Stage-B smoke (development terrain, before any reservation).** On the validation run's two development blocks
  (SCREEN_100_44, SCREEN_100_57), all four arms run. CONT's map hash and objects must equal that run's recorded
  development records; otherwise stop.
- **Reservations first.** Before any VALIDATION pixel is read, three rows are charged through
  `v3_protocol.reserve_validation`: `SEEDS-2026-09-26-S2` and `SEEDS-2026-09-26-S3` (bucket `family`) and
  `SEEDS-2026-09-26-ENS3` (bucket `improvement`). The family bucket has 1 of 3 slots left, so
  `decisions/2026-09-26-validation-family-reallocation.md` moves one unused improvement slot to family: baseline 11,
  family 4, improvement 15, redteam 4, final 6 (total 40, final 6 unchanged). Expected sequences 17, 18, 19; any other
  value stops the stage. A failed execution stays charged; no retry.
- **Inference.** One pass over the 45 mosaics; per tile, three forwards (CONT e41, S2\*, S3\*) and four arms
  (CONT, S2, S3, ENS3), through the validation run's path (`tta_run.objects_multi` tiling and vectorising, the
  half-open interior filter). CONT here is a determinism guard: its objects must equal the locked seq-16 objects.
  More than 2 of 45 mismatching blocks stops the stage before scoring; 1-2 are reported and ENS3 keeps the fresh
  forward it was built from.
- **Lock.** All arm proposals are hashed into a create-only lock before labels are opened.
- **Scoring.** `validation_run.score_population` and `interval`, unchanged: global 60 m NMS and pool key of
  `screen_scorer.build_pool`, K_d = round(d × 1,119.75) for d = 0.5, 1, 2, greedy one-to-one matching, denominator
  all 265 common-class parts, paired block bootstrap (blocks as units, 2,000 draws, seed 20260924, percentiles 5/95).
  Public ADAF AO uses its locked seq-15 proposals (132 at 2/km²); CONT e41 uses its locked seq-16 proposals (157).

## Decision rules (primary density 2 proposals/km²)

**H1 (three-seed confirmation).** D_s = hits_s − hits_AO for s ∈ {CONT e41, S2\*, S3\*}, in points of 265.
- `CONFIRMED`: every D_s > 0 with its 90 % interval lower bound > 0, **and** the three-seed mean D ≥ 5.0 points
  (the original look's margin).
- `PARTIAL`: every D_s > 0, but an interval or the mean fails.
- `NOT_CONFIRMED`: any D_s ≤ 0.

Only `CONFIRMED` sets `three_seed_confirmation = true` for the TEST freeze.

**H2 (ENS3 vs CONT e41).** E = hits_ENS3 − hits_CONT (paired).
- `GAIN` iff the 90 % interval lower bound > 0.
- `HARM` iff the upper bound < 0.
- `NULL` otherwise.

The 0.5 and 1 per km² results are reported as secondary.

**TEST consequence (declared now).** CONT e41 is the primary TEST model unless ENS3 is `GAIN` at 2/km². In that
case ENS3 becomes primary and CONT e41 secondary. Public ADAF AO is the TEST baseline either way. S2\*/S3\* alone are
never promoted, since choosing the best seed would be selection on VALIDATION.

**Descriptive.** Seed spread (max − min hits); ENS3 − mean of members; per class (barrow / enclosure); all-class
hits; the members' holdout tables.

## Resources and guards

- **GPU.** Ceiling 5.0 h for the whole packet, charged to `outputs/seeds_2026-09-26/GPU_TIME.json` by every stage.
  Forecast: training 2 × 1.0 h, holdout 2 × 0.35 h, VALIDATION ~1.2 h. A ceiling stop is `INCOMPLETE`, not a result.
- **One GPU job at a time.** Each stage refuses to start while more than 3,000 MiB of GPU memory is in use.
- **Disk.** Output directory `outputs/seeds_2026-09-26` (uncompressed parent). Declared peak additional allocation
  7.6 GB: per arm 6 epochs × 0.32 GB + `last.pt` 0.86 GB + one atomic `.tmp` 0.86 GB = 3.64 GB, two arms, plus
  proposals/logs 0.3 GB. Every GPU stage runs
  under `verify/v3/storage_guard.py run --peak-gb` (+20 GB reserve).
- **Before any WSL read of the VALIDATION mosaics** (classic NTFS compression; STORAGE_RUNBOOK), convert them to WOF
  XPRESS4K with pre/post SHA-256 equality. This is the same procedure used for `train_raw`, with receipts kept.
- **Launch precondition (owner, 2026-09-26).** Codex's archive receipt for the background run's NEG/CTRL
  intermediates, then a fresh storage-guard check.

## Stop conditions

1. The smoke reproduction gate fails.
2. A frozen hash changes.
3. The training manifest gate is not `PASS`.
4. A reservation sequence is unexpected.
5. A mosaic hash mismatches.
6. More than 2 CONT guard mismatches occur.
7. The GPU ceiling is reached.

Each stop is recorded, and the evidence is preserved. After two failures on the same subproblem, diagnose rather
than retry.

## After the run

- Archive the unselected epochs and both `last.pt` files per STORAGE_RUNBOOK. Keep S2\*, S3\* and the proposals.
- Update `CURRENT_DETECTOR.json` only if H2 is `GAIN`.
- Score the forecasts in a separate outcomes file.
- Record the TEST-freeze inputs: seeds, their holdout and VALIDATION metrics, and sequences 17-19.
