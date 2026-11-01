# ENG-ADAPT-1: English barrow-only adaptation of public ADAF barrow (pre-registration draft, 2026-10-19, Opus)

**Status:** DRAFT. It is frozen by `FREEZE.json`, with the code, plans and scorer, before any EB checkpoint exists. No
EB inference output may exist at the freeze.

**Capability lines:**
- Reproducible detector improvement in England. Can the public barrow specialist be beaten on English HER records?
- Regional generalisation. Is A011's Oldenburg gain local, or a product of the recipe and barrow-only data?

## Why this test

- **England.** ENG-OPS-3 showed CONT e41 ties public ADAF barrow in Wiltshire and beats it only in Hampshire.
  DIRECTION names the next lever: "a matched English adaptation of ADAF barrow". That model does not exist yet.
- **Locality.** A011 (OLDENBURG-TRANSFER-1) showed that D1 (ADAF barrow + recipe F1 on Lower Saxony register barrows)
  beats unchanged ADAF barrow on Oldenburg by +3.86 points. Stage 2 compared D1 with a mixed-class nonlocal replay
  control, so it tested a package, not locality (CONTRIBUTIONS A011 limits; Pro R4-1 "What a class-matched experiment
  would require").
- **One training arm answers both questions.** EB is the same recipe from the same start, with nonlocal
  (English) barrow-only data. The Pro R4-1 conditions are met as follows:
  - barrow-only targets;
  - an explicit policy for co-occurring non-barrow monuments (such windows are excluded);
  - the same number of distinct positive units (852) and the same replay multiplicities, from D1's own draw generator;
  - a selection rule fixed without outcomes;
  - no recipe selection in either arm. EB inherits F1 unchanged, and F1 was selected on D1's local runs, which is
    disclosed as an asymmetry favouring D1.

## 1. Arm EB (trained; ten seeds)

| Property | D1 run s (D1-COLLAPSE-1 session 3) | EB run s | How it is held equal |
|---|---|---|---|
| Start | ADAF barrow tar `51b679fa…`, 861,530,273 B | the same tar | `match_train.fresh_model("BARROW_L")` |
| Recipe | F1, `d1_train.update`; fresh AdamW (lr 1e-4, wd 1e-4); cosine 1e-4 → 1e-6 over 5,116 updates; frozen BN; batch 1; fp32, TF32 off | the same | `d1_train` imported unchanged (sha `2e232c77…`) |
| Updates | 5,116 in epochs 4 and 5; positive at even, background at odd steps | the same | `match_schedule` constants |
| Draw generator | `match_schedule.run_plan(seed, pool, bg)` over the Lower Saxony FIT table | **the same function**, with the English pool and background lists | imported unchanged (ADAPT-MATCH-0 freeze) |
| Crops | `SCH.draw` (512 px, ±160 px, 8 dihedral), streams `default_rng([seed + 30, 2, epoch, step])` | the same | the same function |
| Loader | void policy V, `adapt_train.AdaptSteps` + `loader` | the same; every step is an English `real` step (no regional map) | `AdaptSteps(windows, real, [], {})` |
| Seeds | 2026092805 … 2026092814 | the same ten | `torch.manual_seed(s)` |
| Weights kept | final only | final only | `final.pt` at t = 5,115 |

**English data (the only substantive difference).**
- **Source.** R2 FIT windows (`plans.load_windows()`, R2 mask manifest `f44a36bd…`, 846 × 846 px at 0.5 m). The 804
  holdout windows are never drawn. No Lower Saxony, Welsh or Irish file is read.
- **Pool (positives).** A FIT positive window is eligible only if it has at least one live positive part (pixels > 0)
  and every live positive part is classed `barrow` alone (`classes == ["barrow"]`). Windows containing any live
  enclosure, hillfort, moated or motte part are excluded. This is the policy for co-occurring non-barrow monuments.
  Exactly **852** eligible windows are taken in ascending `sha256("ENG-ADAPT-1|pos|" + window_id)` order. If fewer
  than 852 are eligible, all are used and the shortfall is recorded. 852 equals D1's number of distinct positive units.
- **Background.** All **741** FIT negative windows. D1 had 852; this is the one unavoidable count difference.
- **Masks are unchanged** (label 1 = live positive part, 255 = IGNORE, 0 = background). In the selected windows
  every positive pixel is barrow by construction.
- **Composition is measured before the freeze** (CPU, `COMPOSITION.json`, descriptive only):
  - draws per distinct window;
  - positive-role foreground fraction (mean and quantiles);
  - empty-target positive crops;
  - the same quantities for D1, from its pinned plan and masks where readable, or quoted from Stage 2's
    `COMPOSITION.json`.

## 2. Primary A: England (pre-registered development comparison)

**Terrain and labels.** These are frozen, unchanged, from ENG-OPS-2 and ENG-OPS-3:
- G: 12 cells, Gloucestershire HER via KYP;
- H: 40 cells, Hampshire HCC extract;
- W: 40 cells, Wiltshire & Swindon HER via KYP.

The primary label sets, eligible areas and 50 m matching are exactly those of `eo2_score` and `eo3_score`. All 92
cells exclude R2 fitting windows, which EB's data is drawn from, and labels within 50 m of them.

**Status of this terrain.** It is previously evaluated development terrain. CONT and ADAF were scored on it today.
EB's recipe and data selection did not use it. A positive result needs a fresh-terrain confirmation before any
operating change.

**Inference.** One VM pass over the 92 cells and the 58 WF cells (section 2b) with 7 models:
- EB_05 … EB_09, the first five seeds;
- CONT e41;
- public ADAF barrow.

England uses five EB seeds because 150 cells × 12 models would take about 7.5 GPU hours (ENG-OPS-3 measured
15 s per cell per model). The five seeds are fixed now by number, before any output exists. Oldenburg keeps all ten.

Comparators are re-inferred on the same EA DTM bytes as EB (ST-B12: bytes can change between fetches). Decoding is
the frozen `adaf_configs.vectorise` used for ADAF barrow in ENG-OPS-2/3, for every model. Per-cell DTM sha256 is
recorded. Parity with the saved ENG-OPS proposals is reported, never substituted.

**Contrast A1: EB vs public ADAF barrow, at the comparator's own workload.**
- For each stratum, n_s = the number of ADAF barrow proposals in that stratum (same pass).
- EB_s is cut to its top n_s by `rank_key` (the `eo2_score.primary_block` rule).
- Per cell: D_s,c = found(EB_s) − found(ADAF barrow).
- Statistic: D̄ = mean over the five England seeds (05-09) of Σ_c D_s,c.

**Contrast A2: EB vs CONT e41, at a fixed budget of 1.0 proposal per eligible km² per stratum.**
- n_s = round(1.0 × eligible km²_s). Both lists are cut to their top n_s.
- A list shorter than n_s is used whole and flagged.
- D̄ is defined as for A1.

**Interval.**
- Stratified cluster bootstrap over cells, using the frozen `C.boot_ci` index scheme with seed 20261026 and 10,000
  replicates.
- The resampled quantity is each cell's seed-mean D.
- Bonferroni over A1 and A2: the 97.5 % interval (1.25, 98.75).

**Verdicts.**

| Verdict | Condition |
|---|---|
| **EB_BEATS_BARROW** | A1 adjusted lower bound > 0, and ≥ 4 of 5 seeds with Σ_c D_s,c > 0 |
| **EB_BEATS_CONT** | The same, for A2 |
| **EB_BEATS_BOTH** | Both of the above |
| **LOSES** | Upper bound < 0 for either contrast, reported for that contrast |
| **NOT_DISTINGUISHED** | Otherwise |

**Deployable checkpoint.** EB_05, fixed now by the D1 run-ID rule (lowest seed). Its own A1/A2 results are reported.
It is never chosen by outcome.

**Secondary (descriptive).**
- Per county and per seed;
- 0.5 and 2.0 per km² budgets for A2;
- A1 on the barrow-class subset (the ENG-OPS-3 lens class rule);
- the NHLE ≤ 50 m / > 50 m split (the ENG-OPS-3 audit rule);
- EB vs ADAF barrow at fixed budgets 0.5 / 1 / 2 per km²;
- chance at each workload (frozen `C.chance`).

## 2b. Confirmation C: fresh Wiltshire terrain (pre-registered, labels fixed before any inference there)

**Stratum WF.** Every 5 km BNG cell of the Wiltshire & Swindon HER area that has never been inferred, under ENG-OPS-3's
SPEC3 section 1 rules unchanged:
- reserved, TEST, VALIDATION and SCREEN terrain excluded;
- prior inferred cores now include ENG-OPS-3's 40 cells;
- prior lead points and windows excluded;
- R2/CONT fitting windows excluded;
- at least 5,000 m from every ENG-OPS-1 / FRESH-POOL-1, ENG-OPS-2 and POOL-NEXT-1 cell core.

There is no separation from ENG-OPS-3's own cells. They are excluded as inferred cores only, so WF is never-inferred
terrain beside them. The scout count is 58 cells, 1,346.0 km² eligible (`outputs/fresh_scope_2026-10-19_PRIVATE/`);
the frozen AREA_WF file decides. Every WF cell is used; there is no sampling.

**Labels.** KYP `ext/wilts_her_2022_v2` layers 4-7, one envelope query per cell interior, fetched before the freeze.
The SPEC3 section 2 primary rule is unchanged: PRIMARY_RE type, ≤ 1 ha, rotated-rectangle extent ≤ 150 m, and not
within 50 m of a fitting window.

**Inference.** In the same VM pass as section 2, with the same 7 models and decoding.

**Contrasts.**
- **C1.** EB vs public ADAF barrow at the comparator's own WF workload.
- **C2.** EB vs CONT at 1.0/km².
- Statistic: seed-mean D, as in section 2.
- Interval: one-sided 95 % lower bound (5th percentile of 10,000 cluster-bootstrap replicates over WF cells, seed
  20261027). This is a directional confirmation of the section 2 claim, so there is no Bonferroni here.
- Seed rule: ≥ 4/5 seeds positive.

**Claim levels, fixed now.**

| Level | Condition |
|---|---|
| **CONFIRMED_BEATS_BARROW** | EB_BEATS_BARROW (section 2) and C1 passes |
| **CONFIRMED_BEATS_CONT** | EB_BEATS_CONT and C2 passes |
| Development-only | The section 2 verdict passes without C. Reported as development-only |
| Fresh-only | C passes without section 2. Reported, but not a confirmed claim |

"EB_BEATS_BARROW (section 2)" means the A1 condition holds, so EB_BEATS_BOTH also qualifies; the same applies to CONT.
WF chance at the WF workloads uses seed 20261028 (descriptive only).

## 3. Primary B: Oldenburg locality (pre-registered; training-disjoint, previously inspected district)

**Inference and comparators.**
- `infer_irl.py` (`f3855551…`) runs over the frozen CONFIRM manifest `8fce717e…`, 699 windows, on pinned DGM1 bytes
  (staging plan `e37e3e01…`), for EB_05 … EB_14.
- D1 proposals: the OLDENBURG-TRANSFER-1 session files, sha-pinned at the freeze.
- ADAF barrow proposals: `21421a9c…`.
- Scorer: the frozen `score_v0` used by OLDENBURG-TRANSFER-1 (RAW decoder; N = 959; K = 225 / 451 / 902).

**Contrasts at RAW, b = 0.5.**
- **B1 locality.** Δ_s = recall(D1_s) − recall(EB_s), with seeds paired by number.
- **B2 nonlocal gain.** G_s = recall(EB_s) − recall(ADAF barrow).

**Verdicts.**

| Verdict | Condition |
|---|---|
| **LOCAL_GAIN** | mean Δ ≥ +0.030 (unrounded; ≥ 28.77 matches), ≥ 8/10 Δ_s > 0, and the paired joint 95 % lower bound > 0 (2 km grid, the frozen OLDENBURG-TRANSFER-1 bootstrap) |
| **NONLOCAL_GAIN** | mean G ≥ +0.030 and ≥ 8/10 G_s > 0 |

**Readings, fixed now.**

| B1 | B2 | Reading |
|---|---|---|
| LOCAL_GAIN | no NONLOCAL_GAIN | The A011 gain needs local register labels |
| no LOCAL_GAIN | NONLOCAL_GAIN | Recipe plus barrow-only data is enough; locality not shown |
| LOCAL_GAIN | NONLOCAL_GAIN | Nonlocal data helps, local data helps more |
| no LOCAL_GAIN | no NONLOCAL_GAIN | Neither contrast is established |

**Limit.** This still compares two particular corpora (Pro R4-1). Acquisition, annotation quality (NHLE scheduled
extents vs register footprints), preservation and landscape all remain part of the difference.

## 4. Forecasts (Opus, before the freeze)

| Event | P |
|---|---:|
| EB_BEATS_BARROW (England) | 0.30 |
| EB_BEATS_CONT (England) | 0.22 |
| EB_BEATS_BOTH | 0.12 |
| CONFIRMED_BEATS_BARROW (section 2 and C1) | 0.18 |
| C1 passes (fresh Wiltshire, EB vs ADAF barrow) | 0.30 |
| England A1 point estimate D̄ > 0 | 0.55 |
| Oldenburg LOCAL_GAIN | 0.55 |
| Oldenburg NONLOCAL_GAIN | 0.25 |
| Any EB seed collapses (zero proposals on 699 Oldenburg windows) | 0.05 |

## 5. Execution and resources

**Machine.** One Colab G4 VM. Act only on this session's VM. At most 3 machines run project-wide; at 21:10Z two were
held by another session.

**Job order.**
0. Environment check and step-0 gate, then a Drive custody probe: `vm_archive.sh` uploads the tiny STEP0.json. A
   failure stops the job before training. The probe leaves one small object in the owner's Drive archive folder per
   launch.
1. Train ten seeds, about 6 min each.
2. Write finals to the owner's Drive through `vm_archive.sh`, then local `fastarch.py adopt`.
3. England inference.
4. Pack checkpoint A.
5. Oldenburg staging and inference.
6. Pack checkpoint B.

The VM stops only after both result tars are sha-verified locally and the finals are DRIVE_VERIFIED.

**Ceilings.**
- 14 h of VM time and 130 CU; about 9.5 h is expected. A step that would exceed them stops, and the stop is
  reported. Per-step limits are in `ea_vm_job.py`. The resumable England step reruns until COMPLETE or until its
  limit.
- On the VM: `storage_guard` with a declared peak of 30 GB (DGM1 staging 18 GB, weights 3.2 GB, bundle and outputs).

**Local disk.**
- C: had 18.0 GB free at 21:20Z, below the 20 GB reserve. It then fell to 2.8 GB at 22:20Z from a system-level
  cause outside the repository (BLOCKERS HR-B3).
- A result tar is fetched only while free space minus its size stays at or above 1.5 GB. Otherwise it stays on the
  kept-alive VM and also goes to Drive through `vm_archive.sh`.
- No weight or bulk file comes to C:. Only result tars (proposals and logs, under 0.3 GB) and receipts do.
- The English window bundle is streamed in 48 MiB parts, so the local peak is under 0.1 GB.
- Judgement recorded: under the AGENTS.md storage rule, this upload is not a substantial local download,
  preprocessing, training or inference launch. The substantial job runs, and is guarded, on the VM.

**Integrity and privacy.**
- The scorer runs once, after both tars are verified.
- An independent Haiku 5.5 recount, written from this SPEC alone, is pinned before scoring.
- A technical inference failure is rerun unchanged.
- A changed recipe, data rule, decoder or scorer is a new experiment.
- HER labels stay private (KYP/HCC terms). Committed results carry no coordinates or cell ids.
- Oldenburg stays previously evaluated terrain. Sealed terrain is untouched.
