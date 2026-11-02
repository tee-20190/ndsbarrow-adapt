# DK-REGSUP-1 stage A freeze: does Danish register supervision improve the D recipe on held-out West Jutland?

> **FROZEN 2026-10-29** by the DK-REGSUP-1 session under the owner's standing execution authority (AGENTS.md,
> 2026-09-24), from `work/next_route_2026-10-28_opus/PROPOSAL.md` v2 with every fix in `CRITIQUE.md`. Committed
> **before any training or decode**. A later change is an addendum, made before the output it could affect, with its
> reason. Nothing here changes a loss, selector, threshold or decoder; faint learning stays closed
> (`work/faint_learn2_2026-10-28_codex/NEXT_DECISION.md`).

**Capability line:** regional scale-up. Does supervision from a region's own public register stack on the confirmed
D recipe (dense terrain SSL + four-channel input + F1)?

## 1. Question and scope

| Item | Fixed value |
|---|---|
| Contrast | **A − C**, paired by seed k, K = 15 |
| A_k | the D recipe's F1 from encoder SSL_D_k, with every pair p where (p − e) % 3 = 0 drawn from Danish register supervision (852 of 2,558 pairs; one third) |
| C_k | the D recipe's F1 from the same encoder, same seed, same 5,116 updates and schedule, Lower Saxony only (= D_k) |
| Terrain | the 171 DK-JW cores of CONFIRM-D (exposed development terrain), cross-fitted over 3 spatial folds |
| Records | CONFIRM-D's frozen 5,273 protected round barrows (`DK_RECORDS_PRIVATE.json` `7a581c91…fc53`) |
| Workload | 2 proposals/km² of land + PROX, 15 m one-to-one matching (CONFIRM-D `REACH_DECISION.json`) |

**Scope.** Registered protected round-barrow recall at fixed workload. It is not discovery and not precision on
unknown places (REFERRAL-PRECISION-1 measures that). It is a recipe claim over seeds: single-model recall averaged
over 15 seeds, not an ensemble. Stage B ([reserved region; name redacted]) is not part of this freeze; it needs its own FREEZE, and the
owner's conditional unseal ([decision record; file name redacted]) applies only if §8 passes.

## 2. Data and licences

**Terrain.** The CONFIRM-D 1 m DK-JW mosaics (`PREP_JW.json` `5afae7e2…52e2`): 18 still local, 153 restored from the
owner's Drive by `rs1_restore.py mosaics` (fastarch, size + sha256 verified against PREP_JW; Drive originals kept).
No DHM tile is downloaded; DK-WOOD-1 keeps priority on the owner's key.

**Register.** Every Fund og Fortidsminder WFS layer named in GetCapabilities (12), fetched 2026-10-29 00:27–00:28Z
over the DK-JW box + 1 km by `rs1_ff.py` (`FF_RECEIPT.json` `8f4a69f6…9c3d`). The service itself serves two
`punkt_ikkefredet` rows twice (identical rows; kept, harmless to an IGNORE union).
- **Record layers (9):** `punkt_fredet`, `punkt_ikkefredet`, `punkt_marin`, `areal_fredet`, `areal_alle`,
  `areal_marin`, `linje_fredet`, `linie_alle`, `linie_marin`.
- **Not records (3):** `areal_beskyttelse` (100 m protection zones, median 37,069 m²), `areal_ka` (heritage areas,
  median 0.37 km²) and `areal_adm` (administrative areas with no feature type). They are planning zones; buffering
  them would erase the ground around every protected monument.

**Lower Saxony FIT inputs.** The 1,704 legacy windows and 1,704 product files were restored from FAINT-LEARN-1's
archive instead of re-staging 3,291 LGLN tiles. `rs1_lscheck.py` (`LS_CHECK.json`): 1,704/1,704 windows byte-equal to
D1's `LS_RENDER_LOG`, 1,704/1,704 products byte-equal to stage B's `PRODUCTS.json` on both training VMs. These are the
exact bytes every D_k was trained on.

**Licence re-verification (2026-10-29), as `DK2_LICENCE_NOTE.md` requires.**
- SLKS rights page (`slks.dk/omraader/kulturarv/databaserne/rettigheder-til-data`, "Opdateret 04. november 2022",
  read 2026-10-29, html sha256 `c0445530…aae0`): textual data in the agency's databases are provided under CC0 and may
  be copied freely, not for unlawful use. Geometry is not named separately. Images are CC BY-NC-ND (not used).
- Fund og Fortidsminder "Regler for brug" (read 2026-10-29, `0ae00e34…3dc7`): a free service that "can be used for most
  purposes without the agency's approval"; harvesting of the website's content is not allowed; when in doubt, contact
  the agency. The site's own Download page offers bulk downloads of all monuments (points and areas).
- WFS GetCapabilities (2026-10-29): Fees NONE, AccessConstraints NONE.
- **Assessment:** training labels and internally held weights for research are "most purposes"; the data come from the
  official WFS (the machine route of the offered bulk download), not from harvesting site pages. No register geometry
  or coordinate is redistributed; credit "Slots- og Kulturstyrelsen, Fund og Fortidsminder". **Residual caveat
  (unchanged DK-2):** the geometry licence is not named explicitly; publishing weights trained on this register or
  derived maps waits for a consent-first question to SLKS (not sent here; no correspondence in this task).
- DHM/Terræn: SDFI free-data terms, attribution "Indeholder data fra Styrelsen for Dataforsyning og Infrastruktur,
  Danmarks Højdemodel". LGLN DGM1 CC BY 4.0. ADAF barrow start CC BY-SA 4.0 (shared weights stay CC BY-SA 4.0).

## 3. Spatial folds (`rs1_folds.py`, `FOLDS.json`)

- **Blocks:** the 171 cores ordered by (northing index, easting index) of their 5 km squares, cut into three contiguous
  north–south bands; the cut points minimise the spread of the bands' record counts. Band b is fold b's held-out set.
- **Separation:** fold f trains only on cores of the other bands whose 5 km square lies ≥ 10,000 m (Euclidean, square
  to square) from every fold-f core square. Training pixels outside those squares are NoData (§4), so every training
  pixel, its 97 m product halo and its label lie ≥ 10 km from every held-out core and its decode halo (256 m).
- Every core is held out exactly once; each seed's three held-out decodes join into one out-of-fold proposal set.
  K is 15, not 45: the folds are not seeds.

| Fold | Held-out cores | Held-out records | Training cores | Training records | Cores dropped by the buffer | Min. train–held-out distance |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 62 | 1,758 | 97 | 3,122 | 12 | 10,000 m |
| 1 | 54 | 1,757 | 90 | 2,747 | 27 | 10,000 m |
| 2 | 55 | 1,758 | 97 | 2,837 | 19 | 10,000 m |

`FOLDS_PRIVATE.json` `93b0a7f1…04da` (core ids; private).

## 4. Labels and Danish FIT windows (`rs1_fit.py`, `rs1_stage.py`, `FIT_PLAN.json`)

The Lower Saxony FIT rules (`work/adapt_nds_2026-10-05/plan_fit.py`) applied to each fold's training region.
- **Positives:** protected Rundhøj extents: the `areal_fredet` polygons (union per systemnr) of every locality whose
  `punkt_fredet` anlaegstype is Rundhøj (CONFIRM-D's cohort rule); 11,099 extents in the box. HR1 `masks.py` adds a
  5 m IGNORE band outside each positive.
- **IGNORE ±30 m** around every record of another locality in the 9 record layers (including `punkt_ikkefredet`, so
  ploughed-out and unprotected mounds never become background) and a 30 m disc at each of the 207 DK-POOL-1 stage-2
  proposals [custodian correspondence detail redacted]. A positive's own locality records are not "other".
  **Precedence:** positive > IGNORE > background (masks.py: positives overwrite IGNORE).
- **Positive windows:** FRESH-1 rule 5 groups (60 m centroid linkage); a group belongs to the region when its
  member-centroid mean lies inside it; 852 groups per fold drawn by `default_rng(2026102900 + fold)`; one 423 m
  integer window centred on each group's member-centroid mean.
- **Background windows:** 852 cells per fold of the 423 m grid, wholly inside the region shrunk by 100 m, ≥ 251 m from
  every record geometry and DK-POOL-1 proposal (the strict LS rule), and **on land** (≥ 99 % of the cell's 1 m pixels
  finite in the mosaics), drawn by `default_rng(2026102910 + fold)`. Masks are all-zero.
  - *Pre-freeze design fix:* without the land condition, 117 of fold 0's 852 cells were entirely sea (CONFIRM-D masks
    sea as NoData; the LS FIT union was land by construction). Found by the local staging check before any training.
- **Terrain:** each window and its 97 m halo take, per 1 m pixel, the mosaic of the training core whose square holds
  the pixel centre; pixels outside every training core are NoData (the LS "outside the FIT union" rule). Legacy window
  and products are written exactly as `dt1_stage.py` (`lgln_reader.write_raw`; `dt1_channels.native_products` cropped
  to the window + 1 cell).

| Fold | Eligible groups | Eligible land cells | Positive windows | Background windows | Table sha256 |
|---:|---:|---:|---:|---:|---|
| 0 | 2,493 | 1,211 | 852 | 852 | `5c7df724…8351` |
| 1 | 2,133 | 1,358 | 852 | 852 | `f833de28…0351` |
| 2 | 2,139 | 1,798 | 852 | 852 | `28a3cca9…6417` |

## 5. Arms and training

**Seeds:** K = 15: k ∈ {101, 202, 303, 404, 505, 606, 707, 808, 909} (existing dense-SSL encoders and their D_k as
C_k) and k ∈ {1010, 1111, 1212, 1313, 1414, 1515} (six new encoders).

| Run | Code | Count |
|---|---|---:|
| New SSL encoders | `dt1_ssl.py` unchanged through `rs1_seeds.py` (seed tuple only), 4,096 updates, DT1's edges/pool | 6 |
| New C_k | `dt1_train.py train k --arm D --encoder SSL_D_k` unchanged through `rs1_seeds.py` | 6 |
| A_k^f | `rs1_train.py train f k` (below), f ∈ {0, 1, 2} | 45 |

**`rs1_train.py`.** It imports `dt1_train` unchanged and keeps the model (`dt1_model.build("D")` from the seed's
encoder), recipe F1 (`d1_train.update`), the fresh AdamW, the cosine schedule over 5,116 updates, frozen BatchNorm,
batch 1, fp32 with TF32 off, `torch.manual_seed(k)` after the build, the item (`adapt_train.AdaptSteps`, void policy V,
512 px crops, ±160 px, dihedral; added channels via `dt1_train.extra_channels`), the collapse detector (records only)
and per-epoch resume. **The one change:** on each pair with (p − e) % 3 = 0 (`nds_schedule.is_ls_pair`, ADAPT-NDS-1's
positions) the step draws a Danish (positive, background) pair, `nds_schedule.ls_plan(k + 10, e, dk_pool, dk_bg)`
(ADAPT-NDS-1's own expressions on stream k + 40); each Danish window is drawn once per run. Every other step keeps C's
draw exactly (`match_train.schedule(k, e, ·)`: window, crop, dihedral, update index, learning rate).

**C_k identity (Codex review point 2).**
- Plan: for all 45 (f, k), `PLAN_CHECK.json`: A keeps C's draw on every kept step, Danish rows only on Danish steps,
  roles alternate, 852 + 852 Danish windows each drawn once; and C's reconstructed plan digest equals the
  `plan_summary` in each existing D_k's own `TRAIN_RESULT.json` (9/9).
- Inputs: the LS bytes are the D_k training bytes (§2).
- Environment: D1's step-0 rule must be bit-equal on every training VM, as in stage B. If it fails, no run is trained
  on that VM and the experiment stops as INCOMPLETE (no runtime-drifted A_k is paired with an old C_k).
- Descriptive: run-to-run replay is not bitwise (FL2 `CONTROL_REPLAY_DIAGNOSTIC.json`); that noise is random across
  seeds and is carried by the seed term. If the ceiling allows, C_101 is retrained once on the F VM and decoded on
  fold 0's cores to report the replay difference in hits (never scored in the gate).

**Checks before any training run (VM stage step):** Danish staging (`rs1_stage.py`) and masks (`masks.py build`, every
label_sha256); `rs1_train.py items` (RS1Steps returns DT1Steps' exact tensors on LS steps; Danish items finite, 4 ×
512 × 512); `rs1_train.py plan` for all 45 runs; a 64-update smoke on the production path. Danish staging bytes on the
VM are compared with a local staging of the same tables (descriptive; same numpy 2.5.3 / scipy 1.18.1 / rasterio
1.5.1).

**Failure handling.** A run that exits non-zero is rerun once with identical inputs (its output was never read). A
second failure at the same step stops the experiment with the evidence preserved and the root cause reported.

**Operational choices (they do not change what is computed):** runs per GPU under CUDA MPS (chosen by a 64-update
benchmark), SSL concurrency, decode workers, VM assignment, and the decode device for the new C_k (Colab G4 or the
local RTX 2060; CONFIRM-D showed identical scored hits across devices).

## 6. Power recompute (`rs1_power.py`, `POWER.json`), before any output

Interval Δ ± 1.96 √(se²_core + max(var_obs, 2 × (1.3 pts)²/K)). se_core from CONFIRM-D stage B on the same cores,
records and workload: all 0.371, large 0.467, isolated 0.411 points.

| K | se_core | Seed SD | MDE (80 %) | Power at +1.0 / +1.5 / +2.0 / +3.0 | Guard passes at Δ_g = 0 (large / isolated) |
|---:|---|---|---:|---|---|
| 9 | stage B | floor | 2.01 | 0.29 / 0.55 / 0.80 / 0.99 | 0.74 / 0.77 |
| **15** | **stage B** | **floor** | **1.69** | **0.38 / 0.70 / 0.91 / 1.00** | **0.85 / 0.89** |
| 15 | stage B | 2.0 pts | 1.78 | 0.35 / 0.66 / 0.88 / 1.00 | 0.82 / 0.86 |
| 15 | × 1.5 | floor | 2.05 | 0.28 / 0.54 / 0.78 / 0.98 | 0.66 / 0.73 |

K = 15 gives power 0.91 ≥ 0.8 at +2.0 under the central assumption; K is not reduced. With no true effect on large or
isolated mounds, a guard still fails by chance 11–15 % of the time; that risk is accepted as frozen.

## 7. Decode and scoring (`rs1_score.py`)

- **Decoder:** `dt1_decode.py` unchanged (`af88392a…`), four-channel route, threshold 0.5, area > 30 m², roundness >
  0.4, no NMS, centroid in core. A_k^f decodes only fold f's held-out cores; each new C_k decodes all 171.
- **Proposal sets:** A_k = union over f of A_k^f's held-out objects (`rs1_score.a_union` refuses an object outside its
  fold or a file whose core list differs from the fold). C_k = D_101–303 (CONFIRM-D stage A decode), D_404–909
  (stage B VM decode), C_1010–1515 (this packet).
- **Scoring:** CONFIRM-D stage B unchanged: `dk_score.inputs("JW")`, `cr1_score.score`, PROX per core at 2/km² of land,
  reach 15 m, `dk_a1.populations`.
- **Statistic:** `cd_analysis.recipe_contrast(A hits, C hits, pop, "scaled")`, Δ = mean_k Δ_k.
- **Descriptive only:** small and strong-grouped strata, plain 2/km² and 20/km², the constant-floor interval, reach 10
  and 25 m, per-fold contrasts, per-seed Δ_k, kept counts.
- Nothing is scored until all 45 A_k^f and 15 C_k proposal sets exist.

## 8. Gate and stop rule

**PASS** iff, at K = 15 under the scaled floor: the all-records lower bound > 0, **and** the large (≥ 200 m²) lower
bound > −2 points, **and** the isolated lower bound > −2 points. Otherwise **FAIL**.
- **PASS:** stage B may be frozen under the owner's conditional decision, as one frozen read of a reserved region
  [name redacted] with its own FREEZE committed before any of its DHM tiles is fetched. Not part of this session's task.
- **FAIL:** register supervision of the D recipe in Denmark is closed. The reserved region stays reserved.
- **INCOMPLETE:** the resource ceiling is reached, the step-0 rule fails, or an implementation step fails twice. No
  gate reading; K is never reduced; evidence preserved.
- The small and faint strata never gate (this is not a faint study).

## 9. Checks

- **Scorer smoke on this FREEZE, before any DK-REGSUP-1 output** (`rs1_score.py smoke`, `SMOKE_SCORE.json`): S1 splits
  each D_k by fold, unions it back and scores it against stage B's A_k (D1 recipe), K = 9; it must reproduce
  `RESULT_B_K9.json` exactly in every population and both floors. S2 (A = C) must give Δ = 0 and gate FAIL.
- **Independent recount** (`rs1_check.py`, `CHECK_A.json`): separate variance code from the private hits; every
  population and the gate must match to 1e-9.

## 10. Resources (hard limits from the brief)

- At most **4 owned Colab machines** (`rs1_vm.sh up` refuses a fifth), 6 account-wide; act only on own machines.
- Stage A ≤ **15 G4-hours** and ≤ **120 CU**, counting setup, uploads, staging, training and decode.
- **Measured-rate budget** (stage B: SSL 0.754 s/update, F1 345 s, decode 5.4 s per core-model): pure GPU work is
  5.15 (SSL) + 0.58 (6 C F1) + 4.31 (45 A F1) + 1.54 (C decode) + 3.85 (A decode) = 15.4 G4-h ≈ 137 CU at
  8.9 CU/h, above both limits before overhead. Hence: LS inputs restored, not re-staged (saves ≈ 1.5 G4-h per
  training VM); arm F1 runs share a GPU under CUDA MPS (F1 is latency-bound at batch 1); C_k decode moves to the local
  RTX 2060 (≈ 35 s per core-model, about 10 h) if the CU projection exceeds 112 at that point.
- **VM roles:** two S VMs (SSL + new C_k), one F VM (Danish staging, the 45 A_k^f, decode).
- **Storage:** `archive_gate_v2.py check` and `storage_guard.py` (20 GB reserve) around every substantial launch.

## 11. Forecasts (before any output)

| Event | P |
|---|---:|
| Gate PASS | 0.35 |
| Δ (all) > 0 as a point estimate | 0.70 |
| Δ ≥ +2.0 points | 0.25 |
| Δ ≥ +5.0 points | 0.04 |
| A guard fails (lower ≤ −2) | 0.25 |
| INCOMPLETE (ceiling or implementation) | 0.15 |

## 12. Exposure and reserved terrain

DK-JW was exposed by CONFIRM-D; this stage reads nothing else. The reserved Danish regions [names redacted] and the
sealed Dutch TEST are not read. No other DHM tile is fetched.

## 13. Codex pre-freeze review (`work/dk_regsup1_review_2026-10-29_codex/REVIEW.md`)

1. Split audit: §3 (memberships pinned; 10 km separation enforced on training pixels, products and labels; one
   out-of-fold set per seed; K = 15).
2. Control reuse: §5 (plan identity 9/9, LS byte identity, step-0 gate, replay diagnostic).
3–5. Stage-B points (operating system definition, off-register retention, reading a null) belong to the stage-B
   FREEZE and are carried there. The IGNORE precedence point is settled in §4.

## 14. Pins (sha256)

| Pin | File | sha256 |
|---|---|---|
| Fold split | `rs1_folds.py` | `d69ff6fd5c6f87f51c798598ced44b3efd7efb94e927b2ced560c26b7869909c` |
| FF fetch | `rs1_ff.py` | `601000a31d5e26cd5a6831b715ea4301540a6443692ae435492f431541420da1` |
| Danish FIT plan | `rs1_fit.py` | `1ed5e3c4ebab770463c9aa89fdecf2cd053265d651e6b42c1e690e17d388edaf` |
| Danish staging | `rs1_stage.py` | `710239a9548595ed9a6dc651e7c4775d9b5d894db4f7a8810ce1f281fdc476d2` |
| Arm-A trainer | `rs1_train.py` | `d4b6b38c079b651e4a23254863e590c916cf6a8858a315c4a6e03b2cab0593ab` |
| New-seed wrapper (SSL, C_k) | `rs1_seeds.py` | `6c3a3cfe0a89bdfef419e180a72838053b7039cfa34d5775012e3d55790a1356` |
| Scorer + smoke + assemble | `rs1_score.py` | `43f43fe1bfa93937a04ff1a9a4a323bba75d73f5684345aaf36e21961e24f1b0` |
| Independent recount | `rs1_check.py` | `6fca2575e7101a86bf47b2a4efeff96ae2fa11a42a229362a574f43ab2139c8f` |
| Power | `rs1_power.py` | `0614caebc42e311af34f2b86789a949e02790ed0a9a58a5d282e4bf2a5d68142` |
| Plan checks | `rs1_plancheck.py` | `78340f56223253dca97268fd480d2573c30e45eb6da58892cc0fa52fec35cfe2` |
| LS byte check | `rs1_lscheck.py` | `ee9b6618fb89669f7879939ee6ec79eeccecc32b075bc89c77cdcbde21933458` |
| Drive restores | `rs1_restore.py` | `6059e734cad31a372f10dde0b1477096d2fd4b4c95bd44a03a1417cf84c3c2af` |
| VM job (operational) | `vm/rs1_job.py` | `34d67e14024c5a6e72a89eeb35d7021af94d1ae926974decb37c287a8ac5d2b7` |
| VM driver (operational) | `vm/rs1_vm.sh` | `9d96b8c7c5ac2bdd4c9b1f12faffbede74a0f15741906fd615172784a8c01098` |
| VM bundle (operational) | `vm/rs1_bundle.py` | `555c09a3717b32f7de181c178116ef38fdd685af76dbf496d7ca29f1317111f6` |
| FF receipt | `FF_RECEIPT.json` | `8f4a69f609fabfa3bc4fd2ff8ac7a73b0a2a2bb9dda01a13dba5b16ee7d49c3d` |
| Fold receipt (public) | `FOLDS.json` | `d378bd4ae4c80dddcb3cef4007e1bd40bfbbe62cf7af96dc1f993db77efdc10f` |
| Fit plan (public) | `FIT_PLAN.json` | `c65280b54acc4a8c438e68fb0e9dbcf69f50a4d79409232f7894d1d71969598d` |
| Power | `POWER.json` | `ce7b83ff300f65450a7e2a1456c9a08a5171a6f84aa5274f28f4c1eb629dfe9c` |
| Plan checks | `PLAN_CHECK.json` | `5c83d4808087e1caebca07941baf6bdd0b9f9e4bb39d36081f0a84878099cec8` |
| LS byte check | `LS_CHECK.json` | `e36d9298578b8f378f6d681f09a67cbc8276aa4cfbf9b41118dcd6308c447d0f` |
| Folds (private) | `FOLDS_PRIVATE.json` | `93b0a7f10403ca3f419730ad43a3730b637eab967106b52e2c404d0ceba804da` |
| LS manifest (private) | `LS_MANIFEST_PRIVATE.json` | `612882e396a43fed4cb5675b915817d0bfc722967868d2d7231ba7963de4e456` |
| LS bundle tar (private) | `LS_FIT_BUNDLE.tar` | `76b49c66c2f6e5018598dd0369c9afbd28f1fde336e653caba70ab7c74c972b2` |
| Danish FIT table fold 0 (private) | `FIT_DK_F0_PRIVATE.json` | `5c7df724771e7243f89502e762ffa7d068e06ae517f9b000c0ecea8bc7b38351` |
| Danish FIT table fold 1 (private) | `FIT_DK_F1_PRIVATE.json` | `f833de2883c5219251114b251b911902d7a4b273c11f16aa262c67fd655e0351` |
| Danish FIT table fold 2 (private) | `FIT_DK_F2_PRIVATE.json` | `28a3cca90f67a8d49843edfd94a8757ba898f5dbb5f6de749fd9d2e069076417` |
| Reused, unchanged: dt1_train.py | `work/dense_terrain1_2026-10-25/dt1_train.py` | `f241d7151d6a2e61925be971161c6b66f9ed11e3baaefcb064421c3195168bdd` |
| Reused, unchanged: dt1_ssl.py | `work/dense_terrain1_2026-10-25/dt1_ssl.py` | `c801d2ffa923e2215dd2243cee67bba8d63c8341096110b06c5fefa6379e2c3c` |
| Reused, unchanged: dt1_model.py | `work/dense_terrain1_2026-10-25/dt1_model.py` | `9e3fdbcee88b762cf9d5a9a61531ee0d149df9574dd35132184a86087d083ddd` |
| Reused, unchanged: dt1_channels.py | `work/dense_terrain1_2026-10-25/dt1_channels.py` | `0783d1e97f2111c846c381be3abaabadc05d21836ee58060bdfd90e7ab6f0181` |
| Reused, unchanged: dt1_decode.py | `work/dense_terrain1_2026-10-25/dt1_decode.py` | `af88392ab734fc4959da470ee1ae93ef1d1153e2bdae408fdb2b6cd3ec47bb40` |
| Reused, unchanged: d1_train.py | `work/d1_collapse_2026-10-06/d1_train.py` | `2e232c7759f718f77c5a97443e7b8c6784904f8785d5ae27dc4c11159c3c804e` |
| Reused, unchanged: masks.py (HR1) | `work/holdout_region_2026-10/chain/masks.py` | `bf51105952cc5388f91bda8e1182e4ad5817c6113fde3178ea2dba9ac4f8d1ca` |
| Reused, unchanged: nds_schedule.py | `work/adapt_nds_2026-10-05/nds_schedule.py` | `c85101efdb328fc20912aafbe1766fdf98248f108673ba0d515f9a823edec8dd` |
| Reused, unchanged: cdb_score.py | `work/confirm_d_stage_b_2026-10-26/cdb_score.py` | `c7ae85109a60e11e0ed75055712fe600772d1bab8e0f92a49b58dfe5c8c6aa09` |
| Reused, unchanged: cd_analysis.py | `work/confirm_d_2026-10-25/cd_analysis.py` | `10db9421f6d6a8b52d46b42f54e0a9076d128abe775d9d717298eda892607a69` |
| Reused, unchanged: dk_score.py | `work/confirm_d_2026-10-25/dk_score.py` | `8ae364338e0c14f7881aa8cdbb8bbcdd912488b9fab50b5eedd9811a7aed1117` |
| Reused, unchanged: dk_a1.py | `work/confirm_d_2026-10-25/dk_a1.py` | `c60884b6217f85c48892b4117b9de57893d2379412e2a7c7e0b2d2d5fd04743a` |
| Reused, unchanged: cr1_score.py | `work/context_rescue1_2026-10-23/cr1_score.py` | `3340998668c79d7388ba33d8de499d788a7680ec6a9546a896528b42209b4c31` |
| Reused, unchanged: cr1_analysis.py | `work/context_rescue1_2026-10-23/cr1_analysis.py` | `7ac8cc484066df785752569de9c3605a4163a1ae3a44b2ae11715d80db05bfac` |
| Reused, unchanged: REACH_DECISION.json | `work/confirm_d_2026-10-25/REACH_DECISION.json` | `382efe6638176a575e739b6176acb0df24a385389a1b519194c10dadbffcd69a` |
| Reused, unchanged: DK_RECORDS_PRIVATE.json | `outputs/confirm_d_2026-10-25_PRIVATE/DK_RECORDS_PRIVATE.json` | `7a581c9177c6cb0a84b00ac25449f130bb64d7e8dba60a41ac31554721e7fc53` |
| Reused, unchanged: PREP_JW.json | `outputs/confirm_d_2026-10-25_PRIVATE/jw/PREP_JW.json` | `5afae7e2a28f4c564d5342d5eb86b92dd6ed9c29f2217056bdd19231e05552e2` |
| Reused, unchanged: RESULT_B_K9.json (smoke reference) | `work/confirm_d_stage_b_2026-10-26/RESULT_B_K9.json` | `4c87a5947fffa1c3c8cbc8ace63254ae250d151f4a2a998509905ef0ca7f1dc9` |
