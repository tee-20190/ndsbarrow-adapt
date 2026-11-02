# CONFIRM-D freeze: do D-based detectors beat the incumbent on fresh Danish barrows?

> **FROZEN 2026-10-25** by the coordinator under the owner's standing execution authority (AGENTS.md, 2026-09-24).
> Preconditions met: DK-1 resolved (`DHM_PROBE_RECEIPT.json`); DK-2 accepted as a caveat (`DK2_LICENCE_NOTE.md`); the
> code pins in §12 are filled; this file is committed **before any DHM tile of a DK-JW or DK-FUNEN-DEV core is read**
> (only non-cohort test boxes in Copenhagen and Roskilde were read, to build and test `dk_stage.py`).
> `dk_stage.py cores` refuses to run unless this file is committed and unchanged. Any later change is an addendum
> with its reason, made before the output it could affect.

**Capability line.** Reproducible detector improvement, confirmed prospectively on terrain no project detector has
read. It also extends D1-family scope to a third NW European sand landscape.

**Authority.** Owner standing execution authority (AGENTS.md, 2026-09-24) covers this: bounded Colab inference and
training, licensed public-data downloads, local evaluation. Exceptions: the DHM account (DK-1) is an owner action, and
the FF geometry licence (DK-2) is for the coordinator to accept or query.

## 1. Questions (Codex correction 1; MERGE-1)

| ID | Question | Estimand | Role |
|---|---|---|---|
| **A1** | Does the frozen system **S_D = interleave(D_101, D_202) + PROX** beat the incumbent **D1_05 + PROX**? | Fixed weights; geography random | **Primary** |
| **A2** | Does S_D beat **S_D1 = interleave(D1_05, D1_06) + PROX**, a D1 two-seed ensemble at equal decode cost? | Fixed weights; geography random | Key secondary (fixed sequence after A1) |
| **B** | Is the D recipe (four channels, dense SSL 4,096, F1) better than the D1 recipe (legacy, F1) over training seeds? | Seeds and geography random | Secondary, stage B, separate claim |

**Scope of the claims.**
- A1 and A2 concern those exact checkpoints, decoders and policies. They say nothing about new training runs.
- B concerns recipes. Its average single-model recall is not an ensemble output and must not be advertised as one.

## 2. Terrain and labels (CANDIDATES.md)

### Register
- Fund og Fortidsminder `punkt_fredet` snapshot `a97beee0…f47c8` and `areal_fredet` snapshot `1e98d181…0361`.
- Cohort: Rundhøj, deduplicated at 2 m.
- Size: protected-extent polygon area; small < 200 m².
- Groups: isolated = 200 m group of fewer than 3; grouped = 3 or more.
- Strong-grouped = grouped and H ≥ 2T.
  - T = `cr1_analysis.T_FAINT` (0.329 m).
  - H is measured by E0's dome-height routine on the DK-JW 1 m mosaics, before any detector output.
  - If more than 20 % of grouped records have no measurable H, the guard uses "grouped" instead. This is decided
    in an addendum before any decode.

### Cohorts
- **DK-JW (confirmation):** 171 cores, 5,273 records. Core list `DK_COHORT_PRIVATE.json` `5e3c8ec4…a819`, built by
  `dk_cohort.py` from `DK_CORES_PRIVATE.json` `35f53a53…ed`.
- **DK-FUNEN-DEV (development):** 15 cores, 242 records, at least 30 km from DK-JW. Used only for §5. No D system is
  scored there.

### Terrain
- DHM/Terræn 0.4 m (SDFI/Klimadatastyrelsen free-data terms, attribution required), WCS 1.0.0
  `wcs.datafordeler.dk/DHMNedboer/dhm_wcs/1.0.0/WCS`, coverage `dhm_terraen`, owner's Datafordeler key (DK-1 resolved
  2026-10-25, `DHM_PROBE_RECEIPT.json`). Fetched by `dk_stage.py` in ≤ 1,000 m chunks on the native grid.
- Each mosaic is the core plus a 256 m halo. It is resampled once by GDAL `average` to a **1 m grid on integer UTM 32N
  metres**, so every arm meets the same 1 m source geometry as Lower Saxony DGM1. Chunk edges are integer metres, so
  the per-chunk average equals one whole-mosaic warp (selftest).
- **Sea (added 2026-10-25 from non-cohort test boxes only).** DHM stores sea as exactly 0.0 m, not nodata (98 % of an
  Øresund test box). Exact-0.0 8-connected components of ≥ 1 ha become NaN before the fill, so water meets the
  OPEN-TRANSFER AHN regime (nodata) and budgets count land only.
- Then OPEN-TRANSFER's `fill_gaps`: inverse-distance fill of no-data within 20 m; cells farther away stay NaN.
- **Budget land** is the finite-DTM area of each core after the sea mask and the fill. B_core = round(2 × land km²).

### Rules
- No sealed or reserved terrain.
- No 738-pool core, and no Dutch, Swedish, Norwegian or English terrain.
- Nothing read outside the listed cores plus their halos.

## 3. Systems (pinned)

| System | Members (sha256) | Construction |
|---|---|---|
| S_D | D_101 `5840d9aa…cd6a8`, D_202 `f5c39c60…c407` (DETECTOR_REGISTRY DT1D-F1_BARROW_L_101/202) | MERGE-1 `merged(D_101, D_202)` (`merge1.py` `ee4074b4…`): per core, rank_key order, interleave with D_101 first, 15 m dedup; then PROX |
| Incumbent | D1_05 `d9ece1fb…501e` | single model, rank_key; PROX |
| S_D1 | D1_05 `d9ece1fb…`, D1_06 `f2916949…0bade` | `merged(D1_05, D1_06)`; PROX |
| Descriptive | D_303, D1_07 (`d7052046…3511`), single D_101, single D_202, public ADAF barrow `51b679fa…` | as above |

**Why DD(101, 202).** It was chosen now, on DT1 development numbers only, by a leximin rule: sort each candidate's
DT1 contrasts against D1_05 + PROX over {all, large, isolated, strong-grouped} in ascending order, then compare the
sorted vectors lexicographically.

| Candidate | Sorted contrasts |
|---|---|
| **DD(101, 202)** | (+0.16, +1.28, +2.18, +3.52) |
| DD(202, 303) | (+0.16, +1.09, +1.39, +3.88) |
| DD(303, 101) | (+0.00, +0.79, +1.88, +3.10) |

S_D1 contains the incumbent checkpoint and is MERGE-1's pair 1. No system is changed after any Danish output.

**Decoders.**
- D members use DT1's four-channel adapter: channels from §2's 1 m grid, 97 m filter halo, adapter pins as DT1
  Addendum P.
- D1 and ADAF use the frozen three-channel route, with the legacy chain bitwise as DT1 §3.1.
- Common settings: 1,024 px tiles with 64 px halo; threshold 0.5; area > 30 m²; roundness > 0.4; no NMS; centroid in
  core.

## 4. Scoring (as CR-1)

- `cr1_score.score` unchanged: core assignment, S-v1 strict mask, plain top-B, PROX(B) with R 200 m and f 0.3,
  15 m one-to-one maximum matching.
- The S-v1 mask uses `cr1_score.mask_rule` on an Overpass snapshot per core. The snapshot is taken with
  `work/ls_pool2_2026-10-25/overpass_batch.py` and hashed before any decode.
- Budgets come from §2. They are never pooled. Each system's kept counts and underfilled cores are recorded.
- Primary workload: 2/km² + PROX. Plain 2/km² and plain 20/km² are descriptive.

## 5. Development slice: reach rule (fixed before any DK-JW decode)

- On DK-FUNEN-DEV, decode **D1_05 and public ADAF only**. The parity gate (§10) is run there as well.
- Compute D1_05 + PROX recall at 15 m (R15) and at 25 m (R25).
- **If R15 / R25 < 0.85, the primary reach for every contrast becomes 25 m. Otherwise it stays 15 m.**
- The ratio and the decision are committed before any DK-JW tile is decoded.
- Recall at 10, 15 and 25 m is reported descriptively in every case.

## 6. Statistics

**A1 and A2 (fixed systems).**
- Δ = (hits_X − hits_Y) / N on the cohort.
- se²_core: the `cr1_analysis.contrast` ratio estimator over 5 km cores, with K = 1. **No seed term**: the weights
  are frozen.
- 95 % interval Δ ± 1.96 √se²_core.
- Implemented in a new `cd_analysis.fixed_contrast`, pinned in §12. It calls the unchanged ratio estimator and
  bypasses only `SEED_FLOOR`. Labels: `cd_analysis.a1_label` / `guard_label`.
- Sensitivity, descriptive: 10 km block clusters.

**B (recipe).**
- Pairs k = (D_k, A_k), k ∈ {101, 202, 303, 404, 505, 606, 707, 808, 909}.
  - A_k is the D1 recipe: legacy channel, ADAF barrow start, F1.
  - D_k and A_k share the F1 plan `match_train.schedule(k, …)`. Plan digests must be identical within a pair, as in
    DT1.
- Δ = mean_k Δ_k on the cohort.
- var_seed = max(var(Δ_k, ddof 1) / K, **2 × (1.3 pts)² / K**) (the SCALED floor; POWER.md §2.3).
- 95 % interval Δ ± 1.96 √(se²_core + var_seed).
- Also reported:
  - the DT1-code constant-floor interval;
  - D(9) − D1 R0(05/06/07) by index, descriptive.

## 7. Decision rules

**A1 (primary).**
- **PASS** iff the lower bound > 0, all records, 2/km² + PROX, primary reach (§5).
- **Guards:** S_D − incumbent on large, isolated and strong-grouped records.
  - PASS iff the lower bound > −2 points.
  - HARM iff the upper bound < −2.
  - Otherwise UNRESOLVED (not proof of harm).

**A2.** Tested only if A1 passes (fixed sequence, family-wise α 0.05). PASS iff lower bound > 0. Same guards.

**B.**
- PASS iff the lower bound > 0 (SCALED floor). Guards as A1.
- B is a separate claim and does not gate A.

**Outcome labels for A1.**

| Label | Condition |
|---|---|
| **CONFIRMED** | PASS, no guard HARM |
| **CONFIRMED_GUARD_UNRESOLVED** | PASS, at least one guard UNRESOLVED |
| **PASS_GUARD_HARM** | PASS, at least one guard HARM (reported as a trade-off, not a confirmation) |
| **NOT_DISTINGUISHED** | interval contains 0 |
| **REVERSED** | upper bound < 0 |
| **INCONCLUSIVE_IMPLEMENTATION** | parity, coverage or a pin fails |

A positive smaller gain and a breakthrough (+5 or more) remain separately reported outcomes.

**What changes on CONFIRMED.**
- S_D becomes the evidence-backed candidate operating system for NW European sand-lowland barrows (LS, NL, DK
  scope). A decode costs two models, so a deployment is twice the decode cost of D1_05.
- DETECTOR_HISTORY and the registry are updated by their writer.
- Lower Saxony and Dutch operating choices change only by a separate adoption record.

## 8. Descriptive reports (no decision)

- Recall for every system at 1, 2, 5 and 20/km², plain and PROX.
- Strata: small, large, size unknown, isolated, grouped, strong-grouped.
- Reach at 10, 15 and 25 m.
- Kept counts and underfilled cores.
- Per-core contrast map, aggregated (no coordinates).
- Projection check: the DT1 size-reweighted forecasts (POWER.md §5.3) against the observed values.

## 9. Resources (from DT1 measured rates; 48 VM-h ceiling)

### Rates
- SSL: 0.753 s per update; 4,096 updates = 0.857 h.
- F1: 338 s.
- Decode: **5.2 s per core-model** of wall time on one G4 with 8 workers. Source: DT1 A/B phase 0, 7,583 s for
  7 models × 210 cores; DT1 `FORECAST_AB` `core_pass_s_7models` 288.9 s ÷ 8 workers is about 36 s per core for a
  7-model pass, consistent with the brief's "about 38 s/core".
- Pessimistic: 2× for decode and staging.
- Four-channel prep: 20 s per core.
- DHM download (measured 2026-10-25 on non-cohort boxes, local link): 25 MB per 1 km² chunk; one stream
  0.35–0.57 MB/s; 8 concurrent streams with starts ≥ 1 s apart 2.7 MB/s. A core is 36 chunks (30.4 km², ~0.9 GB raw);
  186 cores ≈ 167 GB raw ≈ 17 h locally at 2.7 MB/s; DEV (15 cores) ≈ 1.4 h. 1 m output ≈ 2.3 MB/km² deflate,
  ≈ 70 MB per core, ≈ 13 GB for all cores.

### Stage A (A1, A2 and descriptive): no training

| Item | Central VM-h | Pessimistic VM-h |
|---|---:|---:|
| DHM staging (171 + 15 cores) + 1 m resample + fill | 3.5 | 6.5 |
| Four-channel prep | 1.0 | 1.0 |
| Decode 7 models × 171 cores (D_101, D_202, D_303, D1_05, D1_06, D1_07, ADAF) | 1.7 | 3.5 |
| DEV slice (2 models × 15 cores + parity) | 1.0 | 1.5 |
| Setup, bundles, gates, read-back | 2.0 | 3.0 |
| **Stage A** | **9.2** | **15.5** |

### Stage B (recipe, K = 9)

| Item | Central VM-h | Pessimistic VM-h |
|---|---:|---:|
| 6 new D: SSL 4,096 + F1 = 6 × 0.951 h | 5.7 | 5.7 |
| 6 new A: F1 = 6 × 0.094 h | 0.6 | 0.6 |
| FIT/halo staging (reuse DT1 bundle if retained; else LGLN at 1 request/s) + env gates | 2.0 | 3.0 |
| Decode 15 models (6 new D + 9 A) × 171 cores | 3.7 | 7.4 |
| Overhead | 1.0 | 2.0 |
| **Stage B** | **13.0** | **18.7** |

### Totals
- **Whole job: 22.2 VM-h central, 34.2 VM-h pessimistic. Ceiling 48.**
- About 200–305 CU at DT1's 8.9 CU/VM-h.
- Elapsed about 12–18 h: two training workers in parallel with the evaluation VM, at most 4 job workers, and at most 6
  Colab machines in total.

### Variants
- **K = 6 stage B** (three new D and three new A): about 15 VM-h total with stage A, central.
- **K = 16** (13 new D): about 31 VM-h central, about 48 pessimistic. Not recommended: power at +1.3 is still only
  0.50–0.82.
- **Fallback:** if a mid-run forecast exceeds 48 VM-h, stage B drops to K = 6 before any stage-B output is scored.
  Stage A always completes first.

### Storage
- `archive_gate_v2 check` and `storage_guard` before every launch.
- **Local:** stage A writes proposals only (under 1 GB). Stage B adds about 5 GB: six SSL encoders of 0.32 GB and
  twelve F1 weights of 0.26 GB. C: had **26 GB free** at 16:40Z, against a need of about 25 GB including the 20 GB
  reserve. Archive DT1's B and G F1 weights and G SSL encoders (about 2.5 GB, RESULT.md archive candidates) first, or
  keep the new SSL encoders on Drive.
- **VM:** stream DHM per 10 km block. Keep only the 1 m mosaics (about 21 GB). Peak about 40 GB.

### Request limits
- DHM and Overpass: at most 1 request/s; stop on HTTP 429.

## 10. Gates (hard stops)

1. Freeze commit, then the reach decision commit (§5), before any DK-JW decode.
2. **Parity on DK-FUNEN-DEV:**
   - The four-channel adapter reproduces the frozen three-channel route bitwise for D1_05 on 2 DEV cores.
   - D_101 reproduces its DT1 proposals on 2 LS gate cores, as in DT1 Addendum E.
3. **Coverage:** every DK-JW core's mosaic is complete after the fill except true no-data; this is reported.
4. **Stage B environment:** D1's step-0 rule on each training VM. SSL point and view manifests hashed per seed. Plan
   digests identical within each pair.

## 11. Forecasts (designer, before any Danish data)

| Event | P |
|---|---:|
| A1 PASS | 0.70 |
| A1 CONFIRMED (no guard HARM, all guards PASS) | 0.50 |
| A2 PASS (given A1 PASS) | 0.45 |
| B PASS, K = 9 | 0.20 |
| Reach switched to 25 m (§5) | 0.30 |
| Implementation or access failure before scores | 0.20 |

## 12. Pins (sha256)

| Pin | File | sha256 |
|---|---|---|
| Statistics (selftest PASS; reproduces DT1 D − R0 PROX and plain to 1e-12 from its private hits) | `cd_analysis.py` | `10db9421f6d6a8b52d46b42f54e0a9076d128abe775d9d717298eda892607a69` |
| DK staging, resample, sea mask, fill (offline selftest PASS; non-cohort boxes) | `dk_stage.py` | `db90e6393b11b85e624609b5e3017b2745f8e13afa2173efa8de3a2cf16def22` |
| S-v1 Overpass snapshots (unchanged `cr1_score.core_query`, `cr1_e0.fetch`) | `dk_osm.py` | `e8dde02e045288cc2e5558ed86d3478872339a87fa9f8c3ea95d72bfe59ae487` |
| E0 H port (`inventory_b_faint.dome_max` unchanged; window equality selftest PASS) | `dk_e0h.py` | `25c0e8055fe2a0e3bf2e7bd516ae2cd023fdfdc818448fb17ae236c29c86908e` |
| Stage-B launch notes | `STAGE_B_NOTES.md` | `fb8deb34b1a9366d9814f81cd0f03d0de37cb71c89a544eeab30456123f7ce70` |
| Cohort builder / core list | `dk_cohort.py`, `DK_COHORT_PRIVATE.json` | `18374177…2c`, `5e3c8ec4…a819` |
| Register records | `DK_RECORDS_PRIVATE.json` | `7a581c91…fc53` |
| DHM access probe receipt | `DHM_PROBE_RECEIPT.json` | `b4af4b2c…f345` |
| MERGE-1 | `work/merge1_2026-10-25/merge1.py` | `ee4074b4…a0` |
| Scorer | `work/context_rescue1_2026-10-23/cr1_score.py` | `33409986…4c31` |
| Ratio estimator | `work/context_rescue1_2026-10-23/cr1_analysis.py` | `7ac8cc48…bfac` |

**Overpass manifest gate.** The snapshot run (`dk_osm.py`) started before this commit and touches OSM only. Its
`OSM_MANIFEST.json` (public counts plus the private manifest's sha256) is committed **before any decode**; every scored
mask is built from those snapshot files.

**Staging location (CD-5).** DK-FUNEN-DEV stages locally. DK-JW stages in batches that are archived to Drive
(fastarch, verified) before the next batch, or on a VM; either route uses `dk_stage.py` unchanged.

## 13. Exposure

- After this one read, DK-JW is exposed development terrain for every system decoded on it.
- DK-FUNEN-DEV is exposed for D1_05 and ADAF only.
- The other Danish regions stay unread. They are reserved for a later confirmation or for discovery [region names
  and record counts redacted for this release; see paper_a/REDACTIONS.md].

## Blockers (detail in CANDIDATES.md)

| # | Status | Summary |
|---|---|---|
| DK-1 | **Resolved 2026-10-25** | Owner's Datafordeler key; probe 200 GeoTIFF on the new host (old dataforsyningen host 403s). |
| DK-2 | Caveat (accepted) | CC0 per SLKS 2018 statement (`DK2_LICENCE_NOTE.md`); evaluation use, credit, no redistribution. |
| DK-3 | Workaround | DAWA kommune API gone (HTTP 410). Finite-DTM land after the sea mask is used instead. |
| CD-1 | Open | FF absolute positional accuracy unknown. Handled by the §5 reach rule. |
| CD-3 | **Measured** | 2.7 MB/s with 8 streams locally: ≈ 17 h for all 186 cores (§9). |
| CD-4 | Resolved | DT1 A/B/G weights archived (3.81 GB); C: 31.6 GB at 16:52Z. |
| CD-5 | Open | 13 GB of 1 m mosaics + 20 GB reserve exceeds local C: (~30 GB). DEV stages locally; DK-JW is staged in batches that are archived to Drive (fastarch) before the next, or on a VM. |
