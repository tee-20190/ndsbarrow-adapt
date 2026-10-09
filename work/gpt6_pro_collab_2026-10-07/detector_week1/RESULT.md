# Detector week 1: miss decomposition across all models, protocol-vs-scorer audit, benchmark v0 sensitivity (2026-10-07)

CPU only, read-only. Nothing in the project was edited, and no inference was run. The work uses saved proposals and
frozen scorers (imported unchanged). Outputs are aggregate counts only: no part ids, no coordinates. Runtime: WSL
`/home/tee-/adaf-h2h-venv/bin/python -B`. That is the venv `region_score.py:21-22` names; `gap_region.py` imports
that scorer.

| File | Content |
|---|---|
| `decomp.py` → `UNION_DECOMP_ALL.json` | Task 1. Imports `work/gap_analysis_2026-10-03/gap_region.py` unchanged (`part_flags`, `load_objects`, `miss_kind`). Adds the any-model union and the oracle ceilings. Input sha256 per model is in the JSON. |
| `tables.py` | Prints the Task 1 tables from the JSON. |
| `ignore_check.py` → `IGNORE_CHECK.json`, `ignore_types.py` | Task 2 side check on IGNORE handling, with types counted. |
| `bench_v0.py` → `BENCH_V0.json` | Task 3. |

## Task 1: MISSING / NMS / RANK for every model on the open development sets

**Scorer, every set.** The frozen HR1 `region_score` (`work/holdout_region_2026-10/step2/region_score.py:157-169`) and
the frozen `score_holdout.recall_at` (`work/tta_ensemble_2026-09-24/score_holdout.py:73-89`), used as
`gap_region.py:35-70` uses them:

- **Reach.** 15 m around the part footprint (`region_score.py:152`).
- **NMS.** Greedy global 60 m, score descending (`work/next_experiment_2026-09-23/r3_curation.py:164-174`).
- **Budget.** The first background-tile proposal beyond floor(d × background km²) ends the list. Proposals in positive tiles are never charged.

England uses `rs.english_manifest()` (`region_score.py:224-238`). That is the frozen English holdout convention:
overlapping windows, and a background area summed over windows rather than unioned. The Verden sets use the Step 0
group and individual manifests and the rebound proposals that `score_fresh.py:135-143` wrote.

**Kinds** (`gap_region.py:4-7`). Each cell below is HIT / RANK / NMS / MISSING:

- **HIT:** found within budget d.
- **RANK:** a post-NMS proposal reaches the part but ranks below the budget.
- **NMS:** pre-NMS proposals reached the part and NMS removed all of them.
- **MISSING:** no in-core proposal reaches the part.

The last column, "top-K all", is the number of parts the model's own frozen pool hits within its first
K = floor(d × scored km²) proposals when every proposal is charged.

**Parity with GAP_TABLE.md (pass).** Running `gap_region` on the inputs the gap analysis used reproduces every count
in `work/gap_analysis_2026-10-03/GAP_TABLE.md`:

- England AO 292/52/155 and CONT 370/68/64; the NMS counts are 7 and 4, as in `SUMMARY_ENGLAND_HOLDOUT.json`.
- Wales enclosures: AO 47/6/24, CONT 27/37/13.
- Ireland: AO 83/13/64, CONT 38/75/47.
- Verden groups at 0.5: ADAF barrow hit 0.507 / rank 0.077 / missing 0.417; S\* 0.397 / 0.303 / 0.300 (`work/fresh_region_2026-10-03/RESULT.md:30-48`).

The England JOINT-1 `refs_holdout` re-inference files have different bytes from the gap-analysis inputs: AO
`a322bb21…` against `ee81259f…`, and CONT `46eb9eff…` against `3fac0cf8…#CONT`. They give identical counts.
S\* is J_2026092803, the FRESH-1 freeze `sstar` (`work/fresh_region_2026-10-03/step2/freeze/FREEZE.json`). Its
England, Wales and Ireland proposals come from JOINT-1 `run_s4`.

### English TRAIN holdout (506 parts: 435 barrows, 71 enclosures; background 33.38 km², summed; scored union 93.85 km²)

| Model | d = 0.5 | d = 1 | d = 2 | Pre-NMS parts | Background/km² | Top-K all 0.5 / 1 / 2 |
|---|---|---|---|---:|---:|---|
| ADAF AO | 292 / 52 / 7 / 155 | 328 / 16 / 7 / 155 | 342 / 2 / 7 / 155 | 351 | 2.13 | 37 / 77 / 154 |
| ADAF barrow | 262 / 76 / 7 / 161 | 311 / 27 / 7 / 161 | 338 / 0 / 7 / 161 | 345 | 1.71 | 29 / 69 / 151 |
| ADAF enclosure | 97 / 123 / 3 / 283 | 152 / 68 / 3 / 283 | 206 / 14 / 3 / 283 | 223 | 2.76 | 29 / 55 / 119 |
| ADAF ringfort | 116 / 70 / 3 / 317 | 160 / 26 / 3 / 317 | 184 / 2 / 3 / 317 | 189 | 2.10 | 34 / 65 / 117 |
| CONT e41 | 370 / 68 / 4 / 64 | 392 / 46 / 4 / 64 | 405 / 33 / 4 / 64 | 442 | 10.99 | 45 / 89 / 163 |
| S\* (J_03) | 358 / 77 / 4 / 67 | 378 / 57 / 4 / 67 | 409 / 26 / 4 / 67 | 439 | 8.09 | 42 / 81 / 161 |

- **Union of six models, pre-NMS:** 466 / 506 (0.921): 405 barrows, 61 enclosures. Post-NMS: 465. No model reaches 40 parts.
- **Any model HIT, each at its own budget:** 402 / 424 / 442 at d = 0.5 / 1 / 2. The best single model, CONT, reaches 370 / 392 / 405.

### Wales HR1 DEV (188: 111 barrows, 77 enclosures; background 300 km²; scored 453.1 km²)

| Model | d = 0.5 | d = 1 | d = 2 | Pre-NMS | Background/km² | Top-K all |
|---|---|---|---|---:|---:|---|
| ADAF AO | 125 / 11 / 1 / 51 | 132 / 4 / 1 / 51 | 136 / 0 / 1 / 51 | 137 | 1.40 | 100 / 128 / 136 |
| ADAF barrow | 87 / 19 / 3 / 79 | 100 / 6 / 3 / 79 | 106 / 0 / 3 / 79 | 109 | 1.60 | 73 / 90 / 106 |
| ADAF enclosure | 58 / 27 / 1 / 102 | 71 / 14 / 1 / 102 | 85 / 0 / 1 / 102 | 86 | 1.70 | 48 / 65 / 85 |
| ADAF ringfort | 56 / 26 / 0 / 106 | 63 / 19 / 0 / 106 | 73 / 9 / 0 / 106 | 82 | 2.85 | 52 / 61 / 72 |
| CONT e41 | 114 / 49 / 0 / 25 | 126 / 37 / 0 / 25 | 139 / 24 / 0 / 25 | 163 | 10.92 | 97 / 114 / 132 |
| S\* | 130 / 35 / 0 / 23 | 146 / 19 / 0 / 23 | 153 / 12 / 0 / 23 | 165 | 8.34 | 113 / 134 / 149 |

- **Union, pre-NMS:** 171 / 188 (0.910): 103 barrows, 68 enclosures. None reach 17.
- **Any model HIT:** 152 / 159 / 168, against S\* 130 / 146 / 153.
- **Enclosures at 0.5** (HIT / RANK / NMS / MISSING): ADAF enclosure head 45 / 12 / 0 / 20; ringfort head 46 / 8 / 0 / 23; AO 47 / 6 / 0 / 24; S\* 46 / 18 / 0 / 13.

### Ireland HR3 DEV (160: 9 barrows, 151 enclosures; background 156.25 km²; scored 232.6 km²)

| Model | d = 0.5 | d = 1 | d = 2 | Pre-NMS | Background/km² | Top-K all |
|---|---|---|---|---:|---:|---|
| ADAF AO | 83 / 13 / 0 / 64 | 94 / 2 / 0 / 64 | 96 / 0 / 0 / 64 | 96 | 1.45 | 56 / 79 / 96 |
| ADAF barrow | 32 / 7 / 0 / 121 | 37 / 2 / 0 / 121 | 39 / 0 / 0 / 121 | 39 | 1.24 | 24 / 32 / 39 |
| ADAF enclosure | 80 / 24 / 1 / 55 | 103 / 1 / 1 / 55 | 104 / 0 / 1 / 55 | 105 | 1.74 | 41 / 78 / 103 |
| ADAF ringfort | 86 / 20 / 0 / 54 | 100 / 6 / 0 / 54 | 106 / 0 / 0 / 54 | 106 | 2.28 | 47 / 83 / 104 |
| CONT e41 | 38 / 75 / 0 / 47 | 55 / 58 / 0 / 47 | 90 / 23 / 0 / 47 | 113 | 8.78 | 21 / 42 / 79 |
| S\* | 84 / 33 / 0 / 43 | 94 / 23 / 0 / 43 | 101 / 16 / 0 / 43 | 117 | 7.18 | 52 / 83 / 98 |

- **Union, pre-NMS:** 133 / 160 (0.831). None reach 27.
- **Any model HIT:** 112 / 119 / 126. The best single model is the ringfort head, at 86 / 100 / 106.

### Verden (FRESH-1; 300 barrow groups, 728 individual barrows; background 300.45 km²; scored 422.8 km²)

ADAF's enclosure and ringfort heads were never run on Verden, so there are no saved proposals for them. They are
absent here, not substituted.

| Unit | Model | d = 0.5 | d = 1 | d = 2 | Pre-NMS | Background/km² | Top-K all |
|---|---|---|---|---|---:|---:|---|
| group | ADAF AO | 53 / 77 / 0 / 170 | 86 / 44 / 0 / 170 | 129 / 1 / 0 / 170 | 130 | 2.09 | 39 / 76 / 124 |
| group | ADAF barrow | 152 / 23 / 0 / 125 | 173 / 2 / 0 / 125 | 175 / 0 / 0 / 125 | 175 | 1.05 | 105 / 155 / 175 |
| group | CONT e41 | 94 / 106 / 2 / 98 | 130 / 70 / 2 / 98 | 150 / 50 / 2 / 98 | 202 | 8.60 | 74 / 108 / 140 |
| group | S\* | 119 / 91 / 0 / 90 | 147 / 63 / 0 / 90 | 170 / 40 / 0 / 90 | 210 | 6.60 | 82 / 126 / 155 |
| individual | ADAF AO | 57 / 107 / 28 / 536 | 100 / 64 / 28 / 536 | 163 / 1 / 28 / 536 | 192 | 2.09 | 43 / 81 / 156 |
| individual | ADAF barrow | 201 / 41 / 83 / 403 | 240 / 2 / 83 / 403 | 242 / 0 / 83 / 403 | 325 | 1.05 | 132 / 207 / 242 |
| individual | CONT e41 | 113 / 150 / 20 / 445 | 162 / 101 / 20 / 445 | 188 / 75 / 20 / 445 | 283 | 8.60 | 90 / 129 / 175 |
| individual | S\* | 147 / 140 / 21 / 420 | 182 / 105 / 21 / 420 | 217 / 70 / 21 / 420 | 308 | 6.60 | 104 / 155 / 197 |

- **Groups.** Union pre-NMS 235 / 300 (0.783); none reach 65. Any model HIT 173 / 193 / 203, against ADAF barrow 152 / 173 / 175.
- **Individual barrows.**
  - Union pre-NMS 430 / 728 (0.591); post-NMS 383. None reach 298.
  - Any model HIT 255 / 305 / 326.
  - NMS removes ADAF barrow's only reaching proposal at 83 barrows.

### Oracle ceilings for the union (parts)

- **Frozen charge (background-tile proposals only).** Proposals in positive tiles are free, so an oracle ranker over
  the union reaches the union's pre-NMS coverage at every d.
  - The "greedy NMS-feasible" column picks proposals more than 60 m apart, so the frozen NMS keeps every one. It is a
    lower bound.
  - The "upper" column is the pre-NMS union.
- **All-charged.** K = floor(d × scored km²), and every proposal costs 1.
  - The upper bound is min(union, the K largest per-proposal reach sets). One proposal can reach several parts.
  - The "greedy" columns are lower bounds, with and without the NMS-feasibility constraint.

| Set | Frozen charge, upper / greedy NMS-feasible (every d) | All-charged d = 0.5: K, upper, greedy | d = 1 | d = 2 |
|---|---|---|---|---|
| England (506) | 466 / 466 | 46, 83, 52 | 93, 130, 99 | 187, 224, 193 |
| Wales (188) | 171 / 171 | 226, 171, 171 | 453, 171, 171 | 906, 171, 171 |
| Ireland (160) | 133 / 132 | 116, 133, 121 | 232, 133, 132 (133 without NMS) | 465, 133, 132 (133) |
| Verden groups (300) | 235 / 235 | 211, 211, 211 | 422, 235, 235 | 845, 235, 235 |
| Verden individual (728) | 430 / 348 | 211, 357, 276 (278) | 422, 430, 348 (430) | 845, 430, 348 (430) |

**Reading Task 1**

1. **NMS is negligible on every group- and part-level set** (≤ 7 parts per model). The exception is Verden individual barrows:
   - NMS costs 20–83 barrows per model;
   - an NMS-feasible oracle over all four models' proposals reaches only 348 / 728 (0.478).
2. **Ranking leaves a large margin to a ceiling the current proposals already contain.**
   - The any-model union covers 0.92 (England), 0.91 (Wales), 0.83 (Ireland) and 0.78 (Verden groups).
   - The best single model at d = 0.5 reaches 0.73 / 0.69 / 0.54 / 0.51.
   - "Any model HIT at its own budget", a per-part oracle choice of model, already gives 0.79 / 0.81 / 0.70 / 0.58 at d = 0.5.
   - So a fusion or re-ranker over the saved pools has 9–29 points of headroom at d = 0.5 before any new proposal is needed. This is diagnostic only: the oracle uses labels.
3. **The specialist heads add coverage mainly where the generalists are weakest.**
   - Ireland: the ringfort and enclosure heads have 54–55 MISSING against AO's 64.
   - Verden groups: the barrow head has 125 MISSING against AO's 170.
   - In England and Wales, CONT and S\* already cover nearly the whole union: 442 and 165 of 466 and 171.
4. **Residual coverage loss** (no model proposes near the part) is 40 (England), 17 (Wales), 27 (Ireland) and 65 (Verden groups). The cause is not separable without probability maps.
5. **The two charge conventions answer different questions.**
   - Under the frozen convention, ceilings do not depend on d: positive-core proposals are free.
   - Under all-charged budgets, the dense English holdout is budget-bound at every d. K is 46–187 for 506 parts.

## Task 2: registered definition against scorer code

| Element | (a) England VALIDATION, CONT vs ADAF AO | (b) ADAPT-NDS-1, Oldenburg CONFIRM | (c) FRESH-1, first use of Verden |
|---|---|---|---|
| Registered unit | NHLE common-class part centroids, all 265 parts owned by the 45 blocks, with no visibility filter (`work/validation_cont_2026-09-26/PACKET.md:72,187`; `PACKET_AMENDMENT_B.md:23,34`) | Barrow group, using FRESH-1's 60 m groups (`work/adapt_nds_2026-10-05/PACKET.md:26-28,55`) | Barrow group: register Grabhügel with centroids ≤ 60 m apart form one MultiPolygon part; individual barrows are secondary (`work/fresh_region_2026-10-03/FRESH1_PREPIXEL.md:29,112-113`) |
| Unit in code | `common = labels with COMMON & classes` (`validation_run.py:396`; `vguard.py:34`). The label points themselves were not opened (VALIDATION). | `region_setup` on the CONFIRM group manifest (`score_adapt.py:74-82,338-340`). The manifest is built by FRESH-1's step0 functions (`confirm_build.py:202-233`). | Single-linkage connected components at `query_pairs(60.0)` (`prepixel_stats.py:50,123-132`). Group geometry is the union of member polygons (`step0.py:759-760`). |
| Registered matching | Greedy in rank order: the nearest unclaimed label within ≤ 60 m, ties to label id; one-to-one; credited to the label's owner block (`PACKET.md:71`) | Reach 15 m, NMS 60 m (`PACKET.md:56`); no one-to-one rule | Reach 15 m, NMS 60 m (`FRESH1_PREPIXEL.md:79`); no one-to-one rule |
| Matching in code | `match_greedy` (`verify/v3/v3_frozen_mixture.py:259-275`), called at `validation_run.py:407`; owner block at `:410` | `rs.ranked_pool` reaches parts through a 15 m buffered footprint tree (`region_score.py:152,167-168`). A proposal may credit several parts. | Same as (b): `score_fresh.py:159-169` → `region_score.py:157-169` |
| Registered pool / NMS | Global 60 m NMS over 45 blocks (score descending, id ascending), then ranked by score, mean probability, area descending, id ascending (`PACKET.md:66`) | NMS 60 m (`PACKET.md:56`) | NMS 60 m (`FRESH1_PREPIXEL.md:79`) |
| Pool in code | `build_pool` (`work/campaign_2026-09-22/screen_scorer.py:68-70`): `global_nms`, radius 60, inclusive (`targets/.../adaf_stage1_2026-09-20/scripts/score_staged.py:28,54-79`); `mean_rank_key` (`screen_scorer.py:51-52`) | `cur.nms` 60 m, inclusive (`r3_curation.py:41,168-174`); `rank_key` (`:164-165`) | Same as (b). DEPLOY is INTERLEAVE, with a synthetic score and NMS re-applied (`work/fusion_2026-10-03/fuse_eval.py:55-62,85-92`; `score_fresh.py:172-175,345`). |
| Registered budget | K_d = round(d × A), A = summed block interiors, **all proposals charged**. Primary: each arm at min(K_d, its own supply) (`PACKET.md:67`; `PACKET_AMENDMENT_B.md:20-21`). | Mean recall at 0.5, 1 and 2 **background** proposals per km² (`PACKET.md:55`) | The same (`FRESH1_PREPIXEL.md:79`) |
| Budget in code | `area` (`validation_run.py:395`); `round(d * area)` (`:420`); top-K slice `[:k]`, which implicitly applies min(K, supply) (`:406`) | `recall_at`: allowance floor(d × background km²); only background-tile proposals counted (`score_holdout.py:73-89`; `region_score.py:166`); primary = mean of 0.5/1/2 (`region_score.py:192`) | Same as (b) (`score_fresh.py:159-169`) |
| Registered grouping / uncertainty | Paired block bootstrap over 45 blocks, 2,000 draws, seed 20260924, 90 % interval on parts. MET iff Δpp ≥ 5 and lower bound > 0 (`PACKET.md:76-85`). | Cluster (2 km square) × seed bootstrap, 2,000 draws, seed 20261005. ADAPT_GAIN: mean ≥ +0.03 and lo95 > 0. Milestone: BEATS if ≥ +0.02 and lo95 > 0; TRAILS if hi95 < 0 (`PACKET.md:57-60`). | `hierarchical_ci`, one pair, 2,000 draws, seed 20261003 (`FRESH1_PREPIXEL.md:45,80`). A1 adds DEPLOY − ADAF barrow (`step2/STEP2_AMENDMENT_A1.md:107-110`, registered before the freeze, `:3-4`). |
| Grouping / uncertainty in code | `interval` (`validation_run.py:372-382,400`); verdict (`:438-453`); constants from `CONFIG.json` (`:38-42`) | `contrast` / `ci_draws` (`score_adapt.py:98-124`); labels (`:57-70`); contrasts (`:282-293`); constants (`nds_common.py:125-127`); seed pairing by index (`score_adapt.py:273-275`) | `contrast` / `label_of` / `a1_label` (`score_fresh.py:178-207`); `primary_verdicts` (`:412-420`); constants (`fresh_common.py:110`). Step 1 row = proceed (v̂ 0.728, `freeze/FREEZE.json` `step1`), so the reading is PRIMARY. |
| **Verdict** | **AGREES** | **AGREES** | **AGREES** (registered primary SUPERIOR and A1 INFERIOR, both reported as registered) |

### Notes behind the verdicts

1. **(a) Rounding is harmless.** `round()` in `validation_run.py:420` is Python round-half-even. A = 1,119.75 km², so
   2 × A = 2,239.5 → 2,240, the same as round-half-up. 0.5 × A and 1 × A are not on a .5 boundary. No effect.
2. **(b) The erratum changes nothing.** `score_adapt_erratum1.py` supplies only the descriptive district label; the
   frozen code path and numbers are unchanged (`ERRATUM_1.md`). The independent recount MATCHes 71/71
   (`adapt_nds_2026-10-05/RESULT.md:4`).
3. **(b), (c): IGNORE wording against code (ambiguity, not a deviation).**
   - Rule 6 says proposals in IGNORE zones are "neither hits nor false alarms" (`FRESH1_PREPIXEL.md:30`). IGNORE is every other register record, buffered 30 m.
   - In code, background boxes exclude IGNORE (`confirm_build.py:244-247`; FRESH-1 §3). Inside positive cores, `region_score` applies no IGNORE test, so an eligible-barrow hit takes precedence.
   - A literal reading is unworkable. *Grabhügelfeld* (barrow cemetery) polygons, and in Oldenburg *Grabungsschutzgebiet* (protected excavation zone) polygons, enclose the eligible barrows. Treating them as IGNORE removes 40–45 % of every system's hits (`IGNORE_CHECK.json`: Verden DEPLOY 0.400 → 0.242; Oldenburg A_02 0.603 → 0.341).
   - With those container areas excluded, the residual literal-rule effect is 1–5 groups per system. The contrasts move as follows (point estimates; both directions unchanged):

     | Contrast | Legacy | Literal residual |
     |---|---:|---:|
     | Verden DEPLOY − AO | +0.102 | +0.099 |
     | Verden DEPLOY − ADAF barrow | −0.156 | −0.159 |
     | Oldenburg A − C | +0.141 | +0.138 |
     | Oldenburg A − ADAF barrow | +0.045 | +0.049 |
4. **(b), (c): two structural conventions, both as registered.**
   - The budget charges only background-tile proposals. Non-hitting proposals inside positive cores are free: 122 of 423 km² in Verden and 151 of 451 km² in Oldenburg.
   - One proposal can credit more than one group.
   - Task 3 measures how much these conventions matter.
5. **(c) Reproduction.** The FRESH-1 legacy group primaries are reproduced exactly with the frozen `region_score`: DEPLOY − AO +0.102, DEPLOY − ADAF barrow −0.156, S\* − AO +0.187 (`BENCH_V0.json`). This uses the same code, not an independent implementation.

## Task 3: benchmark v0 sensitivity (Oldenburg CONFIRM and Verden; saved proposals; legacy kept)

**Entity check.** The source register defines no barrow groups. In both local register snapshots, `objektnummer` is
unique per Gemarkung for every Grabhügel record:

- Verden: 759 records, 759 keys.
- Oldenburg: 967 records, 964 keys. The one 4-record key spans 5 m, duplicate geometries that dedup rule 4 handles.

So the entities are individual barrows, taken from the existing individual manifests: 728 in Verden, 959 in
Oldenburg. *Grabhügelfeld* records stay IGNORE, as in the legacy setup.

**v0 rules.**

- **Matching.** One-to-one maximum-cardinality bipartite matching (`scipy.sparse.csgraph.maximum_bipartite_matching`). A proposal is eligible within 15 m of the footprint polygon.
- **Budget.** K = floor(b × total scored area), counting every in-core proposal: positive and background tiles, IGNORE included.
  - Verden: 422.84 km², K = 211 / 422 / 845.
  - Oldenburg: 451.12 km², K = 225 / 451 / 902.
- **Pools.** NMS60 is the frozen ranked pool. RAW is every in-core proposal with no NMS.
- **Recall caps.** Entities outnumber K at low b, so recall is capped by the budget:
  - Verden: min(K, n) / n = 0.290 / 0.580 / 1.0;
  - Oldenburg: 0.235 / 0.470 / 0.941.
- **Total scored area.** This is the union of all manifest cores. Positive cores are not validity-eroded, unlike background boxes.

**Primary values.** Each cell is the mean over b = 0.5 / 1 / 2. Point estimates; no intervals computed.

| Region | System | Legacy group (frozen) | Legacy individual (frozen) | v0 NMS60 | v0 RAW |
|---|---|---:|---:|---:|---:|
| Verden | ADAF AO | 0.298 | 0.147 | 0.116 | 0.126 |
| Verden | ADAF barrow | 0.556 | 0.313 | 0.236 | 0.286 |
| Verden | CONT e41 | 0.416 | 0.212 | 0.158 | 0.162 |
| Verden | S\* (J_03) | 0.484 | 0.250 | 0.175 | 0.178 |
| Verden | DEPLOY | 0.400 | 0.205 | 0.151 | 0.139 |
| Oldenburg | ADAF AO | 0.328 | 0.129 | 0.090 | 0.102 |
| Oldenburg | ADAF barrow | 0.529 | 0.251 | 0.156 | 0.182 |
| Oldenburg | CONT e41 | 0.429 | 0.176 | 0.116 | 0.118 |
| Oldenburg | A (3-seed mean; 02/03/04) | 0.575 | 0.266 | 0.161 (0.163 / 0.169 / 0.150) | 0.180 (0.188 / 0.191 / 0.162) |
| Oldenburg | C (3-seed mean) | 0.434 | 0.199 | 0.119 | 0.120 |
| Oldenburg | J (3-seed mean) | 0.406 | 0.179 | 0.112 | 0.113 |

Per-b values are in `BENCH_V0.json`.

**Contrasts under each scoring** (point estimates):

| Contrast | Legacy group | Legacy individual | v0 NMS60 | v0 RAW |
|---|---:|---:|---:|---:|
| Verden DEPLOY − ADAF AO (FRESH-1 registered primary) | +0.102 | +0.059 | +0.035 | +0.013 |
| Verden DEPLOY − ADAF barrow (A1) | −0.156 | −0.108 | −0.085 | −0.147 |
| Verden S\* − ADAF barrow | −0.071 | −0.063 | −0.061 | −0.108 |
| Oldenburg A − C (ADAPT primary 1) | +0.141 | +0.067 | +0.041 | +0.060 |
| Oldenburg A − ADAF barrow (milestone) | +0.045 | +0.015 | +0.004 | −0.002 (per seed +0.006 / +0.008 / −0.021) |

**Reading Task 3**

- **No sign flips on a contrast whose legacy label was decisive.** The verdicts were ADAPT_GAIN; DEPLOY > AO; DEPLOY < ADAF barrow. Under v0 these keep their direction.
- **Magnitudes shrink sharply.**
  - A − C: +0.141 → +0.041 (NMS60) or +0.060 (RAW).
  - DEPLOY − AO: +0.102 → +0.035 or +0.013.
  - Part of the shrinkage is the budget cap: K < entity count at b ≤ 1.
- **The Oldenburg milestone moves from +0.045 to about 0**, with one seed negative under RAW. Its legacy label was NOT_DISTINGUISHED, so no recorded claim changes. But under v0, "A is ahead of ADAF barrow" is not supported even as a point estimate.
- **Under RAW (no NMS), ADAF barrow gains most** (+0.05 against NMS60 in both regions). It produces separate proposals for neighbouring mounds that the 60 m NMS discards. That widens its lead over every other system.
- **Uncertainty is not computed for v0.** A cluster bootstrap under v0 is needed before any v0 number is used as a claim.

## Inputs not available (reported, not substituted)

- Verden: ADAF enclosure and ringfort proposals were never produced, so they are not in the Verden union.
- JOINT-1 S\* proposals exist for one seed per region file. Only J_2026092803 (S\*) was used for England, Wales and
  Ireland. The other J seeds exist under `run_s{1..7}` and were not added.
