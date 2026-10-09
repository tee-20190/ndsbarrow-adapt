# STAGE1-D1-V0: result (2026-10-07T21:08Z, Opus 5.5)

Frozen rule: `FREEZE.md` (commit 1ea377ca). CPU only, read-only on existing files; no inference, no training. Not opened:
Oldenburg, reserved terrain, TEST. Runtime: WSL `/home/tee-/adaf-h2h-venv/bin/python -B` (the venv `bench_v0.py:12-13` used).

| File | Content |
|---|---|
| `step0.py` → `STEP0.json` | Step 0 legitimacy check |
| `score_v0.py` → `RESULT.json` | v0 scoring, gate, week-1 reproduction check |
| `tables.py` | Prints the tables below from `RESULT.json` |

## Verdict: **PASS** (gate at b = 0.5, RAW)

| Gate term (`FREEZE.md:30-32`) | Required | Measured |
|---|---|---|
| Mean over 10 runs of (run recall − ADAF barrow recall) | ≥ +0.030 | **+0.0625** (ADAF barrow 0.169; run mean 0.231) |
| Per-run differences > 0 | ≥ 8/10 | **10/10** (range +0.052 to +0.074) |

Consequence (`FREEZE.md:43-45`): Stage 2 (10 matched controls, ~40 CU) is eligible. It needs its own freeze, packet and
storage/archive gates. No checkpoint is adopted by this result.

## Step 0: legitimacy — PASS

**Where D1 trained.** Every BARROW_L confirmation run (seeds 2026092805–14) trained only on ADAPT-NDS-1's Lower Saxony FIT
table, not on Verden. Verden was the evaluation (DEV) region only.

- Each run's `TRAIN_RESULT.json` (`outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/v1/results/runs/*/`)
  records `ls_table` sha256 `58d6aa16…` = `outputs/adapt_nds_2026-10-05/fit/FIT_TABLE_PRIVATE.json`, and
  `updates` = `lower_saxony_updates` = 5,116. The all-local plan draws only FIT rows
  (`work/adapt_match_2026-10-05/match_train.py:56-69`; `match_schedule.py:66-76`). Start: ADAF barrow tar `51b679fa…`, the
  same checkpoint as the comparator.
- FIT table: 852 positive + 852 background 423 m windows in three FIT districts
  (`work/adapt_nds_2026-10-05/PACKET.md:31-32`); window union 280.3 km². Terrain rule: pixel centres outside the FIT
  union are NoData, ≥ 2 km from Verden. Training pixels are confined to the windows (crops sit inside them).

**Separation, measured** (`step0.py:85-92`, `STEP0.json`):

| Check | Value |
|---|---:|
| FIT windows intersecting the Verden district polygon | 0 |
| Min distance, FIT windows → Verden district | 2,036 m |
| Min distance, FIT windows → Verden scored cores | 2,213 m |
| Min distance, FIT windows → Verden inference read windows (incl. margins) | 463 m |
| Min distance, FIT windows → Verden entity footprints | 2,880 m |
| Register records shared between FIT labels and Verden entities | 0 |
| 1 km DGM1 squares shared: training fetch log (2,806 files) vs Verden DEV fetch log (806 files) | 0 |

**Eligible subset:** all 717 Verden tiles, 728 individual barrows, 422.84 km² scored area. Any training-context buffer up
to 2.2 km leaves every scored core and entity eligible, so no split was drawn (`FREEZE.md:12-15` respected).

**Caveat (not a Step 0 failure).** Verden is development terrain: previously evaluated by FRESH-1 and ADAPT-MATCH-0, and
D1's recipe selection used Verden group recall of the three development seeds
(`work/d1_collapse_2026-10-06/PACKET.md`, "Sequence"). The ten confirmation seeds were trained after selection. This is
a development-evidence result, not an untouched-terrain claim.

## Pipeline

- **Proposals.** D1 runs: `outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/v1/results/dev/PROPOSALS_D1F1_BARROW_L_<seed>_PRIVATE.json`,
  sha256 checked against `SESSION_VERIFIED.json` pins (status PASS). ADAF barrow:
  `outputs/fresh_region_2026-10-03/step2/score/individual/PROPOSALS_ADAF_barrow_PRIVATE.json` (FRESH-1). Both made on
  the same inference manifest (`c1e1ba01…`) by the same `infer_irl.py` (`f3855551…`, pinned identically in the FRESH-1
  freeze, the D1 freeze and the session-3 pins). Tiles checked identical (`score_v0.py:89-100`).
- **Scoring.** `score_v0.py:56-87` is a copy of `bench_v0.py:61-103` (`raw_pool`, `match_count`, `v0`), unmodified
  logic: frozen `region_score` (`b3337741…`) setup and 15 m footprint reach; one-to-one maximum-cardinality matching;
  K = floor(b × 422.84) = 211 / 422 / 845, every in-core proposal charged; legacy IGNORE handling (manifest as built).
  RAW sorts by `rank_key` (score, mean probability, area desc, proposal id asc). NMS60 is `region_score.ranked_pool`.
- **Entities:** N = 728. Ceiling c = min(1, K/N) = 0.290 / 0.580 / 1.000.

## Sanity check: week-1 reproduction — IDENTICAL

ADAF barrow under this implementation equals `work/gpt6_pro_collab_2026-10-07/detector_week1/BENCH_V0.json`
(verden → ADAF_barrow → v0_individual) to < 1e-12 on every value:

| Decoder | b = 0.5 | b = 1 | b = 2 | Primary | Supply |
|---|---:|---:|---:|---:|---:|
| RAW | 0.1690 | 0.2912 | 0.3970 | 0.2857 | 794 |
| NMS60 | 0.1635 | 0.2527 | 0.2926 | 0.2363 | 688 |

The week-1 table's 0.286 (RAW) and 0.236 (NMS60) are these primaries. The restriction is identical because Step 0
excludes nothing.

## Per-run results

Each cell: recall (matches). Δ = run − ADAF barrow. Supply = in-core proposals. Matches per 100 slots use K slots.

**RAW (primary decoder)**

| System | Supply | b=0.5 | b=1 | b=2 | Δ0.5 | Δ1 | Δ2 | /100 slots 0.5 / 1 / 2 |
|---|---:|---|---|---|---:|---:|---:|---|
| ADAF barrow | 794 | 0.169 (123) | 0.291 (212) | 0.397 (289) | – | – | – | 58.3 / 50.2 / 34.2 |
| BARROW_L seed 05 | 2531 | 0.225 (164) | 0.376 (274) | 0.507 (369) | +0.056 | +0.085 | +0.110 | 77.7 / 64.9 / 43.7 |
| BARROW_L seed 06 | 2633 | 0.225 (164) | 0.390 (284) | 0.516 (376) | +0.056 | +0.099 | +0.120 | 77.7 / 67.3 / 44.5 |
| BARROW_L seed 07 | 3509 | 0.232 (169) | 0.370 (269) | 0.511 (372) | +0.063 | +0.078 | +0.114 | 80.1 / 63.7 / 44.0 |
| BARROW_L seed 08 | 3471 | 0.228 (166) | 0.378 (275) | 0.516 (376) | +0.059 | +0.087 | +0.120 | 78.7 / 65.2 / 44.5 |
| BARROW_L seed 09 | 2882 | 0.235 (171) | 0.408 (297) | 0.552 (402) | +0.066 | +0.117 | +0.155 | 81.0 / 70.4 / 47.6 |
| BARROW_L seed 10 | 2841 | 0.235 (171) | 0.393 (286) | 0.538 (392) | +0.066 | +0.102 | +0.141 | 81.0 / 67.8 / 46.4 |
| BARROW_L seed 11 | 2470 | 0.236 (172) | 0.387 (282) | 0.521 (379) | +0.067 | +0.096 | +0.124 | 81.5 / 66.8 / 44.9 |
| BARROW_L seed 12 | 2358 | 0.221 (161) | 0.371 (270) | 0.501 (365) | +0.052 | +0.080 | +0.104 | 76.3 / 64.0 / 43.2 |
| BARROW_L seed 13 | 2624 | 0.234 (170) | 0.385 (280) | 0.522 (380) | +0.065 | +0.093 | +0.125 | 80.6 / 66.4 / 45.0 |
| BARROW_L seed 14 | 2588 | 0.243 (177) | 0.391 (285) | 0.530 (386) | +0.074 | +0.100 | +0.133 | 83.9 / 67.5 / 45.7 |
| **10-run mean** | | 0.231 | 0.385 | 0.522 | +0.062 | +0.094 | +0.125 | |
| Runs with Δ > 0 | | | | | 10/10 | 10/10 | 10/10 | |

**NMS60 (secondary decoder)**

| System | Supply | b=0.5 | b=1 | b=2 | Δ0.5 | Δ1 | Δ2 | /100 slots 0.5 / 1 / 2 |
|---|---:|---|---|---|---:|---:|---:|---|
| ADAF barrow | 688 | 0.163 (119) | 0.253 (184) | 0.293 (213) | – | – | – | 56.4 / 43.6 / 25.2 |
| BARROW_L seed 05 | 2135 | 0.210 (153) | 0.309 (225) | 0.371 (270) | +0.047 | +0.056 | +0.078 | 72.5 / 53.3 / 32.0 |
| BARROW_L seed 06 | 2167 | 0.210 (153) | 0.312 (227) | 0.363 (264) | +0.047 | +0.059 | +0.070 | 72.5 / 53.8 / 31.2 |
| BARROW_L seed 07 | 2831 | 0.217 (158) | 0.298 (217) | 0.359 (261) | +0.054 | +0.045 | +0.066 | 74.9 / 51.4 / 30.9 |
| BARROW_L seed 08 | 2821 | 0.217 (158) | 0.302 (220) | 0.367 (267) | +0.054 | +0.049 | +0.074 | 74.9 / 52.1 / 31.6 |
| BARROW_L seed 09 | 2301 | 0.216 (157) | 0.319 (232) | 0.370 (269) | +0.052 | +0.066 | +0.077 | 74.4 / 55.0 / 31.8 |
| BARROW_L seed 10 | 2303 | 0.214 (156) | 0.319 (232) | 0.371 (270) | +0.051 | +0.066 | +0.078 | 73.9 / 55.0 / 32.0 |
| BARROW_L seed 11 | 2039 | 0.221 (161) | 0.313 (228) | 0.368 (268) | +0.058 | +0.060 | +0.076 | 76.3 / 54.0 / 31.7 |
| BARROW_L seed 12 | 1967 | 0.210 (153) | 0.297 (216) | 0.354 (258) | +0.047 | +0.044 | +0.062 | 72.5 / 51.2 / 30.5 |
| BARROW_L seed 13 | 2172 | 0.217 (158) | 0.312 (227) | 0.371 (270) | +0.054 | +0.059 | +0.078 | 74.9 / 53.8 / 32.0 |
| BARROW_L seed 14 | 2124 | 0.212 (154) | 0.305 (222) | 0.365 (266) | +0.048 | +0.052 | +0.073 | 73.0 / 52.6 / 31.5 |
| **10-run mean** | | 0.214 | 0.309 | 0.366 | +0.051 | +0.056 | +0.073 | |
| Runs with Δ > 0 | | | | | 10/10 | 10/10 | 10/10 | |

## Reading

- **Direction is uniform:** 60/60 run × b × decoder cells favour the D1 runs. The smallest run difference (+0.044, NMS60
  b = 1) exceeds the gate threshold.
- **The b = 0.5 cell is close to the ceiling for both.** c = 0.290. D1 runs fill 76–84 % of slots with matches, ADAF
  barrow 58 %.
- **ADAF barrow is supply-limited at b = 2.** Its in-core supply (794 RAW, 688 NMS60) is below K = 845, so its b = 2
  cells use fewer slots. D1 runs supply 2,358–3,509. The gate cell (K = 211) is unaffected.
- **NMS costs D1 runs more at b ≥ 1** (RAW − NMS60 mean: +0.076 at b = 1, +0.156 at b = 2) than at b = 0.5 (+0.017).
- **No interval computed.** The freeze specifies a point-estimate gate; seed spread (range +0.052 to +0.074 at the gate
  cell) is reported instead. A cluster bootstrap would be needed before a claim beyond the gate.
- **Scope:** development terrain, one region, one comparator checkpoint. Not untouched-terrain evidence; not a BENCH-1
  claim.

## Forecast scored

P(PASS): Opus 0.40, Pro 0.45. Outcome PASS. Brier: Opus 0.36, Pro 0.3025.
