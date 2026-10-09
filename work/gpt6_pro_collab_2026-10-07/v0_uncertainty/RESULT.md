# v0 uncertainty, budget ceilings, IGNORE readings, register semantics and adaptation-vs-start (2026-10-07)

This implements GPT-6 Pro Round 3 items R3-1, R3-2, R3-7 (IGNORE) and R3-8 items 2-3
(`../ROUND3_REPLY.md:11-89, 91-273, 646-679, 699-709`).

- CPU only. Nothing outside this directory was modified (checked by `find -newer` after the runs). There was no
  GPU work, no inference and no training. No reserved or TEST data was opened. `work/stage1_d1_v0_2026-10-07/`
  was not touched.
- Runtime: WSL `/home/tee-/adaf-h2h-venv/bin/python -B`, the venv the `detector_week1` scripts use.
- Outputs are aggregates only. They contain no ids, no coordinates and no place names below district level.
- Finished 2026-10-07T21:26Z.

| File | Content |
|---|---|
| `engine.py` | Blocks, incremental matching increments, the exact shortcut, the literal formula and literal copied-graph matching. Pure numpy/scipy/shapely. |
| `v0data.py` | Frozen inputs. Pools come from `detector_week1/bench_v0.py`, imported unchanged (`raw_pool`, `deploy_objs`, frozen `region_score`). |
| `boot_v0.py` → `BOOT_V0.json` | Tasks 1, 2, 3 and 6. Log: `boot_run.log`. |
| `ignore_v0.py` → `IGNORE_V0.json` | Task 4. Log: `ignore_run.log`. |
| `register_semantics.py` → `REGISTER_SEMANTICS.json` | Task 5. |
| `tests/test_engine.py` | Unit tests and the random literal cross-check. |
| `tables_v0.py` | Prints the tables below from the JSON. |

## Proof of correctness (read first)

| Check | Result |
|---|---|
| Unit tests (`python -B -m pytest -p no:cacheprovider -q .../tests`). They cover all-ones = unweighted top-K, a duplicated block, a cutoff partway through one proposal's copies, an edge that crosses an initial grid boundary, a budget larger than the pool, zero weight / zero budget, increments = scipy on every prefix, and offset grids assigning area once. | **13 passed** |
| Random literal cross-check: shortcut = literal formula (F_j, t_j, q, d) = literal copied-graph matching (scipy) on 1,000 random small graphs, every budget from 0 to the copied pool + 1 (`test_random_literal_cross_check`) | 0 mismatches |
| All-ones block weights reproduce every BENCH_V0 v0 score exactly (float equality) and K. Oldenburg: 12 systems × 2 decoders × 3 budgets. Verden: 5 × 2 × 3. | **72/72, 30/30 exact**; K 225/451/902 and 211/422/845 |
| Vectorised shortcut = literal formula on real pools with random multinomial weights (20 replicates × 3 budgets × every pool, 2 km grid) | 1,440 (Oldenburg) and 1,560 (Verden) checks, 0 mismatches |
| Full-pool maximum matching C: incremental = scipy, every pool | pass (`oracle_table`, `boot_v0.py:115-136`) |
| Every proposal-reference edge, across every evaluated pool, lies in one block | 0 of 7,901 (Oldenburg, 24 pools) and 0 of 7,726 (Verden, 26 pools) edges cross blocks, on all three grids |
| Edges whose proposal cell differs from the reference's anchor cell (rescued by the merge) | 31 / 29 / 34 (Oldenburg) and 16 / 10 / 18 (Verden) on the 2 km / 4 km / offset grids |
| Blocks partition the scored area and the references | Σ A_j = A within 1e-6 km²; Σ N_j = N (`boot_v0.py:177-180`) |

The checks verify the combinatorics and reproduction. They do not verify interval coverage (`ROUND3_REPLY.md:93,216`).

## Procedure as implemented

- **Inputs.** Frozen individual manifests: Oldenburg sha256 `3531675c…`, N = 959, A = 451.117 km²; Verden
  `a97b6124…`, N = 728, A = 422.843 km². The complete saved pools are the in-core frozen proposals. NMS60 is
  `region_score.ranked_pool`; RAW is `bench_v0.raw_pool` (`detector_week1/bench_v0.py:61-65`). Both are in
  `rank_key` order, a deterministic tie order (`r3_curation.py:163-164`). Input hashes are in
  `BOOT_V0.json` `inputs_sha256`. Pools are not truncated at the largest budget.
- **Blocks** (`engine.py:231-263`).
  - Grid: 2 km squares aligned to (0, 0) in EPSG:25832.
  - Anchor: the cell holding the footprint centroid.
  - Merging: each anchor cell is merged with every cell whose closed square intersects the reference's 15 m reach
    buffer (the scorer's own `buffer(REACH_M)`, `region_score.py:152`), then union-find closure.
  - Every cell with scored area is kept, so background-only blocks are included.
  - Sensitivities: a 4 km grid at (0, 0), and a 2 km grid offset by (1000, 1000).
- **Weights.** Multinomial(J; 1/J), with A\* = Σ w_j A_j, N\* = Σ w_j N_j and K\*_b = floor(b·A\* + 1e-9). The
  1e-9 is bench_v0's tolerance. One weight matrix per region and grid is shared by every system, seed, budget and
  decoder.
- **Shortcut** (`engine.py:86-110`).
  - δ_i = F(i+1) − F(i), from incremental Kuhn augmentation in frozen rank order (`engine.py:64-72`). Edges lie
    inside blocks, so the δ of block j's proposals form F_j.
  - M\* = Σ_{i<cut} w_{j(i)} δ_i + d·δ_cut. This is algebraically Σ_j w_j F_j(t_j) + d[F_q(t_q+1) − F_q(t_q)].
  - Order: original rank, then copy id.
  - When the copied pool is smaller than K\*, the whole pool is used and the underfill is recorded
    (`underfilled_replicate_share`).
- **Replicates.** 5,000, master seed 20261007, `SeedSequence.spawn` streams `geo_oldenburg`, `train_oldenburg`,
  `geo_verden` and `train_verden` (`boot_v0.py:34-37`). N\* = 0 occurred in 0 replicates. Intervals are pointwise
  2.5/97.5 percentiles of the paired difference. Each checkpoint is scored first and the scores are then averaged
  (no pooled predictions).
- **Training-run dependencies** (`boot_v0.py:40-70`).
  - **Oldenburg: genuinely paired.** A_s and C_s both start from J_s's final weights at the same seed, and C uses
    "identical updates, schedule and draws for the English, Welsh and Irish steps" (`adapt_nds_2026-10-05/PACKET.md:42-48`;
    `nds_train.py:1-13,176-188`: start = `J_<seed>/final.pt`, `torch.manual_seed(seed)`, every non-Lower-Saxony
    step identical).
  - So (J_s, A_s, C_s) is one family. The three families are resampled together with replacement.
  - ADAF barrow, ADAF AO and CONT are fixed single checkpoints.
- **Verden: the registered S\* is a selected checkpoint.** It is J_2026092803, chosen by FRESH-1's rule: the
  highest region-macro DEV primary among the five JOINT-1 confirmation seeds, ties to the lower seed
  (`FRESH1_PREPIXEL.md:75`; `step2/freeze/FREEZE.json` `sstar.resolution.macro`). DEPLOY = INTERLEAVE(ADAF AO, S\*).
  - "Training only" here is **selection-aware**. Each replicate draws 5 seeds with replacement and re-applies the
    frozen rule to the frozen macro values.
  - J_03 was selected in 3,335 of 5,000 replicates, J_06 in 1,273, J_02 in 347, J_05 in 43 and J_04 in 2.
  - Proposals for all five J seeds exist in Verden (`outputs/fresh_region_2026-10-03/step2/score/individual/`), so
    DEPLOY was rebuilt for each with the unchanged `fuse_eval` INTERLEAVE.
  - For a fixed-checkpoint reading of DEPLOY, the training-only interval is the point and joint = geography only.
  - ADAF is fixed.

## Task 1: blocks

| Region | Grid | J | Blocks with labels | Background-only | Largest area share | Largest label share | J_area | J_labels | Cell merges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oldenburg | 2 km | 146 | 79 | 67 | 0.026 | 0.114 | 101.2 | 23.8 | 10 |
| Oldenburg | 4 km | 56 | 36 | 20 | 0.083 | 0.126 | 32.1 | 15.1 | 6 |
| Oldenburg | 2 km offset | 196 | 74 | 122 | 0.028 | 0.114 | 125.7 | 20.7 | 18 |
| Verden | 2 km | 126 | 48 | 78 | 0.037 | 0.096 | 83.5 | 20.3 | 16 |
| Verden | 4 km | 51 | 25 | 26 | 0.120 | **0.261** | 24.2 | **8.3** | 8 |
| Verden | 2 km offset | 191 | 57 | 134 | 0.065 | **0.251** | 87.2 | **11.2** | 18 |

No block has zero area. Label concentration is the weak point. On the primary grid J_labels is about 20-24. On
Verden's 4 km and offset grids, one block holds a quarter of the barrows (J_labels 8-11). By the R3-2 criterion
(`ROUND3_REPLY.md:146`), Verden intervals on those two grids are **fragile**. They are reported, not discarded.

## Task 2: v0 intervals (main table, 2 km grid, percentage points)

Intervals are 95 % percentile intervals over 5,000 replicates. "Per run" gives the point difference for each
training family (A_s − C_s, or A_s − ADAF barrow).

| Region | Contrast | Decoder | b | Point | Per run (seed 02 / 03 / 04) | Geography only | Training only | Joint |
|---|---|---|---|---:|---|---|---|---|
| Oldenburg | A − C | NMS60 | 0.5 | +3.1 | +2.6 / +3.2 / +3.3 | [+2.0, +4.5] | [+2.6, +3.3] | [+1.8, +4.6] |
| Oldenburg | A − C | NMS60 | 1 | +4.4 | +5.1 / +4.1 / +4.0 | [+3.1, +6.3] | [+4.0, +5.1] | [+2.7, +6.7] |
| Oldenburg | A − C | NMS60 | 2 | +4.9 | +5.7 / +4.6 / +4.5 | [+3.7, +6.9] | [+4.5, +5.7] | [+3.4, +7.1] |
| Oldenburg | A − C | NMS60 | mean | +4.1 | +4.5 / +4.0 / +3.9 | [+3.1, +5.7] | [+3.9, +4.5] | [+2.9, +5.8] |
| Oldenburg | A − ADAF barrow | NMS60 | 0.5 | +1.2 | +1.3 / +2.1 / +0.3 | [-0.0, +2.7] | [+0.3, +2.1] | [-0.3, +2.9] |
| Oldenburg | A − ADAF barrow | NMS60 | 1 | +0.6 | +1.0 / +1.3 / -0.5 | [-0.7, +2.0] | [-0.5, +1.3] | [-1.2, +2.4] |
| Oldenburg | A − ADAF barrow | NMS60 | 2 | -0.6 | -0.3 / +0.3 / -1.7 | [-1.8, +1.8] | [-1.7, +0.3] | [-2.1, +2.0] |
| Oldenburg | A − ADAF barrow | NMS60 | mean | +0.4 | +0.7 / +1.2 / -0.6 | [-0.6, +1.9] | [-0.6, +1.2] | [-1.0, +2.1] |
| Oldenburg | A − C | RAW | 0.5 | +3.4 | +2.8 / +3.9 / +3.5 | [+2.2, +5.0] | [+2.8, +3.9] | [+2.0, +5.3] |
| Oldenburg | A − C | RAW | 1 | +5.7 | +7.3 / +5.5 / +4.3 | [+4.3, +8.1] | [+4.3, +7.3] | [+3.8, +8.7] |
| Oldenburg | A − C | RAW | 2 | +8.9 | +10.6 / +8.8 / +7.2 | [+7.2, +11.4] | [+7.2, +10.6] | [+6.4, +11.9] |
| Oldenburg | A − C | RAW | mean | +6.0 | +6.9 / +6.0 / +5.0 | [+4.7, +8.0] | [+5.0, +6.9] | [+4.4, +8.2] |
| Oldenburg | A − ADAF barrow | RAW | 0.5 | +1.0 | +0.9 / +2.2 / +0.0 | [-0.4, +2.5] | [+0.0, +2.2] | [-0.7, +2.9] |
| Oldenburg | A − ADAF barrow | RAW | 1 | -0.1 | +1.3 / +0.6 / -2.3 | [-1.4, +1.9] | [-2.3, +1.3] | [-2.3, +2.5] |
| Oldenburg | A − ADAF barrow | RAW | 2 | -1.5 | -0.4 / -0.3 / -3.9 | [-4.2, +0.8] | [-3.9, -0.3] | [-5.0, +1.5] |
| Oldenburg | A − ADAF barrow | RAW | mean | -0.2 | +0.6 / +0.8 / -2.1 | [-1.6, +1.4] | [-2.1, +0.8] | [-2.4, +1.9] |
| Verden | DEPLOY − ADAF AO | NMS60 | 0.5 | +2.3 | (S\* fixed) | [+1.1, +4.0] | [+2.1, +2.3] | [+1.0, +3.9] |
| Verden | DEPLOY − ADAF AO | NMS60 | 1 | +4.5 | | [+2.4, +6.5] | [+2.7, +4.5] | [+1.6, +6.3] |
| Verden | DEPLOY − ADAF AO | NMS60 | 2 | +3.6 | | [+2.1, +5.6] | [+1.5, +3.6] | [+0.5, +5.4] |
| Verden | DEPLOY − ADAF AO | NMS60 | mean | +3.5 | | [+2.4, +4.9] | [+2.2, +3.5] | [+1.5, +4.8] |
| Verden | DEPLOY − ADAF barrow | NMS60 | 0.5 | -8.7 | | [-12.0, -5.9] | [-8.9, -8.7] | [-12.1, -5.9] |
| Verden | DEPLOY − ADAF barrow | NMS60 | 1 | -10.2 | | [-13.2, -7.4] | [-12.0, -10.2] | [-13.8, -7.6] |
| Verden | DEPLOY − ADAF barrow | NMS60 | 2 | -6.7 | | [-9.1, -3.9] | [-8.8, -6.7] | [-10.6, -4.1] |
| Verden | DEPLOY − ADAF barrow | NMS60 | mean | -8.5 | | [-10.8, -6.3] | [-9.8, -8.5] | [-11.4, -6.5] |
| Verden | DEPLOY − ADAF AO | RAW | 0.5 | +1.6 | | [-0.2, +3.2] | [+1.1, +1.6] | [-0.4, +3.1] |
| Verden | DEPLOY − ADAF AO | RAW | 1 | +2.2 | | [-0.4, +4.6] | [+1.0, +2.2] | [-1.0, +4.5] |
| Verden | DEPLOY − ADAF AO | RAW | 2 | +0.0 | | [-2.2, +3.1] | [-1.9, +0.0] | [-3.1, +2.8] |
| Verden | DEPLOY − ADAF AO | RAW | mean | +1.3 | | [-0.4, +3.0] | [+0.1, +1.3] | [-1.0, +2.9] |
| Verden | DEPLOY − ADAF barrow | RAW | 0.5 | -9.8 | | [-14.3, -7.0] | [-10.3, -9.8] | [-14.5, -7.2] |
| Verden | DEPLOY − ADAF barrow | RAW | 1 | -15.7 | | [-19.6, -12.3] | [-16.9, -15.7] | [-20.3, -12.5] |
| Verden | DEPLOY − ADAF barrow | RAW | 2 | -18.7 | | [-22.7, -13.4] | [-20.6, -18.7] | [-23.3, -13.5] |
| Verden | DEPLOY − ADAF barrow | RAW | mean | -14.7 | | [-18.0, -11.8] | [-15.9, -14.7] | [-18.5, -12.0] |

The points equal BENCH_V0's v0 numbers (`detector_week1/RESULT.md:239-245`). Each system's own recall and
geography interval is in `BOOT_V0.json` `systems_point` and `systems_geo_ci95`.

**Supplementary: DEPLOY built from each J seed, mean over budgets.** These are not the registered system.

| Decoder | Contrast | J_02 | J_03 = S\* | J_04 | J_05 | J_06 |
|---|---|---:|---:|---:|---:|---:|
| NMS60 | DEPLOY − ADAF AO | +2.3 | +3.5 | +0.7 | +2.3 | +2.2 |
| RAW | DEPLOY − ADAF AO | +0.1 | +1.3 | −1.1 | +0.5 | +0.5 |
| NMS60 | DEPLOY − ADAF barrow | −9.7 | −8.5 | −11.3 | −9.7 | −9.8 |
| RAW | DEPLOY − ADAF barrow | −15.9 | −14.7 | −17.0 | −15.5 | −15.4 |

The registered S\* gives the best DEPLOY in all four rows. The selection-aware training interval therefore lies at
or below the point.

**Underfill.** ADAF barrow in Verden supplies 688 (NMS60) or 794 (RAW) proposals against K = 845 at b = 2. In the
replicates it is underfilled 99.9 % (NMS60) and 78.7 % (RAW) of the time at b = 2. In Oldenburg the figures are
27.2 % and 1.1 % at b = 2. No other system is underfilled.

**Grid sensitivity (95 % intervals).**

| Region | Contrast | Decoder | b | 2 km geography | 4 km geography | Offset geography | 2 km joint | 4 km joint | Offset joint |
|---|---|---|---|---|---|---|---|---|---|
| Oldenburg | A − C | NMS60 | 0.5 | [+2.0, +4.5] | [+2.0, +4.3] | [+2.0, +4.5] | [+1.8, +4.6] | [+1.8, +4.5] | [+1.8, +4.6] |
| Oldenburg | A − C | NMS60 | mean | [+3.1, +5.7] | [+3.2, +5.6] | [+3.2, +5.6] | [+2.9, +5.8] | [+3.0, +5.8] | [+3.1, +5.8] |
| Oldenburg | A − ADAF barrow | NMS60 | 0.5 | [-0.0, +2.7] | [+0.1, +2.8] | [-0.1, +2.6] | [-0.3, +2.9] | [-0.3, +3.0] | [-0.4, +2.9] |
| Oldenburg | A − ADAF barrow | NMS60 | mean | [-0.6, +1.9] | [-0.5, +2.0] | [-0.5, +1.8] | [-1.0, +2.1] | [-1.0, +2.2] | [-0.9, +2.1] |
| Oldenburg | A − C | RAW | 0.5 | [+2.2, +5.0] | [+2.2, +4.8] | [+2.2, +5.1] | [+2.0, +5.3] | [+2.0, +5.0] | [+2.0, +5.3] |
| Oldenburg | A − C | RAW | mean | [+4.7, +8.0] | [+4.9, +7.7] | [+4.8, +8.0] | [+4.4, +8.2] | [+4.5, +8.0] | [+4.5, +8.3] |
| Oldenburg | A − ADAF barrow | RAW | 0.5 | [-0.4, +2.5] | [-0.1, +2.5] | [-0.4, +2.4] | [-0.7, +2.9] | [-0.6, +3.0] | [-0.7, +2.9] |
| Oldenburg | A − ADAF barrow | RAW | mean | [-1.6, +1.4] | [-1.6, +1.6] | [-1.5, +1.2] | [-2.4, +1.9] | [-2.4, +2.1] | [-2.3, +1.9] |
| Verden | DEPLOY − ADAF AO | NMS60 | 0.5 | [+1.1, +4.0] | [+1.2, +4.6] | [+1.1, +4.4] | [+1.0, +3.9] | [+1.1, +4.4] | [+1.0, +4.3] |
| Verden | DEPLOY − ADAF AO | NMS60 | mean | [+2.4, +4.9] | [+2.4, +5.3] | [+2.4, +5.3] | [+1.5, +4.8] | [+1.5, +5.1] | [+1.4, +5.2] |
| Verden | DEPLOY − ADAF barrow | NMS60 | 0.5 | [-12.0, -5.9] | [-12.4, -6.0] | [-11.9, -5.9] | [-12.1, -5.9] | [-12.4, -6.1] | [-12.0, -6.0] |
| Verden | DEPLOY − ADAF barrow | NMS60 | mean | [-10.8, -6.3] | [-10.7, -6.0] | [-10.6, -5.7] | [-11.4, -6.5] | [-11.3, -6.2] | [-11.3, -5.9] |
| Verden | DEPLOY − ADAF AO | RAW | 0.5 | [-0.2, +3.2] | [+0.0, +3.4] | [+0.0, +3.3] | [-0.4, +3.1] | [-0.3, +3.2] | [-0.2, +3.1] |
| Verden | DEPLOY − ADAF AO | RAW | mean | [-0.4, +3.0] | [-0.1, +3.1] | [-0.2, +3.3] | [-1.0, +2.9] | [-0.6, +3.0] | [-0.6, +3.1] |
| Verden | DEPLOY − ADAF barrow | RAW | 0.5 | [-14.3, -7.0] | [-15.1, -7.3] | [-14.5, -7.2] | [-14.5, -7.2] | [-15.2, -7.4] | [-14.8, -7.3] |
| Verden | DEPLOY − ADAF barrow | RAW | mean | [-18.0, -11.8] | [-18.4, -11.4] | [-17.5, -11.4] | [-18.5, -12.0] | [-18.9, -11.7] | [-17.8, -11.5] |

**Reading Task 2**

- **A − C (Oldenburg).** Positive at every budget, under both decoders, on all three grids and under all three
  resampling schemes. Every lower bound is ≥ +1.8 points. The direction of the original ADAPT_GAIN survives v0
  scoring with uncertainty. The effect is 3-9 points, not 14.
- **A − ADAF barrow (Oldenburg).** Not distinguished from zero.
  - Every geography and joint interval spans 0, on every grid and decoder. At b = 0.5 NMS60 the geography
    interval reaches −0.0, and it touches +0.1 on the 4 km grid.
  - Points run from +1.2 (b = 0.5) to −1.5 (RAW, b = 2). The advantage, if any, is confined to the tightest budget
    and shrinks or reverses as the budget grows.
  - Neither superiority nor equivalence is shown.
- **DEPLOY − ADAF barrow (Verden).** Negative everywhere, with upper bounds ≤ −3.9 points. ADAF barrow's lead is
  robust to grid, decoder and resampling. It is largest under RAW (−14.7 mean).
- **DEPLOY − ADAF AO (Verden).** Under NMS60 it is positive with uncertainty: joint lower bound +1.5 for the mean
  and +1.0 at b = 0.5. Under RAW it is not distinguished from zero.
- **Training-only intervals rest on 3 (Oldenburg) or 5 (Verden) runs.**
  - With three families, the training-only interval is the range of resampled 3-seed means. For example, RAW
    b = 2 A − ADAF barrow [−3.9, −0.3] only says that all three seeds are negative there.
  - Per R3-2 (`ROUND3_REPLY.md:271`), per-seed values and geography-only intervals are the main presentation, and
    joint intervals are qualified supplements.

## Task 3: budget ceilings, supply and pool headroom (R3-1)

Columns:

- **N:** eligible entities.
- **K** = floor(b·A).
- **M:** one-to-one matches in the top K.
- **R** = M/N.
- **c** = min(1, K/N).
- **C:** maximum matching cardinality over the complete frozen pool.
- **R_oracle** = min(K, C)/N.
- **Supplied:** the pool size.
- **Unused slots** = max(0, K − supplied). Submitted unmatched = min(K, supplied) − M is in the JSON.

Per-budget values are given as b = 0.5 / 1 / 2.

**Oldenburg:** N = 959; K = 225 / 451 / 902; c = 0.235 / 0.470 / 0.941.

| System | Decoder | Supplied | C | M | R | Matches per 100 slots | Unused | R_oracle |
|---|---|---:|---:|---|---|---|---|---|
| ADAF AO | NMS60 | 1068 | 155 | 37 / 81 / 142 | 0.039 / 0.084 / 0.148 | 16.4 / 18.0 / 15.7 | 0 / 0 / 0 | 0.162 / 0.162 / 0.162 |
| ADAF AO | RAW | 1144 | 193 | 38 / 90 / 166 | 0.040 / 0.094 / 0.173 | 16.9 / 20.0 / 18.4 | 0 / 0 / 0 | 0.201 / 0.201 / 0.201 |
| ADAF barrow | NMS60 | 947 | 208 | 90 / 153 / 207 | 0.094 / 0.160 / 0.216 | 40.0 / 33.9 / 22.9 | 0 / 0 / 0 | 0.217 / 0.217 / 0.217 |
| ADAF barrow | RAW | 1164 | 285 | 95 / 173 / 256 | 0.099 / 0.180 / 0.267 | 42.2 / 38.4 / 28.4 | 0 / 0 / 0 | 0.235 / 0.297 / 0.297 |
| CONT | NMS60 | 4395 | 209 | 76 / 114 / 145 | 0.079 / 0.119 / 0.151 | 33.8 / 25.3 / 16.1 | 0 / 0 / 0 | 0.218 / 0.218 / 0.218 |
| CONT | RAW | 4658 | 216 | 76 / 116 / 147 | 0.079 / 0.121 / 0.153 | 33.8 / 25.7 / 16.3 | 0 / 0 / 0 | 0.225 / 0.225 / 0.225 |
| A_02 | NMS60 | 6021 | 278 | 102 / 163 / 204 | 0.106 / 0.170 / 0.213 | 45.3 / 36.1 / 22.6 | 0 / 0 / 0 | 0.235 / 0.290 / 0.290 |
| A_02 | RAW | 7682 | 409 | 104 / 185 / 252 | 0.108 / 0.193 / 0.263 | 46.2 / 41.0 / 27.9 | 0 / 0 / 0 | 0.235 / 0.426 / 0.426 |
| A_03 | NMS60 | 4959 | 268 | 110 / 165 / 210 | 0.115 / 0.172 / 0.219 | 48.9 / 36.6 / 23.3 | 0 / 0 / 0 | 0.235 / 0.279 / 0.279 |
| A_03 | RAW | 6125 | 388 | 116 / 179 / 253 | 0.121 / 0.187 / 0.264 | 51.6 / 39.7 / 28.0 | 0 / 0 / 0 | 0.235 / 0.405 / 0.405 |
| A_04 | NMS60 | 1212 | 204 | 93 / 148 / 191 | 0.097 / 0.154 / 0.199 | 41.3 / 32.8 / 21.2 | 0 / 0 / 0 | 0.213 / 0.213 / 0.213 |
| A_04 | RAW | 1298 | 249 | 95 / 151 / 219 | 0.099 / 0.157 / 0.228 | 42.2 / 33.5 / 24.3 | 0 / 0 / 0 | 0.235 / 0.260 / 0.260 |
| C_02 | NMS60 | 3461 | 197 | 77 / 114 / 149 | 0.080 / 0.119 / 0.155 | 34.2 / 25.3 / 16.5 | 0 / 0 / 0 | 0.205 / 0.205 / 0.205 |
| C_02 | RAW | 3593 | 206 | 77 / 115 / 150 | 0.080 / 0.120 / 0.156 | 34.2 / 25.5 / 16.6 | 0 / 0 / 0 | 0.215 / 0.215 / 0.215 |
| C_03 | NMS60 | 3335 | 213 | 79 / 126 / 166 | 0.082 / 0.131 / 0.173 | 35.1 / 27.9 / 18.4 | 0 / 0 / 0 | 0.222 / 0.222 / 0.222 |
| C_03 | RAW | 3543 | 227 | 79 / 126 / 169 | 0.082 / 0.131 / 0.176 | 35.1 / 27.9 / 18.7 | 0 / 0 / 0 | 0.235 / 0.237 / 0.237 |
| C_04 | NMS60 | 3388 | 202 | 61 / 110 / 148 | 0.064 / 0.115 / 0.154 | 27.1 / 24.4 / 16.4 | 0 / 0 / 0 | 0.211 / 0.211 / 0.211 |
| C_04 | RAW | 3524 | 212 | 61 / 110 / 150 | 0.064 / 0.115 / 0.156 | 27.1 / 24.4 / 16.6 | 0 / 0 / 0 | 0.221 / 0.221 / 0.221 |
| J_02 | NMS60 | 3623 | 199 | 79 / 110 / 145 | 0.082 / 0.115 / 0.151 | 35.1 / 24.4 / 16.1 | 0 / 0 / 0 | 0.208 / 0.208 / 0.208 |
| J_02 | RAW | 3827 | 207 | 79 / 110 / 146 | 0.082 / 0.115 / 0.152 | 35.1 / 24.4 / 16.2 | 0 / 0 / 0 | 0.216 / 0.216 / 0.216 |
| J_03 | NMS60 | 3526 | 204 | 74 / 112 / 149 | 0.077 / 0.117 / 0.155 | 32.9 / 24.8 / 16.5 | 0 / 0 / 0 | 0.213 / 0.213 / 0.213 |
| J_03 | RAW | 3716 | 219 | 75 / 114 / 150 | 0.078 / 0.119 / 0.156 | 33.3 / 25.3 / 16.6 | 0 / 0 / 0 | 0.228 / 0.228 / 0.228 |
| J_04 | NMS60 | 3742 | 197 | 60 / 104 / 137 | 0.063 / 0.108 / 0.143 | 26.7 / 23.1 / 15.2 | 0 / 0 / 0 | 0.205 / 0.205 / 0.205 |
| J_04 | RAW | 3950 | 212 | 60 / 104 / 140 | 0.063 / 0.108 / 0.146 | 26.7 / 23.1 / 15.5 | 0 / 0 / 0 | 0.221 / 0.221 / 0.221 |

**Verden:** N = 728; K = 211 / 422 / 845; c = 0.290 / 0.580 / 1.000. The other J seeds and DEPLOY variants are in
the JSON.

| System | Decoder | Supplied | C | M | R | Matches per 100 slots | Unused | R_oracle |
|---|---|---:|---:|---|---|---|---|---|
| ADAF AO | NMS60 | 1001 | 145 | 39 / 77 / 138 | 0.054 / 0.106 / 0.190 | 18.5 / 18.2 / 16.3 | 0 / 0 / 0 | 0.199 / 0.199 / 0.199 |
| ADAF AO | RAW | 1052 | 171 | 40 / 82 / 153 | 0.055 / 0.113 / 0.210 | 19.0 / 19.4 / 18.1 | 0 / 0 / 0 | 0.235 / 0.235 / 0.235 |
| ADAF barrow | NMS60 | 688 | 213 | 119 / 184 / 213 | 0.163 / 0.253 / 0.293 | 56.4 / 43.6 / 25.2 | 0 / 0 / **157** | 0.290 / 0.293 / 0.293 |
| ADAF barrow | RAW | 794 | 289 | 123 / 212 / 289 | 0.169 / 0.291 / 0.397 | 58.3 / 50.2 / 34.2 | 0 / 0 / **51** | 0.290 / 0.397 / 0.397 |
| CONT | NMS60 | 3851 | 226 | 78 / 113 / 153 | 0.107 / 0.155 / 0.210 | 37.0 / 26.8 / 18.1 | 0 / 0 / 0 | 0.290 / 0.310 / 0.310 |
| CONT | RAW | 4043 | 244 | 79 / 116 / 158 | 0.109 / 0.159 / 0.217 | 37.4 / 27.5 / 18.7 | 0 / 0 / 0 | 0.290 / 0.335 / 0.335 |
| S\* (J_03) | NMS60 | 3059 | 242 | 86 / 131 / 165 | 0.118 / 0.180 / 0.227 | 40.8 / 31.0 / 19.5 | 0 / 0 / 0 | 0.290 / 0.332 / 0.332 |
| S\* (J_03) | RAW | 3194 | 263 | 86 / 133 / 170 | 0.118 / 0.183 / 0.234 | 40.8 / 31.5 / 20.1 | 0 / 0 / 0 | 0.290 / 0.361 / 0.361 |
| DEPLOY | NMS60 | 3485 | 249 | 56 / 110 / 164 | 0.077 / 0.151 / 0.225 | 26.5 / 26.1 / 19.4 | 0 / 0 / 0 | 0.290 / 0.342 / 0.342 |
| DEPLOY | RAW | 4060 | 289 | 52 / 98 / 153 | 0.071 / 0.135 / 0.210 | 24.6 / 23.2 / 18.1 | 0 / 0 / 0 | 0.290 / 0.397 / 0.397 |

**Reading Task 3**

1. **One-to-one recoverability is far below coverage.** Each single pool's C (145-409) is much smaller than the
   union coverage week 1 reported (Verden RAW union 430, `detector_week1/RESULT.md:113`). R_oracle at b = 2
   ranges from 0.16 to 0.43. The pools, not the ranking, cap recall at generous budgets.
2. **Two different limits apply.**
   - At b = 0.5 nearly every system is far below the cap c (R 0.04-0.17 against 0.235/0.290). Ranking is the
     binding constraint: oracle ranking would reach the cap for A (Oldenburg, RAW and two NMS60 seeds) and for
     ADAF barrow, CONT, S\* and DEPLOY (Verden).
   - At b = 2, M approaches C for ADAF barrow (Verden: M = C = 213 under NMS60 and 289 under RAW) while 51-157
     slots stay unused. ADAF barrow's binding limit there is pool size.
3. **RAW raises C most for barrow-targeted pools.** A_02 rises from 278 to 409, ADAF barrow from 208/213 to
   285/289, and the general pools (CONT, C, J, S\*) only by 7-21. The NMS loss is concentrated in the models that
   respond to individual neighbouring barrows.
4. **Matches per 100 slots.** At b = 0.5, ADAF barrow and A reach 40-58 per 100 slots. AO reaches 16-19.

## Task 4: IGNORE under v0 (R3-7)

**Interpretations** (`ignore_v0.py:1-20`). All use the same 2 km blocks, weights and training draws as Task 2.

| Interpretation | IGNORE set | Proposals | Scored area / K | Entities |
|---|---|---|---|---|
| (i) Historical | None inside positive cores. Background boxes already exclude IGNORE (`confirm_build.py:247`; `step0.py:706`; `region_score.py:157-169`). | All charged | Full | All reachable |
| (ii) Literal rule 6 (`FRESH1_PREPIXEL.md:30`, which explicitly lists Grabhügelfeld; `adapt_nds PACKET.md:26-28`) | Every register record that is not an eligible entity, plus edge-supplement records (`step0.py:661-665`), buffered 30 m | Removed before K ("neither hits nor false alarms") | IGNORE area removed | N kept; unreachable entities counted |
| (iii) Explicit precedence | As (ii), minus Grabhügelfeld and Grabungsschutzgebiet polygons; every eligible entity's 15 m reach zone overrides the remaining IGNORE | Removed before K | IGNORE area removed | All reachable by construction |

IGNORE records: Oldenburg 1,203, including 101 Grabhügelfeld and 1 Grabungsschutzgebiet; Verden 1,535, including
73 Grabhügelfeld (district and edge supplement).

| Region | Interpretation | Scored km² | K (0.5 / 1 / 2) | Unreachable entities |
|---|---|---:|---|---:|
| Oldenburg | (i) historical | 451.12 | 225 / 451 / 902 | 0 |
| Oldenburg | (ii) literal | 440.33 | 220 / 440 / 880 | **665 of 959 (69 %)** |
| Oldenburg | (iii) precedence | 450.22 | 225 / 450 / 900 | 0 |
| Verden | (i) historical | 422.84 | 211 / 422 / 845 | 0 |
| Verden | (ii) literal | 420.55 | 210 / 420 / 841 | **445 of 728 (61 %)** |
| Verden | (iii) precedence | 422.18 | 211 / 422 / 844 | 0 |

**Key contrasts.** Points and 95 % geography-only intervals, with joint intervals in braces; percentage points.

| Contrast | Decoder | (i) b = 0.5 | (i) mean | (ii) b = 0.5 | (ii) mean | (iii) b = 0.5 | (iii) mean |
|---|---|---|---|---|---|---|---|
| Oldenburg A − C | NMS60 | +3.1 [+2.0, +4.5] {+1.8, +4.6} | +4.1 [+3.1, +5.7] {+2.9, +5.8} | +2.0 [+1.1, +3.4] {+0.9, +3.6} | +2.3 [+1.3, +3.8] {+1.2, +4.0} | +3.1 [+2.0, +4.5] {+1.8, +4.7} | +4.1 [+3.1, +5.8] {+2.9, +5.9} |
| Oldenburg A − C | RAW | +3.4 [+2.2, +5.0] {+2.0, +5.3} | +6.0 [+4.7, +8.0] {+4.4, +8.2} | +2.1 [+1.1, +3.7] {+1.0, +3.8} | +2.8 [+1.5, +4.8] {+1.4, +4.9} | +3.5 [+2.2, +5.1] {+2.0, +5.3} | +6.1 [+4.7, +8.0] {+4.4, +8.3} |
| Oldenburg A − ADAF barrow | NMS60 | +1.2 [−0.0, +2.7] {−0.3, +2.9} | +0.4 [−0.6, +1.9] {−1.0, +2.1} | +0.7 [−0.2, +1.8] {−0.3, +1.9} | **+1.0 [+0.1, +2.0]** {−0.0, +2.2} | +1.1 [+0.0, +2.7] {−0.3, +3.0} | +0.4 [−0.6, +1.9] {−1.0, +2.1} |
| Oldenburg A − ADAF barrow | RAW | +1.0 [−0.4, +2.5] {−0.7, +2.9} | −0.2 [−1.6, +1.4] {−2.4, +1.9} | +0.5 [−0.4, +1.6] {−0.6, +1.7} | +0.9 [−0.0, +2.0] {−0.2, +2.2} | +0.9 [−0.3, +2.5] {−0.8, +3.0} | −0.2 [−1.5, +1.4] {−2.4, +1.9} |
| Verden DEPLOY − ADAF AO | NMS60 | +2.3 [+1.1, +4.0] | +3.5 [+2.4, +4.9] | +1.8 [+0.4, +2.9] | +2.7 [+1.6, +3.8] | +2.3 [+1.1, +4.0] | +3.5 [+2.4, +5.0] |
| Verden DEPLOY − ADAF AO | RAW | +1.6 [−0.2, +3.2] | +1.3 [−0.4, +3.0] | +1.1 [−0.2, +2.3] | +1.7 [+0.4, +2.8] | +1.2 [−0.2, +3.1] | +1.1 [−0.4, +3.1] |
| Verden DEPLOY − ADAF barrow | NMS60 | −8.7 [−12.0, −5.9] | −8.5 [−10.8, −6.3] | −5.6 [−7.5, −3.6] | −3.6 [−5.0, −2.1] | −8.9 [−12.2, −6.0] | −8.6 [−10.8, −6.3] |
| Verden DEPLOY − ADAF barrow | RAW | −9.8 [−14.3, −7.0] | −14.7 [−18.0, −11.8] | −6.3 [−8.5, −4.4] | −5.2 [−6.9, −3.6] | −10.2 [−14.6, −7.2] | −14.8 [−18.1, −11.8] |

Per-seed values, system recalls, and recall over reachable N under (ii) are in `IGNORE_V0.json`. Proposals
removed under (ii) are 75-439 per pool; under (iii) they are 6-31.

**Reading Task 4**

- **(ii) is unworkable as a benchmark definition.** It makes 61-69 % of the eligible barrows unreachable, because
  rule 6 names the cemetery polygons that contain them. It also shrinks every gap that rests on cemetery barrows:
  - A − C halves (+4.1 → +2.3, NMS60 mean).
  - DEPLOY − ADAF barrow shrinks from −8.5 to −3.6.
  - **A − ADAF barrow becomes marginally positive under NMS60 mean (+1.0, geography [+0.1, +2.0]; joint [−0.0, +2.2]).**

  This is the R3-7 warning in practice: next to a 0.4-point specialist difference, the reading of IGNORE decides
  the sign of the bound. It is not a basis for a claim, because the denominator is mostly unreachable.
- **(iii) reproduces (i) to within 0.4 points everywhere, with the same interval conclusions.** The one borderline
  bound, A − ADAF barrow NMS60 b = 0.5 geography, moves from −0.0 to +0.0. Once
  container polygons do not suppress eligible barrows, the residual IGNORE (other records near the barrows) removes
  6-31 proposals per pool. It changes no conclusion. This confirms the week-1 legacy finding of a 1-5-group residual
  (`detector_week1/RESULT.md:182-189`) under v0.

**Contemporaneous precedence statement: none found for FRESH-1 or ADAPT-NDS-1 scoring.** Searched the FRESH-1
packet, Step 0 and Step 2 documents, the freeze, the ADAPT-NDS-1 packet, BUILD_DECISIONS, the freeze, the
scorers, the HR1/HR3 packets and amendments, and the git log (grep for precedence / priority / IGNORE with
positive / core / scoring):

| Source (commit, UTC) | What it says | Bearing |
|---|---|---|
| `FRESH1_PREPIXEL.md:30` (3d629d0d, 2026-10-03 09:40Z; unchanged since) | Rule 6: IGNORE = every other record, Grabhügelfeld named; proposals there "neither hits nor false alarms" | No scope or precedence. The registered wording. |
| `adapt_nds_2026-10-05/PACKET.md:26-28` (c1cc68f1, 2026-10-05 00:41Z) | "FRESH-1 label rules 1-7 unchanged ... other records IGNORE +30 m" | No precedence |
| `hr3_ireland_2026-10-02/AMENDMENT_1.md:12-15` (11fef773, 2026-10-02 21:14Z) | IGNORE discs defined under "Background"; "the disc area is removed from the background area" | Scopes IGNORE to background construction, in the HR3 design FRESH-1 copies (`FRESH1_PREPIXEL.md:41`). Indirect, and not restated in FRESH-1. |
| `fr_eval_1_2026-10-04/LOCK_DECISIONS.md:8` (a20dcd74, 2026-10-03 22:43Z) | "Zone precedence: label zone first (hit candidate), then IGNORE ..." | An explicit precedence rule, but for a different packet (France). It is hit-zone precedence, not positive-tile precedence. It predates the FRESH-1 freeze (2026-10-04 17:51Z) and the ADAPT packet, but neither cites it. |
| `adapt_nds_2026-10-05/plan_fit.py:11`, `hr3 .../irl_masks.py:5` | "positives overwrite IGNORE" | Training masks only, not scoring |

Under the R3-7 standard (`ROUND3_REPLY.md:648-654`), the FRESH-1 / ADAPT-NDS-1 case is a
**protocol-implementation discrepancy**, not a documented "wording inconsistency". The frozen implementation
applies positive-tile precedence. No FRESH-1 or ADAPT-NDS-1 document states it. (iii) is offered as the explicit
future rule; it is a narrower precedence (reach zone, not whole tile).

## Task 5: register semantics of cemetery and protection polygons (R3-8 item 2)

Source: the local district snapshots of the Denkmalatlas Niedersachsen (`REGISTER_SEMANTICS.json`). Barrow
membership is tested by eligible-barrow centroid in polygon.

| | Verden Grabhügelfeld | Oldenburg Grabhügelfeld | Oldenburg Grabungsschutzgebiet |
|---|---:|---:|---:|
| Records | 60 | 96 (2 MultiPolygon) | 1 |
| Register class | Archäologie; archaeologischesDenkmal; DSchG ND (all) | same (all) | **Infofläche**; ausweisung "Grabungsschutzgebiete" (code 5); designation grabungsschutzgebiet |
| Area, median / max (ha) | 0.84 / 10.6 | 0.035 / 38.3 | 1,033 (one polygon) |
| Polygons containing 0 / 1 / 2-4 / 5-9 / ≥ 10 eligible barrows | 3 / 0 / 17 / 25 / 15 (max 20) | **46** / 3 / 17 / 10 / 20 (max 109) | – / – / – / – / 1 (89) |
| Eligible barrows inside any polygon (+30 m) | 431 (432) of 728 | 617 (688) of 959 | 89 of 959 |
| Descriptions present / stating a barrow count | 2 / 2 | 85 / 83 | 1 / 1, citing § 16 NDSchG |
| Stated count vs individual Grabhügel records inside | equal 1, more 1 | equal 23, more 54, fewer 6 (stated 1,534 vs 540 recorded) | stated about 600 vs 89 recorded |
| Eligible barrows inside sharing the polygon's site number (FStNr) | 55 of 431 | 43 of 617 | – |

Further Oldenburg findings:

- 11 cemeteries lie inside the protection area.
- 26 eligible barrows lie in the protection area but in no cemetery.
- 48 cemetery polygons share an identical area of 0.0112 ha (a disc of about 6 m radius). 46 of them, the 46
  empty polygons, contain no individual record. These are placeholder geometries, not boundaries.

**Reading Task 5**

1. **Grabhügelfeld records are archaeologically defined complexes.** The register lists them as archaeological
   monuments in their own right (Denkmal designation). Most carry a description that counts their mounds, and in
   Verden 57 of 60 contain 2-20 eligible barrows. They are not protection boundaries.
2. **They are not a clean complex endpoint as delivered.**
   - Oldenburg's 46 empty placeholder discs (of 48) contain no individual record.
   - Stated mound counts exceed the individually recorded barrows in 54 of 83 described Oldenburg cemeteries
     (1,534 against 540). Individual recording is incomplete.
   - Members are linked spatially, not by register number (55 of 431 and 43 of 617 share the site number).

   A complex-level endpoint is feasible for the Verden polygons, and for Oldenburg cemeteries with real
   boundaries (50 with ≥ 1 member). It needs a pre-stated rule for placeholder geometries.
3. **The Grabungsschutzgebiet is a legal protection boundary.**
   - Its class is "Infofläche", it is designated under § 16 NDSchG, and its area is 10.3 km².
   - It contains the district's largest cemetery polygon (38.3 ha, wholly inside) and 10 others, plus 26 barrows
     outside any cemetery.
   - It is not a cemetery boundary and should not define a complex (R3-8: "A protection boundary is not
     automatically a cemetery boundary", `ROUND3_REPLY.md:703`).
   - It must not suppress eligible barrows. Interpretation (iii) already excludes it.

## Task 6: adaptation and control against their common start J (R3-8 item 3)

| Contrast | Decoder | b | Point | Per family (02 / 03 / 04) | Geography | Training | Joint |
|---|---|---|---:|---|---|---|---|
| A − J | NMS60 | 0.5 | +3.2 | +2.4 / +3.8 / +3.4 | [+2.1, +4.7] | [+2.4, +3.8] | [+1.9, +5.1] |
| A − J | NMS60 | mean | +4.8 | +4.7 / +5.2 / +4.6 | [+3.7, +6.6] | [+4.6, +5.2] | [+3.6, +6.7] |
| C − J | NMS60 | 0.5 | +0.1 | −0.2 / +0.5 / +0.1 | [−0.3, +0.8] | [−0.2, +0.5] | [−0.6, +1.0] |
| C − J | NMS60 | mean | +0.7 | +0.2 / +1.3 / +0.6 | [+0.3, +1.2] | [+0.2, +1.3] | [+0.0, +1.6] |
| A − J | RAW | 0.5 | +3.5 | +2.6 / +4.3 / +3.6 | [+2.3, +5.2] | [+2.6, +4.3] | [+2.0, +5.7] |
| A − J | RAW | mean | +6.7 | +7.2 / +7.3 / +5.6 | [+5.3, +8.8] | [+5.6, +7.3] | [+5.1, +9.1] |
| C − J | RAW | 0.5 | +0.1 | −0.2 / +0.4 / +0.1 | [−0.3, +0.8] | [−0.2, +0.4] | [−0.6, +1.0] |
| C − J | RAW | mean | +0.7 | +0.2 / +1.2 / +0.6 | [+0.3, +1.2] | [+0.2, +1.2] | [+0.0, +1.5] |

Budgets 1 and 2 are in `BOOT_V0.json`. They show the same pattern: A − J +5.2 to +10.0; C − J +0.8 to +1.1.

- **A improved; C did not deteriorate.** C stays at J's level (+0.1 at b = 0.5, +0.7 mean, never negative beyond
  −0.6). So under v0 the A − C gap (+4.1 / +6.0) is essentially all adaptation gain (A − J +4.8 / +6.7).
- The legacy group metric shows the same split: A − J +16.8 and C − J +2.8
  (`adapt_nds_2026-10-05/RESULT.md:35-36`).
- **The adaptation tool should still return "retain the public specialist" here.** A moved from J's level up to
  ADAF barrow's level, not past it (Task 2): J_s mean 10.5-11.6 % against ADAF barrow 15.6 % (NMS60 mean). The gain
  is real and the specialist comparison remains a tie.

## Limitations

- These are conditional intervals for these landscapes, registers and checkpoints (`ROUND3_REPLY.md:93`). They
  are pointwise: not simultaneous over the 4 contrasts × 2 decoders × 4 summaries × 3 grids inspected.
- Training intervals rest on 3 (Oldenburg) or 5 (Verden) runs.
- Verden's 4 km and offset grids have J_labels 8-11, so those Verden intervals are fragile.
- Verden is previously evaluated development terrain. Oldenburg CONFIRM has been read many times since its frozen
  primary; v0 is a retrospective rescoring of both.
- The IGNORE sets of (ii) and (iii) are rebuilt from the local register snapshot and edge-supplement files with
  `make_valid` and 30 m buffers. The district part matches `step0.ignore_geometries` (`step0.py:661-665`): non-entity
  register records, which include the NODATA failures. This is not the frozen build's exact geometry object.
- The register-semantics regex for stated counts is approximate: first match, digits ≤ 3 places or German number
  words to twelve.

## Inputs not available (reported, not substituted)

- Verden has no A/C proposals in this task's scope, and the task did not ask for them. Verden ADAF enclosure and
  ringfort pools were never produced.
- No missing input blocked tasks 1-6.
