# D1 vs ADAF barrow on Verden: v0 uncertainty and pool terms (DESCRIPTIVE, 2026-10-07, Opus)

**Label: DESCRIPTIVE.** Stage 1's gate was point-only and is unchanged (`work/stage1_d1_v0_2026-10-07/FREEZE.md`,
verdict PASS). These intervals add uncertainty to that result. They do not re-decide it.

Source: GPT-6 Pro Round 4, R4-4 and R4-5 (`ROUND4_REPLY.md:248-306`), adopted in `ROUND4_REVIEW.md` item 7.

Capability line: reproducible detector improvement (regional adaptation, Lower Saxony barrows).

| File | Content |
|---|---|
| `desc_verden.py` → `RESULT.json` | Bootstrap, pool terms, v0.1 complex sensitivity (aggregates only) |
| `tables_desc.py` → `TABLES.md` | Every table, generated from `RESULT.json` |
| `run.log` | Runtime log (213 s) |

**Run conditions.**

- CPU only, WSL `/home/tee-/adaf-h2h-venv/bin/python -B`.
- Read-only on every input. No Oldenburg, reserved or TEST terrain was opened.

## Inputs and procedure

- **Proposals:** Stage 1's, unchanged. ADAF barrow (FRESH-1) and the ten D1 BARROW_L confirmation runs, seeds 05-14.
  Each file's sha256 was checked against Stage 1's provenance (`RESULT.json` `proposals_sha256`).
- **Manifest:** Verden individual manifest `a97b6124…`, N = 728, A = 422.843 km², K = 211 / 422 / 845.
- **Engine:** the frozen v0 bootstrap (`v0_uncertainty/engine.py`, `boot_v0.py`, `v0data.py`), imported unchanged.
  - Blocks: 2 km grid at (0, 0), primary; 4 km and 2 km offset (1000, 1000) as sensitivities. Cells are anchored on
    footprint centroids and merged by 15 m reach.
  - 5,000 replicates; pointwise percentile 95 % intervals.
- **Geography:** Multinomial(J; 1/J) block weights from the v0 stream `geo_verden`, the same weights as `BOOT_V0.json`.
  - Check: ADAF barrow's 2 km geography interval reproduces `BOOT_V0.json` exactly under both decoders.
- **Training:** the ten D1 runs drawn with replacement (10 draws) from a new named stream, `train_d1` =
  `SeedSequence(20261007).spawn(5)[4]`. The v0 streams used children 0-3.
  - ADAF barrow is one fixed public checkpoint.
  - These identical draws are frozen for the Oldenburg transfer check. A joint two-district summary therefore shares
    the checkpoint index (R4-3).
- **Contrasts:** geography-only uses the ten observed runs; training-only uses the point recalls; joint uses both. Each
  checkpoint is scored first, then averaged.
- **Checks passed:**
  - all-ones block weights reproduce every Stage 1 recall exactly, for all 11 systems, both decoders and all budgets
    and grids;
  - no proposal-reference edge crosses a block;
  - incremental C = scipy C for every pool;
  - every M_s(b) equals Stage 1's match count.

## Headline (2 km grid, percentage points, D1 ten-run mean − ADAF barrow)

| Decoder | b | Point | Positive runs | Geography only 95 % | Training only 95 % | **Joint 95 %** |
|---|---|---:|---:|---|---|---|
| **RAW** | **0.5** | **+6.25** | **10/10** | [+3.89, +8.52] | [+5.88, +6.65] | **[+3.89, +8.58]** |
| RAW | 1 | +9.37 | 10/10 | [+6.09, +12.84] | [+8.71, +10.05] | [+6.01, +12.98] |
| RAW | 2 | +12.46 | 10/10 | [+8.70, +16.63] | [+11.61, +13.39] | [+8.68, +16.77] |
| RAW | mean | +9.36 | | [+6.75, +12.20] | [+8.77, +9.98] | [+6.71, +12.25] |
| NMS60 | 0.5 | +5.10 | 10/10 | [+3.58, +6.82] | [+4.88, +5.33] | [+3.52, +6.85] |
| NMS60 | 1 | +5.58 | 10/10 | [+3.78, +7.29] | [+5.12, +6.03] | [+3.67, +7.36] |
| NMS60 | 2 | +7.32 | 10/10 | [+5.36, +9.47] | [+6.96, +7.64] | [+5.32, +9.57] |
| NMS60 | mean | +6.00 | | [+4.61, +7.48] | [+5.71, +6.26] | [+4.56, +7.53] |

**Grid sensitivity, RAW b = 0.5 joint:** 4 km [+3.74, +8.73]; offset [+3.38, +9.03]. Every interval on every grid,
decoder and budget excludes zero. The lowest lower bound anywhere is +3.08 (NMS60, b = 0.5, offset grid, joint). The
full grid table is in `TABLES.md`.

**Per run, RAW b = 0.5:** +5.22 to +7.42. Each run's own geography interval against ADAF barrow excludes zero; the
lowest lower bound is seed 12, +2.91. **Deployable checkpoint, seed 05: +5.63 [+3.56, +8.11]** (RAW b = 0.5,
geography only); +8.52 at b = 1 and +10.99 at b = 2.

## Pool terms (R4-5): C_s, G_ranking,s(b) = min(K_b, C_s) − M_s(b), G_pool,s = N − C_s

| System | Decoder | Supply | C_s | G_pool,s | G_ranking,s at b = 0.5 / 1 / 2 |
|---|---|---:|---:|---:|---|
| ADAF barrow | RAW | 794 | 289 | 439 | 88 / 77 / **0** |
| D1, ten runs (range) | RAW | 2,358–3,509 | **459–487** | **241–269** | **34–50** / 125–153 / 83–110 |
| ADAF barrow | NMS60 | 688 | 213 | 515 | 92 / 29 / **0** |
| D1, ten runs (range) | NMS60 | 1,967–2,831 | 293–311 | 417–435 | 50–58 / 69–90 / 32–43 |

The per-run rows are in `TABLES.md`. At b = 0.5 RAW the D1 mean G_ranking is 42.5 slots (5.8 points), against Pro's
estimate of ≈ 43 from 0.231 × 728.

**Reading.**

1. **D1's limit is ranking at every budget; ADAF barrow's is pool at b = 2.**
   - ADAF barrow reaches its whole-pool maximum at b = 2 (G_ranking = 0, RAW and NMS60).
   - Every D1 run still has 83–110 (RAW) recoverable matches unranked at b = 2, and 34–50 at b = 0.5.
2. **D1 raises recoverability, not only supply.**
   - RAW C_s is 459–487 against 289: +170 to +198 recoverable registered barrows.
   - G_pool falls from 439 to 241–269. The extra supply (3.0–4.4× ADAF's) carries matchable proposals, not only
     volume.
3. **The pool is still the larger gap in absolute terms.**
   - 241–269 registered barrows (33–37 %) remain unrecoverable by any ranking of any D1 pool (RAW).
   - Under v0 that is a register-recovery statement, not mound detection. Recording is incomplete (R4-5).
4. **NMS60 costs D1 most of its recoverability gain.** RAW C_s 459–487 falls to 293–311 under NMS60, while ADAF's falls
   from 289 to 213. Neighbouring barrows within 60 m are suppressed. Any decoder change should be tested first on the
   RAW pool (R4-5 decoder experiment; not started).

## Evaluator v0.1 sensitivity (frozen `0f3443ea`; applied after the freeze)

- **Barrow endpoint: identical to Stage 1** for all 11 systems. Verden has no void, and container records are metadata.
- **Member-supported complex endpoint.** 57 available complexes (2-20 eligible members each, median 7); one hit per
  complex; one complex per proposal. Point values:

| Decoder | b | ADAF barrow | D1 mean | Mean diff (points) | Range | Positive runs |
|---|---|---:|---:|---:|---|---:|
| RAW | 0.5 | 0.596 (34) | 0.712 | +11.58 | +5.26 to +14.04 | 10/10 |
| RAW | 1 | 0.789 (45) | 0.828 | +3.86 | 0.00 to +7.02 | 9/10 |
| RAW | 2 | 0.877 (50) | 0.925 | +4.74 | +1.75 to +7.02 | 10/10 |
| NMS60 | 0.5 | 0.596 (34) | 0.754 | +15.79 | +8.77 to +17.54 | 10/10 |
| NMS60 | 1 | 0.807 (46) | 0.861 | +5.44 | +1.75 to +10.53 | 10/10 |
| NMS60 | 2 | 0.877 (50) | 0.932 | +5.44 | +3.51 to +7.02 | 10/10 |

- **Reading.** The complex-level gain is largest at the tight budget, where D1's extra hits reach more distinct
  cemeteries (39–42 of 57 against 34, RAW).
  - At b ≥ 1 both systems reach most available complexes, and the differences are 1-6 complexes.
  - One D1 run ties ADAF barrow at RAW b = 1.
  - With 57 units, one complex is 1.75 points.
  - No interval was computed for the complex endpoint.

## What this licenses (R4-4)

> On training-disjoint Verden development terrain, D1 (ten runs from public ADAF barrow) exceeded unchanged ADAF barrow
> in registered-barrow recall at 211 total slots by +6.25 points. The 2 km block-bootstrap joint 95 % interval is
> [+3.9, +8.6]. The interval excludes zero on every grid, decoder and budget.

Qualifications:

- the recipe was selected with three earlier Verden development runs;
- Verden is previously evaluated development terrain;
- intervals are conditional on this landscape, register and comparator checkpoint, and are pointwise;
- the training-only interval reflects ten runs of one selected recipe.

Not licensed: transfer beyond Verden (that is the Oldenburg check), a locality effect (Stage 2 does not isolate it
either), or discovery yield.

## Limitations

- **Descriptive.** No gate depends on these intervals.
- **Unchecked coverage.** The coverage of the block bootstrap is unverified (`v0_uncertainty/RESULT.md:44`).
- **Fragile sensitivity grids.** Verden's 4 km and offset grids have J_labels 8-11, so those sensitivities are
  fragile. They agree with the primary grid here.
- **Fixed comparator.** ADAF barrow is a single checkpoint. Its run-to-run variability, had it been retrained, is not
  represented.
