# OLDENBURG-TRANSFER-1: a one-time transfer check of the ten D1 checkpoints (packet, 2026-10-07, Opus)

**Name:** *a prespecified transfer check on a previously inspected, training-disjoint district that was not used to
select this recipe.* Oldenburg is not untouched. It was ADAPT-NDS-1's CONFIRM district and has been rescored and
inspected since (v0, IGNORE readings, register semantics). Its earlier use may have influenced model-family choices
and general strategy. It did not select the D1 recipe.

**Status:**

- **Frozen** (`FREEZE.md`, `FREEZE.json`).
- **Not launched.** No D1 proposal exists on Oldenburg, and none was made here. The one GPU step is
  `INFERENCE_PLAN.md`; the coordinator launches it.
- `score_oldenburg.py` runs once afterwards.

**Source:** GPT-6 Pro Round 4, R4-3 and R4-4 (`work/gpt6_pro_collab_2026-10-07/ROUND4_REPLY.md:185-279`), adopted in
`ROUND4_REVIEW.md` items 5 and 6.

**Capability line:** generalisation. Does a regional adaptation recipe's advantage over the public specialist extend
to a second district of the same state?

**Question.** Does the fixed D1 recipe's Verden advantage over unchanged ADAF barrow (Stage 1: +6.25 points at
211 slots, RAW, 10/10 runs) extend to Oldenburg?

## Frozen rule (Pro R4-3 items 1-9)

| # | Item | Implementation |
|---|---|---|
| 1 | All ten existing D1 checkpoints; no replacement or selection | D1F1_BARROW_L seeds 2026092805-14, the hashes in `INFERENCE_PLAN.md`, final weights of D1-COLLAPSE-1 session 3 (SESSION_VERIFIED PASS). A missing or failed run is reported. It is not replaced. |
| 2 | Unchanged ADAF barrow as comparator | Checkpoint `51b679fa…`. Its existing Oldenburg proposals `outputs/adapt_nds_2026-10-05/score/confirm/individual/PROPOSALS_ADAF_barrow_PRIVATE.json`, sha256 `21421a9c…` (section "Comparator"). |
| 3 | Same raw decoder and exact primary scorer as Stage 1 | `work/stage1_d1_v0_2026-10-07/score_v0.py`, imported unchanged (`v0`, `raw_pool`, `match_count`). RAW primary; NMS60 secondary. |
| 4 | Fixed Oldenburg entity, area, mask and proposal-extraction manifest | Entities: individual manifest `3531675c…`, N = 959 eligible individual barrows. Area: the union of the 699 tile cores, 451.117 km²; K = 225 / 451 / 902. Mask: CONFIRM inside masks `2a47e064…` under staging plan `e37e3e01…` (DGM1 bytes pinned). Extraction: inference manifest `8fce717e…` with `infer_irl.py` `f3855551…` (run + assemble). |
| 5 | Legitimacy check with the Verden logic | `legit.py` → `LEGITIMACY.json`: **PASS** (below). |
| 6 | Primary gate | Mean over the ten runs of (run recall − ADAF barrow recall) at b = 0.5, RAW, **≥ +0.030**, unrounded (float guard 1e-12, as Stage 1), **and ≥ 8/10 differences strictly > 0**. In matches: a mean of ≥ 28.77 more of 959. |
| 7 | Secondary reporting | Every seed; b = 1 and 2; NMS60; supply; C_s, G_ranking,s(b), G_pool,s for each run and ADAF barrow; the frozen v0 bootstrap (below); the deployable checkpoint's own result; the joint two-district summary; v0.1 sensitivity; ADAF parity. |
| 8 | Separate "clear transfer improvement" | Gate PASS **and** the primary joint 95 % lower bound > 0: 2 km grid, RAW, b = 0.5, unrounded. |
| 9 | No modification of recipe or decoder followed by another score presented as this check | This packet scores once. A changed recipe, decoder, checkpoint set, entity set or scorer is a new experiment with a new name. A technical inference failure before scoring is rerun unchanged (`INFERENCE_PLAN.md`). |

Keep Stage 1's point-only pass intact. Nothing here changes the Verden result.

## Comparator: reuse verified by hash

The existing ADAF barrow Oldenburg proposals are reused because every element of their production equals the D1 pass
fixed here.

- **Inference code:** ADAPT-NDS-1 session 1 made them with `infer_irl.py` `f38555510647…`
  (`run_s1_20261005T043612Z/v1/results/chain.log` line 116). It is pinned identically in the ADAPT-NDS-1 and D1
  freezes and is unchanged on disk.
- **Manifest:** the same inference manifest, `8fce717e…`.
- **Input bytes:** the same pinned DGM1 bytes (the staging plan pins sha256 per file).
- **Environment:** the same VM environment freeze, `9d23a2c7…`, recorded by both sessions.
- **Content:** the score copy's objects equal the session file's (`77b68e79…`, SESSION_VERIFIED pin) exactly.
- **Determinism:** G4 inference is deterministic (EQ-1).

ADAF barrow is also re-inferred in the same pass as a **technical parity check**. It is reported and never substituted
for the primary comparator. PASS means:

- every proposal is within EQ-1 tolerance (position < 1e-6 m, |Δscore| ≤ 2e-6);
- v0 match counts are identical.

A DEVIATION is reported beside the result.

Reproduction: the frozen comparator under this scorer equals BENCH_V0 v0 exactly (`work/evaluator_v0_1_2026-10-07/CHECKS.json`).

| Decoder | b = 0.5 | b = 1 | b = 2 |
|---|---|---|---|
| RAW | 0.0991 (95) | 0.1804 (173) | 0.2669 (256) |
| NMS60 | 0.0938 (90) | 0.1595 (153) | 0.2158 (207) |

Ceiling c = min(1, K/N) = 0.235 / 0.470 / 0.941. At RAW, ADAF barrow's whole-pool maximum is C = 285 with
G_ranking(0.5) = 130 (`DRYRUN_FIXTURE.json`, comparator rows).

## Legitimacy check: PASS (run 2026-10-07, `LEGITIMACY.json`)

The check uses the same logic as `work/stage1_d1_v0_2026-10-07/step0.py`.

| Check | Value |
|---|---:|
| Every run trained only on the Lower Saxony FIT table (sha256 `58d6aa16…`), 5,116 of 5,116 updates; session pins match | yes (10/10) |
| FIT windows (852 + 852, three FIT districts, union 280.3 km²) intersecting the Oldenburg district | 0 |
| Min distance, FIT windows → Oldenburg district | 48,281 m |
| → Oldenburg scored cores | 50,066 m |
| → Oldenburg inference read windows (with margins) | 49,289 m |
| → Oldenburg entity footprints | 50,669 m |
| Register records shared: FIT members vs Oldenburg entities; FIT members + incidentals vs the Oldenburg register | 0; 0 |
| 1 km DGM1 squares shared: FIT pins vs Oldenburg staging plan; D1 training fetch log (2,806 files) vs the plan (1,011 files); vs the earlier CONFIRM fetch log | 0; 0; 0 |
| D1 chain logs (sessions 1, 3, 4) staging or reading Oldenburg | 0, 0, 0 |
| D1 freeze pins naming CONFIRM or Oldenburg files (of 148) | 0 |

- Training pixels are confined to the 423 m FIT windows; pixel centres outside the FIT union are NoData.
- With about 48 km of separation, every Oldenburg tile and entity is eligible under any context buffer. No split is
  drawn.
- The recipe was selected on Verden DEV (seeds 02-04; `work/d1_collapse_2026-10-06/PACKET.md` names CONFIRM/Oldenburg
  as not used).

## Uncertainty (frozen; descriptive except item 8)

- **Engine:** the frozen v0 bootstrap (`v0_uncertainty/engine.py`, `boot_v0.py`), unchanged.
  - Blocks: 2 km grid at (0, 0), primary; 4 km and 2 km offset (1000, 1000) as sensitivities.
  - 5,000 replicates; pointwise percentile 95 %.
- **Geography:** v0 stream `geo_oldenburg`, the same block weights as `BOOT_V0.json`. Oldenburg's 2 km grid has
  J = 146 blocks and J_labels 23.8.
- **Training:** the ten D1 runs drawn with replacement from stream `train_d1` = `SeedSequence(20261007).spawn(5)[4]`.
  - These are the identical draws used for the Verden descriptive (`work/d1_verden_descriptive_2026-10-07/desc_verden.py`
    `train_draws`).
  - ADAF barrow is fixed.
- **Contrasts:** geography-only (ten observed runs), training-only and joint, for every b and decoder. Per-run points
  and per-run geography intervals are included.
- **Joint two-district summary (descriptive):** the equal-weight mean of the Verden and Oldenburg D1 − ADAF barrow
  differences.
  - **The checkpoint index is shared across regions in the training resampling**: the same `train_d1` draw is applied
    to both regions in each replicate, because the identical checkpoint is evaluated in both.
  - Geography is resampled separately per region: `geo_verden`, `geo_oldenburg`, 2 km grid.
  - It is not a gate.

## Sensitivities

- **Evaluator v0.1** (`work/evaluator_v0_1_2026-10-07`, frozen `0f3443ea`): role-based records; Oldenburg void table
  empty.
  - The barrow endpoint therefore equals the primary by construction. The scorer recomputes and reports it.
  - **Member-supported complex endpoint**, computed from the register and manifest before any D1 output exists:
    - 93 available complexes (49 with valid extent; 45 without extent, whose members come only from register
      site-number links);
    - 8 unavailable;
    - 296 eligible barrows unassigned.
  - The 6 m discs are not extents.
  - The 10.3 km² legal protection polygon is not a complex; its barrows outside cemeteries stay unassigned.
- **ADAF parity:** see "Comparator".

## Reading table (Pro's sentences, fixed precedence; `score_oldenburg.reading`)

| Condition (unrounded) | Sentence |
|---|---|
| Gate PASS and joint lower bound > 0 | "The prespecified transfer gate passed on Oldenburg." + clear transfer improvement |
| Gate PASS and joint lower bound ≤ 0 | "The prespecified transfer gate passed on Oldenburg." + "Point pass with interval spanning zero: the practical gate passed; the uncertainty criterion … was not met." |
| Mean ≤ 0 | "The Verden benefit did not extend to Oldenburg under this comparison." |
| Mean ≥ +0.030 but fewer than 8/10 positive | "The transfer result did not meet the replication criterion." |
| 0 < mean < +0.030 | "A smaller positive transfer effect was estimated; the practical gate was not met." (+ "The replication criterion was also not met." if fewer than 8/10 positive) |

None of these changes the historical Verden result.

The two studies answer different questions:

- **Oldenburg:** does this fixed recipe's advantage extend to another district?
- **Stage 2:** how does the recipe compare with the specified replay package?

A disappointing Oldenburg score must not trigger a more favourable Stage 2 control. Stage 2 is frozen separately by
its own packet.

## Deployable checkpoint (frozen before any Oldenburg D1 score)

**`D1F1_BARROW_L_2026092805`**, sha256 `d9ece1fb023d2daf72ab0e5f789b5b89a27af73a28307e6260e4ee2fb5ac501e`
(319,970,247 bytes; registry id `D1-F1_BARROW_L_2026092805`; archived in Drive bundle k1).

- **Selection rule:** the earliest run in immutable run-ID order (R4-4; `ROUND4_REVIEW.md` item 6). It was not chosen
  by score.
  - On Verden at b = 0.5 RAW it scores +5.63 points, geography 95 % [+3.56, +8.11]
    (`work/d1_verden_descriptive_2026-10-07/RESULT.md`). That ties seed 06 for second-lowest of ten; the range is
    +5.22 to +7.42.
- **Scope:** development-selected candidate for Lower Saxony barrow surveys under conditions resembling the evaluated
  data (LGLN DGM1, Denkmalatlas register barrows).
  - ADAF barrow remains the fallback elsewhere.
  - It is not an English, Irish or mixed-class policy; earlier adaptations forgot outside their region.
- **Records:** `DETECTOR_REGISTRY.json` (operating choice `lower_saxony_barrow_surveys`) and `DETECTOR_HISTORY.md`.
- **Reporting:** the Oldenburg result reports this checkpoint's own performance beside the ten-run distribution
  (`deployable_checkpoint` in `RESULT.json`).
- **A poor Oldenburg result does not silently move the choice to another seed.** Any change of dispatch is a new,
  documented operating decision.

## Forecasts (scored at the result; conditional on legitimacy PASS and the unchanged frozen evaluation)

| Event | Forecaster | P |
|---|---|---:|
| Gate PASS (mean ≥ +0.030 at RAW b = 0.5, ≥ 8/10 positive) | Coordinator | **0.50** |
| Same (F4-04) | GPT-6 Pro | 0.60 |
| Clear transfer improvement (gate + joint lower bound > 0) | Opus (this packet) | 0.40 |
| Mean D1 − ADAF barrow ≤ 0 | Opus (this packet) | 0.15 |

The coordinator's basis for 0.50:

- **Prior adaptation tied ADAF barrow on Oldenburg.** ADAPT-NDS-1 A − ADAF barrow under v0 was +1.0 points RAW
  b = 0.5, interval spanning 0.
- **ADAF barrow's absolute recall there is low:** 0.156, the legacy NMS60 three-budget mean (v0 RAW b = 0.5: 0.099).

Opus's basis:

- D1 starts from the comparator itself, not from J.
- On Verden every interval excluded zero by ≥ 3 points.
- ADAF barrow leaves 130 ranking slots unused at b = 0.5 on Oldenburg.
- But Oldenburg's register differs: 1,534 stated vs 540 recorded mounds, and many small disc-geometry cemeteries.

## Files

| File | Role |
|---|---|
| `PACKET.md` | This rule |
| `legit.py` → `LEGITIMACY.json` | Legitimacy check (PASS) |
| `score_oldenburg.py` | The one-time scorer: gate, reading, bootstrap, joint summary, v0.1, parity; `--fixture` software check |
| `tests/test_gate_reading.py` → `TESTS.log` | Gate threshold, positivity and reading branches |
| `DRYRUN_FIXTURE.json` | End-to-end software check. Ten synthetic re-rankings of ADAF barrow's proposals stand in for D1; no D1 output is read. Made after the freeze. |
| `INFERENCE_PLAN.md` | The GPU step: exact commands, checkpoints, outputs, verification, VM-sharing trade-off |
| `cu_projection.py` → `CU_PROJECTION.json` | CU and storage projection from measured actuals |
| `STORAGE_CHECK.json` | storage_guard check at the declared 7.5 GB local peak (PASS, 2026-10-07 22:52Z) |
| `FREEZE.md`, `FREEZE.json`, `freeze_oldenburg.py` | Freeze record (sha256 of files and inputs) |

Output directories (single writer: this packet): `work/oldenburg_transfer_2026-10-07/` and
`outputs/oldenburg_transfer_2026-10-07/`.

**Attribution:**

- LGLN DGM1 (CC BY 4.0), modified.
- Denkmalatlas Niedersachsen (CC BY-SA 4.0), modified.
- ADAF weights CC BY-SA 4.0 (Čož et al. 2026); adapted weights stay CC BY-SA 4.0.
