# Independent re-score of the German benchmark-v0 headlines (2026-10-09)

**Verdict: REPRODUCED.** I scored the saved proposals with my own code and got every headline value exactly. I
also matched every per-system cell in the three RESULT.json files: K, matches, recall and supply, plus pool C and
G_ranking for Oldenburg. The result does not depend on any spec ambiguity I could identify: every variant gives
the same match counts.

## Files

| File | Role |
|---|---|
| `rescore.py` | Independent scorer, written from protocol text only. Run it in WSL v3venv, CPU only, about 20 min. |
| `RESCORE_RAW.json` | Its output for 32 systems × 3 budgets. Per system: supply, C, K, matches, recall, G_ranking, tie audit, sensitivity variants, input sha256. Aggregate only, no coordinates. |
| `compare.py` → `COMPARE.json` | The comparison step. It was written and run only after `RESCORE_RAW.json` existed. |

## Independence

- **Not read:** `verify/v3/` scoring code; `work/evaluator_v0_1_2026-10-07/*.py`; `score_v0.py`; `score_s2.py`;
  `score_oldenburg.py`; `region_score.py`; `work/manuscript_2026-10/ignore_rescore/`.
- **Read before scoring:**
  - `SPEC.md` (v0.1, sections 4 and 6);
  - the Stage 1 `FREEZE.md`;
  - the Stage 2 `PACKET.md` ("Primary scorer");
  - the Oldenburg `PACKET.md`;
  - `work/manuscript_2026-10/DRAFT_v1.md` sections 4.1, 4.4 and 4.5 (methods text);
  - Stage 1 `STEP0.json`, only for input paths and hashes.
- **Matcher:** scipy `maximum_bipartite_matching`, cross-checked with networkx `hopcroft_karp_matching` on 4 systems × 3
  budgets. The two agree in every cell.
- **PRIVATE files (disclosure).** Every proposal file and manifest in this project is named `*_PRIVATE.json`. The
  brief excluded `*_PRIVATE*` files but explicitly allowed proposals, manifests and footprints. I read only those
  inputs (listed below), locally, and wrote no coordinates. I opened no other PRIVATE file and no sealed or reserved
  data.

## Inputs

All hashes match the packet pins, and `compare.py` checks every Stage 1 and Stage 2 provenance sha256.

| Input | Path | sha256 (prefix) |
|---|---|---|
| Verden individual manifest (717 tiles, N = 728) | `outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json` | `a97b6124` |
| Verden ADAF barrow | `outputs/fresh_region_2026-10-03/step2/score/individual/PROPOSALS_ADAF_barrow_PRIVATE.json` | `7228b938` |
| Verden D1 05–14 | `outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/v1/results/dev/` | Stage 1 provenance |
| Verden controls 05–14 | `outputs/stage2_d1_controls_2026-10-07/chain_runs/run_s1_20261008T011425Z/v1/results/dev/` | Stage 2 provenance |
| Oldenburg individual manifest (699 tiles, N = 959) | `outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json` | `3531675c` |
| Oldenburg ADAF barrow (frozen comparator) | `outputs/adapt_nds_2026-10-05/score/confirm/individual/PROPOSALS_ADAF_barrow_PRIVATE.json` | `21421a9c` |
| Oldenburg D1 05–14 | `outputs/oldenburg_transfer_2026-10-07/chain_runs/run_s1_20261008T203551Z/v1/results/confirm/` | — (all 30 cells match) |

## Rule as implemented (reading (i))

| Step | Rule |
|---|---|
| Entities | Every part of the individual manifest |
| Area A | Union of the tile core boxes |
| Budget | K = floor(b·A + 1e-9) |
| Proposals | Every saved proposal whose centroid passes a half-open box test against its own tile's core |
| IGNORE | No IGNORE test at score time |
| RAW order | Score desc, mean probability desc, area desc, id asc; top K |
| Edge | Point-to-footprint-polygon distance ≤ 15 m (0 inside) |
| Matching | Maximum-cardinality bipartite matching |

## Targets: my value vs reported value

| # | Target | My value | Reported | Status |
|---|---|---|---|---|
| 1 | Verden scored area / N / K | 422.84319 km², 728; K 211 / 422 / 845 | 422.84319, 728; 211 / 422 / 845 | MATCH (exact) |
| 1 | ADAF barrow recall, b = 0.5 RAW | 0.168956 (123/728) | 0.169 (RESULT.json 0.16895604…) | MATCH (float-equal) |
| 1 | ADAF barrow, b = 1 / 2 | 0.291209 (212) / 0.396978 (289) | 0.29121 / 0.39698 | MATCH |
| 1 | D1 mean recall, b = 0.5 | 0.231456 | 0.231 | MATCH |
| 1 | D1 − ADAF, b = 0.5 | +6.250 points, 10/10 positive | +6.25, 10/10 | MATCH |
| 1 | D1 − ADAF, b = 1 / 2 | +9.368 / +12.459 points, 10/10 each | per-system cells equal | MATCH |
| 1 | Per-run matches, b = 0.5 (seeds 05–14) | 164, 164, 169, 166, 171, 171, 172, 161, 170, 177 | identical (RESULT.json) | MATCH (30/30 cells) |
| 2 | D1 − control per seed, b = 0.5 (matches) | 75, 84, 93, 75, 88, 107, 103, 59, 111, 103; Σ = 898 | identical; Σ 898 | MATCH |
| 2 | Mean D1 − control | +12.335 points, 10/10 | +12.3 (0.12335…), 10/10 | MATCH (float-equal) |
| 2 | Control − ADAF | −6.085 points (control mean 0.108104) | −6.1 (−0.0608516…) | MATCH |
| 2 | Control matches, b = 0.5 (05–14) | 89, 80, 76, 91, 83, 64, 69, 102, 59, 74 | identical | MATCH (30/30 control cells) |
| 3 | Oldenburg area / N / K | 451.11687 km², 959; 225 / 451 / 902 | 451.1, 959, K = 225 | MATCH |
| 3 | ADAF barrow, b = 0.5 | 0.099062 (95) | 0.0991 (95) | MATCH (float-equal) |
| 3 | ADAF barrow, b = 1 / 2 | 0.180396 (173) / 0.266945 (256) | 0.1804 (173) / 0.2669 (256) | MATCH |
| 3 | ADAF pool C / G_ranking(0.5) | 285 / 130 | 285 / 130 | MATCH |
| 3 | D1 − ADAF mean, b = 0.5 | +3.858 points, 10/10 | +3.86 (0.0385819…), 10/10 | MATCH (float-equal) |
| 3 | Run 05 at b = 0.5 / 1 / 2 | +3.858 / +2.815 / +4.380 | +3.9 / +2.8 / +4.4 | MATCH |
| 3 | Per-run D1 − ADAF, b = 0.5 (matches gained) | 37, 29, 39, 36, 33, 34, 38, 42, 43, 39 | per_run_diff identical | MATCH |
| 3 | Re-inferred ADAF barrow (parity) | 95 / 173 / 256 | parity PASS | MATCH |

- Full grid: 32 systems × 3 budgets = 96 K/matches/recall cells, plus 32 supply values. All are equal (`COMPARE.json`,
  `all_cells_equal: true`).
- Pool terms: Oldenburg C and G_ranking are equal for all 11 systems.
- Run 05 equals the mean at b = 0.5 by coincidence: Σ gained = 370, so the mean is 37.0, which is run 05's own gain.
- **DIFFs: none.**

## Spec ambiguities and sensitivity

| # | Ambiguity | Choices tested | Effect on match counts (96 cells) |
|---|---|---|---|
| 1 | Is 15 m inclusive (≤) or strict (<)? | ≤ (literal "within 15 m"), < | None |
| 2 | In-core test: half-open, closed or none? (Saved files might contain halo proposals.) | half-open (manuscript 4.1), closed, no filter | None. The saved files hold 0 out-of-core proposals |
| 3 | K: is there a float guard, and is A the union or the sum of cores? | floor(b·A + 1e-9) (v0.1 text), floor(b·A), floor(b·Σcores) | None. Cores are disjoint (union = sum to 1e-12 km²) |
| 4 | Ties in (score, mean probability, area) at the K cut | id ascending (spec) | None. No cut has a tie group larger than 1, so the id tie-break never decides membership |
| 5 | Which copy of the proposals? | harvest vs results copies are byte-identical. Verden ADAF: `individual/` copy, whose sha equals the Stage 1 provenance (`group/` differs in bytes). Oldenburg: session s1 (all 699 tiles), not s0 (2-tile smoke) | n/a. All chosen files reproduce exactly |
| 6 | "Eligible subset" | All manifest parts: 728 / 959, as STEP0 and the packet state | — |
| 7 | IGNORE handling | Reading (i): no IGNORE at score time. The existing background squares already exclude records (SPEC §6) | Readings (ii) and (iii) not attempted; out of scope |

## Limits of this check

- **Scope.** This independently verifies the scoring layer: area, K, in-core filtering, ranking, the edge rule and
  one-to-one matching.
- **Shared inputs.** It reuses the project's proposal files and manifests. A defect upstream would be shared and stay
  undetected: in proposal extraction (thresholding, vectorisation, the roundness filter), in manifest building (cores,
  entity footprints, eligibility) or in inference.
- **NMS60 and the bootstrap intervals** were not re-implemented. NMS60 is secondary; the bootstrap was optional.
