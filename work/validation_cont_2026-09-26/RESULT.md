# VALIDATION comparison result: CONT e41 vs public ADAF AO

Scored 2026-09-26T05:29:15Z by the frozen stage-B code (`FREEZE_B.json`); packet `PACKET.md` + `PACKET_AMENDMENT_B.md`. Numbers below are generated from `outputs/validation_cont_2026-09-26/RESULT.json` and `RECOUNT.json` by `report.py`. Contains Environment Agency information © Environment Agency and database right (OGL v3); NHLE data © Historic England (OGL v3).

## Verdict: **MET**

- Primary (2 proposals per km², each arm at min(K, its own supply)): CONT 157 vs ADAF 132 of 265 common-class parts; Δ = +25 parts = +9.43 points.
- Paired 90 % block bootstrap (stage 2 code, 2,000 draws, seed 20260924): [+14, +37] parts, [+5.76, +13.31] points.
- Rule: each arm at min(K, its own supply); MET iff delta_points >= 5.0 and the 90% paired block-bootstrap lower bound (parts) > 0. Independent recount: **PASS**.

## Population and charges

- B0: census 45 blocks / 265 common / 275 all-class parts reproduced (735 / 393 / 351 / 46 overall). Run blocks 45, 265 common parts (251 barrow-tagged, 15 enclosure-tagged), 275 all-class parts, interior area 1119.75 km².
- Excluded by the enforced 500 m TEST guard: none.
- Within 5,040 m of a TEST square (descriptive): none.
- Reservations charged: sequences [15, 16] (ADAF AO baseline, CONT e41 improvement) with the D2609 reallocation (baseline 11, family 3, improvement 16, redteam 4, final 6).
- Supply after global 60 m NMS: CONT 9,407, ADAF 2,117.

## Common-class recovery by density

| d (per km²) | K | CONT hits (selected) | ADAF hits (selected) | Δ parts | Δ points | 90 % interval, parts | 90 % interval, points |
|---|---:|---:|---:|---:|---:|---|---|
| 0.5 | 560 | 125 (560) | 90 (560) | +35 | +13.21 | [+21, +50] | [+8.33, +18.45] |
| 1.0 | 1,120 | 146 (1,120) | 114 (1,120) | +32 | +12.08 | [+20, +45] | [+7.81, +16.79] |
| 2.0 | 2,240 | 157 (2,240) | 132 (2,117) | +25 | +9.43 | [+14, +37] | [+5.76, +13.31] |

## Classes, all-class and gained/lost parts

| d | CONT barrow | ADAF barrow | CONT enclosure | ADAF enclosure | CONT all-class | ADAF all-class | CONT gained / lost vs ADAF |
|---|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 125 | 90 | 0 | 0 | 127 | 94 | 38 / 3 |
| 1.0 | 144 | 113 | 2 | 1 | 150 | 118 | 36 / 4 |
| 2.0 | 155 | 130 | 2 | 2 | 163 | 137 | 28 / 3 |

## Equal-K comparison (descriptive only; COORDINATOR_B0_DECISION.md item 2)

| d | shared K | binding | CONT | ADAF | Δ parts | 90 % interval, parts |
|---|---:|---|---:|---:|---:|---|
| 0.5 | 560 | False | 125 | 90 | +35 | [+21, +50] |
| 1.0 | 1,120 | False | 146 | 114 | +32 | [+20, +45] |
| 2.0 | 2,117 | True | 154 | 132 | +22 | [+11, +35] |

## Label context (descriptive)

- Common-class parts outside their block interior: 3; on void source cells: 12. Both stay in the denominator.

## Resources

- GPU ledger: 1.140 h of the 3.0 h ceiling across 3 runs (wall time incl. model loading and fetch waits).
- Kept mosaics: 90 files (45 DTM + 45 validity rasters; the generated line said "90 DTM files", corrected by hand), 1.78 GB, NTFS-compressed folder `outputs/validation_cont_2026-09-26/mosaics/` (access needs a charged reservation; README there).

## Limits

- Conditional comparison on a VALIDATION population with 14 earlier charges; not an untouched test. MET would support only 'CONT e41 beats frozen public ADAF AO on this 45-block English VALIDATION evaluation'.
- ADAF's barrow class includes about 1,939 Historic England scheduled monuments from the same NHLE source as these labels; overlap with these blocks is unknown and may favour ADAF on barrows.
- Proposal density is not a measured false-alarm rate; labels are NHLE part centroids and 60 m matching does not verify outlines. TEST stays sealed (PACKET.md repair (c)).

## Executor notes (added by hand after generation; interpretation, not scoring)

- **Status: READY_FOR_TEST** under the packet's stop rule. TEST stays sealed. Opening it needs the `verify_freeze` prerequisites (VALIDATION-set thresholds, baselines reproduced, three-seed confirmation, three clean red-teams, clean-checkout reproduction) and then the owner typing "unseal TEST". None of these exists yet.
- **The VALIDATION gain is in barrows, not enclosures.**
  - Barrow-tagged parts: CONT 155 vs ADAF 130 of 251 at 2/km², and 125 vs 90 at 0.5/km².
  - Enclosure-tagged parts: 2 vs 2 of 15.
  - This is the reverse of stage 2, where CONT's gain over R2 and AO was almost all enclosures. The development class pattern did not carry over, but the overall direction did.
- **ADAF ran out of proposals at 2/km²** (supply 2,117 < K 2,240).
  - Under the primary per-arm-supply rule, CONT used 123 more proposals.
  - The equal-K comparison (descriptive) still gives +22 parts (+8.3 points), with a 90 % interval of [+11, +35].
  - At 0.5 and 1/km² neither arm was supply-limited. The margins there were +35 [+21, +50] and +32 [+20, +45].
- **Fetch audit.** All 1,620 WCS requests returned status 200, all inside SE/SP. The nearest request lies 31.7 km from a TEST square. No block failed the 500 m guard or came within 5,040 m.
