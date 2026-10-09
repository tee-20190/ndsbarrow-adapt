# OLDENBURG-TRANSFER-1: result (2026-10-09, scored once)

**Verdict: gate PASS, clear transfer improvement.** The ten D1 runs (public ADAF barrow + recipe F1, trained on Lower
Saxony FIT windows only) beat unchanged ADAF barrow on Oldenburg at the frozen operating point. This district is
previously inspected and training-disjoint, and was not used to select the recipe. It is not untouched terrain.

Scored by the frozen `score_oldenburg.py` (packet freeze `19289f9d…`) on session 1
`run_s1_20261008T203551Z` (`SESSION_VERIFIED` PASS, 11/11 proposal sets, `scored: false` at export, ADAF parity
PASS). Numbers below are copied from `RESULT.json`.

| Quantity (RAW decoder, b = 0.5, K = 225 over 451.1 km², N = 959 individual barrows) | Value |
|---|---|
| ADAF barrow recall | 0.0991 (95 matches) |
| D1 − ADAF barrow, mean of 10 runs | **+0.0386** (gate needs ≥ +0.030) |
| Runs positive | **10/10** (gate needs ≥ 8) |
| Per-run range | +0.030 to +0.045 |
| Primary joint 95 % lower bound | **+0.0137** (> 0, so "clear transfer improvement") |
| Deployable run 05: diff at b = 0.5 / 1 / 2 | +0.039 / +0.028 / +0.044 |
| Run 05 geography 95 % interval at b = 0.5 | [+0.018, +0.063] |

**Compared with Verden** (the development terrain where the recipe was selected): +6.25 points there and +3.9 points
here. The gain shrinks off the selection terrain but stays positive in every run, with a lower bound above zero.
ADAPT-NDS-1's earlier adapted model tied ADAF barrow on Oldenburg (+1.0 points, interval spanning 0). The
supported explanation is the specialist-start adaptation package; the recipe alone is not isolated (erratum 2026-10-09, GPT-6 Pro
Round 5: start checkpoint and other package components also differ from ADAPT-NDS-1).

**What this licenses:** D1 (run 05, the pre-registered deployable checkpoint) is the Lower Saxony barrow-survey
default over unchanged ADAF barrow, evidenced on two districts. ADAF barrow stays the default elsewhere.

**Not licensed:** an untouched-terrain or cross-country claim, non-barrow classes, or a claim that locality causes the
gain (Stage 2 tested a package, not locality).

**Cost:** smoke 1.75 CU, run 30.63 CU (ceiling 40).

## Forecasts (pre-registered, PACKET.md:169)

| Event | Forecaster | P | Outcome | Brier |
|---|---|---:|---|---:|
| Gate PASS | Coordinator | 0.50 | TRUE | 0.250 |
| Gate PASS (F4-04) | GPT-6 Pro | 0.60 | TRUE | 0.160 |
| Clear transfer improvement | Opus (packet) | 0.40 | TRUE | 0.360 |
| Mean D1 − ADAF ≤ 0 | Opus (packet) | 0.15 | FALSE | 0.023 |

Attribution: LGLN DGM1 (CC BY 4.0), modified. ADAF weights CC BY-SA 4.0 (Čož et al. 2026); D1 weights stay
CC BY-SA 4.0.
