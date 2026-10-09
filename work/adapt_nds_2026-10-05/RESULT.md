# ADAPT-NDS-1 result: Lower Saxony register labels raise barrow-group recall on an untouched district by 14 points; the lead over ADAF's barrow model is not distinguished (2026-10-05)

Status: **COMPLETE**. Freeze `ba07548b…`, packet fixed 2026-10-05 00:45Z, operative forecasts revised before any data
(00:43Z). Scored 17:17:18Z (`RESULT.json` `a4fb82aa…`). Independent recount MATCH 71/71 (`RECOUNT.json` `f4e817d6…`).
Capability line: regional generalisation (adapting the system to a new country from its public register).

## Verdict (CONFIRM = Landkreis Oldenburg, chosen by count rule; 272 barrow groups, 115 clusters, 300.5 km² background)

| Primary | Mean | 95 % interval | Per seed (02 / 03 / 04) | Label |
|---|---|---|---|---|
| ADAPT effect, A − C | **+0.141** | [+0.090, +0.181] | +0.174 / +0.115 / +0.132 | **ADAPT_GAIN** |
| Milestone, A − ADAF barrow | **+0.045** | [−0.011, +0.096] | +0.074 / +0.061 / +0.001 | **NOT_DISTINGUISHED** |

Metric: FRESH-1's benchmark primary, which is barrow-group recall averaged over 0.5, 1 and 2 background proposals per
km². The interval comes from a cluster × seed bootstrap (2,000 draws, seed 20261005).

- **A:** J continued for 5,116 updates, with one third of the steps on Lower Saxony register barrows from Uelzen,
  Harburg and Heidekreis.
- **C:** the same updates, schedule and draws, with the Lower Saxony steps replaced by English, Welsh and Irish steps.

## Systems on CONFIRM

| System | Primary | 0.5 / 1 / 2 / 4 per km² | Pool |
|---|---|---|---|
| A (3-seed mean; 0.603, 0.591, 0.531) | **0.575** | 0.485 / 0.583 / 0.656 / 0.691 | 6,021 / 4,959 / 1,212 |
| ADAF barrow HRNet (public) | 0.529 | 0.419 / 0.559 / 0.610 / 0.610 | 947 |
| C (0.429, 0.476, 0.398) | 0.434 | 0.336 / 0.439 / 0.528 / 0.604 | ~3,400 |
| CONT e41 | 0.429 | 0.357 / 0.434 / 0.496 / 0.592 | 4,395 |
| J, the start (0.419, 0.433, 0.368) | 0.407 | 0.316 / 0.409 / 0.494 / 0.583 | ~3,600 |
| ADAF AO (public) | 0.328 | 0.176 / 0.338 / 0.471 / 0.471 | 1,068 |
| INTERLEAVE(ADAF barrow, A) (0.581, 0.587, 0.543) | 0.570 | | |

Descriptive contrasts on CONFIRM:

- A − J: +0.168 [+0.120, +0.201].
- C − J: +0.028 [−0.002, +0.055].
- J − ADAF barrow: −0.123 [−0.177, −0.062].
- INTERLEAVE(ADAF barrow, A) − ADAF barrow: +0.041 [+0.006, +0.074].
- INTERLEAVE(ADAF AO, A) − ADAF barrow: −0.038 [−0.087, +0.010].

Individual-record unit, 3-seed means: A 0.266, C 0.199, ADAF barrow 0.251.

**Height strata (individual records, 3-seed means, at 1 proposal per km²):**

| Register height | n | A | C | ADAF barrow |
|---|---|---|---|---|
| < 0.5 m | 191 | 0.073 | 0.040 | 0.079 |
| 0.5–1.0 m or unknown | 453 | 0.207 | 0.149 | 0.185 |
| ≥ 1.0 m | 315 | 0.483 | 0.370 | 0.508 |

A gains over C in every stratum. Against ADAF barrow, A is level or behind on tall barrows at 1-2 per km². Its lead
comes from the strictest budget (0.5 per km²) and from depth: ADAF barrow's pool ends at 0.610. Barrows lower than
0.5 m are still missed by every system.

## Secondary: DEV (Verden; FRESH-1's terrain, previously evaluated; development evidence)

- A − C: +0.141 [+0.071, +0.192] (the ADAPT_GAIN rule would be met).
- A − ADAF barrow: +0.043 [−0.041, +0.122] (not distinguished).
- A means 0.599, against ADAF barrow 0.556 and J 0.424.
- INTERLEAVE(ADAF barrow, A): 0.612, which is +0.057 [+0.005, +0.104] over ADAF barrow.

FRESH-1's gap is now closed on average: the zero-shot J − ADAF barrow was −0.132 [−0.191, −0.051].

## Regressions reported with the gain

- **English retention** (in-training English holdout): A − C −0.072 [−0.123, −0.019]. Per seed, A 0.661 / 0.610 /
  0.715 against C 0.746 / 0.724 / 0.731.
- **Welsh and Irish DEV, seed 02 only** (point values): Wales A − C −0.085 (0.677 against 0.762); Ireland −0.204
  (0.360 against 0.565).
- A gives a third of its steps to Lower Saxony, so it rehearses England, Wales and Ireland less than C does.
  Interference is not separated from reduced rehearsal (the same open point as JOINT-1's English cost).
- A is a Lower Saxony specialist, not a replacement for J elsewhere.

## Forecasts (operative values; Brier)

| Event | P | Occurred | Brier |
|---|---|---|---|
| ADAPT_GAIN | 0.60 | yes | 0.160 |
| BEATS_ADAF_BARROW | 0.30 | no | 0.090 |
| TRAILS | 0.20 | no | 0.040 |
| English retention A − C < −0.010 | 0.30 | yes | 0.490 |
| Within 120 CU | 0.85 | yes (106.02) | 0.023 |

The forecasts underweighted the cost of adaptation to the other regions. The gain was about 4.7 times the +0.03
threshold.

## Resources and checkpoints

- **CU:** session 1 39.86 (reconciled for the concurrent JOINT-2 overlap; apparent 57.04), session 2 28.93, session
  3 37.23. Total **106.02 of 120**.
- **Six final checkpoints**, each 319,970,183 bytes, verified locally under `outputs/adapt_nds_2026-10-05/weights/`
  and not yet archived to Drive:

  | Seed | A_NDS sha256 | C_NDS sha256 |
  |---|---|---|
  | 2026092802 | `2eb8bd81…` | `a5146918…` |
  | 2026092803 | `f44f6489…` | `ac1d2eff…` |
  | 2026092804 | `2105fdd0…` | `cf15438e…` |

- Registry ids: `ADAPT-NDS-A_NDS_<seed>` and `ADAPT-NDS-C_NDS_<seed>`.

## Verification

- **Strict session selection:** exactly one SESSION_VERIFIED PASS per index, under this freeze, with the expected seed
  and trained arms (`run_s1_20261005T043612Z`, `run_s2_20261005T092024Z`, `run_s3_20261005T124915Z`).
- **Erratum 1:** the frozen scorer read a descriptive district field that the freeze lacks. A wrapper supplied it from
  the pinned selection file ([ERRATUM_1.md](ERRATUM_1.md), `SCORE_RECEIPT.json` `a6d1c883…`). No scoring input or rule
  changed.
- **Independent recount** ([recount_adapt.py](recount_adapt.py), using HR1's independent implementation): matches to
  1e-9 for:
  - every system primary;
  - both packet contrasts' means, 95 % bounds and labels, on CONFIRM and DEV;
  - the English retention means.

## Limits

- **Catalogue recovery, not discovery:** these are registered barrows (Denkmalatlas Niedersachsen). Unregistered
  barrows count as false alarms.
- **Adaptation within a new country, not transfer to an unseen one:** CONFIRM shares the state, sensor family and
  register with the fitting districts. The design has one country, one CONFIRM district and three seeds.
- **The gain has two possible sources that are not separated:** terrain adaptation, and learning the register's
  conventions (what is recorded as a Grabhügel).
- **Seed instability:**
  - Seed 04's A made a proposal pool about five times smaller (1,212 against 4,959–6,021).
  - Its whole-pool recall is 0.621 (against 0.80 and 0.78).
  - It is level with ADAF barrow (+0.001 on CONFIRM, −0.044 on DEV).
  - The recipe's reliability against the specialist is therefore not established.
- **ADAF barrow was used zero-shot.** Whether adapting ADAF on the same labels gains as much is ADAPT-MATCH-0's
  question (built, not frozen). BENCH-1 Track A needs that comparator.
- **Oldenburg is now evaluated terrain:** no tuning, selection or seed choice on it.

## Use of the result

- Regional supervision from a public register is the largest lever measured on untouched terrain so far: +0.14 over
  equal updates and +0.17 over the zero-shot start. The project's adapted system now leads ADAF's public barrow
  specialist on average in its home terrain, but not decisively.
- **No seed is adopted** (registry operating choice `lower_saxony_barrow_screening`: candidate_not_adopted). ADAF
  barrow remains the measured reference for Lower Saxony barrows.
  - INTERLEAVE(ADAF barrow, A) is a descriptive candidate for a pre-registered operating rule, not an adoption.
  - Any seed choice must be fixed without Oldenburg or Verden outcomes.
- **Next questions:**
  - adapted public baselines on the same labels (ADAPT-MATCH-0);
  - rehearsal-balanced adaptation if one model must serve several regions;
  - a fixed rule for the seed or ensemble before any operational use.

Attribution: LGLN DGM1 (CC BY 4.0) and Denkmalatlas Niedersachsen (NLD, CC BY-SA 4.0), modified. ADAF AO and barrow
HRNets: Čož, Corns, Curran, Kocev and Kokalj 2026 (weights CC BY-SA 4.0, code Apache-2.0). A and C descend from ADAF
AO, so shared weights carry CC BY-SA 4.0.
