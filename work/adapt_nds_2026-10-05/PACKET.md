# ADAPT-NDS-1: can the system learn a new country from its public register and beat ADAF's barrow model there? (fixed 2026-10-05 00:45Z)

Capability line: regional generalisation (adapting the deployable system to a new country from public data). Owner
steer of 2026-10-05: "We MUST make substantial improvement." Evidence behind the choice:

- Zero-shot changes have not transferred: Verden-fitted rankers lose in France (EXPLORE_RESULT addenda 1-3), the
  Verden detectability floor does not replicate (FAINT-DIAG-2), TTA, AIM masks, background scaling, Dice, implants,
  geometry and resolution changes were null or harmful.
- The largest measured gains came from a region's own labels: JOINT-1 J − English-only C′ Ireland +13.25 and Wales
  +3.72 points (5/5 seeds); HR1 and HR3 ADAPT_GAIN.
- Lower Saxony is where public ADAF beats the project most: FRESH-1 Verden, ADAF barrow 0.556 vs S\* 0.484 and
  DEPLOY 0.400 (A1 INFERIOR). The models reach more groups in their full pools (0.70 vs 0.58); the gap is response
  and ranking on German barrows.
- JOINT-2's session-1 reproduction check failed (J_repro − J: England −0.011, Wales −0.014, Ireland +0.015 with
  identical draws, plan and start). Run-to-run noise is about 1-1.5 points per region, so only interventions with
  large expected effects can be shown quickly. This one is expected to be large.

## Hypothesis

Continuing J with Lower Saxony register barrows from three fitting districts raises barrow-group recall on an
unopened Lower Saxony district above ADAF's public barrow HRNet, and above an equal-update continuation without
Lower Saxony data.

## Eligible data

- **Register:** Denkmalatlas Niedersachsen OGC API (CC BY-SA 4.0), FRESH-1 label rules 1-7 unchanged
  (`work/fresh_region_2026-10-03/FRESH1_PREPIXEL.md` §2: Grabhügel, 4-40 m, condition flags, dedup, 60 m groups,
  other records IGNORE +30 m).
- **Terrain:** LGLN DGM1 1 m (CC BY 4.0), FRESH-1 preprocessing unchanged (`lgln_reader.py`, English 1 m path,
  bilinear 0.5 m, SLRM).
- **FIT districts:** Uelzen, Harburg, Heidekreis (register counts 2,518 / 1,927 / 1,799 Grabhügel). Fitting windows
  lie at least 2 km inside their district boundary on any side that touches Verden or the CONFIRM district.
- **DEV:** Verden, FRESH-1's frozen labels, plan and scorer (previously evaluated terrain; development evidence only).
- **CONFIRM:** chosen now by rule, from register counts and boundaries only, before any of its pixels, imagery or
  detector outputs are read: among Lower Saxony districts other than the four above that share no boundary with a FIT
  district, the one with the most eligible Grabhügel under rules 1-4; it must reach FRESH-1's floor of 200 groups
  after NODATA checks (else the next by count). Its spatial design copies FRESH-1 §3 (positive census, background
  squares in the frozen shuffle order to at least 300 km² net, 2 km clusters).
- Refused in code: TEST, reserved terrain, VALIDATION, French and Amazon data; CONFIRM data in training, thresholds or
  selection.

## Arms (seeds 2026092802, 2026092803, 2026092804; start = JOINT-1 J final weights at each seed, registry hashes)

- **A (ADAPT):** two J epochs (5,116 updates) of JOINT-1's recipe and epoch structure with Lower Saxony added as a
  fourth region taking one third of the steps (positive/background pairs drawn like the Welsh and Irish FIT pairs);
  AdamW reset as in J; J's own learning-rate schedule (warm-up, peak, cosine decay) compressed to 5,116 updates.
- **C (CONTROL):** identical updates, schedule and draws for the English, Welsh and Irish steps; the Lower Saxony
  steps are replaced by steps of the original three regions in J's proportions.
- **References (zero-shot, no training):** J at each seed, ADAF barrow, ADAF AO, CONT e41;
  INTERLEAVE(ADAF barrow, A) and INTERLEAVE(ADAF AO, A) are co-reported.
- No checkpoint selection: final weights only.

## Measures and outcome rule

- Metric: FRESH-1's benchmark primary (barrow-group recall, mean of 0.5, 1 and 2 background proposals per km²;
  `region_score.py`, `hier_uncertainty.py` byte-identical; reach 15 m, NMS 60 m), on CONFIRM and on DEV.
- **Primary 1, ADAPT effect (CONFIRM):** A − C, 3-seed paired, cluster x seed bootstrap (2,000 draws, seed 20261005).
  ADAPT_GAIN if the mean >= +0.03 and the 95 % lower bound > 0; ADAPT_HARM if the mean <= −0.03; else NULL.
- **Primary 2, milestone (CONFIRM):** A (3-seed mean) − ADAF barrow. BEATS_ADAF_BARROW if the mean >= +0.02 and the
  95 % lower bound > 0; TRAILS if the 95 % upper bound < 0; else NOT_DISTINGUISHED.
- Secondary: the same contrasts on DEV (Verden); English retention (A − C on the English holdout, in training);
  Welsh and Irish DEV for seed 2026092802 only (descriptive); height strata; INTERLEAVE systems.
- Forecasts (scored at the result): P(ADAPT_GAIN) = 0.70; P(BEATS_ADAF_BARROW) = 0.40; P(TRAILS) = 0.15;
  P(English retention A − C < −0.010) = 0.30; P(within 120 CU) = 0.85.

## Limits stated in advance

The register lists only preserved above-ground monuments and is incomplete, so unregistered barrows count as false
alarms on CONFIRM (biasing against detectors that find them). CONFIRM is in the same state, sensor family and
register as the FIT districts: a gain shows adaptation within a new country, not transfer to an unseen one. One
country; three seeds.

## Resources and procedure

- Build (CPU): register fetch for FIT, DEV-edge and CONFIRM districts (counts and geometry; CONFIRM labels built but
  not overlaid on any pixel before the frozen run), window plans, HEAD check of every DGM1 tile URL before any VM is
  allocated, tests, a smoke on the production artifact route, freeze.
- GPU: Colab G4, VM-side DGM1 fetch (no local tile store), one VM per session; the first session is one seed (A and
  C) plus the zero-shot references, verified before the remaining seeds (sequential-first). At most 120 CU.
- Local disk at most 1.5 GB (proposals, receipts, final weights). Weight downloads are deferred while C: is below
  24 GB so the live JOINT-2 chain keeps its launch floor. storage_guard and archive_gate for every launch.
- Attribution: "LGLN (2026)", modified (resampled, SLRM); "Niedersächsisches Landesamt für Denkmalpflege,
  Denkmalatlas Niedersachsen, CC BY-SA 4.0, modified". Masks and manifests stay private (or CC BY-SA); aggregate
  metrics carry the attribution.

## Forecast revision before any data (00:43Z)

Revised under the project's calibration rule (own hypotheses at most 0.5 unless an external base rate supports more)
and FRESH-1's power table (single-model contrasts on one district detect about +0.08 to +0.10; paired A − C less):
P(ADAPT_GAIN) 0.70 → **0.60** (base rate: every earlier local-label adaptation gained, JOINT-1 Ireland +13.25 and
Wales +3.72, HR1 and HR3 ADAPT_GAIN); P(BEATS_ADAF_BARROW) 0.40 → **0.30** (A must first close Verden's 0.07 gap);
P(TRAILS) 0.15 → **0.20**. No rule changed.
