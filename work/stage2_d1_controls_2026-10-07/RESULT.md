# STAGE2-D1-CONTROLS: result (scored 2026-10-08T16:4xZ, Opus coordinator)

Scorer: frozen `score_s2.py` (freeze `freeze/FREEZE.json` b5fdd0a9…, commit bd144db0), session 1 top
`outputs/stage2_d1_controls_2026-10-07/chain_runs/run_s1_20261008T011425Z` (`SESSION_VERIFIED.json` PASS). Generated
numbers are in `RESULT.json` and `BOOT_S2.json`; this file summarises them.

## Verdict: **PASS, clear advantage over the prespecified replay control on Verden**

Pro's outcome row (R4-1): *"The selected local barrow-training package exceeded the specified replay package by the
reported amount and met both predefined criteria."*

| Gate term (RAW, b = 0.5, Verden, v0) | Required | Measured |
|---|---|---|
| Mean of the ten D1_s − CTL_s, unrounded (in matches) | ≥ +0.030 (≥ 21.84 matches) | **+0.1234** (89.8 matches) |
| Strictly positive pairs | ≥ 8/10 | **10/10** (range +0.081 to +0.152) |
| Paired joint 95 % interval, 2 km, primary | lower bound > 0 | **[+0.084, +0.172]** |
| Sensitivities: independent-arm resampling; 4 km; 2 km offset | disclosed | all lower bounds +0.077 to +0.085; sign never differs |
| Evaluator v0.1 sensitivity | disclosed | COMPUTED: +0.1234, PASS (identical: Verden has no voids) |
| Collapsed controls | stay in the gate | **0/10** collapsed |
| Stage 1 reproduction (D1 and ADAF barrow) | identical | identical |

## Each arm against unchanged ADAF barrow (Pro R4-1: required for interpretation)

Recall; differences in points. The control is the equal-update nonlocal mixed-class replay control.

| Decoder, b | ADAF barrow | D1 (10-run mean) | Control (10-run mean) | D1 − ADAF | Control − ADAF | Controls above ADAF |
|---|---:|---:|---:|---:|---:|---:|
| RAW 0.5 | 0.169 | 0.231 | 0.108 | +6.25 | **−6.09** | 0/10 |
| RAW 1 | 0.291 | 0.385 | 0.175 | +9.37 | −11.63 | 0/10 |
| RAW 2 | 0.397 | 0.522 | 0.257 | +12.46 | −13.98 | 0/10 |
| NMS60 0.5 | 0.163 | 0.214 | 0.105 | +5.10 | −5.81 | 0/10 |
| NMS60 1 | 0.253 | 0.309 | 0.167 | +5.58 | −8.57 | 0/10 |
| NMS60 2 | 0.293 | 0.366 | 0.235 | +7.32 | −5.80 | 0/10 |

Control recall at RAW b = 0.5 by seed: 0.122, 0.110, 0.104, 0.125, 0.114, 0.088, 0.095, 0.140, 0.081, 0.102. The controls
supply 2,583–5,197 in-core proposals and none collapsed. They respond, but they rank barrows worse than the unchanged specialist.

## Reading

- **About half of the between-arm gap is the replay's harm.** The 12.3-point gap is D1's +6.25-point gain over ADAF barrow plus the
  control's −6.09-point loss. Equal-update training on English/Welsh/Irish mixed barrow-and-enclosure labels degrades
  ADAF barrow on Verden barrows at every budget and under both decoders, in every seed.
- **What this licenses (Pro R4-1 wording):** on Verden development terrain, the selected Lower Saxony barrow-training
  package outperformed the specified nonlocal mixed-class replay package, both starting from ADAF barrow with the same
  update budget and recipe.
- **What it does not license:**
  - It does not show that locality itself is the cause. Class content (the control's positives are 49–59 % barrow),
    foreground fraction (5.9–7.9 % against 1.8–2.0 %), exposure, terrain source and recipe selection all differ
    (PACKET.md deviations 1–7).
  - A plausible reading of the harm is that enclosure-weighted replay moves the specialist away from barrows. That is a
    hypothesis, not tested here.
- **Recipe selection:** the recipe was selected on Verden runs trained with local data (deviation 7). The ten
  confirmation pairs establish repeatability conditional on that recipe.
- **Scope:** one development region. Oldenburg's transfer check is frozen and pending.

## Cost and execution

- Session 0 smoke: 3.61 CU. Session 1: 40.28 CU. Total **43.89 CU** against a 60 CU ceiling; the projection was 47.66.
- Execution: the seed-05 barrier passed before seed 06; ten runs were harvested and verified; export verified; VM
  stopped.
- No score was computed before all ten controls were complete.

## Forecasts scored

| Forecast | P | Outcome | Brier |
|---|---:|---|---:|
| Coordinator P(PASS) | 0.70 | TRUE | 0.09 |
| Coordinator P(clear) | 0.60 | TRUE | 0.16 |
| Pro F4-01: all ten done within 60 CU by 21 Oct | 0.90 | TRUE (43.89 CU, 2026-10-08) | 0.01 |
| Pro F4-02: practical gate | 0.80 | TRUE | 0.04 |
| Pro F4-03: practical gate and positive lower bound | 0.75 | TRUE | 0.0625 |

## Next

- Archive the ten control weights to Drive with independent verification, then reclaim them locally. Register them
  as `S2CTL_BARROW_<seed>`, `candidate_not_adopted`.
- Run the Oldenburg transfer check once the D1 weights can be restored. Disk is binding (C: about 11 GB free on
  2026-10-08 16:40Z).
