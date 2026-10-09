# D1-COLLAPSE-1: result (2026-10-07, Opus 5.5)

Freeze `f323e17e`. Frozen verdict: `RESULT.json` (score_d1.py verdict, 2026-10-07T07:16:40Z). The recipe was selected on development
seeds (`RECIPE_SELECTION.json`, commit c826ddd4) before any confirmation run.

| Session | Top | Result | CU |
|---|---|---|---:|
| Development 1 | run_s1_20261006T200408Z | SESSION_VERIFIED PASS | 17.36 |
| Confirmation 3 | run_s3_20261006T222546Z | SESSION_VERIFIED PASS | 37.09 |
| Confirmation 4 | run_s4_20261007T030004Z | SESSION_VERIFIED PASS | 37.25 |
| **Total** | | | **91.70** (projection 116.4) |

Evidence class: development. Verden was previously evaluated by FRESH-1 and ADAPT-MATCH-0. Capability line: reliable
regional adaptation.

## Frozen verdict: PASS (recipe F1, no Dice on empty-target windows)

| Arm (10 fresh confirmation seeds 05–14) | Collapsed | Verden primary, mean (range) | Threshold | One-sided 95 % upper bound on collapse rate |
|---|---:|---|---:|---:|
| BARROW_L (ADAF barrow start) | 0/10 | 0.701 (0.682–0.717) | 0.519 | 0.26 |
| AO_L (ADAF AO start) | 0/10 | 0.662 (0.636–0.694) | 0.618 | 0.26 |
| Pooled | 0/20 | | | 0.14 |

- **Collapse rate.** 0/10 per arm bounds the per-run collapse rate below about 26 % per arm (14 % pooled) at 95 %. It
  does not show that collapse is eliminated.
- **No condition fired.** For every confirmation run, all three collapse conditions were false:
  - (a) zero proposals: proposal counts were 2,358–4,366;
  - (b) primary below half the reference;
  - (c) online detector ON at the end: the detector was never ON in any F1 run.

## Before and after, same setting

| | Old recipe (ADAPT-MATCH-0) | F1 |
|---|---|---|
| BARROW_L | 3/3 collapsed, Verden 0.000 | Development 0/3 (0.701, 0.716, 0.699; same seeds and draws as old); confirmation 0/10, mean 0.701 |
| AO_L | 1/3 collapsed; healthy seeds 0.672 and 0.638 | Confirmation 0/10, mean 0.662 |
| Probe P0 (old recipe re-run, BARROW_L seed 02, dense logging) | Collapsed again: detector ON from update 255 and at the end | – |

**Validation effect.**

- **AO_L:** the F1 mean of 0.662 is level with the old recipe's healthy seeds (0.655). The difference, +0.007, is
  inside the 0.037 repro noise.
- **BARROW_L:** the old recipe produced no usable model. F1 is +0.145 above ADAF barrow zero-shot (0.556).
- **Comparison with A (descriptive, unpaired).** Under F1, BARROW_L also exceeds ADAPT-NDS-1's adapted J (A, 0.599 on
  Verden, ADAPT-MATCH-0). This is not a BENCH-1 claim:
  - different seeds and no frozen contrast;
  - development terrain;
  - A was trained with the old recipe.

## Mechanism (DIAGNOSIS.md + DIAGNOSIS_ADDENDUM.md)

1. **Onset: the first fresh-AdamW step.** At lr 1e-4 with reset moments, the first update moves every parameter by
   about the lr. In both recipes, positive-window Dice falls from 0.92 to 0.001 and the maximum logit from 27.6 to
   −7.4 by update 2. DIAGNOSIS.md's "within 500 updates" was an upper bound from sparse logs.
2. **Trap: the empty-target Dice term `1 − 1/(Σp + 1)`.** Its gradient grows as the background mass falls.
   - Old recipe: background windows went to Σp ≤ 0.01 by update 8, with a Dice share of the background logit
     gradient of 1.0. They stayed there; recovery is possible only in the low-LR tail (2/6 in ADAPT-MATCH-0).
   - F1: the same windows carry BCE only. The mass recovers within about ten updates and positive Dice recovers by
     update 32–36.
3. **Frozen P0 reading: SUPPORTS.** It is degenerate: all three events fall at update 127, the first update at which
   the 64-per-role trailing windows are defined. The ordering comes from the per-update reading above:
   - H-EMPTY-DICE is confirmed as the absorbing mechanism;
   - it is refuted as the onset mechanism.
4. **What distinguished the old "healthy" seeds:** recovery before the LR ran out, not avoiding collapse.

**BatchNorm** is frozen throughout (running statistics, batch 1). It is not implicated.

## Costs and limits

- **Background mass.** F1 keeps more background probability: the final-epoch background-window Σp is about 260–310,
  against about 0.02 for collapsed runs.
  - Verden recall at fixed background-proposal rates did not fall.
  - Precision elsewhere is not measured here.
- **F1 removes the trap, not the shock.** A warm-up (F2) was pre-registered but not needed. Starts much more fragile
  than ADAF's might still need it.
- **Detector weakness.** The detector's mean-Σp condition is outlier-sensitive: P0 changed state 9 times before
  settling ON. Conditions (a) and (b) on Verden remain the primary collapse evidence.
- **Scope of the evidence.**
  - one region (Lower Saxony FIT → Verden DEV);
  - one schedule length (5,116 updates);
  - two ADAF starts;
  - no J-start runs under F1.

  A J-lineage F1 check belongs in BENCH-1's own packet. In the loss-mechanism probe, the empty-target Dice was 0.26 %
  of CONT's negative-window gradient, so a large J effect is not expected; this is unmeasured.
- **Scope of D1.** It gates fair adapted-model comparisons and reliable fine-tuning only.

## Forecasts (PACKET, scored)

| Forecast | P | Outcome | Brier |
|---|---:|---|---:|
| F1 selected in development | 0.5 | yes | 0.25 |
| Selected recipe passes confirmation | 0.45 | yes | 0.3025 |
| P0 SUPPORTS H-EMPTY-DICE | 0.5 | yes (degenerate, see above) | 0.25 |
| On a PASS, AO_L mean ≥ 0.655 | 0.4 | yes (0.662) | 0.36 |

Mean Brier: 0.29. All four forecasts were on the wrong side of 0.5 or at it, so these were under-confident.

## Checkpoints

- **Registered** in DETECTOR_REGISTRY.json, all `candidate_not_adopted`:
  - families D1-OLD-BARROW_L, D1-F1-BARROW_L and D1-F1-AO_L;
  - 24 checkpoints: P0, 3 development and 20 confirmation.
- **Archived:**
  - development weights: bundle j1;
  - BARROW_L confirmation weights: bundle k1;
  - AO_L confirmation weights: local and verified, pending archive.

## Next

- RECIPE.md fixes F1 for BENCH-1's equally adapted comparator, region-in-a-box and the pretraining pilot's fine-tune
  stage.
- D5 (barrow-teacher distillation) is unblocked. Its natural teacher, BARROW_L under F1, is the strongest Lower Saxony
  model measured on Verden (descriptive). That ranking needs a frozen BENCH-1 contrast before it is claimed.
