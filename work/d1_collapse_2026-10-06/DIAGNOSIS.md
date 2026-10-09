# D1 — why ADAF-start fine-tunes collapse to all-background (diagnosis from existing evidence, 2026-10-06)

Roadmap step D1 (work/roadmap_2026-10-06/ROADMAP.md Track D). No GPU was used for this note. Capability line:
reliable regional adaptation (the comparator recipe for BENCH-1, region-in-a-box and the pretraining pilot).

## Sources read

- ADAPT-MATCH-0 sampled training losses (one record every 500 updates) and the per-epoch means:
  `outputs/adapt_match_2026-10-05/chain_runs/run_s1_20261005T201653Z/v1/job.log` and `results/runs/*/epochs.jsonl`
  (AO_L), and `run_s2_20261006T015302Z` (BARROW_L). Also its 8-update smoke on staged FIT windows:
  `outputs/adapt_match_2026-10-05/smoke/run_20261005T201545Z/SMOKE_TRAIN_*.json`.
- The recipe code: `work/adapt_match_2026-10-05/match_train.py`, `work/amazon_adapt_2026-09-28/adapt_train.py:141-153`
  (update), `work/campaign_2026-09-22/r2_train.py:135-151` (frozen BatchNorm, loss),
  `work/adapt_nds_2026-10-05/nds_schedule.py:115-119` (LR).
- J-start comparators with the same optimiser, loss, batch size, BatchNorm handling and LR schedule:
  - ADAPT-NDS-1 A and C, `outputs/adapt_nds_2026-10-05/chain_runs/run_s{1,2,3}_*`;
  - HR1 Wales, `outputs/holdout_region_2026-10/chain_runs/t2_s*` (including A_W0 seed 2026092803).
- An ADAF AO start on English data (R2): `outputs/campaign_2026-09-22/training/R2_adaf_hrnet_slrm/` (constant LR 1e-4, AdamW,
  same loss).
- The FIT table's per-window positive pixel counts (`outputs/adapt_nds_2026-10-05/fit/FIT_TABLE_PRIVATE.json`).
- **Not available:** per-step logits, intermediate checkpoints, or a split of the loss into its BCE and Dice terms in
  any MATCH run. The trainer logged one combined loss every 500 updates and kept final weights only. The six MATCH
  weights are archived on Drive, not local.

## What the recipe does

- Loss per window (batch 1): `BCE(z, g)` averaged over non-IGNORE pixels **+ 1 − soft Dice**, where
  `Dice = (2 Σ p·g + 1) / (Σ p + Σ g + 1)` and `p = sigmoid(z)` on output channel 1 (`r2_train.py:142-151`).
- The optimiser is a fresh AdamW at lr 1e-4 and weight decay 1e-4. There is no warm-up. The schedule is a cosine from
  1e-4 to 1e-6 over 5,116 updates (5.05e-5 at update 2,558).
- The windows alternate: an even update is a positive window, an odd update is a background window with no positive
  pixel.
- **BatchNorm is frozen.** `set_train_mode` puts every BatchNorm layer in eval mode, so the start's running statistics
  are used throughout and batch-1 statistics never enter. Only the BatchNorm affine weights train.

## Timing: onset and recovery

Positive-window loss at the logged updates (combined BCE + 1 − Dice). The epoch columns give means over 1,279
positive / 1,279 background windows.

| Run | upd 0 | 500 | 1000 | 1500 | 2000 | 2500 | ep 4 pos / bg | 2558 | 3058 | 3558 | 4058 | 4558 | 5058 | ep 5 pos / bg | Verden |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AO_L 02 | 0.23 | 1.07 | 1.09 | 1.02 | 0.91 | 1.02 | 1.25 / 0.069 | 1.08 | 1.04 | 1.01 | 1.03 | 1.06 | 1.19 | 1.27 / 0.020 | **0.000** |
| AO_L 03 | 0.09 | 1.06 | 1.01 | 1.13 | 1.08 | 1.09 | 1.26 / 0.074 | 0.01 | 1.97 | 0.002 | 0.39 | 0.54 | 0.005 | 0.62 / 0.127 | 0.672 |
| AO_L 04 | 0.40 | 1.16 | 1.03 | 1.24 | 1.15 | 1.02 | 1.32 / 0.053 | 1.08 | 0.50 | 0.03 | 0.33 | 0.03 | 0.22 | 0.66 / 0.126 | 0.638 |
| BARROW_L 02 | 0.08 | 1.08 | 1.08 | 1.02 | 1.02 | 0.76 | 1.26 / 0.138 | 1.08 | 1.04 | 1.01 | 1.02 | 1.06 | 1.17 | 1.27 / 0.025 | **0.000** |
| BARROW_L 03 | 1.02 | 1.05 | 1.01 | 1.12 | 1.07 | 1.10 | 1.33 / 0.080 | 1.07 | 1.93 | 1.04 | 1.11 | 1.14 | 1.01 | 1.28 / 0.023 | **0.000** |
| BARROW_L 04 | 0.51 | 1.13 | 1.05 | 1.33 | 0.25 | 1.01 | 1.36 / 0.221 | 1.09 | 1.06 | 1.10 | 1.04 | 1.03 | 1.13 | 1.25 / 0.019 | **0.000** |

Reading (the identical draw digests show that AO_L and BARROW_L see the same windows and offsets at each seed):

1. **The onset is universal and early.**
   - By update 500 (lr 9.8e-5), all six runs sit at a positive-window loss of 1.01–1.16. That is the value of a model
     that predicts nothing on the window: Dice ≈ 1/(Σg + 1) ≈ 0, so the loss is ≈ 1 + BCE.
   - Their first windows scored 0.08–0.51 (five runs) or 1.02 (BARROW_L 03), so each start detected barrows and then
     lost them within the first 500 updates.
   - All six epoch-4 means are collapsed-looking (positive 1.25–1.36). The epoch-5 background means also fall to
     0.019–0.025.
2. **Collapse is the default outcome; what differs between seeds is recovery.**
   - The two "healthy" AO_L seeds were collapsed through epoch 4 too. They recovered during the low-LR half:
     - AO_L 03 by update 2,558 (lr 5.05e-5);
     - AO_L 04 between updates 2,558 and 3,058 (lr ≤ 3.6e-5).
   - AO_L 02 and every BARROW_L seed stayed collapsed until the LR reached 1e-6.
   - **So "healthy" means "recovered before the LR ran out", not "never collapsed".**
   - At the same seed, AO_L and BARROW_L share every draw (identical digests) and differ only in the start. Recovery
     therefore depends on the start's weights as well as the data order: AO 2/3, barrow 0/3.
3. **The collapsed end state is an equilibrium, not a numerical failure.**
   - Every loss is finite and nothing is skipped (`skipped_all_ignore` 0).
   - The background-window loss settles at 0.019–0.025. The loss on an empty-target window is BCE + 1 − 1/(Σp + 1)
     ≈ Σp when Σp is small, so the predicted probability mass over a whole 512 × 512 background window is about 0.02.
   - The positive-window loss settles at about 1.27, i.e. Dice ≈ 0 plus BCE ≈ 0.27 from very negative logits on the
     barrow pixels.

## Factors

| Factor | Evidence | Verdict |
|---|---|---|
| **BatchNorm with batch 1** | Running statistics are frozen in every run (`r2_train.py:135-139`). Batch-1 statistics are never used, and J-start runs with the identical setting do not collapse. | **Not implicated.** The only batch-1 consequence is that Dice is computed per window, so every background step evaluates Dice on an empty target (below). |
| **LR / optimiser** | The onset happens at lr ≈ 1e-4 with fresh AdamW and no warm-up, and the only recoveries happen at lr ≤ 5e-5. But R2 (ADAF AO, constant lr 1e-4, fresh AdamW, same loss, English windows) trained 40 epochs without collapse (epoch-0 positive holdout loss 0.67). ADAPT-NDS-1's six J-start runs used the identical compressed schedule and did not collapse to all-background. | **Amplifier, not sufficient.** High LR makes the descent fast and leaves no budget to climb back, but lr 1e-4 alone does not collapse a model. |
| **Initialisation** | ADAF starts: 6/6 collapsed at onset, 4/6 never recovered. J lineage under the same recipe: ADAPT-NDS-1 A/C 0/6 all-background; A seed 04 partially (English holdout positive loss 1.065 and negative 0.166, against about 0.6 / 1.04 for the others; proposal pool about a fifth of the other seeds'). HR1 Wales (CONT e41 start, same recipe family): 1 of 21 trainings partial (A_W0 03). | **Strong effect.** ADAF's precise, low-background-mass heads are much more exposed than J's. J's holdout negative loss of about 1.04 means its background mass stays large. |
| **Loss weighting** (per-window Dice with smoothing 1 on empty targets) | See the mechanism below: the size of the Dice gradient on a background window grows as 1/(Σp + 1)² as the background mass falls. HR1's A_W0 03 shows the same signature from a CONT e41 start. Its Welsh windows were background only (A_W0 has no Welsh positives). The Welsh loss fell 1.03 → 0.48 while the other A_W0 seeds stayed at about 1.00, and its holdout positive loss rose to 0.724 (others about 0.55). | **Most likely driver of the absorbing state.** |
| **Window repetition** (MATCH sees each FIT window about 3 times; A sees each once) | The onset occurs inside the first pass (update ≤ 500 of 5,116; nothing has repeated yet). | **Not the onset cause.** |

## Mechanism hypothesis

**H-EMPTY-DICE.** On a background window (target all zero) the Dice term is `1 − 1/(Σp + 1)`.

- **The Dice gradient is weak at large mass and strong at small mass.** Its gradient on logit `z_i` is
  `p_i(1 − p_i) / (Σp + 1)²`. The BCE gradient is `p_i / N` (N ≈ 262,144 valid pixels). Their per-pixel ratio is
  therefore about `N / (Σp + 1)²`:
  - while the window's predicted mass Σp is above √N ≈ 512, Dice is negligible. This is the J-lineage regime, where
    the loss-mechanism probe measured Dice at 0.26 % of the negative-window gradient;
  - below that, Dice dominates, by a factor of up to ~10⁵ as Σp → 0.
- **Background steps then push the global logit offset down, harder the lower it already is.** This is positive
  feedback.
- **On positive windows, the restoring Dice gradient saturates.** It scales as `p(1 − p)`, so it vanishes when the
  barrow pixels' p → 0. What remains is the pixel-mean BCE on the barrow pixels, about Σg/N of one unit. The median
  window has 1,991 positive pixels, so Σg/N ≈ 0.8 % (mean 1.2 %).
- **All-background is therefore an absorbing equilibrium.** It sits where the background push `Σp/(Σp + 1)²` balances
  the barrow pull Σg/N, i.e. Σp_bg ≈ Σg/N ≈ 0.01.
- **The equilibrium prediction is in range.** It predicts a background-window loss of about 0.01, and the collapsed
  runs settled at 0.019–0.025. This is an order-of-magnitude check only: it treats the global offset as the only
  direction and ignores Adam's per-parameter scaling.
- **The ADAF/J difference fits this picture.** A start whose background mass is small, or which is pushed down
  quickly by fresh-Adam steps at lr 1e-4, enters the steep part of the curve within a few hundred updates. A start
  with large background mass (J) stays on the flat part, where the empty-window Dice is inert.

**Evidence for it:**

- the universal early onset;
- the collapsed equilibrium's background loss (≈ Σp ≈ 0.02) matches the predicted order;
- recovery happens only when the LR falls;
- the A_W0 03 analogue: background-only regional windows, Welsh background loss falling to 0.48;
- R2 and J-start runs, which sit in the large-mass regime, do not collapse.

**Not established:**

1. What drives the first descent below Σp ≈ 512 in ADAF starts? Candidates are the fresh-AdamW steps at full LR
   (every parameter moves by about lr per update regardless of gradient size) and a low starting background mass.
   - The smoke's background losses at update 1-7 (0.97–1.01) only bound the starting mass loosely (Σp ≳ 30).
   - The confidently wrong positive window in the smoke (loss 7.55 / 5.64) shows ADAF fires strongly on some Lower
     Saxony terrain, which gives large early BCE gradients.
2. Whether removing the empty-target Dice term alone prevents the onset, or only removes the trap. A sufficiently
   strong negative-BCE push could still collapse a model whose features are destroyed early.

**Evidence still needed** (cheap; see PACKET.md, probe P0): one old-recipe BARROW_L run with dense logging (BCE and
Dice separately, Σp and the maximum logit per window, every 16 updates). It should show:

- Σp on background windows falling through ~512;
- the Dice share of the background gradient rising;
- positive-window Dice falling to 0 after that, not before.

If positive Dice collapses first while background mass is still large, H-EMPTY-DICE is wrong about the onset, and the
fallback (warm-up plus lower peak LR) targets the actual driver.

## What distinguishes collapsed from healthy seeds (operational)

- **End-of-training signature** (separates all six MATCH runs; the zero-proposal runs are exactly the collapsed ones):

  | | final-epoch mean positive-window loss | mean background-window loss |
  |---|---|---|
  | collapsed (4 runs) | 1.25–1.28 | 0.019–0.025 |
  | healthy (2 runs) | 0.62–0.66 | 0.126–0.127 |

- **During training,** both kinds pass through the collapsed state. A detector on the final 256 updates separates
  them; one on epoch 4 does not. The online detector in PACKET.md is defined on a trailing window and reported as
  "ever on" and "on at end".
- **Seed and data order** do not explain it alone: the same draws give AO 2/3 healthy and barrow 0/3.

## Consequence for the fix (carried into PACKET.md)

- **Primary (F1):** compute the Dice term only on windows that contain positive pixels; empty-target windows get BCE
  only.
  - This removes the term whose gradient grows as the model empties.
  - It is close to a no-op where J operates (0.26 % of the background gradient).
  - It differs from the stop-listed "background/negative re-weighting". That was a performance lever, rejected twice.
    This one is a stability change, and it is judged on collapse and on no loss of validation recall.
- **Fallback (F2), pre-registered:** F1 plus a 256-update linear warm-up and a peak lr of 5e-5. It targets the
  fresh-AdamW onset. It is used only if F1 fails the development selection rule (three BARROW_L development seeds,
  PACKET.md "Sequence"); confirmation then runs on ten fresh seeds per arm.
