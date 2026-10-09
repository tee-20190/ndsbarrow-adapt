# D1-COLLAPSE-1: a collapse-safe fine-tune recipe for ADAF starts (packet, 2026-10-06)

Status: fixed before any GPU run. This file is committed before development session 1. Roadmap Track D step D1
(`work/roadmap_2026-10-06/ROADMAP.md`); owner authorisation in `decisions/2026-10-06-roadmap-confirmed.md`. Revised
before the freeze to apply the coordinator's independent-review corrections:

- separate development and confirmation seeds;
- operational definitions;
- honest verdict wording;
- scope.

Capability line: reliable regional adaptation.

**Scope.** D1 gates two things only:

- fair adapted-model comparisons (BENCH-1's equally adapted comparator);
- reliable fine-tuning (region-in-a-box, and the pretraining pilot's fine-tune stage).

D1 does not gate label work, metric repairs, frozen-encoder probes, pretraining preparation, or inference with
existing models.

**Evidence class: development.** Verden DEV was previously evaluated by FRESH-1 and ADAPT-MATCH-0. Nothing here is a
claim about untouched terrain.

## Hypothesis (from DIAGNOSIS.md)

H-EMPTY-DICE: on empty-target background windows, the per-window soft Dice term is `1 − 1/(Σp + 1)`.

- Its gradient grows as `1/(Σp + 1)²` while the background mass falls.
- Below Σp ≈ √N it outweighs pixel-mean BCE by up to ~10⁵.
- Positive-window Dice saturates as `p(1 − p)` → 0.
- So all-background is an absorbing state.

In ADAPT-MATCH-0, 6/6 ADAF-start runs collapsed by update 500 and 4/6 never recovered.

## Candidate fixes

- **F1 (primary): no Dice on empty targets.**
  - A window whose valid target has no positive pixel is scored with BCE only. Windows with positives keep
    BCE + 1 − Dice.
  - Everything else is the ADAPT-MATCH-0 recipe:
    - start tars;
    - fresh AdamW, lr 1e-4, weight decay 1e-4;
    - cosine 1e-4 → 1e-6 over 5,116 updates, no warm-up;
    - frozen BatchNorm, batch 1;
    - the all-local plan;
    - final weights only.
- **F2 (fallback, development only if F1 is not selected):** F1 + a linear warm-up over updates 0–255 and a peak lr of
  5e-5. The schedule is a cosine from 5e-5 to 1e-6 over 5,116 updates, multiplied by min(1, (t + 1)/256).
- **P0 (probe, not a candidate):** the old recipe, BARROW_L, seed 2026092802, training only, with dense logging.
  - **Predicted order:**
    1. Σp on background windows falls through ~512;
    2. the Dice share of the background logit gradient rises;
    3. positive-window Dice falls to ~0 after that.
  - **Reported** in RESULT.md as SUPPORTS (all three orderings hold), CONTRADICTS (positive Dice reaches < 0.05 while
    the background Σp is still > 512) or UNCLEAR.

## Test setting (ADAPT-MATCH-0's comparator setting, same data and splits)

- **Arms:** AO_L (ADAF AO tar, sha256 a2232cc1…) and BARROW_L (ADAF barrow tar, sha256 51b679fa…), as pinned in
  `work/adapt_match_2026-10-05/match_common.py`.
- **FIT:** ADAPT-NDS-1's Lower Saxony FIT table, staging pins and masks, read by hash (852 positive and 852 background
  windows).
- **Plan:** ADAPT-MATCH-0's all-local plan (`match_schedule.local_plan`) at each seed: 5,116 alternating updates on
  A's draw streams. The plan does not depend on the recipe.
- **DEV:** Verden, through FRESH-1's frozen label-free inference manifest (717 tiles), HR3 `infer_irl.py`, and
  `score_adapt.score_model`.
  - Primary: FRESH-1's barrow-group recall, the mean over 0.5, 1 and 2 background proposals per km².
- **Not used:**
  - CONFIRM (Oldenburg), TEST, reserved terrain, VALIDATION;
  - English, Welsh and Irish windows. The step-0 environment fixture is used for the environment gate only.

## Seeds: development and confirmation are disjoint

| Set | Seeds | Use |
|---|---|---|
| Development | 2026092802, 03, 04 | ADAPT-MATCH-0's seeds, already used by the diagnosis. Recipe selection uses them only. |
| Confirmation | 2026092805 … 2026092814 (10 per arm) | Never used by any earlier run. Trained only after the recipe is frozen. |

## Dense logging and the online collapse detector

- **Per update** (`curves.jsonl`):
  - role, lr;
  - BCE, soft Dice;
  - Σp and Σg over valid pixels, valid pixel count;
  - max and mean logit;
  - the analytic logit-space BCE and Dice gradient sums and L2 norms;
  - the old-form loss and the loss actually used.
- **Online detector** at update t ≥ 255, over the trailing 256 updates (128 positive, 128 background windows):
  - ON iff mean positive-window soft Dice < 0.05 **and** mean background-window Σp < 1.0;
  - it records only; it never alters training;
  - per run it records the first update ON, the number of updates ON, and **ON at end** (t = 5,115).

## "Collapsed" (exact; applies to every final checkpoint)

A final checkpoint is **collapsed** if any one of the following holds:

- **(a) All-background on the selection set:** zero proposals over Verden DEV's 717-tile manifest. This is the
  ADAPT-MATCH-0 failure.
- **(b) Recall collapse:** Verden primary below half its arm's reference: AO_L < 0.3275, BARROW_L < 0.2778.
- **(c) All-background on the training stream at the end:** the detector is ON at end, i.e. over the last 256
  updates:
  - mean positive-window soft Dice < 0.05;
  - mean background-window predicted mass < 1.0 (a positive-pixel fraction below 1 in 262,144).

Check against ADAPT-MATCH-0's six runs, from their final-epoch signatures:

- the 4 zero-proposal runs meet (a) and the (c) signature;
- the 2 healthy runs meet none (Verden 0.672 and 0.638; positive Dice ≈ 0.4–0.5).

## Acceptable performance loss (exact)

The 10-seed mean Verden primary must be ≥ reference − δ.

- **δ = 0.037**, the largest measured same-seed rerun difference (HR1 C rerun). It exceeds JOINT-2's 1.1–1.5-point
  repro movement and FRESH-1's paired single-checkpoint SE of 0.023.
- **AO_L reference = 0.655**, the mean of the old recipe's healthy seeds 03 (0.672) and 04 (0.638). Threshold: 0.618.
- **BARROW_L reference = 0.556**, ADAF barrow zero-shot (FRESH-1). The old recipe has no healthy BARROW_L seed, so the
  arm's start is the reference. Threshold: 0.519.

## Sequence

1. **Development session 1:**
   - P0 (train only);
   - F1 BARROW_L at the three development seeds (train + Verden DEV).
   - BARROW_L is the stricter arm (old recipe 3/3 collapsed).
2. **Selection rule (development only):**
   - F1 is selected iff 0 of its 3 development runs is collapsed and their mean primary is ≥ 0.519.
   - Otherwise, development session 2 runs F2 BARROW_L at the same three seeds, with the same rule.
   - If neither is selected, stop with two failed candidates. RESULT.md then reports a mechanism diagnosis (P0 plus
     both candidates' curves) without a recipe (two-failure rule).
   - `select_recipe.py` writes `RECIPE_SELECTION.json`, which is committed. **That commit freezes the recipe.**
3. **Confirmation (selected recipe only; no change of recipe, hyperparameter or rule after selection):**
   - session 3/13: BARROW_L at the 10 confirmation seeds;
   - session 4/14: AO_L at the 10 confirmation seeds.
   - Each session is verified before the next one starts.

## Exit rule (confirmation runs only)

**PASS** iff, for **both** arms:

1. 0 of 10 confirmation runs is collapsed;
2. the 10-seed mean Verden primary is ≥ the threshold above.

**FAIL otherwise.** RESULT.md states which condition failed and gives:

- the per-seed table;
- each development run;
- P0's verdict;
- the exact (Clopper–Pearson) one-sided 95 % upper bound on the per-run collapse rate per arm. At 0/10 that is 0.26; at 0/20
  pooled, 0.14.

**Wording of a PASS:** "0/10 collapsed per arm: the collapse rate under this recipe is below about 26 % per arm at
95 %". It is not "collapse is eliminated".

**If confirmation fails:**

- no further candidate is tried inside D1, since tuning after confirmation would spend the fresh seeds;
- RESULT.md reports the mechanism evidence (P0 and the curves) as the alternative outcome: a demonstrated mechanism
  plus a partially effective fix;
- a new candidate needs a new packet with new confirmation seeds.

**Adoption:** no checkpoint is adopted by this packet. On a PASS, RECIPE.md fixes the recipe for BENCH-1,
region-in-a-box and the pretraining pilot's fine-tune stage. All checkpoints are registered as
`candidate_not_adopted`.

## Resources

- **GPU:** Colab G4 through the official CLI route (`work/colab_cli_2026-09-30`), one VM at a time.
  - The driver is a checked copy of ADAPT-MATCH-0's; its frozen files are untouched.
  - The local RTX 2060 is not used: Verden staging needs 16 GB of disk and its inference would be ~3× slower.
- **CU projection:** `CU_PROJECTION.json`, written by `chain/sessions_d1.py projection`.
  - It uses ADAPT-MATCH-0's per-stage model: 509 s per training, a DEV pass of 717 × (2.38 + 1.25 × models) s, and
    8.724 CU/h. It is conservative: ADAPT-MATCH-0's sessions cost about 73 % of its projection.

  | Session | Contents | Projected CU |
  |---|---|---:|
  | Dev 1 | P0 + 3 F1 runs | 23.7 |
  | Dev 2 | 3 F2 runs, only if needed | 22.4 |
  | Confirmation BARROW_L | 10 runs | 46.6 |
  | Confirmation AO_L | 10 runs | 46.1 |
  | **F1 path total** | | **116.4** |
  | **F2 path total** | | **138.8** |

- **Ceilings:**
  - development 55 CU;
  - confirmation 120 CU;
  - each session 1.25 × projection.
  - A measured overrun is handled by a documented revision under the budget-revision policy, not by a stop.
- **Local disk:**
  - every launch through `verify/v3/archive_gate.py run --` and `verify/v3/storage_guard.py`;
  - the declared peak is the session's weights + 0.5 GB;
  - launch rule: C: free ≥ 20 GB reserve + peak + 2 GB page-file headroom;
  - weight downloads wait while C: free − weight < 22 GB (the VM keeps the weight);
  - D1 weights are archived to Drive with an independent read before local reclaim, at the latest before the second
    confirmation session.
- **Artifact rule (owner):** no VM is stopped before every final weight and result has a verified local copy
  (sha256 and size).
- **Output directories (single writer: this packet's driver and scorer):**
  - `outputs/d1_collapse_2026-10-06/` (private);
  - `work/d1_collapse_2026-10-06/` (code, receipts, RESULT).
- **Attribution:**
  - ADAF weights CC BY-SA 4.0 (Čož et al. 2026; Zenodo 10.5281/zenodo.15848662); adapted weights stay CC BY-SA 4.0.
  - LGLN DGM1 (CC BY 4.0), modified; Denkmalatlas Niedersachsen (CC BY-SA 4.0), modified.

## Forecasts (own hypotheses ≤ 0.5)

| Forecast | P |
|---|---:|
| F1 is selected in development | 0.5 |
| The selected recipe passes the confirmation exit rule | 0.45 |
| P0 SUPPORTS H-EMPTY-DICE | 0.5 |
| On a PASS, AO_L confirmation mean ≥ 0.655 | 0.4 |
